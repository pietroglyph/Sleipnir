// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/InteriorPoint.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>

#include <Eigen/SparseCholesky>

#include "optimization/RegularizedLDLT.hpp"
#include "optimization/solver/util/Filter.hpp"
#include "optimization/solver/util/FractionToTheBoundaryRule.hpp"
#include "optimization/solver/util/Bounds.hpp"
#include "optimization/solver/util/ModifiedLagrangian.hpp"
#include "optimization/solver/util/PrimalDualSystem.hpp"
#include "optimization/solver/util/LogBarrierFunctions.hpp"
#include "optimization/solver/util/SwitchingCriterion.hpp"
#include "optimization/solver/util/TerminationCriteria.hpp"
#include "sleipnir/autodiff/Gradient.hpp"
#include "sleipnir/autodiff/Hessian.hpp"
#include "sleipnir/autodiff/Jacobian.hpp"
#include "sleipnir/optimization/SolverExitCondition.hpp"
#include "sleipnir/util/Print.hpp"
#include "sleipnir/util/EigenFormatter.hpp"
#include "sleipnir/util/Spy.hpp"
#include "util/ScopeExit.hpp"
#include "util/ToMilliseconds.hpp"

// See docs/algorithms.md#Works_cited for citation definitions.
//
// See docs/algorithms.md#Interior-point_method for a derivation of the
// interior-point method formulation being used.

namespace sleipnir {

void InteriorPoint(std::span<Variable> decisionVariables,
                   std::span<Variable> inequalityConstraints, Variable& f,
                   function_ref<bool(const SolverIterationInfo& info)> callback,
                   const SolverConfig& config, Eigen::VectorXd& x,
                   SolverStatus* status) {
  const auto solveStartTime = std::chrono::system_clock::now();

  // Given in table 1 of [4]; explanations are included there.
  static constexpr double ε_opt = 1e-6, ε_far = 1e-3, ε_inf = 1e-6,
                          ε_unbd = 1e-12;
  static_assert(0 < ε_opt);
  static_assert(0 < ε_far && ε_far < 1);
  static_assert(0 < ε_inf && ε_inf < 1);
  static_assert(0 < ε_unbd && ε_unbd < 1);
  static constexpr double β_1 = 1e-4, β_2 = 0.01, β_3 = 0.02, β_4 = 0.2,
                          β_5 = 0x1p-5, β_6 = 0.5, β_7 = 0.5, β_8 = 0.9,
                          β_10 = 1e-4, β_11 = 1e-2, β_12 = 1e3;
  static_assert(0 < β_1 && β_1 < 1);
  static_assert(0 < β_2 && β_2 < 1);
  static_assert(β_2 < β_3 && β_3 < 1);
  static_assert(0 < β_4 && β_4 < 1);
  static_assert(0 < β_5 && β_5 < 1);
  static_assert(0 < β_6 && β_6 < 1);
  static_assert(0 < β_7 && β_7 < 1);
  static_assert(0.5 < β_8 && β_8 < 1);
  static_assert(0 < β_10 && β_10 < 1);
  static_assert(0 < β_11);
  static_assert(β_11 <= β_12);

  // Map decision variables and constraints to VariableMatrices so that we can
  // compute the gradient of the constraints and the Hessian of the modified
  // Lagrangian wrt these variables.
  VariableMatrix xAD{decisionVariables};
  VariableMatrix c_iAD{inequalityConstraints};
  Eigen::VectorXd c_i{inequalityConstraints.size()};

  // Make new Variables for the Lagrange multipliers so that we can update them
  // when we need to re-compute the Hessian of the modified Lagrangian.
  Assert(inequalityConstraints.size() <= std::numeric_limits<int>::max());
  VariableMatrix yAD{static_cast<int>(inequalityConstraints.size())};
  // Make a new variable for μ so that we can update it before re-computing the
  // Hessian of the modified Lagrangian.
  Variable μAD;

  // Modified Lagrangian, L_μₖ; we only compute the Hessian from
  // this.
  //
  // L_μₖ = f(xₖ) + (yₖ - μₖβ₁e)ᵀcᵢ(xₖ),
  Variable L = AutodiffModifiedLagrangian(f, yAD, c_iAD, μAD, β_1);
  // Hessian of the modified Lagrangian H
  //
  // Hₖ = ∇²ₓₓL_μₖ(xₖ, yₖ)
  Hessian hessianL{L, xAD};
  // First calculation should happen after we project x₀ onto bounds and first
  // compute μ and set μAD during initialization.
  Eigen::SparseMatrix<double> H;

  // Gradient of f, denoted ∇f in the paper and g in the code
  Gradient gradientF{f, xAD};
  // First calculation should happen after we project x₀ onto bounds.
  Eigen::SparseVector<double> g;

  // Jacobian product ∇cᵢᵀy, used to compute the Lagrangian and termination
  // criteria while re-using other autodiff computations
  Gradient gradientComplimentarity{c_iAD, yAD};
  // Calculations happen on every inner iteration but the first (where we
  // instead use the constraint Jacobian directly).
  Eigen::SparseVector<double> gradientComplimentarityValue;
  // Jacobian product ∇cᵢᵀe, used to compute some termination criteria while
  // re-using other necessary autodiff computations
  Gradient gradientConstraintSum{c_iAD, VariableMatrix::Ones(c_iAD.size(), 1)};
  // Calculations happen on every inner iteration but the first (where we
  // instead use the constraint Jacobian directly).
  Eigen::SparseVector<double> gradientConstraintSumValue;

  // Inequality constraint Jacobian, Aᵢ
  //
  //          [∇ᵀcᵢ₁(xₖ)]
  // Aᵢ(xₖ) = [∇ᵀcᵢ₂(xₖ)]
  //          [    ⋮    ]
  //          [∇ᵀcᵢₘ(xₖ)]
  Jacobian jacobianCi{c_iAD, xAD};
  // We need this to detect bounds in the first place, so we compute it now and
  // re-compute it after projecting x₀ onto bounds.
  Eigen::SparseMatrix<double> A_i = jacobianCi.Value();

  // KKT system solver and the LHS matrix we pass to it
  RegularizedLDLT solver;
  Eigen::SparseMatrix<double> M;

  // Update x₀ and compute y₀, s₀, w, and μ₀ according to section
  // B.2 of [4].
  Eigen::VectorXd y{inequalityConstraints.size()};
  Eigen::VectorXd s{inequalityConstraints.size()};
  Eigen::VectorXd w{inequalityConstraints.size()};
  double μ;
  {
    // Detect bounds by inspecting the inequality constraint Jacobian
    const auto [boundConstraintMask, bounds, conflictingConstraints] =
        GetBounds(decisionVariables, inequalityConstraints, A_i);
    if (!conflictingConstraints.empty()) {
      if (config.diagnostics) {
        sleipnir::println(
            "The problem is infeasible because of conflicting bound "
            "constraints (conflicting indices are in order of inequality "
            "constraint declaration and then equality constraint "
            "declaration):");
        for (const auto& [firstIndex, secondIndex] : conflictingConstraints) {
          sleipnir::println("  Constraint {} conflicts with {}", firstIndex,
                            secondIndex);
        }
        sleipnir::println(
            "\nThe following bounds conflict (decision variable indices are in "
            "order of declaration):");
        for (std::size_t decisionVariableIndex = 0;
             decisionVariableIndex < bounds.size(); decisionVariableIndex++) {
          const auto& [lower, upper] = bounds[decisionVariableIndex];
          if (lower > upper) {
            sleipnir::println(
                "  Decision variable {} has lower bound of {} which is greater "
                "than its upper bound of {}",
                decisionVariableIndex, lower, upper);
          }
        }
      }
      status->exitCondition = SolverExitCondition::kTooFewDOFs;
      return;
    }

    // Project x₀ onto the bounds (should the user be able to disable this?).
    Eigen::VectorXd xOld = x;
    ProjectOntoBounds(x, bounds);

    // Compute cᵢ(x₀ᵇ) and ∇f(x₀ᵇ) for the updated (bound-projected) x₀ᵇ,
    // falling back to the user-provided x₀ if the objective or constraints are
    // non-finite for the bound-projected x₀ᵇ.
    auto UpdateXRecomputeAutodiff = [&xAD, &x, &f, &g, &gradientF, &c_i, &c_iAD,
                                     &A_i,
                                     &jacobianCi](const Eigen::VectorXd& newX) {
      x = newX;
      xAD.SetValue(x);
      g = gradientF.Value();
      c_i = c_iAD.Value();
      A_i = jacobianCi.Value();
      return std::isfinite(f.Value()) && c_i.allFinite();
    };
    if (!UpdateXRecomputeAutodiff(x) && !UpdateXRecomputeAutodiff(xOld)) {
      status->exitCondition =
          SolverExitCondition::kNonfiniteInitialCostOrConstraints;
      return;
    }

    // This section is loosely based on Gertz, Nocedal, and Wright's init
    // strategy for IPMs for NLP, given in algorithm 3.1 of [5].
    //
    // First compute provisional ỹ and s̃, which we use to
    // form M (with H and Aᵢ, which were computed above).
    Eigen::VectorXd yTilde =
        Eigen::VectorXd::Ones(inequalityConstraints.size());
    // Assumes cᵢ(x) ≤ 0. The paper writes this as:
    //   −cᵢ(x₀) + max{−2minⱼ{sⱼ}, β₁₀},
    // and is ambiguous about what s should be here. According to
    // https://github.com/ohinder/OnePhase.jl/blob/4863f8146f1454c353118b3f12b1784dfea60032/src/init/gertz_init.jl#L12-L16
    // we should use s = -cᵢ(x₀) (note the sign flip, since we assume cᵢ(x) ≤ 0
    // following their paper, but their code assumes the opposite).
    double μHat = std::max(2 * c_i.maxCoeff(), β_10);
    μAD.SetValue(μHat);    // So that μ = μ̂ when computing the value of H
    yAD.SetValue(yTilde);  // So that y = ỹ = e when computing the value of H
    Eigen::VectorXd sTilde = -c_i.array() + μHat;
    ComputePrimalDualLHS(M, hessianL, A_i, yTilde, sTilde);
    solver.Compute(M);

    // Compute an affine scaling step and use it to compute a more accurate
    // estimate of yTilde.
    // Again, the paper is ambiguous about the values of μ, γ, and w; inspecting
    // https://github.com/ohinder/OnePhase.jl/blob/4863f8146f1454c353118b3f12b1784dfea60032/src/init/gertz_init.jl#L17
    // and looking at [5] shows that we're supposed to compute an affine scaling
    // step (i.e., γ = 0). Their code sets w to e, which is not like in [5]
    // where they (in our notation) set w = −cᵢ(x) − δ.
    const auto [_, d_y] =
        PrimalDualStep(solver, g, yTilde, yTilde, sTilde, sTilde,
                       Eigen::VectorXd::Ones(w.size()), A_i, A_i, μHat, 0, β_1);

    // This section is based on Mehrotra's init strategy for an LP IPM, given in
    // section 7 of [6].
    static constexpr double negativeScaling =
        2.0;  // Set to 1.5 in [6]; unclear why it's different in [4].
    // Refine the intermediate estimate yTilde based on the above affine scaling
    // step, and then find ε_y so that it is the smallest (times
    // negativeScaling) elementwise increment that makes ỹ positive elementwise.
    yTilde += d_y;
    double ε_y = std::max(-negativeScaling * yTilde.minCoeff(), 0.0);
    yTilde.array() += ε_y;

    // Re-compute the intermediate estimate sTilde so that it isn't negative by
    // adding the smallest elementwise increment (times negativeScaling) that
    // makes s̃ positive.
    sTilde = -c_i;
    // The expression in [4] for ε_s seems to be wrong: 1) they take the max
    // with the scaled Lagrangian instead of with 0, which doesn't match [6] and
    // which makes less sense to me; 2) they write an L₂ norm instead of an L_∞
    // norm of ỹ, which makes less sense (with the L_∞ norm this expression is
    // like one of the termination criteria) and is not consistent with their
    // code. 3) They compute ε_s after updating ỹ  with ε_y in [4], but in their
    // code they compute ε_s before updating ỹ; the former is closer to what
    // happens in [6] and seems to make a little more sense (since the updated ỹ
    // might as well be taken into account).
    double ε_s = std::max(-negativeScaling * sTilde.minCoeff(), 0.0) +
                 ManualGradientModifiedLagrangian(g, A_i, yTilde, 0, β_1)
                         .lpNorm<Eigen::Infinity>() /
                     (yTilde.lpNorm<Eigen::Infinity>() + 1.0);
    sTilde += ε_s * boundConstraintMask;

    // Compute the final refinement of yTilde and sTilde based on the weighted
    // mean complimentarity.
    auto WeightedMeanComp = [](const Eigen::VectorXd& sTilde,
                               const Eigen::VectorXd& yTilde,
                               const Eigen::VectorXd& weight) -> double {
      return sTilde.dot(yTilde) / (2 * weight.sum());
    };
    // All ops here are componentwise, so aliasing shouldn't be an issue
    yTilde = (yTilde.array() + WeightedMeanComp(sTilde, yTilde, sTilde))
                 .min(β_12)
                 .max(β_11);
    sTilde += boundConstraintMask * WeightedMeanComp(sTilde, yTilde, yTilde);
    double μTilde = sTilde.cwiseProduct(yTilde).mean();

    // Set final output values.
    if (config.diagnostics) {
      sleipnir::println("Barrier initial guess scale: {}",
                        config.initialBarrierScale);
    }
    μ = config.initialBarrierScale * μTilde;
    s = sTilde;
    w = (c_i + s) / μ;
    y = yTilde.cwiseMax(β_3 * μ * s.cwiseInverse())
            .cwiseMin((1 / β_3) * μ * s.cwiseInverse());
  }

  if (config.diagnostics) {
    sleipnir::println("Error tolerance: {}\n", config.tolerance);
  }

  // Sparsity pattern files written when spy flag is set in SolverConfig
  std::ofstream H_spy;
  std::ofstream A_i_spy;
  if (config.spy) {
    H_spy.open("H.spy");
    A_i_spy.open("A_i.spy");
  }

  // We count total iterations to check the max iteration exit condition, and
  // acceptable iterations to check the acceptable tolerance exit condition.
  int iterations = 0;
  int acceptableIterCounter = 0;

  // Prints final diagnostics when the solver exits.
  std::chrono::system_clock::time_point allIterationsStartTime;
  scope_exit exit{[&] {
    status->cost = f.Value();

    if (config.diagnostics) {
      auto solveEndTime = std::chrono::system_clock::now();

      sleipnir::println("\nSolve time: {:.3f} ms",
                        ToMilliseconds(solveEndTime - solveStartTime));
      sleipnir::println(
          "  ↳ {:.3f} ms (solver setup)",
          ToMilliseconds(allIterationsStartTime - solveStartTime));
      if (iterations > 0) {
        sleipnir::println(
            "  ↳ {:.3f} ms ({} solver iterations; {:.3f} ms average)",
            ToMilliseconds(solveEndTime - allIterationsStartTime), iterations,
            ToMilliseconds((solveEndTime - allIterationsStartTime) /
                           iterations));
      }
      sleipnir::println("");

      sleipnir::println("{:^8}   {:^10}   {:^14}   {:^6}", "autodiff",
                        "setup (ms)", "avg solve (ms)", "solves");
      sleipnir::println("{:=^47}", "");
      constexpr auto format = "{:^8}   {:10.3f}   {:14.3f}   {:6}";
      sleipnir::println(format, "∇f(x)",
                        gradientF.GetProfiler().SetupDuration(),
                        gradientF.GetProfiler().AverageSolveDuration(),
                        gradientF.GetProfiler().SolveMeasurements());
      sleipnir::println(format, "∇²ₓₓL", hessianL.GetProfiler().SetupDuration(),
                        hessianL.GetProfiler().AverageSolveDuration(),
                        hessianL.GetProfiler().SolveMeasurements());
      sleipnir::println(format, "∂cᵢ/∂x",
                        jacobianCi.GetProfiler().SetupDuration(),
                        jacobianCi.GetProfiler().AverageSolveDuration(),
                        jacobianCi.GetProfiler().SolveMeasurements());
      sleipnir::println("");
    }
  }};

  if (config.diagnostics) {
    allIterationsStartTime = std::chrono::system_clock::now();
  }
  // TODO(declan): Max acceptable iterations is BROKEN because we don't check
  // error tolerance here... Is it even appropriate to continue to track this?
  // What error tolerance would we track?
  while (acceptableIterCounter < config.maxAcceptableIterations) {
    std::chrono::system_clock::time_point outerIterStartTime;
    if (config.diagnostics) {
      outerIterStartTime = std::chrono::system_clock::now();
    }

    // Check for max iterations
    if (iterations >= config.maxIterations) {
      status->exitCondition = SolverExitCondition::kMaxIterationsExceeded;
      return;
    }

    // Check for solve to acceptable tolerance
    // TODO(declan): Again, this is broken... Need to remind myself of IPOPT's
    // rationale for this and determine whether or not we really need it with
    // one-phase.
    // if (E_0 > config.tolerance &&
    //     acceptableIterCounter == config.maxAcceptableIterations) {
    //   status->exitCondition =
    //   SolverExitCondition::kSolvedToAcceptableTolerance; return;
    // }

    // Set autodiff variables to the current (outer iteration) state and perform
    // both cheap (objective gradient) and expensive autodiff (constraint
    // Jacobian and Lagrangian Hessian).
    xAD.SetValue(x);
    g = gradientF.Value();
    c_i = c_iAD.Value();
    A_i = jacobianCi.Value();
    yAD.SetValue(y);
    μAD.SetValue(μ);
    H = hessianL.Value();

    // Write out spy file contents if that's enabled
    if (config.spy) {
      // Gap between sparsity patterns
      if (iterations > 0) {
        A_i_spy << "\n";
        H_spy << "\n";
      }

      Spy(A_i_spy, A_i);
      Spy(H_spy, H);
    }

    // Form the Schur compliment of the block-eliminated primal-dual system
    ComputePrimalDualLHS(M, hessianL, A_i, y, s);

    // Find some δ s.t. M + δI > 0 and then perform a LDLᵀ factorization of δ
    // XXX(declan): I think we handle δ differently on the first iteration than
    // in the paper, which says to set δ ← 0; we're currently using whatever δ
    // we had in init.
    // TODO(declan): Need to incorparate step A.6 from Algorithm 4 in [4].
    solver.Compute(M);

    // We're done computing (potentially multiple) new iteration(s), so we call
    // the user callback.
    // TODO(declan): I'm not calling this for each inner iteration since we
    // don't re-compute H or A_i for the inner iterations... Need to check how
    // this callback is used in practice to determine whether changing the
    // callback is appropriate. In some ways, the "inner iterations" thing is
    // like a second-order correction, and I don't think the user cares about
    // the intermediate updates of the second-order correction unless they all
    // succeed?? Hm or maybe they do?
    // XXX(declan): Previously this skipped the first iteration, I think?
    if (callback({iterations, x, s, g, H, A_i})) {
      status->exitCondition = SolverExitCondition::kCallbackRequestedStop;
      return;
    }

    // Save the iterate we used to compute H and A_i
    const double μHat = μ;
    const Eigen::VectorXd xHat = x;
    const Eigen::VectorXd sHat = s;
    const Eigen::VectorXd yHat = y;

    for (int innerIterCounter = 0; innerIterCounter < 2; innerIterCounter++) {
      bool locallyOptimal, locallyInfeasible;
      if (innerIterCounter == 0) {
        // We've already computed the full constraint Jacobian in this case, so
        // just taking the following matrix-vector products is cheaper than more
        // autodiff
        locallyOptimal = IsLocallyOptimal(y, s, c_i, g, A_i, β_1, ε_opt);
        locallyInfeasible = IsInequalityLocallyInfeasible(
            y, s, c_i, A_i.transpose() * y, ε_far, ε_inf);
      } else {
        // Update autodiff variables with new inner iteration state and perform
        // *only* the cheap autodiff (which is what distinguishes this from an
        // outer iteration).
        xAD.SetValue(x);
        c_i = c_iAD.Value();
        g = gradientF.Value();
        yAD.SetValue(x);
        μAD.SetValue(μ);
        gradientComplimentarityValue = gradientComplimentarity.Value();
        gradientConstraintSumValue = gradientConstraintSum.Value();

        locallyOptimal =
            IsLocallyOptimal(y, s, c_i, g, gradientComplimentarityValue,
                             gradientConstraintSumValue, ε_far, ε_inf);
        locallyInfeasible = IsInequalityLocallyInfeasible(
            y, s, c_i, gradientComplimentarityValue, ε_far, ε_inf);
      }

      // Check if the KKT conditions are satisfied to some accuracy.
      if (locallyOptimal) {
        status->exitCondition = SolverExitCondition::kSuccess;
        return;
      }

      // Check for local inequality constraint infeasibility.
      if (locallyInfeasible) {
        if (config.diagnostics) {
          sleipnir::println(
              "The problem appears to be locally infeasible because it is "
              "approaching a stationary point for a weighted L_∞ infeasibility "
              "measure. If the constraints are convex, this may be a "
              "certificate of infeasibility in the sense of Observation 1 in "
              "[4].");
          sleipnir::println(
              "Violated constraints (cᵢ(x) ≥ 0) in order of declaration (some "
              "of these may have been converted from equality constraints):");
          for (int row = 0; row < c_i.rows(); ++row) {
            if (c_i(row) < 0.0) {
              sleipnir::println("  {}/{}: {} ≥ 0", row + 1, c_i.rows(),
                                c_i(row));
            }
          }
        }

        status->exitCondition = SolverExitCondition::kLocallyInfeasible;
        return;
      }

      // Check for unboundedness/diverging iterates.
      if (IsUnbounded(x, ε_unbd) || !x.allFinite() || !s.allFinite()) {
        status->exitCondition = SolverExitCondition::kDivergingIterates;
        return;
      }

      bool aggressiveStep;
      // The Lagrangian gradient ∇ₓL_{γμ}(x, y); used to compute both aggressive
      // and stabilizing steps. The value of γ depends on whether this is an
      // aggressive or stabilizing step. See equation (7) in [4].
      Eigen::VectorXd b_D;
      if (innerIterCounter == 0) {
        // This is again a special case so that we use the constraint
        // Jacobian we've already computed instead of doing more autodiff
        aggressiveStep = IsAggressiveStepAppropriate(y, s, g, A_i, μ, β_1, β_3);
        b_D = ManualGradientModifiedLagrangian(
            g, A_i, y, (aggressiveStep ? 1.0 : 0.0) * μ, β_1);
      } else {
        aggressiveStep = IsAggressiveStepAppropriate(
            y, s, g, gradientComplimentarityValue, gradientConstraintSumValue,
            μ, β_1, β_3);
        b_D = ManualGradientModifiedLagrangian(
            g, gradientComplimentarityValue, gradientConstraintSumValue,
            (aggressiveStep ? 1.0 : 0.0) * μ, β_1);
      }
      if (aggressiveStep) {
        // Compute an affine-scaling step and use the result to select a barrier
        // multipler γ (which is a centering parameter locally/in the linear
        // case) using Mehrotra's predictor-corrector heuristic, as in [6].
        const auto [d_x, d_y] = PrimalDualStep(solver, y, yHat, s, sHat, w, b_D, A_i, μ, 0);
	
      } else {
        // TODO(declan)
      }

      // Check for max wall clock time
      const auto innerIterEndTime = std::chrono::system_clock::now();
      if (innerIterEndTime - solveStartTime > config.timeout) {
        status->exitCondition = SolverExitCondition::kTimeout;
        return;
      }

      // Diagnostics for current outer iteration
      if (config.diagnostics) {
        /* if (iterations % 20 == 0) {
          sleipnir::println("{:^4}   {:^9}  {:^13}  {:^13}  {:^13}", "iter",
                            "time (ms)", "error", "cost", "infeasibility");
          sleipnir::println("{:=^61}", "");
        }

        sleipnir::println("{:4}  {:9.3f}  {:13e}  {:13e}  {:13e}", iterations,
                          ToMilliseconds(innerIterEndTime - innerIterStartTime),
                          E_0, f.Value(),
                          c_e.lpNorm<1>() + (c_i - s).lpNorm<1>()); */
      }
    }

    // xₖ₊₁ = xₖ + αₖpₖˣ
    // sₖ₊₁ = sₖ + αₖpₖˢ
    // yₖ₊₁ = yₖ + αₖᶻpₖʸ
    // TODO(declan): not exactly like this for one-phase
    // x += α * p_x;
    // s += α * p_s;
    // y += α_z * p_y;

    ++iterations;

    // The search direction has been very small twice, so assume the problem has
    // been solved as well as possible given finite precision and reduce the
    // barrier parameter.
    //
    // See section 3.9 of [2].
    if (stepTooSmallCounter >= 2 && μ > μ_min) {
      UpdateBarrierParameterAndResetFilter();
      continue;
    }
  }
}  // NOLINT(readability/fn_size)

}  // namespace sleipnir
