// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/interior_point.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "optimization/regularized_ldlt.hpp"
#include "optimization/solver/util/error_estimate.hpp"
#include "optimization/solver/util/filter.hpp"
#include "optimization/solver/util/is_locally_infeasible.hpp"
#include "optimization/solver/util/kkt_error.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/util/assert.hpp"
#include "util/print_diagnostics.hpp"
#include "util/scope_exit.hpp"
#include "util/scoped_profiler.hpp"
#include "util/solve_profiler.hpp"

// See docs/algorithms.md#Works_cited for citation definitions.
//
// See docs/algorithms.md#Interior-point_method for a derivation of the
// interior-point method formulation being used.

namespace {

/**
 * Interior-point method step direction.
 */
struct Step {
  /// Primal step.
  Eigen::VectorXd p_x;
  /// Log-domain variable step.
  Eigen::VectorXd p_v;
};

class BarrierParameter {
 public:
  constexpr BarrierParameter(double μ) : μ{μ}, sqrt_μ{std::sqrt(μ)} {}
  static constexpr BarrierParameter from_sqrt(double sqrt_μ) {
    return BarrierParameter{sqrt_μ * sqrt_μ, sqrt_μ};
  }

  constexpr operator double() const { return μ; }
  constexpr double sqrt() const { return sqrt_μ; }

 private:
  constexpr BarrierParameter(double μ, double sqrt_μ) : μ{μ}, sqrt_μ{sqrt_μ} {}

  double μ;
  double sqrt_μ;
};

}  // namespace

namespace slp {

ExitStatus interior_point(
    const InteriorPointMatrixCallbacks& matrix_callbacks, bool is_nlp,
    std::span<std::function<bool(const IterationInfo& info)>>
        iteration_callbacks,
    const Options& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::VectorXd& x) {
  const auto solve_start_time = std::chrono::steady_clock::now();

  gch::small_vector<SolveProfiler> solve_profilers;
  solve_profilers.emplace_back("solver");
  solve_profilers.emplace_back("  ↳ setup");
  solve_profilers.emplace_back("  ↳ iteration");
  solve_profilers.emplace_back("    ↳ feasibility ✓");
  solve_profilers.emplace_back("    ↳ iteration callbacks");
  solve_profilers.emplace_back("    ↳ μ update");
  solve_profilers.emplace_back("    ↳ iter matrix build");
  solve_profilers.emplace_back("    ↳ iter matrix compute");
  solve_profilers.emplace_back("    ↳ iter matrix solve");
  solve_profilers.emplace_back("    ↳ line search");
  solve_profilers.emplace_back("      ↳ SOC");
  solve_profilers.emplace_back("    ↳ next iter prep");
  solve_profilers.emplace_back("    ↳ f(x)");
  solve_profilers.emplace_back("    ↳ ∇f(x)");
  solve_profilers.emplace_back("    ↳ ∇²ₓₓL");
  solve_profilers.emplace_back("    ↳ cᵢ(x)");
  solve_profilers.emplace_back("    ↳ ∂cᵢ/∂x");

  auto& solver_prof = solve_profilers[0];
  auto& setup_prof = solve_profilers[1];
  auto& inner_iter_prof = solve_profilers[2];
  auto& feasibility_check_prof = solve_profilers[3];
  auto& iteration_callbacks_prof = solve_profilers[4];
  auto& μ_update_prof = solve_profilers[5];
  auto& linear_system_build_prof = solve_profilers[6];
  auto& linear_system_compute_prof = solve_profilers[7];
  auto& linear_system_solve_prof = solve_profilers[8];
  auto& line_search_prof = solve_profilers[9];
#if 0
	auto& soc_prof = solve_profilers[10];
#endif
  auto& next_iter_prep_prof = solve_profilers[11];

  // Set up profiled matrix callbacks
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  auto& f_prof = solve_profilers[12];
  auto& g_prof = solve_profilers[13];
  auto& H_prof = solve_profilers[14];
  auto& c_i_prof = solve_profilers[15];
  auto& A_i_prof = solve_profilers[16];

  InteriorPointMatrixCallbacks matrices{
      [&](const Eigen::VectorXd& x) -> double {
        ScopedProfiler prof{f_prof};
        return matrix_callbacks.f(x);
      },
      [&](const Eigen::VectorXd& x) -> Eigen::SparseVector<double> {
        ScopedProfiler prof{g_prof};
        return matrix_callbacks.g(x);
      },
      [&](const Eigen::VectorXd& x, const Eigen::VectorXd& v, double μ,
          double β_1) -> Eigen::SparseMatrix<double> {
        ScopedProfiler prof{H_prof};
        return matrix_callbacks.H(x, v, μ, β_1);
      },
      [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        ScopedProfiler prof{c_i_prof};
        return matrix_callbacks.c_i(x);
      },
      [&](const Eigen::VectorXd& x) -> Eigen::SparseMatrix<double> {
        ScopedProfiler prof{A_i_prof};
        return matrix_callbacks.A_i(x);
      }};
#else
  const auto& matrices = matrix_callbacks;
#endif

  solver_prof.start();
  setup_prof.start();

  double f = matrices.f(x);
  Eigen::VectorXd c_i = matrices.c_i(x);

  int num_decision_variables = x.rows();
  int num_inequality_constraints = c_i.rows();

  Eigen::SparseVector<double> g = matrices.g(x);
  Eigen::SparseMatrix<double> A_i = matrices.A_i(x);

  // Proportionality constant for decreasing infeasibility proportional to μ
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
  Eigen::VectorXd w = bound_constraint_mask.select(
      Eigen::VectorXd::Ones(num_inequality_constraints),
      Eigen::VectorXd::Zero(num_inequality_constraints));
#else
  Eigen::VectorXd w = Eigen::VectorXd::Ones(num_inequality_constraints);
#endif

  // Shifted log-barrier shift parameter
  const double β_1 = 0.1;

  // Barrier parameter minimum
  const BarrierParameter μ_min = options.tolerance / 10.0;

  // Barrier parameter μ
  BarrierParameter μ = 0.0;

  // For reasons given in [7], we require that all iterates satisfy
  //   cᵢ − √(μ)e⁻ᵛ + μw = 0,
  // so we set v⁰ to satisfy this:
  //   cᵢ − √(μ)e⁻ᵛ⁰ + μw = 0
  //   √(μ)e⁻ᵛ⁰ = cᵢ + μw
  //   e⁻ᵛ⁰ = 1/√(μ)(cᵢ + μw)
  //   −v⁰ = ln(1/√(μ)(cᵢ + μw))
  //   v⁰ = −ln(1/√(μ)(cᵢ + μw))
  Eigen::VectorXd v =
      -((1.0 / μ_min.sqrt()) * (c_i + μ * w)).array().log().matrix();

  // eᵛ
  Eigen::VectorXd exp_v{v.array().exp().matrix()};
  // e⁻ᵛ
  Eigen::VectorXd exp_neg_v = exp_v.cwiseInverse();
  // e²ᵛ
  Eigen::VectorXd exp_2v{(2 * v).array().exp().matrix()};
  // s = √(μ)e⁻ᵛ
  Eigen::VectorXd s = μ.sqrt() * exp_neg_v;

  Eigen::SparseMatrix<double> H = matrices.H(x, v, μ, β_1);

  // Ensure matrix callback dimensions are consistent
  slp_assert(g.rows() == num_decision_variables);
  slp_assert(A_i.rows() == num_inequality_constraints);
  slp_assert(A_i.cols() == num_decision_variables);
  slp_assert(H.rows() == num_decision_variables);
  slp_assert(H.cols() == num_decision_variables);

  // Check whether initial guess has finite f(xₖ), and cᵢ(xₖ)
  if (!std::isfinite(f) || !c_i.allFinite()) {
    return ExitStatus::NONFINITE_INITIAL_COST_OR_CONSTRAINTS;
  }

  int iterations = 0;

  Filter filter;

  // Kept outside the loop so its storage can be reused
  gch::small_vector<Eigen::Triplet<double>> triplets;

  RegularizedLDLT solver{num_decision_variables, 0};
  Eigen::SparseMatrix<double> lhs(num_decision_variables,
                                  num_decision_variables);
  Eigen::VectorXd rhs{x.rows()};

  setup_prof.stop();

  auto build_and_compute_lhs = [&]() -> ExitStatus {
    ScopedProfiler linear_system_build_profiler{linear_system_build_prof};

    // lhs = H + Aᵢᵀdiag(e²ᵛ)Aᵢ
    //
    // Don't assign the upper triangle because solver only uses the lower
    // triangle.
    // TODO: This can also be written as a sparse rank update with Aᵢᵀdiag(eᵛ);
    // this is only accelerated by Eigen if we build with LAPACK, however.
    lhs = H + (A_i.transpose() * exp_2v.asDiagonal() * A_i)
                  .triangularView<Eigen::Lower>();

    linear_system_build_profiler.stop();
    ScopedProfiler linear_system_compute_profiler{linear_system_compute_prof};

    // Regularize and factorize the lhs we just computed of the Newton-KKT
    // system
    if (solver.compute(lhs).info() != Eigen::Success) {
      return ExitStatus::FACTORIZATION_FAILED;
    } else {
      return ExitStatus::SUCCESS;
    }
  };

  auto build_rhs = [&](const BarrierParameter& μ) {
    // rhs = -∇f + Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘(cᵢ + μw) − μβ₁e))
    const Eigen::VectorXd μβ_1e =
        Eigen::VectorXd::Constant(num_inequality_constraints, μ * β_1);
    rhs = -g + A_i.transpose() * (2.0 * μ.sqrt() * exp_v -
                                  exp_2v.asDiagonal() * (c_i + μ * w) - μβ_1e);
  };

  auto compute_step = [&](const BarrierParameter& μ) -> Step {
    Step step;

    step.p_x = solver.solve(rhs);
    // pᵥ = e − 1/√(μ) eᵛ∘(Aᵢpₓ + cᵢ − μw).
    step.p_v =
        1.0 +
        exp_v.cwiseProduct((-1.0 / μ.sqrt()) * (A_i * step.p_x + c_i - μ * w))
            .array();

    return step;
  };

  // Initializes the barrier parameter for the current iterate.
  //
  // Returns true on success and false on failure.
  auto init_barrier_parameter = [&] {
    build_rhs(BarrierParameter::from_sqrt(1e15));
    Eigen::VectorXd p_v_0 = compute_step(BarrierParameter::from_sqrt(1e15)).p_v;
    build_rhs(1.0);
    Eigen::VectorXd p_v_1 = compute_step(1.0).p_v - p_v_0;

    // See section 3.2.3 of [6]
    if (double dot = p_v_0.transpose() * p_v_1; dot < 0.0) {
      μ = BarrierParameter::from_sqrt(
          std::max(μ_min.sqrt(), p_v_1.squaredNorm() / -dot));
    } else {
      // Initialization failed, so use a hardcoded value for μ instead
      μ = BarrierParameter::from_sqrt(10.0);
    }
  };

  // Updates the barrier parameter for the current iterate and resets the
  // filter.
  //
  // This should be run when the error estimate is below a desired
  // threshold for the current barrier parameter.
  auto update_barrier_parameter = [&] {
    if (μ.sqrt() == μ_min.sqrt()) {
      return;
    }

    bool found_μ = false;

    if (is_nlp) {
      // Binary search for smallest μ such that |pᵛ|_∞ ≤ 1 starting from
      // the current value of μ. If one doesn't exist, keep the original.

      constexpr double sqrt_μ_line_search_tol = 1e-8;

      BarrierParameter μ_lower = 0.0;
      BarrierParameter μ_upper = μ;

      while (μ_upper.sqrt() - μ_lower.sqrt() > sqrt_μ_line_search_tol) {
        // Search bias that determines which side of range to check. < 0.5
        // is closer to lower bound and > 0.5 is closer to upper bound.
        constexpr double search_bias = 0.75;

        const auto μ_mid =
            BarrierParameter::from_sqrt((1.0 - search_bias) * μ_lower.sqrt() +
                                        search_bias * μ_upper.sqrt());

        build_rhs(μ_mid);
        const Eigen::VectorXd p_v = compute_step(μ_mid).p_v;
        const double p_v_infnorm = p_v.lpNorm<Eigen::Infinity>();

        if (p_v_infnorm <= 1.0) {
          // If step down was successful, decrease upper bound and try
          // again
          μ = μ_mid;
          μ_upper = μ_mid;
          found_μ = true;

          // If μ hit minimum, stop searching
          if (μ <= μ_min) {
            μ = μ_min;
            break;
          }
        } else {
          // Otherwise, increase lower bound and try again
          μ_lower = μ_mid;
        }
      }
    } else {
      // Line search for smallest μ such that |pᵛ|_∞ ≤ 1. If one doesn't
      // exist, keep the original.
      //
      // For quadratic models, this only requires two system solves
      // instead of a binary search.

      constexpr double dinf_bound = 0.99;

      build_rhs(1e15);
      Eigen::VectorXd p_v_0 = compute_step(1e15).p_v;
      build_rhs(1.0);
      Eigen::VectorXd p_v_1 = compute_step(1.0).p_v - p_v_0;

      double α_μ_min = 0.0;
      double α_μ_max = 1e15;

      for (int i = 0; i < v.rows(); ++i) {
        double temp_min = (dinf_bound - p_v_0[i]) / p_v_1[i];
        double temp_max = (-dinf_bound - p_v_0[i]) / p_v_1[i];
        if (p_v_1[i] > 0.0) {
          std::swap(temp_min, temp_max);
        }

        α_μ_min = std::max(α_μ_min, temp_min);
        α_μ_max = std::min(α_μ_max, temp_max);
      }

      if (α_μ_min <= α_μ_max) {
        found_μ = true;
        μ = BarrierParameter::from_sqrt(std::max(μ_min.sqrt(), 1.0 / α_μ_max));
      }
    }

    if (found_μ) {
      // Reset the filter when the barrier parameter is updated
      filter.reset();
    }
  };

  // Variables for determining when a step is acceptable
  constexpr double α_reduction_factor = 0.75;
  constexpr double α_min = 1e-7;

  int full_step_rejected_counter = 0;

  // Error estimate
  double E_0 = std::numeric_limits<double>::infinity();

  // Prints final solver diagnostics when the solver exits
  scope_exit exit{[&] {
    if (options.diagnostics) {
      solver_prof.stop();
      if (iterations > 0) {
        print_bottom_iteration_diagnostics();
      }
      print_solver_diagnostics(solve_profilers);
    }
  }};

  double prev_p_v_infnorm = std::numeric_limits<double>::infinity();
  bool μ_initialized = false;

  // bool can_take_aggressive_step = false;

  // Watchdog (nonmonotone) variables. If a line search fails, accept up
  // to this many steps in a row in case the dual variable steps allow the
  // primal steps to make progress again.
  constexpr int watchdog_max = 1;
  int watchdog_count = 0;

  while (E_0 > options.tolerance) {
    ScopedProfiler inner_iter_profiler{inner_iter_prof};
    ScopedProfiler feasibility_check_profiler{feasibility_check_prof};

    // Check for local inequality constraint infeasibility
    if (is_inequality_locally_infeasible(A_i, c_i)) {
      if (options.diagnostics) {
        print_c_i_local_infeasibility_error(c_i);
      }

      return ExitStatus::LOCALLY_INFEASIBLE;
    }

    // Check for diverging iterates
    if (x.lpNorm<Eigen::Infinity>() > 1e10 || !x.allFinite() ||
        v.lpNorm<Eigen::Infinity>() > 1e10 || !v.allFinite()) {
      return ExitStatus::DIVERGING_ITERATES;
    }

    feasibility_check_profiler.stop();
    ScopedProfiler iteration_callbacks_profiler{iteration_callbacks_prof};

    // Call iteration callbacks
    for (const auto& callback : iteration_callbacks) {
      if (callback({iterations, x, g, H, {}, A_i})) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    iteration_callbacks_profiler.stop();

    if (auto status = build_and_compute_lhs(); status != ExitStatus::SUCCESS) {
      return status;
    }

    // Update the barrier parameter if necessary
    if (!μ_initialized) {
      init_barrier_parameter();
      μ_initialized = true;
    } else if (is_nlp) {
      double E_sqrt_μ = error_estimate(g, A_i, c_i, v, μ.sqrt());
      if (E_sqrt_μ <= 10.0 * μ) {
        ScopedProfiler μ_update_profiler{μ_update_prof};
        update_barrier_parameter();
      }
    } else if (prev_p_v_infnorm <= 1.0) {
      ScopedProfiler μ_update_profiler{μ_update_prof};
      update_barrier_parameter();
    }

    ScopedProfiler linear_system_solve_profiler{linear_system_solve_prof};

    // TODO: build for aggressive or stabiliziation step depending on
    // take_aggressive_step
    build_rhs(μ);

    // Solve the Newton-KKT system for the step
    Step step = compute_step(μ);

    linear_system_solve_profiler.stop();
    ScopedProfiler line_search_profiler{line_search_prof};

    constexpr double α_max = 1.0;
    double α = 1.0;

    // αᵛₖ = min(1, 1/|pᵛ|_∞²)
    double p_v_infnorm = step.p_v.lpNorm<Eigen::Infinity>();
    double α_v = std::min(1.0, 1.0 / (p_v_infnorm * p_v_infnorm));
    prev_p_v_infnorm = p_v_infnorm;

    // Loop until a step is accepted
    while (1) {
      Eigen::VectorXd trial_x = x + α * step.p_x;
      Eigen::VectorXd trial_v = v + α_v * step.p_v;

      double trial_f = matrices.f(trial_x);
      Eigen::VectorXd trial_c_i = matrices.c_i(trial_x);

      // If f(xₖ + αpˣₖ) or cᵢ(xₖ + αpˣₖ) aren't finite,
      // reduce step size immediately
      if (!std::isfinite(trial_f) || !trial_c_i.allFinite()) {
        // Reduce step size
        α *= α_reduction_factor;

        if (α < α_min) {
          return ExitStatus::LINE_SEARCH_FAILED;
        }
        continue;
      }

      Eigen::VectorXd trial_s;
      if (options.feasible_ipm && c_i.cwiseGreater(0.0).all()) {
        // TODO: this, updated to use cᵢ − √(μ)e⁻ᵛ = -μw, should always happen

        // If the inequality constraints are all feasible, prevent them
        // from becoming infeasible again.
        //
        //   cᵢ − √(μ)e⁻ᵛ = 0
        //   √(μ)e⁻ᵛ = cᵢ
        //   e⁻ᵛ = 1/√(μ) cᵢ
        //   −v = ln(1/√(μ) cᵢ)
        //   v = −ln(1/√(μ) cᵢ)
        trial_s = c_i;
        trial_v = -(c_i * (1.0 / μ.sqrt())).array().log().matrix();
      } else {
        trial_s = μ.sqrt() * (-trial_v).array().exp().matrix();
      }

      // Check whether filter accepts trial iterate
      if (filter.try_add(FilterEntry{trial_f, trial_v, trial_c_i, μ.sqrt()},
                         α)) {
        // Accept step
        watchdog_count = 0;
        break;
      }

#if 0
      double prev_constraint_violation = (c_i - s).lpNorm<1>();
      double next_constraint_violation = (trial_c_i - trial_s).lpNorm<1>();

      // Second-order corrections
      //
      // If first trial point was rejected and constraint violation stayed
      // the same or went up, apply second-order corrections
      if (α == α_max &&
          next_constraint_violation >= prev_constraint_violation) {
        // Apply second-order corrections. See section 2.4 of [2].
        auto soc_step = step;

        double α_v_soc = α_v;
        Eigen::VectorXd c_e_soc = c_e;

        bool step_acceptable = false;
        for (int soc_iteration = 0; soc_iteration < 5 && !step_acceptable;
             ++soc_iteration) {
          ScopedProfiler soc_profiler{soc_prof};

          scope_exit soc_exit{[&] {
            soc_profiler.stop();

            if (options.diagnostics) {
              print_iteration_diagnostics(
                  iterations,
                  step_acceptable ? IterationType::ACCEPTED_SOC
                                  : IterationType::REJECTED_SOC,
                  soc_profiler.current_duration(),
                  error_estimate(g, A_e, trial_c_e, trial_y), trial_f,
                  trial_c_e.lpNorm<1>() + (trial_c_i - trial_s).lpNorm<1>(),
                  sqrt_μ * sqrt_μ, solver.hessian_regularization(), 1.0, 1.0,
                  α_reduction_factor, α_v_soc);
            }
          }};

          // Rebuild Newton-KKT rhs with updated constraint values.
          //
          // rhs = −[∇f − Aₑᵀy − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘cᵢ)]
          //        [              cₑˢᵒᶜ              ]
          //
          // where cₑˢᵒᶜ = c(xₖ) + c(xₖ + αpˣₖ)
          c_e_soc += trial_c_e;
          rhs.bottomRows(y.rows()) = -c_e_soc;

          // Solve the Newton-KKT system
          soc_step = compute_step(sqrt_μ);

          trial_x = x + soc_step.p_x;
          trial_y = y + soc_step.p_y;

          // αᵛₖ = 1/max(1, |pᵛ|_∞²)
          double p_v_infnorm = step.p_v.lpNorm<Eigen::Infinity>();
          α_v_soc = 1.0 / std::max(1.0, p_v_infnorm * p_v_infnorm);

          trial_v = v + α_v_soc * soc_step.p_v;
          trial_s = sqrt_μ * (-trial_v).array().exp().matrix();

          trial_f = matrices.f(trial_x);
          trial_c_e = matrices.c_e(trial_x);
          trial_c_i = matrices.c_i(trial_x);

          // Constraint violation scale factor for second-order
          // corrections
          constexpr double κ_soc = 0.99;

          // If constraint violation hasn't been sufficiently reduced,
          // stop making second-order corrections
          next_constraint_violation =
              trial_c_e.lpNorm<1>() + (trial_c_i - trial_s).lpNorm<1>();
          if (next_constraint_violation > κ_soc * prev_constraint_violation) {
            break;
          }

          // Check whether filter accepts trial iterate
          if (filter.try_add(
                  FilterEntry{trial_f, trial_v, trial_c_e, trial_c_i, sqrt_μ},
                  α)) {
            step = soc_step;
            α = 1.0;
            α_v = α_v_soc;
            step_acceptable = true;
          }
        }

        if (step_acceptable) {
          // Accept step
          watchdog_count = 0;
          break;
        }
      }
#endif

      // If we got here and α is the full step, the full step was
      // rejected. Increment the full-step rejected counter to keep track
      // of how many full steps have been rejected in a row.
      if (α == α_max) {
        ++full_step_rejected_counter;
      }

      // If the full step was rejected enough times in a row, reset the
      // filter because it may be impeding progress.
      //
      // See section 3.2 case I of [2].
      if (full_step_rejected_counter >= 4 &&
          filter.max_constraint_violation >
              filter.back().constraint_violation / 10.0) {
        filter.max_constraint_violation *= 0.1;
        filter.reset();
        continue;
      }

      // Reduce step size
      α *= α_reduction_factor;

      // If step size hit a minimum, check if the KKT error was reduced.
      // If it wasn't, report line search failure.
      if (α < α_min) {
        double current_kkt_error = kkt_error(g, A_i, c_i, v, μ.sqrt());

        trial_x = x + α_max * step.p_x;
        trial_v = v + α_v * step.p_v;

        trial_c_i = matrices.c_i(trial_x);

        double next_kkt_error =
            kkt_error(matrices.g(trial_x), matrices.A_i(trial_x), trial_c_i,
                      trial_v, μ.sqrt());

        // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway
        if (next_kkt_error <= 0.999 * current_kkt_error) {
          α = α_max;

          // Accept step
          watchdog_count = 0;
          break;
        }

        // If the dual step is making progress, accept the whole step
        // anyway
        if (p_v_infnorm > α_min && watchdog_count < watchdog_max) {
          // Accept step
          ++watchdog_count;
          break;
        }

        return ExitStatus::LINE_SEARCH_FAILED;
      }
    }

    line_search_profiler.stop();

    // If full step was accepted, reset full-step rejected counter
    if (α == α_max) {
      full_step_rejected_counter = 0;
    }

    // xₖ₊₁ = xₖ + αₖpˣₖ
    // vₖ₊₁ = vₖ + αᵛₖpᵛₖ
    x += α * step.p_x;
    v += α_v * step.p_v;

    exp_v = v.array().exp().matrix();
    exp_neg_v = exp_v.cwiseInverse();
    exp_2v = exp_v.cwiseProduct(exp_v);
    s = μ.sqrt() * exp_neg_v;

    // Update autodiff for Jacobians and Hessian
    f = matrices.f(x);
    A_i = matrices.A_i(x);
    g = matrices.g(x);
    H = matrices.H(x, v, μ.sqrt(), β_1);

    ScopedProfiler next_iter_prep_profiler{next_iter_prep_prof};

    c_i = matrices.c_i(x);

    // Update the error estimate
    E_0 = error_estimate(g, A_i, c_i, v, μ_min.sqrt());

    next_iter_prep_profiler.stop();
    inner_iter_profiler.stop();

    if (options.diagnostics) {
      print_iteration_diagnostics(iterations, IterationType::NORMAL,
                                  inner_iter_profiler.current_duration(), E_0,
                                  f, (c_i - s).lpNorm<1>(), μ,
                                  solver.hessian_regularization(), α, α_max,
                                  α_reduction_factor, α_v);
    }

    ++iterations;

    // Check for max iterations
    if (iterations >= options.max_iterations) {
      return ExitStatus::MAX_ITERATIONS_EXCEEDED;
    }

    // Check for max wall clock time
    if (std::chrono::steady_clock::now() - solve_start_time > options.timeout) {
      return ExitStatus::TIMEOUT;
    }
  }

  return ExitStatus::SUCCESS;
}

}  // namespace slp
