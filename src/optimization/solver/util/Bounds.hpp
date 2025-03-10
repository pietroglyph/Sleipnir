// Copyright (c) Sleipnir contributors

#pragma once

#include "Eigen/SparseCore"
#include "sleipnir/autodiff/Variable.hpp"

#include "sleipnir/util/Assert.hpp"

#include <span>
#include <algorithm>

#include <Eigen/Core>

// See docs/algorithms.md#Works_cited for citation definitions

namespace sleipnir {

//
/**
 * A "bound constraint" is any linear constraint in one scalar variable.
 * Computes which constraints, if any, are bound constraints, whether or not
 * they're feasible (given previously encountered bounds), and the tightest
 * bounds on each decision variable.
 *
 * @param decisionVariables Decision variables corresponding to each column of
 *   A_i.
 * @param inequalityConstraints Variables representing the left-hand side of
 *   cᵢ(decisionVariables) ≤ 0.
 * @param A_i The Jacobian of inequalityConstraints wrt decisionVariables.
 */
inline std::tuple<Eigen::VectorXd, small_vector<std::pair<double, double>>,
                  small_vector<std::pair<Eigen::Index, Eigen::Index>>>
GetBounds(const std::span<Variable> decisionVariables,
          const std::span<Variable> inequalityConstraints,
          const Eigen::SparseMatrix<double, Eigen::ColMajor>& A_i) {
  // A blocked, out-of-place transpose should be much faster than traversing row
  // major on a column major matrix unless we have few linear constraints (using
  // a heuristic to choose between this and staying column major based on the
  // number of constraints would be an easy performance improvement.)
  Eigen::SparseMatrix<double, Eigen::RowMajor> rowmajA_i{A_i};
  // NB: Casting to long is unspecified if the size of decisionVariable.size()
  // is greater than the max long value, but then we wouldn't be able to fill
  // A_i correctly anyway.
  Assert(static_cast<long>(decisionVariables.size()) == rowmajA_i.innerSize());
  Assert(static_cast<long>(inequalityConstraints.size()) ==
         rowmajA_i.outerSize());

  // Used only to report conflicting bounds
  small_vector<std::pair<Eigen::Index, Eigen::Index>>
      decisionVarToConstraintIndices{decisionVariables.size(), {-1, -1}};
  // We return these three
  Eigen::VectorXd boundConstraintMask{inequalityConstraints.size()};
  boundConstraintMask.fill(1.0);
  small_vector<std::pair<double, double>> decisionVarToBounds{
      decisionVariables.size(),
      {-std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity()}};
  small_vector<std::pair<Eigen::Index, Eigen::Index>> conflictingBounds;
  for (decltype(inequalityConstraints)::size_type constraintIndex = 0;
       constraintIndex < inequalityConstraints.size(); constraintIndex++) {
    // Claim: A constraint is a bound iff it's a linear function in one variable
    // and its gradient has a single nonzero value.
    if (inequalityConstraints[constraintIndex].Type() !=
        ExpressionType::kLinear) {
      continue;
    }
    const Eigen::SparseVector<double>& rowA =
        rowmajA_i.innerVector(constraintIndex);
    const auto nonZeros = rowA.nonZeros();
    Assert(nonZeros != 0);
    if (nonZeros > 1) {
      // Constraint is in more than one variable.
      continue;
    }

    // Claim: The bound is given by a bound constraint is the constraint
    // evaluated at zero divided by the nonzero element of the constraint's
    // gradient.
    // Proof: If c(x) is a bound constraint, then by definition c is a linear
    // function in one variable, hence there exist a, b ∈ ℝ s.t. c(x) = axᵢ + b
    // and a ≠ 0. The gradient of c is then aeᵢ (where eᵢ denotes the i-th basis
    // element), and c(0) = b. If c(x) ≤ 0, then since either a < 0 or a > 0, we
    // have either x ≥ -b/a or x ≤ -b/a, respectively. ∎
    Eigen::SparseVector<double>::InnerIterator rowIter(rowA);
    const auto constraintCoefficient =
        rowIter
            .value();  // The nonzero value of the j-th constraint's gradient.
    const auto decisionVariableIndex = rowIter.index();
    const auto decisionVariableValue =
        decisionVariables[decisionVariableIndex].Value();
    double constraintConstant;
    if (decisionVariableValue != 0) {
      decisionVariables[decisionVariableIndex].SetValue(0);
      constraintConstant = inequalityConstraints[constraintIndex].Value();
      decisionVariables[decisionVariableIndex].SetValue(decisionVariableValue);
    } else {
      constraintConstant = inequalityConstraints[constraintIndex].Value();
    }
    Assert(constraintCoefficient !=
           0);  // Shouldn't happen since the constraint is
                // supposed to be linear and not a constant.

    // Update bounds
    auto& [lowerBound, upperBound] = decisionVarToBounds[decisionVariableIndex];
    auto& [lowerIndex, upperIndex] =
        decisionVarToConstraintIndices[decisionVariableIndex];
    // Let xⱼ = decisionVariables[decisionVariableIndex]; we have
    //   constraintCoefficient * xⱼ + constraintConstant ≤ 0,
    // which implies that
    //   xⱼ ≤ -constraintConstant/constraintCoefficient,
    // assuming cᵢ(x) ≤ 0.
    const auto detectedBound = -constraintConstant / constraintCoefficient;
    if (constraintCoefficient < 0 && detectedBound > lowerBound) {
      lowerBound = detectedBound;
      lowerIndex = constraintIndex;
    } else if (constraintCoefficient > 0 && detectedBound < upperBound) {
      upperBound = detectedBound;
      upperIndex = constraintIndex;
    }

    // Update conflicting bounds
    if (lowerBound > upperBound) {
      conflictingBounds.emplace_back(lowerIndex, upperIndex);
    }

    // We track this to set wⱼ = 0 later
    boundConstraintMask[constraintIndex] = 0;
  }
  return {boundConstraintMask, decisionVarToBounds, conflictingBounds};
}

/**
 * Projects the decision variables onto the given bounds, while ensuring some
 * configurable distance from the boundary if possible. This is designed to
 * match the algorithim given in section 3.6 of [2].
 *
 * @param x A vector of decision variables.
 * @param bounds An array of bounds (stored [lower, upper]) for each decision
 *   variable in x (implicitly a map from decision variable index to bound).
 * @param κ_1 A constant controlling distance from the lower or upper bound when
 *   the difference between the upper and lower bound is small.
 * @param κ_2 A constant controlling distance from the lower or upper bound when
 *   the difference between the upper and lower bound is large (including when
 *   one of the bounds is ±∞).
 */
template <typename D>
  requires(EigenMatrixLike<D> &&
           static_cast<bool>(Eigen::MatrixBase<D>::IsVectorAtCompileTime))
inline void ProjectOntoBounds(
    Eigen::MatrixBase<D>& x,
    const std::span<const std::pair<typename Eigen::MatrixBase<D>::Scalar,
                                    typename Eigen::MatrixBase<D>::Scalar>>
        bounds,
    const typename Eigen::MatrixBase<D>::Scalar κ_1 = 1e-2,
    const typename Eigen::MatrixBase<D>::Scalar κ_2 = 1e-2) {
  Assert(κ_1 > 0 && κ_2 > 0 && κ_2 < 0.5);

  Eigen::Index idx = 0;
  for (const auto& [lower, upper] : bounds) {
    typename Eigen::MatrixBase<D>::Scalar& x_i = x[idx++];

    // We assume that bound infeasibility is handled elsewhere.
    Assert(lower <= upper);

    // See B.2 in [4] and section 3.6 in [2]; compare to
    // https://github.com/ohinder/OnePhase.jl/blob/4863f8146f1454c353118b3f12b1784dfea60032/src/init/primal-project.jl#L1-L68,
    // which writes this in equivalent way that is more concise but less
    // clear. Although the equations are the same as in [2], since we
    // convert equalities to bounds, this results into projecting exactly onto
    // the equalities.
    if (std::isfinite(lower) && std::isfinite(upper)) {
      auto p_L =
          std::min(κ_1 * std::max(1.0, std::abs(lower)), κ_2 * (upper - lower));
      auto p_U =
          std::min(κ_1 * std::max(1.0, std::abs(upper)), κ_2 * (upper - lower));
      x_i = std::min(std::max(lower + p_L, x_i), upper - p_U);
    } else if (std::isfinite(lower)) {
      x_i = std::max(x_i, lower + κ_1 * std::max(1.0, std::abs(lower)));
    } else if (std::isfinite(upper)) {
      x_i = std::min(x_i, upper - κ_1 * std::max(1.0, std::abs(upper)));
    }
  }
}

}  // namespace sleipnir
