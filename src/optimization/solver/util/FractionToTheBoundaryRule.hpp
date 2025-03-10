// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>

// See docs/algorithms.md#Works_cited for citation definitions

namespace sleipnir {

/**
 * Computes a componentwise lower bound for the primal iterates which can be
 * used in a fraction-to-the-boundary rule. The bound is
 *
 *   Θ min{s, ‖dₓ‖_∞ (δ + ‖d_y‖_∞ + ‖dₓ‖ᵝ₇_∞)e},
 *
 * where Θ varies depending on whether we're using this to calculate the
 * iterate-acceptance fraction-to-the-boundary rule, or a stronger rule for
 * calculating an initial primal step size.
 *
 * @param s The slack variables.
 * @param d_x The step direction on the primal variables.
 * @param d_y The step direction on the dual variables.
 * @param Θ A vector that scales the bound in a componentwise fashion; used by
 *   [4] to set a more aggressive bound duals corresponding to linear
 * constraints.
 * @param δ The regularization parameter in the L₂-regularized log-barrier
 *   subproblem, which is the same as the Lagrangian Hessian regularization
 *   parameter in [0, ∞) ⊆ ℝ.
 */
inline Eigen::VectorXd PrimalFractionToTheBoundaryRule(
    const Eigen::Ref<const Eigen::VectorXd>& s,
    const Eigen::Ref<const Eigen::VectorXd>& d_x,
    const Eigen::Ref<const Eigen::VectorXd>& d_y,
    const Eigen::Ref<const Eigen::VectorXd>& Θ, const double δ,
    const double β_7) {
  using Eigen::Infinity;
  return s
      .cwiseMin(
          d_x.lpNorm<Infinity>() *
          (δ + d_y.lpNorm<Infinity>() + std::pow(d_x.lpNorm<Infinity>(), β_7)))
      .cwiseProduct(Θ);
}

/**
 * Applies a componentwise fraction-to-the-boundary rule to the primal iterates,
 * then returns a maximum primal step size in [0, 1] to be used by the
 * backtracking line search.
 *
 * @param s The slack variables.
 * @param d_x The step direction on the primal variables.
 * @param d_y The step direction on the dual variables.
 * @param d_s The step direction on the slack variables.
 * @param δ The regularization parameter in the L₂-regularized log-barrier
 *   subproblem, which is the same as the Lagrangian Hessian regularization
 *   parameter in [0, ∞) ⊆ ℝ.
 * @return A maximum primal step size in [0, 1] for the backtracking line
 *   search.
 */
inline double MaxPrimalStepSize(
    const Eigen::Ref<const Eigen::VectorXd>& s,
    const Eigen::Ref<const Eigen::VectorXd>& d_x,
    const Eigen::Ref<const Eigen::VectorXd>& d_y,
    const Eigen::Ref<const Eigen::VectorXd>& d_s,
    const Eigen::Ref<const Eigen::VectorXd>& Θ_p, const double δ,
    const double β_7) {
  // This is a strict lower bound on the primal step size for all indices of dₛ
  // which are positive by equation (28) in [4], and an upper bound for all
  // indices of dₛ which are negative, since dividing by negative dₛ flips the
  // inequality (28) in [4]. Assumes cᵢ(x) ≤ 0, since this makes the bounds
  // positive (otherwise the bounds are flipped).
  Eigen::VectorXd bound =
      PrimalFractionToTheBoundaryRule(s, d_x, d_y, Θ_p, δ, β_7)
          .cwiseQuotient(d_s);

  // We'd like the find the maximum α_P ∈ [0, 1] that satisfies
  // s + α_P dₛ ≥ Θᵖ min{s, ‖dₓ‖_∞ (δ + ‖d_y‖_∞ + ‖dₓ‖_∞^(β₇)) e}.
  // We're only interested in finding the least *upper* bound on α_P; the bound
  // vector encodes lower and upper bounds, so we only take the minimum of the
  // upper bounds (i.e., the elements of the bound vector corresponding to
  // negative elements of dₛ).
  decltype(bound)::Index minCoeffIdx;
  const auto minBound = bound.minCoeff(&minCoeffIdx);
  if (d_s(minCoeffIdx) <= 0.0) {
    return std::max(0.0, std::min(minBound, 1.0));
  }
  return 1.0;
}

/**
 * Computes a componentwise lower bound for the dual iterates which can be
 * used in a very conventional fraction-to-the-boundary rule. The bound is
 *
 *   Θᵇy min{1, ‖dₓ‖_∞}.
 *
 * @param y The dual variables.
 * @param d_x The step direction on the primal variables.
 * @param Θᵇ A vector that scales the bound in a componentwise fashion.
 */
inline Eigen::VectorXd DualFractionToTheBoundaryRule(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::VectorXd>& d_x,
    const Eigen::Ref<const Eigen::VectorXd>& Θ) {
  using Eigen::Infinity;
  return Θ.cwiseProduct(y) * std::min(1.0, d_x.lpNorm<Infinity>());
}

inline std::pair<double, double> MinMaxDualStepSizes(
    const Eigen::Ref<const Eigen::VectorXd>& complimentarityStepGradient,
    const Eigen::Ref<const Eigen::VectorXd>& s,
    const Eigen::Ref<const Eigen::VectorXd>& d_y,
    const Eigen::Ref<const Eigen::VectorXd>& Θ_p, const double μ,
    const double β_2) {
  // Calculate B(s⁺, d_y) (idk about 29a; 29b is like the expanded primal
  // fraction-to-boundary calculation)
  // Calculate unbounded min for (30a) and then project onto B(s⁺, d_y).
  return {};
}

}  // namespace sleipnir
