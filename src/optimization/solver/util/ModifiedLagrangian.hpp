// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"

// See docs/algorithms.md#Works_cited for citation definitions

namespace sleipnir {

/*
 * Form the modified Lagrangian L_μₖ(xₖ, yₖ), defined as
@verbatim
L_μₖ(xₖ, yₖ) := f(xₖ) + (yₖ - μₖβ₁e)ᵀcᵢ(xₖ),
@endverbatim
 * as given in [4]. This modified Lagrangian can be viewed as a version of the
 * Lagrangian of the original inequality-constrained problem with the Lagrange
 * multipliers shifted by β₁μ̅, or as a function given by a condition necessary
 * for optimality of a shifted log-barrier problem denoted ψ_μ̅ in [4], given by
@verbatim
ψ_μ̅(x) := f(x) − μ̅ ∑ⱼ[β₁cᵢ(x)ᵀeⱼ + ln(μ̅wᵀeⱼ − cᵢ(x)ᵀeⱼ)].
@endverbatim
 * We return an autodiff variable so that we can take the Hessian of this wrt
 * the decision variables.
 *
 * @param f An autodiff variable containing the objective function in terms of
 *   the decision variables.
 * @param y A vector of autodiff variables containing the Lagrange multipliers.
 * @param c_i A vector of autodiff variables cᵢ containting the constraint
 *   functions in terms of the decision variables, which are assumed to give the
 *   constraints in the form cᵢ(x) ≤ 0.
 * @param μ A scalar that gives an elementwise shift of the Lagrange
 *   multipliers (usually set to a scaled version of the barrier parameter in
 *   the shifted log-barrier subproblem ψ_μ̅).
 * @param β_1 A shift rescaling constant; the term β₁cᵢ(x)ᵀeⱼ in ψ_μ̅(x) keeps
 *   the shifted log-barrier bounded below.
 */
inline Variable AutodiffModifiedLagrangian(const Variable& f,
                                           const VariableMatrix& y,
                                           const VariableMatrix& c_i,
                                           const Variable& μ,
                                           const double β_1) {
  // Note that the second term is not negated, since
  // we assume cᵢ(x) ≤ 0.
  return f + ((y - μ * β_1 * VariableMatrix::Ones(y.Rows(), 1)).T() * c_i)(0);
}

/*
 * Forms the gradient of the modified Lagrangian given in [4] semi-manually.
 * This modified Lagrangian can be viewed as a version of the Lagrangian of the
 * original inequality-constrained problem with the Lagrange multipliers shifted
 * by β₁μ̅, or as a function given by a condition necessary for optimality of a
 * shifted log-barrier problem denoted ψ_μ̅ in [4], given by
@verbatim
ψ_μ̅(x) := f(x) − μ̅ ∑ⱼ[β₁cᵢ(x)ᵀeⱼ + ln(μ̅wᵀeⱼ − cᵢ(x)ᵀeⱼ)].
@endverbatim
 *
 * We form the gradient semi-manually, since we need to compute this with
 * different values of μ̅ every step, and we need to compute the full constraint
 * Jacobian anyway in order to form the Schur complement of the block-eliminated
 * primal-dual system. As a result, this is cheaper than re-computing a
 * backwards pass on L each time (i.e., fully automatic computation).
 *
 * @param g A sparse vector containing the gradient of the objective ∇f(xₖ).
 * @param A_i A sparse matrix containing the Jacobian of the inequality
 *   constraints ∇cᵢ(xₖ).
 * @param y A dense vector containing the Lagrange multipliers for every
 *   inequality constraint.
 * @param μBar A scalar that gives an elementwise shift of the Lagrange
 *   multipliers (usually set to a scaled version of the barrier parameter in
 *   the shifted log-barrier subproblem ψ_μ̅).
 * @param β_1 A shift rescaling constant; the term β₁cᵢ(x)ᵀeⱼ in ψ_μ̅(x) keeps
 *   the shifted log-barrier bounded below.
 */
inline Eigen::VectorXd ManualGradientModifiedLagrangian(
    const Eigen::SparseVector<double>& g,
    const Eigen::SparseMatrix<double>& A_i, const Eigen::VectorXd& y,
    const double μ, const double β_1) {
  return g + A_i.transpose() * (y - μ * β_1 * Eigen::VectorXd::Ones(y.size()));
};

/*
 * Forms the gradient of the modified Lagrangian given in [4] semi-manually.
 * This modified Lagrangian can be viewed as a version of the Lagrangian of the
 * original inequality-constrained problem with the Lagrange multipliers shifted
 * by β₁μ̅, or as a function given by a condition necessary for optimality of a
 * shifted log-barrier problem denoted ψ_μ̅ in [4], given by
@verbatim
ψ_μ̅(x) := f(x) − μ̅ ∑ⱼ[β₁cᵢ(x)ᵀeⱼ + ln(μ̅wᵀeⱼ − cᵢ(x)ᵀeⱼ)].
@endverbatim
 *
 * We form the gradient semi-manually, since we need to compute this with
 * different values of μ̅ every step, and we need to compute the full constraint
 * Jacobian anyway in order to form the Schur complement of the block-eliminated
 * primal-dual system. As a result, this is cheaper than re-computing a
 * backwards pass on L each time (i.e., fully automatic computation).
 *
 * @param g A sparse vector containing the gradient of the objective ∇f(xₖ).
 * @param complimentarityGradient The matrix-vector product  ∇cᵢ(xₖ)ᵀyₖ.
 * @param constraintSumGradient The matrix-vector product  ∇cᵢ(xₖ)ᵀe.
 * @param y A dense vector containing the Lagrange multipliers for every
 *   inequality constraint.
 * @param μBar A scalar that gives an elementwise shift of the Lagrange
 *   multipliers (usually set to a scaled version of the barrier parameter in
 *   the shifted log-barrier subproblem ψ_μ̅).
 * @param β_1 A shift rescaling constant; the term β₁cᵢ(x)ᵀeⱼ in ψ_μ̅(x) keeps
 *   the shifted log-barrier bounded below.
 */
inline Eigen::VectorXd ManualGradientModifiedLagrangian(
    const Eigen::SparseVector<double>& g,
    const Eigen::SparseVector<double>& gradientComplimentarity,
    const Eigen::SparseVector<double>& gradientConstraintSum, const double μ,
    const double β_1) {
  return g + gradientComplimentarity - μ * β_1 * gradientConstraintSum;
};

}  // namespace sleipnir
