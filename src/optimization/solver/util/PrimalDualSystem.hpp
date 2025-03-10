// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "optimization/solver/util/ModifiedLagrangian.hpp"
#include "optimization/RegularizedLDLT.hpp"
#include "sleipnir/autodiff/Hessian.hpp"
#include "sleipnir/util/Assert.hpp"

// See docs/algorithms.md#Works_cited for citation definitions

namespace sleipnir {

// TODO: rename to PrimalDualSystem.hpp.

/*
 * Form the Schur compliment wrt the upper block of a symmetric version of the
 * LHS of the original primal-dual system (symmetrized by block elimination),
 * denoted M in equation 9 in [4]; we return only the lower-triangle, since the
 * matrix is self-adjoint and we don't use the upper triangle during
 * factorization.
 *
 * @param M The output of this function; we fill only the lower triangle of this
 *   matrix.
 * @param hessianL An object we'll use to recompute the Hessian ∇²ₓₓL_μ(x, y) at
 *   whatever values the autodiff x, y, and μ variables are set to.
 * @param A_i The Jacobian of the constraints, which should have been calculated
 *   at such that xAD.Value() == x.
 * @param y The Lagrange multipliers, which should have the same value as the
 *   value of yAD.
 * @param s The slack variables, which should have the same value as the value
 *   of sAD.
 */
inline void ComputePrimalDualLHS(Eigen::SparseMatrix<double>& M,
                                 Hessian& hessianL,
                                 const Eigen::SparseMatrix<double>& A_i,
                                 Eigen::VectorXd y, Eigen::VectorXd s) {
  // Fills the sparse matrix relatively efficiently, from triplets.
  M = hessianL.Value();
  // Computes an update to the Hessian to give the Schur compliment of the
  // primal-dual system; the term AᵢᵀYS⁻¹Aᵢ can produce unfortunate amounts of
  // fill-in on some problems. We may want to avoid the Schur compliment for
  // computing steps in these cases.
  M += (A_i.transpose() * y.cwiseQuotient(s).asDiagonal() * A_i)
           .triangularView<Eigen::Lower>();
  // NB: The above can  be written as a sparse rank update with Aᵢᵀ(YS⁻¹)⁰ˑ⁵,
  // although this is only currently accelerated on a dense selfadjointView when
  // Eigen is enabled with LAPACK, and possibly has conditioning issues with
  // very small multipliers because of the square root.
}

/*
 * Computes a primal-dual step for the barrier problem ψ_{γμ} (as given in [4])
 * at an iterate denoted (x, s, y, μ, γ) by re-using the δ-regularized
 * factorization of M + δI, where M was computed at a potentially different (!)
 * iterate (x̂, ŝ, ŷ, μ̂, γ); the matrix M is the Schur-compliment of a
 * block reduced version of the primal-dual system for the barrier problem
 * ψ_{γμ̂} at (x̂, ŝ, ŷ). This is equivalent to computing a primal-dual step
 * using a γμ-centered right-hand side at (x, s, y) and a left-hand side
 * given by the primal-dual system for a barrier problem ψ_{γμ̂} at (x̂, ŝ, ŷ).
 *
 * @param solver The solver which has already performed a successful regularized
 *   factorization of M at the iterate (x̂, ŝ, ŷ, μ̂, γ); we will use the
 *   factorization to compute a step with RHS given by (x, s, y, μ, γ).
 * @param y The Lagrange multipliers at the current iterate.
 * @param ŷ The Lagrange multipliers at the factorization iterate, which can be
 *   the same as the current iterate (i.e., y = ŷ).
 * @param s The slack variables at the current iterate.
 * @param ŝ The slack variables at the factorization iterate, which can be the
 *   same as the current iterate (i.e., s = ŝ).
 * @param w The constraint violation allowance. If the element wⱼ = 0 and the
 *   j-th constraint was feasible at (x, s, y, μ, γ), then the step we compute
 *   will keep the j-th constraint feasible; otherwise, the targeted constraint
 *   violation is proportional to wⱼ.
 * @param b_D The gradient of the Lagrangian with barrier parameter γμ computed
 *   at the current iterate, i.e., b_D = ∇ₓL_{γμ}(x, y). See equation (7) in
 *   [4].
 * @param A_iHat The constraint Jacobian computed at the factorization iterate.
 * @param μ The barrier parameter at the current iterate.
 * @param γ The centering parameter, which we assume is the same at the current
 *   and factorization iterates.
 */
inline std::pair<Eigen::VectorXd, Eigen::VectorXd> PrimalDualStep(
    RegularizedLDLT& solver, const Eigen::VectorXd& y,
    const Eigen::VectorXd& yHat, const Eigen::VectorXd& s,
    const Eigen::VectorXd& sHat, const Eigen::VectorXd& w,
    const Eigen::VectorXd& b_D, const Eigen::SparseMatrix<double>& A_iHat,
    const double μ, const double γ) {
  // We assume the user has already performed a successful regularized
  // factorization of M before trying to compute a step.
  Assert(solver.Info() == Eigen::Success);

  // b_P = (1 − γ)μw
  Eigen::VectorXd b_P = (1 - γ) * μ * w;
  // We form b_C implicitly in the following expressions to avoid a few
  // extra multiplications.

  // See equations (26a) and (26c) in [4]. We do not use a primal dual update
  // for s, so we don't compute or return dₛ here.
  Eigen::VectorXd rhs =
      -b_D - A_iHat.transpose() * (((b_P - s).cwiseProduct(y).array() + γ * μ)
                                       .matrix()
                                       .cwiseQuotient(sHat));
  Eigen::VectorXd d_x = solver.Solve(rhs);
  Eigen::VectorXd d_y = ((A_iHat * d_x + b_P).cwiseProduct(yHat) -
                         (y.cwiseProduct(s).array() - γ * μ).matrix())
                            .cwiseQuotient(sHat);
  return std::make_pair(d_x, d_y);
};

}  // namespace sleipnir
