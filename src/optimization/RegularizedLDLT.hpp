// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

// See docs/algorithms.md#Works_cited for citation definitions

namespace sleipnir {

/**
 * Modifies a symmetric matrix M to be positive-definite by adding a diagonal
 * matrix δI and attempting a direct, sparse square-root-free (LDLᵀ) Cholesky
 * factorization on M + δI, increasing δ ≥ 0 until the factorization succeeds on
 * M + δI and D implies that M + δI is sufficiently positive-definite. We must
 * check for the latter condition, since the LDLᵀ form of the Cholesky
 * factorization can succeed for positive semidefinite and some indefinite
 * matrices. This process is simple, but not as efficient or as stable as
 * methods that try to compute δ during the factorization process: see P. E.
 * Gill, W. Murray, and M. H. Wright, “Practical Optimization”, Academic Press,
 * London, (1981); or, Robert B. Schnabel and Elizabeth Eskow, ”A Revised
 * Modified Cholesky Factorization Algorithm”, SIAM Journal on Optimization
 * 9(4), 1135 – 1148 (1999) for such methods.
 *
 * This process can be viewed in two equivalent ways---as approximately
 * L₂−optimal eigenvalue modification, and as L₂ regularization (in a similar
 * sense as Tikhonov regularization):
 * 1. We define an Lₚ-optimal eigenvalue modification as the solution to the
 *    problem
@verbatim
arg min_{ΔA ∈ ℝ^{n×n}} ‖ΔA‖ₚ
            subject to λₘᵢₙ(A + ΔA) ≥ τ,
@endverbatim
 *    where ‖⋅‖ₚ is a operator p-norm. If p = 2, then (from pg. 51 in [1]) the
 *    problem has the unique global solution ΔA = max{0, τ − λₘᵢₙ(A)}I, where
 *    λₘᵢₙ(A) denotes the smallest eigenvalue of A.
 *    We do not find ΔA exactly since an eigendecomposition is very expensive,
 *    but the error between δ and max{0, τ − λₘᵢₙ(A)} for τ = 0 is on the order
 *    of the amount by which we increment δ.
 * 2. An L₂-regularized log-barrier problem
@verbatim
min_{dₓ ∈ ℝⁿ} f(x + dₓ) − μ∑ln(−cᵢ(x)) + δ/2 ‖dₓ‖₂
@endverbatim
 *    has a local solution at dₓ* if the KKT conditions for
 *    min_{dₓ ∈ ℝⁿ} f(x + dₓ) subject to c(x) ≤ 0 (1) hold at dₓ*, but with
 *    the Lagrangian equal to −δdₓ* and not zero. Hence, taking a primal-dual
 *    step with δI added to the Hessian of the Lagrangian of problem (1) is
 *    equivalent to applying Newton's method to first-order necessary conditions
 *    of the L₂-regularized log-barrier problem.
 *    The above log-barrier subproblem is slightly simplified compared to the
 *    actual one in [4]; see section 2.1.1 of [4] for more details.
 * The first view shows that we're approximating an "optimally small" change to
 * the Lagrangian Hessian that makes it positive-definite, which seems
 * reasonable since we want to lose as little curvature information as possible
 * (the same reasoning motivates the method of Gill, Murray, and Wright). This
 * has few downsides when there are only very small positive or negative
 * eigenvalues (i.e., ill-conditioning and mild concavity), but when there are
 * large negative eigenvalues, these minimal perturbation approaches throw away
 * the curvature information in those directions.
 */
class RegularizedLDLT {
 public:
  using Solver = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>,
                                       Eigen::Lower, Eigen::AMDOrdering<int>>;

  /**
   * Constructs a RegularizedLDLT instance; no computation is performed until a
   * call to Compute is made..
   */
  RegularizedLDLT() = default;

  /**
   * Reports whether previous computation was successful.
   */
  Eigen::ComputationInfo Info() { return m_info; }

  /**
   * Returns the δ used for the last successful factorization.
   */
  double Delta() { return m_δOld; }

  /**
   * Computes the regularized LDLᵀ factorization of a matrix according to
   * algorithm 7 in [4]. We regularize the entire matrix, since we expect the
   * Schur compliment of the symmetrized (block-eliminated) primal-dual system,
   * instead of the symmetrized matrix where only the upper left block
   * (corresponding to the Hessian of the Lagrangian) may not be
   * positive-definite (see section 2.1.1 of [4] for more details).
   *
   * @param lhs Left-hand side of the system.
   * @param D_min Minimum componentwise value of the D matrix; set to 0 by
   *   default to match the algorithm in [4]. If set to a small positive value,
   *   this will regularize ill-conditioned but positive-definite lhs (i.e., it
   *   will choose a nonzero δ in this case).
   */
  void Compute(const Eigen::SparseMatrix<double>& lhs,
               const double D_min = 0.0) {
    static constexpr double Δ_dec = std::numbers::pi, Δ_inc = 8.0, Δ_min = 1e-8,
                            Δ_max = 1e50;
    double δ = 0.0;

    // There is a well-known theorem that if M is positive-definite, then its
    // diagonal is strictly positive elementwise; the contrapositive gives the
    // necessary condition for positive-definiteness that we check here.
    double τ = lhs.diagonal().minCoeff();
    if (τ > 0) {
      AnalyzePattern(lhs);
      m_solver.factorize(lhs);

      // See, for example, Golub and Van Loan "Matrix Computations",
      // Section 4.2.3 for a proof that the second condition implies
      // positive-definiteness.
      if (m_solver.info() == Eigen::Success &&
          m_solver.vectorD().minCoeff() > D_min) {
        m_δOld = δ;
        m_info = Eigen::Success;
        return;
      }
      τ = 0;
    }

    Eigen::SparseMatrix<double> sparseEye{lhs.rows(), lhs.cols()};
    sparseEye.setIdentity();
    δ = std::max(m_δOld * Δ_dec, Δ_min - τ);
    while (true) {
      // NB: Algorithm 7 in [4] checks this only on the first iteration, which
      // seems like a typo.
      if (δ >= Δ_max) {
        // If the Hessian perturbation is too high, report failure. We skip
        // updating δOld in this case.
        m_info = Eigen::NumericalIssue;
        return;
      }

      Eigen::SparseMatrix<double> lhsReg = lhs + δ * sparseEye;
      AnalyzePattern(lhsReg);
      m_solver.factorize(lhsReg);

      // If the inertia is ideal, store that value of δ and return.
      // Otherwise, increase δ by an order of magnitude and try again.
      if (m_solver.info() == Eigen::Success &&
          m_solver.vectorD().minCoeff() > D_min) {
        m_δOld = δ;
        m_info = Eigen::Success;
        return;
      }
      δ *= Δ_inc;
    }
  }

  /**
   * Solve the system of equations using a regularized LDLT factorization.
   *
   * @param rhs Right-hand side of the system.
   */
  template <typename Rhs>
  auto Solve(const Eigen::MatrixBase<Rhs>& rhs) {
    return m_solver.solve(rhs);
  }

 private:
  Solver m_solver;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The value of δ from the previous run of Compute().
  double m_δOld = 0.0;

  // Number of non-zeros in LHS.
  int m_nonZeros = -1;

  /**
   * Reanalize LHS matrix's sparsity pattern if it changed.
   *
   * @param lhs Matrix to analyze.
   */
  void AnalyzePattern(const Eigen::SparseMatrix<double>& lhs) {
    int nonZeros = lhs.nonZeros();
    if (m_nonZeros != nonZeros) {
      m_solver.analyzePattern(lhs);
      m_nonZeros = nonZeros;
    }
  }
};

}  // namespace sleipnir
