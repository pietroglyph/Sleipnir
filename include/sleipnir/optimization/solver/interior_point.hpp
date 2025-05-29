// Copyright (c) Sleipnir contributors

#pragma once

#include <functional>
#include <span>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

/**
 * Matrix callbacks for the interior-point method solver.
 */
struct SLEIPNIR_DLLEXPORT InteriorPointMatrixCallbacks {
  /// Cost function value f(x) getter.
  ///
  /// <table>
  ///   <tr>
  ///     <th>Variable</th>
  ///     <th>Rows</th>
  ///     <th>Columns</th>
  ///   </tr>
  ///   <tr>
  ///     <td>x</td>
  ///     <td>num_decision_variables</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>f(x)</td>
  ///     <td>1</td>
  ///     <td>1</td>
  ///   </tr>
  /// </table>
  std::function<double(const Eigen::VectorXd& x)> f;

  /// Cost function gradient ∇f(x) getter.
  ///
  /// <table>
  ///   <tr>
  ///     <th>Variable</th>
  ///     <th>Rows</th>
  ///     <th>Columns</th>
  ///   </tr>
  ///   <tr>
  ///     <td>x</td>
  ///     <td>num_decision_variables</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>∇f(x)</td>
  ///     <td>num_decision_variables</td>
  ///     <td>1</td>
  ///   </tr>
  /// </table>
  std::function<Eigen::SparseVector<double>(const Eigen::VectorXd& x)> g;

  /// Lagrangian Hessian ∇ₓₓ²L(x, v, μ, β₁) getter, where
  ///
  /// L(x, s, z, μ) =
  ///   f(x) − μ ∑ [β₁(cᵢ)ⱼ(x) − ln(√(μ))vⱼ] − √(μ)eᵛᵀ(cᵢ(x) − √(μ)e⁻ᵛ + μw),
  ///            j
  ///
  /// ∇ₓL(x, v, μ, β₁) = ∇ₓf(x) − Aᵢ(x)ᵀ(√(μ)eᵛ − μβ₁e),
  ///
  /// ∇ₓₓ²L(x, v, μ, β₁) = ∇ₓₓ²f(x) − ∇ₓₓ²cᵢ(x)ᵀ(√(μ)eᵛ − μβ₁e).
  ///
  /// <table>
  ///   <tr>
  ///     <th>Variable</th>
  ///     <th>Rows</th>
  ///     <th>Columns</th>
  ///   </tr>
  ///   <tr>
  ///     <td>x</td>
  ///     <td>num_decision_variables</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>v</td>
  ///     <td>num_inequality_constraints</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>μ</td>
  ///     <td>1</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>β₁</td>
  ///     <td>1</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>∇ₓₓ²L(x, y, z, μ, β₁)</td>
  ///     <td>num_decision_variables</td>
  ///     <td>num_decision_variables</td>
  ///   </tr>
  /// </table>
  std::function<Eigen::SparseMatrix<double>(
      const Eigen::VectorXd& x, const Eigen::VectorXd& v, double μ, double β_1)>
      H;

  /// Inequality constraint value cᵢ(x) getter.
  ///
  /// <table>
  ///   <tr>
  ///     <th>Variable</th>
  ///     <th>Rows</th>
  ///     <th>Columns</th>
  ///   </tr>
  ///   <tr>
  ///     <td>x</td>
  ///     <td>num_decision_variables</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>cᵢ(x)</td>
  ///     <td>num_inequality_constraints</td>
  ///     <td>1</td>
  ///   </tr>
  /// </table>
  std::function<Eigen::VectorXd(const Eigen::VectorXd& x)> c_i;

  /// Inequality constraint Jacobian ∂cᵢ/∂x getter.
  ///
  /// @verbatim
  ///         [∇ᵀcᵢ₁(xₖ)]
  /// Aᵢ(x) = [∇ᵀcᵢ₂(xₖ)]
  ///         [    ⋮    ]
  ///         [∇ᵀcᵢₘ(xₖ)]
  /// @endverbatim
  ///
  /// <table>
  ///   <tr>
  ///     <th>Variable</th>
  ///     <th>Rows</th>
  ///     <th>Columns</th>
  ///   </tr>
  ///   <tr>
  ///     <td>x</td>
  ///     <td>num_decision_variables</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>Aᵢ(x)</td>
  ///     <td>num_inequality_constraints</td>
  ///     <td>num_decision_variables</td>
  ///   </tr>
  /// </table>
  std::function<Eigen::SparseMatrix<double>(const Eigen::VectorXd& x)> A_i;
};

/**
Finds the optimal solution to a nonlinear program using the interior-point
method.

A nonlinear program has the form:

@verbatim
     min_x f(x)
subject to cᵢ(x) ≥ 0
@endverbatim

where f(x) is the cost function and cᵢ(x) are the inequality constraints.

@param[in] matrix_callbacks Matrix callbacks.
@param[in] is_nlp If true, the solver uses a more conservative barrier parameter
  reduction strategy that's more reliable on NLPs. Pass false for problems with
  quadratic or lower-order cost and linear or lower-order constraints.
@param[in] iteration_callbacks The list of callbacks to call at the beginning of
  each iteration.
@param[in] options Solver options.
@param[in,out] x The initial guess and output location for the decision
  variables.
@return The exit status.
*/
SLEIPNIR_DLLEXPORT ExitStatus interior_point(
    const InteriorPointMatrixCallbacks& matrix_callbacks, bool is_nlp,
    std::span<std::function<bool(const IterationInfo& info)>>
        iteration_callbacks,
    const Options& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::VectorXd& x);

}  // namespace slp
