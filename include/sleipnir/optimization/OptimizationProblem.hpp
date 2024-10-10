// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <iterator>
#include <optional>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"
#include "sleipnir/optimization/SolverConfig.hpp"
#include "sleipnir/optimization/SolverExitCondition.hpp"
#include "sleipnir/optimization/SolverIterationInfo.hpp"
#include "sleipnir/optimization/SolverStatus.hpp"
#include "sleipnir/optimization/solver/InteriorPoint.hpp"
#include "sleipnir/util/Print.hpp"
#include "sleipnir/util/SymbolExports.hpp"
#include "sleipnir/util/small_vector.hpp"

namespace sleipnir {

/**
 * This class allows the user to pose a constrained nonlinear optimization
 * problem in natural mathematical notation and solve it.
 *
 * This class supports problems that can be converted to the form:
@verbatim
      minₓ f(x)
subject to cₑ(x) = 0
           cᵢ(x) ≥ 0
@endverbatim
 *
 * where f(x) is the scalar cost function, x is the vector of decision variables
 * (variables the solver can tweak to minimize the cost function), cᵢ(x) are the
 * inequality constraints, and cₑ(x) are the equality constraints. Constraints
 * are equations or inequalities of the decision variables that constrain what
 * values the solver is allowed to use when searching for an optimal solution.
 *
 * The nice thing about this class is users don't have to put their system in
 * the form shown above manually; they can write it in natural mathematical form
 * and it'll be converted for them.
 */
class SLEIPNIR_DLLEXPORT OptimizationProblem {
 public:
  /**
   * Construct the optimization problem.
   */
  OptimizationProblem() noexcept = default;

  /**
   * Create a decision variable in the optimization problem.
   */
  [[nodiscard]]
  Variable DecisionVariable() {
    m_decisionVariables.emplace_back();
    return m_decisionVariables.back();
  }

  /**
   * Create a matrix of decision variables in the optimization problem.
   *
   * @param rows Number of matrix rows.
   * @param cols Number of matrix columns.
   */
  [[nodiscard]]
  VariableMatrix DecisionVariable(int rows, int cols = 1) {
    m_decisionVariables.reserve(m_decisionVariables.size() + rows * cols);

    VariableMatrix vars{rows, cols};

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        m_decisionVariables.emplace_back();
        vars(row, col) = m_decisionVariables.back();
      }
    }

    return vars;
  }

  /**
   * Create a symmetric matrix of decision variables in the optimization
   * problem.
   *
   * Variable instances are reused across the diagonal, which helps reduce
   * problem dimensionality.
   *
   * @param rows Number of matrix rows.
   */
  [[nodiscard]]
  VariableMatrix SymmetricDecisionVariable(int rows) {
    // We only need to store the lower triangle of an n x n symmetric matrix;
    // the other elements are duplicates. The lower triangle has (n² + n)/2
    // elements.
    //
    //   n
    //   Σ k = (n² + n)/2
    //  k=1
    m_decisionVariables.reserve(m_decisionVariables.size() +
                                (rows * rows + rows) / 2);

    VariableMatrix vars{rows, rows};

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col <= row; ++col) {
        m_decisionVariables.emplace_back();
        vars(row, col) = m_decisionVariables.back();
        vars(col, row) = m_decisionVariables.back();
      }
    }

    return vars;
  }

  /**
   * Tells the solver to minimize the output of the given cost function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param cost The cost function to minimize.
   */
  void Minimize(const Variable& cost) {
    m_f = cost;
    status.costFunctionType = m_f.value().Type();
  }

  /**
   * Tells the solver to minimize the output of the given cost function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param cost The cost function to minimize.
   */
  void Minimize(Variable&& cost) {
    m_f = std::move(cost);
    status.costFunctionType = m_f.value().Type();
  }

  /**
   * Tells the solver to maximize the output of the given objective function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param objective The objective function to maximize.
   */
  void Maximize(const Variable& objective) {
    // Maximizing a cost function is the same as minimizing its negative
    m_f = -objective;
    status.costFunctionType = m_f.value().Type();
  }

  /**
   * Tells the solver to maximize the output of the given objective function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param objective The objective function to maximize.
   */
  void Maximize(Variable&& objective) {
    // Maximizing a cost function is the same as minimizing its negative
    m_f = -std::move(objective);
    status.costFunctionType = m_f.value().Type();
  }

  /**
   * Tells the solver to solve the problem while satisfying the given equality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  void SubjectTo(const EqualityConstraints& constraint) {
    // Get the highest order expression type of the equality constraint passed
    // by the user, before conversion to two inequalities
    for (const auto& c : constraint.constraints) {
      status.equalityConstraintType =
          std::max(status.equalityConstraintType, c.Type());
    }

    m_combinedInequalityConstraints.reserve(
        m_combinedInequalityConstraints.size() +
        2 * constraint.constraints.size());
    for (const auto& c : constraint.constraints) {
      // Used only for bookkeeping
      m_equalityConstraints.emplace_back(c);
      // Equivalent since cₑ(x) = 0 iff cₑ(x) ≥ 0 and -cₑ(x) ≥ 0
      m_combinedInequalityConstraints.emplace_back(c);
      m_combinedInequalityConstraints.emplace_back(-c);
    }
  }

  /**
   * Tells the solver to solve the problem while satisfying the given equality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  void SubjectTo(EqualityConstraints&& constraint) {
    // Get the highest order expression type of the equality constraint passed
    // by the user, before conversion to two inequalities
    for (const auto& c : constraint.constraints) {
      status.equalityConstraintType =
          std::max(status.equalityConstraintType, c.Type());
    }

    m_combinedInequalityConstraints.reserve(
        m_combinedInequalityConstraints.size() +
        2 * constraint.constraints.size());
    for (const auto& c : constraint.constraints) {
      // Used only for bookkeeping
      m_equalityConstraints.emplace_back(c);
      // Equivalent since cₑ(x) = 0 iff cₑ(x) ≥ 0 and -cₑ(x) ≥ 0
      m_combinedInequalityConstraints.emplace_back(c);
      m_combinedInequalityConstraints.emplace_back(-c);
    }
  }

  /**
   * Tells the solver to solve the problem while satisfying the given inequality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  void SubjectTo(const InequalityConstraints& constraint) {
    // Get the highest order inequality constraint expression type
    for (const auto& c : constraint.constraints) {
      status.inequalityConstraintType =
          std::max(status.inequalityConstraintType, c.Type());
    }

    m_combinedInequalityConstraints.reserve(
        m_combinedInequalityConstraints.size() + constraint.constraints.size());
    std::copy(constraint.constraints.begin(), constraint.constraints.end(),
              std::back_inserter(m_combinedInequalityConstraints));
  }

  /**
   * Tells the solver to solve the problem while satisfying the given inequality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  void SubjectTo(InequalityConstraints&& constraint) {
    // Get the highest order inequality constraint expression type
    for (const auto& c : constraint.constraints) {
      status.inequalityConstraintType =
          std::max(status.inequalityConstraintType, c.Type());
    }

    m_combinedInequalityConstraints.reserve(
        m_combinedInequalityConstraints.size() + constraint.constraints.size());
    std::copy(constraint.constraints.begin(), constraint.constraints.end(),
              std::back_inserter(m_combinedInequalityConstraints));
  }

  /**
   * Solve the optimization problem. The solution will be stored in the original
   * variables used to construct the problem.
   *
   * @param config Configuration options for the solver.
   */
  SolverStatus Solve(const SolverConfig& config = SolverConfig{}) {
    // Create the initial value column vector
    Eigen::VectorXd x = VariableMatrix{m_decisionVariables}.Value();

    status.exitCondition = SolverExitCondition::kSuccess;

    // If there's no cost function, make it zero and continue
    if (!m_f.has_value()) {
      m_f = Variable();
    }

    if (config.diagnostics) {
      constexpr std::array kExprTypeToName{"empty", "constant", "linear",
                                           "quadratic", "nonlinear"};

      // Print cost function and constraint expression types
      sleipnir::println(
          "The cost function is {}.",
          kExprTypeToName[static_cast<int>(status.costFunctionType)]);
      sleipnir::println(
          "The equality constraints are {}.",
          kExprTypeToName[static_cast<int>(status.equalityConstraintType)]);
      sleipnir::println(
          "The inequality constraints are {}.",
          kExprTypeToName[static_cast<int>(status.inequalityConstraintType)]);
      sleipnir::println("");

      // Print problem dimensionality
      sleipnir::println("Number of decision variables: {}",
                        m_decisionVariables.size());
      sleipnir::println(
          "Number of equality constraints: "
          "{}\n",
          m_equalityConstraints.size());
      sleipnir::println(
          "Number of inequality constraints (before addition of converted "
          "equalities): "
          "{}\n",
          m_combinedInequalityConstraints.size() -
              m_equalityConstraints.size());
    }

    // If the problem is empty or constant, there's nothing to do
    if (status.costFunctionType <= ExpressionType::kConstant &&
        status.equalityConstraintType <= ExpressionType::kConstant &&
        status.inequalityConstraintType <= ExpressionType::kConstant) {
      return status;
    }

    // Check for an overconstrained problem
    if (m_equalityConstraints.size() > m_decisionVariables.size()) {
      if (config.diagnostics) {
        sleipnir::println("The problem has too few degrees of freedom.");
        sleipnir::println(
            "Violated constraints (cₑ(x) = 0) in order of declaration:");
        VariableMatrix c_eAD{m_equalityConstraints};
        Eigen::VectorXd c_e = c_eAD.Value();
        for (Eigen::Index row = 0; row < c_e.rows(); ++row) {
          if (c_e(row) < 0.0) {
            sleipnir::println("  {}/{}: {} = 0", row + 1, c_e.rows(), c_e(row));
          }
        }
      }

      status.exitCondition = SolverExitCondition::kTooFewDOFs;
      return status;
    }

    // TODO(declan): THIS IS A HACK---the one-phase paper, hence our solver,
    // expects constraints of the form cᵢ(x) ≤ 0, even though the original Julia
    // implementation expects the opposite. It would be faster to convert
    // constraints to this form as we receive them, but I would like to
    // eventually refactor the solver itself, so I'm keeping this easy-to-remove
    // here.
    for (auto& constraint : m_combinedInequalityConstraints) {
      constraint = -constraint;
    }

    // Solve the optimization problem
    Eigen::VectorXd s =
        Eigen::VectorXd::Ones(m_combinedInequalityConstraints.size());
    InteriorPoint(m_decisionVariables, {}, m_combinedInequalityConstraints,
                  m_f.value(), m_callback, config, false, x, s, &status);

    if (config.diagnostics) {
      sleipnir::println("Exit condition: {}", ToMessage(status.exitCondition));
    }

    // Assign the solution to the original Variable instances
    VariableMatrix{m_decisionVariables}.SetValue(x);

    return status;
  }

  /**
   * Sets a callback to be called at each solver iteration.
   *
   * The callback for this overload should return void.
   *
   * @param callback The callback.
   */
  template <typename F>
    requires requires(F callback, const SolverIterationInfo& info) {
      { callback(info) } -> std::same_as<void>;
    }
  void Callback(F&& callback) {
    m_callback = [=, callback = std::forward<F>(callback)](
                     const SolverIterationInfo& info) {
      callback(info);
      return false;
    };
  }

  /**
   * Sets a callback to be called at each solver iteration.
   *
   * The callback for this overload should return bool.
   *
   * @param callback The callback. Returning true from the callback causes the
   *   solver to exit early with the solution it has so far.
   */
  template <typename F>
    requires requires(F callback, const SolverIterationInfo& info) {
      { callback(info) } -> std::same_as<bool>;
    }
  void Callback(F&& callback) {
    m_callback = std::forward<F>(callback);
  }

 private:
  // The list of decision variables, which are the root of the problem's
  // expression tree
  small_vector<Variable> m_decisionVariables;

  // The cost function: f(x)
  std::optional<Variable> m_f;

  // The list of inequality constraints: cᵢ(x) ≥ 0; includes converted equality
  // constraints
  small_vector<Variable> m_combinedInequalityConstraints;

  // The list of equality constraints: cₑ(x) = 0; each underlying Expression is
  // also pointed to by one element of m_inequalityConstraints
  small_vector<Variable> m_equalityConstraints;

  // The user callback
  std::function<bool(const SolverIterationInfo& info)> m_callback =
      [](const SolverIterationInfo&) { return false; };

  // The solver status
  SolverStatus status;
};

}  // namespace sleipnir
