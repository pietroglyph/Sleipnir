// Copyright (c) Sleipnir contributors

#pragma once

#include "sleipnir/autodiff/ExpressionType.hpp"
#include "sleipnir/optimization/SolverExitCondition.hpp"
#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
 * Return value of OptimizationProblem::Solve() containing the cost function and
 * constraint types and solver's exit condition.
 */
struct SLEIPNIR_DLLEXPORT SolverStatus {
  /// The cost function type detected by the solver.
  ExpressionType costFunctionType = ExpressionType::kNone;

  /// The expression type of the equality constraints as given by the user;
  /// these are converted to equivalent inequality constraints before being
  /// passed to the solver―i.e., the solver operates on only inequality
  /// constraints of type max(equalityConstraintType, inequalityConstraintType).
  ExpressionType equalityConstraintType = ExpressionType::kNone;

  /// The expression type of the inequality constraints as given by the user;
  /// these are converted to equivalent inequality constraints before being
  /// passed to the solver―i.e., the solver operates on only inequality
  /// constraints of type max(equalityConstraintType, inequalityConstraintType).
  ExpressionType inequalityConstraintType = ExpressionType::kNone;

  /// The solver's exit condition.
  SolverExitCondition exitCondition = SolverExitCondition::kSuccess;

  /// The solution's cost.
  double cost = 0.0;
};

}  // namespace sleipnir
