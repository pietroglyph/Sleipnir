// Copyright (c) Sleipnir contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CatchStringConverters.hpp"

TEST_CASE("Initialization - Matches OnePhase.jl for Quadradic Objective",
          "[Initialization]") {
  static constexpr sleipnir::SolverConfig initOnly{.maxIterations = 0, .diagnostics = true};
  SECTION("One decision variable") {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable();
    problem.Minimize(x);
    problem.SubjectTo(x * x >= 1.0);
    problem.SubjectTo(x >= -1);

    SECTION("Feasible start") {
      x.SetValue(-3.0);
      auto status = problem.Solve(initOnly);
      CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);
    }
    SECTION("Infeasible start") {
      x.SetValue(3.0);
      auto status = problem.Solve(initOnly);
      CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);
    }
  }

  SECTION("Multiple decision variables") {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable();
    auto y = problem.DecisionVariable();
    problem.Minimize(sleipnir::pow(y + 100, 2) + 0.01 * x * x);
    problem.SubjectTo(y - sleipnir::cos(x) >= 0);
    problem.SubjectTo(x * x <= 5);

    SECTION("Feasible start") {
      x.SetValue(0.0);
      y.SetValue(1.0);
      auto status = problem.Solve(initOnly);
      CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);
    }
    SECTION("Infeasible start") {
      x.SetValue(std::numbers::pi);
      y.SetValue(0.0);
      auto status = problem.Solve(initOnly);
      CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);
    }
  }
}
