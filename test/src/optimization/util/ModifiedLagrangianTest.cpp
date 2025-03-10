// Copyright (c) Sleipnir contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/Gradient.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <optimization/solver/util/ModifiedLagrangian.hpp>

#include "CatchStringConverters.hpp"

TEST_CASE("ModifiedLagrangian - Manual Matches Autodiff",
          "[ModifiedLagrangian]") {
  // Rosenbrock function constrained with a cubic and a line
  sleipnir::Variable x, z;
  auto f = sleipnir::pow(1 - x, 2) + 100 * sleipnir::pow(z - x * x, 2);
  sleipnir::VariableMatrix c_i{
      {{sleipnir::pow(x - 1, 3) - z + 1}, {x + z - 2}}};
  auto y = sleipnir::VariableMatrix::Ones(c_i.Rows(), c_i.Cols());

  // Values don't matter, they just need to be the same in both calls.
  static constexpr double μ = 15.0, β_1 = 1e-4;

  auto L = sleipnir::AutodiffModifiedLagrangian(f, y, c_i, μ, β_1);
  auto autodiffGradL = sleipnir::Gradient{L, x}.Value();

  auto g = sleipnir::Gradient{f, x}.Value();
  auto A_i = sleipnir::Jacobian{c_i, x}.Value();
  auto manualGradL = sleipnir::ManualGradientModifiedLagrangian(g, A_i, y.Value(), μ, β_1);
}
