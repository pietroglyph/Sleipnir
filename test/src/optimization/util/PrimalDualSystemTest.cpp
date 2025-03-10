// Copyright (c) Sleipnir contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <sleipnir/autodiff/Hessian.hpp>

#include "CatchStringConverters.hpp"
#include "optimization/solver/util/PrimalDualSystem.hpp"

TEST_CASE("PrimalDualSystem - Form LHS", "[PrimalDualSystem]") {
  static constexpr double β_1 = 1e-4;
  sleipnir::VariableMatrix x{1200}, y{4000};
  x.SetValue(Eigen::VectorXd::Random(x.Rows()));
  y.SetValue(Eigen::VectorXd::Random(y.Rows()));
  sleipnir::Variable μAD{12.0};
  sleipnir::VariableMatrix c_i{y.Rows()};
  for (int i = 0; i < c_i.Rows(); i++) {
    c_i.Row(i) = x.Row(i % x.Rows())(0) - std::rand();
  }

  sleipnir::Variable f = x.Row(3).CwiseTransform(sleipnir::sin)(
      0);  //(x.Block(98, 0, 12, 0).CwiseTransform(sleipnir::exp).T() *
           //x.Block(3, 0, 12, 0))(0) / (x.Block(9, 0, 95, 0).T() *
           //sleipnir::VariableMatrix::Ones(95, 1))(0);
  auto L =
      f + ((y - μAD * β_1 * sleipnir::VariableMatrix::Ones(y.Rows(), 1)).T() *
           c_i)(0);
  sleipnir::Hessian hessianL{L, x};
  sleipnir::Jacobian jacobianCi{c_i, x};

  Eigen::SparseMatrix<double> M = hessianL.Value();  // Don't care
  Eigen::SparseMatrix<double> A_i = jacobianCi.Value();
  Eigen::VectorXd yVec = y.Value(), s = Eigen::VectorXd::Random(y.Rows());
  sleipnir::ComputePrimalDualLHS(M, hessianL, jacobianCi.Value(), y.Value(), s);
}
