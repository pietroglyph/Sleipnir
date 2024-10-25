// Copyright (c) Sleipnir contributors

#include <array>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <sleipnir/util/Print.hpp>

#include "optimization/RegularizedLDLT.hpp"

#include "CatchStringConverters.hpp"

const Eigen::Matrix3d rotation =
    Eigen::AngleAxisd{0.25, Eigen::Vector3d{1, 0.5, 1}.normalized()}
        .toRotationMatrix();

TEST_CASE("RegularizedLDLT - Triviality on good positive-definite matrices",
          "[RegularizedLDLT]") {
  const auto mats = std::to_array<Eigen::MatrixXd>({
      Eigen::MatrixXd::Identity(200, 200),
      Eigen::DiagonalMatrix<double, 4>{4, 9, 1, 5},
      rotation * Eigen::DiagonalMatrix<double, 3>{3, 2, 1} *
          rotation.transpose(),
  });

  sleipnir::RegularizedLDLT regularizer{};
  for (const Eigen::MatrixXd& M : mats) {
    regularizer.Compute(M.sparseView(), 0.0);
    REQUIRE(M.isApprox(M.transpose()));  // Test is ill-formed otherwise
    CHECK(regularizer.Info() == Eigen::Success);
    CHECK(regularizer.Delta() == 0);
  }
}

TEST_CASE("RegularizedLDLT - Step correctness", "[RegularizedLDLT]") {
  sleipnir::RegularizedLDLT regularizer{};
  SECTION("Correct steps on M = I") {
    const Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    sleipnir::RegularizedLDLT regularizer{};
    regularizer.Compute(M.sparseView(), 0.0);
    for (int i = 0; i < 10; i++) {
      Eigen::Vector4d rhs = Eigen::Vector4d::Random();
      REQUIRE(regularizer.Solve(rhs) == rhs);
    }
  }
  SECTION("Correct steps on semidefinite M") {
    Eigen::Matrix3d M1 = Eigen::DiagonalMatrix<double, 3>{1, 1, 0};
    Eigen::Vector3d v1{0, 0, 1};
    regularizer.Compute(M1.sparseView(), 0.0);
    Eigen::Vector3d step1 = regularizer.Solve(v1);
    REQUIRE_THAT(step1.z(),
                 Catch::Matchers::WithinRel(
                     v1.z() / (M1.diagonal().z() + regularizer.Delta())));

    Eigen::Matrix3d M2 = Eigen::DiagonalMatrix<double, 3>{1, 0, 1};
    Eigen::Vector3d v2{0, 1, 0};
    regularizer.Compute(M2.sparseView(), 0.0);
    Eigen::Vector3d step2 = regularizer.Solve(v2);
    REQUIRE_THAT(step2.y(),
                 Catch::Matchers::WithinRel(
                     v2.y() / (M2.diagonal().y() + regularizer.Delta())));
  }
  SECTION("Correct steps on changing sparsity patterns") {
    Eigen::Vector4d v{0, 0, 1, 1};

    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    regularizer.Compute(M.sparseView(), 0.0);
    REQUIRE(regularizer.Solve(v) == v);

    Eigen::Matrix4d semidefM;
    // clang-format off
    semidefM << 1, 1, 0, 0,
		1, 1, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0;
    // clang-format on
    regularizer.Compute(semidefM.sparseView(), 0.0);
    regularizer.Solve(v);

    regularizer.Compute(M.sparseView(), 0.0);
    REQUIRE(regularizer.Solve(v) == v);
  }
}

TEST_CASE("RegularizedLDLT - Success on semi- and in-definite matrices",
          "[RegularizedLDLT]") {
  const auto mats = std::to_array<Eigen::MatrixXd>({
      Eigen::DiagonalMatrix<double, 4>{-1, -12, -1000, -1e-3},
      (Eigen::Matrix3d{} << -1, 2, 3, 2, 4, 5, 3, 5, 6).finished(),
      rotation * Eigen::DiagonalMatrix<double, 3>{1e-3, -2000, 1} *
          rotation.transpose(),
      Eigen::DiagonalMatrix<double, 5>{9, 8, 1, 1, 0},
      -1e12 * Eigen::MatrixXd::Identity(200, 200),
  });

  sleipnir::RegularizedLDLT regularizer{};
  for (const Eigen::MatrixXd& M : mats) {
    regularizer.Compute(M.sparseView(), 0.0);
    REQUIRE(M.isApprox(M.transpose()));  // Test is ill-formed otherwise
    CHECK(regularizer.Info() == Eigen::Success);
    CHECK(regularizer.Delta() > 0);
    // This check is sufficient to show that M + δI will be PSD, since we've
    // checked above that M is symmetric
    CHECK(
        regularizer.Delta() >=
        std::max(0.0,
                 -M.selfadjointView<Eigen::Lower>().eigenvalues().minCoeff()));
  }
}
