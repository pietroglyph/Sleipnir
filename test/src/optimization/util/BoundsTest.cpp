// Copyright (c) Sleipnir contributors

#include <array>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/matchers/catch_matchers_quantifiers.hpp>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/autodiff/Jacobian.hpp>
#include <optimization/solver/util/Bounds.hpp>

#include "CatchStringConverters.hpp"

constexpr auto inf = std::numeric_limits<double>::infinity();

template <typename O, typename GettableByType, std::size_t N>
constexpr auto extract(const std::array<GettableByType, N>& array) {
  constexpr auto impl = []<std::size_t... I>(
                            const std::array<GettableByType, N>& array,
                            std::index_sequence<I...>) -> std::array<O, N> {
    return {std::get<O>(array[I])...};
  };
  return impl(array, std::make_index_sequence<N>{});
}

TEST_CASE("Bounds - Detection", "[Bounds]") {
  sleipnir::Variable x, y, z, w, v;
  auto decisionVariables = std::to_array<sleipnir::Variable>({x, y, z, w, v});

  sleipnir::Variable a = -z + 1e-12;
  auto constraintVariables = std::to_array<sleipnir::Variable>({
      x + y - 3,
      x * y,
      3,
      x - 3,
      x - 4,
      (3 + 4 * y - y) * 4,
      sleipnir::sin(w),
      a,
      -z,
      -v + 8,
      v - 8,
      v - 7,
      v - 6.5,
  });

  constexpr auto correctBounds = std::to_array<std::pair<double, double>>({
      {-inf, 3},
      {-inf, -1},
      {1e-12, inf},
      {-inf, inf},
      {8, 6.5},
  });  // Assumes c(x) ≤ 0.
  static_assert(correctBounds.size() == decisionVariables.size());
  constexpr auto correctBoundConstraintIndices = std::to_array<Eigen::Index>({
      3,
      4,
      5,
      7,
      8,
      9,
      10,
      11,
      12,
  });
  constexpr auto correctConflictingBounds =
      std::to_array<std::pair<Eigen::Index, Eigen::Index>>({
          {9, 11},
          {9, 12},
      });

  sleipnir::VariableMatrix xAD{decisionVariables};
  sleipnir::VariableMatrix cAD{constraintVariables};
  sleipnir::Jacobian jacobianC{cAD, xAD};
  Eigen::SparseMatrix<double> A = jacobianC.Value();
  const auto [boundConstraintIndices, decisionVarToBounds, conflictingBounds] =
      sleipnir::GetBounds(decisionVariables, constraintVariables, A);

  using Catch::Matchers::UnorderedRangeEquals;
  CHECK_THAT(decisionVarToBounds, UnorderedRangeEquals(correctBounds));
  CHECK_THAT(boundConstraintIndices,
               UnorderedRangeEquals(correctBoundConstraintIndices));
  CHECK_THAT(conflictingBounds,
               UnorderedRangeEquals(correctConflictingBounds));
}

TEST_CASE("Bounds - Projection", "[Bounds]") {
  auto bounds = std::to_array<std::pair<double, double>>({
      {-inf, inf},
      {-inf, 3},
      {2, 2},
      {12, 12.1},
      {-1, -1e-12},
      {2, inf},
  });
  Eigen::Vector<double, bounds.size()> x, xCorrect;

  // This tests that we exactly match section 3.6 in [2]
  SECTION("Initial value already mostly in bounds") {
    constexpr double κ_1 = 1e-2, κ_2 = 1e-2;
    x.setZero();
    xCorrect << 0, 0, 2, 12 + κ_2 * 0.1,
        -1e-12 - std::min(κ_1, κ_2 * (1 - 1e-12)), 2 + κ_1 * 2;
    sleipnir::ProjectOntoBounds(x, bounds, κ_1, κ_2);
    CHECK(x == xCorrect);
  }

  // This tests that we match the spirit of bound projection, without relying on
  // any details of the specific method
  const auto boundsAreSane = [](auto x, auto bounds, bool stickToLower) {
    for (std::size_t i = 0; i < bounds.size(); i++) {
      const auto& [lower, upper] = bounds[i];
      if (stickToLower && std::isfinite(lower)) {
        CHECK(std::abs(lower - x[i]) <= std::abs(upper - x[i]));
      } else if (std::isfinite(upper)) {
        CHECK(std::abs(lower - x[i]) >= std::abs(upper - x[i]));
      }
      if (lower == upper) {
        CHECK(lower == x[i]);
      } else {
        CHECK(lower < x[i]);
        CHECK(x[i] < upper);
      }
    }
  };
  SECTION("Initial value below all bounds") {
    constexpr auto bigNegative = -1000;
    x.setConstant(bigNegative);
    sleipnir::ProjectOntoBounds(x, bounds);
    boundsAreSane(x, bounds, true);
  }
  SECTION("Initial value above all bounds") {
    constexpr auto bigPositive = 1000;
    x.setConstant(bigPositive);
    xCorrect << bigPositive, 3, 2, 12.1, -1e-12, bigPositive;
    sleipnir::ProjectOntoBounds(x, bounds);
    boundsAreSane(x, bounds, false);
  }
}
