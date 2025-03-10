// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "optimization/solver/util/LagrangeMultiplierRescaler.hpp"
#include "optimization/solver/util/ModifiedLagrangian.hpp"

namespace sleipnir {
inline double KKTMeritFunction(const Eigen::SparseVector<double>& g,
                               const Eigen::SparseVector<double>& A_i,
                               const Eigen::VectorXd& s,
                               const Eigen::VectorXd& y, const double μ,
                               const double β_1) {
  constexpr auto σ = LagrangeMultiplierRescaler;
  // 𝕂_μ(x, s, y) := σ(y) max{‖∇L_μ(x, y)‖_∞, ‖Sy − μe‖_∞}
  // XXX(declan): filter_ls.jl:55 does σ(y)(‖∇L_μ(x, y)‖_∞ + ‖Sy − μe‖_∞)?? WTF.
  return σ(y) *
         std::max(ManualGradientModifiedLagrangian(g, A_i, y, μ, β_1)
                      .lpNorm<Eigen::Infinity>(),
                  (s.cwiseProduct(y).array() - μ).lpNorm<Eigen::Infinity>());
}
}  // namespace sleipnir
