// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "optimization/solver/util/LagrangeMultiplierRescaler.hpp"
#include "optimization/solver/util/ModifiedLagrangian.hpp"

namespace sleipnir {
inline bool IsAggressiveStepAppropriate(const Eigen::VectorXd& y,
                                        const Eigen::VectorXd& s,
                                        const Eigen::VectorXd& gradL,
                                        const double duallessGradLNorm,
                                        const double μ, const double β_3) {
  // σ(y) = 100/max{100, ‖y‖_∞}
  auto σ = LagrangeMultiplierRescaler;
  // [[sᵢyᵢ/μ : i ∈ {1, … }]]ᵀ
  const Eigen::ArrayXd scaledComplimentarity = s.cwiseProduct(y) / μ;
  // Equations (20a-c) in [4]:
  // ‖∇L_μ(x, y)‖_∞ ≤ μ ∧
  // ‖∇L_μ(x, y)‖₁  ≤ ‖∇f(x) − β₁μeᵀ∇cᵢ(x)‖₁ + sᵀy ∧
  // ∀i ∈ {1, …, m} (sᵢyᵢ/μ ∈ [β₃, 1/β₃])
  return gradL.lpNorm<1>() <= duallessGradLNorm + s.dot(y) &&
         σ(y) * gradL.lpNorm<Eigen::Infinity>() <= μ &&
         β_3 <= scaledComplimentarity && scaledComplimentarity <= 1.0 / β_3;
}

inline bool IsAggressiveStepAppropriate(const Eigen::VectorXd& y,
                                        const Eigen::VectorXd& s,
                                        const Eigen::SparseVector<double>& g,
                                        const Eigen::SparseMatrix<double>& A_i,
                                        const double μ, const double β_1,
                                        const double β_3) {
  // ∇L_μ(x, y)
  const Eigen::VectorXd gradL =
      ManualGradientModifiedLagrangian(g, A_i, y, μ, β_1);
  // ∇L_μ(x, 0) = ∇f(x) − β₁μeᵀ∇cᵢ(x)
  const double duallessGradLNorm =
      ManualGradientModifiedLagrangian(g, A_i, Eigen::VectorXd::Zero(y.size()),
                                       μ, β_1)
          .lpNorm<1>();
  return IsAggressiveStepAppropriate(y, s, gradL, duallessGradLNorm, μ, β_3);
}

inline bool IsAggressiveStepAppropriate(
    const Eigen::VectorXd& y, const Eigen::VectorXd& s,
    const Eigen::SparseVector<double>& g,
    const Eigen::SparseVector<double>& gradientComplimentarity,
    const Eigen::SparseMatrix<double>& gradientConstraintSum, const double μ,
    const double β_1, const double β_3) {
  // ∇L_μ(x, y)
  const Eigen::VectorXd gradL = ManualGradientModifiedLagrangian(
      g, gradientComplimentarity, gradientConstraintSum, μ, β_1);
  // ∇L_μ(x, 0) = ∇f(x) − β₁μeᵀ∇cᵢ(x)
  const double duallessGradLNorm =
      ManualGradientModifiedLagrangian(
          g, Eigen::SparseVector<double>(gradientComplimentarity.size()),
          gradientConstraintSum, μ, β_1)
          .lpNorm<1>();
  return IsAggressiveStepAppropriate(y, s, gradL, duallessGradLNorm, μ, β_3);
}
}  // namespace sleipnir
