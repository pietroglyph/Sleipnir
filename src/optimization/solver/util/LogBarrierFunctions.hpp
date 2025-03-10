// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/autodiff/VariableMatrix.hpp"
#include "sleipnir/util/Concepts.hpp"

namespace sleipnir {

namespace detail {
template <typename Derived>
inline auto cwiseLog(const Eigen::MatrixBase<Derived>& v) {
  return v.array().log().matrix();
}

inline sleipnir::VariableMatrix cwiseLog(const sleipnir::VariableMatrix& v) {
  return v.CwiseTransform(
      [](const sleipnir::Variable& x) { return sleipnir::log(x); });
}
}  // namespace detail

template <MatrixLike V, ScalarLike S>
inline S ψ(const S& f, const V& c_i, const Eigen::VectorXd& w, const S& μ,
           const double β_1) {
  return f - μ * (V::Ones(1, c_i.size()) *
                  (β_1 * c_i + detail::cwiseLog(μ * V{w} - c_i)))(0, 0);
}

inline double ϕ(const double f, const Eigen::VectorXd& c_i,
                         const Eigen::VectorXd& s, const Eigen::VectorXd& y,
                         const Eigen::VectorXd& w, const double μ,
                         const double β_1) {
  const double compMeasure =
      std::pow(
          (s.cwiseProduct(y).array() - μ).matrix().lpNorm<Eigen::Infinity>(),
          3) /
      (μ * μ);
  return ψ(f, c_i, w, μ, β_1) + compMeasure;
}

// VariableMatrix ModifiedLogBarrier(const VariableMatrix& f,
//                                   const VariableMatrix& c_i,
//                                   const Eigen::VectorXd& w, const Variable&
//                                   μ, const double β_1) {
//   return f - μ * (VariableMatrix::Ones(1, c_i.size()) *
//                   (β_1 * c_i + cwiseLog(μ * VariableMatrix{w} - c_i))(0, 0) *
//                   VariableMatrix::Ones(1, f.size()));
// }
//
// Eigen::VectorXd ModifiedLogBarrier(const Eigen::VectorXd& f,
//                                    const Eigen::VectorXd& c_i,
//                                    const Eigen::VectorXd& w, const double μ,
//                                    const double β_1) {
//   return f - μ *
//                  (Eigen::VectorXd::Ones(1, c_i.size()) *
//                   (β_1 * c_i + (μ * Eigen::VectorXd{w} - c_i)))(0, 0) *
//                  Eigen::VectorXd::Ones(1, f.size());
// }
//
// inline void baz() {
//   Eigen::VectorXd f, c_i, w;
//   double μ, β_1;
//   ψ(f, c_i, w, μ, β_1, Eigen::log);
// }
}  // namespace sleipnir
