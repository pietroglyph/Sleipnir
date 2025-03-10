// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <Eigen/Core>

// See docs/algorithms.md#Works_cited for citation definitions

namespace sleipnir {

/*
 * Function to rescale expressions containing a Lagrange multiplier. We usually
 * apply it to the dual multiplier y; in this case, this gives the function σ(y)
 * from section 2.3 of [4]. The expression is related to s_d and s_c in [2].
 *
 * @param langrangeMultiplier A Lagrange multiplier vector.
 */
inline double LagrangeMultiplierRescaler(
    const Eigen::Ref<const Eigen::VectorXd> lagrangeMultiplier) {
  return 100.0 / std::max(100.0, lagrangeMultiplier.lpNorm<Eigen::Infinity>());
}
}  // namespace sleipnir
