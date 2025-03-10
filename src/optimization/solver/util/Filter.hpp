// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/autodiff/Variable.hpp"
#include "optimization/solver/util/LogBarrierFunctions.hpp"
#include "optimization/solver/util/KKTMeritFunction.hpp"
#include "sleipnir/util/small_vector.hpp"

namespace sleipnir {

/**
 * Filter entry consisting of cost and constraint violation.
 */
struct FilterEntry {
  /// The objective added to a shifted log barrier term and a complimentarity
  /// measure, denoted ϕ_μₖ(xₖ, sₖ, yₖ) in [4].
  double modifiedLogBarrierObjective = 0.0;

  /// A truncated version of the full KKT error that measures just (scaled) dual
  /// feasibility and complimentarity, denoted 𝕂_μₖ(xₖ, sₖ, yₖ) in [4].
  double truncatedKKTError = 0.0;

  /// The primal step size used to get this step, denoted αₚ in [4].
  double primalStepSize = 0.0;

  constexpr FilterEntry() = default;

  /**
   * Constructs a FilterEntry.
   *
   * @param modifiedLogBarrierObjective The value of the augmented, shifted log
   *   barrier function computed at (xₖ, sₖ, yₖ); i.e., ϕ_μₖ(xₖ, sₖ, yₖ) in [4].
   * @param truncatedKKTError The max of the L_∞ norms of the dual feasibility
   * and complimentarity, multiplied by a dual Lagrange multiplier scaling
   *   factor; i.e., 𝕂_μₖ(xₖ, sₖ, yₖ).
   * @param primalStepSize The primal step size αₚ used to get the iterate
   *   (xₖ, sₖ, yₖ); i.e., (xₖ, sₖ, yₖ) = (xₖ₋₁, sₖ₋₁, yₖ₋₁) + αₚdₖ₋₁.
   */
  FilterEntry(const double modifiedLogBarrierObjective,
              const double truncatedKKTError, const double primalStepSize)
      : modifiedLogBarrierObjective{modifiedLogBarrierObjective},
        truncatedKKTError{truncatedKKTError},
        primalStepSize{primalStepSize} {}

  /**
   * Constructs a FilterEntry.
   *
   * @param f The cost function.
   * @param μ The barrier parameter.
   * @param s The inequality constraint slack variables.
   * @param c_e The equality constraint values (nonzero means violation).
   * @param c_i The inequality constraint values (negative means violation).
   */
  /* FilterEntry(Variable& f, const double μ, const double β_1, const
     Eigen::VectorXd& s, const Eigen::VectorXd& y, const Eigen::VectorXd& c_i,
     const Eigen::VectorXd& w) : modifiedLogBarrierObjective{ϕ(f.Value(), c_i,
     s, y, w, μ, β_1)}, truncatedKKTError{KKTMeritFunction()} {} */
};

/**
 * Interior-point step filter.
 */
class Filter {
 public:
  /**
   * Construct an empty filter.
   *
   * @param f The cost function.
   * @param μ The barrier parameter.
   */
  explicit Filter(Variable& f, const double μ, const double β_7) : m_β_7{β_7} {
    m_f = &f;
    m_μ = μ;

    // There is no initial filter entry; [4] and Julia implementation seem to
    // accept the first step unconditionally. This is not like [2].
  }

  /**
   * Reset the filter when the amount of constraint violation changes. Since we
   * enforce, for all k, that a(xₖ) + sₖ = wμₖ, then two iterates have the same
   * constraint violation if and only if they have the same barrier parameter.
   *
   * @param μ The new barrier parameter.
   */
  void Reset(double μ) {
    m_μ = μ;
    m_filter.clear();

    // There is no initial filter entry; [4] and Julia implementation seems to
    // accept the first step unconditionally. This is not like [2].
  }

  /**
   * Creates a new filter entry.
   *
   * @param s The inequality constraint slack variables.
   * @param c_e The equality constraint values (nonzero means violation).
   * @param c_i The inequality constraint values (negative means violation).
   */
  /* FilterEntry MakeEntry(Eigen::VectorXd& s, const Eigen::VectorXd& c_e,
                        const Eigen::VectorXd& c_i) {
    return FilterEntry{*m_f, m_μ, s, c_e, c_i};
  } */

 private:
  /**
   * For some FilterEntry lhs, returns a function f such that f(rhs) = true only
   * if lhs dominates rhs. The predicate f can be viewed as a partial order
   * where rhs ≤ lhs only if f(rhs) = true.
   *
   * @param lhsEntry A filter entry that will be on the lhs of the partial order.
   */
  auto SufficientDecreaseTotalOrderPredicate(const FilterEntry& lhsEntry) {
    // TODO(declan): this is a filter trivially, since we're checking against a total order?? This is quite unlike the usual terminology in constrained optimization.
    return [&](const FilterEntry& rhsEntry) {
      return lhsEntry.truncatedKKTError <=
                 (1 - m_β_7 * lhsEntry.primalStepSize) *
                     rhsEntry.truncatedKKTError &&
             lhsEntry.modifiedLogBarrierObjective <=
                 rhsEntry.modifiedLogBarrierObjective +
                     std::sqrt(rhsEntry.truncatedKKTError);
    };
  }

 public:
  /**
   * Add a new entry to the filter. Does not check whether the entry  is
   * acceptable.
   *
   * @param entry The entry to add to the filter.
   */
  void Add(const FilterEntry& entry) {
    // Remove dominated entries.
    // XXX(declan): This has a cost and we don't currently rely on only holding
    // only non-dominated entries. It does prevent the small_vector from getting
    // huge, but since we reset on every barrier parameter change we probably
    // wouldn't have any huge changes anyway.
    // TODO(declan): Check whether or not we should use a different predicate
    // here?
    //erase_if(m_filter, DominatesPredicate(entry));

    m_filter.push_back(entry);
  }

  /**
   * Add a new entry to the filter. Does not check whether the entry  is
   * acceptable.
   *
   * @param entry The entry to add to the filter.
   */
  void Add(FilterEntry&& entry) {
    // Remove dominated entries.
    //erase_if(m_filter, DominatesPredicate(entry));

    m_filter.push_back(entry);
  }

  /**
   * Add a new entry to the filter only if it is acceptable to the filter;
   * returns true if the given iterate is accepted.
   *
   * @param entry The entry to attempt adding to the filter.
   */
  bool TryAdd(const FilterEntry& entry) {
    if (IsAcceptable(entry)) {
      Add(entry);
      return true;
    } else {
      return false;
    }
  }

  /**
   * Returns true if the given iterate is accepted by the filter.
   *
   * @param entry The entry to attempt adding to the filter.
   */
  bool TryAdd(FilterEntry&& entry) {
    if (IsAcceptable(entry)) {
      Add(std::move(entry));
      return true;
    } else {
      return false;
    }
  }

  /**
   * Returns true if the given entry is acceptable to the filter.
   *
   * @param entry The entry to check.
   */
  bool IsAcceptable(const FilterEntry& entry) {
    if (!std::isfinite(entry.modifiedLogBarrierObjective) ||
        !std::isfinite(entry.truncatedKKTError)) {
      return false;
    }

    // If current filter entry is better than all prior ones in some respect,
    // accept it.
    return std::all_of(m_filter.begin(), m_filter.end(),
                       SufficientDecreaseTotalOrderPredicate(entry));
  }

 private:
  const double m_β_7;
  Variable* m_f = nullptr;
  double m_μ = 0.0;
  small_vector<FilterEntry> m_filter;
};

}  // namespace sleipnir
