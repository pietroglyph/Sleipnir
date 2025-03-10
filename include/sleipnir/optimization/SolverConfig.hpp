// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>
#include <limits>

#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
 * Solver configuration.
 */
struct SLEIPNIR_DLLEXPORT SolverConfig {
  /// The solver will scale its initial estimate of the barrier parameter by
  /// this value, which should be in (0, ∞).
  double initialBarrierScale = 1.0;

  /// The solver will stop once the error is below this tolerance.
  double tolerance = 1e-8;

  /// The maximum number of solver iterations before returning a solution.
  int maxIterations = 5000;

  /// The solver will stop once the error is below this tolerance for
  /// `acceptableIterations` iterations. This is useful in cases where the
  /// solver might not be able to achieve the desired level of accuracy due to
  /// floating-point round-off.
  double acceptableTolerance = 1e-6;

  /// The solver will stop once the error is below `acceptableTolerance` for
  /// this many iterations.
  int maxAcceptableIterations = 15;

  /// The maximum elapsed wall clock time before returning a solution.
  std::chrono::duration<double> timeout{
      std::numeric_limits<double>::infinity()};

  /// Enables diagnostic prints.
  bool diagnostics = false;

  /// Enables writing sparsity patterns of H and Aᵢ to files named H.spy and
  /// A_i.spy, respectively, during solve.
  ///
  /// Use tools/spy.py to plot them.
  bool spy = false;
};

}  // namespace sleipnir
