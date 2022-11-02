// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <sleipnir/optimization/OptimizationProblem.h>
#include <units/time.h>

/**
 * Creates a cart-pole nonlinear optimization problem with Sleipnir.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
sleipnir::OptimizationProblem CartPoleSleipnir(units::second_t dt, int N);
