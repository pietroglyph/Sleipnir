# Internal algorithms

This document contains internal algorithm documentation for Sleipnir. We assume that all costs (objectives) take the form f : **R**ⁿ → **R** and that all constraints take the form c : **R**ᵐ → **R**ᵐ, and that all cost and constraint functions are continuously differentiable.

## Reverse accumulation automatic differentiation

In reverse accumulation AD, the dependent variable to be differentiated is fixed and the derivative is computed with respect to each subexpression recursively. In a pen-and-paper calculation, the derivative of the outer functions is repeatedly substituted in the chain rule:

(∂y/∂x) = (∂y/∂w₁) ⋅ (∂w₁/∂x) = ((∂y/∂w₂) ⋅ (∂w₂/∂w₁)) ⋅ (∂w₁/∂x) = ...

In reverse accumulation, the quantity of interest is the adjoint, denoted with a bar (w̄); it is a derivative of a chosen dependent variable with respect to a subexpression w: ∂y/∂w.

Given the expression f(x₁,x₂) = sin(x₁) + x₁x₂, the computational graph is:
@mermaid{reverse-autodiff}

The operations to compute the derivative:

w̄₅ = 1 (seed)<br>
w̄₄ = w̄₅(∂w₅/∂w₄) = w̄₅<br>
w̄₃ = w̄₅(∂w₅/∂w₃) = w̄₅<br>
w̄₂ = w̄₃(∂w₃/∂w₂) = w̄₃w₁<br>
w̄₁ = w̄₄(∂w₄/∂w₁) + w̄₃(∂w₃/∂w₁) = w̄₄cos(w₁) + w̄₃w₂

https://en.wikipedia.org/wiki/Automatic_differentiation#Beyond_forward_and_reverse_accumulation

## Unconstrained optimization

We want to solve the following optimization problem.

```
   min f(x)
    x
```

where f(x) is the cost function.

### Lagrangian

The Lagrangian of the problem is

```
  L(x) = f(x)
```

### Gradients of the Lagrangian

The gradients are

```
  ∇ₓL(x) = ∇f
```

The first-order necessary conditions for optimality are

```
  ∇f = 0
```

### Newton's method

Next, we'll apply Newton's method to the optimality conditions. Let H be ∂²L/∂x² and pˣ be the step for x.

```
  ∇ₓL(x + pˣ) ≈ ∇ₓL(x) + ∂²L/∂x²pˣ
  ∇ₓL(x) + Hpˣ = 0
  Hpˣ = −∇ₓL(x, y)
  Hpˣ = −(∇f)
```

### Final results

In summary, the following system gives the iterate pˣₖ.

```
  Hpˣ = −∇f(x)
```

The iterate is applied like so

```
  xₖ₊₁ = xₖ + pˣₖ
```

## Sequential quadratic programming

We want to solve the following optimization problem.

```
   min f(x)
    x
  s.t. cₑ(x) = 0
```

where f(x) is the cost function and cₑ(x) is the equality constraints.

### Lagrangian

The Lagrangian of the problem is

```
  L(x, y) = f(x) − yᵀcₑ(x)
```

### Gradients of the Lagrangian

The gradients are

```
  ∇ₓL(x, y) = ∇f − Aₑᵀy
  ∇_yL(x, y) = −cₑ
```

The first-order necessary conditions for optimality are

```
  ∇f − Aₑᵀy = 0
  −cₑ = 0
```

where Aₑ = ∂cₑ/∂x. We'll rearrange them for the primal-dual system.

```
  ∇f − Aₑᵀy = 0
  cₑ = 0
```

### Newton's method

Next, we'll apply Newton's method to the optimality conditions. Let H be ∂²L/∂x², pˣ be the step for x, and pʸ be the step for y.

```
  ∇ₓL(x + pˣ, y + pʸ) ≈ ∇ₓL(x, y) + ∂²L/∂x²pˣ + ∂²L/∂x∂ypʸ
  ∇ₓL(x, y) + Hpˣ − Aₑᵀpʸ = 0
  Hpˣ − Aₑᵀpʸ = −∇ₓL(x, y)
  Hpˣ − Aₑᵀpʸ = −(∇f − Aₑᵀy)
```
```
  ∇_yL(x + pˣ, y + pʸ) ≈ ∇_yL(x, y) + ∂²L/∂y∂xpˣ + ∂²L/∂y²pʸ
  ∇_yL(x, y) + Aₑpˣ = 0
  Aₑpˣ = −∇_yL(x, y)
  Aₑpˣ = −cₑ
```

### Matrix equation

Group them into a matrix equation.

```
  [H   −Aₑᵀ][pˣ] = −[∇f(x) − Aₑᵀy]
  [Aₑ   0  ][pʸ]    [     cₑ     ]
```

Invert pʸ.

```
  [H   Aₑᵀ][ pˣ] = −[∇f(x) − Aₑᵀy]
  [Aₑ   0 ][−pʸ]    [     cₑ     ]
```

### Final results

In summary, the reduced 2x2 block system gives the iterates pˣₖ and pₖʸ.

```
  [H   Aₑᵀ][ pˣ] = −[∇f(x) − Aₑᵀy]
  [Aₑ   0 ][−pʸ]    [     cₑ     ]
```

The iterates are applied like so

```
  xₖ₊₁ = xₖ + pˣₖ
  yₖ₊₁ = yₖ + pₖʸ
```

Section 6 of [^3] describes how to check for local infeasibility.

## Log-domain interior-point method

We want to solve the following optimization problem <a name="original-ipm-problem">(1)</a>

```
   min f(x),  (1)
    x
  s.t. cₑ(x) = 0
       ĉᵢ(x) ≥ 0
```

where f(x) is the cost function, cₑ(x) is the vector of equality constraints, and ĉᵢ(x) is vector of inequality constraints.

We'll reformulate the equality constraints as two inequality constraints: that is, we define a new inequality constraint vector cᵢ(x) = concat(cₑ(x), −cₑ(x), ĉᵢ(x)) (also see the end of section 2 of [^5] for more information on this unusual choice). This gives a new but equivalent problem

```
   min f(x),
    x
  s.t. cᵢ(x) ≥ 0.
```

We would like to control the rate at which we reduce the primal infeasibility since, for reasons outlined in [^7], we would like the rate of decrease to be proportional to the rate at which we decrease complementarity. We can achieve this by making primal infeasibility and complementarity proportional to a parameter μ ∈ (0, ∞), since for a sequence of these parameters (μₖ)ₖ ⊆ (0, ∞), the rate of decrease in complementarity and primal infeasibility will be μₖ₊₁/μₖ for all k ∈ **Z**₊ if and only if the respective proportionality constants are fixed across iterations. We choose such a constant w ∈ **R**ᵐ to be fixed across iterations and define a new modified problem <a name="homotopy-ipm-problem">(2)</a>


```
   min f(x),  (2)
    x
  s.t. cᵢ(x) ≥ -μw;
```

note that in the above problem, μ only controls the rate of decrease of primal infeasibility and *not* complementarity, which is a goal we will return to later.

Take care to note that problem [(2)](#homotopy-ipm-problem) is equivalent to the original problem [(1)](#original-ipm-problem) if and only if μ = 0. We can take advantage of this structure by solving a sequence of these problems with a sequence of (μₖ)ₖ that converges to 0. This is known as a homotopy method, and indeed the final algorithm we derive in the remainder of the section can also be derived as a homotopy method applied to the slightly modified KKT conditions of the following version of problem [(2)](#homotopy-ipm-problem) (see section 19.1 in [^1] for some information on this duality):

```
   min f(x).
   x,s
   s.t. s = cᵢ(x) + μw
        s ≥ 0
```

Instead of deriving our algorithm as a homotopy method, will will derive it as a barrier method since this allows us to incorparate μ as a control term for complementarity in a more theoretically justified manner. We can eliminate the inequality constraints by adding a "log-barrier term" to the objective which penalizes constraint violation---this gives a new modified problem <a name="primal-log-barrier-ipm-problem">(3)</a>

```
   min f(x) − μ ∑ ln[(cᵢ)ⱼ + μwⱼ].  (3)
    x           j
```

In general, the above problem is neither equivalent to the first modified problem [(2)](#homotopy-ipm-problem) nor to the original problem [(1)](#original-ipm-problem) for any μ. Note that we have scaled the log-barrier by μ since it makes the log-barrier term vanish as μ vanishes, which is why μ is typically called the "barrier parameter" (there is another important reason for scaling the log-barrier by μ which will soon become clear under a different problem formulation.)

At this point, we could solve a sequence of these primal log-barrier problems [(3)](#primal-log-barrier-ipm-problem) with decreasing barrier parameters with any unconstrained optimization algorithm. However, for reasons outlined in section 19.6 of [^1], the primal log-barrier objective is highly nonlinear as μ approaches 0 which results in slow convergence when solving a series of primal log-barrier problems [(3)](#primal-log-barrier-ipm-problem).

To remove this nonlinearity, we define slack variables s = cᵢ(x) + μw which we add to and substitute into problem [(3)](#primal-log-barrier-ipm-problem): this gives an equivalent problem <a name="primal-dual-log-barrier-ipm-problem">(4)</a>

```
   min f(x) − μ ∑ ln(sⱼ).  (4)
   x,s          j
   s.t. cᵢ(x) − s = -μw
```

Note that the state is now (x, s) ∈ **R**ⁿ × **R**ᵐ, and that this is still not equivalent to the original problem [(1)](#original-ipm-problem), although again we could solve a sequence of these problems as μ → 0. We will also show in the next section that scaling the sum by μ also causes μ to control the rate of decrease in complementarity, which achieves our goal of decreasing complementarity and primal infeasibility at the same rate.

Finally, following [^5], we add another term to the sum to bound each summand below so that the primal iterates do not spuriously diverge, which gives our final problem <a name="shifted-primal-dual-log-barrier-ipm-problem">(5)</a>

```
  min f(x) − μ ∑ [β₁(cᵢ)ⱼ(x) + ln(sⱼ)],  (5)
  x,s          j
  s.t. cᵢ(x) − s + μw = 0
```

where β₁ ∈ (0, ∞).

### Lagrangian

The Lagrangian of the final barrier problem [(5)](#shifted-primal-dual-log-barrier-ipm-problem) is

```
  L(x, s, z) = f(x) − μ ∑ [β₁(cᵢ)ⱼ(x) + ln(sⱼ)] − zᵀ(cᵢ(x) − s + μw).
                        j
```

### Gradients of the Lagrangian

The gradient of the Lagrangian of the barrier problem [(5)](#shifted-primal-dual-log-barrier-ipm-problem) with respect to the state (x, s) ∈ **R**ⁿ × **R**ᵐ is

```
  ∇ₓL(x, s, z) = ∇f − Aᵢᵀ(z − μβ₁e)
  ∇ₛL(x, s, z) = z − μS⁻¹e,
```

where ∇f = ∇f(x), Aᵢ = ∂cᵢ/∂x(x), S = diag(s), and e is a column vector of ones.

### First-order necessary conditions

We will now write the first-order necessary conditions: if (x, s) ∈ **R**ⁿ × **R**ᵐ is a local solution to the barrier problem [(5)](#shifted-primal-dual-log-barrier-ipm-problem) at which an appropriate constraint qualification holds, then there exists a Lagrange multiplier z ∈ **R**ᵐ such that

```
  ∇ₓL(x, s, z) = ∇f − Aᵢᵀ(z − μβ₁e) = 0
  ∇ₛL(x, s, z) = z − μS⁻¹e = 0
  Z(cᵢ − s + μw) = 0
  cᵢ − s + μw = 0
  z ≥ 0
  s > 0,
```

where cᵢ = cᵢ(x). Everything but the final inequality is due to the standard KKT theorem. The final inequality is easy to show by contradiction: if (x, s) is a local solution for which s ≤ 0, then the objective is not defined (this also works for the complex logarithm, since the objective will be lower for (x, s') where s' is any positive real number.)

We will simplify these conditions to make it easier to apply Newton's method to the equalities. Since s ≥ 0, we can left-multiply the second equation in the above necessary condition by S without changing the solution. Furthermore, the third and fourth equation are redundant, so we eliminate the third equation. These modifications give the following KKT conditions, which are equivalent, i.e., they hold if and only if the previous conditions hold (the proof is left as an exercise):

```
  ∇f − Aᵢᵀ(z − μβ₁e) = 0
  Sz − μe = 0
  cᵢ − s + μw = 0
  z ≥ 0
  s > 0.
```

We can now see how choosing to scale the log-barrier term by μ in problem [(3)](#primal-log-barrier-ipm-problem) allows μ to control the rate of decrease in complementarity by setting Sz = μe. When we apply Newton's method to these conditions, our Newton steps will try to force complimentarity to reduce to whatever μ is. This is a relatively standard choice for primal-dual IPMs; targeting cᵢ − s = −μw is the nonstandard piece that allows us to bound the duals.

We make a final clever substitution, due originally to [^8], which ensures all steps satisfy complementarity and strict positivity of s and z. The substitution rests on the following claim:

Claim: Let s, z ∈ **R**ᵐ and μ > 0. We have z = √(μ)eᵛ and s = √(μ)e⁻ᵛ if and only if s, z > 0 and Sz = μe.

Proof: We note that since μ > 0, the absolute value |μ| = μ.
If z = √(μ)eᵛ and s = √(μ)e⁻ᵛ, then Sz = |μ|eᵛ⁻ᵛ = μe⁰ = μe. Furthermore, since μ > 0 and the image of the exponential is the set of positive reals, we have s, z > 0.
Conversely, if Sz = μe and s, z > 0, then since the exponential is surjective onto the positive reals and z/√(μ) > 0, there exists v ∈ **R**ᵐ such that eᵛ = z/√(μ), hence √(μ)eᵛ = z. We have Sz = |μ| √(μ)√(μ)e, therefore Sz/√(μ) = √(μ) and we can substitute in z = √(μ)eᵛ to get Seᵛ = √(μ) and then perform the right- Hadamard (elementwise) product with e⁻ᵛ to get Seᵛ⁻ᵛ = Se⁰ = s = √(μ)e⁻ᵛ. ∎

As a result of the previous claim, if the necessary conditions for the shifted, primal-dual log-barrier problem [(5)](#shifted-primal-dual-log-barrier-ipm-problem) hold and μ > 0, then Sz = μe and s > 0 and z is **strictly** positive (due to μ > 0 and Sz = μe), hence z = √(μ)eᵛ and s = √(μ)e⁻ᵛ. We can therefore substitute these equalities into the first-order necessary conditions to get a new set of necessary conditions:

```
  ∇f − Aᵢᵀ(√(μ)eᵛ − μβ₁e) = 0
  μe − μe = 0
  cᵢ − √(μ)e⁻ᵛ + μw = 0
  √(μ)eᵛ  ≥ 0
  √(μ)e⁻ᵛ > 0.
```

The complementarity condition and non-negativity/strict-positivity conditions are now always satisfied, so they can be omitted to get the final log-domain necessary conditions for the shifted, primal-dual log-barrier problem [(5)](#shifted-primal-dual-log-barrier-ipm-problem)

```
  F₁(x, v) = ∇ₓL(x, v) = ∇f − Aᵢᵀ(√(μ)eᵛ − μβ₁e) = 0,
  F₂(x, v)             = cᵢ − √(μ)e⁻ᵛ + μw = 0.
```

### Step computation

In effect, we would like to find a solution to the nonlinear equation F(x, v) = concat(F₁(x, v), F₁(x, v)) = 0. We follow the standard Newton's method approach (for nonlinear equations) to approximately solve for the step that gives the solution F(x + pˣ, v + pᵛ) = 0: we form a linear model of a step about the current iterate (x, v) by taking the first-order Taylor series expansion, which gives

```
F(x, v) + J(x, v)[pˣ, pᵛ] = 0,
```

where J(x, v) is the Jacobian of F(x, v) with respect to (x, v). We can expand the Jacobian in the above equation to get

```
  [∂F₁/∂x  ∂F₁/∂v][pˣ]   [-F₁]
  [              ][  ] = [   ].
  [∂F₂/∂x  ∂F₂/∂v][pᵛ]   [-F₂]
```

We now write the sub-Jacobians, letting H = ∂²L/∂x² for brevity:

```
  ∂F₁/∂x = ∂²L/∂x²  = H
  ∂F₁/∂v = ∂²L/∂x∂v = -√(μ)Aᵢᵀeᵛ
  ∂F₂/∂x = Aᵢ
  ∂F₂/∂v = √(μ)diag(e⁻ᵛ).
```

We substitute these into the previous matrix equation and expand the right-hand side

```
  [H    -√(μ)Aᵢᵀeᵛ  ][pˣ]   [ -∇f + Aᵢᵀ(√(μ)eᵛ − μβ₁e) ]
  [                 ][  ] = [                          ].
  [Aᵢ  √(μ)diag(e⁻ᵛ)][pᵛ]   [    -cᵢ + √(μ)e⁻ᵛ − μw    ]
```

Solve the third row for pᵛ

```
  Aᵢpˣ + √(μ)e⁻ᵛ∘pᵛ = -cᵢ + √(μ)e⁻ᵛ − μw
  √(μ)e⁻ᵛ∘pᵛ = -Aᵢpˣ − cᵢ + √(μ)e⁻ᵛ − μw
  pᵛ = −1/√(μ) Aᵢeᵛ∘pˣ − 1/√(μ) eᵛ∘cᵢ + e − μ/√(μ) eᵛ∘w
  pᵛ = e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ − μw).
```

Substitute the explicit formula for pᵛ into the first row

```
  Hpˣ − √(μ)Aᵢᵀeᵛ∘pᵛ = −∇f + Aₑᵀy + √(μ)Aᵢᵀeᵛ
  Hpˣ − √(μ)Aᵢᵀeᵛ∘(e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ − μw)) = -∇f + Aᵢᵀ(√(μ)eᵛ − μβ₁e).
```

Expand and simplify

```
  Hpˣ − Aᵢᵀeᵛ∘(√(μ)e − eᵛ∘(Aᵢpˣ + cᵢ − μw)) = -∇f + Aᵢᵀ(√(μ)eᵛ − μβ₁e)
  Hpˣ − √(μ)Aᵢᵀeᵛ + Aᵢᵀe²ᵛ∘(Aᵢpˣ + cᵢ − μw) = -∇f + Aᵢᵀ(√(μ)eᵛ − μβ₁e)
  Hpˣ − √(μ)Aᵢᵀeᵛ + Aᵢᵀdiag(e²ᵛ)Aᵢpˣ + Aᵢᵀe²ᵛ∘cᵢ − μAᵢᵀe²ᵛ∘w = -∇f + Aᵢᵀ(√(μ)eᵛ − μβ₁e)
  Hpˣ + Aᵢᵀdiag(e²ᵛ)Aᵢpˣ + Aᵢᵀe²ᵛ∘cᵢ − μAᵢᵀe²ᵛ∘w = -∇f + 2√(μ)Aᵢᵀeᵛ − μβ₁Aᵢᵀe
  Hpˣ + Aᵢᵀdiag(e²ᵛ)Aᵢpˣ = -∇f + 2√(μ)Aᵢᵀeᵛ − Aᵢᵀe²ᵛ∘cᵢ + μAᵢᵀe²ᵛ∘w − μβ₁Aᵢᵀe
  (Hpˣ + Aᵢᵀdiag(e²ᵛ)Aᵢ)pˣ = -∇f + 2√(μ)Aᵢᵀeᵛ − Aᵢᵀe²ᵛ∘cᵢ + μAᵢᵀe²ᵛ∘w − μβ₁Aᵢᵀe
  (Hpˣ + Aᵢᵀdiag(e²ᵛ)Aᵢ)pˣ = -∇f + Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘cᵢ + μ(e²ᵛ∘w − β₁e)).
```

Substitute the new first and second rows into the system

```
  [H + Aᵢᵀdiag(e²ᵛ)Aᵢ  0][pˣ] = -[∇f − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘cᵢ + μ(e²ᵛ∘w − β₁e))]
  [        0           I][pᵛ]    [       e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ − μw)       ].
```

Eliminate the second row and column

```
  [H + Aᵢᵀdiag(e²ᵛ)Aᵢ][ pˣ] = -[∇f − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘cᵢ + μ(e²ᵛ∘w − β₁e))];
```

notably, if one substitutes s and z back into the system on the left-hand side, the re-substituted system is the "primal" Schur compliment of the original reduced KKT system (see sub-section 2.1.1 of [^5]).

### Final results

In summary, the reduced system gives the iterate pˣₖ

```
  [H + Aᵢᵀdiag(e²ᵛ)Aᵢ][pˣₖ ] = -[∇f − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘cᵢ + μ(e²ᵛ∘w − β₁e))].
```

The iterate pᵛ is given by

```
  pᵛ = e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ − μw).
```

The iterates are applied like so

```
  αᵛₖ = min(1, 1/|pᵛₖ|_∞²),

  xₖ₊₁ = xₖ + αₖ  pˣₖ
  vₖ₊₁ = vₖ + αᵛₖ pᵛₖ.
```

where αₖ is found via backtracking line search. A filter method determines acceptance of pˣ.

Section 6 of [^3] describes how to check for local infeasibility.

## Works cited

[^1]: Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19. Springer, 2006.

[^2]: Wächter, A. and Biegler, L. "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming", 2005. [http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf](http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf)

[^3]: Byrd, R. and Nocedal, J. and Waltz, R. "KNITRO: An Integrated Package for Nonlinear Optimization", 2005. [https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf](https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf)

[^4]: Gu, C. and Zhu, D. "A Dwindling Filter Algorithm with a Modified Subproblem for Nonlinear Inequality Constrained Optimization", 2014. [https://sci-hub.st/10.1007/s11401-014-0826-z](https://sci-hub.st/10.1007/s11401-014-0826-z)

[^5]: Hinder, O. and Ye, Y. "A one-phase interior point method for nonconvex optimization", 2018. [https://arxiv.org/pdf/1801.03072.pdf](https://arxiv.org/pdf/1801.03072.pdf)

[^6]: Permenter, F. "Log-domain interior-point methods for convex quadratic programming", 2022. [https://arxiv.org/pdf/2212.02294](https://arxiv.org/pdf/2212.02294)

[^7]: https://arxiv.org/pdf/1707.07327
