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

In summary, the following system gives the iterate pₖˣ.

```
  Hpˣ = −∇f(x)
```

The iterate is applied like so

```
  xₖ₊₁ = xₖ + pₖˣ
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

In summary, the reduced 2x2 block system gives the iterates pₖˣ and pₖʸ.

```
  [H   Aₑᵀ][ pˣ] = −[∇f(x) − Aₑᵀy]
  [Aₑ   0 ][−pʸ]    [     cₑ     ]
```

The iterates are applied like so

```
  xₖ₊₁ = xₖ + pₖˣ
  yₖ₊₁ = yₖ + pₖʸ
```

Section 6 of [^3] describes how to check for local infeasibility.

## Log-domain interior-point method

We want to solve the following optimization problem <a name="original-ipm-problem">(1)</a>

```
   min f(x),
    x
  s.t. cₑ(x) = 0
       ĉᵢ(x) ≥ 0
```

where f(x) is the cost function, cₑ(x) is the vector of equality constraints, and ĉᵢ(x) is vector of inequality constraints.

We'll reformulate the equality constraints as two inequality constraints: that is, we define a new inequality constraint vector cᵢ(x) = concat(cₑ(x), −cₑ(x), ĉᵢ(x)) (also see the end of section 2 of [6] for more information on this unusual choice). This gives a new but equivalent problem

```
   min f(x),
    x
  s.t. cᵢ(x) ≥ 0.
```

We would like to control the rate at which we reduce the primal feasibility since, for reasons outlined in [7], we would like the rate of decrease to be proportional to the rate at which we decrease complimentarity. We can achieve this by making primal feasibility and complimentarity proportional to a parameter μ ∈ (0, ∞), since for a sequence of these parameters (μₖ)ₖ ⊆ (0, ∞), the rate of decrease in complimentarity and primal feasibility will be μₖ₊₁/μₖ for all k ∈ **Z**₊ if and only if the respective proportionality constants are fixed across iterations. We choose such a constant w ∈ **R**ᵐ to be fixed across iterations and define a new modified problem <a name="homotopy-ipm-problem">(2)</a>


```
   min f(x),
    x
  s.t. cᵢ(x) ≥ μw;
```

note that in the above problem, μ only controls the rate of decrease of primal feasibility and *not* complimentarity, which is a goal we will return to later.

Also note that problem [(2)](#homotopy-ipm-problem) is equivalent to the original problem [(1)](#original-ipm-problem) if and only if μ = 0

We can eliminate the inequality constraints by adding a "log-barrier term" to the objective which penalizes constraint violation---this gives a new modified problem <a name="primal-log-barrier-ipm-problem">(3)</a>

```
   min f(x) - μ ∑ ln[(cᵢ)ⱼ − μwⱼ].
    x           j
```

In general, the above problem is neither equivalent to the first modified problem [(2)](#homotopy-ipm-problem) nor to the original problem [(1)](#original-ipm-problem) for any μ. At this point, we could solve a sequence of these primal log-barrier problems with decreasing barrier parameters with any unconstrained optimization algorithm. This is one of the reasons we have also scaled the log-barrier by μ since it makes the log-barrier vanish as μ vanishes (there is another important reason under a different problem formulation, which we will return to shortly).

For reasons outlined in section 19.6 of [1], the primal log-barrier objective is highly nonlinear as μ approaches 0 which results in slow convergence when solving a series of primal log-barrier problems [(3)][#primal-log-barrier-ipm-problem]. To deal with this issue, we define "slack variables" s = cᵢ(x) which we add to and substitute into problem [(3)](#primal-log-barrier-ipm-problem) 


<!-- We give up on the above primal log-barrier problem [(3)](#primal-log-barrier-ipm-problem) due to the nonlinearity, but we include it above because it is the objective they show in [6]. We still want to control the rate of decrease in primal feasibility, so we define "slack variables" s = cᵢ(x) which we add to and substitute into problem [(2)](#homotopy-ipm-problem) to get an equivalent problem <a name="slack-homotopy-ipm-problem">(4)</a>

```
   min f(x).
   x,s
  s.t. s = cᵢ(x) (⇔ cᵢ(x) − s = 0)
       s ≥ μw    (⇔ s − μw ≥ 0)
``` -->

Note that the state is now (x, s) ∈ **R**ⁿ × **R**ᵐ, and that this is not equivalent to the original problem [(1)](#original-ipm-problem), although again we could solve a sequence of these problems as μ → 0. Indeed, the final algorithm we derive in the remainder of the section can also be derived as a homotopy method applied to the KKT conditions of the above problem (see section 19.1 in [1] for some information on this duality.) 

Instead of deriving our algorithm as a homotopy method, will will derive it as a barrier method (we need to find the barrier objective since it ends up being our merit function): we again eliminate the inequality constraints (now on the slack s) by adding a log-barrier term to the objective, which gives a new modified, non-equivalent problem <a name="primal-dual-log-barrier-ipm-problem">(5)</a>

```
   min f(x) - μ ∑ ln[sⱼ − μwⱼ].
   x,s          j
   s.t. cᵢ(x) − s = 0
```

Note that, as in [(3)](#primal-log-barrier-ipm-problem), we have placed the same primal feasibility decrease control term μ outside the sum. As mentioned previously, this allows the sequence of solutions to [(5)](#primal-dual-log-barrier-ipm-problem) to approach the solution to the original problem since as μ vanishes the log-barrier vanishes, which is why μ is traditionally called the barrier parameter. We will also show in the next section that scaling the sum by μ also causes μ to control the rate of decrease in complimentarity, which achieves our goal of decreasing complimentarity and primal feasibility at the same rate.

<!-- where μ is the barrier parameter, β₁ ∈ **R**, and w ∈ [0, ∞)ⁿ is a vector parameter fixed across iterations which we will examine in more detail later. Take care to note that the state is now (x, s)∈ **R**ⁿ × **R**ᵐ. -->

Finally, following [7], we add another term to the sum to bound each summand below so that the primal iterates do not spuriously diverge, which gives the following problem <a name="shifted-primal-dual-log-barrier-ipm-problem">(6)</a>

```
  min f(x) − μ ∑ [β₁(cᵢ)ⱼ(x) + ln(μwⱼ + sⱼ)],  (*)
  x,s          j
  s.t. cᵢ(x) - s = 0
```

where β₁ ∈ **R**.

### Lagrangian

The Lagrangian of the final barrier problem [(6)](#shifted-primal-dual-log-barrier-ipm-problem) is

```
  L(x, s, z) = f(x) − μ ∑ [β₁(cᵢ)ⱼ(x) + ln(μwⱼ + sⱼ)] − zᵀ(cᵢ(x) - s)
                        j
```

### Gradients of the Lagrangian

The gradient of the Lagrangian of the barrier problem [(6)](#shifted-primal-dual-log-barrier-ipm-problem) with respect to the state (x, s) ∈ **R**ⁿ × **R**ᵐ is

```
  ∇ₓL(x, s, z) = ∇f − Aᵢᵀ(z - μβ₁e)
  ∇ₛL(x, s, z) = z − μ diag(s + μw)⁻¹e
```

where ∇f = ∇f(x), Aᵢ = ∂cᵢ/∂x, and e is a column vector of ones.

We will now write the first-order necessary conditions: if (x, s) ∈ **R**ⁿ × **R**ᵐ is a local solution to the barrier problem [(6)](#shifted-primal-dual-log-barrier-ipm-problem) at which an appropriate constraint qualification holds, then there exists a Lagrange multiplier z ∈ **R**ᵐ such that

```
  ∇ₓL(x, s, z) = ∇f − Aᵢᵀ(z - μβ₁e) = 0
  ∇ₛL(x, s, z) = z − μ diag(s + μw)⁻¹e = 0
  Z(cᵢ − s) = 0
  cᵢ - s = 0
  z ≥ 0,
```

where cᵢ = cᵢ(x).

```
  z − μ diag(s + μw)⁻¹e = 0 ⇔  z = μ diag(s + μw)⁻¹e ⇔  diag(z) = Z = μ diag(s + μw)⁻¹
  Z(cᵢ − s) = 0
imply
  μ diag(s + μw)⁻¹(cᵢ − s) = 0
implies
  μ diag(s + μw)⁻¹cᵢ - 
```


We will simplify these conditions to make it easier to apply Newton's method to the equalities. If we let S = diag(s) and W = diag(w), then 

```
  z − μ diag(s + μw)⁻¹e = 0
  diag(s + μw) z        = μe
  Sz + μWz              = μe
  (S + μW)z             = μe
  μ⁻¹Sz + Wz            =  e
```

```
  ∇f − Aₑᵀy − Aᵢᵀz = 0
  Sz − μe = 0
  cₑ = 0
  cᵢ − s = 0
  s ≥ 0
```

To ensure s ≥ 0 and z ≥ 0, make the following substitutions.

```
  s = √(μ)e⁻ᵛ
  z = √(μ)eᵛ
```
```
  ∇f − Aₑᵀy − Aᵢᵀ√(μ)eᵛ = 0
  cₑ = 0
  cᵢ − √(μ)e⁻ᵛ = 0

  ∇f − Aₑᵀy − √(μ)Aᵢᵀeᵛ = 0
  cₑ = 0
  cᵢ − √(μ)e⁻ᵛ = 0
```

The complementarity condition is now always satisfied, so it can be omitted.

### Newton's method

Next, we'll apply Newton's method to the optimality conditions. Let H be ∂²L/∂x², pˣ be the step for x, pʸ be the step for y, and pᵛ be the step for v.

```
  ∇ₓL(x + pˣ, y + pʸ, v + pᵛ)
    ≈ ∇ₓL(x, y, v) + ∂²L/∂x²pˣ + ∂²L/∂x∂ypʸ + ∂²L/∂x∂vpᵛ
  ∇ₓL(x, y, v) + Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘pᵛ = 0
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘pᵛ = −∇ₓL(x, y, v)
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘pᵛ = −(∇f − Aₑᵀy − √(μ)Aᵢᵀeᵛ)
```
```
  ∇_yL(x + pˣ, y + pʸ, v + pᵛ)
    ≈ ∇_yL(x, y, v) + ∂²L/∂y∂xpˣ + ∂²L/∂y²pʸ + ∂²L/∂y∂vpᵛ
  ∇_yL(x, y, v) + Aₑpˣ = 0
  Aₑpˣ = −∇_yL(x, y, v)
  Aₑpˣ = −cₑ
```
```
  ∇ᵥL(x + pˣ, y + pʸ, v + pᵛ)
    ≈ ∇ᵥL(x, y, v) + ∂²L/∂v∂xpˣ + ∂²L/∂v∂ypʸ + ∂²L/∂v²pᵛ
  ∇ᵥL(x, y, v) + Aᵢpˣ + √(μ)e⁻ᵛ∘pᵛ = 0
  Aᵢpˣ + √(μ)e⁻ᵛ∘pᵛ = −∇ᵥL(x, y, v)
  Aᵢpˣ + √(μ)e⁻ᵛ∘pᵛ = −(cᵢ − √(μ)e⁻ᵛ)
```

### Matrix equation

Group them into a matrix equation.

```
  [H   −Aₑᵀ  −√(μ)Aᵢᵀeᵛ][pˣ]    [∇f − Aₑᵀy − √(μ)Aᵢᵀeᵛ]
  [Aₑ   0         0    ][pʸ] = −[          cₑ         ]
  [Aᵢ   0     √(μ)e⁻ᵛ  ][pᵛ]    [    cᵢ − √(μ)e⁻ᵛ     ]
```

Invert pʸ.

```
  [H   Aₑᵀ  −√(μ)Aᵢᵀeᵛ][ pˣ]    [∇f − Aₑᵀy − √(μ)Aᵢᵀeᵛ]
  [Aₑ   0       0     ][−pʸ] = −[         cₑ          ]
  [Aᵢ   0    √(μ)e⁻ᵛ  ][ pᵛ]    [    cᵢ − √(μ)e⁻ᵛ     ]
```

Solve the third row for pᵛ.

```
  Aᵢpˣ + √(μ)e⁻ᵛ∘pᵛ = −cᵢ + √(μ)e⁻ᵛ
  √(μ)e⁻ᵛ∘pᵛ = −Aᵢpˣ − cᵢ + √(μ)e⁻ᵛ
  pᵛ = −1/√(μ) Aᵢeᵛ∘pˣ − 1/√(μ) eᵛ∘cᵢ + e
  pᵛ = e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ)
```

Substitute the explicit formula for pᵛ into the first row.

```
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘pᵛ = −∇f + Aₑᵀy + √(μ)Aᵢᵀeᵛ
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ∘(e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ)) = −∇f + Aₑᵀy + √(μ)Aᵢᵀeᵛ
```

Expand and simplify.

```
  Hpˣ − Aₑᵀpʸ − Aᵢᵀeᵛ∘(√(μ) − eᵛ∘(Aᵢpˣ + cᵢ)) = −∇f + Aₑᵀy + √(μ)Aᵢᵀeᵛ
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ + Aᵢᵀe²ᵛ∘(Aᵢpˣ + cᵢ) = −∇f + Aₑᵀy + √(μ)Aᵢᵀeᵛ
  Hpˣ − Aₑᵀpʸ − √(μ)Aᵢᵀeᵛ + Aᵢᵀdiag(e²ᵛ)Aᵢpˣ + Aᵢᵀe²ᵛ∘cᵢ = −∇f + Aₑᵀy + √(μ)Aᵢᵀeᵛ
  Hpˣ − Aₑᵀpʸ + Aᵢᵀdiag(e²ᵛ)Aᵢpˣ + Aᵢᵀe²ᵛ∘cᵢ = −∇f + Aₑᵀy + 2√(μ)Aᵢᵀeᵛ
  Hpˣ − Aₑᵀpʸ + Aᵢᵀdiag(e²ᵛ)Aᵢpˣ = −∇f + Aₑᵀy + 2√(μ)Aᵢᵀeᵛ − Aᵢᵀe²ᵛ∘cᵢ
  (Hpˣ + Aᵢᵀdiag(e²ᵛ)Aᵢ)pˣ − Aₑᵀpʸ = −∇f + Aₑᵀy + 2√(μ)Aᵢᵀeᵛ − Aᵢᵀe²ᵛ∘cᵢ
  (Hpˣ + Aᵢᵀdiag(e²ᵛ)Aᵢ)pˣ − Aₑᵀpʸ = −∇f + Aₑᵀy + Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘cᵢ)
```

Substitute the new first and third rows into the system.

```
  [H + Aᵢᵀdiag(e²ᵛ)Aᵢ  Aₑᵀ  0][ pˣ]    [∇f − Aₑᵀy − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘cᵢ)]
  [        Aₑ           0   0][−pʸ] = −[               cₑ                ]
  [        0            0   I][ pᵛ]    [    e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ)    ]
```

Eliminate the third row and column.

```
  [H + Aᵢᵀdiag(e²ᵛ)Aᵢ  Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘cᵢ)]
  [        Aₑ           0 ][−pʸ]    [               cₑ                ]
```

### Final results

In summary, the reduced 2x2 block system gives the iterates pₖˣ and pₖʸ.

```
  [H + Aᵢᵀdiag(e²ᵛ)Aᵢ  Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘cᵢ)]
  [        Aₑ           0 ][−pʸ]    [               cₑ                ]
```

The iterate pᵛ is given by

```
  pᵛ = e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ)
```

The iterates are applied like so

```
  αₖᵛ = min(1, 1/|pᵛ|_∞²)

  xₖ₊₁ = xₖ + αₖpₖˣ
  yₖ₊₁ = yₖ + αₖpₖʸ
  vₖ₊₁ = vₖ + αₖᵛpₖᵛ
```

where αₖ is found via backtracking line search. A filter method determines acceptance of pˣ.

Section 6 of [^3] describes how to check for local infeasibility.

## Works cited

[^1]: Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19. Springer, 2006.

[^2]: Wächter, A. and Biegler, L. "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming", 2005. [http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf](http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf)

[^3]: Byrd, R. and Nocedal, J. and Waltz, R. "KNITRO: An Integrated Package for Nonlinear Optimization", 2005. [https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf](https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf)

[^4]: Gu, C. and Zhu, D. "A Dwindling Filter Algorithm with a Modified Subproblem for Nonlinear Inequality Constrained Optimization", 2014. [https://sci-hub.st/10.1007/s11401-014-0826-z](https://sci-hub.st/10.1007/s11401-014-0826-z)

[^5]: Permenter, F. "Log-domain interior-point methods for convex quadratic programming", 2022. [https://arxiv.org/pdf/2212.02294](https://arxiv.org/pdf/2212.02294)

[^6]: https://arxiv.org/pdf/1707.07327

[^7]: https://arxiv.org/pdf/1801.03072
