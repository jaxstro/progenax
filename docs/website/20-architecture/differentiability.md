---
title: Differentiability rules
description: The patterns that preserve gradient flow through progenax — fixed-iteration scan over while-loop, smooth-threshold sigmoids over hard masks, soft argmin over hard — and the antipatterns that silently break differentiability.
---

# Differentiability rules

Differentiability is a contract between an input parameter, an observable, a
parameter domain and a randomness convention. Progenax supports continuous
construction and sampling derivatives on qualified paths; this does not guarantee
an accurate gradient for every downstream observable or discrete population choice.

This page describes implementation patterns and their limitations. Measured
accuracy belongs to the relevant validation page.

(engine-a-ad)=
## Engine A construction and sampling

All three Engine A constructors accept a static `adjoint` choice. The default
supports checkpointed reverse AD (`jax.grad`, `jax.jacrev`). Select
`diffrax.ForwardMode()` for forward AD (`jax.jvp`, `jax.jacfwd`), or
`diffrax.DirectAdjoint()` when both modes are needed. These choices concern the
ODE solve; the complete observable still needs a derivative check.

`from_imf` differentiates its converged central-fraction constraint using an
implicit Jacobian–vector product. Inspect the returned `residual`: differentiating
the constraint does not qualify an unconverged fit. The tidal-root, radial CDF and
velocity sampler share the unclipped monotone-cubic potential representation;
the public potential array remains nonnegative.

:::{important} Two different gradient contracts
A fixed-realization derivative holds random draws and component assignments
locally fixed. A population-expectation derivative must also account for changing
component probabilities. `estimate_expectation` combines pathwise and categorical
score terms with a leave-one-out baseline. Its identity checks pass; statistical
precision and full-constructor population-observable qualification remain open.
Higher derivatives are not qualified. See the
[Engine A validation matrix](../50-validation/multimass-equilibrium.md#engine-a-matrix).
:::

## Three preservation patterns

```{list-table}
:header-rows: 1

* - Pattern
  - What it replaces
  - Examples in progenax
* - **Fixed-iteration `lax.scan`**
  - `while_loop` with data-dependent termination
  - Fixed-iteration Newton solvers and inverse-CDF lookups
* - **Smooth threshold (sigmoid)**
  - Hard mask `where(x > t, 1, 0)`
  - Approved smooth environment-IMF continuation
* - **Sort-by-value gradient flow**
  - argmax / argmin in critical paths
  - Value-based statistics with locally fixed ordering
```

Each pattern has a specific purpose. None of them is the "right
answer" universally; they are tools for specific obstructions.

## Fixed-iteration scan

`jax.lax.while_loop` does not support ordinary reverse-mode AD because its
memory requirements lack a static bound; this is not a statement that the
underlying mathematical solution has no derivative. A fixed-length `lax.scan`
is one option. Implicit differentiation or an ODE adjoint can differentiate a
converged solution without differentiating every stopping decision. See the
[JAX loop contract](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html).

```python
# WRONG — while_loop with data-dependent termination
def buggy_kepler_solve(M, e):
    def cond(state):
        E, residual = state
        return jnp.abs(residual) > 1e-12   # Data-dependent
    def body(state):
        E, _ = state
        new_E = E - (E - e * jnp.sin(E) - M) / (1.0 - e * jnp.cos(E))
        return new_E, new_E - e * jnp.sin(new_E) - M
    final, _ = jax.lax.while_loop(cond, body, (M, jnp.inf))
    return final

# RIGHT — fixed iteration count
def correct_kepler_solve(M, e, n_iter=10):
    def body(E, _):
        new_E = E - (E - e * jnp.sin(E) - M) / (1.0 - e * jnp.cos(E))
        return new_E, None
    final, _ = jax.lax.scan(body, M, jnp.arange(n_iter))
    return final
```

10 Newton iterations gives double-precision convergence for $e \le
0.9$; 20 iterations covers $e \le 0.99$. The fixed count is the
"tunable conservatism" — pick it large enough to converge in the
worst case the function will see, then live with the constant cost.

```{admonition} How to pick the iteration count
:class: note
Run the solver with $n = 100$ on a representative parameter sweep,
plot the residual at each step, and identify the smallest $n$ at
which the worst-case residual is below the desired tolerance. Add a
small safety margin. progenax's Newton solvers use $n = 10$–$30$
typically; the King-ODE adaptive integrator uses up to $n = 1000$
internal Tsit5 steps.
```

## Smooth thresholds

An indicator such as `jnp.where(x > t, 1, 0)` has zero classical derivative
away from its threshold and no classical derivative at the jump. AD does not
supply the missing probability-boundary contribution. A sigmoid defines a
different, smooth model; use it only when that approximation is intended and
approved. Physical support boundaries must not be softened merely to obtain a
gradient. Keep both branches numerically valid: an unselected NaN can contaminate
reverse derivatives ([JAX `where` notes](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html)).

```python
# WRONG — hard mask, gradient is zero almost everywhere
mask = jnp.where(s > s_t, 1.0, 0.0)

# RIGHT — soft sigmoid, smooth gradient
kappa = 10.0   # Width parameter; large = closer to hard
mask = jax.nn.sigmoid(kappa * (s - s_t))
```

The width $\kappa$ controls the trade-off:

```{list-table}
:header-rows: 1

* - $\kappa$
  - Behavior
  - Use case
* - $\to 0$
  - Smooth — mask varies linearly across full $s$
  - Gradient evaluation; small for inference
* - 10
  - Standard "soft" mask
  - **progenax default** for BM19 tail mask
* - 100
  - Sharp — approaches hard threshold
  - When the soft transition has biased the answer
* - $\to \infty$
  - Hard mask
  - Forward Monte Carlo only; not for HMC
```

The BM19 example concerns the excluded experimental subsystem, not a released
core default. A sigmoid width changes the forward model and must be justified
for its intended observable.

The same pattern applies elsewhere:

- The Fundamental Plane threshold $\hat x \ge -0.87$ in
  [](../10-theory/imfs/environment.md) uses a sigmoid.
- The "in or out of bounds" check in `apply_tidal_truncation` uses a
  sigmoid by default (the hard-mask version is available but breaks
  inference).

## Sort-by-value gradient flow

The permutation selected by sorting is locally constant when values are
distinct. The sorted values are locally a permutation of the inputs, so their
Jacobian is a permutation matrix, not zero. Ties and order-dependent downstream
choices need a separate contract:

```python
# Wrong understanding — but progenax does this correctly
sorted_values = jnp.sort(values)
mean_value = jnp.mean(sorted_values)   # Gradient flows!
```

The gradient `∂(mean_value)/∂(values)` is well-defined: it does not
depend on the permutation (since the mean is symmetric). progenax
exploits this in several places:

- **Mass segregation** ([](../10-theory/tidal-and-substructure/mass-segregation.md)):
  the energy-ordered orbit assignment uses `argsort` on energies, but
  the gradient flows through the masses.

A derivative through sorted values is local to an ordering region. An
order-dependent assignment can change discontinuously at a tie, and a
population expectation may require contributions from changing assignments.

## Antipatterns to avoid

The following patterns require an explicit transformation and derivative
contract. The local gradient registry checks selected entry-point, parameter,
and observable combinations; it does not establish every builder derivative.

### Antipattern 1: `while_loop` in core code

For reverse AD, use a fixed-length scan or an appropriate implicit/adjoint rule;
verify convergence and the derivative of the intended solution.

### Antipattern 2: hard masks `where(x > t, ...)`

Preserve physical support and define the required derivative contract. A smooth
replacement changes the model and needs scientific justification; a categorical
expectation requires its probability contribution, as in Engine A above.

### Antipattern 3: in-place mutation `array[i] = val`

Use `array.at[i].set(val)` (functional update) instead.

### Antipattern 4: rejection sampling

Variable-cost-per-particle rejection sampling cannot be `vmap`'d
efficiently. Use inverse-CDF sampling with a fixed-iteration root
finder instead.

### Antipattern 5: `argmin` / `argmax` for critical decisions

The "nearest-neighbour" choice in the kNN-based CW04 Q approximation
([](jax-native-substructure-q.md)) uses `argmin` on distances, which
is not differentiable in the choice. progenax accepts this as a
"partially differentiable" diagnostic, with the documentation
making the limitation explicit.

### Antipattern 6: data-dependent shape changes

Functions whose output shape depends on the data (e.g. filtering and
returning a smaller array) are not JIT-able and not directly
differentiable. progenax uses *masks* of fixed shape instead, with
the consumer responsible for honouring the mask.

## Validating differentiability

A finite-gradient smoke check can miss a silently blocked zero. For example,
the following illustrative assertion alone is insufficient:

```python
@pytest.mark.parametrize("builder", [
    plummer_builder, king_builder, eff_builder,
])
def test_builder_grad_finite(builder):
    """Gradient of any scalar observable w.r.t. r_h is finite."""
    def loss_fn(r_h):
        ic = builder(r_h=r_h, alpha=2.3, key=jax.random.PRNGKey(0))
        return jnp.sum(ic.state.positions ** 2)   # Some observable

    grad = jax.grad(loss_fn)(1.0)
    assert jnp.isfinite(grad), "Gradient is not finite"
```

Compare the intended observable's derivative with an independent analytic
result or a converged finite-difference sequence. Keep random draws fixed for
a pathwise check, state the parameter domain, and test the required forward
or reverse mode. The [gradient audit](../50-validation/differentiability-audit.md)
records existing cases and limitations. A local check does not imply that remote
CI is enabled.

## What "partial differentiability" means

Some progenax modules are explicitly *partially* differentiable:

```{list-table}
:header-rows: 1

* - Module
  - Differentiable in
  - Not differentiable in
* - `KingProfile`
  - $r_c$, $M_{\mathrm{tot}}$, $W_0$ (profile shape, via `diffrax`), **and the
    scalar tidal radius $r_t$** (zero-crossing carries $\partial r_t/\partial W_0$ —
    see [](#roadmap-differentiable-rt))
  - —
* - Environment IMF helpers
  - $\rho_{\mathrm{cl}}$, $M_{\mathrm{ecl}}$, [Fe/H], slope continuation
  - Mass-bin boundaries (need sigmoid for full)
* - kNN substructure Q
  - Distances
  - NN-choice permutation
```

Each partial-differentiability case is documented in its module
docstring with the specific limitation and the workaround if one
exists. For inference targets that need gradients in a partially-
differentiable parameter, consult the module documentation for the
finite-difference fallback.

(roadmap-differentiable-rt)=
## Differentiable King tidal radius ($\partial r_t/\partial W_0$)

```{admonition} Tidal-root contract
:class: note
The tidal radius $r_t$ is the first zero of the dimensionless relative potential
$\psi$. With core radius $r_c$, define $\xi=r/r_c$ and $\xi_t=r_t/r_c$;
$W_0$ is the central dimensionless potential. Root derivatives require the
**unclamped** potential and a simple zero with nonzero radial slope.
A clipped zero plateau cannot retain that boundary sensitivity.
```

### Implicit root derivative

The boundary satisfies $\psi(\xi_t,W_0)=0$, so

$$\frac{\partial\xi_t}{\partial W_0}
=-\frac{\partial\psi/\partial W_0}{\partial\psi/\partial\xi}
\Bigg|_{\xi_t}.$$

The shared root helper uses a monotone cubic with common slopes at neighboring
saved nodes. This keeps the first root derivative continuous when the zero
crosses a node. A bracketed solve locates the zero; `jax.lax.custom_root`
differentiates its implicit equation, including the nodal slopes, rather than
the bisection decisions. Accuracy still depends on the ODE solution and saved grid.

### Shared implementation

King, Michie, LIMEPY, and multi-component Engine A use
`_find_tidal_radius` with the unclamped potential. Engine A also interpolates
that unclamped potential for radial CDF construction and velocity sampling;
physical density and speed support is restricted to the nonnegative-potential domain. The helper returns the domain endpoint if no
crossing exists; that fallback is not evidence of a physically truncated model.
A flat multiple zero is outside the differentiable root contract.

The finite-model factories `KingProfile.from_W0_rc`, `MichieProfile.from_W0_rc`
and `LIMEPYProfile.from_W0_rc`, the standalone `KingVelocityDF`, `MichieVelocityDF`
and `LIMEPYVelocityDF` constructors, and completed Engine A models require finite
saved potential values, a positive first value and an adjacent
positive-to-nonpositive bracket. This is the structural $W=0$ crossing condition;
it introduces no endpoint tolerance. Construction rejects a failed condition in
eager, compiled and supported first-order differentiated calls. Raw solvers and
manual constructors receiving precomputed profile arrays retain their separate
diagnostic contracts. Passing the factory check does not establish saved-grid,
ODE, observable or derivative accuracy; those still require convergence checks.

### Science cases it enables

```{list-table}
:header-rows: 1

* - Case
  - Uses $\partial r_t/\partial W_0$?
  - Why
* - **Galactic potential / orbits from cluster limiting radii**
  - ✅ headline
  - $r_t \approx r_J = \big(M_{\rm cl}/3M_{\rm gal}(<R)\big)^{1/3} R$; backprop
    observed truncation → $M_{\rm gal}(<R)$ and pericenter (HMC over structure +
    orbit + MW mass, with Gaia orbits).
* - **Roche-filling / tidal-coupling prior** $r_t = r_J(\text{orbit})$
  - ✅
  - Links cluster structure to its orbit; the hook for integrating
    `progenax.tidal` ([](../10-theory/tidal-and-substructure/tidal.md)) into a
    differentiable fit.
* - Catalog-concentration fitting $(c_{\rm model}-c_{\rm cat})^2/\sigma_c^2$
  - ✅
  - $c = \log_{10}(r_t/r_c)$ now backprops through $r_t$ directly.
* - Outer-edge observables (tidal-tail onset, count beyond $0.9\,r_t$)
  - ⚠️ downstream
  - Real, but routed through gravax / rendering.
* - Ordinary concentration inference
  - ✅ (already)
  - Covered by $W_0$ via the profile shape, now also via $r_t$.
```

The finite-difference grad-check for $r_t$ is part of the gradient-audit suite
([](../50-validation/differentiability-audit.md)); see
[](../50-validation/king-profile.md) for the King physics validation.

## References

The differentiability constraints follow directly from JAX's
programming model; see the [JAX docs on autodiff](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).
The "soft threshold for autodiff" pattern is widely used across the
JAX ML ecosystem. progenax's specific patterns are inspired by
{cite:t}`Equinox`'s "differentiable Python" idioms.
