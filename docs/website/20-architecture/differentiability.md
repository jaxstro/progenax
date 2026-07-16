---
title: Differentiability rules
description: The patterns that preserve gradient flow through progenax — fixed-iteration scan over while-loop, smooth-threshold sigmoids over hard masks, soft argmin over hard — and the antipatterns that silently break differentiability.
---

# Differentiability rules

progenax's promise is that every IC parameter — half-mass radius,
IMF slope, virial Q, mass-segregation strength (and, in the experimental
`gravoturb` package, gravoturbulent Mach and α) — is
**differentiable** through `jax.grad`. Gradients flow from any
downstream observable (final-snapshot energy, mock observation
likelihood) all the way back to the IC parameters. This is what makes
HMC inference of cluster properties tractable.

The promise is non-trivial. Real-world code is full of operations
that *silently* break differentiability — hard thresholds, while-loops
with data-dependent termination, in-place mutation, sort permutations.
This chapter catalogues the patterns progenax uses to preserve
differentiability, the antipatterns it forbids, and the
"gradient-not-defined" failure modes to watch for.

## Three preservation patterns

```{list-table}
:header-rows: 1

* - Pattern
  - What it replaces
  - Examples in progenax
* - **Fixed-iteration `lax.scan`**
  - `while_loop` with data-dependent termination
  - Newton solvers, ODE integrators, inverse-CDF lookups
* - **Smooth threshold (sigmoid)**
  - Hard mask `where(x > t, 1, 0)`
  - BM19 transition density, Fundamental Plane $\hat x \ge -0.87$ threshold
* - **Sort-by-value gradient flow**
  - argmax / argmin in critical paths
  - Fractal radial remap (gradients flow through values, not permutation)
```

Each pattern has a specific purpose. None of them is the "right
answer" universally; they are tools for specific obstructions.

## Fixed-iteration scan

`jax.lax.while_loop` is **not differentiable**. The iteration count
depends on data, so the gradient is undefined. The replacement is
`lax.scan` (or `lax.fori_loop`) with a *fixed* iteration count
chosen large enough to ensure convergence in the worst case.

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

Hard masks `jnp.where(x > t, 1, 0)` produce *zero* gradients almost
everywhere and a *delta function* at $x = t$. The gradient is
mathematically defined as a distribution but is not numerically
useful — autodiff returns zero everywhere except at the threshold,
where it returns infinity (or NaN under finite precision). The
replacement is a sigmoid:

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
  - Behaviour
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

progenax uses $\kappa = 10$ by default in BM19, giving a transition
width $\sim 0.1$ in $s$ ($\sim 10\%$ in $\rho$). The width can be
exposed as a free parameter for inference if needed.

The same pattern applies elsewhere:

- The Fundamental Plane threshold $\hat x \ge -0.87$ in
  [](../10-theory/imfs/environment.md) uses a sigmoid.
- The "in or out of bounds" check in `apply_tidal_truncation` uses a
  sigmoid by default (the hard-mask version is available but breaks
  inference).

## Sort-by-value gradient flow

Sorting is *piecewise constant* in its inputs — small changes to one
value can cause a discontinuous jump in the permutation. The
gradient of the sort operation itself is zero (or undefined), but the
gradient of *quantities computed from the sorted values* is well-defined
and useful:

```python
# Wrong understanding — but progenax does this correctly
sorted_values = jnp.sort(values)
mean_value = jnp.mean(sorted_values)   # Gradient flows!
```

The gradient `∂(mean_value)/∂(values)` is well-defined: it does not
depend on the permutation (since the mean is symmetric). progenax
exploits this in several places:

- **Fractal radial remap** ([](../10-theory/tidal-and-substructure/fractal.md)):
  the rank-based radius remap involves `argsort`, but the gradient
  flows through the *target radii* (which are values being sorted)
  rather than the permutation.
- **Mass segregation** ([](../10-theory/tidal-and-substructure/mass-segregation.md)):
  the energy-ordered orbit assignment uses `argsort` on energies, but
  the gradient flows through the masses.

The rule: **avoid `argmax` / `argmin` in critical paths**, but `sort`
itself is OK as long as the downstream code does not depend on the
permutation indices directly.

## Antipatterns to avoid

The following antipatterns silently break differentiability. progenax's
test suite includes gradient-validity tests for all builders to catch
them at CI time.

### Antipattern 1: `while_loop` in core code

Use `lax.scan` with fixed iteration count instead.

### Antipattern 2: hard masks `where(x > t, ...)`

Use sigmoid soft masks instead. If a hard mask is *strictly* required
(e.g. for a forward Monte Carlo where gradients are not needed),
isolate it in a `progenax.diagnostics`-style non-JIT module.

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

progenax's test suite includes per-builder gradient tests:

```python
@pytest.mark.parametrize("builder", [
    plummer_builder, king_builder, eff_builder, fractal_builder,
])
def test_builder_grad_finite(builder):
    """Gradient of any scalar observable w.r.t. r_h is finite."""
    def loss_fn(r_h):
        ic = builder(r_h=r_h, alpha=2.3, key=jax.random.PRNGKey(0))
        return jnp.sum(ic.state.positions ** 2)   # Some observable

    grad = jax.grad(loss_fn)(1.0)
    assert jnp.isfinite(grad), "Gradient is not finite"
```

Tests like these run on every CI build; a regression that breaks
differentiability gets caught immediately rather than at the next
HMC chain failure.

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
* - Fractal `chi`
  - $\chi$, $\sigma_u$, $\lambda_{\mathrm{frac}}$
  - Mode wavevectors / phases (frozen by `stop_gradient`)
```

Each partial-differentiability case is documented in its module
docstring with the specific limitation and the workaround if one
exists. For inference targets that need gradients in a partially-
differentiable parameter, consult the module documentation for the
finite-difference fallback.

(roadmap-differentiable-rt)=
## Differentiable King tidal radius ($\partial r_t/\partial W_0$)

```{admonition} Status: resolved
:class: tip
The King concentration $W_0$ is differentiable both for the density/velocity
*shape* observables one fits to data **and** for the **scalar tidal radius**
$r_t$. The original silent-zero hazard ($\partial r_t/\partial W_0 = 0$) was found
and fixed by the gradient audit ([](../50-validation/differentiability-audit.md)):
the public `KingProfile.from_W0_rc(W0, r_c).r_t` now carries a finite, exact
$\partial r_t/\partial W_0$ — verified AD-vs-FD to $\sim 10^{-8}$ (e.g.
$\partial r_t/\partial W_0 \approx 22.9$ at $W_0=7$, $r_c=1$). The forward value of
$r_t$ is unchanged by the fix; only the gradient differs.
```

### What was blocked, and why

`solve_king_profile` clamps the density-side potential with
`psi_clamped = jnp.maximum(psi_raw, 0.0)` — the clamp that keeps the
lowered-Maxwellian density gradient-safe at $\psi=0$. If the *tidal-radius finder*
is fed the **clamped** $\psi$, the first crossing node has $\psi_1 = 0$ exactly, so
the linear-interpolation weight $t = \psi_0/(\psi_0 - \psi_1) = 1$ snaps $\xi_t$ to
the fixed grid node — an integer index with zero gradient. The very clamp that
makes the *density* safe was what zeroed the *tidal-radius* slope.

### The fix (unclamped zero-crossing = implicit function theorem)

$\xi_t$ is implicitly defined by $\psi(\xi_t, W_0) = 0$, so by the implicit
function theorem

$$\frac{\partial \xi_t}{\partial W_0} = -\frac{\partial\psi/\partial W_0}{\partial\psi/\partial\xi}\Bigg|_{\xi_t}.$$

`solve_king_profile` returns *both* potentials — the clamped one for
density/CDF/velocity and the **unclamped** `psi_raw` (which goes negative just past
the crossing). `_find_tidal_radius` is fed `psi_raw`, so the linear-interpolation
crossing $\xi_t = \xi_0 + \psi_0/(\psi_0-\psi_1)\,(\xi_1-\xi_0)$ stays smooth in
$(\psi_0,\psi_1)$ and $\partial\xi_t/\partial W_0$ flows through the `diffrax`
solve — to grid accuracy, this *is* the implicit-function-theorem result above. No
`custom_jvp` is needed; the differentiability falls out of the unclamped-$\psi$
dual return. The same path differentiates $r_t$ across King, Michie, LIMEPY, and
the multi-component (Engine A + B) models.

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
