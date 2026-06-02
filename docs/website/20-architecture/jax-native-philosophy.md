---
title: JAX-native philosophy
description: Why progenax forbids numpy/scipy in core code, requires jax.lax.scan over Python loops, and uses equinox modules for state — the three constraints that make every IC differentiable end-to-end.
---

# JAX-native philosophy

progenax is **JAX-native by construction**. Every line of core code
operates on `jax.Array` objects, every loop is a `jax.lax` primitive,
every stateful object is an `equinox.Module`. This is not aesthetic
preference; it is the *minimum* discipline required to make every
IC differentiable, JIT-compilable, and `vmap`-able end to end.

This chapter documents the three constraints that follow from the
JAX-native commitment, the antipatterns they forbid, and why the
discipline pays for itself many times over in practice.

## The three constraints

```{list-table}
:header-rows: 1

* - Constraint
  - Implication
* - **No `numpy` or `scipy` in core code**
  - Use `jax.numpy` exclusively. Even read-only operations: `np.sqrt(x)` is a JAX-incompatible call when `x` is a traced JAX array under JIT
* - **No Python `for` / `while` loops in hot paths**
  - Use `jax.lax.scan` (fixed iteration count) or `jax.lax.fori_loop` (runtime-known fixed count). Python loops over traced arrays force unrolling, breaking JIT for large $N$
* - **No mutable state, no `array[i] = val`**
  - JAX arrays are immutable. Updates use `array.at[i].set(val)` (out-of-place) or `lax.scan` carry. Stateful classes are `equinox.Module` PyTrees
```

The constraints compose: code that respects all three is
JIT-compilable, `vmap`-able, and differentiable through every line —
the four operations (JIT, vmap, grad, scan) that make HMC inference
of cluster parameters tractable.

## Why no numpy/scipy

The most common transgression is mixing `numpy` with `jax.numpy`:

```python
# WRONG — np.sqrt forces a non-traced computation
import numpy as np
import jax.numpy as jnp

def buggy_radius(x, y, z):
    return np.sqrt(jnp.power(x, 2) + jnp.power(y, 2) + jnp.power(z, 2))

# RIGHT — jnp.sqrt stays in the JAX trace
def correct_radius(x, y, z):
    return jnp.sqrt(jnp.power(x, 2) + jnp.power(y, 2) + jnp.power(z, 2))
```

The `np.sqrt` version *works* outside JIT — it returns a numpy array
that JAX silently converts back. Inside JIT, the conversion happens
at trace time and produces a *concrete* numpy array, breaking the
trace. The error message is usually a cryptic
"`ConcretizationTypeError`" pointing at a line that looks innocent.

scipy presents the same problem with bigger consequences:
`scipy.integrate.quad` cannot be JIT-traced because its iteration
count depends on the value of the integrand. progenax's gravoturb
quadratures use fixed-grid trapezoidal or Gauss-Legendre instead —
both of which are pure-JAX operations.

The progenax convention: *zero* `import numpy as np` or
`import scipy` in `progenax/`. The diagnostics module
(`progenax.diagnostics`) is the deliberate exception — it is documented
as non-JIT, non-differentiable, and consumed only at validation time.

## Why no Python loops

Python `for` loops over JAX arrays *unroll* under JIT — each
iteration becomes a separate operation in the compiled XLA graph.
For $N = 10^4$ particles, this produces a 10,000-step XLA program
that takes minutes to compile and gigabytes of memory to hold. The
compiled program is also *not* `vmap`-able, since the unroll happened
at trace time and `vmap` cannot un-unroll it.

The replacement is `jax.lax.scan`:

```python
# WRONG — Python loop over particles
def buggy_kepler_solve(M_array, e):
    E_array = []
    for M in M_array:
        E = newton_solve_kepler(M, e)
        E_array.append(E)
    return jnp.array(E_array)

# RIGHT — vmap'd Newton solver
def correct_kepler_solve(M_array, e):
    return jax.vmap(lambda M: newton_solve_kepler(M, e))(M_array)
```

`jax.vmap` is the right tool for *parallel* loops (each particle
independently); `jax.lax.scan` is the right tool for *sequential*
loops (each step depends on the previous). progenax uses both.

```{warning}
**`jax.lax.while_loop` is forbidden in core code.** It is not
differentiable (the iteration count depends on data, so the gradient
is not defined). Convergence-based iteration must use `lax.scan`
with a *fixed* iteration count chosen large enough to ensure
convergence in the worst case. progenax's Newton solvers, ODE
integrations, and inverse-CDF lookups all use fixed-iteration
`scan`.
```

## Why immutable state

JAX arrays are immutable. The `array[i] = val` syntax does not work:
it raises `TypeError: '<class 'jax.numpy.ndarray'>' object does not
support item assignment`. The replacement uses functional updates:

```python
# WRONG — mutation
def buggy_zero_at(arr, i):
    arr[i] = 0  # TypeError under JAX
    return arr

# RIGHT — functional update
def correct_zero_at(arr, i):
    return arr.at[i].set(0.0)
```

For stateful model classes, progenax uses `equinox.Module`:

```python
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

class ExampleSolution(eqx.Module):
    y_grid: Float[Array, "N"]
    x_grid: Float[Array, "N"]

    def evaluate(self, x):
        return jnp.interp(x, self.x_grid, self.y_grid)
```

Equinox modules are PyTrees — JAX traces them automatically, so
methods like `evaluate` work under JIT, `vmap`, and `grad` without
any special handling. Updates produce *new* instances:

```python
sol_new = eqx.tree_at(lambda s: s.xi_t, sol_old, new_xi_t)
```

This out-of-place pattern is what makes everything differentiable —
gradients flow through the *new* PyTree, not through any in-place
modification.

## What about diagnostics?

The diagnostics layer (`progenax.diagnostics`) is *deliberately*
non-JAX-native. It uses `numpy` and `scipy` (e.g. `scipy.spatial.distance.pdist`,
`scipy.sparse.csgraph.minimum_spanning_tree`) because:

1. It runs *outside* the gradient-evaluation hot path.
2. The relevant algorithms (MST, Delaunay, convex hull) are not
   straightforward to port to JAX.
3. Diagnostic values are consumed at validation/plotting time, not
   inside HMC chains.

The boundary is enforced by the import structure: `progenax.profiles`,
`progenax.kinematics`, `progenax.imf`, `progenax.binaries`,
`progenax.gravoturb` may not import from `progenax.diagnostics`. The
reverse is allowed.

For diagnostic functions that *do* need to be JAX-native — e.g. the
JAX-native CW04 Q approximation in [](jax-native-substructure-q.md) —
they live alongside the scipy reference in `progenax.diagnostics.q_jax`,
clearly labelled.

## The dividend

The discipline pays for itself in three ways:

1. **HMC inference is tractable.** Posterior sampling of cluster
   parameters from observed data requires gradients of the entire
   IC pipeline. Every JAX-native function flows gradients
   automatically; without the discipline, finite-difference gradients
   would be the only option (slow, noisy, not HMC-compatible).
2. **GPU acceleration is free.** A `@jax.jit`-compiled function runs
   on GPU via `jax.devices('gpu')`. With the discipline, no code
   changes are needed; without it, mixed numpy/JAX code requires
   per-line auditing.
3. **Vectorisation is free.** Time-series of ICs, parameter sweeps,
   or batched inference all become single `vmap` calls. Without the
   discipline, Python loops produce per-batch retraces and 100×
   slowdowns.

A typical progenax research project that pays the upfront discipline
cost runs $\sim 100\times$ faster than its numpy-mixed equivalent at
$N = 10^4$ — the constant-factor speedup that makes posterior chains
of $\sim 10^6$ likelihood evaluations finish in hours rather than
weeks.

## References

The JAX programming model is documented at the [JAX
docs](https://jax.readthedocs.io). Equinox is {cite:t}`Equinox`. The
"no while_loop" convention is shared across the JAX scientific-computing
ecosystem (Diffrax, Optax, NumPyro all enforce it).
