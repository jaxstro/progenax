---
title: JAX-native philosophy
description: JAX array, loop, and state conventions, with the separate checks needed for transformation support and derivative accuracy.
---

# JAX-native philosophy

Progenax uses JAX arrays and transformations on its numerical core paths.
Equinox modules hold model state. These conventions make transformation support
possible; they do not establish differentiability or derivative accuracy for
every parameter, boundary, discrete choice, or observable.

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

Transformation support also depends on shapes, static arguments, control flow,
and derivative rules. Verify `jit`, the required AD mode, and batching for the
intended public path. See [](differentiability.md).

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
at trace time and attempts to convert a tracer to a NumPy array. It can raise
`TracerArrayConversionError`; the traced value is not available to NumPy.

scipy presents the same problem with bigger consequences:
`scipy.integrate.quad` cannot be JIT-traced because its iteration
count depends on the value of the integrand. The gravoturbulence
quadratures (now in the experimental `gravoturb` package) use
fixed-grid trapezoidal or Gauss-Legendre instead — both of which are
pure-JAX operations.

The progenax convention: *zero* `import numpy as np` or
`import scipy` in `progenax/`. The diagnostics module
(`progenax.diagnostics`) is the deliberate exception — it is documented
as non-JIT, non-differentiable, and consumed only at validation time.

## Why no Python loops

Python `for` loops over JAX arrays *unroll* under JIT — each
iteration becomes a separate operation in the compiled XLA graph.
A long unrolled loop can increase graph size and compilation cost. Static
Python loops can still support `vmap` and AD; unrolling alone does not prohibit
those transformations. Measure trace, compile, and warm execution separately
before replacing a loop.

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
`jax.lax.while_loop` supports forward AD but ordinary reverse AD requires a
static memory bound. A converged mathematical solution may still have an
implicit derivative. Progenax uses fixed-iteration kernels on some paths,
Diffrax adjoints for adaptive ODE solves, and an implicit derivative around
Engine A's adaptive central-fraction fit. Check convergence and the derivative
of the intended solution. See the
[JAX loop contract](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html).
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

Functional updates preserve JAX tracing. The updated expression still needs a
valid derivative rule and a suitable parameter domain.

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
`progenax.kinematics`, `progenax.imf`, and `progenax.binaries` may not
import from `progenax.diagnostics`. The reverse is allowed.

For diagnostic functions that *do* need to be JAX-native — e.g. the
JAX-native CW04 Q approximation in [](jax-native-substructure-q.md) —
they live alongside the scipy reference in `progenax.diagnostics.q_approx`,
clearly labelled.

## The dividend

The discipline pays for itself in three ways:

1. **HMC inference is tractable.** Posterior sampling of cluster
   parameters from observed data requires gradients of the entire
   IC pipeline. JAX can evaluate derivatives on supported paths; a physical
   likelihood and independent gradient checks remain necessary.
2. **Device execution can be selected explicitly.** Backend availability,
   precision, supported operations, and measured performance determine whether
   an accelerator is useful. Device enumeration does not place a computation.
3. **Batching can expose independent work.** `vmap` often replaces a Python
   batch loop, but static shapes and memory cost still matter. Benchmark the
   intended batch size and retain a bounded-memory route when needed.

Measured timings and their hardware, precision, and problem sizes belong in
[performance and memory](../50-validation/performance-memory.md). These
implementation conventions alone imply no speedup factor.

## References

The JAX programming model is documented at the [JAX
docs](https://jax.readthedocs.io). Equinox is {cite:t}`Equinox`. Adaptive loops and their derivative rules must be assessed separately; the
[JAX control-flow guide](https://docs.jax.dev/en/latest/control-flow.html)
describes the transformation contracts.
