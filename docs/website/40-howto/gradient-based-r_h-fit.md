---
title: Fit the half-mass radius with jax.grad
description: "Recipe — recover a Plummer cluster's half-mass radius r_h by differentiating through the IC generator with jax.grad and running gradient descent on a structural observable."
---

(howto-rh-fit)=
# Fit the half-mass radius with `jax.grad`

**Goal.** Recover a cluster's half-mass radius $\rh$ by **differentiating
through the IC generator itself**. progenax samples every IC with
`jax.lax.scan` over a fixed iteration count (never `while_loop`), so
`build_spatial_ic` is differentiable end to end: `jax.grad` flows from an
observable all the way back to $\rh$.

## Inputs and assumptions

```{list-table} Recipe inputs
:header-rows: 1
:label: tbl-rhfit-inputs

* - Input
  - Meaning and role
  - Fiducial
* - $\rh^{\rm true}$
  - The half-mass radius of the synthetic "observed" cluster.
  - 1.3 pc (**recovered**)
* - observable
  - The scalar matched in the loss. We use the RMS radius, which is smooth and monotone in $\rh$.
  - RMS radius
* - `key`
  - **Frozen** PRNG key — held fixed across the optimization so the forward map is deterministic.
  - `PRNGKey(0)`
* - `G` (`STELLAR.G`)
  - Required gravitational constant; threaded through the velocity sampling and virial scaling.
  - `STELLAR`
```

```{important}
:label: imp-rhfit-frozen-key
**Freeze the seed across the optimization.** Differentiable inference here
treats the PRNG key as *fixed*: the same random draws are reused at every
$\rh$, so the forward map $\rh \mapsto \text{RMS radius}$ is a smooth,
deterministic function (here exactly linear, RMS $\approx 2.34\,\rh$ for this
seed). Re-keying every step would inject sampling noise into the gradient.
This isolates the *structural* gradient — the same pattern underlies the
full Fisher-information inference demos in
[](../60-science-demos/index.md).
```

## Recipe

```python
import jax, jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import PlummerProfile, PlummerVelocityDF, build_spatial_ic

G = STELLAR.G
key = jax.random.PRNGKey(0)
masses = jnp.ones(2000)

# 1. Synthetic "observed" cluster at the true half-mass radius.
r_h_true = 1.3
obs = build_spatial_ic(PlummerProfile(r_h=r_h_true), masses,
                       PlummerVelocityDF(r_h=r_h_true), key=key, G=G, Q=0.5)
target = jnp.sqrt(jnp.mean(jnp.sum(obs.positions**2, axis=1)))   # RMS radius

# 2. Forward map: rebuild the IC at a trial r_h and measure the same observable.
def predicted_rms(r_h):
    ic = build_spatial_ic(PlummerProfile(r_h=r_h), masses,
                          PlummerVelocityDF(r_h=r_h), key=key, G=G, Q=0.5)
    return jnp.sqrt(jnp.mean(jnp.sum(ic.positions**2, axis=1)))

def loss(r_h):
    return (predicted_rms(r_h) - target) ** 2

grad_loss = jax.grad(loss)              # d(loss)/d(r_h) THROUGH the IC builder
print(f"loss(1.0) = {loss(1.0):.4e}, grad = {grad_loss(1.0):+.4f}")

# 3. Gradient descent through the generator (frozen key -> deterministic).
r_h = 1.0
for step in range(80):
    r_h = r_h - 0.1 * grad_loss(r_h)
print(f"recovered r_h = {r_h:.4f}  (truth {r_h_true})")
```

## Verified output

Measured (`PRNGKey(0)`, $N=2000$, 80 GD steps, learning rate 0.1):

```
loss(1.0) = 4.9163e-01, grad = -3.2775
recovered r_h = 1.3000  (truth 1.3)
```

The gradient is non-zero and points toward the truth; gradient descent
converges to the exact $\rh^{\rm true}$.

```{warning}
**Step-size stability.** For this seed the forward map is linear,
RMS $\approx c\,\rh$ with $c \approx 2.34$, so the loss is quadratic and
gradient descent is stable only for $\eta < 1/c^2 \approx 0.18$. We use
$\eta = 0.1$. A larger step (e.g. 0.3) overshoots and diverges — for
production inference prefer a proper optimizer (`optax.adam`) or a
Gauss–Newton / Fisher step rather than hand-tuned fixed-step descent.
```

## See also

- [](../20-architecture/differentiability.md) — why `lax.scan` (not `while_loop`)
  makes the whole pipeline `jax.grad`-safe.
- [](set-up-virial-cluster.md) — the forward builder this recipe differentiates.
- [](../50-validation/differentiability-audit.md) — the AD-vs-FD gradient audit.
