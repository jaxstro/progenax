---
title: Taking gradients through an IC
description: Take a fixed-key Plummer derivative through the IC builder, and identify the separate requirements for posterior inference.
---

# Taking gradients through an IC

```{admonition} Running this tutorial
:class: note

Use the [installed float64 environment](installation.md) for the Python examples.
Keep the random key fixed when comparing a derivative or optimizing a realization.
Posterior sampling has a separate demonstration linked below, with its own
assumptions and convergence requirements.
```

This tutorial differentiates a Plummer realization with a fixed random key.
Its half-mass-radius derivative is a continuous scaling derivative. Other
parameters, discrete assignments, and population expectations have separate
[contracts](../20-architecture/differentiability.md).

## A first gradient

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF
from progenax.imf import Maschberger
from progenax.builders import (
    virial_scale, compute_kinetic_energy, compute_potential_energy,
)

def loss_fn(r_h):
    """Loss = (Q_vir - 0.5)^2 — should always be near zero."""
    key = jax.random.PRNGKey(0)
    key_mass, key_pos, key_vel = jax.random.split(key, 3)
    masses = Maschberger(alpha=2.3).sample(key_mass, 1000)
    profile = PlummerProfile(r_h=r_h)
    df = PlummerVelocityDF(r_h=r_h)
    positions = profile.sample_positions(masses, key_pos)
    velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
    velocities = virial_scale(
        positions, velocities, masses, Q_target=0.5, G=STELLAR.G,
    )

    T = compute_kinetic_energy(velocities, masses)
    V = compute_potential_energy(positions, masses, G=STELLAR.G)
    Q = T / jnp.abs(V)
    return (Q - 0.5) ** 2

grad_fn = jax.grad(loss_fn)
r_h_test = jnp.float64(1.0)
print(f"Loss at r_h=1: {loss_fn(r_h_test):.6e}")
print(f"Gradient: {grad_fn(r_h_test):.6e}")
```

Expected: tiny loss ($< 10^{-5}$, since virial scaling enforces
$Q = 0.5$ exactly), tiny gradient (the loss surface is essentially
flat near the optimum).

## A more interesting gradient

The above is too well-behaved to be illustrative. Let's optimize
$r_h$ to match a *target* total kinetic energy:

```python
def total_KE(r_h, target_KE):
    key = jax.random.PRNGKey(0)
    key_mass, key_pos, key_vel = jax.random.split(key, 3)
    masses = Maschberger(alpha=2.3).sample(key_mass, 1000)
    profile = PlummerProfile(r_h=r_h)
    df = PlummerVelocityDF(r_h=r_h)
    positions = profile.sample_positions(masses, key_pos)
    velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
    KE = compute_kinetic_energy(velocities, masses)
    return (KE - target_KE) ** 2

target_KE = jnp.float64(50.0)   # Pick something
loss = total_KE
grad_loss = jax.grad(loss)

# Gradient descent, 50 steps
r_h = jnp.float64(2.0)   # Start at 2 pc
for step in range(50):
    grad = grad_loss(r_h, target_KE)
    r_h = r_h - 5e-4 * grad
    if step % 10 == 0:
        print(f"Step {step}: r_h = {r_h:.4f}, loss = {loss(r_h, target_KE):.4e}")
```

The cluster's KE depends on $r_h$ via the Plummer DF: $\sigma^2 \propto
1/r_h$, so $T \propto N\,\sigma^2 / 2 \propto 1/r_h$. Gradient descent
converges on the $r_h$ that produces the target KE — here $r_h \approx
2.16$ pc, with the loss driven to $\sim 10^{-27}$ within ~20 steps:

```
Step 0: r_h = 2.1091, loss = 1.5433e+00
Step 10: r_h = 2.1614, loss = 2.3783e-07
Step 20: r_h = 2.1615, loss = 5.2865e-14
Step 30: r_h = 2.1615, loss = 1.1754e-20
Step 40: r_h = 2.1615, loss = 3.2312e-27
```

(The step size matters: because $\partial\mathrm{KE}/\partial r_h$ is
large here, a learning rate of $0.01$ overshoots and diverges — $5\times
10^{-4}$ converges cleanly.)

## Why this is hard without progenax

Most N-body codes are *not* end-to-end differentiable. They use:

- Python loops over particles → unrollable, but slow
- Random sampling with rejection → variable per-particle cost
- Mutable state → no JAX trace
- scipy operations → not in JAX

Finite differences provide an independent check of
$\partial \mathrm{KE}/\partial r_h$. Hold the random key fixed and reduce
the step until truncation error gives way to roundoff or solver error. AD
differentiates the implemented numerical map; agreement at one step or a finite
gradient alone does not establish the derivative of the intended physical model.

## HMC inference: the bigger picture

A usable likelihood, a prior, and a checked gradient are needed before an IC
generator can enter a posterior calculation. A differentiable generator alone
does not define a likelihood for observed stars.

The existing [IMF and equipartition demo](../60-science-demos/imf-equipartition.md)
owns the BlackJAX example, its opt-in posterior route, and its stated limits.
It separates deterministic engineering checks from posterior recovery. Any new
recovery claim requires convergence diagnostics, including R-hat and effective
sample size, and representative synthetic populations. No posterior result is
claimed by the fixed-key gradient examples on this page.

## Next step

[](imf-sampling.md) walks through the four canonical IMF
parameterizations.
