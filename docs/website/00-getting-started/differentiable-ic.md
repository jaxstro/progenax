---
title: Taking gradients through an IC
description: Hands-on tutorial — take `jax.grad` through the IC builder, then run an HMC chain to recover input parameters.
---

# Taking gradients through an IC

```{admonition} Executable notebook coming later
:class: tip

The hands-on **executable** version of this tutorial — `differentiable-ic.ipynb`
with pre-executed JAX cells, ready to launch in Binder or Colab — is
a Phase E deliverable still to come. Until then, the prose walkthrough
below is sufficient to follow along: the core progenax code blocks run
as-is when copied into a Python session with progenax installed (see
[](installation.md)); the NumPyro block additionally requires NumPyro.
```

The headline feature of progenax: every IC parameter is **differentiable**
through `jax.grad`. This page demonstrates that with a concrete
example, then shows how the same machinery powers HMC posterior
inference of cluster parameters.

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

The above is too well-behaved to be illustrative. Let's optimise
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
    r_h = r_h - 0.01 * grad
    if step % 10 == 0:
        print(f"Step {step}: r_h = {r_h:.4f}, loss = {loss(r_h, target_KE):.4e}")
```

The cluster's KE depends on $r_h$ via the Plummer DF: $\sigma^2 \propto
1/r_h$, $T \propto N\,\sigma^2 / 2$. Gradient descent will find the
$r_h$ that produces the target KE.

## Why this is hard without progenax

Most N-body codes are *not* end-to-end differentiable. They use:

- Python loops over particles → unrollable, but slow
- Random sampling with rejection → variable per-particle cost
- Mutable state → no JAX trace
- scipy operations → not in JAX

To compute $\partial \mathrm{KE} / \partial r_h$ in such a code,
you'd use **finite differences**: compute KE at $r_h$ and $r_h +
\epsilon$, take the difference, divide by $\epsilon$. This works
but is slow ($\mathcal{O}(N_{\mathrm{params}})$ extra builds), noisy
(Monte-Carlo noise dominates for small $\epsilon$), and HMC-incompatible.

progenax's analytic gradients are exact, fast, and HMC-friendly —
the foundation for cluster-parameter inference.

## HMC inference: the bigger picture

The same machinery powers full Bayesian inference. Given observed
cluster data — say, an observed mass distribution — you can infer
the IMF slope $\alpha$ via:

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(observed_masses):
    alpha = numpyro.sample("alpha", dist.Uniform(0.5, 4.0))
    log_lik = Maschberger(alpha=alpha).logpdf(observed_masses).sum()
    numpyro.factor("ll", log_lik)

# Run NUTS on synthetic data
true_alpha = 2.35
synth_masses = Maschberger(alpha=true_alpha).sample(jax.random.PRNGKey(0), 1000)

mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(1), observed_masses=synth_masses)
mcmc.print_summary()
```

NUTS uses gradients — provided automatically by JAX through the
progenax IMF — to take large effective steps in parameter space.
The chain converges in ~500 warmup steps and produces a posterior
on $\alpha$ centred near the true value.

For the full binary-aware version of this — where the inferred
$\alpha$ accounts for unresolved binaries — see
[](../10-theory/imfs/binary.md).

## Next step

[](imf-sampling.md) walks through the four canonical IMF
parameterisations.
