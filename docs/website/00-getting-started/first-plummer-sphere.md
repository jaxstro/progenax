---
title: Your first Plummer sphere
description: Hands-on tutorial — generate a 1000-particle Plummer cluster from scratch and measure its energy ratio.
---

# Your first Plummer sphere

```{admonition} Executable notebook coming later
:class: tip

The hands-on **executable** version of this tutorial — `first-plummer-sphere.ipynb`
with pre-executed JAX cells, ready to launch in Binder or Colab — is
a Phase E deliverable still to come. Until then, the prose walkthrough
below is sufficient to follow along: every code block runs as-is when
copied into a Python session with progenax installed (see
[](installation.md)).
```

Five-minute walkthrough: generate a 1000-particle Plummer cluster IC
from scratch and measure its energy ratio. By the end
of this page you'll know what every step does and why it's needed.

## Setup

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF
from progenax.imf import Maschberger
from progenax.builders import (
    virial_scale, to_com_frame,
    compute_kinetic_energy, compute_potential_energy,
)

key = jax.random.PRNGKey(42)
G = STELLAR.G       # ≈ 0.00450 pc³ M☉⁻¹ Myr⁻²
N = 1000
```

## Step 1: sample masses from the IMF

```python
imf = Maschberger(alpha=2.3)        # Default Salpeter-like high-mass slope
key_mass, key = jax.random.split(key)
masses = imf.sample(key_mass, N)
print(f"Mean mass: {masses.mean():.2f} M☉, total: {masses.sum():.0f} M☉")
```

The printed mean and total mass describe this finite IMF realization.

## Step 2: sample positions from the spatial profile

```python
profile = PlummerProfile(r_h=1.0)   # Half-mass radius = 1 pc
key1, key = jax.random.split(key)
positions = profile.sample_positions(masses, key1)
radii = jnp.linalg.norm(positions, axis=1)
order = jnp.argsort(radii)
enclosed_mass = jnp.cumsum(masses[order])
half_index = jnp.searchsorted(enclosed_mass, 0.5 * enclosed_mass[-1])
sample_r_h = radii[order[half_index]]
print(f"Position shape: {positions.shape}, sampled half-mass radius: {sample_r_h:.3f} pc")
```

This estimates the realization's half-mass radius about the sampling origin using
the first particle radius that encloses half the total mass. The median radius
instead estimates the half-number radius. After changing the origin to the center
of mass, recompute the diagnostic if that is the desired convention.

## Step 3: sample velocities from the matched DF

```python
df = PlummerVelocityDF(r_h=1.0)
key2, key = jax.random.split(key)
velocities = df.sample_velocities(positions, masses, key2, G=G)
print(f"Velocity shape: {velocities.shape}, mean speed: {jnp.linalg.norm(velocities, axis=1).mean():.3f} pc/Myr")
```

The output has shape `(1000, 3)`; its speed scale depends on the realized total mass.

## Step 4: center the realization and optionally rescale velocities

```python
positions, velocities = to_com_frame(positions, velocities, masses)
velocities = virial_scale(
    positions, velocities, masses, Q_target=0.5, G=G,
)

T = compute_kinetic_energy(velocities, masses)
V = compute_potential_energy(positions, masses, G=G)
Q = T / jnp.abs(V)
print(f"Virial Q = {Q:.4f} (expected 0.5)")
```

The optional rescaling sets the measured energy ratio to the requested value,
within numerical precision. It changes the raw sampled velocities and does not
establish a stationary equilibrium.

## What just happened

```{list-table}
:header-rows: 1

* - Step
  - What it did
* - 1
  - Drew $N$ stellar masses from the {cite:t}`Maschberger2013` IMF, with low-mass turnover and Salpeter high-mass slope
* - 2
  - Drew $N$ positions from the Plummer density profile via inverse-CDF sampling
* - 3
  - Drew $N$ velocities from the matched isotropic Plummer DF — the equilibrium kinematics
* - 4
  - Shifted to the centre-of-mass frame, then optionally rescaled velocities to the requested energy ratio
```

## Visualising the result

```python
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Spatial distribution (xy projection)
ax1.scatter(positions[:, 0], positions[:, 1], s=masses * 5, alpha=0.3)
ax1.set_xlabel("x [pc]"); ax1.set_ylabel("y [pc]")
ax1.set_title("Plummer cluster, xy projection")
ax1.set_aspect('equal')

# Mass distribution
ax2.hist(jnp.log10(masses), bins=30)
ax2.set_xlabel("log₁₀(M / M☉)"); ax2.set_ylabel("N")
ax2.set_title("IMF samples")
plt.tight_layout()
plt.savefig("first_plummer.png", dpi=120)
```

## What you can do next

- **Take a gradient** through the IC builder — see
  [](differentiable-ic.md).
- **Try other profiles** — King ([](../10-theory/spatial-profiles/king.md))
  for tidally-truncated old globulars, EFF ([](../10-theory/spatial-profiles/eff.md))
  for young massive clusters.
- **Add binaries** — [](../40-howto/add-binary-population.md).
- **Hand off to gravax** for evolution — [](../40-howto/interface-with-gravax.md).
