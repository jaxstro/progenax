---
title: Mix Plummer positions with King velocities
description: "Recipe — use progenax's protocol-based composition to pair a Plummer spatial profile with a King velocity DF in one IC, and understand why an unmatched profile/DF pair needs an explicit virial rescale."
---

(howto-mix-plummer-king)=
# Mix Plummer positions with King velocities

**Goal.** Build an IC whose *positions* come from a Plummer profile but whose
*velocities* come from a King lowered-Maxwellian DF. progenax's
protocol-based design makes any `SpatialProfile` composable with any
`VelocityDF` — but mixing **unmatched** models breaks detailed equilibrium,
so this recipe also shows how to restore $Q = 0.5$.

## How composition works

Every spatial profile implements the `SpatialProfile` protocol
(`sample_positions`) and every velocity DF implements `VelocityDF`
(`sample_velocities`), so the two axes are independent:

```python
from progenax.protocols import SpatialProfile, VelocityDF
from progenax import PlummerProfile, KingVelocityDF

profile: SpatialProfile = PlummerProfile(r_h=1.0)
df: VelocityDF      = KingVelocityDF(W0=7.0, r_c=1.0)

isinstance(profile, SpatialProfile)   # True
isinstance(df, VelocityDF)            # True
```

```{important}
:label: imp-mix-not-equilibrium
**A mixed profile/DF pair is *not* a self-consistent equilibrium.** The King
DF is the equilibrium kinematics of the *King* density, not of a Plummer
density. Sampling Plummer positions with King velocities gives a system whose
raw virial ratio is **far from 0.5** (measured $Q \approx 0.19$ below). Pass
`Q=0.5` to `build_spatial_ic` to rescale the velocities back onto the
virial-equilibrium balance — or pass `Q=None` if you deliberately want the
unscaled, non-equilibrium kinematics. For a *self-consistent* cluster, use
the matched DF instead (`PlummerProfile` ↔ `PlummerVelocityDF`), which
`build_cluster` wires up automatically — see
[](set-up-virial-cluster.md).
```

## Recipe

```python
import jax, jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import (
    PlummerProfile, KingVelocityDF, build_spatial_ic,
    compute_kinetic_energy, compute_potential_energy,
)

G = STELLAR.G
key = jax.random.PRNGKey(0)
masses = jnp.ones(3000)

profile = PlummerProfile(r_h=1.0)          # spatial axis
df      = KingVelocityDF(W0=7.0, r_c=1.0)  # velocity axis (independent)

def virial_Q(ic):
    T = compute_kinetic_energy(ic.velocities, ic.masses)
    V = compute_potential_energy(ic.positions, ic.masses, G=G)
    return T / jnp.abs(V)

# Unscaled: see the raw mismatch (Q=None disables the virial rescale).
ic_raw = build_spatial_ic(profile, masses, df, key=key, G=G, Q=None)
print(f"unscaled mix:  Q = {virial_Q(ic_raw):.4f}")   # -> 0.1888 (NOT 0.5)

# Virialized: rescale velocities to the equilibrium balance Q = 0.5.
ic = build_spatial_ic(profile, masses, df, key=key, G=G, Q=0.5)
print(f"Q=0.5 mix:     Q = {virial_Q(ic):.4f}")        # -> 0.5000
```

## Verified output

Measured (`PRNGKey(0)`, $N=3000$):

```{list-table}
:header-rows: 1

* - Build
  - Measured $Q = T/|V|$
  - Interpretation
* - `Q=None` (unscaled)
  - 0.1888
  - Sub-virial — King kinematics are too cold for a Plummer potential.
* - `Q=0.5`
  - 0.5000
  - Velocities rescaled to virial equilibrium.
```

```{tip}
`KingVelocityDF` takes `W0` (concentration) and `r_c` (core radius) only — it
derives its tidal extent from the King ODE, so there is **no** `r_t` argument.
The velocity DF also needs `G` (passed through by `build_spatial_ic`) to set
its central velocity scale $\sigma^2 = GM/(9\,r_c\,\mu(W_0))$.
```

## See also

- [](../20-architecture/protocols.md) — the `SpatialProfile` / `VelocityDF` protocols.
- [](../10-theory/velocity-dfs/king-dfs.md) — the King lowered-Maxwellian DF.
- [](set-up-virial-cluster.md) — matched profile/DF builders (self-consistent equilibria).
