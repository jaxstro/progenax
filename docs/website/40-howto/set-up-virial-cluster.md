---
title: Set up a virial-equilibrium cluster
description: "Recipe — build a virial-equilibrium (Q = T/|V| = 0.5) Plummer cluster IC and verify the virial ratio, at three levels of API (convenience builder, build_cluster, build_spatial_ic)."
---

(howto-virial-cluster)=
# Set up a virial-equilibrium cluster

**Goal.** Generate a star-cluster initial condition that sits in virial
equilibrium — $Q \equiv T/|V| = 0.5$, the virial-theorem balance
$2T + V = 0$ — so it neither collapses ($Q<0.5$) nor expands ($Q>0.5$)
when handed to an integrator.

## Inputs and assumptions

```{list-table} Recipe inputs
:header-rows: 1
:label: tbl-virial-inputs

* - Input
  - Meaning and role
  - Fiducial
* - profile / `r_h`
  - Spatial density model (Plummer here) and its half-mass radius.
  - Plummer, $\rh = 1$ pc
* - `n` (or `masses`)
  - Particle count; `n` gives equal $1\,\Msun$ masses, or pass explicit `masses`.
  - 2000 (equal mass)
* - `units` → `G`
  - Unit system carrying the gravitational constant. **Required** (no default `G`).
  - `STELLAR`
* - `Q`
  - Target virial ratio $T/|V|$. `0.5` rescales velocities to equilibrium.
  - 0.5
```

```{important}
:label: imp-virial-G
**`G` is mandatory.** Every velocity sample and energy evaluation needs an
explicit gravitational constant — pass `units=STELLAR` (or `G=STELLAR.G`).
There is no implicit fallback. The virial scaling at `Q=0.5` rescales
velocities under the **exact** ($\varepsilon=0$) Newtonian force law by
default, matching the analytic equilibrium DFs and collisional integration.
```

## Recipe A — convenience builder (shortest path)

```python
import jax, jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import (
    build_plummer_cluster,
    compute_kinetic_energy, compute_potential_energy,
)

key = jax.random.PRNGKey(0)
ic = build_plummer_cluster(key=key, n=2000, r_h=1.0, units=STELLAR)

T = compute_kinetic_energy(ic.velocities, ic.masses)
V = compute_potential_energy(ic.positions, ic.masses, G=STELLAR.G)
print(f"virial Q = {T / jnp.abs(V):.4f}")   # -> virial Q = 0.5000
```

`build_plummer_cluster` defaults to `Q=0.5`, so the returned `ICResult` is
already virialized and centred on the cluster's centre of mass.

## Recipe B — explicit profile via `build_cluster`

When you want to choose the profile object yourself (and let progenax pick
the *matched* equilibrium velocity DF), use `build_cluster`:

```python
from progenax import PlummerProfile, build_cluster

ic = build_cluster(PlummerProfile(r_h=1.0), key=key, n=2000,
                   units=STELLAR, Q=0.5)
# virial Q = 0.5000
```

`matched_velocity_df` reads the profile's own scale fields, so the velocity
DF can never desync from the profile's $\rh$ — the classic footgun is removed.

## Recipe C — full control with `build_spatial_ic`

The lowest-level builder takes an explicit profile **and** velocity DF — use
it when you are composing a profile and DF by hand (see
[](mix-plummer-positions-king-velocities.md)):

```python
from progenax import PlummerProfile, PlummerVelocityDF, build_spatial_ic

masses = jnp.ones(2000)
ic = build_spatial_ic(
    PlummerProfile(r_h=1.0), masses, PlummerVelocityDF(r_h=1.0),
    key=key, G=STELLAR.G, Q=0.5,
)
# virial Q = 0.5000
```

## Verified output

All three recipes produce the same equilibrium (measured, `PRNGKey(0)`,
$N=2000$):

```{list-table}
:header-rows: 1

* - Recipe
  - Measured $Q = T/|V|$
* - A `build_plummer_cluster`
  - 0.5000
* - B `build_cluster`
  - 0.5000
* - C `build_spatial_ic`
  - 0.5000
```

```{tip}
Pass `Q=None` to **disable** the rescale. For the true-equilibrium DFs (King,
EFF, Michie, LIMEPY) this yields the faithful *unscaled* equilibrium, whose
measured $Q$ already lands near 0.5 — see
[](../50-validation/king-profile.md). The isotropic Plummer DF likewise
samples its own equilibrium; the explicit `Q=0.5` rescale only removes the
finite-$N$ scatter.
```

## See also

- [](gradient-based-r_h-fit.md) — differentiate through this builder to fit $\rh$.
- [](../20-architecture/q-virial-convention.md) — the $Q = T/|V|$ convention.
- [](../50-validation/plummer-equilibrium.md) — the equilibrium validation suite.
