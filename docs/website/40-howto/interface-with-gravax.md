---
title: Hand cataloged binary ICs to gravax
description: Build a cataloged primordial-binary cluster, inspect contact margins, and pass its physical state to gravax without confusing birth provenance with numerical ownership.
---

(howto-interface-gravax)=
# Hand cataloged binary ICs to `gravax`

**Goal.** Generate a mixed single/binary population with durable birth
provenance, inspect whether any sampled binary overlaps at periapsis, and hand
the physical state to [gravax](https://github.com/jaxstro/gravax).

## 1. Build the cataloged IC

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import (
    MoeCompanions,
    PlummerProfile,
    PlummerVelocityDF,
    PowerLawIMF,
    Systems,
    build_cataloged_binary_cluster,
)

ic = build_cataloged_binary_cluster(
    profile=PlummerProfile(r_h=1.0),
    velocity_df=PlummerVelocityDF(r_h=1.0),
    primary_imf=PowerLawIMF.kroupa(),
    companion_model=MoeCompanions(),
    target=Systems(1_000),
    key=jax.random.PRNGKey(7),
    units=STELLAR,
    Q=0.5,
    softening=0.0,
    compact=True,
)

catalog = ic.primordial_systems
binary_margin = catalog.periapsis_contact_margins[catalog.is_binary]
n_binary = int(jnp.sum(catalog.is_binary))
n_contacting = int(jnp.sum(binary_margin <= 0.0))
print(n_binary, n_contacting)
```

`softening=0.0` is used while virialising the system centers of mass and is the
collisional convention expected by the direct Gravax baseline. It is not stored
as an IC property.

## 2. Interpret the contact report

The catalog stores

$$
\Delta r_{\rm contact}=a(1-e)-(R_1+R_2)
$$

in the same length unit as `ic.positions`. A positive value is detached; a
negative value overlaps at Keplerian periapsis under progenax's current
main-sequence collision-radius approximation.

The builder does not silently repair a negative value. Decide explicitly
whether the scientific setup admits initial contact, applies a population-level
detached-only policy, or requires a collision/stellar-evolution model.

## 3. Hand over the physical state today

The released Gravax IC adapter is duck typed over particle arrays:

```python
from gravax import ParticleSystem

system = ParticleSystem.from_ic(
    ic,
    units=STELLAR,
    softening=0.0,
)
```

This currently transfers positions, velocities, masses, and radii. Keep `ic`
beside `system` as the immutable birth-provenance record.

```{warning}
At the time this page was written, released Gravax does **not yet** use
`PrimordialSystemCatalog` to choose the initial direct-hierarchy plan, and its
`ParticleSystem.from_ic` adapter does not yet preserve cataloged IDs. A normal
run therefore must not be described as catalog-certified hierarchy execution.
The catalog-aware initialization contract is under active implementation in
Gravax.
```

## 4. The catalog-aware hierarchy boundary

The intended Gravax boundary accepts this same `ic` object without a Progenax
import in Gravax core. At the initial synchronized boundary Gravax will:

1. preserve `ic.ids` as particle identity;
2. read binary candidate IDs from `ic.primordial_systems`;
3. check resolved contact and negative two-body energy;
4. apply its coupling, owner, schedule, handoff, phase, and hierarchy budgets;
5. resolve overlapping candidates collectively;
6. authenticate an exact field/compact pair partition; and
7. execute the accepted plan on the first physical map.

A primordial pair can become Kepler, SDAR, an algorithmic-regularization owner,
or remain in the field. The catalog never makes that numerical decision.

## 5. Use the masked form for IC differentiation

```python
masked = build_cataloged_binary_cluster(
    profile=PlummerProfile(r_h=1.0),
    velocity_df=PlummerVelocityDF(r_h=1.0),
    primary_imf=PowerLawIMF.kroupa(),
    companion_model=MoeCompanions(),
    target=Systems(1_000),
    key=jax.random.PRNGKey(7),
    units=STELLAR,
    compact=False,
)

real_positions = masked.positions[masked.is_real]
```

The masked product has exactly `2 * target.n` rows and is suitable for
fixed-shape JAX transformations. Its zero-mass ghost secondaries are generation
slots, not particles to send to Gravax. Use the compact product for evolution.

## What is and is not preserved

```{list-table}
:header-rows: 1

* - Quantity
  - Handoff meaning
* - Cartesian state
  - The physical state Gravax evolves.
* - Stable particle and system IDs
  - Immutable birth identity; compact IDs can contain gaps.
* - Sampled orbital elements
  - Generation provenance, not continuously updated osculating elements.
* - Contact margin
  - Initial physical diagnostic in position units.
* - Numerical owner
  - Not supplied by progenax; certified by Gravax.
* - Current binary membership
  - Must be measured from evolved phase space.
```

For the underlying ownership rationale, see
[](../20-architecture/primordial-system-handoff.md). For population construction
and its two-scale energy budget, see [](add-binary-population.md).
