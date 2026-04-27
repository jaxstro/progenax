---
title: Binary populations
description: progenax's binary-population machinery — Kepler orbital elements, the four period-distribution families, and the three eccentricity distributions.
---

# Binary populations

A **binary population** is the kinematic complement to the binary IMF
discussed at [](../imfs/binary.md). The IMF chapter covers the
*statistical* framework — what fraction of stars are in binaries, what
mass ratios they have, and how that biases inferred IMF slopes. This
chapter family covers the **orbital mechanics**: given the binary
fraction and mass ratios, how to assign Keplerian orbits to each
binary, sample the orbital phase, and produce the resolved component
positions and velocities that an N-body integrator consumes.

## Map of the section

```{list-table}
:header-rows: 1
:widths: 28 72

* - Chapter
  - Scope
* - [](kepler-elements.md)
  - The seven orbital elements $(a, e, i, \Omega, \omega, M, t_p)$, their physical meaning, and the conversion from elements to phase-space coordinates.
* - [](period-distributions.md)
  - Four period-distribution families: log-uniform (Öpik), log-normal ({cite:t}`Sana2012` for OB-type, Duquennoy-Mayor for solar), {cite:t}`Sana2012` and {cite:t}`Moe2017` empirical fits.
* - [](eccentricity.md)
  - Three eccentricity-distribution families: thermal ($f(e) = 2e$), uniform, period-dependent {cite:t}`Moe2017`. Tidal-circularisation effects at short period.
```

## What a "binary" is in progenax

A binary in progenax is two particles linked by a `KeplerElements`
PyTree. The two particles are stored in the same `(masses, positions,
velocities)` arrays as single stars — they are not "second-class"
entries. The binary status is recorded in a separate `binary_id`
array indexing which pairs of particles form binaries:

```python
masses     # shape (N_total,)   — every star's mass, including secondaries
positions  # shape (N_total, 3) — 3D positions in cluster frame
velocities # shape (N_total, 3) — 3D velocities in cluster frame
binary_id  # shape (N_total,)   — 0 for singles, integer >= 1 for paired binaries
```

The advantage of this representation is that downstream consumers
(integrators, renderers, observation operators) treat singles and
binary components uniformly. The disadvantage is that the joint
distribution of binary parameters has to be reconstructed from the
component data when needed — for which progenax provides
`progenax.binaries.elements_from_state(positions, velocities, masses, binary_id)`.

## Composability

Binary populations layer on top of any IMF + spatial profile + velocity
DF combination:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Component
  - Role for binaries
* - [](../imfs/binary.md)
  - Decides *which* primaries get companions and *what* the secondary masses are
* - [](../spatial-profiles/index.md)
  - Places primaries in 3D — secondaries are placed at the same position then offset by the orbital separation
* - [](../velocity-dfs/index.md)
  - Sets the primary's bulk velocity — secondaries inherit it plus the orbital velocity
* - [Kepler elements](kepler-elements.md)
  - Specify the relative orbit (semi-major axis, eccentricity, orientation)
* - [Period distributions](period-distributions.md)
  - Set the per-binary semi-major axis from the population-wide $f(P)$
* - [Eccentricity distributions](eccentricity.md)
  - Set the per-binary $e$ from the population-wide $f(e)$
```

The composition is order-independent: the result is the same whether
you place primaries first and decorate with binaries, or sample
binary parameters first and place the resolved system. progenax's
default builder follows the latter ordering for vectorisation efficiency.

## Resolved vs unresolved

A "resolved" binary in progenax is one where both components are
present in the output `(masses, positions, velocities)` arrays. An
"unresolved" treatment merges the two components into a single
phase-space entry at the binary's centre of mass. Both are supported:

```python
from progenax.binaries import build_binaries

# Resolved (default): both components present
masses, positions, velocities, binary_id = build_binaries(
    primary_masses, primary_positions, primary_velocities,
    binary_mask, q_samples, kepler_elements,
    resolved=True,
)

# Unresolved: combined into binary CoM
masses_u, positions_u, velocities_u = build_binaries(
    ..., resolved=False,
)
```

For N-body integration, *resolved* is the right choice — the
integrator needs to evolve the orbital motion. For mock observations
of unresolved photometric surveys, *unresolved* matches the
data-generating process. Both forms are differentiable in the
underlying parameters.

## Connection to the binary IMF

The binary fraction $f_b(m_1)$ from [](../imfs/binary.md) decides the
*number* of binaries; the chapters in this section decide their
*orbital properties*. Both come from the same {cite:t}`Moe2017`
calibration when consistency matters — period distributions from
[](period-distributions.md), mass ratios from
[](../imfs/mass-ratio-distributions.md), eccentricities from
[](eccentricity.md) — and they share the period-conditional
non-separability noted at [](../imfs/multiplicity-statistics.md).

## References

Kepler-element machinery is standard textbook material. The period
distributions follow {cite:t}`Sana2012` (OB-type), Duquennoy-Mayor
(solar), and {cite:t}`Moe2017` (joint $f(P)$). The eccentricity
distributions follow {cite:t}`Moe2017` and earlier compilations.
