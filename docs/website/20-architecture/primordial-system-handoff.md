---
title: Primordial-system provenance and downstream ownership
description: The cataloged binary-cluster boundary — immutable birth systems in progenax, numerical ownership in gravax, and dynamical identity measured from evolved state.
---

# Primordial-system provenance and downstream ownership

A binary cluster crosses three conceptually different boundaries. Keeping them
separate prevents a birth label from being mistaken for either a physical claim
about an evolved system or an instruction to a numerical integrator.

```{list-table} Three identities
:header-rows: 1

* - Layer
  - Owner
  - Meaning
* - Birth provenance
  - progenax
  - Which sampled particles began in one primordial single or binary system, with which orbital elements.
* - Numerical ownership
  - gravax
  - Which field or compact method owns each Newtonian interaction during one accepted integration epoch.
* - Dynamical identity
  - Measured from evolved phase space
  - Which particles are bound after disruption, capture, exchange, or higher-order encounters.
```

The first layer is immutable provenance. The other two can change. In
particular, a primordial binary is a **candidate** for a compact integration
owner; it is not automatically a Kepler, SDAR, or algorithmic-regularization
assignment.

## The opt-in cataloged API

The established `build_binary_cluster(...)` API remains available with its
existing `ICResult` / `ResolvedBinaries` returns. New workflows that need a
durable system-level handoff use `build_cataloged_binary_cluster(...)`, which
returns a `CatalogedBinaryClusterIC` with the ordinary particle arrays plus a
`PrimordialSystemCatalog`.

```python
ic.positions
ic.velocities
ic.masses
ic.stellar_radii
ic.ids
ic.is_real
ic.primordial_system_id
ic.is_primordial_secondary
ic.primordial_systems
```

The particle-local arrays remain convenient for grouping rows. The catalog is
the canonical record of each sampled system:

```python
catalog.system_ids
catalog.is_binary
catalog.component_particle_ids
catalog.component_active
catalog.semimajor_axes
catalog.eccentricities
catalog.inclinations
catalog.longitudes_ascending_node
catalog.arguments_periapsis
catalog.mean_anomalies
catalog.periapsis_contact_margins
```

All orbital angles are in radians. Semimajor axes and contact margins use the
same length unit as `positions`. Stellar radii on the particle product retain
the established `ICResult` convention of solar radii.

## Stable logical birth IDs

Masked generation uses two interleaved logical slots per sampled system:
primary `2s` and secondary `2s+1`. A single star has an inactive secondary
component with catalog ID `-1`; its masked ghost slot is not a real particle.

Compaction removes ghosts but does not renumber real birth IDs. A compact
cataloged result can therefore have gaps such as `[0, 2, 3, 4]`. Downstream
code must use `ids` and `component_particle_ids` as identities, not array-row
numbers.

The legacy builder intentionally keeps its current contiguous row IDs. That
separation is the compatibility boundary for a future deliberate hard cutover.

## Initial contact margin

For each sampled binary, progenax reports

$$
\Delta r_{\rm contact} = a(1-e) - (R_1 + R_2).
$$

- $\Delta r_{\rm contact}>0$: detached at Keplerian periapsis under the adopted stellar-radius model.
- $\Delta r_{\rm contact}=0$: tangent at periapsis.
- $\Delta r_{\rm contact}<0$: overlapping at periapsis.

Singles store a finite zero and are excluded with `catalog.is_binary`.
Progenax reports negative margins without clipping or resampling: contact is a
property of the sampled IC. Whether a downstream physical model stops,
merges, resets, or rejects that system belongs to that model.

## Compact and masked products

`compact=True` contains real particles only and is the ordinary handoff to an
N-body code. Its `is_real` array is all true. `compact=False` retains `2S`
interleaved slots and the real-particle mask; it is the fixed-shape route for
JAX transformations and requires `Systems(n)`.

Both products retain the same sampled orbital and contact provenance. Their
real-particle Cartesian states agree exactly at a fixed random key.

## The gravax boundary

progenax deliberately has no dependency on gravax. The catalog contains no
integrator name, hierarchy slot, routing margin, or ownership decision.
gravax consumes the physical state and, once its catalog adapter is released,
will independently check identity, contact, binding, perturbation budgets, and
pair-partition completeness before the first hierarchy map.

See [](../40-howto/interface-with-gravax.md) for the current physical-state
handoff and the catalog-aware hierarchy boundary.
