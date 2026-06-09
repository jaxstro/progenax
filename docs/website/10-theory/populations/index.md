---
title: Multi-component populations
description: progenax's multi-component cluster machinery — two-component clusters, mixed dynamical states, and the per-component composability with profiles, DFs, and modifiers.
---

# Multi-component populations

```{seealso}
For single-component clusters built from one IMF + one profile + one
DF, see the chapter families [](../spatial-profiles/index.md),
[](../velocity-dfs/index.md), and [](../imfs/index.md). This chapter
covers the case where a single IC contains *multiple* such
components — the simplest non-trivial example being a two-component
cluster with a young dense core embedded in an older diffuse envelope.
```

A **multi-component population** is an IC that contains stars drawn
from more than one IMF + profile + DF combination. Real clusters
often need multi-component descriptions:

- **Two-population globular clusters.** Many old globular clusters
  show two stellar populations of different ages and chemical
  compositions; these have distinct radial distributions and need
  distinct sampling.
- **Embedded cluster + halo.** Young embedded clusters often have a
  dense compact core embedded in a diffuse halo of expelled stars.
- **Multi-mass equipartition.** Old clusters reach approximate
  equipartition of kinetic energy across mass bins; the resulting
  per-mass-bin radial distributions differ.

progenax's `TwoComponentConfig` and `generate_two_component_cluster`
provide the current two-component primitive: one already-sampled mass
array, two profiles, two velocity DFs, and a returned population-ID
mask. Richer N-component builders and per-component IMF sampling are
composition patterns, not a separate public API in this checkout.

## Map of the section

```{list-table}
:header-rows: 1

* - Chapter
  - Scope
* - [](two-component.md)
  - Two-component clusters with separate masses, radii, and dynamical states. The simplest multi-component case and the one progenax exposes as a primitive.
```

## What "multi-component" means in progenax

The implemented two-component primitive treats the masses as an input
array, splits those systems into populations A and B, then samples
positions and velocities from the corresponding profile/DF pair. A
more general multi-component recipe can give each component its own
IMF, spatial profile, velocity DF, and virial state, but that recipe is
currently user-side composition rather than a packaged N-component
builder.

```{warning}
**Per-component virial state is not preserved by joint rescaling.**
If component A is at $Q_A = 0.5$ and component B at $Q_B = 0.7$, joint
rescaling to $Q_{\mathrm{global}} = 0.5$ moves both away from their
input values. For multi-component ICs where you need both components
in equilibrium, use `Q_target=None` to skip the joint rescale and
rely on per-component DFs to set the local equilibrium. See
[](../../20-architecture/q-virial-convention.md).
```

## Composability

Each component composes orthogonally with all the modifier layers:

- Tidal truncation ([](../tidal-and-substructure/tidal.md)) applies
  to the union (or per-component if needed).
- Fractal substructure ([](../tidal-and-substructure/fractal.md))
  can be applied per-component (typical for the dense core) or to
  the union.
- Mass segregation ([](../tidal-and-substructure/mass-segregation.md))
  is per-component, since the energy ranking is component-relative.

The general N-component recipe is:

```python
components = []
for cfg in component_configs:
    masses_i = cfg.imf.sample(cfg.N, key)
    positions_i = cfg.profile.sample_positions(masses_i, key)
    velocities_i = cfg.df.sample_velocities(positions_i, masses_i, key, G=G)
    components.append((masses_i, positions_i, velocities_i))

# Concatenate and shift to global COM
masses = jnp.concatenate([c[0] for c in components])
positions = jnp.concatenate([c[1] for c in components])
velocities = jnp.concatenate([c[2] for c in components])
positions, velocities, masses = to_com_frame(positions, velocities, masses)
```

For the implemented two-component case, progenax exposes the simpler
single API call documented at [](two-component.md).

## References

Multi-component cluster modelling is standard in N-body work; the
{cite:t}`Aarseth1974` numerical scheme and {cite:t}`Kuepper2011`
McLuster code both support multi-population ICs. The lowered-model
family formalized by {cite:t}`Gieles2015` is the natural framework when
*self-consistent* multi-mass equilibrium is required (rather than
the layered single-mass-DF-per-component approach progenax takes);
progenax plans to implement this family natively as its own
differentiable generalization (see
[](../spatial-profiles/lowered-model-family.md)).
