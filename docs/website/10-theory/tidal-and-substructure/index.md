---
title: Tidal physics & substructure
description: progenax's tidal physics and substructure section — Jacobi-radius computation, fractal-substructure theory + the CW04 Q diagnostic (the FDF generator moved to the experimental gravoturb package), and Baumgardt energy-ranked mass segregation.
---

# Tidal physics & substructure

This section covers three orthogonal *modifier* layers that progenax
applies on top of the base spatial profile + velocity DF + IMF
combination ([](../ic-philosophy.md)):

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning the modifier layers progenax applies on top of a base equilibrium IC — tidal truncation, substructure, and mass segregation; no prior cluster-dynamics literature assumed.
**Prerequisites:** [spatial profiles](../spatial-profiles/index.md) and the [IC philosophy](../ic-philosophy.md) (these modifiers perturb a base profile + velocity DF + IMF).
**You'll get:** what the three modifier layers add to an IC, why each maps to an observed cluster deviation, and which are released vs. experimental.
:::

```{list-table}
:header-rows: 1

* - Modifier
  - What it adds to the IC
* - [](tidal.md)
  - Truncation at the Jacobi (tidal) radius set by the host galaxy's tidal field
* - [](fractal.md)
  - Theory of fractal/clumpy substructure ({cite:t}`Goodwin2004`) and the CW04 $Q$ diagnostic. The differentiable *generator* moved to the experimental, repo-only `gravoturb` package in the 2026-06 clean-room rewrite; released progenax keeps the diagnostics (`progenax.diagnostics`)
* - [](mass-segregation.md)
  - Mass segregation two ways: the energy-ranked PRIMORDIAL generator (`energy_sorted_segregation`, after {cite:t}`Baumgardt2008`) and the differentiable EQUILIBRIUM route (`MultiComponentCluster.from_mass_segregation`)
```

Tidal truncation is a standalone array utility and can be applied
around either path. Turbulent/fractal substructure ICs live outside
the released package (experimental `gravoturb`), so within released
progenax the segregation routes and tidal truncation are the two
modifiers you can actually apply.

## Why these three?

The three correspond to the three observed deviations of real young /
old clusters from smooth-equilibrium-sphere idealisations:

1. **Truncation.** No real cluster extends to infinite radius. Old
   globular clusters are tidally limited; young clusters are limited
   by the embedding gas cloud's mass. The tidal modifier captures
   this.
2. **Substructure.** Young clusters inherit clumpy substructure from
   their parent molecular clouds {cite:p}`Goodwin2004,Allison2009`.
   The fractal modifier captures this.
3. **Mass segregation.** Old clusters show massive stars preferentially
   in their cores. Whether this is primordial {cite:p}`Baumgardt2008`
   or dynamical {cite:p}`Allison2009` is a debated observational
   question. The mass-segregation modifier seeds the primordial case.

All three are *post-equilibrium* in the sense that they perturb a
base equilibrium IC. Whether the resulting non-equilibrium configuration
is the right starting state for a given science target depends on
the science: cool fractal ICs are appropriate for studying violent
relaxation; smooth virial-equilibrium-plus-segregation is appropriate
for studying long-timescale relaxation.

## Current API sketch

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import MultiComponentCluster
from progenax.tidal import jacobi_radius, apply_tidal_truncation

# Equilibrium mass segregation: two components in ONE shared potential,
# with the equipartition law w_j = mu_j^(-delta) (differentiable in delta)
cluster = MultiComponentCluster.from_mass_segregation(
    alpha_j=jnp.array([0.5, 0.5]),   # central density fractions
    m_j=jnp.array([0.3, 1.0]),       # representative stellar masses [Msun]
    W0=7.0, g=1.0, delta=0.5,
)
ic = cluster.sample_cluster(jax.random.PRNGKey(42), n_stars=1000, G=STELLAR.G)
# ic.component_id labels each star's generating component

r_J = jacobi_radius(
    M_cluster=1e4,
    M_galaxy=1e10,
    R_galactic=8000.0,
)
positions, velocities, masses, keep_mask = apply_tidal_truncation(
    ic.positions, ic.velocities, ic.masses, r_t=r_J,
)
```

For *primordial* (non-equilibrium) segregation, use
`progenax.energy_sorted_segregation` to energy-rank an orbit pool drawn
from any equilibrium profile — see [](mass-segregation.md). The legacy
string-dispatch generator (`generate_cluster_ic` with
`SpatialStructureParams` layers) was retired in the 2026-06 unified
redesign.

## Implementation, validation & references

- **In code:** tidal truncation is `src/progenax/tidal.py`; the
  equilibrium and primordial segregation routes are
  `src/progenax/cluster/multicomponent.py` and
  `src/progenax/cluster/mass_segregation.py`; the substructure
  diagnostics are `src/progenax/diagnostics/`. See the
  [tidal API](../../30-api/tidal.md), the [cluster API](../../30-api/cluster.md),
  and the [diagnostics API](../../30-api/diagnostics.md); each chapter
  below carries its exact module path. (The fractal *generator* moved to
  the experimental, repo-only `gravoturb` package; only the CW04 $Q$
  diagnostic remains in released progenax.)
- **Validated in:** [tidal truncation](../../50-validation/tidal-truncation.md),
  [fractal substructure](../../50-validation/fractal-substructure.md),
  and [mass segregation](../../50-validation/mass-segregation.md).
- **Primary sources:** tidal physics is standard textbook material (the
  Jacobi-radius approximation traces to Roche), with cluster-specific
  applications following {cite:t}`Aarseth1974` and {cite:t}`Kuepper2011`;
  fractal substructure {cite:t}`Goodwin2004` (recursive tree) and
  {cite:t}`Allison2009`; mass segregation {cite:t}`Baumgardt2008`
  (energy-ordered construction) and {cite:t}`Kuepper2011` (McLuster
  S-shuffle). Full notes in the
  [bibliography](../../99-bibliography/index.md); each chapter below
  points at the specific result(s) used.
