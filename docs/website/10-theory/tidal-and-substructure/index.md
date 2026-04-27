---
title: Tidal physics & substructure
description: progenax's tidal physics and substructure section — Jacobi-radius computation, fractal IC generation via the FDF method, and Baumgardt energy-ranked mass segregation.
---

# Tidal physics & substructure

This section covers three orthogonal *modifier* layers that progenax
applies on top of the base spatial profile + velocity DF + IMF
combination ([](../ic-philosophy.md)):

```{list-table}
:header-rows: 1
:widths: 30 70

* - Modifier
  - What it adds to the IC
* - [](tidal.md)
  - Truncation at the Jacobi (tidal) radius set by the host galaxy's tidal field
* - [](fractal.md)
  - Fractal-style spatial substructure via the differentiable FDF method (replaces the non-differentiable {cite:t}`Goodwin2004` recursive tree)
* - [](mass-segregation.md)
  - Energy-ranked primordial mass segregation per {cite:t}`Baumgardt2008`, with smooth blending for HMC compatibility
```

The modifiers are conceptually composable, but the current high-level
cluster API keeps fractal substructure and mass segregation mutually
exclusive. Tidal truncation is a standalone array utility and can be
applied around either path.

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
from jaxstro.units import STELLAR
from progenax.cluster import FractalLayer, SpatialStructureParams, generate_cluster_ic
from progenax.imf import PowerLawIMF
from progenax.tidal import jacobi_radius, apply_tidal_truncation

cluster = generate_cluster_ic(
    key=jax.random.PRNGKey(42),
    N_stars=1000,
    M_total=1000.0,
    R_half=1.0,
    imf_params=PowerLawIMF.kroupa(),
    structure_params=SpatialStructureParams(
        base_profile="plummer",
        fractal=FractalLayer(D=2.0, lambda_frac=0.5, virial_ratio=0.3),
    ),
    G=STELLAR.G,
)

r_J = jacobi_radius(
    M_cluster=1e4,
    M_galaxy=1e10,
    R_galactic=8000.0,
)
positions, velocities, masses, keep_mask = apply_tidal_truncation(
    cluster.positions, cluster.velocities, cluster.masses, r_t=r_J,
)
```

For primordial mass segregation, replace the `FractalLayer` with
`MassSegregationLayer` in `SpatialStructureParams`. The high-level
generator will raise `ValueError` if both are supplied at once.

## References

Tidal physics: standard textbook material; the Jacobi-radius
approximation traces to Roche; cluster-specific applications follow
{cite:t}`Aarseth1974` and {cite:t}`Kuepper2011`. Fractal substructure:
{cite:t}`Goodwin2004` for the original recursive tree;
{cite:t}`Allison2009` for the dynamical-segregation consequence;
progenax's FDF method is original. Mass segregation:
{cite:t}`Baumgardt2008` for the energy-ordered construction;
{cite:t}`Kuepper2011` for the McLuster S-shuffle.
