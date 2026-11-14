---
title: Theory & methods
description: Roadmap for the theory section — physical and mathematical foundations for every model in progenax.
---

# Theory & methods

These chapters explain the physical models, their assumptions, and the numerical
choices used in Progenax. Read each model alongside its implementation and validation
references. Theory develops the physical relations; method notes explain how they
are evaluated and sampled. The API reference owns signatures, and architecture
pages explain software organization and JAX integration.

## Suggested reading order

The [map below](#map-of-the-section) is the *index*; this is the *path*.
If you are learning the science from scratch, read the chapters in this
order — each step builds on the conventions and models of the ones
before it:

1. [](ic-philosophy.md) — **start here.** Defines what an "initial
   condition" *is* in progenax (the three ingredients —
   density, velocity DF, masses), the virial-$Q$ convention, units, and
   the COM frame. Everything downstream assumes these conventions.
2. **Spatial profiles**, in increasing generality:
   [](spatial-profiles/plummer.md) (the closed-form baseline) →
   [](spatial-profiles/king.md) (tidal truncation via an ODE) →
   [](spatial-profiles/eff.md) (a free outer slope) →
   [](spatial-profiles/lowered-model-family.md) (the unified
   Woolley/King/Wilson family) →
   [](spatial-profiles/multimass-equipartition.md) (multi-mass
   equilibrium). Plummer first because its analytic relations make the sampling steps explicit; the family and multimass pages last
   because they generalize King.
3. [](velocity-dfs/index.md) — **how each density is given velocities.**
   Equilibrium requires a compatible density, potential, and DF; read this
   directly after the spatial profiles.
4. [](imfs/index.md) — **how stars get masses.** Mass sampling can be independent of positions in an unsegregated model;
   mass segregation and multi-mass equilibria introduce correlations.
5. [](binaries/index.md) — **multiplicity.** Builds on the IMF: orbital
   elements, period/eccentricity distributions, and resolving binaries
   into component kinematics.
6. [](tidal-and-substructure/index.md) — **modifiers and diagnostics.**
   Tidal truncation, fractal/substructured ICs, and mass-segregation
   layers that act *on top of* a base profile, plus the diagnostics that
   measure them.
7. [](populations/index.md) — **composite clusters.** Composite populations require an explicit treatment of their shared
   gravitational potential and component distributions; assembling components
   does not by itself establish equilibrium.

For developers extending the package, pair each theory chapter with the
[architecture section](../20-architecture/index.md) — see [Reading
paths](#reading-paths) below.

(map-of-the-section)=

## Map of the section

```{list-table}
:header-rows: 1

* - Chapter family
  - Scientific scope
* - [](ic-philosophy.md)
  - What an "initial condition" *is* in progenax. Conventions, virial theorem, units, COM frame, half-mass radius. Read first if you are new.
* - [](spatial-profiles/index.md)
  - The spatial density profiles: [](spatial-profiles/plummer.md), [](spatial-profiles/king.md), and [](spatial-profiles/eff.md).
* - [](velocity-dfs/index.md)
  - Velocity distribution functions paired with each spatial profile to produce dynamical equilibrium ICs, including the anisotropic [](velocity-dfs/michie-king.md), and anisotropy and rotation extensions.
* - [](imfs/index.md)
  - The initial mass function — from canonical Salpeter / Kroupa / Chabrier / Maschberger through binary-aware {cite:t}`MoeDiStefano2017` and environment-dependent {cite:t}`Marks2012,Jerabkova2018` variants.
* - [](binaries/index.md)
  - Kepler orbital elements, period and eccentricity distributions ({cite:t}`Sana2012,MoeDiStefano2017`), resolved binary kinematics.
* - [](tidal-and-substructure/index.md)
  - Tidal physics (Jacobi radius, differentiable truncation), the theory of fractal / substructured initial conditions, and {cite:t}`Baumgardt2008` energy-ranked mass segregation.
* - [](gravoturbulence/index.md)
  - From the {cite:t}`FederrathKlessen2012` density PDF through the {cite:t}`ParmentierPasquali2020` magnification factor and the {cite:t}`Burkhart2018,BurkhartMocz2019` dense-gas SFR framework.
* - [](populations/index.md)
  - Multi-component clusters and other composite distributions.
```

## Theory, methods, and implementation

Theory pages derive the physical relations and state their assumptions. Brief
method notes connect those relations to numerical sampling, including approximations
and endpoint treatment. Function signatures belong in the [API reference](../30-api/index.md);
software ownership and JAX execution patterns belong in [architecture](../20-architecture/index.md).
The [validation pages](../50-validation/index.md) report the checks actually performed.

(reading-paths)=

## Reading paths

For students: read [](ic-philosophy.md) first, then any one chapter
family that matches what you want to learn. Each family is
self-contained.

For developers extending progenax: read the architecture section
([](../20-architecture/index.md)) alongside the relevant theory chapter
— theory tells you what physics to preserve, architecture tells you the
patterns that preserve it under JAX.

For paper-writing: every theory chapter ends with a `References` block
containing `{cite}` keys you can copy directly into your manuscript.
The full bibliography (with DOIs / arXiv links) lives at
[](../99-bibliography/bibliography.md).
