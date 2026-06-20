---
title: Theory & methods
description: Roadmap for the theory section — physical and mathematical foundations for every model in progenax.
---

# Theory & methods

This section is the scientific heart of progenax: every model the
package implements, derived from first principles, anchored on its
canonical references, and accompanied by the implementation choices
that made the model practical to differentiate through.

Every chapter follows the same template:

1. **Physical motivation** — what observational phenomenon the model
   captures and why it matters.
2. **Mathematical formulation** — closed-form derivations, ODEs, or
   sampling procedures, written so a non-specialist astronomer can
   follow the argument without needing to fall back on the source paper.
3. **Implementation in progenax** — the class signature, the JAX
   patterns used to keep it differentiable, and any historical bugs
   that future readers should know about.
4. **Domain of validity and caveats** — where the model fails,
   numerical-safety clips, and which alternative model to reach for
   when this one breaks.
5. **References** — `{cite}`-rendered links to the canonical paper(s),
   at least one pedagogical review, and any progenax-internal note
   the chapter draws on.

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

## What this section is *not*

Theory chapters describe *what the package computes and why it is correct*.
They deliberately do not describe *how to use the API* — that lives in
[](../30-api/index.md). They also do not describe *what the package
chose not to do*; that level of architectural rationale lives in
[](../20-architecture/index.md). When a chapter needs to point at an
implementation detail, it cross-links to the API page rather than
duplicating signatures.

The boundary between theory and architecture is sometimes fuzzy. As a
working rule: if the chapter content is invariant under a JAX-to-
PyTorch port, it belongs in theory. If it depends on JAX-specific
patterns (`lax.scan`, `stop_gradient`, smart dispatch), it belongs in
architecture.

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
