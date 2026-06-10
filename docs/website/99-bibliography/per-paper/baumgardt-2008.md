---
title: Baumgardt, De Marchi & Kroupa (2008)
description: Annotated reference for H. Baumgardt et al. — Evidence for primordial mass segregation in globular clusters.
---

# Baumgardt, De Marchi & Kroupa (2008)

```{admonition} Evidence for primordial mass segregation in globular clusters
:class: note

**Authors.** H. Baumgardt, G. De Marchi, P. Kroupa

**Reference.** *The Astrophysical Journal* **685, 247** (2008).

**DOI.** [10.1086/590488](https://doi.org/10.1086/590488)
```

## Abstract (paraphrased)

Demonstrates that primordial mass segregation can explain both the IMF-slope-vs-concentration trend and the depletion of low-mass stars in globular clusters. progenax uses this paper as the physical motivation for primordial segregation; the implemented energy-ordered IC machinery is exposed through the `progenax.cluster` layer.

## Use in progenax

- [](../../10-theory/tidal-and-substructure/mass-segregation.md) — energy-ranked algorithm, two released routes
- `progenax.energy_sorted_segregation` — the PRIMORDIAL energy-ordered generator (public API)
- `MultiComponentCluster.from_mass_segregation(delta)` — the separate *differentiable* equilibrium route (not from this paper; equipartition law $w_j = \mu_j^{-\delta}$ after Gieles & Zocchi 2015)

## Notes

The canonical primordial segregation IC algorithm: most massive stars onto
the most bound orbits of an equilibrium pool, with bin widths proportional
to cumulative mass.

**Deliberate, documented departure from the published recipe.**
Baumgardt+2008 draw orbits *randomly within* cumulative-mass bins; progenax's
`energy_sorted_segregation` instead uses a **deterministic isotonic rounding**
of the bin-centre targets to a strictly increasing orbit-index sequence. The
random per-bin draw fails for steep IMFs — cumulative-mass bins collapse
below one orbit, forcing many low-mass ranks onto the same orbit (coincident
stars, $V = -\infty$). The deterministic assignment guarantees a distinct
orbit per star for any mass spectrum; realisation variety comes from
re-drawing the random orbit pool. See the module docstring of
`src/progenax/cluster/mass_segregation.py`.

The historical λ_seg catalog blend that once smoothed this construction was
retired in the 2026-06 unified redesign (its intermediate states drift from
per-mass-group virial balance); differentiable segregation control now comes
from the equilibrium `MultiComponentCluster.from_mass_segregation` route.
