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

- [](../../10-theory/tidal-and-substructure/mass-segregation.md) — Energy-ranked algorithm + λ_seg blending
- `progenax.cluster.MassSegregationLayer` and `progenax.cluster.generate_cluster_ic` — high-level implementation
- `progenax.cluster.mass_segregation.energy_sorted_segregation` — lower-level energy-ordered assignment

## Notes

The canonical primordial segregation IC algorithm. progenax extends it with smooth λ_seg blending for HMC compatibility.
