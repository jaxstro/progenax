---
title: Cartwright & Whitworth (2004)
description: Annotated reference for A. Cartwright et al. — The statistical analysis of star clusters.
---

# Cartwright & Whitworth (2004)

```{admonition} The statistical analysis of star clusters
:class: note

**Authors.** A. Cartwright, A. P. Whitworth

**Reference.** *Monthly Notices of the Royal Astronomical Society* **348, 589** (2004).

**DOI.** [10.1111/j.1365-2966.2004.07360.x](https://doi.org/10.1111/j.1365-2966.2004.07360.x)
```

## Abstract (paraphrased)

Introduces the dimensionless Q parameter for quantifying the spatial structure of stellar clusters. Q combines the normalised mean MST edge length and the normalised mean inter-star separation. Distinguishes substructured ($Q < 0.79$) from centrally-concentrated ($Q > 0.79$) distributions.

## Use in progenax

- [](../../20-architecture/jax-native-substructure-q.md) — JAX-native kNN approximation + scipy reference
- [](../../10-theory/tidal-and-substructure/fractal.md) — Q as the calibration target for FDF $\chi$

## Notes

The CW04 Q is **distinct** from the virial Q = T/|V|. progenax separates these into different modules.
