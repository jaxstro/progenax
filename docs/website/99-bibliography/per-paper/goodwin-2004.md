---
title: Goodwin & Whitworth (2004)
description: Annotated reference for S. P. Goodwin et al. — The dynamical evolution of fractal star clusters.
---

# Goodwin & Whitworth (2004)

```{admonition} The dynamical evolution of fractal star clusters
:class: note

**Authors.** S. P. Goodwin, A. P. Whitworth

**Reference.** *Astronomy and Astrophysics* **413, 929** (2004).

**DOI.** [10.1051/0004-6361:20031529](https://doi.org/10.1051/0004-6361:20031529)
```

## Abstract (paraphrased)

Introduces the recursive-tree fractal IC generation algorithm, parameterised by fractal dimension $D \in [1.6, 3.0]$. Demonstrates that fractal substructure can survive $\sim 1$ crossing time of dynamical evolution.

## Use in progenax

- [](../../10-theory/tidal-and-substructure/fractal.md) — GW04 algorithm + the differentiable FDF replacement
- [](../../50-validation/fractal-substructure.md) — Calibration target for FDF $\chi \leftrightarrow D$

## Notes

The standard fractal IC. progenax keeps GW04 in `cluster.fractal_gw_legacy` and replaces it for production with the differentiable FDF method.
