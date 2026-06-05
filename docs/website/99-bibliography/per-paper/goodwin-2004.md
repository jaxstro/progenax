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

- [](../../10-theory/tidal-and-substructure/fractal.md) — GW04 algorithm + the fractal-dimension $D$ ↔ CW04 $Q$ relationship (theory)
- [](../../50-validation/fractal-substructure.md) — current substructure-validation status (CW04 $Q$ diagnostic)

## Notes

The standard fractal IC and the source of the fractal-dimension $D$ parameter.
progenax's differentiable Fractal Displacement Field replacement (`cluster.fdf`,
`cluster.fractal_gw_legacy`) was **removed in the 2026-06 clean-room rewrite** and
has no released successor; GW04 now appears only as theory pedagogy and as the
fractal-dimension ladder behind the CW04 $Q$ diagnostic (the surviving substructure
tool). Turbulent-density ICs are now the experimental `gravoturb_fdf` package
(a *density*-field method, distinct from the GW04 displacement fractal).
