---
title: Moe & Di Stefano (2017)
description: "Annotated reference for M. Moe et al. — Mind your Ps and Qs: The interrelation between period (P) and mass-ratio (Q) distributions of binary stars."
---

# Moe & Di Stefano (2017)

```{admonition} Mind your Ps and Qs: The interrelation between period (P) and mass-ratio (Q) distributions of binary stars
:class: note

**Authors.** M. Moe, R. Di Stefano

**Reference.** *The Astrophysical Journal Supplement Series* **230, 15** (2017).

**DOI.** [10.3847/1538-4365/aa6fb6](https://doi.org/10.3847/1538-4365/aa6fb6)

    **arXiv.** [1606.05347](https://arxiv.org/abs/1606.05347)
```

## Abstract (paraphrased)

Comprehensive census of binary star properties combining spectroscopic, eclipsing, interferometric, AO, and CPM surveys. Derives the intrinsic joint distribution $f(M_1, q, P, e)$. Two key findings: binary properties are mass-dependent; the joint distribution is not separable.

## Use in progenax

- [](../../10-theory/imfs/multiplicity-statistics.md) — Joint $f(M_1, q, P, e)$
- [](../../10-theory/imfs/mass-ratio-distributions.md) — $g(q \mid M_1)$
- [](../../10-theory/imfs/binary.md) — Underpins binary-aware IMF framework
- [](../../10-theory/binaries/period-distributions.md) — Joint $f(P, M_1)$
- [](../../10-theory/binaries/eccentricity.md) — Period-dependent $f(e)$
- `progenax.imf.MoeDiStefano2017` — mass-dependent mass-ratio implementation
- `progenax.imf.BinaryIMF` — binary-system sampling framework
- `progenax.binaries.MoeEccentricity` — period-dependent eccentricity sampler

## Notes

**The single most-referenced paper across progenax binary modelling.** Cited extensively throughout the binary chapters.
