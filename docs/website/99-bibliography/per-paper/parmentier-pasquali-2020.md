---
title: Parmentier & Pasquali (2020)
description: "Annotated reference for G. Parmentier et al. — A new parameterization of the star formation rate–dense gas mass relation: Embracing gas density gradients."
---

# Parmentier & Pasquali (2020)

```{admonition} A new parameterization of the star formation rate–dense gas mass relation: Embracing gas density gradients
:class: note

**Authors.** G. Parmentier, A. Pasquali

**Reference.** *The Astrophysical Journal* **903, 56** (2020).

**DOI.** [10.3847/1538-4357/abb8d3](https://doi.org/10.3847/1538-4357/abb8d3)

    **arXiv.** [2009.10652](https://arxiv.org/abs/2009.10652)
```

## Abstract (paraphrased)

Introduces the magnification factor ζ that quantifies the geometric SFR boost a centrally-concentrated cloud gets over a uniform top-hat. Closed form for power-law profiles: $\zeta(p) = (3-p)^{3/2}/[2.6(2-p)]$. Diverges as $p \to 2$. The canonical $\zeta(1.67) \approx 1.79$ matches Kainulainen+14.

## Use in progenax

- [](../../10-theory/gravoturbulence/pp20.md) — Full ζ(p) derivation, equivalence proof, spot values
- [](../../10-theory/gravoturbulence/cored-profiles.md) — Generalisation to cored profiles
- `progenax.gravoturb.magnification_factor` — primary implementation

## Notes

**The reference for ζ(p).** The 2026-04-28 transcription bug fix at [](../../90-development-log/2026-04-28-pp20-fix.md) corrected an earlier mangled implementation.
