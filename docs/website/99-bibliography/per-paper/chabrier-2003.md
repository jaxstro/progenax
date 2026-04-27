---
title: Chabrier (2003)
description: Annotated reference for G. Chabrier — Galactic stellar and substellar initial mass function.
---

# Chabrier (2003)

```{admonition} Galactic stellar and substellar initial mass function
:class: note

**Authors.** G. Chabrier

**Reference.** *Publications of the Astronomical Society of the Pacific* **115, 763** (2003).

**DOI.** [10.1086/376392](https://doi.org/10.1086/376392)
```

## Abstract (paraphrased)

Comprehensive synthesis of stellar IMF observations from substellar masses through the high-mass tail. The low-mass IMF is better described by a lognormal (peaked near $0.22\,\Msun$, width $\sigma_{\log m} \approx 0.57$) than by a broken power-law. The high-mass tail follows Salpeter; lognormal and power-law are joined continuously at $m = 1\,\Msun$.

## Use in progenax

- [](../../10-theory/imfs/classic.md) — Lognormal+Salpeter form
- `progenax.imf.Chabrier` — implementation with Newton-solver sampling

## Notes

Right for unresolved-population integrated colours. For HMC inference where analytic invertibility matters, prefer {cite:t}`Maschberger2013`.
