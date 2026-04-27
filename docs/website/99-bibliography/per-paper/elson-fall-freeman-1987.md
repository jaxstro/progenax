---
title: Elson, Fall & Freeman (1987)
description: Annotated reference for R. A. W. Elson et al. — The structure of young star clusters in the Large Magellanic Cloud.
---

# Elson, Fall & Freeman (1987)

```{admonition} The structure of young star clusters in the Large Magellanic Cloud
:class: note

**Authors.** R. A. W. Elson, S. M. Fall, K. C. Freeman

**Reference.** *The Astrophysical Journal* **323, 54** (1987).

**DOI.** [10.1086/165807](https://doi.org/10.1086/165807)
```

## Abstract (paraphrased)

Empirical surface-brightness fits to 10 young massive star clusters in the LMC, finding power-law outer fall-off rather than exponential King-style cutoffs. Introduces the EFF profile $\rho(r) = \rho_0\,[1 + (r/a)^2]^{-\gamma/2}$ with free slope parameter $\gamma$. The profile reduces to Plummer at $\gamma = 5$ and is the standard description for unrelaxed young clusters.

## Use in progenax

- [](../../10-theory/spatial-profiles/eff.md) — Full derivation and progenax implementation
- `progenax.profiles.EFFProfile` — young-cluster spatial profile

## Notes

The right choice for young massive clusters where outer-slope is a free observational parameter. Unlike King, EFF has no built-in tidal truncation.
