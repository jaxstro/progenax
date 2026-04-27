---
title: Plummer (1911)
description: Annotated reference for H. C. Plummer — On the problem of distribution in globular star clusters.
---

# Plummer (1911)

```{admonition} On the problem of distribution in globular star clusters
:class: note

**Authors.** H. C. Plummer

**Reference.** *Monthly Notices of the Royal Astronomical Society* **71, 460** (1911).

**DOI.** [10.1093/mnras/71.5.460](https://doi.org/10.1093/mnras/71.5.460)
```

## Abstract (paraphrased)

The original derivation of the Plummer model — a smooth, finite, self-consistent gravitational equilibrium first proposed to fit the observed star counts of globular clusters. Combines the Lane-Emden formalism with the empirical observation that cluster densities are finite at the centre and fall off smoothly at large radius, producing the now-canonical $\rho(r) = (3M/4\pi a^3)\,[1 + (r/a)^2]^{-5/2}$ with closed-form mass, potential, and (via Eddington inversion) DF.

## Use in progenax

- [](../../10-theory/spatial-profiles/plummer.md) — Full derivation and progenax implementation
- [](../../10-theory/velocity-dfs/plummer-dfs.md) — Eddington-inversion DF $f(E) \propto (-E)^{7/2}$
- `progenax.profiles.PlummerProfile` — production-default spatial profile
- `progenax.kinematics.PlummerVelocityDF` — matched equilibrium velocity DF

## Notes

The Plummer model is the simplest self-consistent equilibrium with a closed-form DF. It serves as both a research tool (production ICs) and a pedagogical example (canonical illustration of Eddington inversion).
