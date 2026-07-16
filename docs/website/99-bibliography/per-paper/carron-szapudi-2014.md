---
title: Carron & Szapudi (2014)
description: Annotated reference for J. Carron & I. Szapudi — sufficient observables for discrete (Poisson-sampled) galaxy fields, the counts-in-cells analogue of the optimal-transform theory.
---

# Carron & Szapudi (2014)

```{admonition} Sufficient observables for large-scale structure in galaxy surveys
:class: note

**Authors.** J. Carron, I. Szapudi

**Reference.** *Monthly Notices of the Royal Astronomical Society Letters* **439, L11–L15** (2014).

**DOI.** [10.1093/mnrasl/slt167](https://doi.org/10.1093/mnrasl/slt167)

**Verified.** Abstract and §1 checked against the held PDF (2026-06). The contribution progenax uses:
the optimal-transform / sufficient-statistics theory extended to **discrete, Poisson-sampled
lognormal fields** — the counts-in-cells regime.
```

## The big idea

Real surveys deliver **points, not a continuous field**, so the elegant continuous-field results of
[](carron-szapudi-2013.md) need a discreteness-aware version. Carron & Szapudi solve the one-point
case exactly and sketch the multipoint case for the **Poisson sampling of an underlying lognormal
density field** — the same generative model as the FDF (lognormal-type field → local Poisson star
counts). They show the corresponding optimal non-linear transformation is directly related to the
**maximum-a-posteriori Bayesian reconstruction of the underlying continuous field with a lognormal
prior**. The practical payoff is that "simple recipes for realizing the sufficient observables can be
built on previously proposed algorithms" — the log/Gaussianizing transforms already implemented and
tested in simulations.

## Use in progenax

- **Counts-in-cells sufficiency.** This is the discrete analogue that licenses progenax's CIC block:
  the stellar counts are a local-Poisson sampling of a lognormal-type field (the
  [](szapudi-pan-2004.md) CIC relation), and for such a field the sufficient observables are the
  ones the [](../../10-theory/gravoturbulence/inference.md) likelihood already uses (the count distribution and its
  two-point clustering), not high-order count moments.
- **Connects the count likelihood to a principled estimator** rather than an ad hoc summary — the
  [`count_loglike`](../../../../src/experimental/gravoturb/inference/likelihood.py) block sits in
  this lineage.

## Notes

- Together with [](carron-szapudi-2013.md) (continuous), [](neyrinck-2009.md) and
  [](neyrinck-2011.md) (empirical, matter and galaxies), this completes the argument that
  log-density 1pt + 2pt statistics are sufficient and information-optimal for a lognormal-type field —
  continuous or Poisson-sampled.
- The lognormal prior here is the pure-lognormal case; BM19 adds the self-gravity power-law tail,
  which progenax handles with the peaks-over-threshold tail block rather than a moment expansion.
