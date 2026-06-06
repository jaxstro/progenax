---
title: Carron, Wolk & Szapudi (2014)
description: Annotated reference for J. Carron, M. Wolk & I. Szapudi — fast exact FFT generation of cosmological random fields with a prescribed two-point function, the realization-and-covariance lineage of the FDF.
---

# Carron, Wolk & Szapudi (2014)

```{admonition} On fast generation of cosmological random fields
:class: note

**Authors.** J. Carron, M. Wolk, I. Szapudi

**Reference.** *Monthly Notices of the Royal Astronomical Society* **444, 994–1000** (2014).

**DOI.** [10.1093/mnras/stu1527](https://doi.org/10.1093/mnras/stu1527)

**Verified.** Abstract and §1 checked against the held PDF (2026-06). The relevant content: **fast,
exact FFT generation of a random field with a prescribed two-point statistic**, and consistent
counts-in-cells **covariances** — the realization and mock-covariance lineage of the FDF.
```

## The big idea

To do inference you need not only a model but **mocks and a covariance**. Carron, Wolk & Szapudi give
a fast, exact method to simulate a wide class of random fields with a prescribed two-point function on
a periodic grid of twice the survey side using FFTs. They show that a finite survey boundary breaks
the field's translation invariance and *correlates* the observable Fourier modes — the "supersurvey"
or "beat-coupling" modes that dominate the power-spectrum covariance on large scales — and that these
are accounted for exactly by the non-linear transformations of the Gaussian field. Applied to a real
galaxy field (CFHTLS) modelled as Poisson sampling of a lognormal field, the method reproduces power
spectra, counts-in-cells distributions, and covariances consistent with the data, and recovers the
information plateau seen in SDSS and $N$-body simulations.

## Use in progenax

- **The GRF realization step.** progenax builds its field with the same idea — draw Gaussian Fourier
  amplitudes scaled by $\sqrt{P(k)} = k^{-\beta/2}$ and inverse-FFT
  ([`gaussian_random_field`](../../../../src/experimental/gravoturb_fdf/field/field.py)) — then apply
  the monotone copula map. This is the FFT-based, prescribed-spectrum construction of this paper.
- **Mock covariances.** The realization simulator supplies the mock covariance used (Hartlap-corrected)
  in the Fisher forecast and the Gaussian-likelihood block; the supersurvey/beat-coupling caveat is
  the reason the analytic Gaussian band-power covariance *underestimates* the true mock covariance for
  the non-Gaussian log-density field — a finding the [](../../10-theory/gravoturbulence/differentiable-inference.md) layer guards
  against by using the mock covariance.

## Notes

- Part of the Szapudi-group lineage with [](coles-jones-1991.md), [](szapudi-pan-2004.md),
  [](carron-szapudi-2013.md), [](carron-szapudi-2014.md); the realization here is the practical
  companion to that theory.
- progenax differs in target marginal: the FDF imposes the BM19 lognormal + power-law marginal via a
  rank copula, not a pure lognormal.
