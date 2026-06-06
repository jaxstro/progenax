---
title: Szapudi, Pan, Prunet & Budavári (2005)
description: Annotated reference for Szapudi, Pan, Prunet & Budavári — fast FFT-based edge-corrected estimators for the two-point correlation function and the power spectrum, and the ξ↔P duality used by the FDF measurement code.
---

# Szapudi, Pan, Prunet & Budavári (2005)

```{admonition} Fast edge-corrected measurement of the two-point correlation function and the power spectrum
:class: note

**Authors.** István Szapudi, Jun Pan, Simon Prunet, Tamás Budavári

**Reference.** *The Astrophysical Journal Letters* **631, L1–L4** (2005).

**DOI.** [10.1086/496971](https://doi.org/10.1086/496971)

**Verified.** Abstract and §1 checked against the held PDF (2026-06). The relevant content: an
$N\log N$ FFT estimator for $\xi(r)$ and $P(k)$ via the **Wiener–Khinchin theorem**, and the
$\xi \leftrightarrow P$ **Hankel-transform duality** — the measurement machinery of the FDF oracle.
```

## The big idea

The two-point correlation function $\xi(r)$ and the power spectrum $P(k)$ are a Fourier-transform
pair (Wiener–Khinchin), yet in practice they are measured by different, complementary methods — pair
counting on small scales, FFT band-powers on large scales — with different edge-correction problems.
Szapudi, Pan, Prunet & Budavári present a unified, fast pair: an FFT algorithm that produces an
edge-corrected $\xi(r)$ matching a direct pair-count, plus a numerical Hankel transform (with a
Gauss–Bessel quadrature) that turns the measured $\xi(r)$ into an edge-corrected $P(k)$, both at
$N\log N$ cost. The key practical point is that $\xi(r)$ and $P(k)$ are **equivalent** — one can use
whichever is convenient and convert between them robustly.

## Use in progenax

- **The measurement oracle.** progenax's
  [`measure.py`](../../../../src/experimental/gravoturb_fdf/validation/measure.py) uses exactly this
  Wiener–Khinchin route: `autocovariance_3d` computes $\xi(r)$ as the inverse FFT of the squared
  field transform, and the inference covariance layer measures **power-spectrum band-powers** from the
  same field. These are the ground-truth measurements against which the analytic
  [`gaussianized_xi`](../../../../src/experimental/gravoturb_fdf/theory/gaussianization.py) prediction
  and the Limber-projected two-point are validated.
- **The $\xi \leftrightarrow P$ duality** is what lets the [](../../10-theory/gravoturbulence/differentiable-inference.md) 2-point
  block be expressed either as the configuration-space $\xi_s(r)$ (Gaussianization series) or as
  Fourier band-powers (the likelihood data vector), interchangeably.

## Notes

- Same lead author and CIC lineage as [](szapudi-pan-2004.md); here the focus is the estimator rather
  than the density model.
- Edge correction matters for real, bounded survey footprints; the FDF works on periodic simulation
  boxes, where the estimator is exact and the edge term is absent — but the same duality and FFT
  machinery apply.
