---
title: Carron & Szapudi (2013)
description: Annotated reference for J. Carron & I. Szapudi — the information-optimal non-linear transform is the log/power transform tied to the spectrum slope, and why moments fail to capture information in the high-variance (fat-tail) regime.
---

# Carron & Szapudi (2013)

```{admonition} Optimal non-linear transformations for large-scale structure statistics
:class: note

**Authors.** J. Carron, I. Szapudi

**Reference.** *Monthly Notices of the Royal Astronomical Society* **434, 2961–2970** (2013).

**DOI.** [10.1093/mnras/stt1215](https://doi.org/10.1093/mnras/stt1215)

**Verified.** Abstract and §1 checked against the held PDF (2026-06). The result progenax cites: the
**log transform is the information-optimal Gaussianizer**, and **moments lose information in the
high-variance regime** — the formal backing for the log-density choice and for avoiding the
tail-divergent linear statistics.
```

## The big idea

Neyrinck's result — that Gaussianizing restores power-spectrum information — raises a sharper
question: *which* non-linear transform is best, and how much information can any local transform
recover? Carron & Szapudi answer it with Fisher information and cosmological perturbation theory.
They show that at each perturbative order there is a **polynomial that exhausts the information** on a
given parameter; this polynomial is the Taylor expansion of the maximally efficient "sufficient"
observable. The corresponding optimal local transform "is essentially the simple power transform with
an exponent related to the slope of the power spectrum; when this is $-1$, it is indistinguishable
from the logarithmic transform." The transform **Gaussianizes the distribution and recovers the
linear density contrast** — a direct equivalence between *undoing the non-linear dynamics* and
*efficiently capturing Fisher information*. Their transforms stay close to optimal even deep into the
non-linear regime, $\sigma^2 \sim 10$.

## Why this matters for the fat tail

The companion observation (building on Carron 2011) is the one that bites in the gravoturbulent
problem: **in the large-variance regime a large fraction of the information escapes the entire
hierarchy of $N$-point moments.** When the density PDF has a heavy power-law tail — BM19 with
$\alpha \le 2$, where $\langle\rho^2\rangle$ formally diverges — the moments are dominated by the
rarest cells and carry almost no information about the parameters. The fix is not "more moments" but a
*non-linear transform that Gaussianizes the distribution*, after which the (transformed) two-point
function is information-rich. This is precisely why progenax carries the **log-density** two-point
$\xi_s$ and the peaks-over-threshold tail estimator rather than linear-density moments.

## Use in progenax

- **Formal justification for the log-density variable** $s = \ln(\rho/\rho_0)$ used throughout
  [](../../10-theory/gravoturbulence/inference.md): the log transform is (near-)optimal for a $k^{-\beta}$ spectrum,
  Gaussianizing the field and concentrating the information in $\xi_s$.
- **Explains why the linear-density 2-point is avoided** — its moments are information-poor and
  divergent for the collapsing slopes $\alpha \le 2$.

## Notes

- The discrete (Poisson-sampled) extension of these "sufficient observables" is
  [](carron-szapudi-2014.md); the empirical demonstration on the Millennium simulation is in
  [](neyrinck-2011.md).
- "Optimal" here is for *local* (one-point) transforms; genuine phase information (filaments) still
  requires higher-order or morphological statistics, consistent with the 1pt+2pt scope of the
  progenax inference.
