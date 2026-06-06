---
title: Burkhart (2018)
description: Annotated reference for B. Burkhart — the star formation rate from the gravoturbulent density PDF (Part I of the BM19 framework).
---

# Burkhart (2018)

```{admonition} The star formation rate in the gravoturbulent interstellar medium
:class: note

**Authors.** B. Burkhart

**Reference.** *The Astrophysical Journal* **863, 118** (2018).

**DOI.** [10.3847/1538-4357/aad002](https://doi.org/10.3847/1538-4357/aad002)

**Verified.** **No held PDF.** This page is grounded *second-hand* via the held BM19 PDF
([](burkhart-mocz-2019.md), Burkhart & Mocz 2019) — of which this paper is Part I — and the
[](../../10-theory/gravoturbulence/bm19.md) / [](../../10-theory/gravoturbulence/pdf-and-fdf.md)
chapters. Burkhart-2018-specific equation numbers and numerical values are **not** independently
verified; treat them as framework-level, not primary-source, claims.
```

## The big idea

Star formation is gated by the **densest gas**, and a turbulent cloud spreads its mass across a wide
range of densities. Burkhart (2018) closes the loop between the two by predicting the
**cloud-integrated star formation rate (SFR) directly from the density probability distribution
function (PDF)**. The recipe: take the gravoturbulent density PDF — a lognormal body from supersonic
turbulence joined to a power-law high-density tail from self-gravity — weight each density by its
star-forming efficiency (the freefall-density factor $\rho/t_{\rm ff} \propto \rho^{3/2}$), and
integrate over the self-gravitating tail. The result is a forward model for the SFR as a function of
the cloud's turbulence: the sonic Mach number $\mathcal{M}$, the forcing parameter $b$, and the tail
slope $\alpha$.

This is **Part I** of the framework progenax adopts; the companion [](burkhart-mocz-2019.md) (Part II,
BM19) makes the construction self-consistent by deriving the **transition density** $s_t$ — where the
lognormal hands off to the power-law tail — from the condition that the Jeans length equals the sonic
length, and by giving the closed-form self-gravitating mass fraction $f_{\rm dense}$.

## Where it sits in the chain

```{math}
\mathrm{SFR}_{\rm cloud} \;\propto\; \int_{\rho_t}^{\infty}
\left(\frac{\rho}{\langle\rho\rangle}\right)^{3/2} p_V(\rho)\,\mathrm{d}\rho ,
```

the same integral developed in [](../../10-theory/gravoturbulence/pdf-and-fdf.md). The lower limit
$\rho_t$ (equivalently $s_t = \ln(\rho_t/\rho_0)$) encodes the physical assumption that *only
self-gravitating gas forms stars*: below it, turbulence compresses and re-expands gas without forming
stars; above it, collapse proceeds and the local SFR follows the freefall kernel. The Parmentier &
Pasquali magnification factor $\zeta$ ([](parmentier-pasquali-2020.md)) is the geometric,
radial-profile dual of this density-space integral.

## Use in progenax

- [](../../10-theory/gravoturbulence/bm19.md) — the BM19 forward chain (Part I → Part II).
- [](../../10-theory/gravoturbulence/pdf-and-fdf.md) — the PDF + freefall-density-factor SFR formula.
- The 1-point physics it parameterises ($\sigma_s^2$, $s_t$, $f_{\rm dense}$) is implemented in
  [`theory/bm19.py`](../../../../src/experimental/gravoturb_fdf/theory/bm19.py); these scalars are the
  inputs the differentiable-inference layer
  ([](../../10-theory/gravoturbulence/differentiable-inference.md)) recovers from observed
  substructure.

## Notes

- The self-gravitating-gas-fraction and the closed-form $s_t$ used throughout `gravoturb_fdf` are the
  **held, PDF-verified** Part II results of {cite:t}`BurkhartMocz2019` ([](burkhart-mocz-2019.md)) —
  refer to that note for the grounded equations. This page covers the SFR-prediction motivation only.
- To upgrade this page to a fully PDF-verified note (with a "Verified" line like the other
  gravoturbulence references), add the Burkhart (2018) PDF to `docs/core-papers/`.
