---
title: Burkhart (2018)
description: Annotated reference for Blakesley Burkhart — the analytic star formation rate from a piecewise lognormal + power-law density PDF, the SFR Part I of the BM19 framework.
---

# Burkhart (2018)

```{admonition} The Star Formation Rate in the Gravoturbulent Interstellar Medium
:class: note

**Authors.** Blakesley Burkhart

**Reference.** *The Astrophysical Journal* **863, 118** (11pp, 2018).

**DOI.** [10.3847/1538-4357/aad002](https://doi.org/10.3847/1538-4357/aad002)

**Verified.** Abstract, §1–3 (Eqs. 1–19) checked against the held PDF (2026-06). The two facts
`gravoturb` depends on: the **piecewise lognormal + power-law density PDF** (Eq. 18 — the
`bm19_volume_pdf` form) and the **positive-α convention** $p_{\rm PL}\propto e^{-\alpha s}$ (Eq. 6).
```

## The big idea

Earlier analytic star-formation-rate (SFR) models — Krumholz & McKee (2005), Padoan & Nordlund
(2011), Hennebelle & Chabrier (2011), Federrath & Klessen (2012) — integrate the freefall-weighted
density over a **purely lognormal (LN)** density PDF set by supersonic turbulence. Burkhart (2018)
extends this to the form actually seen in simulations and column-density observations of giant
molecular clouds: a **piecewise LN + power-law (PL)** PDF, where self-gravity carves a high-density
power-law tail onto the turbulent lognormal body. The central physical narrative: gas becomes
gravitationally unstable past a critical density $\rho_{\rm crit}$ and forms the PL tail; **as the
cloud collapses, the transition density $\rho_t$ between LN and PL moves to lower density while the
PL slope $\alpha$ becomes increasingly shallow**, and the SFR *accelerates* beyond the LN-only
prediction. This explains why star-formation efficiency per free-fall time increases with shallower
PL slopes, and why depletion times vary across local and extragalactic clouds — without invoking
extreme variations in turbulence.

This is the **SFR Part I** of the framework progenax adopts; the companion [](burkhart-mocz-2019.md)
(Part II, BM19) makes the construction self-consistent by deriving the transition density $s_t$ from
the condition that the Jeans length equals the sonic length, and gives the closed-form
self-gravitating mass fraction $f_{\rm dense}$.

## Core results

**The turbulent lognormal width (Eq. 5).** The LN body width is set by the sonic Mach number and the
forcing parameter,

```{math}
:label: bk18-sigma
\sigma_s^2 \;=\; \ln\!\big[\,1 + b^2 \mathcal{M}_s^2\,\big],
```

with $s\equiv\ln(\rho/\rho_0)$ (Eq. 3) and the mass-conserving mean $s_0=-\tfrac12\sigma_s^2$ (Eq. 4).

**The piecewise LN + PL PDF (Eq. 18).** The density PDF is a lognormal body joined at $s_t$ to a
power-law tail,

```{math}
:label: bk18-pdf
p_{\rm LN+PL}(s) =
\begin{cases}
N\,\dfrac{1}{\sqrt{2\pi\sigma_s^2}}\,e^{-(s-s_0)^2/2\sigma_s^2}, & s < s_t,\\[1.2ex]
N\,C\,e^{-\alpha s}, & s > s_t,
\end{cases}
```

normalised by $N$ (Eq. 19, a closed form in $C$, $\alpha$, $s_t$, $\sigma_s$). Requiring
$p_{\rm LN+PL}$ to be **continuous and differentiable** at $s_t$ fixes the amplitude $C$ and the
transition $s_t$ analytically. This is exactly the
[`bm19_volume_pdf`](../../../../src/experimental/gravoturb/theory/density_cdf.py) implemented in
`gravoturb`.

**The α sign convention (Eq. 6).** Burkhart writes the tail as $p_{\rm PL}(s)=C\,e^{-\alpha s}$ for
$s>s_t$ and notes explicitly that *"in our definition of the PL slope $\alpha$ is positive since the
minus sign appears in the exponent separately."* progenax uses this same convention throughout
(`bm19_volume_pdf`, the peaks-over-threshold tail block), so $\alpha$ is positive and a *steeper*
tail means a *larger* $\alpha$.

**The SFR integral (Eqs. 7–8).** The SFR per free-fall time is the freefall-weighted integral over
the PDF above the critical density,

```{math}
:label: bk18-sfr
\mathrm{SFR}_{\rm ff} = \epsilon_0 \int_{s_{\rm crit}}^{\infty}
\frac{t_{\rm ff}(\rho_0)}{t_{\rm ff}(\rho)}\,\frac{\rho}{\rho_0}\,p_{\rm LN+PL}(s)\,\mathrm{d}s,
\qquad
\mathrm{SFR} = \frac{M_{\rm cloud}}{t_{\rm ff}(\rho_0)}\,\mathrm{SFR}_{\rm ff},
```

split into an LN integral from $\rho_{\rm crit}$ to $\rho_t$ plus a PL integral from $\rho_t$ to the
maximum density. The paper reviews how the critical density $\rho_{\rm crit}$ differs between the
KM05, PN11, and Hennebelle–Chabrier models (Eqs. 9–17).

## Use in progenax

- The piecewise PDF {eq}`bk18-pdf` is [`bm19_volume_pdf`](../../../../src/experimental/gravoturb/theory/density_cdf.py);
  {eq}`bk18-sigma` is [`sigma_s_squared`](../../../../src/experimental/gravoturb/theory/density_pdf.py).
  These 1-point scalars are the inputs the differentiable-inference layer
  ([](../../10-theory/gravoturbulence/inference.md)) recovers from observed
  substructure.
- The SFR integral {eq}`bk18-sfr` is the forward chain of
  [](../../10-theory/gravoturbulence/density-pdf-and-fdf.md) and [](../../10-theory/gravoturbulence/bm19.md);
  the geometric, radial-profile dual is the Parmentier & Pasquali ζ ([](parmentier-pasquali-2020.md)).
- The positive-α convention (Eq. 6) is the one used by the BM19 tail and the peaks-over-threshold
  estimator.

## Notes

- Burkhart (2018) establishes the LN+PL PDF and its SFR; the **self-consistent** transition density
  $s_t=(\alpha-\tfrac12)\sigma_s^2$ (Jeans = sonic) and the closed-form $f_{\rm dense}$ used in the
  code are the **Part II** results of {cite:t}`BurkhartMocz2019` ([](burkhart-mocz-2019.md)).
- The LN+PL form is motivated by observations: the dense star-forming gas PDF takes a piecewise LN+PL
  shape in both 3-D density and column density (Kainulainen et al. 2009; Collins et al. 2012;
  Burkhart et al. 2017; [](kainulainen-2014.md)); some GMCs may be fully power-law.
