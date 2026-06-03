---
title: Elson, Fall & Freeman (1987)
description: Annotated reference for R. A. W. Elson, S. M. Fall & K. C. Freeman — The structure of young star clusters in the Large Magellanic Cloud.
---

# Elson, Fall & Freeman (1987)

```{admonition} The structure of young star clusters in the Large Magellanic Cloud
:class: note

**Authors.** Rebecca A. W. Elson, S. Michael Fall, Kenneth C. Freeman

**Reference.** *The Astrophysical Journal* **323**, 54–78 (1987 December 1); received 1987
February 9, accepted 1987 May 27.

**ADS.** [1987ApJ...323...54E](https://ui.adsabs.harvard.edu/abs/1987ApJ...323...54E) ·
**DOI.** [10.1086/165807](https://doi.org/10.1086/165807)
```

## Abstract (paraphrased)

Surface-brightness profiles (from star counts + aperture photometry) for 10 young massive
star clusters in the LMC (NGC 1818, 1831, 1866, 2004, 2156, 2157, 2159, 2164, 2172, 2214),
ages $8\times10^6$–$3\times10^8$ yr. At large radii the **projected** density falls off as a
power law $\propto r^{-\gamma}$ with $2.2 \lesssim \gamma \lesssim 3.2$ and median $\gamma
\approx 2.6$ — *not* the abrupt King-style tidal cutoff. The clusters do not appear tidally
truncated; the authors suggest up to ~50% of the mass may lie in unbound halos.

## The EFF formula (verified against the paper, p. 61)

EFF fit the **surface brightness** $\mu(r)$ (the figures plot $V$ mag arcsec$^{-2}$) with

```{math}
:label: eff-surface
\mu(r) = \mu_0 \left(1 + \frac{r^2}{a^2}\right)^{-\gamma/2}  \qquad\text{(EFF Eq. 1)}
```

where **$r$ is the *projected* radius**, $\mu_0$ the central surface brightness, $a$ a scale
length, and $\gamma$ the **projected (surface-brightness) slope** ($\gamma \in [2.2, 3.2]$,
median 2.6; Table 1). "This form was chosen purely for mathematical convenience." Integrating
gives the enclosed projected luminosity (EFF Eq. 2):

```{math}
:label: eff-Lp
L_p(r) = \frac{2\pi\mu_0 a^2}{\gamma - 2}\left[1 - \left(1 + \frac{r^2}{a^2}\right)^{1-\gamma/2}\right].
```

```{admonition} Surface brightness vs. 3-D density — progenax's convention
:class: important
EFF Eq. 1 is the **projected surface brightness** $\mu(r)$, **not** the 3-D volume density.
`progenax.profiles.EFFProfile` adopts the *same functional form* as the **3-D volume density**
$\rho(r) = \rho_0\,[1 + (r/a)^2]^{-\gamma/2}$ — the standard convention in N-body/IC codes
(e.g. McLuster). Consequently **progenax's $\gamma$ is a 3-D density slope**, distinct from
EFF87's observed *surface* slope: a 3-D density $\propto r^{-\gamma}$ deprojects/projects with a
one-power offset (Abel), so an EFF87 surface slope $\gamma_{\rm EFF}$ corresponds to a 3-D slope
$\approx \gamma_{\rm EFF}+1$. At $\gamma = 5$ the 3-D form reduces exactly to Plummer. EFF has
**no built-in tidal truncation**; progenax adds an optional truncation radius $r_t$.
```

## Use in progenax

- [](../../10-theory/spatial-profiles/eff.md) — derivation and progenax implementation
- `progenax.profiles.EFFProfile(a, gamma, r_t)` — young-cluster 3-D density profile with optional
  tidal truncation; Eddington-inversion velocity DF in `progenax.kinematics.EFFVelocityDF`

## Notes

The right choice for young massive clusters where the outer slope is a free observational
parameter rather than a tidal cutoff. When comparing progenax's $\gamma$ to EFF87 / observed
surface-brightness fits, mind the surface-vs-3-D convention above.
