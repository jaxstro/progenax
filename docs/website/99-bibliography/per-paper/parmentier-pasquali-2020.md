---
title: Parmentier & Pasquali (2020)
description: "Annotated reference for G. Parmentier & A. Pasquali — a new parameterization of the SFR–dense-gas-mass relation embracing gas density gradients."
---

# Parmentier & Pasquali (2020)

```{admonition} A new parameterization of the star formation rate–dense gas mass relation: embracing gas density gradients
:class: note

**Authors.** G. Parmentier, A. Pasquali

**Reference.** *The Astrophysical Journal* **903, 56** (2020).

**DOI.** [10.3847/1538-4357/abb8d3](https://doi.org/10.3847/1538-4357/abb8d3) ·
**arXiv.** [2009.10652](https://arxiv.org/abs/2009.10652)

**Verified.** Equations + ranges checked against the held PDF (pp. 1–5, Eqs. 1–9, Fig. 1; re-verified 2026-06).
```

## The big idea

Two clouds with the same **dense-gas mass** $M_\mathrm{dg}$ can form stars at different
rates if their density *gradients* differ: a centrally-concentrated clump has more of its
mass at high density, where the local free-fall time $\tau_\mathrm{ff}\propto\rho^{-1/2}$
is short, so it forms stars faster than a uniform ("top-hat") clump of the same mass. PP20
capture this geometric boost with a single **magnification factor** $\zeta$, so that

$$
\frac{\mathrm{SFR}_\mathrm{dg}}{M_\mathrm{dg}}
   = \zeta\,\frac{\epsilon_\mathrm{ff,int}}{\langle\tau_\mathrm{ff,dg}\rangle}
\qquad\text{(their Eq. 7).}
$$

$\zeta=1$ is the top-hat lower limit; $\zeta>1$ for any centrally-concentrated profile.

## Physical & observational context (Sections 1–3)

The dense-gas star-formation law — $N_\mathrm{YSO}\propto M_\mathrm{dg}$, equivalently
$\mathrm{SFR}\propto M_\mathrm{dg}$ — is observationally near-linear (Lada et al. 2010 find
$N_\mathrm{YSO}=0.18\,M_\mathrm{dg}$) but carries real *scatter*. PP20's thesis is that the
scatter is not purely random: it partly tracks the cloud's internal **density gradient**.
Steeper gradients (larger $p$) push more mass into the short-free-fall-time inner region,
raising $\mathrm{SFR}/M_\mathrm{dg}$. The two key inputs PP20 separate (their Eq. 2) are the
*intrinsic* efficiency per free-fall time $\epsilon_\mathrm{ff,int}$ (a true SF-physics
number) and the purely *geometric* boost $\zeta$ — so that measuring a cloud's gradient
unlocks its $\zeta$ and lets one read $\epsilon_\mathrm{ff,int}$ cleanly from the data.

The radial slope $p$ is inferred observationally from the cloud $\rho$-pdf or projected
$\Sigma$-pdf: for a power-law tail of index $n$ in the $\Sigma$-pdf, $p = 1 + 2/n$ (Kritsuk
et al. 2011); $p\equiv\kappa$ in the Kainulainen et al. (2014) notation. The Kainulainen
sample has a mean **$p \approx 1.67$** (range $1<p<2.2$), which is the canonical observational
anchor: $\zeta(1.67)\approx1.79$. PP20 also apply the framework to the Central Molecular Zone
(CMZ), whose clouds sit ~10× below the nearby-cloud locus in $(p,\,\mathrm{SFR}/M_\mathrm{dg})$.

## Core equations

**Why $\zeta$ has this form.** The dense-gas SFR is the mass-weighted free-fall rate,
$\propto\langle\rho^{1/2}\rangle_\mathrm{mass}$, while the "naive" rate uses the mean
density, $\propto\langle\rho\rangle^{1/2}$. Hence

$$
\zeta = \frac{\langle\rho^{1/2}\rangle_\mathrm{mass}}{\langle\rho\rangle^{1/2}}
      = \frac{\int \rho^{3/2}\,dV\;(\int dV)^{1/2}}{(\int \rho\,dV)^{3/2}} .
$$

**Closed form for a power-law sphere $\rho\propto r^{-p}$ (Eq. 6, in Eq. 9).** Performing
the integrals over the sphere gives

$$
\zeta(p) = \frac{(3-p)^{3/2}}{\tfrac{3^{3/2}}{2}\,(2-p)}
         = \frac{(3-p)^{3/2}}{2.6\,(2-p)} ,
\qquad 0 \le p < 2 .
$$

PP20 print the constant as the rounded **2.6**; the exact value is $3^{3/2}/2 = 2.598$,
fixed by the physical top-hat limit $\zeta(0)=1$. The factor $(2-p)$ in the denominator
means $\zeta$ **diverges only as $p\to2$** — there is *no* pole at $p=1.3$ (a previously
caught transcription fabrication). Spot values:

| $p$ | 0 | 1 | 1.5 | 1.67 |
|---|---|---|---|---|
| $\zeta(p)$ | 1 (exact) | 1.089 | $\sqrt2 = 1.414$ (exact) | 1.79 |

**Forbidden regions (their Fig. 1).** Equation (6) is the *upper* limit on $\zeta$ (pure
power law), and $\zeta=1$ is the *lower* limit (top-hat); real cored profiles lie between.
For $p\ge2$ a pure power law would drive the central density — and the SFR — to infinity, so
PP20 add a flat inner **core**: a cored profile $\rho(r)=\rho_c[1+(r/r_c)^2]^{-p/2}$ has a
finite $\zeta$ even for $p\ge2$, obtained by numerically integrating the ratio above over
$r/R\in[0,1]$. The core *damps* the boost: a relative core size $r_c/r_\mathrm{clump}>0.1$
already reduces $\zeta$ significantly (their Fig. 1), because it shrinks the centre-to-edge
density contrast.

**Link to BM19.** The PDF slope $\alpha$ (Burkhart & Mocz) and the radial slope $p$ are
the same quantity: $p = \kappa = 3/\alpha$. So $\alpha\in[1.5,3]\Leftrightarrow p\in[1,2]$.

## Use in progenax

- [](../../10-theory/gravoturbulence/magnification-factor.md) — full $\zeta(p)$ derivation, equivalence proof, spot values, and the generalisation to cored and direct-3D profiles.
- `experimental/gravoturb/theory/dense_gas_sfr.py` — `magnification_factor` (analytic),
  `magnification_factor_with_core` (trapezoid), `zeta_from_field` (field estimator).

Validation: anchors $\zeta(0)=1,\ \zeta(1)=1.089,\ \zeta(1.5)=\sqrt2,\ \zeta(1.67)=1.79$
to $<0.1\%$ (AC3); the direct field estimator matches the analytic $\zeta(p)$ to $\sim3\%$
on a sampled power law (AC4).

## Notes

- $\zeta$ is a **Part-III / SFR-interpretation** quantity: it is *not* needed to *generate*
  the gravoturbulent ICs, only to interpret the dense-gas SFR afterwards.
- The 2026-04-28 transcription fix at [](../../90-development-log/2026-04-28-pp20-fix.md)
  removed the spurious $p=1.3$ pole; the clean-room `pp20.py` re-derives the exact
  $3^{3/2}/2$ constant from first principles.
