---
title: Gravoturbulence
description: progenax's gravoturbulence section — the chain from molecular-cloud density PDFs to the freefall-density factor to the Parmentier & Pasquali (2020) magnification factor and the Burkhart 2018/2021 dense-gas SFR framework.
---

# Gravoturbulence

This section covers the framework that links **cloud-scale density
structure** to the **integrated star formation rate**. The chain is:

1. **Density PDF** — the volume-density distribution $p_V(\rho)$ in
   a turbulent self-gravitating cloud, parameterised by Mach number
   and forcing geometry {cite:p}`FederrathKlessen2012`.
2. **Freefall-density factor (FDF)** — the kernel $\rho/t_{\mathrm{ff}}(\rho)
   \propto \rho^{3/2}$ that weights local density by its star-forming
   efficiency.
3. **PP20 ζ(p) magnification factor** — the geometric SFR boost over
   a uniform-density "top-hat" cloud, parameterised by the radial
   density-profile slope $p$ {cite:p}`ParmentierPasquali2020`.
4. **BM19 framework** — the dense-gas SFR formalism that combines all
   the above into a predictive forward model for cloud-integrated SFR
   {cite:p}`Burkhart2018,Burkhart2021`.

## Map of the section

```{list-table}
:header-rows: 1

* - Chapter
  - Scope
* - [](density-pdf-fundamentals.md)
  - The {cite:t}`FederrathKlessen2012` lognormal + power-law density PDF; Mach-number scaling; turbulence-driving forcing parameter $b$.
* - [](freefall-density-factor.md)
  - The functional $\rho/t_{\mathrm{ff}}(\rho) \propto \rho^{3/2}$ and why it is the right SFR kernel.
* - [](pdf-and-fdf.md)
  - Combining the density PDF with the FDF to give the cloud-integrated SFR.
* - [](pp20.md)
  - {cite:t}`ParmentierPasquali2020` ζ(p) — the magnification factor for power-law profiles, with the canonical analytic form.
* - [](cored-profiles.md)
  - `magnification_factor_with_core` — numerical-integration ζ for cored profiles $\rho \propto [1+(r/r_c)^2]^{-p/2}$.
* - [](direct-3d-zeta.md)
  - `zeta_fdf_direct` — measure ζ directly from a 3D density field with no power-law assumption.
* - [](bm19.md)
  - The {cite:t}`Burkhart2018,Burkhart2021` framework that consumes ζ in a forward model for dense-gas SFR.
```

## Reading order

For a student first encountering the framework: read in TOC order
(density PDF → FDF → PDF+FDF → PP20 → BM19). Each chapter assumes
only the conventions established in the previous one.

For a researcher already familiar with the literature: jump
directly to [](pp20.md) for the PP20 derivation and the Historical
Note on the 2026-04-28 transcription bug fix, or to [](bm19.md) for
the full forward chain that consumes ζ.

For implementation work: each chapter ends with a code snippet showing
the corresponding progenax API. The full module reference is at
[](../../30-api/gravoturb.md).

## Why progenax computes ζ multiple ways

The three ζ-computation modes ([](pp20.md), [](cored-profiles.md),
[](direct-3d-zeta.md)) are not redundant — each captures a different
physical situation:

- **PP20 analytic ζ(p)** is exact for *pure power-law* profiles,
  which is a useful idealisation but rarely realistic for individual
  clouds.
- **Cored profile ζ** numerically integrates a $\rho \propto
  [1+(r/r_c)^2]^{-p/2}$ profile that is flat in the inner core and
  power-law outside — a much better description of real molecular
  clouds with thermal-pressure-supported centres.
- **Direct 3D ζ** measures ζ from an arbitrary 3D density field with
  no parametric assumption — the right choice when you have a
  simulation snapshot or detailed observation.

For HMC-based inference of cloud parameters from observed SFR, all
three are differentiable and progenax exposes them through a unified
API. The choice of which to use depends on the level of cloud
parameterisation in the inference target.

## References

The density-PDF framework is {cite:t}`FederrathKlessen2012`. The
PP20 magnification factor is {cite:t}`ParmentierPasquali2020`, with
the {cite:t}`Kainulainen2014` observational anchor for $p \approx 5/3$.
The BM19 framework is {cite:t}`Burkhart2018,Burkhart2021`. The
{cite:t}`TanKrumholzMcKee2006` cluster-formation framework provides
some of the structural underpinning.
