---
title: Gravoturbulence
description: progenax's gravoturbulence section — the chain from molecular-cloud density PDFs to the freefall-density factor to the Parmentier & Pasquali (2020) magnification factor and the Burkhart 2018/2021 dense-gas SFR framework.
---

# Gravoturbulence

```{admonition} Experimental — not in the released wheel
:class: warning
The gravoturbulent + fractal-density-field (FDF) pipeline was rebuilt **clean-room** (2026-06) as
the standalone **`gravoturb_fdf`** package — a follow-up-paper feature **excluded from the released
progenax wheel**. Import it as `gravoturb_fdf` (repo-only, under `src/experimental/`), **not** as
`progenax.gravoturb` (removed in the 2026-06 rewrite). Fresh validation:
`src/experimental/gravoturb_fdf/VALIDATION_SUMMARY.md`.
```

This section covers the framework that links **cloud-scale density
structure** to the **integrated star formation rate**. The chain is:

1. **Density PDF** — the volume-density distribution $p_V(\rho)$ in
   a turbulent self-gravitating cloud, parameterised by Mach number
   and forcing geometry {cite:p}`FederrathKlessen2012`.
2. **Freefall-density factor (FDF)** — the kernel $\rho/t_{\mathrm{ff}}(\rho)
   \propto \rho^{3/2}$ that weights local density by its star-forming
   efficiency.
3. **ζ magnification factor** — the geometric SFR boost over a
   uniform-density "top-hat" cloud, parameterised by the radial
   density-profile slope $p$ {cite:p}`ParmentierPasquali2020`, computed
   three ways (analytic / cored / direct-3D).
4. **BM19 framework** — the dense-gas SFR formalism that combines all
   the above into a predictive forward model for cloud-integrated SFR
   {cite:p}`Burkhart2018,BurkhartMocz2019`.
5. **Inference** — running the chain *backwards* to recover natal cloud
   parameters from observed substructure.

## Map of the section

```{list-table}
:header-rows: 1

* - Chapter
  - Scope
* - [](density-pdf-and-fdf.md)
  - The {cite:t}`FederrathKlessen2012` lognormal + power-law density PDF (Mach scaling, forcing $b$), the $\rho^{3/2}$ freefall-density kernel, the cloud-integrated SFR they combine to give, and the single canonical α↔p mapping.
* - [](magnification-factor.md)
  - The magnification factor ζ three ways: {cite:t}`ParmentierPasquali2020` analytic ζ(p) for power-law profiles, `magnification_factor_with_core` for cored profiles, and `zeta_fdf_direct` for an arbitrary 3D field — with the "which ζ-mode when" table.
* - [](bm19.md)
  - The {cite:t}`Burkhart2018,BurkhartMocz2019` framework that consumes ζ in a forward model for dense-gas SFR.
* - [](inference.md)
  - **The inference capstone** — running the chain *backwards*: natal cloud parameters $(\mathcal{M}, b, \alpha, \beta)$ from observed cluster substructure, via differentiable predicted statistics (Gaussianization/Mehler 2-point + counts-in-cells + a peaks-over-threshold tail block) and HMC, in 3-D and in projection. Makes the BM19 tail slope $\alpha$ inferable and gives the differentiable $\beta$ successor to $Q$/MST.
```

## Reading order

For a student first encountering the framework: read in TOC order
([](density-pdf-and-fdf.md) → [](magnification-factor.md) →
[](bm19.md)), then [](inference.md) as the capstone that inverts the
forward chain. Each chapter assumes only the conventions established
in the previous one.

For a researcher already familiar with the literature: jump directly to
[](magnification-factor.md) for the ζ derivation, the three computation
modes, and the Historical Note on the 2026-04-28 transcription bug fix,
or to [](bm19.md) for the full forward chain that consumes ζ.

For implementation work: each chapter ends with a code snippet showing
the corresponding `gravoturb_fdf` API. The module reference is the package
source under `src/experimental/gravoturb_fdf/` (see its `README.md` and
`VALIDATION_SUMMARY.md`); this experimental subsystem has no generated
website API page.

## Why ζ is computed three ways

The three ζ-computation modes (all in [](magnification-factor.md)) are
not redundant — each captures a different physical situation:

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
three are differentiable and `gravoturb_fdf` exposes them through a
unified API. The choice of which to use depends on the level of cloud
parameterisation in the inference target.

## References

The density-PDF framework is {cite:t}`FederrathKlessen2012`. The
PP20 magnification factor is {cite:t}`ParmentierPasquali2020`, with
the {cite:t}`Kainulainen2014` observational anchor for $p \approx 5/3$.
The BM19 framework is {cite:t}`Burkhart2018,BurkhartMocz2019`. The
{cite:t}`TanKrumholzMcKee2006` cluster-formation framework provides
some of the structural underpinning.
