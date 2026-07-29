---
title: Validation plot gallery
description: Curated gallery of physics-validation figures — sampled-vs-analytic densities, equilibrium dispersions, IMF recoveries, binary results, engines, tidal physics, and the gradient audit, each linking to its quantitative per-suite page.
---
# Validation plot gallery

A curated visual tour of the physics-validation evidence — one or two headline
figures per model family, rendered by the per-suite validation scripts and
committed under `50-validation/figures/`. **The numbers live on the per-suite
pages** (linked from every caption); this page is for the eyes.

## Spatial profiles & velocity DFs

```{figure} figures/plummer_density.png
:width: 85%
Sampled vs analytic Plummer density — the inverse-CDF position sampler against
$\rho(r) \propto (1+r^2/a^2)^{-5/2}$. Numbers: [](plummer-equilibrium.md).
```

```{figure} figures/plummer_velocity_equilibrium.png
:width: 85%
Plummer velocity equilibrium: sampled dispersion against
$\sigma_r^2 = GM/(6\sqrt{r^2+a^2})$, unscaled $Q_{\rm vir} = 0.5$.
Numbers: [](plummer-equilibrium.md).
```

```{figure} figures/king_w0_sweep.png
:width: 85%
The King family across $W_0$ — concentration $c(W_0)$ against King (1966)
Table II. Numbers: [](king-profile.md).
```

```{figure} figures/king_velocity_equilibrium.png
:width: 85%
King lowered-Maxwellian equilibrium: dispersion falls outward and vanishes at
$r_t$ (cold outskirts). Numbers: [](king-profile.md).
```

```{figure} figures/eff_gamma_family.png
:width: 85%
The EFF power-law-halo family across $\gamma$ ($\gamma = 5$ is Plummer).
Numbers: [](eff-profile.md).
```

```{figure} figures/michie_beta_profile.png
:width: 85%
Michie–King anisotropy: realised $\beta(r)$ rises outward and sits below the
pure Osipkov–Merritt ceiling. Numbers: [](michie-anisotropy.md).
```

## Initial mass functions

```{figure} figures/imf_pdf_overlay.png
:width: 85%
Salpeter / Kroupa / Chabrier / Maschberger sampled PDFs against their analytic
forms. Numbers: [](imf-statistics.md).
```

```{figure} figures/env_fundamental_plane.png
:width: 85%
The environment-dependent high-mass slope $\alpha_3(\rho_{\rm cl}, [\mathrm{Fe/H}])$
— the Marks+2012 fundamental plane with the GC anchors.
Numbers: [](imf-statistics.md).
```

## Binary populations

```{figure} figures/binaries_moe_qdist.png
:width: 85%
Moe & Di Stefano mass-ratio statistics: sampled $g(q \mid M_1, P)$ with the
twin excess. Numbers: [](binary-imf.md).
```

```{figure} figures/binaries_confidently_wrong.png
:width: 85%
The "confidently wrong" headline: a binary-blind IMF fit biases the recovered
slope with shrinking error bars; the binary-aware fit recovers the truth.
Numbers: [](binary-imf.md).
```

## Multi-component engines

```{figure} figures/seg_multimass_equilibrium.png
:width: 85%
Engine A multimass equilibrium: per-component $Q_j = 0.5$ unscaled across the
mass spectrum. Numbers: [](two-component.md).
```

```{figure} figures/equipartition_saturation.png
:width: 85%
The honest sub-equipartition physics: $\sigma(m)$ saturates at low mass
(escape-speed ceiling) and follows $m^{-1/2}$ in the deep-well limit, with the
DERIVED $m_{\rm eq}$ crossover. Numbers: [](two-component.md).
```

```{figure} figures/engine_b_eddington.png
:width: 85%
Engine B density-defined equilibria: prescribed component densities, the
shared-$\Psi$ Eddington DFs, and the realizability margin.
Numbers: [](engine-b-eddington.md).
```

## Tidal physics & diagnostics

```{figure} figures/tidal_jacobi_vs_l1.png
:width: 85%
The Jacobi radius against the directly-computed L1 Lagrange distance.
Numbers: [](tidal-truncation.md).
```

```{figure} figures/q_mbar_sbar_plane.png
:width: 85%
The CW04 $(\bar m, \bar s)$ plane: substructured, uniform, and centrally
concentrated configurations separate cleanly.
Numbers: [](fractal-substructure.md).
```

```{figure} figures/lambda_msr_regimes.png
:width: 85%
$\Lambda_{\rm MSR}$ across segregation regimes.
Numbers: [](mass-segregation.md).
```

## Stellar relations & the gradient gate

```{figure} figures/zams_hr_diagram.png
:width: 85%
The Tout+1996 ZAMS in the HR diagram (75/75 coefficients PDF-verified).
Numbers: [](test-dashboard.md).
```

```{figure} figures/grad_audit_summary.png
:width: 80%
The 99-case gradient gate: 98 clean, 1 pinned known-limitation, 0 hazards.
Numbers: [](differentiability-audit.md).
```

```{seealso}
Every figure above (and ~70 more committed alongside them) also appears on its
per-suite page with the quantitative tables: [](plummer-equilibrium.md),
[](king-profile.md), [](eff-profile.md), [](michie-anisotropy.md),
[](imf-statistics.md), [](binary-imf.md), [](fractal-substructure.md),
[](mass-segregation.md), [](two-component.md), [](engine-b-eddington.md),
[](tidal-truncation.md), [](analytical-test-cases.md),
[](differentiability-audit.md).
```

## Removed suites (history)

The fractal-displacement-field *generator* and the legacy `bm19_fdf_suite`
plot scripts were removed in the 2026-06 clean-room rewrite; the surviving
substructure tooling is the CW04 $Q$ diagnostic ([](fractal-substructure.md)).
The experimental `gravoturb` acceptance suite *prints* its numbers
(AC1–AC17) rather than committing figures — run
`PYTHONPATH=src:src/experimental python -m gravoturb.validation.acceptance`
and see `src/experimental/gravoturb/VALIDATION_SUMMARY.md`. Publication
figures for that subsystem await the jaxstroviz port.

## Plot regeneration

Figures are produced by the per-suite validation scripts
(`scripts/validate_*.py`; `scripts/release_gate.sh` runs all 24 and refreshes
`validation_runs.json`). The committed copies under `50-validation/figures/`
are the deliberate exception to the otherwise-gitignored `validation/plots/`.
Methodology: [](methodology.md).
