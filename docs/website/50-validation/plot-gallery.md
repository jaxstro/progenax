---
title: Validation plot gallery
description: Curated gallery of physics-validation figures rendered from the validation/plots/ directory — visual confirmation of the quantitative results documented in the per-suite pages.
---
# Validation plot gallery

A curated gallery of physics-validation figures. Each plot
visualises the quantitative result of a corresponding per-suite
test page. The filenames below are the figures that currently exist
under `validation/plots/` (PNG).

```{seealso}
For the *quantitative* numbers behind each plot, see the per-suite
validation pages: [](plummer-equilibrium.md), [](king-profile.md),
[](eff-profile.md), [](imf-statistics.md), [](binary-imf.md),
[](fractal-substructure.md), [](mass-segregation.md),
[](gravoturbulent-pp20.md), [](two-component.md), [](tidal-truncation.md),
[](analytical-test-cases.md).
```

## Spatial profiles

`validation/plots/profiles/profiles_plummer_density.png`,
`profiles_king_density.png`, `profiles_eff_density.png` — sampled vs
analytical density profiles for each model.

`validation/plots/profiles/profiles_comparison.png` — the three
profiles overlaid; `profiles_isotropy.png` — velocity isotropy check.

## Initial mass functions

`validation/plots/imf/{salpeter,kroupa,chabrier}_{pdf,cdf,tail}.png` —
PDF, CDF, and high-mass tail for each analytic IMF family.

## Binary IMF recovery

`validation/imf/plots/binary_aware_recovery.png` — the headline
"confidently wrong" result: a naive (binary-blind) fit biases the
recovered slope while the binary-aware fit recovers the truth.

`validation/imf/plots/{maschberger,piecewise}_recovery.png` — slope
recovery for the smooth and piecewise IMFs.

## HMC inference

`validation/plots/hmc_recovery/imf_hmc_recovery.png` and the
per-environment posteriors `imf_hmc_posterior_*.png` (solar, low-Z,
massive GC, NGC 7078, starburst, YMC).

## Fractal substructure

The fractal-displacement-field *generator* was removed in the 2026-06
rewrite, so its calibration figures ($Q_{\mathrm{CW}}$ /
$\sigma_\Sigma/\langle\Sigma\rangle$ vs $D$) are no longer produced. The
surviving substructure tooling is the CW04 $Q$ diagnostic
([](fractal-substructure.md)) and the experimental `gravoturb_fdf`
$Q(f_{\mathrm{sub}})$ calibration (below).

## Mass segregation

**Committed, embedded figures** (under `docs/website/50-validation/figures/`, shown on
[](../10-theory/tidal-and-substructure/mass-segregation.md)):

- `lambda_msr_regimes.png`, `lambda_msr_monotonic_convergence.png`,
  `lambda_msr_binary_caveat.png` — the $\Lambda_{\mathrm{MSR}}$ *diagnostic* validated against
  analytic ground truth (regenerate: `python scripts/validate_mass_segregation.py`).
- `cluster_ic_energy_sorted_segregation.png` (+ `cluster_ic_plummer_equilibrium.png`,
  `cluster_ic_two_component.png`) — the released-core cluster-IC generators
  (regenerate: `python scripts/validate_cluster_ic.py`).

These are the curated set committed for the docs (a deliberate exception to the
otherwise-gitignored `validation/plots/`). The *dynamical* mass-segregation figures
(Λ_MSR(t) evolution) await the deferred gravax N-body experiment —
`docs/notes/2026-06-08-gravax-segregation-validation-followup.md`.

## Gravoturbulence (BM19 / PN11 / PP20)

The legacy `bm19_fdf_suite` plot scripts were removed in the 2026-06
clean-room rewrite. The successor is the experimental **`gravoturb_fdf`**
package, whose acceptance suite currently *prints* its numbers (AC1–AC10)
rather than committing figures — run
`PYTHONPATH=src:src/experimental python -m gravoturb_fdf.validation.acceptance`
and see `src/experimental/gravoturb_fdf/VALIDATION_SUMMARY.md`. Publication
figures await the jaxstroviz port (this is a repo-only, follow-up-paper
subsystem).

## Environment-dependent IMF

`validation/plots/env_imf/env_imf_alpha3_surface.png`,
`env_imf_shapes.png`, `env_imf_gc_validation.png` — the
$\alpha_3$(density) surface, IMF shapes, and GC validation.

## Plot regeneration

There is no single unified plot-regeneration entry point in the current
checkout (the legacy `validation/run_all.py` was removed with the
gravoturbulent suite). Figures are produced by the per-suite validation
scripts that survive — for example the IMF / inference figures under
`validation/imf/`. A unified regenerator and the gravoturbulent figures
await the jaxstroviz port (see the TODO in `progenax/CLAUDE.md`).

## References

Each plot's underlying test is documented in the corresponding
per-suite page. Methodology is at [](methodology.md).
