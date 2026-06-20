---
title: Validation
description: progenax's validation section — what the three-tier test suite asserts, the methodology behind quantitative pass/fail criteria, and per-physics summary pages.
---

# Validation

This section documents what progenax's test suite *proves about the
physics*. Where the API reference says "this function exists and has
this signature," validation says "this function reproduces ζ(1.67) =
1.789 from {cite:t}`Kainulainen2014` to within 0.02." Validation is
where the package's scientific credibility lives.

The validation suite is **physics-anchored**: every test asserts a
quantitative match between progenax output and an analytical or
published-observational ground truth, with explicit pass/fail
tolerances.

See the [**validation audit report**](audit-report.md) for trustworthiness tiers, limits, recommendations, and the remaining/incomplete roadmap (point-in-time, 2026-06-08).

Where this section proves each model reproduces its ground truth, the
[**science demos**](../60-science-demos/index.md) run the models *backwards* —
recovering cluster parameters from mock observations by differentiable MLE / NUTS.
That inference rests on gradient integrity: the
[**differentiability gradient audit**](differentiability-audit.md) measures every public
entry point's autodiff gradient against finite differences (gradient integrity = Fisher
integrity), and reports the two hazards it found and fixed.

## Live status

:::{important} The V&V state is generated, not hand-maintained
The per-module test census, line-coverage, and registry fill now live in the
**generated** [](test-dashboard.md) — built by `scripts/build_test_dashboard.py`
from the test suite + the four registry manifests, staleness-gated in CI, and
**never hand-edited**. That page is the single source of truth for "what is
validated"; this section's prose pages explain *how* each model is validated.

The concepts behind the dashboard — the four frozen-literal registries
(API-coverage, physics-validation, provenance, differentiability) and how they
ratchet coverage against `progenax.__all__` — are documented in
[](testing-architecture.md).
:::

## Map of the section

```{list-table}
:header-rows: 1

* - Page
  - Scope
* - [](methodology.md)
  - The three-tier test architecture (unit / integration / validation), tolerance conventions, the anchor-on-defining-condition lesson, how to add new validation tests
* - [](testing-architecture.md)
  - The validation backbone: the four frozen-literal registries (API-coverage, physics, provenance, differentiability), the generated dashboard, and the four-part release gate
* - [](differentiability-audit.md)
  - The per-entry-point autodiff-vs-finite-difference gradient registry (gradient integrity = Fisher integrity); the two found-and-fixed silent-zero hazards
* - [](plummer-equilibrium.md)
  - Virial Q recovery, density-profile sampling, velocity-dispersion radial profile, energy conservation
* - [](king-profile.md)
  - ODE integration vs King (1966) Table II concentrations, tidal-truncation behaviour, $W_0$ sweep
* - [](eff-profile.md)
  - Density-profile sampling, asymptotic-slope verification
* - [](michie-anisotropy.md)
  - Anisotropy β(r) vs the DF oracle, isotropic King limit, anisotropic dispersions
* - [](rotation-om-anisotropy.md)
  - Solid-body & differential rotation; Osipkov-Merritt β(r) for Plummer/EFF
* - [](imf-statistics.md)
  - Salpeter / Kroupa / Chabrier / Maschberger sampling: KS-test goodness-of-fit and recovered $\alpha$ vs truth
* - [](environment-imf.md)
  - Environment-dependent IMF: Marks (2012) Fundamental Plane $\alpha_3$ and Jeřábková (2018) low-mass slopes vs published tables
* - [](zams-relations.md)
  - Tout (1996) ZAMS $L(M)$/$R(M)$ fits vs the published coefficients and stellar anchors
* - [](binary-imf.md)
  - End-to-end forward-model + likelihood: reproduces "confidently wrong" regime at $N \gtrsim 10^4$
* - [](fractal-substructure.md)
  - CW04 Q substructure diagnostic: (s̄,m̄) plane, Table 1, differentiable q_approx
* - [](mass-segregation.md)
  - $\Lambda_{\mathrm{MSR}}$ diagnostic (analytic), energy-ranked generator, and the
    differentiable segregation observables (soft $\Lambda_{\mathrm{MSR}}$ / radial / $\Sigma$–$m$)
* - [](gravoturbulent-pp20.md)
  - PP20 ζ(p) regression suite + BM19 forward chain (now in the experimental `gravoturb_fdf` package; historical record of the 2026-04-28 transcription-bug fix)
* - [](multimass-equilibrium.md)
  - Engine A: coupled multi-mass LIMEPY equilibrium — per-component σ(r) vs the DF moment, Q_j across δ, anisotropic β(r) vs the DF, DF-table budgets
* - [](engine-b-eddington.md)
  - Engine B: prescribed-density shared-potential Eddington equilibria — King A-vs-B cross-engine anchor, analytic Plummer DF oracles, OM anisotropy, realizability
* - [](two-component.md)
  - Superseded API (deleted 2026-06); pointer to the MultiComponentCluster engines + the surviving two-population check
* - [](cluster-builders.md)
  - The `build_cluster` convenience layer — virial Q across all 5 aliases, density recovery, tidal cut, rotation $L_z$, OM anisotropy; bit-identical-to-`build_spatial_ic` sugar
* - [](performance-memory.md)
  - DF-table acceleration and memory budgets: speed-CDF-table routing vs the exact quadrature oracle, construction/sampling speedups
* - [](tidal-truncation.md)
  - Jacobi-radius computation and truncation behaviour
* - [](analytical-test-cases.md)
  - Two-body Kepler, three-body figure-eight, harmonic oscillator — exact-solution sanity tests
* - [](physics-tests.md)
  - Cross-cutting physics validations not specific to one module
* - [](plot-gallery.md)
  - Rendered figures from `validation/plots/`
```

## What "validated" means

A progenax module is "validated" when it has at least one test in each
of three tiers — **unit** (per-function mechanical correctness),
**integration** (end-to-end builder/pipeline behaviour), and
**validation** (quantitative match to analytic or published physics).
A module without all three tiers is treated as *experimental* — it can
be used, but its results are not signed off for production research.

The tier definitions, the pass/fail **tolerance conventions**
(closed-form $10^{-12}$, finite-$N$ statistical $5\times10^{-3}$,
approximation $5\times10^{-2}$, observational anchor $10^{-2}$), and the
single most important methodology lesson — **anchor on the defining
condition, not the derived constant** (test $M(<r_h)=M/2$, not
$a = 0.7664\,r_h$) — all live in one place: [](methodology.md). The
backbone that *enforces* coverage (the three-tier suite, the four
registries, the generated dashboard, and the release gate) is
documented at [](testing-architecture.md).

## References

The three-tier methodology, tolerance conventions, and anchor lesson
are documented in detail at [](methodology.md). The PP20 ζ(p) regression
suite is the largest single validation contribution and is documented at
[](gravoturbulent-pp20.md).
