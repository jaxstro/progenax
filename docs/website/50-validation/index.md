---
title: Validation
description: progenax's validation section — what the three-tier test suite asserts, the methodology behind quantitative pass/fail criteria, and per-physics summary pages.
---

# Validation

This section records numerical and physical checks at specified configurations.
A passing test supports its stated quantity, parameter domain, reference,
precision, and criterion; it does not prove universal correctness. Retained
measurements and figures must be distinguished from new runs on the current
checkout. Experimental Gravoturb results do not qualify the released core.

The [point-in-time validation audit](audit-report.md) retains its June 2026
scope. The [current release verdict](../95-release/index.md) owns unresolved
release qualification. [Science demos](../60-science-demos/index.md) have their
own likelihood and recovery assumptions; a derivative check alone does not
qualify their statistical inference.

## Live status

:::{important} The V&V state is generated, not hand-maintained
The per-module test census, line-coverage, and registry fill now live in the
**generated** [](test-dashboard.md) — built by `scripts/build_test_dashboard.py`
from the test suite + the four registry manifests, staleness-gated in CI, and
**never hand-edited**. That page inventories tests and retained artifacts; counts, coverage, and
registry membership do not establish scientific qualification. The owning
validation pages state the checks and limits for each claim.

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
  - PP20 ζ(p) regression suite + BM19 forward chain (now in the experimental `gravoturb` package; historical record of the 2026-04-28 transcription-bug fix)
* - [](multimass-equilibrium.md)
  - Engine A: coupled multi-mass LIMEPY equilibrium — per-component σ(r) vs the DF moment, Q_j across δ, anisotropic β(r) vs the DF, DF-table budgets
* - [](engine-b-eddington.md)
  - Engine B: prescribed-density shared-potential Eddington equilibria — King A-vs-B cross-engine anchor, analytic Plummer DF oracles, OM anisotropy, realizability
* - [](two-component.md)
  - Superseded API (deleted 2026-06); pointer to the MultiComponentCluster engines + the surviving two-population check
* - [](cluster-builders.md)
  - The `build_cluster` convenience layer — virial Q across all 5 aliases, density recovery, tidal cut, rotation $L_z$, OM anisotropy; bit-identical-to-`build_spatial_ic` sugar
* - [](performance-memory.md)
  - DF-table acceleration and memory budgets: speed-CDF-table routing vs the numerical quadrature reference, construction/sampling speedups
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

Validation is claim-specific. Identify the physical model, parameter domain,
observable, numerical method, independent comparison, and acceptance criterion.
Unit, integration, and validation tests contribute different evidence; having
one test in each directory does not certify a module for production research.
Unperformed, unresolved, and failed comparisons remain visible.

[Methodology](methodology.md) explains independent references, scale-aware
error budgets, and test design. [Testing architecture](testing-architecture.md)
describes coverage metadata and its limits. Neither a registry entry nor a
successful documentation build replaces those scientific checks.

## References

The three-tier methodology, error-budget rules, and anchor lesson
are documented in detail at [](methodology.md). Experimental PP20 material is documented separately at
[](gravoturbulent-pp20.md) and remains outside release qualification.
