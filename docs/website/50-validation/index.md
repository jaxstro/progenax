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

## Status dashboard

The single place to see the V&V state. ✅ = tests pass *and* verified figure(s)
embedded on the page; ⚠️ = tested but page/figures pending the module audit.
Released-core suite (2026-06-10): **1163 tests**
(unit 895 / integration 34 / **validation 234**), full gate green.

```{list-table}
:header-rows: 1

* - Module
  - Validated (tests pass)
  - Figures on page
  - Last verified
  - Run command
* - [profiles/Plummer](plummer-equilibrium.md)
  - ✅ (20)
  - ✅ 5
  - 2026-06-08
  - `python scripts/validate_plummer.py`
* - [profiles/King](king-profile.md)
  - ✅ (32)
  - ✅ 5
  - 2026-06-08
  - `python scripts/validate_king.py`
* - [profiles/EFF](eff-profile.md)
  - ✅ (23)
  - ✅ 5
  - 2026-06-08
  - `python scripts/validate_eff.py`
* - [kinematics/Michie-King](michie-anisotropy.md)
  - ✅ (12)
  - ✅ 5
  - 2026-06-08
  - `python scripts/validate_michie.py`
* - [kinematics/rotation & anisotropy](rotation-om-anisotropy.md)
  - ✅ (10)
  - ✅ 5
  - 2026-06-08
  - `python scripts/validate_rotation_anisotropy.py`
* - [substructure/CW04 Q + azimuthal](fractal-substructure.md)
  - ✅ (14)
  - ✅ 8
  - 2026-06-08
  - `python scripts/validate_substructure_q.py`
* - [cluster/multi-mass LIMEPY equilibrium — Engine A](multimass-equilibrium.md)
  - ✅ (6 + 42 unit)
  - ✅ 3
  - 2026-06-10
  - `python scripts/validate_multimass_equilibrium.py` (+ `_anisotropy`, `_df_tables`)
* - [cluster/multi-component Eddington — Engine B](engine-b-eddington.md)
  - ✅ (6)
  - ✅ 1
  - 2026-06-10
  - `python scripts/validate_multicomponent_eddington.py`
* - [cluster/two-component (superseded)](two-component.md)
  - ✅ (via `MultiComponentCluster`)
  - ✅ 1
  - 2026-06-10
  - `python scripts/validate_cluster_ic.py`
* - [diagnostics/Λ_MSR + segregation](mass-segregation.md)
  - ✅ (8 + 13 + 27 + 4 primordial)
  - ✅ 9
  - 2026-06-10
  - `python scripts/validate_segregation_approx.py`
* - [imf statistics](imf-statistics.md)
  - ✅ (25)
  - ✅ 5
  - 2026-06-08
  - `python scripts/validate_imfs.py`
* - [binary / Moe IMF](binary-imf.md)
  - ✅ (24)
  - ✅ 5
  - 2026-06-08
  - `python scripts/validate_binaries.py`
* - [environment IMF (Marks/IGIMF)](environment-imf.md)
  - ✅ (12)
  - ✅ 5
  - 2026-06-08
  - `python scripts/validate_environment.py`
* - [analytical test cases](analytical-test-cases.md)
  - ✅ (12)
  - ✅ 5
  - 2026-06-09
  - `python scripts/validate_analytical.py`
* - [tidal truncation](tidal-truncation.md)
  - ✅ (9 + 15 unit)
  - ✅ 5
  - 2026-06-09
  - `python scripts/validate_tidal.py`
* - [gravoturbulent / PP20](gravoturbulent-pp20.md)
  - ✅
  - ⚠️ pending
  - —
  - `pytest tests/validation -k pp20`
* - [performance & memory](performance-memory.md)
  - ✅ (gated: 7 RSS stages + 16× benchmark)
  - ✅ 3
  - 2026-06-10
  - `python scripts/benchmark_batch_a.py`
```

## Map of the section

```{list-table}
:header-rows: 1

* - Page
  - Scope
* - [](methodology.md)
  - The three-tier test architecture (unit / integration / validation), tolerance conventions, how to add new validation tests
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
* - [](binary-imf.md)
  - End-to-end forward-model + likelihood: reproduces "confidently wrong" regime at $N \gtrsim 10^4$
* - [](fractal-substructure.md)
  - CW04 Q substructure diagnostic: (s̄,m̄) plane, Table 1, differentiable q_approx
* - [](mass-segregation.md)
  - $\Lambda_{\mathrm{MSR}}$ diagnostic (analytic), energy-ranked generator, and the
    differentiable segregation observables (soft $\Lambda_{\mathrm{MSR}}$ / radial / $\Sigma$–$m$)
* - [](gravoturbulent-pp20.md)
  - PP20 ζ(p) regression suite (35 tests), BM19 forward chain
* - [](multimass-equilibrium.md)
  - Engine A: coupled multi-mass LIMEPY equilibrium — per-component σ(r) vs the DF moment, Q_j across δ, anisotropic β(r) vs the DF, DF-table budgets
* - [](engine-b-eddington.md)
  - Engine B: prescribed-density shared-potential Eddington equilibria — King A-vs-B cross-engine anchor, analytic Plummer DF oracles, OM anisotropy, realizability
* - [](two-component.md)
  - Superseded API (deleted 2026-06); pointer to the MultiComponentCluster engines + the surviving two-population check
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

A progenax module is "validated" when it has at least one test in
each of three categories:

```{list-table}
:header-rows: 1

* - Tier
  - Verifies
* - **Unit**
  - Per-function correctness on synthetic inputs. Covers signatures, return shapes, JIT-compatibility, gradient-finiteness
* - **Integration**
  - End-to-end behaviour through builder/integrator chains. Covers the user-facing entry points
* - **Validation**
  - Quantitative match to analytical or published values. The "scientific" tests
```

A module without all three tiers is treated as *experimental* — it
can be used, but its results are not signed off for production
research.

## Pass/fail conventions

```{list-table}
:header-rows: 1

* - Test type
  - Default tolerance
  - Rationale
* - Closed-form analytic
  - $1\!\times\!10^{-12}$ relative
  - Float64 precision; no statistical noise
* - Finite-$N$ statistical
  - $5\!\times\!10^{-3}$ relative at $N = 10^4$
  - Poisson noise on Monte Carlo means scales as $1/\sqrt{N}$
* - Approximation match
  - $5\!\times\!10^{-2}$ relative
  - e.g. cored ζ vs analytic ζ
* - Observational anchor
  - $1\!\times\!10^{-2}$ absolute (in the relevant quantity)
  - e.g. Kainulainen+14 ζ ≈ 1.79 ± 0.02
```

Tests use `pytest.approx(expected, rel=tolerance)` or `pytest.approx(expected,
abs=tolerance)` to express these conventions explicitly. A test that
passes by a wider margin than its tolerance is considered "OK"; a
test that fails its declared tolerance is a regression that blocks
merging.

## Three failure modes the suite catches

The validation suite has empirically caught three classes of
regression:

1. **Closed-form transcription bugs** — the 2026-04-28 PP20 ζ(p) bug
   ([](../90-development-log/2026-04-28-pp20-fix.md)) was caught by
   anchoring tests on the analytic value $\zeta(1.5) = \sqrt{2}$;
   the buggy formula gave $\infty$. Without the anchor, the bug
   would have shipped unnoticed because all other tests were
   self-consistent with the buggy formula.
2. **Inversion bugs** — the Plummer half-mass-radius bug
   (see the historical note in [](../10-theory/spatial-profiles/plummer.md)),
   where $a = r_h/\sqrt{2^{2/3}-1}$ was used instead of $a = r_h\sqrt{2^{2/3}-1}$,
   was caught by `test_half_mass_radius` checking $M(<r_h) = M/2$
   directly rather than checking $a = $ some constant.
3. **Differentiability regressions** — accidental introduction of
   `jnp.where` (instead of sigmoid) in a critical path is caught by
   the per-builder gradient-finiteness tests
   ([](../20-architecture/differentiability.md)).

Each failure mode is now anchored by a regression test. The pattern
is **anchor on the defining condition, not the derived constant** —
test "$M(<r_h) = M/2$" rather than "$a = 0.7664\,r_h$".

## How the validation suite runs

```{list-table}
:header-rows: 1

* - Frequency
  - What runs
  - Time budget
* - Per-push (CI)
  - Unit tests + fast integration
  - $< 5$ minutes
* - Pre-merge (CI)
  - + Slow integration + validation
  - $< 30$ minutes
* - Nightly
  - + Stress tests, large-N, long chains
  - $< 4$ hours
* - Pre-release
  - + Full reproducibility check on prior validation outputs
  - Manual
```

The "validation" tier specifically runs only on pre-merge to main —
running it on every push would slow developer iteration. The
trade-off is that a developer pushing to a feature branch sees only
unit + fast-integration results; a regression in the validation tier
shows up at the merge step.

## References

The three-tier methodology is documented in detail at [](methodology.md).
The PP20 ζ(p) regression suite is the largest single validation
contribution and is documented at [](gravoturbulent-pp20.md).
