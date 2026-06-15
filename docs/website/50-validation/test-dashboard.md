---
title: Test Dashboard
description: Generated single source of truth for progenax's per-module test census, line coverage, registry fill, durations, and validation runs.
---
# Test Dashboard

```{warning}
This page is **generated** — do not hand-edit. Regenerate it with

    uv run python scripts/build_test_dashboard.py --emit --render

which stamps the timestamp, emits `validation/data/test_dashboard.json`, and re-renders this page.
```

Generated at `2026-06-15T05:31:05.651237+00:00`. Line coverage: 95.7% (floor 90%). `Line-cov %` cells read **pending (Phase 2)** until the committed full-suite `coverage.json` exists, then show the statement-weighted per-directory coverage. The `Grad-audit fill` column reports the repo-wide differentiability registry (audited / audited+exempt, hazard count). Built registries: differentiability, API-coverage, physics-validation, provenance.

```{list-table} Per-module test + coverage matrix
:header-rows: 1
:align: left

* - Module
  - Unit
  - Integration
  - Validation
  - Line-cov %
  - Grad-audit fill
  - Slowest test
  - Validation PASS
* - `analytical`
  - 7
  - 0
  - 0
  - 93.4
  - 38/119 audited, 0 haz
  - —
  - —
* - `api_coverage`
  - 0
  - 0
  - 5
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `binaries`
  - 128
  - 0
  - 0
  - 95.8
  - 38/119 audited, 0 haz
  - —
  - —
* - `builders`
  - 43
  - 0
  - 0
  - 93.7
  - 38/119 audited, 0 haz
  - —
  - —
* - `cluster`
  - 63
  - 0
  - 0
  - 98.6
  - 38/119 audited, 0 haz
  - —
  - —
* - `diagnostics`
  - 28
  - 0
  - 0
  - 92.4
  - 38/119 audited, 0 haz
  - —
  - —
* - `dynamics`
  - 28
  - 0
  - 0
  - 97.8
  - 38/119 audited, 0 haz
  - —
  - —
* - `grad_audit`
  - 0
  - 0
  - 26
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `imf`
  - 269
  - 0
  - 0
  - 97.1
  - 38/119 audited, 0 haz
  - —
  - —
* - `kinematics`
  - 101
  - 0
  - 0
  - 99.1
  - 38/119 audited, 0 haz
  - —
  - —
* - `physics_registry`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `profiles`
  - 201
  - 0
  - 0
  - 98.8
  - 38/119 audited, 0 haz
  - —
  - —
* - `provenance_registry`
  - 0
  - 0
  - 7
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `stellar`
  - 25
  - 0
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `substructure`
  - 34
  - 0
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_analytical_physics`
  - 0
  - 0
  - 28
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_azimuthal_variation_physics`
  - 0
  - 0
  - 5
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_binary_cluster`
  - 0
  - 11
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_binary_physics`
  - 0
  - 0
  - 19
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_builders`
  - 14
  - 0
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_cluster_builders_integration`
  - 0
  - 7
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_dashboard_fresh`
  - 0
  - 0
  - 3
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_dashboard_gen`
  - 0
  - 0
  - 14
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_documented_api`
  - 16
  - 0
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_eff_physics`
  - 0
  - 0
  - 24
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_end_to_end`
  - 0
  - 9
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_engine_b_physics`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_environment_physics`
  - 0
  - 0
  - 15
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_grad_audit`
  - 0
  - 0
  - 92
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_imf_physics`
  - 0
  - 0
  - 23
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_jax_compatibility`
  - 0
  - 6
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_king_physics`
  - 0
  - 0
  - 35
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_limepy_reference_parity`
  - 0
  - 0
  - 3
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_mass_segregation_physics`
  - 0
  - 0
  - 9
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_michie_physics`
  - 0
  - 0
  - 10
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_multimass_equilibrium_physics`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_numerics`
  - 8
  - 0
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_plummer_physics`
  - 0
  - 0
  - 20
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_protocols`
  - 14
  - 0
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_public_api`
  - 2
  - 0
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_readme_examples`
  - 0
  - 9
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_rotation_anisotropy_physics`
  - 0
  - 0
  - 8
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_segregation_approx_physics`
  - 0
  - 0
  - 13
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_segregation_equilibrium_physics`
  - 0
  - 0
  - 4
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_strict_refs`
  - 0
  - 0
  - 1
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_substructure_q_physics`
  - 0
  - 0
  - 8
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_tidal`
  - 15
  - 0
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_tidal_physics`
  - 0
  - 0
  - 9
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_units_through_pipeline`
  - 0
  - 2
  - 0
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
* - `test_zams_physics`
  - 0
  - 0
  - 34
  - pending (Phase 2)
  - 38/119 audited, 0 haz
  - —
  - —
```
