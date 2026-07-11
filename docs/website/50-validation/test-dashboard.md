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

Generated at `2026-07-11T03:25:23.532929+00:00`. Line coverage: 96.2% (floor 90%). `Line-cov %` cells read **pending (Phase 2)** until the committed full-suite `coverage.json` exists, then show the statement-weighted per-directory coverage. The `Grad-audit fill` column reports the repo-wide differentiability registry (audited / audited+exempt, hazard count). Built registries: differentiability, API-coverage, physics-validation, provenance.

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
  - 93.2
  - 41/122 audited, 0 haz
  - test_circular_orbit_com_at_origin (0.3s)
  - PASS
* - `api_coverage`
  - 0
  - 0
  - 5
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_line_coverage_above_floor (0.0s)
  - —
* - `binaries`
  - 130
  - 0
  - 0
  - 95.9
  - 41/122 audited, 0 haz
  - test_Q_com_recovers_virial_target (19.4s)
  - PASS
* - `builders`
  - 43
  - 0
  - 0
  - 93.7
  - 41/122 audited, 0 haz
  - test_matched_limepy_anisotropic_threads_r_a (12.4s)
  - —
* - `cluster`
  - 67
  - 0
  - 0
  - 98.8
  - 41/122 audited, 0 haz
  - test_engine_b_global_virial_is_half_unscaled (58.2s)
  - —
* - `diagnostics`
  - 28
  - 0
  - 0
  - 92.4
  - 41/122 audited, 0 haz
  - test_returns_expected_keys (5.6s)
  - —
* - `dynamics`
  - 28
  - 0
  - 0
  - 97.8
  - 41/122 audited, 0 haz
  - test_single_group_reproduces_global_virial (4.9s)
  - —
* - `grad_audit`
  - 0
  - 0
  - 26
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_run_audit_emits_required_keys (420.8s)
  - —
* - `imf`
  - 275
  - 0
  - 0
  - 97.0
  - 41/122 audited, 0 haz
  - test_p_q_interrelation (12.1s)
  - —
* - `kinematics`
  - 138
  - 0
  - 0
  - 99.9
  - 41/122 audited, 0 haz
  - test_differentiable_in_g_through_table (39.0s)
  - —
* - `physics_registry`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_every_cited_test_node_id_resolves (26.5s)
  - —
* - `profiles`
  - 212
  - 0
  - 0
  - 98.9
  - 41/122 audited, 0 haz
  - test_grad_ra_hat_vs_fd_quadrature (110.3s)
  - —
* - `provenance_registry`
  - 0
  - 0
  - 7
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_no_new_unprovenanced_literal_in_allowlist_modules (2.6s)
  - —
* - `stellar`
  - 20
  - 0
  - 0
  - 100.0
  - 41/122 audited, 0 haz
  - test_inverse_round_trip_wide_grid (1.6s)
  - —
* - `substructure`
  - 35
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_calibration_produces_valid_factors (73.0s)
  - —
* - `test_analytical_physics`
  - 0
  - 0
  - 28
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_eccentric_orbit_closes_and_conserves (1.4s)
  - —
* - `test_azimuthal_variation_physics`
  - 0
  - 0
  - 5
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_anticorrelates_with_cw04_Q (1.6s)
  - —
* - `test_binary_cluster`
  - 0
  - 11
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_totalmass_reaches_budget (16.7s)
  - —
* - `test_binary_physics`
  - 0
  - 0
  - 19
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_finite_through_e_to_one (1.8s)
  - —
* - `test_builders`
  - 14
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_build_with_plummer (4.7s)
  - —
* - `test_cluster_builders_integration`
  - 0
  - 7
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_each_profile_builds_near_virial[michie] (7.0s)
  - —
* - `test_dashboard_fresh`
  - 0
  - 0
  - 3
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_committed_dashboard_matches_fresh_regeneration (60.2s)
  - —
* - `test_dashboard_gen`
  - 0
  - 0
  - 14
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_registries_full_flag_gates_on_built_and_full (73.1s)
  - —
* - `test_demo_oed_binary`
  - 5
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_cli_binary_quick_smoke (3.4s)
  - —
* - `test_dispersion_physics`
  - 0
  - 0
  - 24
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_projection_empirical_los_and_pm (21.6s)
  - —
* - `test_documented_api`
  - 16
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - —
  - —
* - `test_eff_physics`
  - 0
  - 0
  - 24
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_velocity_isotropy (9.5s)
  - —
* - `test_end_to_end`
  - 0
  - 9
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_chabrier_to_king_ic (8.8s)
  - —
* - `test_engine_b_physics`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_plummer_halo_eff_core_equilibrium (174.7s)
  - —
* - `test_environment_physics`
  - 0
  - 0
  - 15
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_corrected_marks_equals_jerabkova (0.6s)
  - —
* - `test_grad_audit`
  - 0
  - 0
  - 100
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_gradient_audit[MultiComponentCluster.sample_cluster[EngineA]1] (97.6s)
  - —
* - `test_imf_physics`
  - 0
  - 0
  - 23
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_chabrier_mean_mass_reasonable (4.5s)
  - —
* - `test_jax_compatibility`
  - 0
  - 6
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_powerlaw_grad (2.7s)
  - —
* - `test_king_physics`
  - 0
  - 0
  - 35
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_concentration_matches_king_table_ii[9-2.12] (6.0s)
  - —
* - `test_limepy_reference_parity`
  - 0
  - 0
  - 3
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_parity_with_reference_limepy[twocomp_ra_eta05] (49.9s)
  - —
* - `test_mass_segregation_physics`
  - 0
  - 0
  - 9
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_estimator_converges_with_random_samples (60.6s)
  - —
* - `test_michie_physics`
  - 0
  - 0
  - 10
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_wrt_mass_velocity_scale (9.0s)
  - —
* - `test_multimass_equilibrium_physics`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_anisotropic_sampled_cluster_is_equilibrium_and_correctly_anisotropic (37.3s)
  - —
* - `test_numerics`
  - 79
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_differentiable_in_weight_parameter (1.5s)
  - —
* - `test_plummer_physics`
  - 0
  - 0
  - 20
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_all_particles_bound (2.5s)
  - —
* - `test_protocols`
  - 14
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - —
  - —
* - `test_public_api`
  - 2
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - —
  - —
* - `test_ratchet_characterization`
  - 0
  - 0
  - 12
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_harness_orchestration_reproduces_zero_holes (3.9s)
  - —
* - `test_readme_examples`
  - 0
  - 9
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_readme_block_executes[1] (4.2s)
  - —
* - `test_rotation_anisotropy_physics`
  - 0
  - 0
  - 8
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_plummer_beta_matches_exact_om (6.0s)
  - —
* - `test_segregation_approx_physics`
  - 0
  - 0
  - 13
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_positions_finite (6.3s)
  - —
* - `test_segregation_equilibrium_physics`
  - 0
  - 0
  - 4
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_primordial_full_segregation_is_per_group_equilibrium (17.8s)
  - —
* - `test_strict_refs`
  - 0
  - 0
  - 1
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - —
  - —
* - `test_substructure_q_physics`
  - 0
  - 0
  - 8
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_q_roughly_n_independent (5.9s)
  - —
* - `test_tidal`
  - 15
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_vmap_over_r_t_monotone (1.2s)
  - —
* - `test_tidal_physics`
  - 0
  - 0
  - 9
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_bound_mass_matches_analytic_plummer_enclosed (2.6s)
  - —
* - `test_units_through_pipeline`
  - 0
  - 2
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_virial_rescale_masks_dropped_G (4.8s)
  - —
* - `test_zams_physics`
  - 0
  - 0
  - 34
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_inverse_differentiable (2.6s)
  - —
```

## Validation scripts

Exit status of every `scripts/validate_*.py` (from `validation/data/validation_runs.json`): **24 scripts, 0 failing**.

```{list-table} Validation script runs
:header-rows: 1
:align: left

* - Script
  - Status
* - `validate_analytical.py`
  - PASS
* - `validate_azimuthal_variation.py`
  - PASS
* - `validate_binaries.py`
  - PASS
* - `validate_cluster_builders.py`
  - PASS
* - `validate_cluster_ic.py`
  - PASS
* - `validate_df_tables.py`
  - PASS
* - `validate_eff.py`
  - PASS
* - `validate_environment.py`
  - PASS
* - `validate_equipartition_saturation.py`
  - PASS
* - `validate_imfs.py`
  - PASS
* - `validate_king.py`
  - PASS
* - `validate_limepy_reference.py`
  - PASS
* - `validate_mass_segregation.py`
  - PASS
* - `validate_michie.py`
  - PASS
* - `validate_multicomponent_eddington.py`
  - PASS
* - `validate_multimass_anisotropy.py`
  - PASS
* - `validate_multimass_equilibrium.py`
  - PASS
* - `validate_plummer.py`
  - PASS
* - `validate_rotation_anisotropy.py`
  - PASS
* - `validate_segregation_approx.py`
  - PASS
* - `validate_speed_routing.py`
  - PASS
* - `validate_substructure_q.py`
  - PASS
* - `validate_tidal.py`
  - PASS
* - `validate_zams.py`
  - PASS
```
