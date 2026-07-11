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

Generated at `2026-07-11T09:45:19.090945+00:00`. Line coverage: 96.2% (floor 90%). `Line-cov %` cells read **pending (Phase 2)** until the committed full-suite `coverage.json` exists, then show the statement-weighted per-directory coverage. The `Grad-audit fill` column reports the repo-wide differentiability registry (audited / audited+exempt, hazard count). Built registries: differentiability, API-coverage, physics-validation, provenance.

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
  - test_circular_orbit_com_at_origin (1.1s)
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
  - test_recovers_primordial_hard_binaries (30.2s)
  - PASS
* - `builders`
  - 43
  - 0
  - 0
  - 93.7
  - 41/122 audited, 0 haz
  - test_matched_limepy_anisotropic_threads_r_a (28.7s)
  - —
* - `cluster`
  - 67
  - 0
  - 0
  - 98.8
  - 41/122 audited, 0 haz
  - test_engine_b_global_virial_is_half_unscaled (95.3s)
  - —
* - `diagnostics`
  - 28
  - 0
  - 0
  - 92.4
  - 41/122 audited, 0 haz
  - test_diagnostics_import_without_scipy_gives_actionable_error (7.9s)
  - —
* - `dynamics`
  - 28
  - 0
  - 0
  - 97.8
  - 41/122 audited, 0 haz
  - test_memory_bounded_smoke_n20000 (9.2s)
  - —
* - `grad_audit`
  - 0
  - 0
  - 26
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_run_audit_emits_required_keys (685.1s)
  - —
* - `imf`
  - 275
  - 0
  - 0
  - 97.0
  - 41/122 audited, 0 haz
  - test_p_q_interrelation (24.9s)
  - —
* - `kinematics`
  - 138
  - 0
  - 0
  - 99.9
  - 41/122 audited, 0 haz
  - test_differentiable_in_g_through_table (44.4s)
  - —
* - `physics_registry`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_every_cited_test_node_id_resolves (34.5s)
  - —
* - `profiles`
  - 212
  - 0
  - 0
  - 98.9
  - 41/122 audited, 0 haz
  - test_grad_ra_hat_vs_fd_quadrature (279.9s)
  - —
* - `provenance_cards`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_validation_node_ids_resolve_and_assert (47.1s)
  - —
* - `provenance_registry`
  - 0
  - 0
  - 7
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_no_new_unprovenanced_literal_in_allowlist_modules (6.4s)
  - —
* - `stellar`
  - 20
  - 0
  - 0
  - 100.0
  - 41/122 audited, 0 haz
  - test_inverse_round_trip_wide_grid (1.9s)
  - —
* - `substructure`
  - 35
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_calibration_produces_valid_factors (113.1s)
  - —
* - `test_analytical_physics`
  - 0
  - 0
  - 28
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_eccentric_orbit_closes_and_conserves (2.1s)
  - —
* - `test_azimuthal_variation_physics`
  - 0
  - 0
  - 5
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_anticorrelates_with_cw04_Q (0.8s)
  - —
* - `test_binary_cluster`
  - 0
  - 11
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_invariants_with_moe_joint (20.8s)
  - —
* - `test_binary_physics`
  - 0
  - 0
  - 19
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_finite_through_e_to_one (2.2s)
  - —
* - `test_builders`
  - 14
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_build_with_plummer (2.9s)
  - —
* - `test_cluster_builders_integration`
  - 0
  - 7
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_each_profile_builds_near_virial[plummer] (13.8s)
  - —
* - `test_dashboard_fresh`
  - 0
  - 0
  - 3
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_committed_dashboard_matches_fresh_regeneration (122.1s)
  - —
* - `test_dashboard_gen`
  - 0
  - 0
  - 14
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_build_dashboard_has_all_blocks (122.1s)
  - —
* - `test_demo_oed_binary`
  - 5
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_cli_binary_quick_smoke (8.2s)
  - —
* - `test_dispersion_physics`
  - 0
  - 0
  - 24
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_projection_empirical_los_and_pm (34.8s)
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
  - test_eff_eddington_virial_ratio_mild_truncation (11.8s)
  - —
* - `test_end_to_end`
  - 0
  - 9
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_chabrier_to_king_ic (9.8s)
  - —
* - `test_engine_b_physics`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_plummer_halo_eff_core_equilibrium (267.8s)
  - —
* - `test_environment_physics`
  - 0
  - 0
  - 15
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_corrected_marks_equals_jerabkova (0.8s)
  - —
* - `test_grad_audit`
  - 0
  - 0
  - 100
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_gradient_audit[MultiComponentCluster.sample_cluster[EngineA]0] (158.5s)
  - —
* - `test_imf_physics`
  - 0
  - 0
  - 23
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_massive_more_common_with_lower_alpha (5.3s)
  - —
* - `test_jax_compatibility`
  - 0
  - 6
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_powerlaw_grad (4.8s)
  - —
* - `test_king_physics`
  - 0
  - 0
  - 35
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_auto_domain_preserves_differentiability_high_W0 (9.9s)
  - —
* - `test_limepy_reference_parity`
  - 0
  - 0
  - 3
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_parity_with_reference_limepy[twocomp_ra_eta05] (88.4s)
  - —
* - `test_mass_segregation_physics`
  - 0
  - 0
  - 9
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_estimator_converges_with_random_samples (56.0s)
  - —
* - `test_michie_physics`
  - 0
  - 0
  - 10
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_beta_matches_df_oracle (9.1s)
  - —
* - `test_multimass_equilibrium_physics`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_anisotropic_sampled_cluster_is_equilibrium_and_correctly_anisotropic (73.6s)
  - —
* - `test_numerics`
  - 79
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_ad_equals_fd_in_e[0.0] (2.9s)
  - —
* - `test_plummer_physics`
  - 0
  - 0
  - 20
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_radial_dispersion_profile (6.7s)
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
  - test_harness_orchestration_reproduces_zero_holes (5.0s)
  - —
* - `test_readme_examples`
  - 0
  - 10
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_readme_block_executes[2] (10.7s)
  - —
* - `test_rotation_anisotropy_physics`
  - 0
  - 0
  - 8
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_v_phi_linear_in_R (6.2s)
  - —
* - `test_segregation_approx_physics`
  - 0
  - 0
  - 13
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_positions_finite (14.1s)
  - —
* - `test_segregation_equilibrium_physics`
  - 0
  - 0
  - 4
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_primordial_full_segregation_is_per_group_equilibrium (35.6s)
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
  - test_q_roughly_n_independent (15.7s)
  - —
* - `test_tidal`
  - 15
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_vmap_over_r_t_monotone (2.4s)
  - —
* - `test_tidal_physics`
  - 0
  - 0
  - 9
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_bound_mass_equals_mass_within_rt (5.5s)
  - —
* - `test_units_through_pipeline`
  - 0
  - 2
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_build_spatial_ic_respects_G_without_virial_rescale (7.2s)
  - —
* - `test_zams_physics`
  - 0
  - 0
  - 34
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_inverse_differentiable (11.4s)
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
