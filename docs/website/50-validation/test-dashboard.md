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

Generated at `2026-07-11T02:42:46.242026+00:00`. Line coverage: 96.2% (floor 90%). `Line-cov %` cells read **pending (Phase 2)** until the committed full-suite `coverage.json` exists, then show the statement-weighted per-directory coverage. The `Grad-audit fill` column reports the repo-wide differentiability registry (audited / audited+exempt, hazard count). Built registries: differentiability, API-coverage, physics-validation, provenance.

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
  - test_all_survive_at_t0 (16.2s)
  - PASS
* - `builders`
  - 43
  - 0
  - 0
  - 93.7
  - 41/122 audited, 0 haz
  - test_matched_limepy_anisotropic_threads_r_a (12.1s)
  - —
* - `cluster`
  - 67
  - 0
  - 0
  - 98.8
  - 41/122 audited, 0 haz
  - test_engine_b_global_virial_is_half_unscaled (90.8s)
  - —
* - `diagnostics`
  - 28
  - 0
  - 0
  - 92.4
  - 41/122 audited, 0 haz
  - test_returns_expected_keys (5.0s)
  - —
* - `dynamics`
  - 28
  - 0
  - 0
  - 97.8
  - 41/122 audited, 0 haz
  - test_single_group_reproduces_global_virial (3.1s)
  - —
* - `grad_audit`
  - 0
  - 0
  - 26
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_run_audit_emits_required_keys (810.7s)
  - —
* - `imf`
  - 275
  - 0
  - 0
  - 97.0
  - 41/122 audited, 0 haz
  - test_p_q_interrelation (18.1s)
  - —
* - `kinematics`
  - 138
  - 0
  - 0
  - 99.9
  - 41/122 audited, 0 haz
  - test_grad_df_moment_michie_high_W0_ad_correct (29.7s)
  - —
* - `physics_registry`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_every_cited_test_node_id_resolves (27.4s)
  - —
* - `profiles`
  - 212
  - 0
  - 0
  - 98.9
  - 41/122 audited, 0 haz
  - test_grad_ra_hat_vs_fd_quadrature (485.8s)
  - —
* - `provenance_registry`
  - 0
  - 0
  - 7
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_no_new_unprovenanced_literal_in_allowlist_modules (0.7s)
  - —
* - `stellar`
  - 20
  - 0
  - 0
  - 100.0
  - 41/122 audited, 0 haz
  - test_differentiable (2.2s)
  - —
* - `substructure`
  - 35
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_calibration_produces_valid_factors (35.7s)
  - —
* - `test_analytical_physics`
  - 0
  - 0
  - 28
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_eccentric_orbit_closes_and_conserves (2.2s)
  - —
* - `test_azimuthal_variation_physics`
  - 0
  - 0
  - 5
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_anticorrelates_with_cw04_Q (2.6s)
  - —
* - `test_binary_cluster`
  - 0
  - 11
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_totalmass_reaches_budget (18.8s)
  - —
* - `test_binary_physics`
  - 0
  - 0
  - 19
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_finite_through_e_to_one (2.6s)
  - —
* - `test_builders`
  - 14
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_build_with_plummer (4.4s)
  - —
* - `test_cluster_builders_integration`
  - 0
  - 7
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_through_build_cluster_from_params_bites_each_channel (8.3s)
  - —
* - `test_dashboard_fresh`
  - 0
  - 0
  - 3
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_committed_dashboard_matches_fresh_regeneration (144.0s)
  - —
* - `test_dashboard_gen`
  - 0
  - 0
  - 14
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_registries_full_flag_gates_on_built_and_full (142.8s)
  - —
* - `test_demo_oed_binary`
  - 5
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_deterministic_sweep_span_and_marg_costs_info (36.1s)
  - —
* - `test_dispersion_physics`
  - 0
  - 0
  - 24
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_projection_empirical_los_and_pm (60.3s)
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
  - test_eff_all_particles_bound (9.5s)
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
  - test_plummer_halo_eff_core_equilibrium (351.4s)
  - —
* - `test_environment_physics`
  - 0
  - 0
  - 15
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_slopes_steepen_with_metallicity (1.9s)
  - —
* - `test_grad_audit`
  - 0
  - 0
  - 100
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_gradient_audit[MultiComponentCluster.sample_cluster[EngineA]1] (154.9s)
  - —
* - `test_imf_physics`
  - 0
  - 0
  - 23
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_salpeter_mean_mass (4.6s)
  - —
* - `test_jax_compatibility`
  - 0
  - 6
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_powerlaw_vmap (3.0s)
  - —
* - `test_king_physics`
  - 0
  - 0
  - 35
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_sampled_core_mass_matches_dense_reference[7.0] (17.8s)
  - —
* - `test_limepy_reference_parity`
  - 0
  - 0
  - 3
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_parity_with_reference_limepy[twocomp_iso] (59.2s)
  - —
* - `test_mass_segregation_physics`
  - 0
  - 0
  - 9
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_estimator_converges_with_random_samples (113.1s)
  - —
* - `test_michie_physics`
  - 0
  - 0
  - 10
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_beta_matches_df_oracle (16.4s)
  - —
* - `test_multimass_equilibrium_physics`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_sampled_per_group_virial_converges_to_theory (43.9s)
  - —
* - `test_numerics`
  - 79
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_differentiable_in_weight_parameter (2.7s)
  - —
* - `test_plummer_physics`
  - 0
  - 0
  - 20
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_all_particles_bound (17.4s)
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
  - test_representative_node_id_resolves_and_asserts (6.7s)
  - —
* - `test_readme_examples`
  - 0
  - 9
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_readme_block_executes[2] (9.0s)
  - —
* - `test_rotation_anisotropy_physics`
  - 0
  - 0
  - 8
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_plummer_beta_matches_exact_om (12.1s)
  - —
* - `test_segregation_approx_physics`
  - 0
  - 0
  - 13
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_positions_finite (7.6s)
  - —
* - `test_segregation_equilibrium_physics`
  - 0
  - 0
  - 4
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_primordial_full_segregation_is_per_group_equilibrium (17.2s)
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
  - test_q_roughly_n_independent (8.4s)
  - —
* - `test_tidal`
  - 15
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_zero_mass_ghosts_are_inert_in_potential_energy (1.9s)
  - —
* - `test_tidal_physics`
  - 0
  - 0
  - 9
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_bound_mass_equals_mass_within_rt (2.2s)
  - —
* - `test_units_through_pipeline`
  - 0
  - 2
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_build_spatial_ic_respects_G_without_virial_rescale (3.1s)
  - —
* - `test_zams_physics`
  - 0
  - 0
  - 34
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_inverse_differentiable (3.6s)
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
