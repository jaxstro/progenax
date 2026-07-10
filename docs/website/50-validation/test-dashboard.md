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

Generated at `2026-07-10T20:31:34.069543+00:00`. Line coverage: 96.1% (floor 90%). `Line-cov %` cells read **pending (Phase 2)** until the committed full-suite `coverage.json` exists, then show the statement-weighted per-directory coverage. The `Grad-audit fill` column reports the repo-wide differentiability registry (audited / audited+exempt, hazard count). Built registries: differentiability, API-coverage, physics-validation, provenance.

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
  - test_circular_orbit_com_at_origin (0.4s)
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
  - 128
  - 0
  - 0
  - 95.8
  - 41/122 audited, 0 haz
  - test_p_q_interrelation_in_secondary_masses (16.6s)
  - PASS
* - `builders`
  - 43
  - 0
  - 0
  - 93.7
  - 41/122 audited, 0 haz
  - test_matched_limepy_anisotropic_threads_r_a (15.0s)
  - —
* - `cluster`
  - 65
  - 0
  - 0
  - 98.8
  - 41/122 audited, 0 haz
  - test_engine_b_global_virial_is_half_unscaled (70.5s)
  - —
* - `diagnostics`
  - 28
  - 0
  - 0
  - 92.4
  - 41/122 audited, 0 haz
  - test_returns_expected_keys (5.8s)
  - —
* - `dynamics`
  - 28
  - 0
  - 0
  - 97.8
  - 41/122 audited, 0 haz
  - test_single_group_reproduces_global_virial (3.9s)
  - —
* - `grad_audit`
  - 0
  - 0
  - 26
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_mean_radius_gradient_finite_on_zeros (0.7s)
  - —
* - `imf`
  - 269
  - 0
  - 0
  - 97.1
  - 41/122 audited, 0 haz
  - test_p_q_interrelation (16.4s)
  - —
* - `kinematics`
  - 138
  - 0
  - 0
  - 99.9
  - 41/122 audited, 0 haz
  - test_differentiable_in_g_through_table (23.2s)
  - —
* - `physics_registry`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_every_cited_test_node_id_resolves (20.2s)
  - —
* - `profiles`
  - 205
  - 0
  - 0
  - 98.9
  - 41/122 audited, 0 haz
  - test_jit_grad_matches_eager_alpha_imf (63.7s)
  - —
* - `provenance_registry`
  - 0
  - 0
  - 7
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_no_new_unprovenanced_literal_in_allowlist_modules (1.2s)
  - —
* - `stellar`
  - 25
  - 0
  - 0
  - 100.0
  - 41/122 audited, 0 haz
  - test_differentiable (2.7s)
  - —
* - `substructure`
  - 34
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_scales_to_large_n (2.4s)
  - —
* - `test_analytical_physics`
  - 0
  - 0
  - 28
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_eccentric_orbit_closes_and_conserves (1.8s)
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
  - test_totalmass_reaches_budget (11.6s)
  - —
* - `test_binary_physics`
  - 0
  - 0
  - 19
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_finite_through_e_to_one (1.2s)
  - —
* - `test_builders`
  - 14
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_build_with_plummer (2.0s)
  - —
* - `test_cluster_builders_integration`
  - 0
  - 7
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_through_build_cluster_from_params_bites_each_channel (6.6s)
  - —
* - `test_dashboard_fresh`
  - 0
  - 0
  - 3
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_coverage_provenance_teeth_fire_on_stale_src (1.3s)
  - —
* - `test_dashboard_gen`
  - 0
  - 0
  - 14
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_build_dashboard_has_all_blocks (95.0s)
  - —
* - `test_demo_oed`
  - 13
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_criteria_grads_AD_vs_FD (3.0s)
  - —
* - `test_demo_oed_binary`
  - 36
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_cross_model_bias_only_fits_kept_bins (25.1s)
  - —
* - `test_demo_oed_concentration`
  - 13
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_jacobian_lntheta_shape_and_W0_column_nonzero (95.7s)
  - —
* - `test_demo_oed_depth`
  - 9
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_cli_dynamical_mass_smoke (240.3s)
  - —
* - `test_demo_selection`
  - 4
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_apparent_mag_and_distance_modulus (0.3s)
  - —
* - `test_dispersion_physics`
  - 0
  - 0
  - 24
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_eff_isotropic_jeans_equals_ftable (11.2s)
  - —
* - `test_documented_api`
  - 16
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_symbol_importable_from_root[ConstantBinaryFraction] (0.0s)
  - —
* - `test_eff_physics`
  - 0
  - 0
  - 24
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_gamma3_default_subvirial_offset_is_pinned (4.7s)
  - —
* - `test_end_to_end`
  - 0
  - 9
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_chabrier_to_king_ic (4.9s)
  - —
* - `test_engine_b_physics`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_gradients_ad_vs_fd (13.1s)
  - —
* - `test_environment_physics`
  - 0
  - 0
  - 15
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_corrected_marks_equals_jerabkova (0.4s)
  - —
* - `test_grad_audit`
  - 0
  - 0
  - 100
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_gradient_audit[MultiComponentCluster.sample_cluster[EngineA]1] (100.5s)
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
  - test_powerlaw_grad (1.8s)
  - —
* - `test_king_physics`
  - 0
  - 0
  - 35
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_auto_domain_preserves_differentiability_high_W0 (6.8s)
  - —
* - `test_limepy_reference_parity`
  - 0
  - 0
  - 3
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - —
  - —
* - `test_mass_segregation_physics`
  - 0
  - 0
  - 9
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_estimator_converges_with_random_samples (49.1s)
  - —
* - `test_michie_physics`
  - 0
  - 0
  - 10
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_wrt_mass_velocity_scale (6.1s)
  - —
* - `test_multimass_equilibrium_physics`
  - 0
  - 0
  - 6
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - —
  - —
* - `test_numerics`
  - 8
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_differentiable_in_weight_parameter (1.1s)
  - —
* - `test_plummer_physics`
  - 0
  - 0
  - 20
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_radial_dispersion_profile (1.1s)
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
  - test_representative_node_id_resolves_and_asserts (5.3s)
  - —
* - `test_readme_examples`
  - 0
  - 9
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_readme_block_executes[2] (9.2s)
  - —
* - `test_rotation_anisotropy_physics`
  - 0
  - 0
  - 8
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_plummer_beta_matches_exact_om (6.2s)
  - —
* - `test_segregation_approx_physics`
  - 0
  - 0
  - 13
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_grad_positions_finite (7.5s)
  - —
* - `test_segregation_equilibrium_physics`
  - 0
  - 0
  - 4
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_energy_finite_and_no_coincident_stars (9.8s)
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
  - test_q_roughly_n_independent (5.6s)
  - —
* - `test_tidal`
  - 15
  - 0
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_forward_is_exact_hard_cut (0.8s)
  - —
* - `test_tidal_physics`
  - 0
  - 0
  - 9
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_bound_mass_equals_mass_within_rt (1.5s)
  - —
* - `test_units_through_pipeline`
  - 0
  - 2
  - 0
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_build_spatial_ic_respects_G_without_virial_rescale (4.5s)
  - —
* - `test_zams_physics`
  - 0
  - 0
  - 34
  - pending (Phase 2)
  - 41/122 audited, 0 haz
  - test_finite_across_Z_box (1.2s)
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
