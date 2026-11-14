---
title: Test Dashboard
description: Generated display of stored progenax test, coverage, registry, duration, and validation artifacts, with a separate scoped release verdict.
---
# Test Dashboard

```{note}
**Recorded measurements.** PASS cells, coverage percentages and hazard counts display saved artifacts. See the [current release verdict](../95-release/index.md#release-readiness-verdict) for the revision, input-freshness checks and completed gate stages.
The dashboard generator does not set a full-suite verdict; the release page records executed test and gate results.

This page is generated. Display-only regeneration from the unchanged stored JSON is:

    uv run --no-sync python scripts/build_test_dashboard.py --render

Updating the census with `--emit` is separate. Neither operation reruns coverage, gradient checks, or validation scripts.
```

Census generated at `2026-11-14T03:26:29.233244+00:00`. Recorded combined statement/branch coverage: 96.6% (floor 90%). Coverage, durations and validation PASS entries come from stored artifacts; regenerating this page does not rerun those checks. Per-module `Line-cov %` cells show statement-only coverage, weighted by statement counts across each directory; they exclude branch coverage. They show **not recorded** when no value is present. A missing value does not imply a pending measurement for a test-only module. The gradient column reports the stored repo-wide differentiability registry (audited / audited+exempt, hazard count). Built registries: differentiability, API-coverage, physics-validation, provenance.

Coverage measurement recorded at `2026-11-14T03:26:29.183269+00:00` for commit `59b96f300e0784ca932bc13e6f0143494d50f724`. A commit stamp does not identify different dirty-tree inputs. See [testing architecture](testing-architecture.md) for current input-freshness requirements. The stored census and recorded measurements can have different dates.

```{list-table} Per-module test + coverage matrix
:header-rows: 1
:align: left

* - Module
  - Unit
  - Integration
  - Validation
  - Line-cov %
  - Recorded grad-audit fill
  - Slowest test
  - Recorded validation
* - `analytical`
  - 8
  - 0
  - 0
  - 93.2
  - 42/125 audited, 0 haz
  - test_analytic_metadata_retains_traced_periods_and_energies (5.6s)
  - PASS
* - `api_coverage`
  - 0
  - 0
  - 5
  - not recorded
  - 42/125 audited, 0 haz
  - —
  - —
* - `binaries`
  - 143
  - 0
  - 0
  - 94.4
  - 42/125 audited, 0 haz
  - test_Q_com_recovers_virial_target (81.6s)
  - PASS
* - `builders`
  - 100
  - 0
  - 0
  - 95.2
  - 42/125 audited, 0 haz
  - test_binary_imf_derivatives_preserve_mass_units[units0] (105.1s)
  - —
* - `cluster`
  - 153
  - 0
  - 0
  - 98.9
  - 42/125 audited, 0 haz
  - test_direct_per_component_anisotropy (179.0s)
  - —
* - `diagnostics`
  - 28
  - 0
  - 0
  - 94.6
  - 42/125 audited, 0 haz
  - test_missing_scipy_diagnostic_modules_give_actionable_error (36.0s)
  - —
* - `dynamics`
  - 28
  - 0
  - 0
  - 97.8
  - 42/125 audited, 0 haz
  - test_single_group_reproduces_global_virial (19.7s)
  - —
* - `grad_audit`
  - 0
  - 0
  - 26
  - not recorded
  - 42/125 audited, 0 haz
  - test_mean_radius_gradient_finite_on_zeros (10.7s)
  - —
* - `imf`
  - 300
  - 0
  - 0
  - 97.8
  - 42/125 audited, 0 haz
  - test_range_and_twin_fraction (42.0s)
  - —
* - `kinematics`
  - 176
  - 0
  - 0
  - 99.9
  - 42/125 audited, 0 haz
  - test_differentiable_in_g_through_table (160.1s)
  - —
* - `physics_registry`
  - 0
  - 0
  - 6
  - not recorded
  - 42/125 audited, 0 haz
  - test_every_cited_test_node_id_resolves (81.1s)
  - —
* - `profiles`
  - 280
  - 0
  - 0
  - 99.1
  - 42/125 audited, 0 haz
  - test_grad_ra_hat_vs_fd_quadrature (129.8s)
  - —
* - `provenance_cards`
  - 0
  - 0
  - 6
  - not recorded
  - 42/125 audited, 0 haz
  - test_validation_node_ids_resolve_and_assert (57.0s)
  - —
* - `provenance_registry`
  - 0
  - 0
  - 7
  - not recorded
  - 42/125 audited, 0 haz
  - test_no_new_unprovenanced_literal_in_allowlist_modules (0.7s)
  - —
* - `stellar`
  - 21
  - 0
  - 0
  - 100.0
  - 42/125 audited, 0 haz
  - test_inverse_zams_preserves_matrix_shape_and_inverse_derivatives (14.7s)
  - —
* - `substructure`
  - 40
  - 0
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_calibration_produces_valid_factors (36.4s)
  - —
* - `test_analytical_physics`
  - 0
  - 0
  - 28
  - not recorded
  - 42/125 audited, 0 haz
  - test_energy_conserved_over_period (7.9s)
  - —
* - `test_api_reference`
  - 14
  - 0
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_check_detects_staleness_without_writing (0.5s)
  - —
* - `test_azimuthal_variation_physics`
  - 0
  - 0
  - 5
  - not recorded
  - 42/125 audited, 0 haz
  - test_anticorrelates_with_cw04_Q (1.1s)
  - —
* - `test_binary_cluster`
  - 0
  - 19
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_legacy_equivalence_for_compact_targets[target2] (147.5s)
  - —
* - `test_binary_physics`
  - 0
  - 0
  - 19
  - not recorded
  - 42/125 audited, 0 haz
  - test_circular_orbit_mean_equals_eccentric (5.5s)
  - —
* - `test_builders`
  - 14
  - 0
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_build_with_plummer (7.9s)
  - —
* - `test_cluster_builders_integration`
  - 0
  - 7
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_grad_through_build_cluster_from_params_bites_each_channel (69.1s)
  - —
* - `test_dashboard_fresh`
  - 0
  - 0
  - 16
  - not recorded
  - 42/125 audited, 0 haz
  - —
  - —
* - `test_dashboard_gen`
  - 0
  - 0
  - 12
  - not recorded
  - 42/125 audited, 0 haz
  - test_inventory_has_modules_and_counts (223.4s)
  - —
* - `test_demo_oed_binary`
  - 5
  - 0
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_vbin_and_sigma_ratio_bites (39.5s)
  - —
* - `test_dispersion_physics`
  - 0
  - 0
  - 24
  - not recorded
  - 42/125 audited, 0 haz
  - test_projection_empirical_los_and_pm (83.9s)
  - —
* - `test_documented_api`
  - 16
  - 0
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - —
  - —
* - `test_eff_physics`
  - 0
  - 0
  - 24
  - not recorded
  - 42/125 audited, 0 haz
  - test_velocity_isotropy (34.2s)
  - —
* - `test_end_to_end`
  - 0
  - 9
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_chabrier_to_king_ic (29.1s)
  - —
* - `test_engine_a_ad_modes`
  - 0
  - 0
  - 2
  - not recorded
  - 42/125 audited, 0 haz
  - test_engine_a_forward_reverse_and_finite_difference[0.0] (48.3s)
  - —
* - `test_engine_a_radial_derivatives`
  - 0
  - 0
  - 1
  - not recorded
  - 42/125 audited, 0 haz
  - test_radial_derivatives_with_potential_boundaries (14.8s)
  - —
* - `test_engine_b_physics`
  - 0
  - 0
  - 6
  - not recorded
  - 42/125 audited, 0 haz
  - test_king_density_engine_b_matches_engine_a (107.0s)
  - —
* - `test_environment_physics`
  - 0
  - 0
  - 15
  - not recorded
  - 42/125 audited, 0 haz
  - test_slopes_steepen_with_metallicity (3.3s)
  - —
* - `test_grad_audit`
  - 0
  - 0
  - 101
  - not recorded
  - 42/125 audited, 0 haz
  - test_gradient_audit[MultiComponentCluster.from_mass_segregation[EngineA]] (296.2s)
  - —
* - `test_imf_physics`
  - 0
  - 0
  - 23
  - not recorded
  - 42/125 audited, 0 haz
  - test_massive_star_fraction_salpeter (22.6s)
  - —
* - `test_jax_compatibility`
  - 0
  - 6
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_powerlaw_vmap (13.2s)
  - —
* - `test_king_physics`
  - 0
  - 0
  - 35
  - not recorded
  - 42/125 audited, 0 haz
  - test_auto_domain_preserves_differentiability_high_W0 (60.0s)
  - —
* - `test_limepy_reference_adapter`
  - 12
  - 0
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - —
  - —
* - `test_limepy_reference_parity`
  - 0
  - 0
  - 6
  - not recorded
  - 42/125 audited, 0 haz
  - test_parity_with_reference_limepy[imf4] (78.2s)
  - —
* - `test_mass_segregation_physics`
  - 0
  - 0
  - 9
  - not recorded
  - 42/125 audited, 0 haz
  - test_estimator_converges_with_random_samples (17.5s)
  - —
* - `test_michie_physics`
  - 0
  - 0
  - 10
  - not recorded
  - 42/125 audited, 0 haz
  - test_beta_matches_df_oracle (42.6s)
  - —
* - `test_multimass_equilibrium_physics`
  - 0
  - 0
  - 6
  - not recorded
  - 42/125 audited, 0 haz
  - test_anisotropic_sampled_cluster_is_equilibrium_and_correctly_anisotropic (99.9s)
  - —
* - `test_multimass_figure_helpers`
  - 0
  - 0
  - 2
  - not recorded
  - 42/125 audited, 0 haz
  - test_half_mass_radius_uses_each_interval_width[radii1] (3.1s)
  - —
* - `test_numerics`
  - 79
  - 0
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_differentiable_in_weight_parameter (8.3s)
  - —
* - `test_plummer_physics`
  - 0
  - 0
  - 19
  - not recorded
  - 42/125 audited, 0 haz
  - test_central_velocity_dispersion (10.0s)
  - —
* - `test_protocols`
  - 14
  - 0
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_plummer_instances_conform (2.0s)
  - —
* - `test_public_api`
  - 2
  - 0
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - —
  - —
* - `test_ratchet_characterization`
  - 0
  - 0
  - 9
  - not recorded
  - 42/125 audited, 0 haz
  - test_representative_node_id_resolves_and_asserts (1.1s)
  - —
* - `test_readme_examples`
  - 0
  - 10
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_readme_block_executes[2] (53.1s)
  - —
* - `test_release_gate_status`
  - 0
  - 0
  - 13
  - not recorded
  - 42/125 audited, 0 haz
  - —
  - —
* - `test_rotation_anisotropy_physics`
  - 0
  - 0
  - 8
  - not recorded
  - 42/125 audited, 0 haz
  - test_plummer_beta_matches_exact_om (30.6s)
  - —
* - `test_segregation_approx_physics`
  - 0
  - 0
  - 13
  - not recorded
  - 42/125 audited, 0 haz
  - test_no_nan_on_degenerate_inputs (25.8s)
  - —
* - `test_segregation_equilibrium_physics`
  - 0
  - 0
  - 4
  - not recorded
  - 42/125 audited, 0 haz
  - test_primordial_full_segregation_is_per_group_equilibrium (48.8s)
  - —
* - `test_strict_refs`
  - 0
  - 0
  - 1
  - not recorded
  - 42/125 audited, 0 haz
  - —
  - —
* - `test_substructure_q_physics`
  - 0
  - 0
  - 8
  - not recorded
  - 42/125 audited, 0 haz
  - test_approx_preserves_ordering (2.8s)
  - —
* - `test_tidal`
  - 15
  - 0
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_forward_is_exact_hard_cut (7.1s)
  - —
* - `test_tidal_physics`
  - 0
  - 0
  - 9
  - not recorded
  - 42/125 audited, 0 haz
  - test_bound_mass_equals_mass_within_rt (6.0s)
  - —
* - `test_units_through_pipeline`
  - 0
  - 2
  - 0
  - not recorded
  - 42/125 audited, 0 haz
  - test_build_spatial_ic_respects_G_without_virial_rescale (15.2s)
  - —
* - `test_zams_physics`
  - 0
  - 0
  - 34
  - not recorded
  - 42/125 audited, 0 haz
  - test_round_trip_recovers_mass (8.2s)
  - —
```

## Validation scripts

Recorded exit status of the released-core validation scripts (from `validation/data/validation_runs.json`): **24 scripts: 24 passed, 0 failed, 0 incomplete**. Overall recorded status: **PASS**.
These are stored results; rendering this page does not execute the scripts. A missing, timeout, bool, float or otherwise invalid exit record is incomplete.

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
