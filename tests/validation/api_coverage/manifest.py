"""API-coverage source of truth (Phase 2 / Task 2.1).

Mirrors the grad-audit ``manifest.py`` frozen-literal pattern: THREE hand-curated,
independent dict literals (NOT computed from coverage or ``__all__`` at runtime — a
derived map cannot catch a deleted symbol or a fabricated mapping). Every one of the
114 ``progenax.__all__`` symbols lands in EXACTLY ONE of the three dicts.

  SYMBOL_TESTS : symbol -> the test that CONSTRUCTS/CALLS the symbol AND ASSERTS on its
                 output. The pointed-at test is the asserting witness, NOT a grep-mention
                 (a substring hit is coverage *theater*). Each entry below was verified by
                 OPENING the candidate test and confirming an `assert` derived from the
                 symbol's behavior; the node id is `file::Class::test` or `file::test`.
  EXEMPT       : symbol legitimately NOT directly exercised by an asserting test
                 (typing Protocols asserted via conformance, pure PyTree containers,
                 unit-system constants). Reason given per symbol. Mirrors the grad-audit
                 EXEMPT taxonomy where it applies.
  UNTESTED     : a PUBLIC symbol with NO asserting test found — a REAL hole, filled in
                 Task 2.3 with Anna's per-item approval. Honesty over coverage: an honest
                 UNTESTED is correct; a fabricated SYMBOL_TESTS entry is a failure.

Enforced by tests/validation/api_coverage/test_api_coverage.py. The website API-coverage
section is generated from here (Task 2.2).
"""

# --- SYMBOL_TESTS: symbol -> asserting test (file::Class::test | file::test) -------------
# Every entry verified by opening the test and confirming an assert on the symbol's output.
SYMBOL_TESTS: dict[str, str] = {
    # --- Spatial profiles (asserting on density / sampling physics) ---
    "PlummerProfile": "tests/validation/test_plummer_physics.py::TestPlummerDensityProfile::test_half_mass_radius_statistical",  # assert |frac_within_r_h - 0.5| < HALF_MASS on sample_positions output
    "KingProfile": "tests/validation/test_king_physics.py::TestKingDensityProfile::test_density_decreases_with_radius",  # assert sampled King density monotone-decreasing
    "MichieProfile": "tests/validation/test_michie_physics.py::TestMichieAnisotropyProfile::test_beta_matches_df_oracle",  # assert sampled beta(r) matches DF oracle
    "EFFProfile": "tests/validation/test_eff_physics.py::TestEFFDensityFormula::test_central_density_unity",  # assert EFF central density == 1
    "LIMEPYProfile": "tests/validation/test_limepy_reference_parity.py::test_parity_with_reference_limepy",  # assert LIMEPY density/potential matches the reference-LIMEPY oracle
    "UniformSphereProfile": "tests/unit/substructure/test_q_baselines.py::TestUniformSphereBaseline::test_q_matches_cw04_range",  # assert 0.75 < Q_mean < 0.90 on sample_positions output (CW04 uniform-sphere; sensitive to uniform-density correctness)
    "solve_king_profile": "tests/unit/profiles/test_king.py::TestDifferentiableTidalRadius::test_forward_r_t_value_pinned",  # assert ODE-solved King r_t pinned to a tight value
    "solve_michie_profile": "tests/unit/kinematics/test_michie_df.py::TestMichieSamplerOptimization::test_cached_table_bit_identical_to_fresh_build",  # assert_array_equal on solve_michie_profile psi output
    "solve_limepy_profile": "tests/unit/profiles/test_limepy.py::TestSolveLimepyProfile::test_g1_profile_matches_king_solver",  # assert_allclose g=1 LIMEPY potential vs King solver
    "solve_multimass_limepy": "tests/unit/profiles/test_limepy_multimass.py::TestMultiMassCoreDelta0::test_delta0_recovers_single_mass_potential",  # assert_allclose delta=0 multimass psi vs single-mass
    "find_alpha_for_masses": "tests/unit/profiles/test_limepy_multimass.py::TestEigenvalueSolve::test_realized_masses_match_targets",  # assert residual < 1e-3 on solved alpha_j
    # --- Velocity DFs (asserting on virial Q / isotropy / dispersion) ---
    "PlummerVelocityDF": "tests/validation/test_plummer_physics.py::TestPlummerVirialEquilibrium::test_virial_ratio",  # assert Q = T/|V| ~ 0.5 on sample_velocities output
    "KingVelocityDF": "tests/validation/test_king_physics.py::TestKingEquilibriumVelocityDF::test_virial_ratio_is_half_unscaled",  # assert unscaled King DF Q ~ 0.5
    "MichieVelocityDF": "tests/validation/test_michie_physics.py::TestMichieEquilibrium::test_virial_ratio_half_unscaled",  # assert unscaled Michie DF Q ~ 0.5
    "EFFVelocityDF": "tests/validation/test_eff_physics.py::TestEFFVelocityDF::test_eff_eddington_virial_ratio_mild_truncation",  # assert EFF Eddington-DF Q ~ 0.5
    "LIMEPYVelocityDF": "tests/unit/builders/test_cluster_builders.py::test_matched_limepy_isotropic_passes_none_r_a",  # constructs LIMEPYVelocityDF, assert isinstance + isotropic r_a non-finite
    # --- Kinematic transforms / rotation overlays ---
    "apply_solid_body_rotation": "tests/unit/kinematics/test_rotation.py::TestSolidBodyRotation::test_adds_rotation",  # assert |v| == omega*R after overlay
    "apply_differential_rotation": "tests/unit/kinematics/test_rotation.py::TestDifferentialRotation::test_peak_at_R_peak",  # assert |v| == v_peak at R_peak
    "sample_isotropic_orientations": "tests/unit/binaries/test_population.py::TestIsotropicOrientations::test_inclination_isotropic",  # assert <cos i> ~ 0 on sampled inclinations
    # --- Dispersion forward models (jeans / B&M82 projection) ---
    "jeans_dispersion": "tests/unit/kinematics/test_dispersion.py::test_plummer_isotropic_closed_form",  # assert dp.sigma_1d == sqrt(GM/(6 sqrt(r^2+a^2))) (rtol 3e-3) + beta==0 + sigma_r==sigma_t on jeans_dispersion output
    "project_dispersion": "tests/validation/test_dispersion_physics.py::test_projection_isotropic_all_equal",  # assert pj.sigma_los == sigma_pm_r == sigma_pm_t (beta=0 B&M82 kernel collapse) on project_dispersion output
    "df_moment_dispersion": "tests/unit/kinematics/test_dispersion.py::test_df_moment_export_and_shapes",  # assert df_moment_dispersion exported + returns DispersionProfile with correct (sigma_r/sigma_t/sigma_1d/beta) shapes
    # --- IMFs ---
    "PowerLawIMF": "tests/unit/imf/test_imf_core.py::TestCDFProperties::test_cdf_at_m_min",  # assert |cdf(m_min)| < 1e-6
    "ChabrierIMF": "tests/unit/imf/test_imf_core.py::TestChabrierSpecific::test_A_pl_continuity",  # assert lognormal/power-law continuity at m_c
    "Maschberger": "tests/unit/imf/test_sampling_modes.py::TestSampleFixedN::test_reachable_target_is_hit",  # assert sampled total mass hits target
    "TruncatedIMF": "tests/unit/imf/test_imf_core.py::TestCDFProperties::test_cdf_at_m_max",  # assert |cdf(m_max) - 1| < 1e-6
    "BinaryIMF": "tests/unit/imf/test_binary.py::TestBinaryIMF::test_binary_fraction_matches_target",  # assert realized binary fraction ~ target
    # --- Mass-ratio distributions ---
    "FlatMassRatio": "tests/unit/imf/test_binary.py::TestMassRatioDistributions::test_flat_pdf_normalization",  # assert pdf integral ~ 1
    "PowerLawMassRatio": "tests/unit/imf/test_binary.py::TestPowerLawMassRatioArrayBranches::test_cdf_array_monotonic_and_boundaries",  # assert cdf monotone + boundaries
    "TwinPeakedMassRatio": "tests/unit/imf/test_binary.py::TestTwinPeakedMassRatioArrayBranches::test_cdf_array_monotonic_and_boundaries",  # assert cdf monotone + boundaries
    "MoeDiStefano2017": "tests/unit/imf/test_binary.py::TestMoeDiStefano2017::test_gamma_varies_with_mass",  # assert gamma_low > gamma_solar > gamma_massive
    "MoeDiStefano2017Full": "tests/unit/imf/test_moe_full.py::TestTable13Interpolation::test_gamma_largeq_cells",  # assert_allclose gamma_largeq vs Moe Table 13
    "MoePeriod": "tests/unit/imf/test_moe_full.py::TestMoePeriod::test_range",  # assert sampled log P within [0.2, 8.0]
    "MoeJointOrbit": "tests/unit/imf/test_moe_full.py::TestMoeJointOrbit::test_shapes_and_ranges",  # assert joint (P,q,e) shapes/ranges
    "MoeEccentricity": "tests/unit/binaries/test_population.py::TestFaithfulMoeEccentricity::test_eta_eq17_late_type",  # assert_allclose eta(Eq.17) late-type
    # --- Binary-fraction models ---
    "ConstantBinaryFraction": "tests/unit/imf/test_binary.py::TestBinaryFractionModels::test_constant_returns_constant",  # assert model(masses) == const
    "MassDependentBinaryFraction": "tests/unit/imf/test_binary.py::TestBinaryFractionModels::test_mass_dependent_increases_with_mass",  # assert f_low < f_solar < f_massive
    "RadialBinaryFraction": "tests/unit/binaries/test_population.py::TestRadialBinaryFraction::test_A_positive_core_enhanced",  # assert fb(center) > fb(outer)
    "CombinedBinaryFraction": "tests/unit/binaries/test_fraction_unification.py::TestCombinedBinaryFraction::test_product_clipped",  # assert_allclose combined probability == expected
    # --- Period distributions ---
    "LogNormalPeriod": "tests/unit/binaries/test_population.py::TestLogNormalPeriod::test_mean_log_period",  # assert <log P> ~ 4.0
    "LogUniformPeriod": "tests/unit/binaries/test_population.py::TestLogUniformPeriod::test_log_uniform_distribution",  # assert <log P> ~ 4.0
    "SanaOBPeriod": "tests/unit/binaries/test_population.py::TestSanaOBPeriod::test_mean_shorter_than_solar_type",  # assert mean_sana < mean_solar
    # --- Eccentricity distributions ---
    "ThermalEccentricity": "tests/unit/binaries/test_population.py::TestThermalEccentricity::test_thermal_mean",  # assert <e> ~ thermal expectation
    "UniformEccentricity": "tests/unit/binaries/test_population.py::TestUniformEccentricity::test_uniform_mean",  # assert <e> ~ 0.5
    "LogisticThermalEccentricity": "tests/unit/binaries/test_population.py::TestLogisticThermalEccentricity::test_short_periods_more_circular",  # assert <e>(short P) < <e>(long P)
    # --- Kepler / orbital state ---
    "KeplerElements": "tests/unit/binaries/test_binaries.py::TestKeplerElements::test_circular_orbit_creation",  # assert elements.a == approx(1.0)
    "BinaryOrbitalState": "tests/unit/binaries/test_binaries.py::TestBinaryOrbitalState::test_from_log_period",  # assert state.m1 == approx(1.0)
    "CartesianState": "tests/unit/binaries/test_binaries.py::TestKeplerElements::test_to_state",  # to_state() returns CartesianState; assert state.position/.velocity shape == (3,)
    "BinaryState": "tests/unit/binaries/test_binaries.py::TestKeplerElements::test_to_binary_state",  # to_binary_state() returns BinaryState; assert COM=(m1*r1+m2*r2)/M ~ 0
    "compute_period": "tests/unit/binaries/test_binaries.py::TestPeriodFunctions::test_compute_period",  # assert |T - Kepler-III| < 0.01
    "period_to_semimajor_axis": "tests/unit/binaries/test_binaries.py::TestPeriodFunctions::test_period_to_semimajor_axis",  # assert round-trip P -> a -> P recovers T
    # --- Binary connector / companions ---
    "resolve_binary_components": "tests/unit/binaries/test_assembly.py::TestResolveBinaryComponents::test_com_conserved_per_binary",  # assert per-binary COM conserved to 1e-12
    "ResolvedBinaries": "tests/unit/binaries/test_assembly.py::TestResolveBinaryComponents::test_output_shape_2N",  # assert rb.positions.shape == (2N, 3)
    "CompanionElements": "tests/unit/binaries/test_companions.py::TestCompanionElements::test_namedtuple_fields",  # assert el.m2.shape / el.a.shape
    "IndependentCompanions": "tests/unit/binaries/test_companions.py::TestIndependentCompanions::test_shapes_and_singles",  # assert singles have m2 == 0
    "MoeCompanions": "tests/unit/binaries/test_companions.py::TestMoeCompanions::test_shapes_singles_ranges",  # assert singles have m2 == 0 + ranges
    "batch_elements_to_resolved": "tests/unit/binaries/test_binaries.py::TestBatchOperations::test_batch_elements_to_resolved",  # assert resolved r1.shape == (N, 3)
    "sample_mass_dependent_orbits": "tests/unit/binaries/test_population.py::TestMassDependentOrbits::test_routes_by_mass",  # assert median period routes by mass
    "MassDependentBinaryConfig": "tests/unit/binaries/test_population.py::TestMassDependentOrbits::test_routes_by_mass",  # config drives mass-routed orbit sampling; assert <m2/m1> low > high
    # --- Binary diagnostics + energy budget ---
    "relative_energy": "tests/unit/binaries/test_diagnostics.py::TestRelativeEnergy::test_circular_binary_energy",  # assert |E - E_vis_viva|/|E| < 1e-6
    "find_bound_pairs": "tests/unit/binaries/test_diagnostics.py::TestFindBoundPairs::test_recovers_primordial_hard_binaries",  # assert pairs.shape[0] == 40
    "find_bound_multiples": "tests/unit/binaries/test_diagnostics.py::TestFindBoundMultiples::test_isolated_binaries_multiplicity_2",  # assert all multiplicity == 2
    "primordial_survival": "tests/unit/binaries/test_diagnostics.py::TestPrimordialSurvival::test_all_survive_at_t0",  # assert survived == 30
    "binary_energy_budget": "tests/unit/binaries/test_energy_budget.py::TestEInternal::test_vis_viva_single_binary",  # assert |E_internal - vis-viva|/|E| < 1e-9
    "BinaryEnergyBudget": "tests/unit/binaries/test_energy_budget.py::TestEInternal::test_vis_viva_single_binary",  # budget container from binary_energy_budget; assert b.n_binaries == 1
    # --- Builders / cluster assembly ---
    "build_spatial_ic": "tests/integration/test_end_to_end.py::TestIMFToICPipeline::test_kroupa_to_plummer_ic",  # build_spatial_ic IMF->Plummer; assert positions/velocities shape + masses > 0
    "build_cluster": "tests/unit/builders/test_cluster_builders.py::test_build_cluster_is_bit_identical_to_manual_base_case",  # assert build_cluster IC bit-identical to manual base case
    "build_cluster_from_params": "tests/unit/builders/test_cluster_builders.py::test_cluster_params_wrapper_identical_to_build_cluster",  # assert ClusterParams-wrapper IC == build_cluster IC
    "build_plummer_cluster": "tests/unit/builders/test_cluster_builders.py::test_plummer_alias_identical",  # assert plummer alias IC == build_cluster IC
    "build_king_cluster": "tests/unit/builders/test_cluster_builders.py::test_king_alias_identical",  # assert king alias IC == build_cluster IC
    "build_eff_cluster": "tests/unit/builders/test_cluster_builders.py::test_eff_alias_identical",  # assert eff alias IC == build_cluster IC
    "build_michie_cluster": "tests/unit/builders/test_cluster_builders.py::test_michie_alias_identical",  # assert michie alias IC == build_cluster IC
    "build_limepy_cluster": "tests/unit/builders/test_cluster_builders.py::test_limepy_alias_identical",  # assert limepy alias IC == build_cluster IC
    "build_binary_cluster": "tests/integration/test_binary_cluster.py::TestBuildBinaryCluster::test_count_and_provenance",  # build_binary_cluster; assert particles == systems + secondaries, masses > 0
    "matched_velocity_df": "tests/unit/builders/test_cluster_builders.py::test_matched_plummer_scale_matched",  # assert matched DF type + scale matches profile
    "ClusterParams": "tests/unit/builders/test_cluster_builders.py::test_cluster_params_wrapper_identical_to_build_cluster",  # ClusterParams drives the wrapper; assert wrapper IC == build_cluster IC
    "RotationSpec": "tests/unit/builders/test_cluster_builders.py::test_rotation_spec_solid_matches_float",  # RotationSpec(omega) drives overlay; assert IC == float-omega IC
    "MultiComponentCluster": "tests/unit/cluster/test_multicomponent.py::TestFromComponents::test_unit_w_recovers_single_mass_density",  # assert_allclose multi-component density recovers single-mass
    # --- IC result containers (asserted via type + field) ---
    "ICResult": "tests/unit/cluster/test_multicomponent.py::TestSampleClusterICResult::test_icresult_fields_and_component_id",  # assert isinstance(ic, ICResult) + field shapes on sample_cluster output
    "Systems": "tests/unit/binaries/test_target_budget.py::TestTargetTypes::test_construct_and_fields",  # assert Systems(10).n == 10
    "Stars": "tests/unit/binaries/test_target_budget.py::TestTargetTypes::test_construct_and_fields",  # assert Stars(500).n == 500
    "TotalMass": "tests/unit/binaries/test_target_budget.py::TestTargetTypes::test_construct_and_fields",  # assert TotalMass(1e4).m == 1e4
    # --- Energy / kinematic / segregation utilities ---
    "compute_kinetic_energy": "tests/unit/test_builders.py::TestVirialScale::test_virial_ratio_is_target",  # T from compute_kinetic_energy enters Q; assert |Q - 0.5| < 0.01
    "compute_potential_energy": "tests/unit/test_builders.py::TestVirialScale::test_virial_ratio_is_target",  # V from compute_potential_energy enters Q; assert |Q - 0.5| < 0.01
    "to_com_frame": "tests/unit/test_builders.py::TestToCOMFrame::test_com_is_zero_after_transform",  # assert COM ~ 0 after transform
    "virial_scale": "tests/unit/test_builders.py::TestVirialScale::test_virial_ratio_is_target",  # assert |Q - 0.5| < 0.01 after virial_scale
    "compute_stellar_radii": "tests/unit/test_builders.py::TestComputeStellarRadii::test_solar",  # assert |R(1 Msun) - 1.06| < 0.01
    "energy_sorted_segregation": "tests/unit/cluster/test_mass_segregation.py::TestEnergySortedSegregation::test_no_orbit_reuse_for_any_mass_spectrum",  # assert n_unique == N (no orbit reuse)
    # --- ZAMS stellar relations (Tout+1996; asserting on published solar anchors / round-trip) ---
    "zams_luminosity": "tests/unit/stellar/test_zams.py::TestZAMSLuminosity::test_sun_anchor",  # assert zams_luminosity(1 Msun) ~ 0.698 Lsun (Tout+1996 Sun anchor)
    "zams_radius": "tests/unit/stellar/test_zams.py::TestZAMSRadius::test_sun_anchor",  # assert zams_radius(1 Msun) ~ 0.888 Rsun (Tout+1996 Sun anchor)
    "zams_effective_temperature": "tests/unit/stellar/test_zams.py::TestZAMSEffectiveTemperature::test_sun_anchor",  # assert zams_effective_temperature(1 Msun) ~ 5600 K (Stefan-Boltzmann from verified L,R)
    "zams_surface_gravity": "tests/unit/stellar/test_zams.py::TestZAMSSurfaceGravity::test_sun_anchor",  # assert zams_surface_gravity(1 Msun) ~ 4.54 dex (log10 G M / R^2)
    "inverse_zams_luminosity": "tests/unit/stellar/test_zams.py::TestInverseZAMSLuminosity::test_round_trip",  # assert inverse_zams_luminosity(zams_luminosity(m)) ~ m (rtol 1e-5 over M in [0.5,20])
    # --- Tidal physics ---
    "jacobi_radius": "tests/unit/test_tidal.py::TestJacobiRadius::test_jacobi_radius_formula",  # assert r_J^3 == R^3 M_c/(3 M_g)
    "jacobi_radius_isothermal": "tests/unit/test_tidal.py::TestJacobiRadiusIsothermal::test_satisfies_defining_relation",  # assert r_J^3 == G M R^2/(2 V^2)
    "apply_tidal_truncation": "tests/unit/test_tidal.py::TestTidalTruncation::test_forward_is_exact_hard_cut",  # assert masses beyond r_t == 0
    "fill_factor_to_r_h": "tests/unit/test_tidal.py::TestFillFactor::test_formula",  # assert fill_factor_to_r_h(0.2, 10) == 2.0
    # --- Analytical ICs / solar system (asserting on output) ---
    "two_body_kepler": "tests/unit/analytical/test_analytical.py::TestTwoBodyKepler::test_circular_orbit_com_at_origin",  # assert COM ~ 0
    "two_body_period": "tests/unit/analytical/test_analytical.py::TestTwoBodyKepler::test_period_circular_orbit",  # assert |T - expected| < 0.01
    "two_body_energy": "tests/integration/test_end_to_end.py::TestAnalyticalValidation::test_two_body_energy_conservation",  # assert |E - E_analytical|/|E| < 0.01
    "three_body_figure_eight": "tests/unit/analytical/test_analytical.py::TestThreeBodyFigureEight::test_figure_eight_com_zero",  # assert figure-eight COM ~ 0
    "solar_system_inner_4": "tests/validation/test_analytical_physics.py::TestSolarSystemPhysics::test_inner4_shares_table",  # assert_allclose inner-4 masses vs SOLAR_SYSTEM_PLANETS table
    "solar_system_full": "tests/validation/test_analytical_physics.py::TestSolarSystemPhysics::test_barycentric_and_finite",  # assert full-system mass shape + barycentric/finite
    "get_planet": "tests/unit/analytical/test_analytical.py::TestSolarSystemData::test_get_planet_earth",  # assert get_planet('earth')['a'] == approx(1.0)
    "SOLAR_SYSTEM_PLANETS": "tests/unit/analytical/test_analytical.py::TestSolarSystemData::test_solar_system_planets_count",  # assert len(SOLAR_SYSTEM_PLANETS) == 8
    "earth_sun_2body": "tests/validation/test_analytical_physics.py::TestEarthSunTwoBody::test_bound_energy_matches_vis_viva",  # assert E < 0 and E == -G M1 M2/2a (bound vis-viva); class also asserts COM=0, mass, Kepler-III period ~1yr
    "earth_sun_eccentric": "tests/validation/test_analytical_physics.py::TestEarthSunEccentric::test_recovers_specified_eccentricity",  # assert LRL |e| == 0.0167 (factory's specified e) + vis-viva a==1; class also asserts bound + barycentric
    "sun_earth_jupiter_3body": "tests/validation/test_analytical_physics.py::TestSunEarthJupiterThreeBody::test_bound_and_finite",  # assert 3-body bound (E<0) + finite; class also asserts total mass = M_sun+M_earth+M_jup, COM=0
    "harmonic_oscillator": "tests/validation/test_analytical_physics.py::TestHarmonicOscillator::test_energy_is_sho_constant",  # assert E = 1/2 m v^2 + 1/2 m w^2 x^2 == 1/2 m w^2 A^2 (SHO energy); class also pins x0/v0/period
    "harmonic_solution": "tests/validation/test_analytical_physics.py::TestHarmonicSolution::test_satisfies_sho_ode",  # assert closed-form x(t) satisfies xddot = -w^2 x (central diff) at several t; sibling pins x(0)/v(0) to oscillator IC
    "figure_eight_period": "tests/validation/test_analytical_physics.py::TestFigureEightPeriod::test_default_is_chenciner_montgomery_constant",  # assert default period == 6.32591398 (Chenciner-Montgomery-Simo); class also asserts factory-consistency + scaling law
    # --- Protocols (asserted via runtime conformance: isinstance/issubclass) ---
    "SpatialProfile": "tests/unit/test_protocols.py::test_spatial_profiles_conform",  # assert issubclass(Plummer/King/EFF, SpatialProfile)
    "VelocityDF": "tests/unit/test_protocols.py::test_velocity_dfs_conform",  # assert issubclass(DFs, VelocityDF)
    "IMFProtocol": "tests/unit/test_protocols.py::test_imfs_conform",  # assert isinstance(imfs, IMFProtocol)
    "PeriodDistribution": "tests/unit/binaries/test_population.py::TestDistributionProtocols::test_period_distributions_conform",  # assert isinstance(period dists, PeriodDistribution)
    "EccentricityDistribution": "tests/unit/binaries/test_population.py::TestDistributionProtocols::test_unconditional_eccentricity_conform",  # assert isinstance(Thermal/Uniform, EccentricityDistribution)
    "ConditionalEccentricityDistribution": "tests/unit/binaries/test_population.py::TestDistributionProtocols::test_conditional_eccentricity_protocols",  # assert isinstance(LogisticThermal, ConditionalEccentricityDistribution)
    "MassPeriodEccentricityDistribution": "tests/unit/binaries/test_population.py::TestDistributionProtocols::test_conditional_eccentricity_protocols",  # assert isinstance(MoeEccentricity, MassPeriodEccentricityDistribution)
    "BinaryFractionModel": "tests/unit/binaries/test_fraction_unification.py::TestBinaryFractionProtocol::test_all_models_conform",  # assert isinstance(models, BinaryFractionModel)
    "CompanionModel": "tests/unit/binaries/test_companions.py::TestProtocolConformance::test_both_are_companion_models",  # assert isinstance(Independent/Moe, CompanionModel)
}

# --- EXEMPT: symbol -> reason (legitimately not directly exercised by an asserting test) -
# Pure PyTree/data containers (no behavior to assert beyond construction, which the
# producing-symbol tests already cover) and unit-system constants. Mirrors the grad-audit
# EXEMPT_CONTAINER taxonomy. Anna approves each.
EXEMPT: dict[str, str] = {
    "AnalyticalIC": "pure eqx.Module container (base class for analytical ICs); holds "
    "positions/velocities/masses fields, no standalone behavior to assert. "
    "Mirrors grad-audit EXEMPT_CONTAINER. Its concrete IC factories are the "
    "behavior-bearing symbols (several are UNTESTED holes below).",
    "DEFAULT_UNITS": "module-level UnitSystem constant (carries G for convenience wrappers); "
    "not a callable/sampler. asserted-as-identity in test_cluster_builders.py "
    "(`assert DEFAULT_UNITS is STELLAR`) but it is a constant, not a behavior. "
    "Mirrors grad-audit EXEMPT_CONTAINER.",
}

# --- UNTESTED: symbol -> hole note (REAL holes; filled in Task 2.3 with Anna's approval) --
# Honest holes: NO test constructs the symbol and asserts on its output. Existence guards
# (`hasattr(progenax, name)` in test_public_api.py) do NOT count — they assert importability,
# not behavior (exactly the coverage-theater the C2 rule rejects).
#
# CLOSED in Task 2.3 (Anna-approved): the six analytical-IC factories/helpers
# (earth_sun_2body, earth_sun_eccentric, sun_earth_jupiter_3body, harmonic_oscillator,
# harmonic_solution, figure_eight_period) now each have an asserting physics test in
# tests/validation/test_analytical_physics.py and were moved to SYMBOL_TESTS. Zero holes
# remain, so test_no_untested_holes is now a HARD assertion.
UNTESTED: dict[str, str] = {}

# Line-coverage floor (ratchet-up-only). Enforced by test_line_coverage_above_floor against
# the COMMITTED full-suite validation/data/coverage.json (Task 2.2). SKIPs until that file is
# committed (Task 2.2's full --cov run).
LINE_COV_FLOOR = 90.0

# The selector value the committed coverage.json must carry (full suite, NOT `-m "not slow"`,
# which understates coverage and would spuriously fail the floor).
FULL_SELECTOR = "full"
