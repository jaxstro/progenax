"""Physics-validation source of truth (Phase 4 / Task 4.1).

Mirrors the grad-audit / api-coverage frozen-literal pattern: FOUR hand-curated,
independent dict literals (NOT computed from the validation tests at runtime — a
derived map cannot catch a deleted model or a fabricated invariant). Every one of the
125 ``progenax.__all__`` symbols lands in EXACTLY ONE of the four dicts.

  MODEL_INVARIANTS : model -> {invariant phrase -> the validation test that ASSERTS it}.
                     "Model" = a profile / velocity-DF / IMF / cluster builder /
                     equilibrium engine whose correctness IS a physics claim. Each
                     invariant points at a test (node id ``file::Class::test`` or
                     ``file::test``) verified by OPENING it and confirming an ``assert``
                     derived from the named physics — NOT a grep-mention (substring hits
                     are coverage *theater*; the C2 anti-theater rule). The asserting
                     witness for 2-3 entries is quoted inline as proof of verification.
  EXEMPT_NON_MODEL : non-model __all__ symbols (utilities, containers, distributions,
                     factories, helpers, analytical ICs) — no physics-equilibrium
                     invariant is required of them. Reason per symbol. Mirrors the
                     grad-audit EXEMPT_CONTAINER / EXEMPT_HELPER / EXEMPT_ANALYTICAL_IC /
                     EXEMPT_NON_FISHER_DIAGNOSTIC taxonomy.
  EXEMPT_NON_EQUILIBRIUM_MODEL : models whose physics is REFERENCE-PARITY (LIMEPY vs
                     Gieles & Zocchi 2015) or UNIFORM-DENSITY RECOVERY (UniformSphere vs
                     CW04 Q-baseline) rather than equilibrium-Q — a documented carve so
                     the exclusion from the equilibrium invariant class is AUDITABLE, not
                     silent. The cited test is still a real, asserting physics test.
  UNTESTED_MODELS  : an OPERATIONAL model with NO enumerated/asserting physics invariant
                     today — a REAL hole (a Task-4.2 item for Anna). Honesty over
                     coverage: an honest hole is correct; a fabricated MODEL_INVARIANTS
                     entry is a failure. EMPTY as of Task 4.1 (every operational model
                     has at least one asserting validation/physics test).

Enforced by tests/validation/physics_registry/test_physics_coverage.py. The operational
IS_MODEL definition (issubclass vs the runtime-checkable protocols / ``build_*_cluster``)
lives in the test, NOT here — the test computes it so a new real model with no manifest
entry reds CI (the ratchet). The website physics section is generated from here (Task 4.2).
"""

# --- MODEL_INVARIANTS: model -> {invariant -> asserting validation test} ----------------
# Every entry verified by opening the cited test and confirming an assert on the invariant.
MODEL_INVARIANTS: dict[str, dict[str, str]] = {
    # =========================== Spatial profiles ====================================
    "PlummerProfile": {
        "scale-radius formula a = r_h*sqrt(2^(2/3)-1)": "tests/validation/test_plummer_physics.py::TestPlummerScaleRadius::test_scale_radius_formula_exact",
        # QUOTED WITNESS: assert jnp.allclose(profile.a, r_h * SCALE_RADIUS_FACTOR, rtol=1e-6)
        "density closed-form / inverse-CDF M(<r)/M = r^3/(r^2+a^2)^{3/2}": "tests/validation/test_plummer_physics.py::TestPlummerDensityProfile::test_cdf_formula_accuracy",
        # QUOTED WITNESS: for each r_test, assert |measured frac(r<r_test) - r^3/(r^2+a^2)^1.5| < 0.03
        "half-mass radius (50% of sampled mass within r_h)": "tests/validation/test_plummer_physics.py::TestPlummerDensityProfile::test_half_mass_radius_statistical",
        "spatial isotropy (component means ~ 0)": "tests/validation/test_plummer_physics.py::TestPlummerDensityProfile::test_positions_isotropic",
    },
    "KingProfile": {
        "ODE solution / boundary conditions psi(0)=W0, psi->0 at r_t": "tests/validation/test_king_physics.py::TestKingODESolution::test_boundary_conditions",
        "tidal truncation (100% of mass within r_t)": "tests/validation/test_king_physics.py::TestKingTidalTruncation::test_all_particles_within_tidal_radius",
        "density monotone-decreasing (lowered-Maxwellian shape)": "tests/validation/test_king_physics.py::TestKingDensityProfile::test_density_decreases_with_radius",
        "concentration c(W0)=log10(r_t/r_c) vs King (1966) Table II": "tests/validation/test_king_physics.py::test_concentration_matches_king1966_table_ii",
        # QUOTED WITNESS: assert |c - c_ref| <= 0.02 for W0 in {3,7,9} vs King Table II {0.67,1.53,2.12}
        "lowered-Maxwellian volume-density shape vs direct velocity integral": "tests/validation/test_king_physics.py::TestKingLoweredMaxwellianDensity::test_density_shape_matches_direct_velocity_integral",
    },
    "MichieProfile": {
        "isotropic limit (r_a -> inf) recovers King density": "tests/validation/test_michie_physics.py::TestMichieIsotropicLimit::test_density_matches_king_at_large_ra",
        "anisotropy beta(r) tracks the DF's own analytic 2nd-moment oracle": "tests/validation/test_michie_physics.py::TestMichieAnisotropyProfile::test_beta_matches_df_oracle",
        "lowered model: beta(r) below the pure Osipkov-Merritt ceiling r^2/(r^2+r_a^2)": "tests/validation/test_michie_physics.py::TestMichieAnisotropyProfile::test_beta_below_osipkov_merritt_ceiling",
        "stronger anisotropy => more extended (larger r_t at fixed W0,r_c)": "tests/validation/test_michie_physics.py::TestMichieAnisotropyStructure::test_more_anisotropic_more_extended",
    },
    "EFFProfile": {
        "density closed-form rho(0)=1, rho(a)=2^{-gamma/2}, rho~r^{-gamma} asymptote": "tests/validation/test_eff_physics.py::TestEFFDensityFormula::test_central_density_unity",
        "power-law asymptotic slope rho ~ r^{-gamma}": "tests/validation/test_eff_physics.py::TestEFFDensityFormula::test_power_law_slope_asymptotic",
        "tidal truncation (density = 0 and 100% of mass within r_t)": "tests/validation/test_eff_physics.py::TestEFFTidalTruncation::test_all_particles_within_tidal_radius",
        "gamma concentration (higher gamma => smaller median radius)": "tests/validation/test_eff_physics.py::TestEFFGammaConcentration::test_higher_gamma_more_concentrated",
    },
    # =========================== Velocity DFs ========================================
    "PlummerVelocityDF": {
        "virial Q = T/|V| ~ 0.5 (unscaled equilibrium)": "tests/validation/test_plummer_physics.py::TestPlummerVirialEquilibrium::test_virial_ratio",
        # QUOTED WITNESS: Q = T/|V| with V = -3*pi*G*M^2/(32a); assert |Q - 0.5| < VIRIAL_RATIO
        "velocity dispersion profile sigma^2(r) = GM/(6 sqrt(r^2+a^2))": "tests/validation/test_plummer_physics.py::TestPlummerVelocityDispersion::test_radial_dispersion_profile",
        "velocity isotropy (<vx^2> ~ <vy^2> ~ <vz^2>)": "tests/validation/test_plummer_physics.py::TestPlummerVelocityDispersion::test_velocity_isotropy",
        "Beta(3/2,9/2) speed distribution: <q^2> = 0.25": "tests/validation/test_plummer_physics.py::TestPlummerBetaDistribution::test_q_squared_mean",
        "all particles bound (v < v_esc)": "tests/validation/test_plummer_physics.py::TestPlummerBoundParticles::test_all_particles_bound",
    },
    "KingVelocityDF": {
        "virial Q = T/|V| ~ 0.5 unscaled (true lowered-Maxwellian equilibrium)": "tests/validation/test_king_physics.py::TestKingEquilibriumVelocityDF::test_virial_ratio_is_half_unscaled",
        # QUOTED WITNESS: Q = T / |V| from the sampled IC; assert |Q - 0.5| < 0.05 (no external rescale)
        "velocity isotropy (<vx^2> ~ <vy^2> ~ <vz^2>)": "tests/validation/test_king_physics.py::TestKingVelocityDF::test_velocity_isotropy",
        "all velocities bound against the King escape speed v_esc(r) = sigma sqrt(2 psi(r))": "tests/validation/test_king_physics.py::TestKingVelocityDF::test_velocities_bound_against_king_escape_speed",
        "sampled sigma_1d(r) matches the analytic lowered-Maxwellian 2nd moment": "tests/validation/test_king_physics.py::TestKingEquilibriumVelocityDF::test_dispersion_profile_matches_king_moment",
    },
    "MichieVelocityDF": {
        "virial Q = T/|V| ~ 0.5 unscaled (anisotropic equilibrium)": "tests/validation/test_michie_physics.py::TestMichieEquilibrium::test_virial_ratio_half_unscaled",
        "all velocities bound against v_esc(r) = sigma sqrt(2 W(r))": "tests/validation/test_michie_physics.py::TestMichieEquilibrium::test_all_particles_bound",
        "radial anisotropy grows outward (beta_inner < beta_outer)": "tests/validation/test_michie_physics.py::TestMichieAnisotropyProfile::test_beta_increases_outward",
    },
    "EFFVelocityDF": {
        "Eddington-DF virial Q ~ 0.5 for mild truncation (gamma=5)": "tests/validation/test_eff_physics.py::TestEFFVelocityDF::test_eff_eddington_virial_ratio_mild_truncation",
        # QUOTED WITNESS: gamma=5,r_t=15 build; assert |Q - 0.5| < 0.05 (unscaled, mild truncation)
        "tabulated Eddington f(E) physical: non-negative and increasing in E": "tests/validation/test_eff_physics.py::TestEFFVelocityDF::test_eff_eddington_f_is_physical",
        "velocity isotropy (<vx^2> ~ <vy^2> ~ <vz^2>)": "tests/validation/test_eff_physics.py::TestEFFVelocityDF::test_velocity_isotropy",
        "all velocities bound (v <= v_esc from the shared potential)": "tests/validation/test_eff_physics.py::TestEFFVelocityDF::test_eff_all_particles_bound",
    },
    # =========================== IMFs ================================================
    "PowerLawIMF": {
        "Salpeter (1955) high-mass slope alpha = 2.35": "tests/validation/test_imf_physics.py::TestSalpeterSlope::test_salpeter_high_mass_slope",
        "exact analytic mean mass (vs fine log-grid reference)": "tests/validation/test_imf_physics.py::TestMeanMassAccuracy::test_mean_mass_resolution_converged",
        "Kroupa (2001) breakpoint masses + segment slopes + PDF continuity": "tests/validation/test_imf_physics.py::TestKroupaBreakpoints::test_kroupa_segment_slopes",
        "massive-star rarity scales correctly with slope": "tests/validation/test_imf_physics.py::TestIMFMassiveStars::test_massive_more_common_with_lower_alpha",
    },
    "ChabrierIMF": {
        "Chabrier (2003) characteristic mass m_c = 0.08 + lognormal width sigma = 0.69": "tests/validation/test_imf_physics.py::TestChabrierParameters::test_chabrier_characteristic_mass",
        "high-mass slope = Chabrier (2003) Table 1 (x=1.3 => alpha=2.3)": "tests/validation/test_imf_physics.py::TestChabrierParameters::test_chabrier_high_mass_slope",
        "lognormal<->power-law value-continuous at m_trans=1 Msun (A_pl)": "tests/validation/test_imf_physics.py::TestChabrierParameters::test_chabrier_pdf_continuous_at_mtrans",
        "resolution-converged mean mass (vs fine log-grid reference)": "tests/validation/test_imf_physics.py::TestMeanMassAccuracy::test_mean_mass_resolution_converged",
    },
    "Maschberger": {
        "Maschberger (2013) peak mass mu = 0.2 Msun": "tests/validation/test_imf_physics.py::TestMaschbergerProperties::test_maschberger_peak_mass",
        "high-mass slope ~ 2.3 (near Salpeter)": "tests/validation/test_imf_physics.py::TestMaschbergerProperties::test_maschberger_high_mass_salpeter",
        "resolution-converged mean mass (vs fine log-grid reference)": "tests/validation/test_imf_physics.py::TestMeanMassAccuracy::test_mean_mass_resolution_converged",
    },
    "TruncatedIMF": {
        # TruncatedIMF re-normalizes a base IMF over a narrower [m_min, m_max]. The
        # normalization invariant CDF(m_min)=0 / CDF(m_max)=1 IS its physics: the
        # parametrized fixture includes the TruncatedChabrier case, so these node ids
        # assert ON a TruncatedIMF instance specifically.
        "renormalized CDF(m_max) = 1 after truncation": "tests/unit/imf/test_imf_core.py::TestCDFProperties::test_cdf_at_m_max[TruncatedChabrier]",
        # QUOTED WITNESS: cdf_max = imf.cdf(m_max); assert |cdf_max - 1.0| < 1e-6 (imf = TruncatedChabrier)
        "renormalized CDF(m_min) = 0 after truncation": "tests/unit/imf/test_imf_core.py::TestCDFProperties::test_cdf_at_m_min[TruncatedChabrier]",
    },
    # =========================== Multi-component equilibrium engine ===================
    "MultiComponentCluster": {
        # An equilibrium model (not a protocol-conformer / not build_*_cluster), so it is
        # hand-listed here; the operational ratchet does not flag it, but a missing entry
        # would still leave its physics unguarded — enumerated explicitly.
        "Engine-A: global virial Q = T/|V| ~ 0.5 unscaled across segregation delta": "tests/validation/test_multimass_equilibrium_physics.py::test_global_virial_is_half_across_delta",
        "Engine-A: theoretical per-component Q_j exactly 0.5 (each mass group in equilibrium)": "tests/validation/test_multimass_equilibrium_physics.py::test_theoretical_component_virial_is_exactly_half",
        "Engine-A: delta=0 is single-mass; segregation grows monotonically with delta": "tests/validation/test_multimass_equilibrium_physics.py::test_delta0_is_single_mass_and_segregation_grows",
        "Engine-A: anisotropic sampler is equilibrium AND carries the right beta_j(r)": "tests/validation/test_multimass_equilibrium_physics.py::test_anisotropic_sampled_cluster_is_equilibrium_and_correctly_anisotropic",
        "Engine-B: King A-vs-B cross-engine agreement (r_t / theory Q_j / sigma_1d(r) / radial KS)": "tests/validation/test_engine_b_physics.py::test_king_density_engine_b_matches_engine_a",
        "Engine-B: Plummer halo + EFF core is a true shared-potential equilibrium (headline Q)": "tests/validation/test_engine_b_physics.py::test_plummer_halo_eff_core_equilibrium",
        "Engine-B: OM anisotropy realized in sampled velocities beta_halo(r)=r^2/(r^2+r_a^2)": "tests/validation/test_engine_b_physics.py::test_om_beta_profile_realized",
        "Engine-B: DF-density inversion fidelity (rho_DF,j == augmented rho_presc,j interior)": "tests/validation/test_engine_b_physics.py::test_df_density_fidelity_interior",
    },
    # =========================== Cluster builders (build_*_cluster) ===================
    "build_cluster": {
        "bit-identical to the manual build_spatial_ic composition (pure sugar, no drift)": "tests/unit/builders/test_cluster_builders.py::test_build_cluster_is_bit_identical_to_manual_base_case",
        # QUOTED WITNESS: build_cluster(p, masses, key) compared field-by-field (== on
        # positions/velocities/masses/stellar_radii) to build_spatial_ic(p, M, df, K, Q=0.5)
        "tidal modifier zeroes outer masses beyond r_t": "tests/unit/builders/test_cluster_builders.py::test_tidal_zeroes_outer_masses",
        "OM anisotropy modifier threads into the DF (radial velocity bias)": "tests/unit/builders/test_cluster_builders.py::test_anisotropy_threads_into_df_radial_bias",
    },
    "build_plummer_cluster": {
        "Plummer-family alias IC bit-identical to build_cluster(Plummer)": "tests/unit/builders/test_cluster_builders.py::test_plummer_alias_identical",
    },
    "build_king_cluster": {
        "King-family alias IC bit-identical to build_cluster(King)": "tests/unit/builders/test_cluster_builders.py::test_king_alias_identical",
    },
    "build_eff_cluster": {
        "EFF-family alias IC bit-identical to build_cluster(EFF)": "tests/unit/builders/test_cluster_builders.py::test_eff_alias_identical",
    },
    "build_michie_cluster": {
        "Michie-family alias IC bit-identical to build_cluster(Michie)": "tests/unit/builders/test_cluster_builders.py::test_michie_alias_identical",
    },
    "build_limepy_cluster": {
        "LIMEPY-family alias IC bit-identical to build_cluster(LIMEPY)": "tests/unit/builders/test_cluster_builders.py::test_limepy_alias_identical",
    },
    "build_cluster_from_params": {
        "ClusterParams theta-PyTree wrapper IC identical to build_cluster": "tests/unit/builders/test_cluster_builders.py::test_cluster_params_wrapper_identical_to_build_cluster",
    },
    "build_binary_cluster": {
        "Kepler's third law survives the binary->spatial assembly (P round-trip)": "tests/integration/test_binary_cluster.py::TestBuildBinaryCluster::test_units_kepler_third_law_roundtrip",
        # QUOTED WITNESS: recover P from the resolved primary/companion separation; assert |P - 100 d| < 1e-3
        "cluster COM and Vcom conserved to 1e-10 after assembly": "tests/integration/test_binary_cluster.py::TestBuildBinaryCluster::test_com_preserved",
        "particle count = primaries + secondaries; no ghost masses": "tests/integration/test_binary_cluster.py::TestBuildBinaryCluster::test_count_and_provenance",
    },
    "build_cataloged_binary_cluster": {
        "physical particle state is identical to the legacy binary-cluster assembly at a fixed key": "tests/integration/test_binary_cluster.py::TestCatalogedBinaryCluster::test_legacy_equivalence_for_compact_targets",
        "sampled orbital elements are retained exactly for every primordial binary": "tests/integration/test_binary_cluster.py::TestCatalogedBinaryCluster::test_retains_sampled_orbital_elements",
        "periapsis contact margin equals a(1-e)-(R1+R2) in position units": "tests/integration/test_binary_cluster.py::TestCatalogedBinaryCluster::test_contact_margin_is_in_position_units",
    },
}

# --- EXEMPT_NON_MODEL: non-model __all__ symbol -> reason --------------------------------
# Utilities, PyTree/data containers, distributions, factories, helpers, analytical ICs, and
# diagnostics. No equilibrium/physics invariant is required of these (they are not models in
# the operational sense — not profiles/DFs/IMFs by protocol, not build_*_cluster). Their
# behavior is exercised by the api-coverage registry's asserting tests; here we record only
# that no MODEL invariant applies. Mirrors the grad-audit EXEMPT taxonomy.
EXEMPT_NON_MODEL: dict[str, str] = {
    # --- runtime-checkable typing Protocols (the yardstick, not a measured model) ---
    "SpatialProfile": "typing Protocol, not a concrete model (grad-audit EXEMPT_PROTOCOL)",
    "VelocityDF": "typing Protocol, not a concrete model (grad-audit EXEMPT_PROTOCOL)",
    "IMFProtocol": "typing Protocol, not a concrete model (grad-audit EXEMPT_PROTOCOL)",
    "PeriodDistribution": "typing Protocol for period distributions, not a model",
    "EccentricityDistribution": "typing Protocol for ecc distributions, not a model",
    "ConditionalEccentricityDistribution": "typing Protocol (p(e|P)), not a model",
    "MassPeriodEccentricityDistribution": "typing Protocol (p(e|P,M1)), not a model",
    "BinaryFractionModel": "typing Protocol for binary-fraction models, not a model",
    "CompanionModel": "typing Protocol for the companion/orbit layer, not a model",
    # --- PyTree / data containers (no standalone equilibrium physics) ---
    "ICResult": "IC-result PyTree container (positions/velocities/masses), no model physics",
    "Systems": "population-target container (n systems), not a model",
    "Stars": "population-target container (n stars), not a model",
    "TotalMass": "population-target container (total mass budget), not a model",
    "ResolvedBinaries": "resolved-binary PyTree container (2N slots), not a model",
    "CatalogedBinaryClusterIC": "catalog-bearing particle-state PyTree container, not a model",
    "PrimordialSystemCatalog": "immutable system-level birth-provenance PyTree, not a model",
    "CompanionElements": "companion-elements NamedTuple container, not a model",
    "BinaryOrbitalState": "binary orbital-state container, not a model",
    "CartesianState": "Cartesian (r, v) state container, not a model",
    "BinaryState": "two-body Cartesian state container, not a model",
    "BinaryEnergyBudget": "binary energy-budget result container, not a model",
    "AnalyticalIC": "base eqx.Module container for analytical ICs, not an equilibrium model",
    "MassDependentBinaryConfig": "mass-routed orbit-sampling config container, not a model",
    "RotationSpec": "rotation-overlay spec PyTree, not a model",
    "ClusterParams": "theta-PyTree bundle (profile + modifier knobs), not a model",
    "DEFAULT_UNITS": "module-level UnitSystem constant (carries G), not a model",
    "SOLAR_SYSTEM_PLANETS": "solar-system data table constant, not a model",
    # --- Mass-ratio distributions (binary-statistics components, not stellar models) ---
    "FlatMassRatio": "binary mass-ratio distribution, not a spatial/velocity/IMF model",
    "PowerLawMassRatio": "binary mass-ratio distribution, not a model",
    "TwinPeakedMassRatio": "binary mass-ratio distribution, not a model",
    "MoeDiStefano2017": "Moe+2017 mass-ratio gamma(M1) model, not a stellar-structure model",
    "MoeDiStefano2017Full": "Moe+2017 full Table-13 q model, not a stellar-structure model",
    "MoeJointOrbit": "Moe+2017 joint (P,q,e) sampler, not a stellar-structure model",
    "MoePeriod": "Moe+2017 period component, not a stellar-structure model",
    # --- Period / eccentricity distributions (binary-orbit statistics) ---
    "LogNormalPeriod": "binary period distribution (DM91), not a model",
    "LogUniformPeriod": "binary period distribution (Opik), not a model",
    "SanaOBPeriod": "binary period distribution (Sana+2012 OB), not a model",
    "ThermalEccentricity": "binary eccentricity distribution (f(e)=2e), not a model",
    "UniformEccentricity": "binary eccentricity distribution, not a model",
    "MoeEccentricity": "period/mass-conditional eccentricity distribution, not a model",
    "LogisticThermalEccentricity": "circular->thermal blend ecc distribution, not a model",
    # --- Binary-fraction models (per-star f_bin, not stellar-structure models) ---
    "ConstantBinaryFraction": "per-star binary-fraction model, not a stellar-structure model",
    "MassDependentBinaryFraction": "f_bin(m) model, not a stellar-structure model",
    "RadialBinaryFraction": "f_bin(r) model, not a stellar-structure model",
    "CombinedBinaryFraction": "f_bin(m)xf_bin(r) model, not a stellar-structure model",
    # --- Binary IMF + companion samplers (composition layers, not protocol IMFs) ---
    "BinaryIMF": "primary-IMF + binary-statistics composition; not an IMFProtocol conformer "
    "(lacks mean_mass at class level) and not a spatial/velocity model. Its "
    "binary-fraction behavior is asserted in tests/unit/imf/test_binary.py.",
    "IndependentCompanions": "companion sampler (CompanionModel), not a stellar-structure model",
    "MoeCompanions": "Moe+2017 companion sampler (CompanionModel), not a stellar-structure model",
    # --- Kepler / orbital mechanics ---
    "KeplerElements": "Kepler orbital-element container + transforms, not an equilibrium model",
    # --- Profile/DF SOLVERS (return arrays/PyTrees, not sampleable model objects) ---
    "solve_king_profile": "King ODE solver (returns xi/psi grids), not a model object",
    "solve_michie_profile": "Michie ODE solver, not a model object",
    "solve_limepy_profile": "LIMEPY ODE solver, not a model object",
    "solve_multimass_limepy": "multimass LIMEPY coupled solver, not a model object",
    "find_alpha_for_masses": "eigenvalue solve for alpha_j, not a model object",
    "matched_velocity_df": "factory pairing a profile -> its equilibrium DF, not a model itself",
    # --- Kinematic transforms / overlays / orientation ---
    "apply_solid_body_rotation": "velocity overlay (adds Omega x R), not a model",
    "apply_differential_rotation": "velocity overlay, not a model",
    "sample_isotropic_orientations": "isotropic orientation sampler, not a model",
    # --- Dispersion forward models (free functions over a profile, not equilibrium models) ---
    "jeans_dispersion": "anisotropic-Jeans (Osipkov-Merritt) dispersion forward model "
    "(sigma_r/sigma_t/sigma_1d/beta for a profile under OM r_a); a "
    "forward-model helper, not an equilibrium model. Physics anchored in "
    "tests/validation/test_dispersion_physics.py.",
    "project_dispersion": "Binney & Mamon (1982) line-of-sight projection forward model "
    "(sigma_los/sigma_pm_r/sigma_pm_t); a forward-model helper, not an "
    "equilibrium model. Anchored in test_dispersion_physics.py.",
    "df_moment_dispersion": "exact Michie-King DF second-moment dispersion "
    "(sigma_r/sigma_t/sigma_1d/beta); a forward-model helper, not an "
    "equilibrium model. Physics anchored in "
    "tests/validation/test_dispersion_physics.py (sampler all-radii + "
    "Tier-A Jeans consistency + beta-vs-_michie_beta_oracle).",
    # --- Binary connector + diagnostics + energy/kinematic kernels ---
    "resolve_binary_components": "binary->spatial connector, not a model (grad-audit AUDITED "
    "as a Fisher path, but it is a transform, not a model object)",
    "batch_elements_to_resolved": "Kepler-elements->resolved batch transform, not a model",
    "sample_mass_dependent_orbits": "mass-routed orbit sampler helper, not a model",
    "relative_energy": "binary relative-energy diagnostic, not a model",
    "find_bound_pairs": "bound-pair diagnostic, not a model",
    "find_bound_multiples": "bound-multiple diagnostic, not a model",
    "primordial_survival": "primordial-binary survival diagnostic, not a model",
    "binary_energy_budget": "binary energy-budget diagnostic, not a model",
    "energy_sorted_segregation": "primordial mass-segregation IC helper/diagnostic, not a model",
    "compute_kinetic_energy": "T kernel, not a model",
    "compute_potential_energy": "V kernel, not a model",
    "to_com_frame": "COM-frame transform utility, not a model",
    "virial_scale": "virial-rescale utility, not a model",
    "compute_stellar_radii": "mass->radius ZAMS helper, not a model",
    # --- ZAMS stellar relations (Tout+1996 mass-relations, not equilibrium models) ---
    "zams_luminosity": "Stellar mass-relation (Tout+1996 ZAMS), not an equilibrium model; "
    "validated vs published anchors in test_zams_physics.py.",
    "zams_radius": "Stellar mass-relation (Tout+1996 ZAMS), not an equilibrium model; "
    "validated vs published anchors in test_zams_physics.py.",
    "zams_effective_temperature": "Stellar mass-relation (Tout+1996 ZAMS), not an equilibrium "
    "model; validated vs published anchors in test_zams_physics.py.",
    "zams_surface_gravity": "Stellar mass-relation (Tout+1996 ZAMS), not an equilibrium model; "
    "validated vs published anchors in test_zams_physics.py.",
    "inverse_zams_luminosity": "Stellar mass-relation (Tout+1996 ZAMS), not an equilibrium "
    "model; validated vs published anchors in test_zams_physics.py.",
    "compute_period": "Kepler-III period helper, not a model",
    "period_to_semimajor_axis": "Kepler-III inverse helper, not a model",
    # --- Tidal physics helpers ---
    "jacobi_radius": "Jacobi-radius formula helper, not a model",
    "jacobi_radius_isothermal": "isothermal Jacobi-radius helper, not a model",
    "apply_tidal_truncation": "tidal-truncation transform, not a model",
    "fill_factor_to_r_h": "fill-factor -> r_h helper, not a model",
    "get_planet": "solar-system data accessor, not a model",
    # --- Analytical ICs / closed-form test cases (fixed-config, exact-solution) ---
    "two_body_kepler": "fixed-config analytic 2-body IC, not an equilibrium model",
    "two_body_period": "closed-form 2-body period, not a model",
    "two_body_energy": "closed-form 2-body energy, not a model",
    "three_body_figure_eight": "fixed-config figure-eight IC, not a model",
    "figure_eight_period": "closed-form figure-eight period, not a model",
    "harmonic_oscillator": "fixed-config SHO IC, not a model",
    "harmonic_solution": "closed-form SHO x(t), not a model",
    "earth_sun_2body": "fixed-config Earth-Sun IC, not a model",
    "earth_sun_eccentric": "fixed-config eccentric Earth-Sun IC, not a model",
    "sun_earth_jupiter_3body": "fixed-config 3-body IC, not a model",
    "solar_system_inner_4": "fixed-config inner-4 solar system IC, not a model",
    "solar_system_full": "fixed-config full solar system IC, not a model",
    # --- Top-level IC assembly (composition entry point, not a model) ---
    "build_spatial_ic": "generic profile+DF IC-assembly entry point; the models it composes "
    "(profiles/DFs) carry the equilibrium invariants, not the assembler.",
}

# --- EXEMPT_NON_EQUILIBRIUM_MODEL: model -> documented non-equilibrium-physics reason ----
# These ARE operational models (profiles/DFs), but their correctness is a REFERENCE-PARITY
# or UNIFORM-DENSITY-RECOVERY claim rather than an equilibrium-Q claim — a documented carve
# so their exclusion from the equilibrium invariant class is auditable. The cited test is a
# real, asserting physics test (NOT a grep-mention).
EXEMPT_NON_EQUILIBRIUM_MODEL: dict[str, str] = {
    "LIMEPYProfile": (
        "physics is REFERENCE-PARITY vs the canonical Gieles & Zocchi (2015) numpy/scipy "
        "LIMEPY (rho_j shapes, sigma_j shapes, alpha_j, mass fractions, concentration, r_h), "
        "tested in tests/validation/test_limepy_reference_parity.py::test_parity_with_reference_limepy "
        "against cached reference outputs — not an equilibrium-Q claim."
    ),
    "LIMEPYVelocityDF": (
        "the LIMEPY velocity DF's equilibrium is exercised THROUGH MultiComponentCluster "
        "(Engine A virial / dispersion / anisotropy tests) and its profile parity is the "
        "reference-parity oracle above; as a standalone __all__ symbol its asserting unit "
        "test is tests/unit/builders/test_cluster_builders.py::test_matched_limepy_isotropic_passes_none_r_a "
        "(constructs it, asserts isotropic r_a non-finite). Reference-parity / composed "
        "equilibrium, not a direct equilibrium-Q claim of its own."
    ),
    "UniformSphereProfile": (
        "physics is UNIFORM-DENSITY RECOVERY (a CW04 Q-substructure baseline, Q ~ 0.79), "
        "tested in tests/unit/substructure/test_q_baselines.py::TestUniformSphereBaseline::test_q_matches_cw04_range "
        "(assert 0.75 < Q_mean < 0.90 on sampled positions) — not an equilibrium model "
        "(no velocity DF / no virial Q)."
    ),
}

# --- UNTESTED_MODELS: operational model -> hole note (REAL holes; Task-4.2 items) --------
# EMPTY as of Task 4.1: every operational model (profile/DF/IMF by protocol, build_*_cluster)
# and the MultiComponentCluster engine has >= 1 enumerated invariant mapped to an asserting
# validation/physics test. A NEW model whose invariant is not yet checked re-populates this
# and reds test_no_untested_model_holes until the validation test exists.
UNTESTED_MODELS: dict[str, str] = {}
