"""Gradient-coverage source of truth (design Q1/Q2/Q6).

THREE frozen structures, all hand-curated independent literals (NOT computed from REGISTRY
or __all__ at runtime — a derived manifest could not catch a deleted case / a new symbol):

  MUST_AUDIT       : the (id, param) coverage units the registry MUST cover (ratchet target).
  SYMBOL_CATEGORY  : every progenax.__all__ symbol -> AUDITED | EXEMPT_* (cross-check target).
  PARAM_ALLOWLIST  : registry (id, param) that are legitimately NOT FD-consistent (carry-forward).

Seeded by tests/validation/grad_audit/_gen_manifest_seed.py, then frozen. Enforced by
tests/validation/grad_audit/test_manifest_coverage.py. The website coverage section is
generated from here.
"""

# --- Categories -------------------------------------------------------------
AUDITED = "AUDITED"  # has >=1 registry case (in MUST_AUDIT)
EXEMPT_PROTOCOL = "EXEMPT_PROTOCOL"  # a typing Protocol, not an entry point
EXEMPT_CONTAINER = "EXEMPT_CONTAINER"  # dataclass/PyTree container, not a sampler
EXEMPT_ANALYTICAL_IC = "EXEMPT_ANALYTICAL_IC"  # fixed-config analytic IC/test fixture
EXEMPT_NON_FISHER_DIAGNOSTIC = (
    "EXEMPT_NON_FISHER_DIAGNOSTIC"  # diagnostic/energy kernel, own tests
)
EXEMPT_HELPER = "EXEMPT_HELPER"  # helper/constant/orientation util
EXEMPT_COVERED_ELSEWHERE = (
    "EXEMPT_COVERED_ELSEWHERE"  # differentiable sampler with a scattered
)
# FD test; deferred from THIS arc's registry
# scope (Tier-4 inventory "future candidate")

# --- MUST_AUDIT: (id, param) -> rationale -----------------------------------
# Seeded from REGISTRY (Phase A = current coverage only; D4 entries appended in Phase B).
# Every (id, param) coverage unit the registry MUST keep covering (75 units).
# NOTE: MUST_AUDIT ids are NOT a subset of the AUDITED public symbols — several are
# internal/submodule entry points legitimately absent from progenax.__all__ (the frozen-edge
# binners, q_approx, lambda_msr_approx, Schechter, IMFParams, PlummerVelocityDF+OM). They are
# still ratcheted by test_every_must_audit_entry_is_covered (a direct (id,param) set check).
MUST_AUDIT: dict[tuple[str, str], str] = {
    ("ChabrierIMF.ppf", "alpha"): "Chabrier inverse-CDF in low-mass slope",
    ("ChabrierIMF.ppf", "m_c"): "Chabrier inverse-CDF in characteristic mass",
    ("ChabrierIMF.ppf", "sigma"): "Chabrier inverse-CDF in lognormal width",
    ("ChabrierIMF.ppf[H6 boundary]", "m_c"): "H6 Newton-clamp boundary probe (benign)",
    ("EFFProfile.sample_positions", "a"): "EFF position sampler scale radius",
    (
        "EFFProfile.sample_positions",
        "gamma",
    ): "EFF position sampler slope (near-divergent edge)",
    ("EFFProfile.sample_positions", "r_t"): "EFF position sampler truncation radius",
    ("EFFVelocityDF.sample_velocities", "a"): "EFF Eddington DF scale radius",
    (
        "EFFVelocityDF.sample_velocities",
        "gamma",
    ): "EFF Eddington DF slope (virial point)",
    ("IMFParams.log_prob_nll", "alpha3"): "4-segment IMF NLL Fisher channel",
    (
        "IndependentCompanions.sample",
        "e_max",
    ): "thermal ecc ceiling THROUGH the full companion assembly (is_binary draw + m2 gate + pytree); complements B5's bare-dist case",
    ("KeplerElements.from_state", "v_scale"): "state->elements inverse, recovered a",
    ("KeplerElements.to_state", "M0"): "elements->Cartesian M0 column",
    ("KeplerElements.to_state", "a"): "elements->Cartesian a column",
    (
        "KeplerElements.to_state",
        "e",
    ): "elements->Cartesian e column (near-parabolic edge)",
    ("KingProfile.r_t", "W0"): "King tidal radius differentiability in W0",
    ("KingProfile.sample_positions", "W0"): "King position sampler in W0 (W0=12 edge)",
    ("KingProfile.sample_positions", "r_c"): "King position sampler in core radius",
    ("KingVelocityDF.sample_velocities", "W0"): "King lowered-Maxwellian DF in W0",
    ("KingVelocityDF.sample_velocities", "r_c"): "King DF in core radius",
    ("LogNormalPeriod.sample", "mu_log_P"): "Raghavan+2010 log-normal period location",
    ("LogUniformPeriod.sample", "log_P_max"): "Opik log-uniform period upper bound",
    (
        "LogisticThermalEccentricity.sample",
        "e_max",
    ): "DM91 circular->thermal blend ecc ceiling (FD-target)",
    ("Maschberger.ppf", "alpha"): "Maschberger inverse-CDF in slope",
    ("Maschberger.ppf", "beta"): "Maschberger inverse-CDF in beta",
    ("Maschberger.ppf", "mu"): "Maschberger inverse-CDF in mu",
    ("MichieProfile.density[log rho(r)]", "W0"): "Michie closed-form density obs in W0",
    (
        "MichieProfile.density[log rho(r)]",
        "r_c",
    ): "Michie closed-form density obs in r_c",
    ("MichieProfile.r_t", "W0"): "Michie tidal radius differentiability in W0",
    (
        "MichieProfile.sample_positions",
        "W0",
    ): "Michie anisotropic position sampler in W0",
    ("MichieVelocityDF.sample_velocities", "W0"): "Michie anisotropic DF in W0",
    (
        "MoeCompanions.sample",
        "m1_scale",
    ): "Moe P-q-e companion sampler, <e> vs m1 scale",
    (
        "MoeEccentricity.sample",
        "e_max",
    ): "Moe+2017 ecc ceiling at long P (Roche cap binds; FD-target)",
    (
        "MultiComponentCluster.from_components[EngineA]",
        "w_j",
    ): "Engine-A velocity-scale ratio (Fisher target)",
    (
        "MultiComponentCluster.from_mass_segregation[EngineA]",
        "r_a",
    ): "Engine-A equipartition anisotropy radius (OM, Fisher target)",
    (
        "MultiComponentCluster.sample_cluster[EngineA]",
        "W0",
    ): "Engine A multimass in W0 (H2 edge)",
    (
        "MultiComponentCluster.sample_cluster[EngineA]",
        "delta",
    ): "Engine A mass-segregation exponent",
    (
        "MultiComponentCluster.sample_cluster[EngineA]",
        "g",
    ): "Engine A truncation sharpness",
    (
        "MultiComponentCluster.sample_cluster[EngineB]",
        "gamma",
    ): "Engine B core EFF slope",
    (
        "MultiComponentCluster.sample_cluster[EngineB]",
        "r_a",
    ): "Engine B halo OM anisotropy radius",
    (
        "MultiComponentCluster.sample_cluster[EngineB]",
        "r_h",
    ): "Engine B halo Plummer scale",
    (
        "PlummerProfile.enclosed_mass_fraction[N(r) model]",
        "r_h",
    ): "N(r) model Fisher column",
    ("PlummerProfile.sample_positions", "r_h"): "headline spatial sampler",
    (
        "PlummerVelocityDF+OM.sample_velocities",
        "r_a",
    ): "Plummer OM anisotropy radius (edge)",
    ("PlummerVelocityDF.sample_velocities", "r_h"): "Plummer velocity DF in r_h",
    ("PowerLawIMF.cdf[H4]", "m_min"): "H4 clip probe, cdf vs m_min (benign)",
    ("PowerLawIMF.mean_mass", "alpha"): "analytic E[m] in slope (alpha=1 edge)",
    (
        "PowerLawIMF.ppf[Salpeter]",
        "alpha",
    ): "Salpeter inverse-CDF in slope (alpha=1 edge)",
    ("PowerLawIMF.ppf[m_min]", "m_min"): "Salpeter inverse-CDF lower-support edge",
    ("PowerLawIMF.sample[Salpeter]", "alpha"): "Salpeter reparam sampler in slope",
    ("SanaOBPeriod.sample", "power"): "Sana+2012 OB period power-law index",
    ("Schechter.ppf", "alpha"): "Schechter grid-CDF+Newton inverse in slope",
    (
        "ThermalEccentricity.sample",
        "e_max",
    ): "Heggie+1975 thermal f(e)=2e ecc scale (<sqrt u>~2/3 closed-form)",
    (
        "UniformEccentricity.sample",
        "e_max",
    ): "uniform ecc upper bound (<u>~0.5 closed-form)",
    (
        "apply_differential_rotation",
        "R_peak",
    ): "differential rotation overlay, nonlinear R_peak",
    (
        "apply_differential_rotation",
        "v_peak",
    ): "differential rotation overlay, linear v_peak",
    ("apply_solid_body_rotation", "omega"): "solid-body rotation overlay, linear omega",
    (
        "binned_number_density[data, pinned non-diff]",
        "r_h",
    ): "frozen-count data side, pinned non-diff",
    ("binned_sigma1d[Plummer]", "r_h"): "binned sigma_1d Fisher channel in r_h",
    (
        "binned_sigma_beta[Plummer+OM]",
        "r_a",
    ): "binned beta(r) anisotropy Fisher channel",
    (
        "build_binary_cluster",
        "r_h",
    ): "flagship IMF->companion->spatial binary-cluster Fisher path in r_h",
    (
        "build_cataloged_binary_cluster",
        "r_h",
    ): "catalog-bearing wrapper over the binary-cluster Fisher path in r_h",
    (
        "build_cluster[Plummer]",
        "r_h",
    ): "convenience builder headline (bit-identical sugar; machine-exact)",
    (
        "build_cluster[Plummer+OM]",
        "anisotropy_radius",
    ): "build_cluster OM anisotropy channel via matched DF",
    (
        "build_cluster[Plummer+rotation]",
        "omega",
    ): "build_cluster solid-body rotation overlay channel",
    (
        "build_king_cluster",
        "r_c",
    ): "King family THROUGH the alias (r_c; W0 concrete -> consistent ODE domain)",
    ("build_eff_cluster", "gamma"): "EFF family through the alias (density slope)",
    (
        "build_michie_cluster",
        "W0",
    ): "Michie family through the alias (fixed xi_max=800 -> W0 consistent)",
    (
        "build_limepy_cluster",
        "W0",
    ): "LIMEPY family through the alias (fixed xi_max=300 -> W0 consistent)",
    (
        "build_cluster_from_params[ClusterParams]",
        "r_h",
    ): "ClusterParams theta-PyTree wrapper path (r_h leaf)",
    ("build_spatial_ic[Plummer]", "r_h"): "headline end-to-end IC positions in r_h",
    (
        "build_spatial_ic[Plummer].velocities",
        "r_h",
    ): "end-to-end IC velocities (virial-scaled) in r_h",
    ("lambda_msr_approx", "core_scale"): "soft Lambda_MSR segregation surrogate",
    ("q_approx[EFF]", "gamma"): "CW04 substructure-Q surrogate in EFF slope",
    ("resolve_binary_components", "a"): "binary->spatial connector, all-binary",
    (
        "resolve_binary_components[mixed]",
        "a",
    ): "connector mixed mask (sanitization Fisher check)",
    (
        "zams_luminosity",
        "mass",
    ): "Tout+1996 L(M) rational-function differentiability in mass",
    (
        "zams_radius",
        "mass",
    ): "Tout+1996 R(M) rational-function differentiability in mass",
    (
        "zams_effective_temperature",
        "mass",
    ): "ZAMS T_eff(M) (Stefan-Boltzmann L,R composite) in mass",
    ("zams_surface_gravity", "mass"): "ZAMS log g(M) = log10(G M / R^2) in mass",
    (
        "inverse_zams_luminosity",
        "L_target",
    ): "differentiable Newton/scan invert dM/dL at an "
    "in-range L_target (M~5 MS point, NOT a clipped plateau)",
    # --- Dispersion forward models (Phase 0 Task 8) ---
    ("jeans_dispersion[Plummer+OM]", "r_a"): "OM Jeans sigma_r in anisotropy radius",
    ("jeans_dispersion[Plummer]", "M"): "Jeans sigma_r in total mass",
    ("project_dispersion[Plummer+OM]", "r_a"): "B&M82 sigma_los in anisotropy radius",
    (
        "project_dispersion[Plummer+OM].pm_t",
        "r_a",
    ): "B&M82 sigma_pm_t (beta-carrying) in r_a",
    ("df_moment_dispersion[Michie]", "M"): "Michie DF-moment sigma_r in total mass",
    ("df_moment_dispersion[Michie].W0", "W0"): "Michie DF-moment sigma_r in W0 "
    "(fixed-node interp => AD-correct, no kink; closes the deferred W0 axis)",
    (
        "jeans_dispersion[EFF spans r_t]",
        "gamma",
    ): "EFF Jeans sigma_r over a grid spanning r_t (safe-sqrt-at-0)",
    (
        "project_dispersion[EFF spans r_t]",
        "a",
    ): "EFF B&M82 sigma_los over a grid spanning r_t (safe-sqrt-at-0)",
}

# --- SYMBOL_CATEGORY: every __all__ symbol -> category ----------------------
# Every progenax.__all__ symbol (114) categorized exactly once.
SYMBOL_CATEGORY: dict[str, str] = {
    # --- AUDITED (owns >=1 registry id) ---
    "PlummerProfile": AUDITED,
    "KingProfile": AUDITED,
    "MichieProfile": AUDITED,
    "EFFProfile": AUDITED,
    "MultiComponentCluster": AUDITED,
    "PowerLawIMF": AUDITED,
    "ChabrierIMF": AUDITED,
    "Maschberger": AUDITED,
    "PlummerVelocityDF": AUDITED,
    "KingVelocityDF": AUDITED,
    "MichieVelocityDF": AUDITED,
    "EFFVelocityDF": AUDITED,
    "build_spatial_ic": AUDITED,
    "resolve_binary_components": AUDITED,
    "MoeCompanions": AUDITED,
    "KeplerElements": AUDITED,
    "apply_solid_body_rotation": AUDITED,  # rotation overlay omega case (now root-exported)
    "apply_differential_rotation": AUDITED,  # rotation overlay v_peak/R_peak cases
    "build_binary_cluster": AUDITED,  # B3: end-to-end IMF->companion->spatial Fisher path (r_h)
    "build_cataloged_binary_cluster": AUDITED,
    # --- cluster convenience builders (cluster-builders arc) ---
    "build_cluster": AUDITED,  # r_h + anisotropy_radius + omega (3 FD-consistent cases)
    "build_king_cluster": AUDITED,  # King family through the alias (r_c)
    "build_eff_cluster": AUDITED,  # EFF family through the alias (gamma)
    "build_michie_cluster": AUDITED,  # Michie family through the alias (W0)
    "build_limepy_cluster": AUDITED,  # LIMEPY family through the alias (W0)
    "build_cluster_from_params": AUDITED,  # ClusterParams theta-PyTree wrapper (r_h)
    # build_plummer_cluster / matched_velocity_df: gradient path is identical to / subsumed
    # by build_cluster[Plummer] (audited). The tidal_radius channel inherits apply_tidal_
    # truncation's EXEMPT_HELPER straight-through status (covered by a LIVE-gradient teeth test,
    # NOT an FD-consistent Case — see test_grad_audit.py::test_cluster_tidal_gradient_has_teeth).
    "build_plummer_cluster": EXEMPT_HELPER,
    "matched_velocity_df": EXEMPT_HELPER,  # factory pairing profile->equilibrium DF
    "RotationSpec": EXEMPT_CONTAINER,  # rotation-overlay spec PyTree
    "ClusterParams": EXEMPT_CONTAINER,  # theta-PyTree bundle (profile + modifier knobs)
    # --- ZAMS stellar relations (Tout+1996; P3 grad-audit cases) ---
    "zams_luminosity": AUDITED,  # P3: Tout L(M) in mass (M=5 MS point)
    "zams_radius": AUDITED,  # P3: Tout R(M) in mass (M=5 MS point)
    "zams_effective_temperature": AUDITED,  # P3: T_eff(M) Stefan-Boltzmann composite in mass
    "zams_surface_gravity": AUDITED,  # P3: log g(M) in mass
    "inverse_zams_luminosity": AUDITED,  # P3: Newton/scan invert dM/dL at in-range L
    "SanaOBPeriod": AUDITED,  # B4: Sana+2012 OB period power-law index (power)
    "LogNormalPeriod": AUDITED,  # B4: log-normal period location (mu_log_P, closed-form grad=1)
    "LogUniformPeriod": AUDITED,  # B4: Opik log-uniform period upper bound (log_P_max, <u>~0.5)
    # --- Dispersion forward models (Phase 0 Task 8) ---
    "jeans_dispersion": AUDITED,  # OM Jeans sigma_r in r_a + M (interior radii; FD-consistent)
    "project_dispersion": AUDITED,  # B&M82 sigma_los + sigma_pm_t (beta-carrying) in r_a
    "df_moment_dispersion": AUDITED,  # Michie DF-moment sigma_r in M + W0 (interior radii; FD-consistent; W0 now audited)
    # --- EXEMPT_PROTOCOL (runtime-checkable typing Protocols) ---
    "SpatialProfile": EXEMPT_PROTOCOL,
    "VelocityDF": EXEMPT_PROTOCOL,
    "IMFProtocol": EXEMPT_PROTOCOL,
    "PeriodDistribution": EXEMPT_PROTOCOL,
    "EccentricityDistribution": EXEMPT_PROTOCOL,
    "ConditionalEccentricityDistribution": EXEMPT_PROTOCOL,
    "MassPeriodEccentricityDistribution": EXEMPT_PROTOCOL,
    "BinaryFractionModel": EXEMPT_PROTOCOL,
    "CompanionModel": EXEMPT_PROTOCOL,
    # --- EXEMPT_CONTAINER (dataclass / PyTree containers, not samplers) ---
    "ICResult": EXEMPT_CONTAINER,
    "Systems": EXEMPT_CONTAINER,
    "Stars": EXEMPT_CONTAINER,
    "TotalMass": EXEMPT_CONTAINER,
    "ResolvedBinaries": EXEMPT_CONTAINER,
    "CatalogedBinaryClusterIC": EXEMPT_CONTAINER,
    "PrimordialSystemCatalog": EXEMPT_CONTAINER,
    "CompanionElements": EXEMPT_CONTAINER,
    "BinaryOrbitalState": EXEMPT_CONTAINER,
    "CartesianState": EXEMPT_CONTAINER,
    "BinaryState": EXEMPT_CONTAINER,
    "BinaryEnergyBudget": EXEMPT_CONTAINER,
    "AnalyticalIC": EXEMPT_CONTAINER,
    "MassDependentBinaryConfig": EXEMPT_CONTAINER,
    "DEFAULT_UNITS": EXEMPT_CONTAINER,  # a UnitSystem constant (carries G); not a sampler
    # --- EXEMPT_ANALYTICAL_IC (fixed-config analytic IC / closed-form helpers) ---
    "two_body_kepler": EXEMPT_ANALYTICAL_IC,
    "two_body_period": EXEMPT_ANALYTICAL_IC,
    "two_body_energy": EXEMPT_ANALYTICAL_IC,
    "three_body_figure_eight": EXEMPT_ANALYTICAL_IC,
    "figure_eight_period": EXEMPT_ANALYTICAL_IC,
    "harmonic_oscillator": EXEMPT_ANALYTICAL_IC,
    "harmonic_solution": EXEMPT_ANALYTICAL_IC,
    "earth_sun_2body": EXEMPT_ANALYTICAL_IC,
    "earth_sun_eccentric": EXEMPT_ANALYTICAL_IC,
    "sun_earth_jupiter_3body": EXEMPT_ANALYTICAL_IC,
    "solar_system_inner_4": EXEMPT_ANALYTICAL_IC,
    "solar_system_full": EXEMPT_ANALYTICAL_IC,
    "SOLAR_SYSTEM_PLANETS": EXEMPT_ANALYTICAL_IC,
    "get_planet": EXEMPT_ANALYTICAL_IC,
    # --- EXEMPT_NON_FISHER_DIAGNOSTIC (diagnostics + energy/kinematic kernels, own tests) ---
    "relative_energy": EXEMPT_NON_FISHER_DIAGNOSTIC,
    "find_bound_pairs": EXEMPT_NON_FISHER_DIAGNOSTIC,
    "find_bound_multiples": EXEMPT_NON_FISHER_DIAGNOSTIC,
    "primordial_survival": EXEMPT_NON_FISHER_DIAGNOSTIC,
    "binary_energy_budget": EXEMPT_NON_FISHER_DIAGNOSTIC,
    "energy_sorted_segregation": EXEMPT_NON_FISHER_DIAGNOSTIC,
    "compute_kinetic_energy": EXEMPT_NON_FISHER_DIAGNOSTIC,
    "compute_potential_energy": EXEMPT_NON_FISHER_DIAGNOSTIC,
    "to_com_frame": EXEMPT_NON_FISHER_DIAGNOSTIC,
    "virial_scale": EXEMPT_NON_FISHER_DIAGNOSTIC,
    "compute_stellar_radii": EXEMPT_NON_FISHER_DIAGNOSTIC,
    # --- EXEMPT_HELPER (helpers / orientation / tidal utilities) ---
    "sample_isotropic_orientations": EXEMPT_HELPER,
    "batch_elements_to_resolved": EXEMPT_HELPER,
    "compute_period": EXEMPT_HELPER,
    "period_to_semimajor_axis": EXEMPT_HELPER,
    "fill_factor_to_r_h": EXEMPT_HELPER,
    "jacobi_radius": EXEMPT_HELPER,
    "jacobi_radius_isothermal": EXEMPT_HELPER,
    "apply_tidal_truncation": EXEMPT_HELPER,
    "sample_mass_dependent_orbits": EXEMPT_HELPER,
    # --- EXEMPT_COVERED_ELSEWHERE (differentiable samplers deferred from this arc's registry) ---
    "BinaryIMF": EXEMPT_COVERED_ELSEWHERE,
    "TruncatedIMF": EXEMPT_COVERED_ELSEWHERE,
    "FlatMassRatio": EXEMPT_COVERED_ELSEWHERE,
    "PowerLawMassRatio": EXEMPT_COVERED_ELSEWHERE,
    "TwinPeakedMassRatio": EXEMPT_COVERED_ELSEWHERE,
    "MoeDiStefano2017": EXEMPT_COVERED_ELSEWHERE,
    "MoeDiStefano2017Full": EXEMPT_COVERED_ELSEWHERE,
    "MoePeriod": EXEMPT_COVERED_ELSEWHERE,
    "MoeJointOrbit": EXEMPT_COVERED_ELSEWHERE,
    "ConstantBinaryFraction": EXEMPT_COVERED_ELSEWHERE,
    "MassDependentBinaryFraction": EXEMPT_COVERED_ELSEWHERE,
    "RadialBinaryFraction": EXEMPT_COVERED_ELSEWHERE,
    "CombinedBinaryFraction": EXEMPT_COVERED_ELSEWHERE,
    "LIMEPYProfile": EXEMPT_COVERED_ELSEWHERE,
    "LIMEPYVelocityDF": EXEMPT_COVERED_ELSEWHERE,
    "UniformSphereProfile": EXEMPT_COVERED_ELSEWHERE,
    "solve_king_profile": EXEMPT_COVERED_ELSEWHERE,
    "solve_michie_profile": EXEMPT_COVERED_ELSEWHERE,
    "solve_limepy_profile": EXEMPT_COVERED_ELSEWHERE,
    "solve_multimass_limepy": EXEMPT_COVERED_ELSEWHERE,
    "find_alpha_for_masses": EXEMPT_COVERED_ELSEWHERE,
    # Phase-B D4 targets — ALL promoted to AUDITED (B3-B6 cases landed; no holes remain).
    "LogisticThermalEccentricity": AUDITED,  # B5: DM91 circular->thermal blend ecc ceiling (e_max, Anna A1)
    # SanaOBPeriod / LogNormalPeriod / LogUniformPeriod promoted to AUDITED above (B4)
    "ThermalEccentricity": AUDITED,  # B5: thermal f(e)=2e ecc scale (e_max, <sqrt u>~2/3 closed-form)
    "UniformEccentricity": AUDITED,  # B5: uniform ecc upper bound (e_max, <u>~0.5 closed-form)
    "MoeEccentricity": AUDITED,  # B5: Moe+2017 ecc ceiling at long P (e_max, FD-target)
    "IndependentCompanions": AUDITED,  # B6: thermal e_max through the full companion assembly (FD-target; complements B5)
    # build_binary_cluster promoted to AUDITED above (B3)
}

# --- PARAM_ALLOWLIST: registry (id, param) legitimately not FD-consistent ----
PARAM_ALLOWLIST: dict[tuple[str, str], str] = {
    # The three alpha=1.0 IMF branch points formerly carried here
    # (PowerLawIMF.ppf[Salpeter]/mean_mass alpha, IMFParams.log_prob_nll alpha3)
    # were FIXED by audit S4 (expm1-stable segment kernels in progenax.numerics):
    # their edges are now expect="consistent" (AD FD-exact at exactly alpha=1),
    # so they no longer belong on the known-limitations carry-forward.
    (
        "binned_number_density[data, pinned non-diff]",
        "r_h",
    ): "frozen-edge count is a sum of "
    "indicators; AD=0 correct-by-design, the N(r) Fisher gradient lives in the model p_k",
    # du-monotonicity: the inverse-CDF samplers' grad wrt the frozen uniform draw u is a data-side
    # property out of the param-channel scope; pinned in TestBoundaryGradients, not a registry case.
}
