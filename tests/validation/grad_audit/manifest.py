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
AUDITED = "AUDITED"                                  # has >=1 registry case (in MUST_AUDIT)
EXEMPT_PROTOCOL = "EXEMPT_PROTOCOL"                  # a typing Protocol, not an entry point
EXEMPT_CONTAINER = "EXEMPT_CONTAINER"               # dataclass/PyTree container, not a sampler
EXEMPT_ANALYTICAL_IC = "EXEMPT_ANALYTICAL_IC"       # fixed-config analytic IC/test fixture
EXEMPT_NON_FISHER_DIAGNOSTIC = "EXEMPT_NON_FISHER_DIAGNOSTIC"  # diagnostic/energy kernel, own tests
EXEMPT_HELPER = "EXEMPT_HELPER"                      # helper/constant/orientation util
EXEMPT_COVERED_ELSEWHERE = "EXEMPT_COVERED_ELSEWHERE"  # differentiable sampler with a scattered
                                                       # FD test; deferred from THIS arc's registry
                                                       # scope (Tier-4 inventory "future candidate")

# --- MUST_AUDIT: (id, param) -> rationale -----------------------------------
# Seeded from REGISTRY (Phase A = current coverage only; D4 entries appended in Phase B).
# Every (id, param) coverage unit the registry MUST keep covering (56 units).
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
    ("EFFProfile.sample_positions", "gamma"): "EFF position sampler slope (near-divergent edge)",
    ("EFFProfile.sample_positions", "r_t"): "EFF position sampler truncation radius",
    ("EFFVelocityDF.sample_velocities", "a"): "EFF Eddington DF scale radius",
    ("EFFVelocityDF.sample_velocities", "gamma"): "EFF Eddington DF slope (virial point)",
    ("IMFParams.log_prob_nll", "alpha3"): "4-segment IMF NLL Fisher channel",
    ("KeplerElements.from_state", "v_scale"): "state->elements inverse, recovered a",
    ("KeplerElements.to_state", "M0"): "elements->Cartesian M0 column",
    ("KeplerElements.to_state", "a"): "elements->Cartesian a column",
    ("KeplerElements.to_state", "e"): "elements->Cartesian e column (near-parabolic edge)",
    ("KingProfile.r_t", "W0"): "King tidal radius differentiability in W0",
    ("KingProfile.sample_positions", "W0"): "King position sampler in W0 (W0=12 edge)",
    ("KingProfile.sample_positions", "r_c"): "King position sampler in core radius",
    ("KingVelocityDF.sample_velocities", "W0"): "King lowered-Maxwellian DF in W0",
    ("KingVelocityDF.sample_velocities", "r_c"): "King DF in core radius",
    ("Maschberger.ppf", "alpha"): "Maschberger inverse-CDF in slope",
    ("Maschberger.ppf", "beta"): "Maschberger inverse-CDF in beta",
    ("Maschberger.ppf", "mu"): "Maschberger inverse-CDF in mu",
    ("MichieProfile.density[log rho(r)]", "W0"): "Michie closed-form density obs in W0",
    ("MichieProfile.density[log rho(r)]", "r_c"): "Michie closed-form density obs in r_c",
    ("MichieProfile.r_t", "W0"): "Michie tidal radius differentiability in W0",
    ("MichieProfile.sample_positions", "W0"): "Michie anisotropic position sampler in W0",
    ("MichieVelocityDF.sample_velocities", "W0"): "Michie anisotropic DF in W0",
    ("MoeCompanions.sample", "m1_scale"): "Moe P-q-e companion sampler, <e> vs m1 scale",
    ("MultiComponentCluster.from_components[EngineA]", "w_j"): "Engine-A velocity-scale ratio (Fisher target)",
    ("MultiComponentCluster.sample_cluster[EngineA]", "W0"): "Engine A multimass in W0 (H2 edge)",
    ("MultiComponentCluster.sample_cluster[EngineA]", "delta"): "Engine A mass-segregation exponent",
    ("MultiComponentCluster.sample_cluster[EngineA]", "g"): "Engine A truncation sharpness",
    ("MultiComponentCluster.sample_cluster[EngineB]", "gamma"): "Engine B core EFF slope",
    ("MultiComponentCluster.sample_cluster[EngineB]", "r_a"): "Engine B halo OM anisotropy radius",
    ("MultiComponentCluster.sample_cluster[EngineB]", "r_h"): "Engine B halo Plummer scale",
    ("PlummerProfile.enclosed_mass_fraction[N(r) model]", "r_h"): "N(r) model Fisher column",
    ("PlummerProfile.sample_positions", "r_h"): "headline spatial sampler",
    ("PlummerVelocityDF+OM.sample_velocities", "r_a"): "Plummer OM anisotropy radius (edge)",
    ("PlummerVelocityDF.sample_velocities", "r_h"): "Plummer velocity DF in r_h",
    ("PowerLawIMF.cdf[H4]", "m_min"): "H4 clip probe, cdf vs m_min (benign)",
    ("PowerLawIMF.mean_mass", "alpha"): "analytic E[m] in slope (alpha=1 edge)",
    ("PowerLawIMF.ppf[Salpeter]", "alpha"): "Salpeter inverse-CDF in slope (alpha=1 edge)",
    ("PowerLawIMF.ppf[m_min]", "m_min"): "Salpeter inverse-CDF lower-support edge",
    ("PowerLawIMF.sample[Salpeter]", "alpha"): "Salpeter reparam sampler in slope",
    ("Schechter.ppf", "alpha"): "Schechter grid-CDF+Newton inverse in slope",
    ("apply_differential_rotation", "R_peak"): "differential rotation overlay, nonlinear R_peak",
    ("apply_differential_rotation", "v_peak"): "differential rotation overlay, linear v_peak",
    ("apply_solid_body_rotation", "omega"): "solid-body rotation overlay, linear omega",
    ("binned_number_density[data, pinned non-diff]", "r_h"): "frozen-count data side, pinned non-diff",
    ("binned_sigma1d[Plummer]", "r_h"): "binned sigma_1d Fisher channel in r_h",
    ("binned_sigma_beta[Plummer+OM]", "r_a"): "binned beta(r) anisotropy Fisher channel",
    ("build_spatial_ic[Plummer]", "r_h"): "headline end-to-end IC positions in r_h",
    ("build_spatial_ic[Plummer].velocities", "r_h"): "end-to-end IC velocities (virial-scaled) in r_h",
    ("lambda_msr_approx", "core_scale"): "soft Lambda_MSR segregation surrogate",
    ("q_approx[EFF]", "gamma"): "CW04 substructure-Q surrogate in EFF slope",
    ("resolve_binary_components", "a"): "binary->spatial connector, all-binary",
    ("resolve_binary_components[mixed]", "a"): "connector mixed mask (sanitization Fisher check)",
}

# --- SYMBOL_CATEGORY: every __all__ symbol -> category ----------------------
# Every progenax.__all__ symbol (104) categorized exactly once.
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
    "apply_solid_body_rotation": AUDITED,   # rotation overlay omega case (now root-exported)
    "apply_differential_rotation": AUDITED,  # rotation overlay v_peak/R_peak cases
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
    # Phase-B D4 targets — promote to AUDITED when each one's registry case lands.
    "LogisticThermalEccentricity": EXEMPT_COVERED_ELSEWHERE,  # Phase-B D4 target (Anna A1) — promote to AUDITED when case lands
    "SanaOBPeriod": EXEMPT_COVERED_ELSEWHERE,  # Phase-B D4 target — promote to AUDITED when case lands
    "LogNormalPeriod": EXEMPT_COVERED_ELSEWHERE,  # Phase-B D4 target — promote to AUDITED when case lands
    "LogUniformPeriod": EXEMPT_COVERED_ELSEWHERE,  # Phase-B D4 target — promote to AUDITED when case lands
    "ThermalEccentricity": EXEMPT_COVERED_ELSEWHERE,  # Phase-B D4 target — promote to AUDITED when case lands
    "UniformEccentricity": EXEMPT_COVERED_ELSEWHERE,  # Phase-B D4 target — promote to AUDITED when case lands
    "MoeEccentricity": EXEMPT_COVERED_ELSEWHERE,  # Phase-B D4 target — promote to AUDITED when case lands
    "IndependentCompanions": EXEMPT_COVERED_ELSEWHERE,  # Phase-B D4 target — promote to AUDITED when case lands
    "build_binary_cluster": EXEMPT_COVERED_ELSEWHERE,  # Phase-B D4 target — promote to AUDITED when case lands
}

# --- PARAM_ALLOWLIST: registry (id, param) legitimately not FD-consistent ----
PARAM_ALLOWLIST: dict[tuple[str, str], str] = {
    ("PowerLawIMF.ppf[Salpeter]", "alpha"): "alpha=1.0 edge is a branch-limited removable "
        "singularity (known_blocked); alpha=0.999 is FD-consistent",
    ("PowerLawIMF.mean_mass", "alpha"): "alpha=1.0 Z-denominator branch (known_blocked)",
    ("IMFParams.log_prob_nll", "alpha3"): "alpha3=1.0 branch-limited (known_blocked)",
    ("binned_number_density[data, pinned non-diff]", "r_h"): "frozen-edge count is a sum of "
        "indicators; AD=0 correct-by-design, the N(r) Fisher gradient lives in the model p_k",
    # du-monotonicity: the inverse-CDF samplers' grad wrt the frozen uniform draw u is a data-side
    # property out of the param-channel scope; pinned in TestBoundaryGradients, not a registry case.
}
