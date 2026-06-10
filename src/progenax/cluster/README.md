# `progenax.cluster` — multi-component equilibrium clusters + turbulence relations

Released-core tools for star-cluster initial conditions: the unified
**`MultiComponentCluster`** equilibrium model (N populations in ONE self-consistent shared
potential, via two engines — Engine A: DF-defined lowered-isothermal/LIMEPY family;
Engine B: density-defined Eddington inversion of Plummer/EFF/King components), the
**primordial** energy-ordered mass-segregation generator, plus the **turbulence scaling
relations** used to derive gas-cloud properties from cluster parameters.

Smooth single-population ICs are built with `progenax.build_spatial_ic` (any
`SpatialProfile` × `VelocityDF`). The legacy string-dispatch generator
(`generate_cluster_ic`/`ClusterState`) and the `lambda_seg` blend were retired in the
2026-06 unified redesign (pre-launch, no backwards compat).

> **Turbulent / fractal substructure ICs are not here.** The gravoturbulent density-PDF +
> fractal-density-field (FDF) pipeline was rebuilt clean-room in the experimental
> `gravoturb_fdf` package (a follow-up-paper feature, excluded from the released wheel).
> The legacy `fdf*`, `gravoturbulent`, and `gravoturb/` modules were removed in the
> 2026-06 clean-room rewrite. See `src/experimental/gravoturb_fdf/` and the per-paper notes
> under `docs/website/99-bibliography/per-paper/` (Burkhart & Mocz 2019, Parmentier &
> Pasquali 2020, Padoan & Nordlund 2011, Federrath 2010, Kim & Ryu 2005, Heyer 2009, Lomax 2018).

## Modules

| Module | Contents |
|--------|----------|
| `multicomponent.py` | `MultiComponentCluster` — the unified equilibrium model. Engine A constructors: `from_components`, `from_mass_segregation` (w_j = μ_j^(−δ)), `from_imf`. Engine B constructor: `from_density_profiles`. `sample_cluster` → `ICResult` with `component_id`; `component_virial_ratios` is the quadrature oracle. |
| `sampling.py` | JIT-compiled per-star sampling kernel for `sample_cluster` (Engines A and B). |
| `eddington_engine.py` | Engine B internals: shared-potential assembly, per-component Eddington inversion, realizability gate, `engine_b_component_virials`. |
| `mass_segregation.py` | `energy_sorted_segregation` — PRIMORDIAL energy-ordered orbit assignment (deterministic monotonic; documented departure from Baumgardt+2008's random per-bin draw). |
| `turbulence.py` | Federrath+2010 / Larson / Kim & Ryu turbulence relations (below). |
| `constants.py` | Turbulence constants (`B_DEFAULT`, `BETA_KOLMOGOROV`, `SIGMA_V0_DEFAULT`, `ALPHA_LARSON`, …). |

The Engine-A coupled-Poisson core (`solve_multicomponent_limepy` / `solve_multimass_limepy`,
`find_alpha_for_masses`) and the differentiable DF-table primitives live in
`progenax.profiles` (`limepy_multimass.py`, `limepy_tables.py`); the shared-potential
quadrature for Engine B is `progenax.profiles.density_poisson`.

## Turbulence relations (`turbulence.py`)

All JAX-native and differentiable. Pass a consistent unit system (M☉, pc, Myr/km·s⁻¹).

| Function | Relation | Reference |
|----------|----------|-----------|
| `cloud_radius_from_density(M_ecl, sfe, ρ_cl)` | `R = (3 M_gas / 4πρ_cl)^{1/3}` | geometry |
| `larson_sigma_v(R, σ_v0, α)` | `σ_v = σ_v0 (R/pc)^α` | Larson 1981 / Solomon+1987 |
| `turbulent_mach_from_cloud(R, c_s, …)` | `ℳ = σ_v(R)/c_s` | — |
| `sigma_ln_rho_from_mach(ℳ, b)` | `σ_s² = ln(1 + b²ℳ²)` | **FK10 Eq. 19** |
| `b_from_environment(log ρ_cl)` | solenoidal (1/3) → compressive (~0.7) interpolation | FK10 (tentative) |
| `spectral_slope_from_mach(ℳ)` | **density** spectrum `β(ℳ)=3.788−1.203·log₁₀ℳ`, clipped [2, 11/3] — *flattens* with Mach | **Kim & Ryu 2005** |

Note (2026-06 grounding): `σ_s²` is FK10 **Eq. 19** (not "Eq. 14", which is the Azzalini
skewed-lognormal). `spectral_slope_from_mach` returns the **density** power-spectrum slope,
which *decreases* with Mach (Kim & Ryu 2005) — it is **not** the velocity Kolmogorov/Burgers
slope, and it does **not** feed the IMF (the environment-dependent IMF slopes depend on
`ρ_cl` and `[Fe/H]`; see Marks+2012 / Jerabkova+2018).

## Substructure diagnostic

The CW04 `Q` parameter lives in `progenax.diagnostics.substructure.compute_q_parameter`
(numpy/scipy, non-differentiable), using the CW04 area convention `A = π R_cluster²`
(reproduces CW04 Table 1 to <0.01). A differentiable kNN approximation is in
`progenax.diagnostics.q_approx`.
