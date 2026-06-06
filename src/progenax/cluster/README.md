# `progenax.cluster` — cluster IC assembly + turbulence relations

Released-core tools for assembling star-cluster initial conditions from **smooth spatial
profiles** with optional **mass segregation**, plus the **turbulence scaling relations**
used to derive gas-cloud properties from cluster parameters.

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
| `core.py` | `generate_cluster_ic`, `ClusterState`, `SpatialStructureParams`, `sample_velocities_for_profile` — assemble a cluster IC (smooth profile + optional mass-segregation layer). |
| `mass_segregation.py` | `energy_sorted_segregation`, `MassSegregationLayer` — energy-ordered mass segregation. |
| `turbulence.py` | Federrath+2010 / Larson / Kim & Ryu turbulence relations (below). |
| `constants.py` | Turbulence constants (`B_DEFAULT`, `BETA_KOLMOGOROV`, `SIGMA_V0_DEFAULT`, `ALPHA_LARSON`, …). |
| `validation.py` | Plotting/validation helpers for the smooth + mass-seg path. |

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
