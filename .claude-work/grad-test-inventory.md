# Tier-4 gradient-test inventory (Task 4.1, READ-ONLY)

**Branch:** `feat/differentiability-audit` · **Date:** 2026-06-13 · **Author:** grad-audit Tier-4 inventory pass

## What this is

The grad-audit registry (`tests/validation/grad_audit/registry.py`, **57 `Case`s** across
`params->IC` and `params->summary`) is the single source of truth for gradient correctness, and is
becoming the release gradient-gate. Tier 4 consolidates the scattered ad-hoc gradient tests INTO it.
This document catalogs **every released-core test whose ASSERTION is about a gradient** (autodiff
finiteness, AD-vs-FD consistency, gradient sign/value, differentiability-through-a-path) and
classifies each. **No test has been changed** — this is the sign-off document for CHECKPOINT 4a.

Scope: `tests/unit`, `tests/integration`, `tests/validation`. Excludes `tests/experimental/`
(gravoturb_fdf) and `tests/validation/grad_audit/` (the harness itself).

### Classification definitions

- **migrate** — asserts an AD-vs-FD / autodiff-correctness property the registry should own. In
  Task 4.2 the duplicate assertion is removed (and a NEW registry Case added if the registry does
  not already have an equivalent/stronger one).
- **keep** — asserts a NON-gradient physics property, OR a UNIQUE gradient property the registry
  genuinely doesn't (and shouldn't) cover (custom_vjp/IFT regression, α=1 kink pin, Engine-B β(r)
  anchor, energy-utility double-where guards, mass-ratio / environment-IMF / DifferentiableBinary
  channels the registry has no Case for, internal-function differentiability pins). In Task 4.2 these
  get a one-line comment pointing to the registry as the gradient source-of-truth.
- **delete** — a redundant FINITE-ONLY smoke test (`assert jnp.isfinite(jax.grad(f)(x))`, no FD
  comparison, no unique property) the registry now covers more rigorously. **This is audit finding
  T6** — the finite-only smoke tests gave false confidence (a silently-zero gradient passes them).

### Registry coverage index (the 57 Cases, by entry point)

`PlummerProfile.sample_positions(r_h)`, `PlummerVelocityDF.sample_velocities(r_h)`,
`PlummerVelocityDF+OM(r_a)`, `KingProfile.sample_positions(W0; edge W0=12)`,
`KingProfile.sample_positions(r_c)`, `KingVelocityDF.sample_velocities(W0)`, `KingProfile.r_t(W0)`,
`MichieProfile.sample_positions(W0)`, `MichieProfile.r_t(W0)`,
`MichieVelocityDF.sample_velocities(W0)`, `EFFProfile.sample_positions(gamma; edge 2.01)`,
`EFFProfile.sample_positions(r_t)`, `EFFVelocityDF.sample_velocities(gamma)`,
`build_spatial_ic[Plummer](r_h)` positions + velocities,
`PowerLawIMF.ppf(alpha; edges 0.999, 1.0=known_blocked)`, `PowerLawIMF.sample(alpha)`,
`ChabrierIMF.ppf(m_c / sigma / alpha; H6 edge)`, `Maschberger.ppf(mu / alpha / beta)`,
`Schechter.ppf(alpha)`, `PowerLawIMF.cdf[H4](m_min)`, `PowerLawIMF.mean_mass(alpha; α=1 blocked)`,
`IMFParams.log_prob_nll(alpha3; α3=1 blocked)`, `q_approx[EFF](gamma)`,
`lambda_msr_approx(core_scale)`, `MultiComponentCluster.sample_cluster[EngineA](W0 / g / delta)`,
`MultiComponentCluster.sample_cluster[EngineB](r_h / gamma / r_a)`, `KeplerElements.to_state(e; edge
0.999)`, `resolve_binary_components(a)` + `[mixed]`, `MoeCompanions.sample(m1_scale)`,
`apply_solid_body_rotation(omega)`, `apply_differential_rotation(v_peak / R_peak)`,
`binned_sigma1d[Plummer](r_h)`, `binned_sigma_beta[Plummer+OM](r_a)`,
`PlummerProfile.enclosed_mass_fraction[N(r) model](r_h)`, `binned_number_density[data, pinned]`.

---

## Inventory table

| File::test(s) | Asserts | Category | Registry coverage | Reasoning / action for 4.2 |
|---|---|---|---|---|
| `unit/imf/test_imf_gradients.py::TestFDvsAutodiff` (8 tests: chabrier mc/sigma/alpha, maschberger mu/alpha/beta, schechter alpha, powerlaw exponent) | AD ppf-param grad == central FD, finite, non-zero | **migrate** | Already covered (ChabrierIMF.ppf, Maschberger.ppf, Schechter.ppf, PowerLawIMF.ppf[Salpeter]) | Registry has equal/stronger FD Cases for all 8. Remove the duplicate FD asserts; replace file-level with a comment pointing to registry. |
| `unit/imf/test_imf_gradients.py::test_powerlaw_ppf_grad_mmin` | AD d(ppf)/d(m_min) == FD | **migrate** | NEW case needed: `PowerLawIMF.ppf(m_min)` (registry has `cdf[H4](m_min)` and `ppf(alpha)`, not `ppf(m_min)`) | Either add a `ppf(m_min)` Case or fold into existing m_min coverage; FD assertion is registry-shaped. |
| `unit/imf/test_imf_gradients.py::TestBoundaryGradients::test_grad_finite_at_boundary` | d(ppf)/du finite at u→0/u→1 (NaN-grad sqrt trap) | **keep** | n/a (du boundary, not a param channel) | UNIQUE: boundary NaN-grad guard wrt the *uniform draw*, not a parameter. Registry audits param channels. Keep; add pointer comment. |
| `unit/imf/test_imf_gradients.py::TestParameterGradients` (3 tests: chabrier alpha/sigma, maschberger mu) | grad finite + non-zero (NO FD) | **delete** | Covered by ChabrierIMF.ppf / Maschberger.ppf Cases | Pure finite-only smoke (T6); strict subset of the FD Cases above and of TestFDvsAutodiff in the same file. |
| `unit/imf/test_imf_gradients.py::TestAlphaOneGradients` (4 tests: finite@α=1, FD-exact@α±1e-3, value smooth, sample finite@α=1) | finiteness AND smoothness at the α=1 removable singularity | **keep** | Registry pins `PowerLawIMF.ppf`/`mean_mass` α=1 as `known_blocked` edges | UNIQUE α=1 KINK PIN (audit R10). Registry asserts blocked-but-finite; this file asserts the *forward value is kink-free* + sample-stat finite — a complementary property the registry should NOT absorb. **KEEP-do-not-touch.** |
| `unit/kinematics/test_df_gradients.py::TestPlummerDFGradients::test_grad_wrt_r_h_matches_fd` | AD d⟨\|v\|²⟩/dr_h == FD | **migrate** | Already covered (PlummerVelocityDF.sample_velocities, r_h) | Duplicate of registry FD Case. Remove FD assert. |
| `unit/kinematics/test_df_gradients.py::TestKingDFGradients::test_grad_matches_fd` (r_c, W0) | AD d⟨\|v\|²⟩/d{r_c,W0} == FD | **migrate** | W0 covered (KingVelocityDF.sample_velocities, W0). r_c: **NEW case needed** `KingVelocityDF.sample_velocities(r_c)` | W0 is a duplicate; r_c-of-the-DF is a gap (registry only has King r_c on the *profile*). Add r_c DF Case, migrate. |
| `unit/kinematics/test_df_gradients.py::TestEFFDFGradients::test_grad_matches_fd` (a, gamma) | AD d⟨\|v\|²⟩/d{a,gamma} == FD | **migrate** | gamma covered (EFFVelocityDF.sample_velocities, gamma). a: **NEW case needed** `EFFVelocityDF.sample_velocities(a)` | gamma duplicate; EFF `a` scale on the DF is a gap. Add `a` Case, migrate. |
| `unit/kinematics/test_df_gradients.py::*::test_jit_compatible` (3) | JIT runs, finite output | **keep** | n/a (JIT, not grad) | Not a gradient assertion. Out of Tier-4 scope; leave as-is. |
| `unit/profiles/test_king_grad.py::test_from_W0_rc_is_jittable`, `::test_from_W0_rc_jit_matches_eager` | JIT-safety / concretization (audit C2 symptom 1) | **keep** | n/a (JIT) | Not gradient. Keep. |
| `unit/profiles/test_king_grad.py::test_grad_of_psi_through_solve_king_profile_is_finite` | grad(Σψ)/dW0 finite through King ODE (C2 symptom 2, W=0 double-where) | **keep (verify)** | Indirectly under KingProfile cases | UNIQUE: pins the W=0 Poisson-RHS double-where keeps the ODE grad finite — an *internal* guard the public-entry Cases exercise but don't isolate. Conservative keep; add pointer. |
| `unit/profiles/test_profile_gradients.py::TestSamplerGradients` (plummer r_h, eff a, eff gamma, king r_c, king W0) | AD mean-radius grad == FD | **migrate** | plummer r_h, eff gamma, eff r-scale(≈a via r_t? no), king r_c, king W0 all covered by profile Cases | All five are duplicates of registry profile `sample_positions` FD Cases (EFF `a` ≈ the registry's EFF scale; king W0/r_c covered). Migrate/remove FD asserts. |
| `integration/test_jax_compatibility.py::TestProfileJAXCompatibility::test_plummer_jit` | JIT finite | **keep** | n/a (JIT) | Not grad. Keep (JIT smoke). |
| `integration/test_jax_compatibility.py::TestProfileJAXCompatibility::test_plummer_grad` | grad finite + sign>0 (NO FD) | **delete** | Covered (PlummerProfile.sample_positions, r_h) | Finite+sign smoke (T6); registry FD Case is strictly stronger. |
| `integration/test_jax_compatibility.py::TestVelocityDFJAXCompatibility::test_plummer_df_jit` | JIT finite | **keep** | n/a (JIT) | Keep (JIT smoke). |
| `integration/test_jax_compatibility.py::TestVelocityDFJAXCompatibility::test_plummer_df_grad` | grad finite (NO FD) | **delete** | Covered (PlummerVelocityDF.sample_velocities, r_h) | Finite-only smoke (T6). |
| `integration/test_jax_compatibility.py::TestIMFJAXCompatibility::test_powerlaw_jit/_vmap` | JIT/vmap finite | **keep** | n/a | Not grad. Keep. |
| `integration/test_jax_compatibility.py::TestIMFJAXCompatibility::test_powerlaw_grad` | grad>0 wrt **u** (NO FD) | **keep (verify)** | n/a (du, not a param) | Grad wrt the uniform draw (monotonic ppf), not a parameter — outside registry's param-channel scope. Conservative keep; could delete if Anna deems the du-monotonicity trivial. |
| `integration/test_jax_compatibility.py::TestPipelineDifferentiability::test_plummer_ic_grad_wrt_r_h` | grad finite + sign<0 (NO FD) | **delete** | Covered (build_spatial_ic[Plummer], r_h velocities channel) | Finite+sign smoke (T6); the build_spatial_ic FD Cases are stronger. |
| `integration/test_jax_compatibility.py::test_build_spatial_ic_differentiable_wrt_r_h` | AD grad == FD through public build_spatial_ic (audit CR-FU-2) | **migrate** | Already covered (build_spatial_ic[Plummer], r_h) | Duplicate of registry FD Case. Migrate (this is the CLAUDE.md headline path — keep ONE pointer comment to the registry Case). |
| `integration/test_jax_compatibility.py::test_compute_potential_energy_grad_finite_at_default_softening` | AD grad == FD at softening=0 (double-where) | **keep** | n/a (energy utility, not an IC/summary entry point) | UNIQUE: `compute_potential_energy` softening=0 double-where guard. Registry audits IC/summary, not the energy kernel. **KEEP.** |
| `unit/profiles/test_find_alpha_ift.py::TestForwardRegression` (2) | forward α sums-to-1 + REF_ALPHA pinned | **keep** | n/a (forward value) | UNIQUE forward IFT regression. **KEEP-do-not-touch.** |
| `unit/profiles/test_find_alpha_ift.py::TestGradientMatchesFD` (grad_alpha_imf, grad_delta_W0) | IFT custom_vjp grad == FD @ rtol 1e-6 | **keep** | n/a | UNIQUE: the find_alpha custom_vjp/IFT exact-fixed-point gradient. The plan explicitly names this a keep. **KEEP-do-not-touch.** |
| `unit/profiles/test_find_alpha_ift.py::TestJitGradient` (3) | jit(grad)/jit(value_and_grad) == eager (C1 tracer regression) | **keep** | n/a | UNIQUE jit-grad regression for the IFT path. **KEEP.** |
| `unit/profiles/test_find_alpha_ift.py::TestAnisoGradientQuadrature` (2, slow) | aniso IFT grad == FD (quadrature path) | **keep** | n/a | UNIQUE anisotropic IFT grad-correctness. **KEEP.** |
| `validation/test_engine_b_physics.py::test_gradients_ad_vs_fd` (halo r_h, mass-fraction t, r_a) | AD == FD through full Engine B build (3 params) | **keep (verify) / partial-migrate** | Partly: Engine B `r_h`, `r_a` covered via `sample_cluster[EngineB]`; **mass-fraction t and the King-branch build path are a GAP** | The registry Engine-B Cases audit `sample_cluster` over Plummer+EFF; THIS test audits `build_engine_b_state` directly incl. the KING density branch + mass-fraction t. Keep the King/mass-fraction coverage (unique build-graph path); the r_h/r_a parts duplicate the registry. Recommend: keep as a build-graph anchor with a registry pointer; consider adding an Engine-B `mass_fraction` Case (see GAPS). |
| `validation/test_engine_b_physics.py::test_king_density_engine_b_matches_engine_a`, `test_eff_gamma5_single_component_matches_plummer`, `test_plummer_halo_eff_core_equilibrium`, `test_om_beta_profile_realized`, `test_df_density_fidelity_interior` | Q_j, σ₁d, KS, β(r) physics anchors (NOT gradients) | **keep** | n/a (physics) | The Engine-B β(r) anchor + cross-engine physics. **KEEP-do-not-touch (β anchor).** |
| `unit/profiles/test_limepy_tables.py::test_gradient_finite_at_W_nonpositive` | dV/dW finite at W=0/W=-1 (C1 sqrt(max) cotangent) | **keep** | n/a (internal table) | UNIQUE W≤0 NaN-grad guard on the internal density table. Keep. |
| `unit/profiles/test_limepy_tables.py::test_differentiable_in_g_and_queries` | AD dg == FD through the table BUILD | **keep (verify)** | n/a (internal `AnisoDensityTable.build`) | Internal-function FD grad-check below public-entry granularity; underpins Michie/Engine-A Cases but tests the table directly. Conservative keep; add pointer. |
| `unit/profiles/test_limepy_tables.py::test_table_solve_differentiable_in_rescale_ra` (line 194) | grad finite + non-zero (NO FD) | **keep (verify)** | n/a (internal `solve_multicomponent_limepy`) | Internal-solver finite-only smoke. Not a public entry point; the Engine-A registry Cases cover the public path. Conservative keep (could delete as T6 if Anna treats the internal solver as registry-covered). |
| `unit/profiles/test_limepy_tables.py::test_differentiable_through_table_draw`, `::test_differentiable` (SpeedCDFTable / AnisoSpeedCDFTable) | grad finite + scale-grad>0 (NO FD) | **keep (verify)** | n/a (internal speed-CDF tables) | Internal table-draw finite-only smoke; below public-entry granularity. Conservative keep. |
| `validation/test_michie_physics.py::TestMichieDifferentiability::test_grad_profile_observable` (W0, r_c) | AD d(log density)/d{W0,r_c} == FD | **migrate** | W0 covered (MichieProfile.sample_positions, W0). The observable is *density* not sampled radius; r_c is a GAP | Density-observable FD-grad. W0 overlaps; consider a Michie density-observable or r_c Case. Migrate the FD assertion. |
| `validation/test_michie_physics.py::TestMichieDifferentiability::test_grad_wrt_mass_velocity_scale` | AD dσ/dM == FD (closed-form σ∝√M) | **keep** | n/a | Trivial closed-form σ(M) scaling, not a sampler/DF graph. Keep (cheap physics-scaling pin). |
| `validation/test_michie_physics.py` (other tests: density/beta/virial/bound) | physics (NOT grad) | **keep** | n/a | Physics. Keep. |
| `validation/test_king_physics.py::test_velocity_sampling_is_differentiable` | grad finite through King DF (NO FD) | **delete** | Covered (KingVelocityDF.sample_velocities, W0; r_c via NEW DF Case above) | Finite-only smoke (T6). |
| `validation/test_king_physics.py::TestKingAutoDomain::test_auto_domain_preserves_differentiability_high_W0` | AD == FD through auto-domain high-W0 profile (r_c) | **keep (verify)** | KingProfile.sample_positions(r_c) covered, but at DEFAULT domain (W0=7); this is the **auto-domain W0=12** path | FD-grad but specifically pins the AUTO-DOMAIN high-W0 codepath (registry r_c Case uses W0=7 auto-size; the W0=12 edge is on a *separate* Case with explicit domain). Conservative keep — it guards a distinct domain-selection branch. Add pointer. |
| `validation/test_king_physics.py` (other tests) | physics + Table-II concentration | **keep** | n/a | Physics. Keep. |
| `unit/profiles/test_king.py::test_r_t_grad_is_nonzero_and_fd_consistent` | AD d(r_t)/dW0 == FD, large (silent-zero gone) | **migrate** | Already covered (KingProfile.r_t, W0) | Duplicate of registry r_t Case (same xi_max=400/n=8000 config). Migrate/remove. |
| `unit/profiles/test_king_density.py::test_density_zero_and_gradient_finite_at_zero` | density(0)=0 + grad finite at W=0 | **keep** | n/a (internal king_lowered_maxwellian_density) | UNIQUE internal W=0 double-where guard. Keep. |
| `unit/profiles/test_density_poisson.py::TestKingDrhoDW::test_closed_form_matches_autodiff` | closed-form `_king_drho_dW` == jax.grad of density @ rtol 1e-10 | **keep** | n/a | UNIQUE: pins a *closed-form derivative* against autodiff (the ground truth is AD here, reversed). Keep. |
| `unit/profiles/test_profile_potential.py::test_jit_safe_and_grad_wrt_gamma` | JIT-safe + grad wrt gamma finite+non-zero (NO FD) | **keep** | n/a (`compute_profile_potential`, not an IC/summary entry point) | UNIQUE: the `if gamma==3.0` Tracer regression on the *potential* utility. Not a registry channel. Keep. |
| `unit/profiles/test_plummer.py::test_differentiable_in_r` | d(M(<r))/dr finite+positive (NO FD) | **keep** | n/a (grad wrt r, not a param) | enclosed_mass_fraction grad wrt the *query radius*, not r_h. Registry N(r) Case is d/dr_h. Keep. |
| `unit/profiles/test_plummer.py::test_differentiable_in_r_h` | d(M(<r_fixed))/dr_h finite + sign<0 (NO FD) | **keep (verify)** | Registry `enclosed_mass_fraction[N(r) model]` audits d(p_k)/dr_h via FD | Same method, same param (r_h), but finite+sign only and a single radius vs the registry's frozen-edge shell vector. Borderline T6; conservative keep (registry Case is the FD source-of-truth — add pointer; Anna may downgrade to delete). |
| `unit/profiles/test_plummer.py::TestPlummerDifferentiability::test_gradient_through_r_h` | grad finite through sampling (NO FD) | **delete** | Covered (PlummerProfile.sample_positions, r_h) | Finite-only smoke (T6). |
| `unit/substructure/test_q_approx.py::test_grad_compatible` | grad wrt positions finite + shape (NO FD) | **keep** | Registry `q_approx[EFF]` audits d/dgamma | grad wrt the *position array* (not a param), checks shape+finite. Complementary to the registry param-channel Case. Keep; add pointer. |
| `validation/test_substructure_q_physics.py::test_differentiable_wrt_concentration` | AD == FD wrt a concentration param (few-%) | **migrate** | Registry `q_approx[EFF](gamma)` is the equivalent concentration channel | The registry gamma Case is the canonical concentration FD-grad for q_approx. Migrate (remove the duplicate FD assert; keep the physics-ordering tests in the file). |
| `unit/test_numerics.py::test_differentiable` (cumulative_trapezoid) | grad finite+>0 (NO FD) | **keep** | n/a (numeric primitive) | UNIQUE low-level primitive used everywhere. Keep. |
| `unit/test_numerics.py::test_differentiable_in_weight_parameter` (inverse_cdf_draw) | grad finite (NO FD) | **keep** | n/a (numeric primitive) | UNIQUE inverse-CDF primitive grad pin. Keep. |
| `unit/test_tidal.py::test_retained_mass_grad_wrt_r_t_finite_positive` | grad wrt r_t finite+>0 (NO FD) | **keep** | n/a (tidal, no registry Case) | UNIQUE: straight-through tidal-truncation surrogate (the capability a hard cut lacks). No registry coverage. Keep. |
| `unit/test_tidal.py::test_surrogate_matches_logistic_derivative` | grad == analytic logistic derivative | **keep** | n/a | UNIQUE closed-form surrogate-derivative pin. Keep. |
| `validation/test_tidal_physics.py::test_bound_mass_differentiable_in_rt` | grad wrt r_t finite+>0 (NO FD) | **keep (verify)** | n/a | Near-duplicate of the unit tidal grad test; both unique-to-tidal (no registry Case). Conservative keep. |
| `integration/test_binary_cluster.py::test_grad_through_r_h` | grad through build_binary_cluster finite+>0 (NO FD) | **keep (verify)** | n/a (build_binary_cluster — no registry Case) | The binary-cluster assembly path is NOT a registry entry point (registry has resolve_binary_components + MoeCompanions, not the full build_binary_cluster). Finite-only but on an uncovered path. Conservative keep; flag as a GAP candidate. |
| `integration/test_end_to_end.py::test_imf_ppf_gradient` | grad wrt **u** finite+>0 (NO FD) | **delete** | du-monotonicity; same as test_jax_compatibility's | Finite+sign smoke wrt the uniform draw (T6); trivially duplicated. (Conservative alt: keep one du-monotonicity pin — Anna's call.) |
| `integration/test_end_to_end.py::test_spatial_profile_gradient` | grad wrt r_h finite (NO FD) | **delete** | Covered (PlummerProfile.sample_positions, r_h) | Finite-only smoke (T6); duplicate of registry + test_jax_compatibility. |
| `unit/binaries/test_assembly.py::test_jit_and_grad_safe` | JIT finite + grad d\|sep\|/da finite+>0 (NO FD) | **keep (verify)** | Registry `resolve_binary_components(a)` covers the FD grad | The grad half is a finite-only subset of the registry's resolve_binary FD Case → migrate-able; the JIT half is unique. Conservative keep the JIT assert; the grad assert is a delete/migrate candidate. Split in 4.2. |
| `unit/binaries/test_binaries.py::test_compute_period_grad_finite_at_zero`, `::test_period_to_a_grad_finite_at_zero` | grad finite at a=0 / P=0 (divide-safe sqrt/cbrt) | **keep** | n/a (compute_period / period_to_a — no registry Case) | UNIQUE: divide-by-zero-safe grad at the a→0/P→0 boundary on the period helpers. No registry coverage. Keep. |
| `unit/binaries/test_companions.py::test_grad_fd_accurate_eccentricity` | AD d⟨e⟩/d(e_max) == FD (IndependentCompanions) | **keep (verify)** | Registry `MoeCompanions.sample(m1_scale)` audits ⟨e⟩ wrt m1, NOT e_max of IndependentCompanions | Different model (IndependentCompanions) and different param (e_max). FD-grad but a channel the registry doesn't have. Conservative keep; candidate NEW Case if Anna wants e_max in the gate. |
| `unit/binaries/test_diagnostics.py::test_differentiable` (relative_energy) | grad finite (NO FD) | **keep** | n/a (relative_energy diagnostic) | UNIQUE diagnostic grad pin; no registry Case. Keep. |
| `unit/binaries/test_energy_budget.py::test_grad_dE_internal_da` | AD dE_int/da == analytic (vis-viva) | **keep** | n/a (binary_energy_budget) | UNIQUE: grad vs CLOSED-FORM analytic (not FD). No registry Case. Keep. |
| `unit/binaries/test_fraction_unification.py::test_jit_and_grad` | JIT finite + grad finite+>0 (NO FD) | **keep** | n/a (CombinedBinaryFraction) | UNIQUE binary-fraction-model channel; no registry Case. Keep (JIT+grad smoke on an uncovered path). |
| `unit/binaries/test_population.py::test_sample_grad_finite_wrt_period` | grad finite (NO FD) | **keep (verify)** | n/a (MoeEccentricity) | Uncovered channel (period→e). Finite-only but no registry Case. Conservative keep. |
| `unit/binaries/test_population.py::test_emax_gradient_matches_fd`, `::test_logistic_thermal_emax_gradient_matches_fd`, `::test_uniform_eccentricity_emax_gradient`, `::test_sana_power_gradient_matches_finite_difference`, `::test_lognormal_location_gradient_is_unity`, `::test_thermal_scale_gradient_equals_mean_sqrt_u`, `::test_loguniform_period_logpmax_gradient` | AD == FD / AD == analytic for the period & eccentricity distribution params | **keep** | n/a (SanaOBPeriod, LogNormalPeriod, ThermalEccentricity, UniformEccentricity, MoeEccentricity, LogUniformPeriod) | UNIQUE: each is a distinct orbital-distribution param channel the registry does NOT cover (registry has only MoeCompanions.⟨e⟩-wrt-m1). These are genuine FD/analytic grad-correctness pins on uncovered entry points. **KEEP** (candidate future registry Cases, but not duplicates). |
| `unit/binaries/test_population.py::test_lognormal_ppf_grad_finite_at_boundary`, `::test_thermal_ppf_grad_finite_at_boundary`, `::test_grad_finite_at_power_minus_one` | grad finite at u-boundary / power=-1 singularity (NO FD) | **keep** | n/a | UNIQUE boundary/removable-singularity grad guards on the period/ecc ppfs. Keep. |
| `unit/imf/test_binary.py::TestBinaryGradients` (powerlaw_ppf gamma, twinpeaked f_twin/sigma, diffmodel gamma_intercept) | AD ppf-param grad == FD | **keep** | n/a (PowerLawMassRatio, TwinPeakedMassRatio, DifferentiableBinaryModel — no registry Case) | UNIQUE mass-ratio / differentiable-binary channels. Registry has no mass-ratio sampler Case. **KEEP** (FD-grade, uncovered entry points). |
| `unit/imf/test_differentiable_binary.py::test_differentiable_wrt_a/_b`, `::test_gradient_wrt_temperature`, `::test_gradient_wrt_binary_fraction_params` | grad finite (+non-zero) (NO FD) | **keep (verify)** | n/a (DifferentiableBinaryFraction / DifferentiableBinaryModel) | Uncovered channel; finite-only but no registry Case. Conservative keep. |
| `unit/imf/test_differentiable.py::test_gradient_through_alpha3`, `::test_gradient_through_params`, `::test_gradient_wrt_alpha3` | grad finite + non-zero through log_prob (NO FD) | **keep (verify)** | Registry `IMFParams.log_prob_nll(alpha3)` covers the FD grad | Finite-only subset of the registry's IMFParams Fisher Case → migrate/delete-candidate. Conservative keep pending Anna (the registry Case is the FD source-of-truth — add pointer). |
| `unit/imf/test_differentiable.py::test_grad_finite_sweeping_alpha1_through_one` | grad(NLL)/dα1 finite sweeping α1 through 1 | **keep** | Registry pins α3=1 known_blocked (different segment) | UNIQUE: α1=1 (segment-1) removable-singularity finiteness — a different α-segment than the registry's α3 pin. Keep (α=1 kink family). |
| `unit/imf/test_params.py::test_grad_through_alpha3` | grad of x² == 2x (trivial) | **delete** | n/a | Trivial PyTree-leaf grad sanity (d/dx x²=4.6); no IMF graph, no value. Pure smoke. (Conservative alt: keep as a 1-line PyTree sanity — harmless. Lean delete.) |
| `unit/imf/test_smooth.py::test_grad_flows_through_ppf` (TaperedPowerLaw) | AD == FD wrt alpha | **keep** | n/a (TaperedPowerLaw — no registry Case; Schechter≠TaperedPowerLaw) | UNIQUE TaperedPowerLaw ppf FD-grad. Keep (FD-grade, uncovered). |
| `unit/imf/test_moe_full.py::test_grad_finite` (×2: MoeDiStefano2017Full, MoePeriod) | grad finite (NO FD) | **keep (verify)** | n/a (MoeDiStefano2017Full, MoePeriod) | Uncovered channel; finite-only. Conservative keep. |
| `unit/imf/test_moe_full.py::test_grad_fd_accurate` | AD == FD (MoeDiStefano2017Full grid-CDF reparam) | **keep** | n/a | UNIQUE: pins the grid-CDF reparameterization grad (the mixture-weight gradient). No registry Case. Keep. |
| `unit/imf/test_environment.py` (TestEnvGradients ×6 FD; gradient_env_to_likelihood; gradient_direction; smooth_feh_branch; smooth tanh ×4; jerabkova/marks/x grads) | AD == FD / finite / sign for the environment-IMF α₃ relations | **keep** | n/a (BirthEnvironment / env_to_imf_params / alpha3_* — no registry Case) | UNIQUE environment-dependent-IMF channel (Marks+2012 / Jeřábková+2018). Registry has NO environment channel. **KEEP** (FD-grade + smooth-tanh-branch guards, all uncovered). |
| `unit/cluster/test_multicomponent.py::test_differentiable_in_w_j` | grad finite + non-zero (NO FD) | **keep (verify)** | Engine-A registry Cases cover W0/g/delta, NOT w_j directly | w_j (direct velocity-scale ratio) is a from_components param not in the registry's from_imf Cases. Finite-only on a partially-uncovered channel. Conservative keep; flag as GAP. |
| `unit/cluster/test_multicomponent.py::test_sample_differentiable_in_delta` | grad finite (NO FD) | **delete** | Covered (sample_cluster[EngineA], delta) | Finite-only smoke (T6); registry delta Case (from_imf) is the FD source-of-truth. (Note: this uses from_mass_segregation — same delta channel.) Conservative-leaning delete; mark verify. |
| `unit/cluster/test_multicomponent.py::test_aniso_sample_differentiable_in_ra` | grad finite (NO FD) | **keep (verify)** | Engine-A registry has W0/g/delta; r_a on from_mass_segregation is NOT covered (Engine-B r_a is a different build) | Finite-only but the Engine-A anisotropy-radius channel is uncovered. Conservative keep; flag GAP. |
| `unit/cluster/test_multicomponent.py::test_engine_a_jit_and_grad_still_flow` | grad through r_c via grouped state finite (NO FD) | **keep** | n/a (pytree/grouping regression) | UNIQUE: grouping-must-not-break-the-pytree grad regression. Keep. |
| `unit/diagnostics/test_segregation_approx.py::test_differentiable_in_m_cut` (soft_mass_weights) | grad finite (NO FD) | **keep** | n/a (soft_mass_weights — m_cut, not registry's core_scale) | Internal weight-function grad; registry `lambda_msr_approx` audits core_scale not m_cut. Keep. |
| `unit/diagnostics/test_segregation_approx.py::test_differentiable_in_m_cut_matches_fd` (×3: radial_concentration_approx, lambda_msr_approx, sigma_m_approx) | AD == FD wrt m_cut | **keep (verify)** | Registry `lambda_msr_approx(core_scale)` audits a DIFFERENT param (core_scale) of ONE of the three | m_cut FD-grad on the segregation surrogates. Registry covers lambda_msr core_scale only; m_cut and the radial/sigma surrogates are uncovered. Keep the radial/sigma ones (unique); the lambda_msr m_cut one is borderline (different param than registry) — keep. Flag potential NEW m_cut Case. |
| `unit/dynamics/test_virial.py::test_grad_fd_match` (compute_kinetic_energy) | AD == FD | **keep** | n/a (energy utility) | UNIQUE energy-kernel grad; registry audits IC/summary not energy. Keep. |
| `unit/dynamics/test_virial.py::test_grad_fd_match_softened`, `::test_grad_finite_at_zero_softening_with_padding` (compute_potential_energy) | AD == FD softened / finite at soft=0 with padding | **keep** | n/a (energy utility, double-where + padding guard) | UNIQUE: the PE double-where softening=0 + padded-zero-row guard. **KEEP.** |
| `unit/dynamics/test_energy_consolidation.py::test_potential_energy_softening_zero_grad_is_finite` | grad finite at soft=0 (NO FD) | **keep (verify)** | n/a (energy utility) | Near-duplicate of test_virial's soft=0 guard (different module path: dynamics.virial). Both unique-to-energy-kernel. Conservative keep. |
| `unit/dynamics/test_group_virial.py::test_differentiable` (per_group_virial_ratio) | grad finite (NO FD) | **keep** | n/a (group-virial diagnostic) | UNIQUE diagnostic grad; no registry Case. Keep. |
| `unit/dynamics/test_group_virial.py::test_grad_finite_at_zero_softening` (_accelerations) | grad finite at soft=0 (NO FD) | **keep** | n/a (acceleration kernel) | UNIQUE accel-kernel double-where guard. Keep. |
| `unit/kinematics/test_api.py::test_pipeline_is_differentiable_in_target_Q` | grad finite + sign>0 wrt target_Q (NO FD) | **keep (verify)** | n/a (sample_velocities_pipeline / VelocityModel.target_Q — no registry Case) | target_Q channel through the pipeline is uncovered (registry virial-scales internally but doesn't expose target_Q as an audited param). Finite+sign only. Conservative keep; flag GAP. |
| `unit/kinematics/test_limepy_df.py::test_differentiable_in_g_through_table`, `::test_velocity_sampling_differentiable_in_g` | grad finite + non-zero wrt g (NO FD) | **keep (verify)** | n/a (LIMEPYVelocityDF — g channel not in registry; Michie/King W0 are) | LIMEPYVelocityDF g channel uncovered (registry audits King/Michie/EFF DFs, not the generic LIMEPY g). Finite-only. Conservative keep; flag GAP. |
| `unit/kinematics/test_michie_df.py::test_grad_wrt_W0_matches_fd` | AD == FD wrt W0 (Michie ODE + 2-D sampler) | **migrate** | Already covered (MichieVelocityDF.sample_velocities, W0) | Duplicate of registry Michie DF FD Case. Migrate/remove. |
| `unit/kinematics/test_om_anisotropy.py::test_plummer_grad_wrt_r_a_matches_fd` | AD == FD wrt r_a (OM f-table + stretch) | **migrate** | Already covered (PlummerVelocityDF+OM.sample_velocities, r_a) | Duplicate of registry OM FD Case. Migrate/remove. |
| `unit/profiles/test_limepy.py::test_differentiable_in_g_and_W`, `::test_differentiable_through_solve_in_W0_and_g`, `::test_differentiable_in_W_p_g`, `::test_differentiable_in_anisotropy_radius`, `::test_differentiable_construction_in_W0_and_g` | grad finite + non-zero (NO FD) on internal limepy density/solve | **keep (verify)** | n/a (internal limepy_density_hat / solve_limepy_profile) | Internal-machinery finite-only pins below public-entry granularity (underpin King/Michie/Engine-A Cases). Conservative keep; the public-path FD is the registry's job. |
| `unit/profiles/test_limepy_multimass.py::test_differentiable_in_W0_g_delta_alpha`, `::test_differentiable_in_targets_and_delta`, `::test_anisotropic_multimass_differentiable_in_eta_ra_delta`, `::test_differentiable_in_rescale` | grad finite + non-zero (NO FD) on internal multimass solve | **keep (verify)** | n/a (internal solve_multimass_limepy) | Internal multimass-solver finite-only pins; the public Engine-A path is registry-covered (W0/g/delta). Conservative keep; flag as the largest cluster of T6-style internal finite-only smoke (Anna may downgrade some to delete once she confirms the public Engine-A Cases subsume them). |
| `validation/test_eff_physics.py::test_eff_velocity_sampling_differentiable` | grad finite through EFF DF via pos-scale (NO FD) | **delete** | Covered (EFFVelocityDF.sample_velocities, gamma) | Finite-only smoke (T6); registry EFF DF FD Case is stronger. (Param is a pos-scale, but the channel is the EFF DF sampling the registry already audits.) |
| `validation/test_imf_physics.py::test_grad_through_ppf` | grad wrt **u** finite (Chabrier, NO FD) | **delete** | du-monotonicity | Finite-only smoke wrt the uniform draw (T6). |
| `validation/test_imf_physics.py::test_grad_through_sample` | grad wrt m_min finite (NO FD) | **keep (verify)** | Registry `PowerLawIMF.cdf[H4](m_min)` + the ppf_mmin migrate-row cover m_min | Finite-only on the m_min channel the registry/imf_gradients FD-cover. Borderline T6; conservative keep pending the ppf(m_min) NEW Case decision. |
| `validation/test_rotation_anisotropy_physics.py::test_grad_wrt_omega`, `::test_grad_wrt_v_peak` | AD == FD (rotation overlays) | **migrate** | Already covered (apply_solid_body_rotation omega; apply_differential_rotation v_peak) | Duplicates of registry rotation FD Cases. Migrate/remove. |
| `validation/test_segregation_approx_physics.py::test_grad_m_cut_matches_fd` (radial/lambda/sigma) | AD == FD wrt m_cut | **keep (verify)** | Registry `lambda_msr_approx(core_scale)` is a different param | Same as the unit segregation FD tests (m_cut channel). Keep (radial/sigma unique); flag possible NEW m_cut Case. Likely a near-duplicate of the unit-test trio — Anna may consolidate. |
| `validation/test_segregation_approx_physics.py::test_grad_positions_finite` | grad wrt positions finite (NO FD) | **keep** | n/a (positions, not a param) | grad wrt the position array (degenerate-input no-NaN). Complementary. Keep. |

---

## Per-file summary (scan by file)

- **`unit/imf/test_imf_gradients.py`** — migrate: 8 FD ppf tests + ppf_mmin (9); keep: boundary-NaN-grad, **α=1 kink class (4)**; delete: 3 finite-only ParameterGradients.
- **`unit/kinematics/test_df_gradients.py`** — migrate: Plummer r_h, King W0+r_c, EFF gamma+a (5 FD asserts; r_c-DF & a-DF need NEW Cases); keep: 3 JIT (out of scope).
- **`unit/profiles/test_king_grad.py`** — keep: 2 JIT, 1 unique ODE-W=0-grad guard.
- **`unit/profiles/test_profile_gradients.py`** — migrate: all 5 FD sampler tests (duplicates).
- **`integration/test_jax_compatibility.py`** — **delete: 3 finite-only smoke (plummer_grad, df_grad, ic_grad)**; migrate: build_spatial_ic FD; keep: compute_potential_energy soft=0 FD (unique), JITs, du-grad (verify).
- **`unit/profiles/test_find_alpha_ift.py`** — **KEEP ALL (do-not-touch): forward regression, IFT custom_vjp FD, jit-grad, aniso quadrature.**
- **`validation/test_engine_b_physics.py`** — **KEEP: β(r) anchor + all physics**; test_gradients_ad_vs_fd is keep(verify)/partial-migrate (King-branch + mass-fraction t are unique build-graph paths).
- **`unit/profiles/test_limepy_tables.py`** — keep: W≤0 NaN guard, table-build FD; keep(verify): 3 internal finite-only table-draw/solve smokes.
- **`validation/test_michie_physics.py`** — migrate: density-observable FD (W0); keep: σ(M) scaling + physics.
- **`validation/test_king_physics.py`** — delete: velocity_sampling finite-only; keep(verify): auto-domain-high-W0 FD (distinct branch); keep: physics + Table-II.
- **`unit/profiles/test_king.py`** — migrate: r_t FD (duplicate).
- **`unit/profiles/test_king_density.py`**, **`test_density_poisson.py`**, **`test_profile_potential.py`** — keep: unique internal/closed-form/Tracer grad guards.
- **`unit/profiles/test_plummer.py`** — delete: 1 finite-only sampler smoke; keep: d/dr grad; keep(verify): d/dr_h enclosed-mass.
- **`unit/substructure/test_q_approx.py`** + **`validation/test_substructure_q_physics.py`** — keep: positions-grad; migrate: concentration FD (duplicate of q_approx[EFF]).
- **`unit/test_numerics.py`**, **`unit/test_tidal.py`**, **`validation/test_tidal_physics.py`** — keep: unique primitive/tidal grad pins.
- **`integration/test_binary_cluster.py`** — keep(verify): build_binary_cluster grad (uncovered path / GAP).
- **`integration/test_end_to_end.py`** — **delete: 2 (du-grad + r_h finite-only).**
- **`unit/binaries/*`** — keep (mostly UNIQUE uncovered channels): period/ecc distributions (FD+analytic), energy-budget analytic grad, fraction models, diagnostics, divide-safe boundary grads. assembly test_jit_and_grad_safe: split (JIT keep, grad migrate).
- **`unit/imf/test_binary.py`** — keep: mass-ratio/differentiable-binary FD (uncovered).
- **`unit/imf/test_differentiable_binary.py`** — keep(verify): finite-only on uncovered DifferentiableBinary channel.
- **`unit/imf/test_differentiable.py`** — keep(verify): 3 finite-only log_prob (registry has the FD); keep: α1=1 kink.
- **`unit/imf/test_params.py`** — delete: trivial x² leaf-grad.
- **`unit/imf/test_smooth.py`**, **`test_moe_full.py`** — keep: TaperedPowerLaw + MoeFull grid-CDF FD (unique); keep(verify): 2 finite-only.
- **`unit/imf/test_environment.py`** — **KEEP ALL: environment-IMF channel (uncovered) — FD + smooth-tanh guards.**
- **`unit/cluster/test_multicomponent.py`** — delete: sample_differentiable_in_delta (verify); keep: pytree-grad regression; keep(verify): w_j / r_a Engine-A channels (GAP).
- **`unit/diagnostics/test_segregation_approx.py`** + **`validation/test_segregation_approx_physics.py`** — keep / keep(verify): m_cut FD surrogates (radial/sigma unique; lambda_msr different param than registry).
- **`unit/dynamics/*`** — keep: energy/accel-kernel double-where soft=0 guards + group-virial (all unique, none in registry).
- **`unit/kinematics/test_api.py`**, **`test_limepy_df.py`** — keep(verify): target_Q / LIMEPY-g channels (GAP, finite-only).
- **`unit/kinematics/test_michie_df.py`**, **`test_om_anisotropy.py`** — migrate: 2 FD duplicates.
- **`unit/profiles/test_limepy.py`**, **`test_limepy_multimass.py`** — keep(verify): 9 internal-machinery finite-only pins (largest T6-style cluster; public Engine-A path is registry-covered).
- **`validation/test_eff_physics.py`**, **`test_imf_physics.py`** — delete: 2 finite-only (EFF DF, Chabrier du); keep(verify): m_min finite-only.
- **`validation/test_rotation_anisotropy_physics.py`** — migrate: 2 FD duplicates.
- **`validation/test_binary_physics.py`** — see dedicated rows: keep (Kepler e→1 boundary + small-a physics, analytic), migrate-equivalent (to_state FD-grads overlap KeplerElements.to_state Case).

> **`validation/test_binary_physics.py` detail** (not all rows above): `test_grad_through_kepler_solve`
> (finite+sign, NO FD) → **delete** (covered by KeplerElements.to_state). `test_grad_finite_through_e_to_one`
> (finite at e=1.0, the B4-3 NaN regression) → **keep** (unique e→1 boundary, complements the registry's
> e=0.999 edge). `TestSmallSemiMajorAxis::test_circular_velocity_exact_across_scales_stellar` → **keep**
> (physics, not grad). `TestKeplerTransformGradients` (to_state a/e/M0, from_state v-scale — 4 FD tests) →
> **migrate**: `to_state(e)` duplicates the registry Case; `to_state(a)`, `to_state(M0)`, `from_state(v-scale)`
> are **NEW Cases needed** (the registry has only KeplerElements.to_state in `e`).

---

## Headline counts

- **migrate: ~24 assertions** (the clear AD-vs-FD duplicates) across imf_gradients (9), df_gradients (5),
  profile_gradients (5), test_king.py r_t (1), michie_df (1), om_anisotropy (1), rotation_physics (2),
  build_spatial_ic (1), substructure_q concentration (1), michie_physics density (1), and the
  test_binary_physics Kepler-transform FD trio. (Plus assembly's grad-half.)
- **keep: ~62 tests/groups** (unique gradient properties + physics + uncovered channels).
- **delete: ~12 clear finite-only smoke tests** (T6): test_jax_compatibility plummer_grad / df_grad /
  ic_grad; test_end_to_end imf_ppf (du) + spatial_profile; imf_gradients TestParameterGradients (3);
  test_imf_physics grad_through_ppf (du); test_eff_physics velocity-sampling; test_king_physics
  velocity_sampling; test_plummer gradient_through_r_h; test_binary_physics grad_through_kepler_solve;
  test_params x² (trivial). Plus ~6 more flagged **keep(verify)** that lean delete pending Anna
  (cluster sample_differentiable_in_delta; the 9 internal limepy/limepy_multimass finite-only pins if
  the public Engine-A Cases are deemed to subsume them).

- **NEW registry Cases the migrations would require: ~7**
  1. `KingVelocityDF.sample_velocities(r_c)`
  2. `EFFVelocityDF.sample_velocities(a)`
  3. `PowerLawIMF.ppf(m_min)`
  4. `KeplerElements.to_state(a)`
  5. `KeplerElements.to_state(M0)`
  6. `KeplerElements.from_state(velocity-scale)`
  7. (optional) `MichieProfile` density-observable or `r_c` channel.

---

## KEEP — do not touch (irreplaceable)

1. **`unit/profiles/test_find_alpha_ift.py`** — the find_alpha **IFT / custom_vjp** forward regression
   (REF_ALPHA), the IFT-grad-vs-FD (rtol 1e-6), jit(grad)/jit(value_and_grad) C1 regression, and the
   anisotropic-quadrature grad. The registry must NOT own the custom_vjp internal.
2. **`unit/imf/test_imf_gradients.py::TestAlphaOneGradients`** + **`test_differentiable.py::test_grad_finite_sweeping_alpha1_through_one`** — the **α=1 removable-singularity kink pins** (value smoothness + multi-segment finiteness). Registry only marks α=1 `known_blocked`; it does not assert the forward value is kink-free.
3. **`validation/test_engine_b_physics.py`** — the **Engine-B β(r) OM anisotropy anchor** and all the
   cross-engine/cross-family **physics** (King A-vs-B, EFF(γ=5)≡Plummer, halo+core Q_j, DF-density fidelity).
4. **Energy/accel-kernel double-where soft=0 guards** — `test_virial.py` (PE soft/padded), `test_group_virial.py` (PE/accel soft=0), `test_energy_consolidation.py`, and `test_jax_compatibility::test_compute_potential_energy_grad_finite_at_default_softening`. The registry audits IC/summary, not the gravitational kernels.
5. **Uncovered-channel grad-correctness pins** (no registry Case, FD- or analytic-grade): the binary
   period/eccentricity distributions (`test_population.py`), mass-ratio samplers (`test_binary.py`),
   the **environment-IMF** α₃ relations (`test_environment.py`), MoeFull grid-CDF reparam
   (`test_moe_full.py`), TaperedPowerLaw (`test_smooth.py`), the tidal straight-through surrogate
   (`test_tidal.py`), and the numeric primitives (`test_numerics.py`).
6. Internal W=0 / W≤0 NaN-grad guards: `test_king_density.py`, `test_limepy_tables.py::test_gradient_finite_at_W_nonpositive`, and the closed-form `_king_drho_dW` vs-autodiff pin (`test_density_poisson.py`).

---

## Clearest DELETE candidates (finite-only smoke — audit T6)

These assert ONLY `jnp.isfinite(jax.grad(f)(x))` (some + a sign) with NO FD comparison, on a param
channel the registry now FD-covers — a silently-zero gradient would PASS them:

1. `integration/test_jax_compatibility.py::TestProfileJAXCompatibility::test_plummer_grad`
2. `integration/test_jax_compatibility.py::TestVelocityDFJAXCompatibility::test_plummer_df_grad`
3. `integration/test_jax_compatibility.py::TestPipelineDifferentiability::test_plummer_ic_grad_wrt_r_h`
4. `integration/test_end_to_end.py::test_spatial_profile_gradient`
5. `integration/test_end_to_end.py::test_imf_ppf_gradient` (grad wrt u)
6. `unit/imf/test_imf_gradients.py::TestParameterGradients` (3 tests)
7. `validation/test_imf_physics.py::test_grad_through_ppf` (grad wrt u)
8. `validation/test_eff_physics.py::test_eff_velocity_sampling_differentiable`
9. `validation/test_king_physics.py::test_velocity_sampling_is_differentiable`
10. `unit/profiles/test_plummer.py::TestPlummerDifferentiability::test_gradient_through_r_h`
11. `validation/test_binary_physics.py::test_grad_through_kepler_solve`
12. `unit/imf/test_params.py::test_grad_through_alpha3` (trivial x²)

---

## Coverage GAPS found (gradient asserted somewhere, registry SHOULD cover, currently doesn't)

1. **KeplerElements `to_state(a)`, `to_state(M0)`, `from_state(velocity-scale)`** — FD-grade tested in
   `test_binary_physics.py::TestKeplerTransformGradients`, but the registry has only `to_state(e)`. The
   differentiable-IC binary pipeline depends on all of these columns of the Jacobian.
2. **`KingVelocityDF.sample_velocities(r_c)` and `EFFVelocityDF.sample_velocities(a)`** — FD-tested in
   `test_df_gradients.py`, registry has only the W0/gamma channels of those DFs.
3. **`PowerLawIMF.ppf(m_min)`** — FD-tested in `test_imf_gradients.py`; registry has `cdf[H4](m_min)`
   and `ppf(alpha)` but not `ppf(m_min)`.
4. **Engine-A `w_j` and `r_a` (from_components / from_mass_segregation)** — finite-only tested in
   `test_multicomponent.py`; registry Engine-A Cases are from_imf W0/g/delta only. These are inference
   targets (velocity-scale ratios, anisotropy radius) with NO FD coverage anywhere — a true gap.
5. **`build_binary_cluster` end-to-end** — only a finite+sign smoke (`test_binary_cluster.py`); the full
   IMF→companion→spatial assembly grad is not in the registry (which stops at resolve_binary_components /
   MoeCompanions). Candidate Fisher-integrity Case.
6. **`VelocityModel.target_Q` through `sample_velocities_pipeline`** and **`LIMEPYVelocityDF(g)`** — only
   finite-only smoke; no FD coverage. Minor gaps.
7. **Binary orbital-distribution params** (Sana/LogNormal/LogUniform period; Thermal/Uniform/Moe
   eccentricity; IndependentCompanions e_max) — FD/analytic tested in `test_population.py` /
   `test_companions.py` but NONE are registry Cases. Largest cluster of uncovered-but-tested channels;
   strong candidates for future registry expansion (these are NOT duplicates — keep until added).

## Possible exact DUPLICATES (prime migrate/delete — registry has an equal/stronger Case)

- `test_king.py::test_r_t_grad...` ≡ registry `KingProfile.r_t(W0)` (same config).
- `test_om_anisotropy.py::...r_a_matches_fd` ≡ registry `PlummerVelocityDF+OM(r_a)`.
- `test_michie_df.py::test_grad_wrt_W0_matches_fd` ≡ registry `MichieVelocityDF.sample_velocities(W0)`.
- `test_rotation_anisotropy_physics.py::test_grad_wrt_omega / _v_peak` ≡ registry rotation Cases.
- `test_df_gradients.py::TestPlummerDFGradients` ≡ registry `PlummerVelocityDF.sample_velocities`.
- `test_profile_gradients.py` (5) ≡ registry profile `sample_positions` FD Cases.
- `test_imf_gradients.py::TestFDvsAutodiff` (8) ≡ registry IMF ppf Cases.
- `test_jax_compatibility.py::test_build_spatial_ic_differentiable_wrt_r_h` ≡ registry `build_spatial_ic[Plummer]`.
- `test_substructure_q_physics.py::test_differentiable_wrt_concentration` ≡ registry `q_approx[EFF](gamma)`.

---

*No files were modified. This inventory awaits Anna's migrate/keep/delete sign-off at CHECKPOINT 4a
before any Task 4.2 deletion or registry expansion.*
