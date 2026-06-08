# FDF spherical-cluster ICs — design (envelope + turbulent velocities + substructure diagnostic)

**Date:** 2026-06-07 · **Branch:** `gravoturb-fdf-sbc-validation` (experimental subsystem) · nothing pushed.
**Context:** strategic pivot (see brain inbox 2026-06-07 + STATUS.md) — the 2D-projected-β *inference*
is banked as a methods result; gravoturb_fdf is repositioned as a **forward generative tool**:
realistic, substructured, *spherical* young-cluster ICs for N-body / binary studies. This design adds
a spherical shape, coherent turbulent velocities, and a clean substructure parameterization to the
existing positions-only turbulent-cube pipeline.

## Current state

`build_fdf_field` → `mass_conserving_copula_field` → `cloud_to_stars`/`sample_positions` produces star
**positions** in a periodic cubic box following a BM19 turbulent density field (⟨e^s⟩=1). No spherical
envelope, **no velocities**. Validated by AC1–AC10 (BM19 marginal, mass conservation, f_dense_realized
cornerstone AC6, Q(f_sub), gradients). Core `progenax` provides `PlummerProfile`/`EFFProfile` (each with
`.density(r)`), `compute_q_parameter`/`q_approx`, and `virial_scale(pos, vel, m, Q_target, G)` (cites
Goodwin & Whitworth 2004).

## Decisions

1. **Spherical envelope — separable log-space.** `s_total(x) = s_turb(x) + ln ρ_env(r)`, where
   `s_turb` is the existing BM19 turbulent fluctuation (⟨e^s⟩=1, the **substructure**: β, ℳ, α) and
   `ρ_env(r)` is a radial profile (the **shape**: r_h, concentration). `ρ_total = ρ_env(r)·e^{s_turb}`
   — centrally concentrated **and** clumpy. The split keeps all existing BM19 validation (s_t,
   f_dense, AC6) on `s_turb` (dense clumps = *local* overdensities `s_turb > s_t`). Envelope = any
   progenax `SpatialProfile` via `.density(r)`; **default `PlummerProfile`** (EFF for young-cluster
   realism). Envelope centered in the box, taper well inside (r_h ≲ box/4) so periodic edges are empty.

2. **Velocities — turbulent + coherent, scaled to a chosen Q.** A 3-component Gaussian velocity field
   with a turbulent spectrum P_v(k) ∝ k^{−β_v} (reuse `gaussian_random_field` per component),
   trilinear-interpolated to star positions → spatially coherent stellar velocities (clumps move
   together; Goodwin & Whitworth 2004). Then `virial_scale(pos_total, v, m, Q_target, G)` sets the
   amplitude to a **chosen Q** (free IC parameter; **default 0.5**, sub-virial Q<0.5 supported — the
   dynamically interesting young-cluster regime). `virial_scale` computes |V| from the actual
   positions, so the **envelope is automatically accounted for** (deeper potential → larger v at fixed
   Q). **β_v default to be grounded against the Goodwin & Whitworth 2004 PDF before coding** (no
   assumption-from-memory).

3. **Substructure diagnostic — CW04 Q + (m̄, s̄) plane.** Reuse `compute_q_parameter`; report the
   **(m̄ clumpiness, s̄ concentration) components** separately so substructure is decoupled from the
   envelope. Calibrate m̄ ↔ β at fixed envelope (substructure tracks β) and confirm m̄ ≈ const as
   concentration varies at fixed β (decoupling). Q bridges to observed-cluster catalogs.

## Validation (AC-style, experimental)

(1) sampled ρ(r) matches the analytic envelope; (2) m̄(β) monotonic **and** concentration-decoupled;
(3) BM19/AC6 still hold on `s_turb`; (4) velocity field has the target spectrum, achieves target Q,
and shows near-neighbour velocity **coherence**; (5) `jax.grad` flows through the density construction.

## Plots (figure gallery)

3D + projected scatter (spherical & clumpy); ρ(r) sampled vs analytic; velocity-coherence map;
Q–(m̄,s̄) vs β; density PDF; power spectrum.

## Build order (TDD; released-core **814** invariant held; experimental-only)

1. `field/envelope.py`: `radius_grid`, `apply_spherical_envelope` (separable log-space) + tests.
2. `field/velocity.py`: `turbulent_velocity_field`, `sample_turbulent_velocities` (+ G&W β_v
   grounding) + tests; wire `virial_scale`.
3. diagnostic extension: (m̄, s̄) components, concentration-decoupled + tests.
4. validation script (AC-style) + figure gallery.

## Defaults

envelope `PlummerProfile`; Q_target 0.5 (sub-virial supported); β_v ← Goodwin & Whitworth 2004 (verify).
