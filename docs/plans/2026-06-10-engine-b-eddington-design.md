# Engine B design — density-defined multi-component equilibria (Eddington in a shared potential)

**Status:** design approved by Anna 2026-06-10 (brainstormed section-by-section).
**Master plan:** `~/.claude/plans/we-are-continuing-progenax-s-crispy-rain.md` (Phase 2).
**Predecessor state:** Phase 1 (Engine A) + 1.5 (DF tables) + 1c (legacy retirement)
complete on `feat/multimass-limepy-equilibrium`; released-core 1064 green.

## Goal

`MultiComponentCluster` gains a second equilibrium engine: components defined by
**prescribed densities** (Plummer, EFF, King-density) sharing ONE self-consistent
potential, each component's DF obtained by **Eddington inversion in the shared
relative potential Ψ** (+ optional per-component Osipkov-Merritt anisotropy).
Answers the "Plummer halo + EFF core as a true joint equilibrium" science ask.

## Key structural insight

Engine A needs an ODE because its density is a functional of ψ (self-consistency
loop). Engine B's total density is prescribed: ρ_tot(r) = Σ_j (M_j/M_tot)·ρ̂_j(r),
so the shared potential is ONE direct quadrature pass (cumulative trapezoid:
M(<r), Φ = −G[M(<r)/r + 4π∫_r^{rt} ρ s ds], Ψ = Φ(r_t) − Φ). The genuinely new
work is per-component Eddington inversion in the shared Ψ plus the f_j ≥ 0
realizability gate.

## Decisions (locked with Anna)

1. **API shape (Q1 = A):** same class, new constructor —
   `MultiComponentCluster.from_density_profiles(profiles, mass_fractions, m_j,
   r_a_j=None, r_t=None, f_enc=0.995)`. One user-facing class (master-plan
   decision); engine is a constructor detail. Internals: static `engine` field
   ("A"/"B") + optional Engine-B field group; methods dispatch on engine.
2. **Domain/truncation (Q2 = C, derived):** a component's extent is part of the
   prescribed model. King ends at its own r_t (smoothly — no Eddington kink);
   EFF carries an explicit r_t parameter; Plummer is infinite (extent None).
   Cluster edge r_t = max over finite component extents; if ALL components are
   infinite, the radius enclosing `f_enc` (default 0.995) of the summed mass.
   Explicit `r_t=` override available, never required for a well-posed mix.
   "Too small r_t" cannot hide: per-component truncated-mass fractions
   M_j(r_t)/M_j(∞) stored, f_j ≥ 0 gate, exact-quadrature Q_j oracle. A
   divergent-mass profile (EFF γ ≤ 3) with no finite extent anywhere raises
   (physics-explaining error), the code never invents a radius.
3. **f_j < 0 policy (Q3 = A, refuse loudly):** genuine negativity
   (min f_j < −1e-3·max|f_j|, separating physics from grid ringing) raises
   ValueError on concrete inputs, naming the component and the remedy
   ("density too shallow to be supported in this shared potential — steepen
   it, raise its mass fraction, or raise r_a_j"). Traced builds (grad/jit)
   necessarily skip the raise but ALWAYS store `f_min_j` diagnostics — the
   two-tier pattern EFFVelocityDF already uses. Never clamp silently:
   a clamped DF integrates back to a *different* density than prescribed.
4. **v1 scope (Q4):** Plummer + EFF + King-density components; per-component
   OM r_a_j (augmented-density inversion, machinery already generic). King-
   density is the TRUST ANCHOR: it is the only configuration where Engine A
   (g=1) and Engine B describe the identical cluster → the direct A-vs-B
   cross-validation. Excluded from v1: rotation inside the constructor (stays
   a post-hoc `apply_*_rotation` transform) and mixed A+B engines in one
   potential (master-plan deferral).
5. **Mass semantics:** `mass_fractions` = M_j/M_total (the one free amplitude
   per prescribed shape; what observers think in), NOT Engine A's central-
   density α_j. Number fractions for the categorical draw: N_j ∝ M_j/m_j
   (m_j decoupled labels, as in Engine A). Strict check: raise if
   |Σ mass_fractions − 1| > 1e-8 (a wrong sum is a user bug, not a convention).

## Numerical core

1. **`eddington_invert(r_grid, rho_grid, drho_dr_grid, Psi_grid, dPsi_dr_grid,
   r_a=None, n_e=1000) → (E_grid, f_grid)`** in `kinematics/eddington.py`:
   `_eff_eddington_table` lines 52–107 with the EFF-specific density build
   stripped. Carries over verbatim: OM augmentation ρ_Q = (1+r²/r_a²)ρ; the
   double-where dρ/dΨ guard at r→0 (NaN-cotangent protection — hard-won);
   d²ρ/dΨ² via jnp.gradient; the u = √(E−Ψ) substitution (kills the integrable
   singularity); the truncation boundary term (dρ/dΨ)|_{Ψ=0}/√E. EFFVelocityDF
   becomes a thin caller — bit-identical output is the extraction gate.
2. **`profiles/density_poisson.py`:** evaluate ρ_j, dρ_j/dr per profile on a
   common fixed-size r-grid (n_r = 6000, EFF precedent; jit/grad-safe), scale
   by mass fractions, sum, one cumulative-trapezoid pass → M(<r), Φ, Ψ, plus
   per-component enclosed-mass CDFs. Domain resolved BEFORE grid construction
   (decision 2).
3. **Per-component inversion:** Python loop over the small static component
   count (Engine A precedent): `eddington_invert(..., r_a_j)` → shared E_grid,
   stacked f_j table.

## Sampling (Engine B branch of the jitted core)

1. Component: categorical, N_frac_j ∝ mass_fractions_j/m_j.
2. Position: per-component inverse-CDF on M_j(<r)/M_j(r_t), isotropic angles.
3. Speed: `sample_speed_from_f_table(key, Ψ(r_i), E_grid, f_j_grid)` (existing
   differentiable per-star inverse-CDF); velocity scale √(G·Σm_i/(4π·μ)) from
   the ACTUAL sampled mass (the Engine A lesson — never an independent M_total).
4. Direction: `assign_om_directions` with the star's component r_a_j.
5. NO external virial rescale — equilibrium must emerge from the DF or the
   validation fails honestly.

Speed-table acceleration (Phase 1.5-style) deliberately deferred: correctness
first; the quadrature path becomes the oracle if profiling later justifies it.

## Diagnostics stored on the model

`f_min_j / max|f_j|` (realizability margin), truncated-mass fractions
M_j(r_t)/M_j(∞), derived r_t + its provenance (which component's extent, or
the f_enc criterion).

## Error handling (all physics-explaining, none silent)

- Genuine f_j < 0 → ValueError naming component + remedy (concrete inputs).
- All-infinite mix with divergent mass (EFF γ ≤ 3 unbounded) → ValueError
  explaining the divergence.
  [NOTE: structurally unreachable — EFFProfile always carries a finite r_t;
  downgraded to the documented trunc_frac_j = 0.0 diagnostic per the
  implementation plan.]
- King natural r_t conflicting with an explicit override → ValueError (no
  silent re-truncation).
- |Σ mass_fractions − 1| > 1e-8 → ValueError.
- Honest caveat carried over from EFFVelocityDF: a sharply truncated empirical
  profile is only approximately stationary at the edge; the diagnostics
  quantify it (mild truncation ≈ 1% virial; severe worse) rather than hide it.

## Validation gates (independent oracles; never weakened)

1. **Extraction regression:** post-refactor EFFVelocityDF tables bit-identical.
2. **Analytic DF oracle:** isotropic Plummer f(E) ∝ E^{7/2} closed form
   reproduced (rtol ~1e-3 away from boundary energies) — bypasses ALL our
   numerics.
3. **EFF γ=5 ≡ Plummer** end-to-end through the new path.
4. **A-vs-B anchor:** single King-density component (Engine B) ≡ Engine A
   (g=1, same W0): Ψ(r), σ(r), Q_j agree.
5. **Equilibrium headline:** Plummer halo + EFF core — theory Q_j = 0.5
   (moment quadrature), sampled global Q = 0.5 UNSCALED, per-component Q_j
   converges with N.
6. **OM realization:** sampled β_j(r) ≡ r²/(r²+r_a_j²) per component.
7. **Negative tests:** unrealizable mix raises naming the component;
   divergent γ=3 EFF with no extent raises.
8. **Gradients:** AD-vs-FD through profile params, mass_fractions, r_a_j.

## Phasing (TDD; full released-core gate green at each step)

- **2a** extract `eddington_invert` + EFF bit-identical regression.
- **2b** `density_poisson` (shared Ψ) + Plummer analytic-DF oracle + domain-
  derivation tests (max-extent, f_enc, divergence raise, override conflict).
- **2c** `from_density_profiles` + sampler branch + King A-vs-B anchor +
  equilibrium gates.
- **2d** OM anisotropy + f_j negative tests + gradient checks.
- **2e** `scripts/validate_multicomponent_eddington.py` (PASS table + figures)
  + close-out doc + STATUS/CLAUDE.md.

## Honest scope / risks

- Eddington realizability genuinely constrains decompositions: some physically
  plausible-looking mixes (shallow component in a concentrated companion's
  potential) DO NOT EXIST as equilibria — the engine reports this as physics,
  not failure.
- Truncated empirical profiles are approximately stationary at the edge
  (inherited, documented, quantified by Q_j).
- Mixed A+B engines in one φ: deferred (master plan); the constructor design
  leaves room.
- Phase 3 docs sweep (16 inventoried pages referencing the retired API)
  remains separate.
