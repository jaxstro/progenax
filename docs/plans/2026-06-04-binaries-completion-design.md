# Binaries completion & hardening — design (2026-06-04)

**Goal (Anna):** complete and harden the `progenax` binary machinery to a "finalized,
trustworthy" state before moving to Batch 5. Primary use = **both** N-body cluster ICs
*and* statistically faithful binary populations.

This design was brainstormed and validated decision-by-decision (see "Decisions" below).
It follows Batches 4a–4e (orbital primitives hardened: STELLAR mean-motion fix, faithful
Moe Eq-3 Roche cap, grad-safe period utils, `kepler.py` split).

## Landscape — what already exists (do not rebuild)

- **Mass domain — `imf/binary/` is essentially complete.** `BinaryIMF.sample_systems(key, n)
  → (m1, m2, is_binary)` (primaries from any IMF, binary decision via a fraction model, q →
  secondary). q-distributions: `FlatMassRatio`, `PowerLawMassRatio(γ)`, `TwinPeakedMassRatio`,
  `MoeDiStefano2017`. Fraction models: `ConstantBinaryFraction`, `MassDependentBinaryFraction`
  (Moe Table 13), `DifferentiableBinaryFraction`. **Mass bookkeeping = primary-preserving
  (Kroupa):** `m2 = m1·q` is *added*, so binaries increase cluster mass; the IMF samples primaries.
- **Orbit domain — `binaries/` is complete** (post 4a–4e): period/eccentricity/orientation
  distributions + `KeplerElements.to_binary_state` → resolved (r₁,v₁,r₂,v₂) in the COM frame.
- **Spatial — `builders.build_spatial_ic(profile, masses, df, …) → ICResult`** places N system COMs.

So the "completeness gaps" collapse to: a missing **connector** that joins mass+orbit+space, a
**dynamic binary/multiple finder** (measurement side), **fraction-model unification**, and the
faithful **Moe P–q–e interrelation**.

## gravax handoff contract (verified 2026-06-04)

**Design for the best contract; fix gravax to match later (Anna, 2026-06-04).** We do NOT contort
progenax's `ICResult` to fit gravax's current loose constructor. `ICResult` is the **rich,
authoritative handoff object** (positions/velocities/masses/radii/softening + **primordial
provenance** + units). The ideal consumer is **`gravax.ParticleSystem.from_ic(ic: ICResult)`** —
which gravax should grow (deferred gravax work).

Current gravax reality (verified `core/state.py:243`): the only array constructor is
`ParticleSystem.from_velocities(positions, velocities, masses, units, radii=None, softening=0.01,
time=0.0, ids=None)`.

- ⚠️ **`ParticleSystem.from_ic(ic, units)` advertised in progenax docs/CLAUDE.md does NOT exist yet** —
  the best fix is to ADD `from_ic(ICResult)` to gravax (not to downgrade progenax docs to
  `from_velocities`). Deferred gravax work.
- **Binaries require `softening=0.0` + a collisional integrator** (Hermite-direct or IAS15; close
  encounters resolved by adaptive timestep). `gravax.core.softening.ZeroSoftening` exists "for
  regularized subsystems." Fixed-ε collisionless schemes (PEFRL/Yoshida/Leapfrog) are WRONG for
  resolved binaries. The orchestrator documents this and emits `softening=0.0` guidance; ideally
  `from_ic` defaults softening from the `ICResult`.
- `ParticleSystem` carries `radii` (collision detection) and `ids` (tuple). Primordial bookkeeping
  lives on `ICResult` as arrays (not crammed into the `ids` tuple); `from_ic` carries it through.

## Decisions (validated with Anna, 2026-06-04)

1. **Primary use:** both (cluster ICs + population statistics). Sequence A+C+D, then B; defer E/F/G.
2. **Connector shape:** layered — a pure `resolve_binary_components` primitive in `binaries/` +
   a thin `build_binary_cluster` orchestrator in `builders`. Composable AND single entry point.
3. **Binaries evolve → pairings change.** IC-time labeling is **primordial provenance, not dynamic
   state** (ionization/formation/exchange make it stale). Named `primordial_system_id` /
   `is_primordial_secondary`; never mutated by the integrator.
4. **No per-particle softening flags.** Binaries ⇒ collisional integrator (global ε=0); the
   orchestrator documents this rather than tagging particles.
5. **Dynamic finder included now.** Energy-based bound-system detection makes binaries trustworthy
   end-to-end (set up → evolve → measure).
6. **`find_bound_pairs` AND `find_bound_multiples`** (Anna). Pairs = 2-body primitive; multiples =
   hierarchical (collapse bound pairs → COM pseudo-particles → repeat to fixed max depth), giving
   triples/quadruples on the *measurement* side (gap E measured even though setup is deferred).
7. **Moe P–q–e (B) is forward-compatible now:** the connector primitive takes *final* arrays
   (m1,m2,a,e,orientation), agnostic to how they were sampled — so the joint sampler is later an
   orchestrator-level change with zero primitive rework.

## §1 — Connector (Batch 4f)

**Primitive — `binaries/assembly.py`:**
```
resolve_binary_components(
    com_pos[N,3], com_vel[N,3], m1[N], m2[N], is_binary[N],
    a[N], e[N], inc[N], Omega[N], omega[N], M_anom[N], *, G,
) -> ResolvedBinaries(positions, velocities, masses,
                      primordial_system_id, is_primordial_secondary)
```
Per binary i: `(δr₁,δv₁,δr₂,δv₂) = KeplerElements(a,e,…).to_binary_state(m1,m2,G)`; place
`r₁=X_i+δr₁, v₁=V_i+δv₁, r₂=X_i+δr₂, v₂=V_i+δv₂`. Since `m1·δr₁+m2·δr₂=0`, the binary COM stays at
`(X_i,V_i)` exactly — cluster phase space preserved, internal structure resolved. Singles pass
through. Output length = N_single + 2·N_binary. vmapped, differentiable.

**Orchestrator — `builders.build_binary_cluster(profile, velocity_df, binary_imf, period_dist,
ecc_dist, key, *, G, units) -> ICResult`:** (1) `binary_imf.sample_systems` → m1,m2,is_binary;
(2) `build_spatial_ic` on **system masses** (m1+m2) → COMs; (3) sample P→a (`period_to_semimajor_axis`
with `day_in_time_units` from `units`), e, isotropic orientation; (4) `resolve_binary_components`;
(5) expanded `ICResult` (+ primordial fields).

**Units gotcha (tested):** periods sampled in **days**, `a` in cluster length (pc), G in pc³/M⊙/Myr².
Convert via `day_in_time_units = 1 day in time-units` derived from `units`. End-to-end Kepler-III
test asserts consistency.

**ICResult additions:** `primordial_system_id[N]:int`, `is_primordial_secondary[N]:bool` (watch the
15-field limit). Pairs derivable by grouping on the id; generalizes to triples later.

## §2 — Dynamic finder (Batch 4g)

`binaries/diagnostics.py`:
- `relative_energy(r_i,r_j,v_i,v_j,m_i,m_j,*,G)` — ½μ|Δv|² − G m_i m_j/|Δr| (differentiable).
- `find_bound_pairs(positions, velocities, masses, *, G) -> (pair_idx[K,2], E_rel[K])` —
  mutual-nearest-neighbour AND E_rel<0 (standard NBODY/kira criterion). O(N²); jittable but **not
  differentiable** (uses `argmin`) — correct, it's a measurement.
- `find_bound_multiples(positions, velocities, masses, *, G, max_levels=3) -> (system_id[N],
  multiplicity[N])` — iterate: find pairs → collapse to COM pseudo-particles → repeat to fixed depth
  (bounded `lax.scan`, no `while_loop`). Detects triples/quadruples.
- `primordial_survival(system_id, primordial_system_id) -> {survived, disrupted, newly_formed}` —
  multiplicity-aware.

Round-trip test: `resolve_binary_components` → `find_bound_pairs` recovers exactly the primordial
pairs (hard binaries) at t=0.

## §3 — Fraction unification (Batch 4h)

`BinaryFractionModel` protocol in `protocols.py`: `probability(masses, radii=None) -> f_bin[N]`.
Mass-based models ignore `radii`; `RadialBinaryFraction` ignores `masses`; combined uses both.
**Ordering:** radius-dependent fraction needs positions *before* the binary decision → orchestrator
samples COMs first, then decides membership (vs the mass-only fast path that decides in `BinaryIMF`).
Both paths tested. Re-export the scattered fraction models for discoverability.

## §4 — Moe P–q–e interrelation (Batch 4i)

Faithful `MoeJointOrbit(M1, key) -> (P, q, e)`: sample logP, then q|(M1,logP) (Moe Eqs 2/13–15
two-slope + twin), then e|(M1,logP) (Eqs 17–18, the Roche-capped `MoeEccentricity` already built).
Couples `imf` q + `binaries` P/e. The connector consumes its outputs unchanged (decision 7).

## Sequence

| Sub-batch | Deliverable | Closes |
|---|---|---|
| 4f | connector primitive + orchestrator + primordial provenance | A, C |
| 4g | `find_bound_pairs` + `find_bound_multiples` + survival | dynamic finder |
| 4h | `BinaryFractionModel` protocol + wiring (both orderings) | D |
| 4i | faithful `MoeJointOrbit` P–q–e joint sampler | B |
| → | Batch 5 `analytical/` | — |

Each sub-batch: TDD RED→GREEN, FD-grad-checks on differentiable entry points, full suite green under
both jax envs, 3-gate HITL commit + push.

## Deferred (ticketed)

- **E — triples/quadruples SETUP** (we *measure* them in 4g; generating hierarchical primordial
  systems is a later feature).
- **F — Kroupa-1995 birth-period distribution** (pre-dynamical-processing periods; PDF held).
- **G — observational selection operators** (for likelihood/inference).
- **gravax doc fix** — `ParticleSystem.from_ic` → `from_velocities` in progenax docs/CLAUDE.md +
  gravax README; surfaced by the gravax maturity survey.
