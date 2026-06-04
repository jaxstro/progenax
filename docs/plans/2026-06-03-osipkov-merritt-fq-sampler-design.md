# Design: Osipkov–Merritt f(Q) anisotropic velocity sampler (Batch 2b)

**Date:** 2026-06-03 · **Status:** design (gate before TDD) · **Paper:** Merritt (1985), AJ 90, 1027
(grounded in `docs/website/99-bibliography/per-paper/merritt-1985.md`).

## Decisions locked (Gate 1 + brainstorm)

- **Architecture C** — anisotropy via a shared **augmented-density Eddington core**; isotropic is
  the r_a→∞ limit.
- **API surface** — an optional `anisotropy_radius` (r_a) on each DF: `PlummerVelocityDF`,
  `KingVelocityDF`, `EFFVelocityDF`. `r_a=None` ⇒ isotropic (current behavior, untouched);
  a float ⇒ true Osipkov–Merritt.
- **Remove** the heuristic `apply_osipkov_merritt` transform, `AnisotropyParams`, and the pipeline
  "anisotropy stage" (no backwards-compat shim; update callsites directly).
- **All three** DFs get OM in 2b.

## The math (verified)

OM Type I: f = f(𝒬), 𝒬 = Ψ(r) − v_r²/2 − v_t²(1+r²/r_a²)/2 (relative-potential frame, Ψ(r_t)=0).
The DF reproducing a fixed density ρ(r) is the Eddington inversion of the **augmented density**

    ρ_Q(r) = (1 + r²/r_a²) ρ(r)             (Merritt Eq. 9)

and the anisotropy is **profile-independent**: β(r) = r²/(r²+r_a²), σ_r²/σ_t² = 1+r²/r_a² (Eq. 15/17).

**Sampling = isotropic in a stretched velocity space** (derived, verified to give Eq. 15):
substitute w_t = v_t·√(1+r²/r_a²) ⇒ 𝒬 = Ψ − (v_r²+w_t²)/2; in polar (v_r,w_t)=s(cosθ,sinθ) the
joint density factorizes:

1. s ~ g(s) ∝ s²·f(Ψ(r) − s²/2) on [0,√(2Ψ)]   — the **same 1-D inverse-CDF** as isotropic;
2. cosθ ~ U[−1,1];
3. v_r = s·cosθ along r̂;  v_t = s·sinθ / √(1+r²/r_a²) in a random azimuthal direction ⊥ r̂.

No rejection, no while-loop, differentiable in r_a.

## Module design

New `src/progenax/kinematics/eddington.py` (shared, ≤~200 LOC):

- `augmented_eddington_table(r_grid, rho_grid, r_a) -> (Psi_grid, E_grid, f_grid)`
  Eddington inversion of ρ_Q = (1+r²/r_a²)·ρ. `r_a=None`/∞ ⇒ plain ρ (isotropic). Carries the
  Batch-2a double-`where` guard so it is grad-safe at the center.
- `om_sample_velocities(key, positions, r_a, Psi_of_r, E_grid, f_grid, velocity_scale) -> (N,3)`
  the stretched-isotropic sampler above. `r_a=None` ⇒ stretch factor 1 (pure isotropic).

`EFFVelocityDF._eff_eddington_table` is **refactored to call** `augmented_eddington_table` (DRY;
isotropic EFF becomes the r_a=∞ case), so the existing isotropic EFF path is provably unchanged.

Each DF gains `anisotropy_radius: Optional[float] = None`:

| DF | r_a=None (unchanged) | r_a=float (new OM) |
|----|----------------------|--------------------|
| **Plummer** | exact Beta(3/2,9/2) sampler | speed inverse-CDF from the **analytic** Merritt Eq. 45 f(Q); stretch |
| **EFF** | current Eddington table | `augmented_eddington_table` with (1+r²/r_a²); stretch |
| **King** | current lowered-Maxwellian sampler | augmented-density Eddington of the King density; stretch |

Plummer uses the analytic Eq. 45 DF (exact); the numeric core is cross-checked against it.

## ⚠️ King interpretation — needs your confirmation

Two distinct "anisotropic King" models exist:

- **(A) Merritt's method on the fixed King density** — hold the King ρ(r) fixed, find the OM f(𝒬)
  that reproduces it (augmented-density Eddington, exactly like Plummer/EFF). Density stays King;
  β(r)=r²/(r²+r_a²). **Unified with the other two, fully validatable, tractable.**
- **(B) Self-consistent Michie (1963) model** — specify f(E,J) ∝ exp(−J²/2r_a²σ²)[exp(−E/σ²)−1]
  and re-solve Poisson for a *new* (anisotropic) density/potential — a modified King ODE. Different
  density profile; its own validation; substantially more work.

**Recommendation: (A) for 2b** (consistent core, validatable, matches "augmented-density route");
note (B) self-consistent Michie as a future extension. (Your Gate-1 note said the augmented-density
route "also works for King," which is interpretation A.)

## Non-negativity (real constraint, not a tolerance)

OM DFs go negative if r_a is too small. Merritt Eq. 46: Plummer requires **r_a ≥ 0.75 r₀**. Plan:
build f on the grid, and if `jnp.min(f) < 0` flag it. Since this must stay JIT/grad-safe (no Python
raise on traced values), enforce by (i) a **constructor-time eager check** when r_a is a concrete
float (raise `ValueError` with the bound), and (ii) document the per-profile bound. We do **not**
silently clamp f<0 to 0 (that would fake an unphysical model); we refuse it.

## Differentiability

`r_a=None` vs float is a static Python branch (isotropic vs OM is a modeling choice, not a traced
value) — fine. *Within* the OM path, grad w.r.t. r_a flows (augmented density, table, stretch are
all smooth). FD-vs-autodiff grad-check on r_a is part of the gate.

## Removal + callsite updates

- Delete `kinematics/anisotropy.py` (`apply_osipkov_merritt`), `AnisotropyParams`, and the pipeline
  Stage-2 anisotropy block in `api.py`.
- Update: `kinematics/__init__.py` exports, `api.py` (`VelocityModel.anisotropy`, imports),
  `tests/unit/kinematics/test_anisotropy.py` (delete), `test_api.py` (drop anisotropy-stage tests),
  and any `cluster/` callsite constructing `AnisotropyParams` (verify; cluster is deferred but must
  still import-clean). Grep-driven, exhaustive.

## Validation plan (TDD, RED→GREEN; never weaken)

1. **Plummer numeric == analytic** — `augmented_eddington_table` on the Plummer density matches
   Merritt Eq. 45 f(Q) within table tolerance.
2. **Realized β(r) == r²/(r²+r_a²)** for all three DFs (the test the old heuristic FAILED): sample a
   shell at several r, measure β = 1 − σ_t²/(2σ_r²), compare to target. This is the headline RED.
3. **Isotropic limit** — large r_a (e.g. 50·scale) reproduces the existing isotropic DF's σ(r) and Q.
4. **Virial Q ≈ 0.5** — OM ICs of the fixed density remain in equilibrium (no rescale).
5. **Non-negativity** — r_a below the bound raises; r_a ≥ bound gives f ≥ 0.
6. **FD grad-check w.r.t. r_a**, JIT-compat, byte-stable RNG.

## Out of scope / future

Self-consistent Michie King (B); tangentially anisotropic (Type II) models; linear superpositions
of r_a (Merritt §IV).
