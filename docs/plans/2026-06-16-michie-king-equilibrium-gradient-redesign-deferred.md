# Deferred sub-arc — Michie/King equilibrium-solve gradient redesign (FD-consistency)

**Date:** 2026-06-16
**Status:** DEFERRED to the **differentiability-gradient-audit arc** (it owns profile-gradient integrity
and the shared `_find_tidal_radius`). Surfaced by the OED Phase-0.5 dispersion-hardening brainstorm.
**Decision (Anna, HITL):** Option 1 — the dispersion arc documents + regression-gates this limitation
(it never blocks the OED, which differentiates `r_a`/`M` on Plummer, never `W0`); the *fix* is a
properly-scoped sub-arc here, not rushed into dispersion hardening.

## The defect

`jax.grad` of a Michie-equilibrium observable w.r.t. `W0` is **FD-inconsistent at ~5e-3 relative**
(above the 1e-3 grad gate). Reproduced via `jeans_dispersion(MichieProfile.from_W0_rc(W0,...))`:
`σ_r(r=1)` at `W0=6, r_c=1, r_a=5`, `AD=−5.1554778e-2` vs converged `FD≈−5.176e-2`.
EFF (prescribed `r_t`, no ODE) is **clean** (rel ~1e-8); the defect is specific to the **solved**
equilibrium (King shares the identical solver pattern → same latent issue w.r.t. `W0`).

## Root cause (systematic-debugging, 3 hypotheses tested 2026-06-16)

The Michie/King `σ(W0)` is differentiable only **"to grid accuracy"** (the code's own words,
`king.py::_find_tidal_radius` docstring) because it is built from three *entangled discrete*
operations whose W0-sensitivity AD cannot fully see:
1. **Adaptive step controller** (`PIDController`) — the step schedule depends on `W0` through
   non-differentiable accept/reject branches; AD freezes the schedule, FD re-adapts it.
2. **`argmax` + linear-interp tidal-radius crossing** (`_find_tidal_radius`) — frozen node selection;
   the crossing-slope `dξ_t/dW0` is a two-node linear model, exact only to grid spacing.
3. **Grid interpolation** of `ψ(ξ;W0)` onto the integration radii.

### Negative results (do NOT re-run these — they are refuted)
| Hypothesis | Test | Verdict |
|---|---|---|
| Grid resolution | `n_ode_points` 3000→24000 | **Refuted** — gap resolution-independent (~5e-3); fwd derivative itself drifts |
| Adjoint *method* | `RecursiveCheckpoint` / `DirectAdjoint` / `BacksolveAdjoint` | **Refuted** — all give *bit-identical* AD (−5.1554778e-2); the continuous adjoint does NOT match FD |
| Adaptive vs fixed step | `Tsit5 + ConstantStepSize(dt=0.1)` | **Partial** — 4.4e-3 → 1.08e-3, but **changes the forward solution** (−5.143 vs −5.155) |

**Conclusion:** not a gradient-only adjoint swap. The non-smoothness is in the *forward*
discretization, so a robust fix must change the forward solve.

## Fix direction (for the gradient-audit arc)

A robust <1e-3, FD-consistent gradient needs the differentiable equilibrium-solve path redesigned:
- **Fixed-step (or W0-independent-schedule) integration** so the step sequence carries a clean
  derivative (removes contributor #1); choose `dt`/solver-order for forward accuracy.
- **Differentiable tidal-radius crossing** — event-detected / smooth root (e.g. solve-to-event, or an
  IFT custom_jvp `dξ_t/dW0 = −(∂ψ/∂W0)/(∂ψ/∂ξ)|_{ξ_t}` using `dψ/dξ = y[1]` at the crossing) to
  remove the `argmax`/linear-interp grid limit (contributor #2).
- Possibly a finer / analytically-anchored ψ-interpolation (contributor #3).

### HARD regression constraints (shared infrastructure)
- `_find_tidal_radius`, `solve_king_profile`, `solve_michie_profile` are **shared with King**. Any
  forward-solve change **must** re-validate the full King/Michie suites, especially the sensitive
  **King c(W₀) vs King (1966) Table II anchor at max|Δlog₁₀c| = 0.002** — a fixed-step switch *will*
  perturb it. Re-tune resolution to hold that anchor, or the fix is not acceptable.
- Cross-check both King AND Michie `∂σ/∂W0` AD-vs-FD <1e-3 after the fix.

## What the dispersion arc does instead (Option A, now)

Phase 0.5 Task B regression-gates the clean paths (EFF `r_t`/`γ`; King `W0` and Michie `W0` *if* they
pass) and adds a **documented gate** recording this Michie-`W0` limitation with a cross-ref to this
note. No forward-physics or integrator change in the dispersion arc.
