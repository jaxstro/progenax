# Phase 0.5 — Dispersion engine SoTA hardening — ratified design

**Date:** 2026-06-15
**Status:** RATIFIED (Anna HITL, brainstorming complete).
**Branch:** `feat/dispersion-hardening` (to be created off `main`; local only).
**Scope guard:** packaged `src/progenax/kinematics/dispersion.py` only. No Phase-1 OED demo code, no
gravax/fluxax/startrax. Gated separately from Phase 1 (same logic that made Phase 0 its own gate).

This is a hardening mini-arc **between** the merged Phase 0 (differentiable dispersion capability,
`docs/plans/2026-06-15-oed-dispersion-phase0.md`) and Phase 1 (the OED demo). Its purpose: take the
dispersion forward model from "sound but with four documented caveats" to **SoTA at all radii for every
profile**, so the OED rides on an engine whose B14 caveat audit is ~empty. Anna's explicit goal is
*science ASAP with SoTA now* — a solid engine serves that (shorter caveat audit, faster Fisher build,
real-data-ready versatility), and doing it now is cheaper than re-loading the dispersion context later.

## The four caveats and their SoTA fixes

| # | Caveat (Phase-0 `(d)`) | Type | SoTA fix | Task |
|---|------------------------|------|----------|------|
| 1 | OM-only anisotropy; Michie-under-OM valid inner-region only | physics model | general-β Jeans (Tier A) + exact Michie **DF-moment** (Tier B) | **D** |
| 2 | ~8.6e-4 Plummer truncation-tail bias (∫ to 30a, not ∞) | numerical | algebraic **coordinate compactification** `s=a·t/(1−t)` | **C** |
| 3 | `project_dispersion` re-solves Jeans per-R inside the vmap | numerical (perf) | **tabulate-once-then-project** (1 master solve + interp) | **A** |
| 4 | `r_t` gradient (re-scoped 2026-06-16, see below) | numerical (grad) | **regression-gate** clean paths; defer the Michie/King solver-gradient redesign | **B** |

Three are *numerical-method* fixes that change **zero physics** (validated as regression against the
existing 31 anchors). Only #1 grows the physics surface and gets a new validation anchor + a HITL
checkpoint on its oracle design.

## Ratified decisions (brainstorm Q1–Q4)

**Q1 — API surface (Option 1).** Extend `jeans_dispersion` *in place* with an optional `beta_fn=None`
(default → OM from `r_a`); add **one** new public symbol `df_moment_dispersion(df, r, ...)` for the
exact anisotropic DF moment (Tier B). Registry cost: **1** new `__all__` symbol (all four registries +
dashboard re-stamp), `jeans_dispersion` row updated. The merged OM signature/behavior stays
backward-compatible.

**Q2 — Tier B quadrature + oracle.** The Michie DF at radius `r` is
`f(v_r,v_t) ∝ exp(−(r v_t)²/(2 r_a²σ²))·[exp((Ψ(r)−½(v_r²+v_t²))/σ²) − 1]` on `v_r²+v_t² ≤ 2Ψ`,
`d³v = 2π v_t dv_t dv_r`. Moments: `ρ=∫f`, `ρσ_r²=∫v_r²f`, `ρσ_t²=½∫v_t²f` (since `<v_t²>=2σ_t²`).
- **Quadrature — polar speed–angle on a FIXED domain.** `v_r=w cosα, v_t=w sinα`,
  `w∈[0,√(2Ψ)]`, `α∈[0,π/2]` (×2 for v_r-symmetry), Jacobian `dv_r dv_t = w dw dα`. The energy bound
  `E=Ψ−w²/2≥0` becomes the **w upper-limit** → fixed rectangle `[0,√(2Ψ)]×[0,π/2]`, **no boundary
  masking** (smooth/differentiable; mirrors the existing `ftable_sigma_r_isotropic` 1-D speed moment).
  `E` depends only on `w`; `α` enters only the `exp(−(r w sinα)²/…)` factor.
- **Oracle — 3-leg anchor (no fitted fudge):** (1) **DF sampler at ALL radii** (N≈2e5 from
  `MichieVelocityDF.sample_velocities`, binned σ_r/σ_t, 5% MC) — the leg that proves "inner-region-only"
  is killed, and the **independent physical truth**; (2) **Tier A consistency** — feed (ρ, M,
  β=1−σ_t²/σ_r² from Tier B) into the general-β Jeans factor → must reproduce Tier B's σ_r² (a true
  equilibrium satisfies Jeans); (3) **zeroth-moment self-consistency** — `∫f d³v` equals
  `profile.density(r)` (DF and profile share Ψ). No source-verified closed form for the Michie σ_r(r)
  exists; we will **not** fabricate one — the sampler is the truth-leg.

**Q3 — Tier A integrating factor.** (a) `beta_fn=None` keeps the **exact analytic OM factor**
`f(r)=r²+r_a²` — current path bit-for-bit preserved; the general **numerical** factor
`f(r)=exp(2∫β(s)/s ds)` (a `cumulative_trapezoid`; the lower-limit constant cancels in the `f(s)/f(r)`
ratio; `β/s` is integrable at `s→0` for OM and Michie) fires **only** on the explicit `beta_fn` path.
(b) The Michie cross-check (oracle leg 2) sources β from Tier B's moments; the sampler stays the
independent truth-leg.

**Q4 — numerical methods.**
- **A (tabulate-once):** one master Jeans solve on the fine s-grid → R-independent `σ_r²(s), β(s),
  ρ(s)`; per-R projection `jnp.interp`s those onto `r=√(R²+u²)` instead of re-calling
  `jeans_dispersion`. Regression: `project_dispersion` numerically identical (~1e-9) + structural
  speedup (m→1 solves).
- **B (RE-SCOPED 2026-06-16 — regression-gate, no reparam):** adversarial pre-verification (TDD)
  showed the planned normalized-`x=r/r_t` reparam is **empirically unnecessary** — EFF `∂σ/∂(r_t, γ)`
  and the dispersion integrator's own `r_t` handling are **already clean** (rel ~1e-8), because the
  existing `linspace(·, r_t, n_s)` already scales grid points with `r_t`. The one real defect found is
  **upstream**: Michie/King `∂σ/∂W0` is FD-inconsistent at ~5e-3, rooted in the *equilibrium ODE
  solver* (adaptive-controller schedule + `argmax`/linear-interp tidal-radius crossing), NOT in
  `dispersion.py`. Per Anna's HITL decision (Option 1), the solver-gradient redesign is **deferred to
  the differentiability-gradient-audit arc** (spec: `2026-06-16-michie-king-equilibrium-gradient-
  redesign-deferred.md`). Task B therefore: add AD-vs-FD **regression grad-gates** locking the clean
  paths (EFF `r_t`/`γ`; King `W0`, Michie `W0` *only if* they pass), and one **documented** gate
  recording the Michie-`W0` upstream limitation with a cross-ref. No integrator/physics change.
- **C (compactification):** Plummer's `[s_min, 30a]` → algebraic map `s = a·t/(1−t)`, `t∈[0,1)`,
  integrate in **uniform t** with `ds/dt = a/(1−t)²` folded into the integrand (preserves
  `cumulative_trapezoid`'s uniform-dx contract). Residual → toward machine precision; O(h²) preserved;
  King/EFF (finite r_t) untouched.

## Hard constraints (this arc)

- **Paper-grounding before encoding (D0).** Do NOT encode the Michie DF form from memory or the existing
  (unverified) docstring/note. Read `docs/core-papers/michie-1963.pdf` (and `king1966.pdf`) directly,
  confirm the DF / β(r) / cutoff, then verify/correct the per-paper note
  `docs/website/99-bibliography/per-paper/michie-1963.md`.
- **Doc reconciliation in the SAME pass.** Whenever an equation is verified against a PDF, reconcile the
  website pages too (`docs/website/10-theory/spatial-profiles/king.md`,
  `docs/website/50-validation/michie-anisotropy.md`, and any page stating the Michie DF / β(r) / Jeans /
  B&M82 equations). Every equation encoded in code must be documented identically and correctly — no
  stale/fabricated form survives. This is part of each task's DoD, not deferred to a MyST step.
- **TDD; never loosen a gate.** Fix the physics, or change *what* a test probes — anchors are physical.
- **JAX-native only** (jnp/equinox; reverse-mode differentiable). Zero new deps.
- **Verify LOCALLY; HITL.** Nothing pushed/merged without Anna's explicit go. Commit per task; messages
  end with the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer. Stage files explicitly.

## Tasks (risk-ascending; TDD, one subagent each, code review between)

**Tier 1 — safe numerical refactors (no physics change; regression against existing 31 anchors):**
- **A — tabulate-once-then-project (#3).** Restructure the projection integrator. *First*, because B
  touches the same code. DoD: ~1e-9 regression + projection physics green + speedup + docs reconciled.
- **B — regression-gate `r_t`/profile-param gradients (#4, re-scoped).** No reparam (empirically
  unneeded). DoD: AD-vs-FD regression gates for EFF `r_t`/`γ` + King `W0` (clean); one documented gate
  for Michie `W0` cross-ref'ing `2026-06-16-michie-king-equilibrium-gradient-redesign-deferred.md`.
- **C — algebraic compactification (#2).** DoD: Plummer residual → machine precision, O(h²) preserved,
  King/EFF untouched + docs reconciled.

**Tier 2 — the capability increment (new physics surface; HITL checkpoint before starting):**
- **D0 — paper grounding.** Read Michie-1963/King-1966 PDFs; verify+correct the per-paper note and the
  Michie/King website theory+validation pages. **Anna HITL checkpoint on the verified DF form + oracle
  design before any D1 code.**
- **D1 — general-β `jeans_dispersion`** (`beta_fn`, default OM analytic; numerical factor on opt-in).
  OM regression bit-preserved; Tier A reduces to current solver for β_OM.
- **D2 — `df_moment_dispersion`** (Tier B exact anisotropic Michie moment; polar fixed-domain quadrature).
- **D3 — new Michie all-radii anchor:** Tier A(native β)≈Tier B at all radii (kills inner-region-only) +
  Tier A(β_OM)==current (regression) + sampler 5% MC at all radii + zeroth-moment self-consistency +
  AD-vs-FD grads for the new path.
- **D4 — registries:** register `df_moment_dispersion` across all four (api_coverage, physics
  EXEMPT_NON_MODEL, grad_audit SYMBOL_CATEGORY+MUST_AUDIT+Case, provenance Michie-1963 row) + regenerate
  the dashboard + re-stamp coverage (the Phase-0 dogfood playbook).

## Definition of Complete (Phase 0.5)

- [ ] A: `project_dispersion` ~1e-9 regression + structural speedup; projection physics green.
- [ ] B: EFF `r_t`/`γ` + King `W0` AD-vs-FD regression gates clean; Michie `W0` documented gate +
      deferred-note cross-ref; existing physics unchanged.
- [ ] C: Plummer isotropic residual → machine precision; O(h²) convergence preserved.
- [ ] D0: Michie-1963 DF form verified against PDF; per-paper note + website pages reconciled.
- [ ] D: general-β Jeans (OM regression bit-preserved) + `df_moment_dispersion`; Michie correct at ALL
      radii (Tier A≈Tier B + sampler 5% MC + zeroth-moment self-consistency); new-path AD-vs-FD grads clean.
- [ ] D4: four registries updated (1 new symbol) + dashboard re-stamp + coverage re-measure; all green.
- [ ] FULL released-core gate green (count captured); completion doc `.claude-work/`; STATUS + brain +
      memory; **Anna merge-go** before Phase 1.

## Sequencing & HITL

A → B → C → (HITL checkpoint) → D0 → (HITL checkpoint on verified DF + oracle) → D1 → D2 → D3 → D4 →
FULL gate → Anna merge-go → **then Phase 1 OED on the SoTA engine**. Anna approves every decision, the
plan, and the merge. Verify locally (CI minutes tight). One PR at the arc's end (Phase 0.5 may fold into
the OED-arc PR or stand alone — Anna's call at merge time).
