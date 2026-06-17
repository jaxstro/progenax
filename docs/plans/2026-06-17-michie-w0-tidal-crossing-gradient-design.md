# Design — W₀-differentiable dispersion for OED/Fisher (all truncated models)

**Date:** 2026-06-17 (revised after the 2026-06-17 implementation-arc prototypes)
**Status:** ROOT CAUSE RE-PINNED + REAL FIX VALIDATED. The earlier "normalized-coordinate"
fix (and ADR-0016 as first written) was **falsified by prototype** — it is an exact no-op;
the genuine fix is a **C¹ smooth back-interpolation**. Ready for a TDD plan. THREE earlier fix
hypotheses were refuted by prototype (kept below for provenance).
**Owner:** Anna (HITL).

## Goal (Anna, 2026-06-17)

OED/Fisher must differentiate `σ(W₀)` cleanly for **all** truncated progenax models
(King / Michie / EFF), since they share the `jeans_dispersion` / `project_dispersion` forward
model. Bar: FD-consistent `∂σ/∂(profile param)` < 1e-3 at the standard gate step, model-general.

## Validated root cause (discriminating experiments, 2026-06-17)

The Michie-W₀ xfail is **not** a wrong gradient (AD matches FD to ~1e-8 at fine h). There are
TWO distinct effects, only one of which is a code defect:

| Finding | Test (this arc) | Verdict |
|---|---|---|
| Literal normalized-coordinate swap `interp(r,s,T)` → `interp(r/r_t, s/r_t, T)` | `max\|·−·\|` over the master tables | **NO-OP — exactly 0.000e+00.** `jnp.interp` is scale-equivariant; relabeling the abscissa changes nothing for a fixed physical query radius. **This falsifies the ADR-0016 / prior-doc fix.** |
| **C⁰ piecewise-linear back-interp bracket kink** | C¹ cubic-Hermite back-interp, fixed physical r | **DOMINANT, REAL DEFECT.** W₀=6 rel 5.1e-3 → **3.4e-4** (passes) with a C¹ interpolant. |
| High-W₀ residual (W₀=7) | fixed-`u` query, `n_s` 4000→64000 | **Flat ~3.15e-3 — NOT resolution.** Survives removing the back-interp kink. |
| Is the W₀=7 residual a real gradient defect? | FD-vs-AD at W₀=7, h-sweep 1e-3→1e-6 (fixed-`u`) | **No.** FD **converges monotonically to AD** (3.2e-3 → 2.1e-5); AD is correct. `r_t`/`ρ` gradients individually clean (~1e-4). |
| Why is coarse-h FD unreliable at W₀=7? | `r_t(W₀)` sweep | **Near mass-divergence:** `r_t` = 28→58→133→**545** over W₀ 6.0→7.0; no finite truncation past W₀≈7.1 (r_a=5). The forward map is genuinely stiff; a fixed-step FD has large O(h²·f‴) truncation error there. |

**Mechanism (the real defect).** For finite-`r_t` profiles the master s-grid is
`linspace(1e-4·r_t, r_t, n_s)`; its endpoint **moves with r_t(W₀)**. The final
`jnp.interp(r, s_grid, σ_table)` is **piecewise-linear (C⁰)**. For a fixed physical query r,
as r_t(W₀) moves the bracketing nodes switch and the interpolant's *slope* w.r.t. W₀ **jumps** →
a ~1e-4-scale kink a coarse central FD straddles. The cure is **not** the abscissa coordinate
(proven a no-op) but the **smoothness of the interpolant**: a C¹ scheme has no slope jump at a
node crossing.

**The high-W₀ effect is not a code bug.** Near the Michie mass-divergence the AD gradient is
correct (FD→AD as h→0); the coarse-gate-h inconsistency is a finite-difference *truth-proxy*
artifact in a high-curvature region. No code change can or should remove it.

## Validated fix

1. **C¹ smooth back-interpolation** of the master tables onto the query radii in
   `_sigma_r2_from_tables` (replace the three `jnp.interp` calls — `rho`, `I_outward`,
   `F_shifted` — with a C¹ interpolant, e.g. monotone cubic Hermite / PCHIP on the monotone
   `s`-grid). Model-general: King/Michie/EFF share this path; Plummer (compactified `s`) shares
   it too, so `∂/∂r_h` is covered. **Validated:** W₀=6 5.1e-3 → 3.4e-4.
   - *Caveat from prototype:* a naive central-difference-slope cubic adds edge noise (made
     W₀=7 worse). Use a **monotone/limited-slope** C¹ scheme (PCHIP-style) and re-measure.
2. **Gate methodology for the near-divergence regime** (replaces the speculative "fixed
   integration grid" — unnecessary, since the high-W₀ effect is not a defect): keep the Michie
   gate within the well-truncated regime (W₀=6) and, where a stiff forward map is tested,
   assert AD against a **converged / Richardson-extrapolated** FD rather than a single coarse
   step. Document that beyond W₀≈7 (r_a=5) the model approaches mass-divergence (r_t→∞) where
   the gradient is still correct but a fixed-step FD is a poor check.

### Hard constraints (shared `kinematics/dispersion.py`)
Shared by Plummer/King/EFF and `project_dispersion`. A C¹ interpolant **changes forward values**
relative to today's linear interp (different interpolation error) → MUST re-validate / re-pin the
tight anchors: Plummer isotropic rtol 1e-3, King c(W₀) Table II (≤0.002), and the
`project_dispersion` **rtol-1e-9 regression baseline** (a pin of the *current* linear-interp
values — it WILL move and must be re-pinned, not weakened). The DF-side cross-check
(`ftable_sigma_r_isotropic`) and `df_moment_dispersion` (separate quadrature path) are unaffected.

---

## (Superseded hypotheses, retained for provenance)

### (Superseded) — differentiable tidal-radius crossing

The "differentiable tidal-radius crossing" design (below) was **disproven during prototyping**:

- The **current `_find_tidal_radius` gradient is already clean** — `d(ξ_t)/dW₀` AD-vs-FD is
  ~2e-8 at W₀=5,6 *even at the coarse gate step h=6e-4*; a custom_jvp prototype made it
  **worse** (1e-2). r_t is **not** the source of the σ_r(W₀) kink.
- **Localization (decisive):** the σ_r(W₀) coarse-h inconsistency **scales with the jeans
  integration grid `n_s`** (rel = 2.9e-4 / 5.1e-3 / 3.1e-3 / 1.2e-3 at n_s = 2000/4000/8000/
  16000) and is **flat in `n_ode`** (~5–6e-3 at 3000/6000/12000 — confirming the prior doc's
  "ODE-resolution-independent" note).
- **True root cause:** the density **truncation `jnp.where(r ≤ r_t, ρ, 0)`** applied in the
  **jeans integrand on a fixed `n_s` grid**. As r_t(W₀) sweeps across a fixed jeans node, that
  node flips in/out of the truncation → the Jeans integral steps → σ_r(W₀) gets a ~1e-4-scale
  kink a coarse FD straddles. The kink is in the **dispersion integral's moving-truncation
  handling**, not the profile or the tidal radius.

**Corrected fix surface:** make the jeans/dispersion integral treat r_t as a **smooth
integration limit** (or smoothly-tapered truncation) instead of a hard mask on a fixed `n_s`
grid — in `kinematics/dispersion.py`, which is **shared by Plummer/King/EFF** and carries TIGHT
anchors (Plummer isotropic rtol 1e-3; `project_dispersion` rtol-1e-9 regression baseline).
Higher-stakes than a profile-local custom_jvp; needs its own brainstorm + regression plan.
**Do not implement the tidal-crossing design below.**

---

## (Superseded) original design — differentiable tidal-radius crossing

**Status:** REFUTED (see correction above). Retained for provenance.

## What the diagnosis actually found (2026-06-17 discriminating experiments)

The deferred doc reported "Michie `∂σ_r/∂W₀` ~5e-3 FD-inconsistent" and proposed a forward-solve
redesign (fixed-step integration). Re-diagnosis overturned two of its hypotheses and pinned the cause:

1. **The AD gradient is CORRECT.** With the xfail's exact setup
   (`jeans_dispersion(MichieProfile.from_W0_rc(W0, r_c=1, r_a=5), None, [1.0], 400, G).sigma_r`,
   `r_a=None` ⇒ isotropic Jeans on the Michie density), AD vs central-FD:

   | W₀ | rel @ h=6e-4 (gate) | rel @ h=1e-5 |
   |----|---------------------|--------------|
   | 5  | 3.1e-4 (passes)     | **4.1e-9**   |
   | 6  | 5.1e-3 (fails)      | **2.7e-8**   |
   | 7  | 2.4e-2 (fails)      | 1.4e-3       |

   At a fine FD step the gradient matches to ~1e-8. The "inconsistency" is a **coarse-FD artifact**.

2. **The inner `michie_density` quadrature is EXONERATED.** `dρ̂/dW` is AD-vs-FD clean to ~1e-8 at
   every `n_u` ∈ {256, 1024, 4096, 16384}, every `s`, every `W` — `n_u`-independent. (The doc never
   tested this axis; it is not the cause.)

3. **Domain size is not the driver.** The gap is *non-monotonic* in `xi_max` (200→4.8e-3, 400→1.0e-3,
   800→6.3e-4) — refuting "more steps over a larger domain amplifies it."

4. **Root cause = the `argmax` node-pair selection in `_find_tidal_radius`.** The crossing
   `ξ_t = ξ₀ + ψ₀/(ψ₀−ψ₁)·(ξ₁−ξ₀)` is already smooth *within* a node interval (a prior resolved arc,
   `2026-06-08-king-differentiable-tidal-radius-deferred.md`), but `first_zero_idx = argmax(ψ≤0)`
   makes the (ψ₀,ψ₁,ξ₀,ξ₁) bracket **jump** when the crossing slides across a grid node as W₀ varies
   ⇒ ξ_t(W₀) is continuous but **kinked** (slope-discontinuous) at node boundaries, at the ~1e-4 W₀
   scale. A coarse central FD (gate h=6e-4) straddles a kink and reports a secant that disagrees with
   the true local slope AD computes. King is clean because its smaller domain → finer ξ-spacing →
   far smaller r_t(W₀) steps at the same W₀.

This is the deferred doc's contributor #2 (the discrete crossing), not #1 (adaptive schedule) or #3
(inner quadrature). The forward micro-kinks are negligible for forward physics (≪ the King Table II
0.002 anchor) but break a coarse-FD gradient check and could bite large-step optimizers.

## Fix — primal-preserving `custom_jvp` smooth crossing

Wrap the crossing in a `custom_jvp`:

- **Primal:** bit-identical to today's `argmax` + linear-interp. Every forward value (King c(W₀),
  LIMEPY parity, multimass equilibrium, every `r_t`) is **unchanged** — the strongest possible
  regression guarantee for shared infrastructure.
- **Tangent:** the implicit-function-theorem derivative evaluated *at the continuously-moving* ξ_t,
  independent of which node-pair brackets it:

  ```
  dξ_t/dW₀ = − (∂ψ/∂W₀)|_{ξ_t} / (∂ψ/∂ξ)|_{ξ_t}
  δξ_t = − interp(ξ_t, ξ_grid, δψ_grid) / interp(ξ_t, ξ_grid, dψdξ_grid)
  ```

  Numerator = the W₀-induced perturbation of ψ at the *fixed* crossing radius; denominator = the
  exact ODE-carried slope ψ′ = `y[1]` interpolated to ξ_t. Both interps are smooth in ξ_t and in the
  grid tangents ⇒ **no node-crossing kink** in dξ_t/dW₀.

### Plumbing
- `_find_tidal_radius(xi_grid, psi_grid)` → `_find_tidal_radius(xi_grid, psi_grid, dpsidxi_grid=None)`.
  `None` ⇒ exactly today's behaviour (zero risk for un-migrated callers); supplied ⇒ smooth-tangent path.
- `solve_king_profile` / `solve_michie_profile` return the already-computed `solution.ys[:,1]` grid.
- Pass it at the King / Michie / LIMEPY / LIMEPY-multimass / multicomponent call sites (7 total).
- `xi_grid` is a fixed `linspace` (W₀-independent) ⇒ carries no tangent; only ψ-grids do.

## Hard regression guards (shared infra — 7 call sites)
- King c(W₀) vs King 1966 Table II: `max|Δlog₁₀c| ≤ 0.002`, **bit-identical** (assert ==, not ≈).
- King `∂σ/∂W₀` stays clean (smooth tangent is its continuous extension).
- LIMEPY reference parity + multimass equilibrium forward values byte-identical pre/post.
- Michie `∂σ/∂W₀` < 1e-3 at the gate h=6e-4, across W₀∈{5,6,7} within the physical-truncation range
  (W₀≲8 at r_a=5; beyond that the model has no finite tidal radius — a separate, correct `ValueError`).

## Test plan (TDD, measured-first; each guard RED→GREEN)
1. **Unit — smooth tangent:** `dξ_t/dW₀` matches a *Richardson-converged* FD (h∈{1e-3…1e-6}) to <1e-4;
   secant-slope spread across a node-boundary W₀-sweep collapses (direct kink regression).
2. **Primal bit-identity:** `_find_tidal_radius(...,dpsidxi)` returns `==` ξ_t as the 2-arg path; King/
   LIMEPY/multimass forward `r_t` byte-identical.
3. **Gate flip:** un-xfail `test_grad_jeans_michie_wrt_W0_DEFERRED`; assert `_assert_ad_fd` passes at h=6e-4.
4. **King Table II** anchor unchanged (≤0.002).
5. **Full released-core gate + grad-audit** regenerate: 0 hazards, Michie-W₀ row clean, `registries_full`
   holds, coverage floor holds. Remove the Michie-W₀ `PARAM_ALLOWLIST`/known-limitation entry.
6. **OED/Fisher unaffected:** Plummer/King grad cases unchanged (`audit_gradients.py` spot-check).

## Scope / sequencing
Shared-infra change touching 7 call sites + a `custom_jvp` + full re-validation ⇒ a properly-scoped
implementation arc (subagent-driven TDD, per-task review, Anna HITL checkpoint), **off `main`**, not
folded into the release-readiness audit branch. Does not block the v0.1.0 release (the xfail is fenced
and the OED/Fisher path never differentiates Michie-W₀).
