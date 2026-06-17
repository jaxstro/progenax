# Design — Michie-W₀ gradient via a differentiable tidal-radius crossing

**Date:** 2026-06-17
**Status:** DESIGN (brainstormed + diagnosis-pinned this session). Supersedes the
fix-direction in `docs/plans/2026-06-16-michie-king-equilibrium-gradient-redesign-deferred.md`
(that doc's root-cause narrowing was right; its proposed *fix* targeted the wrong contributor).
**Owner:** Anna (HITL). **Branch:** to be cut off `main` (not the audit branch).

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
