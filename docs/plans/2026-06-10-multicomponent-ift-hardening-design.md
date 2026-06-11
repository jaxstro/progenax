# Multicomponent Engine A — Implicit-Function-Theorem Hardening + B2 Speed — Design

**Date:** 2026-06-10 (brainstormed with Anna; astro-code-dev + brainstorming skills)
**Status:** APPROVED design v2 — **hand-rolled `jax.custom_vjp` IFT** (optimistix
dropped: empirically the slow path). Anna: drop optimistix, fold in cheap wins,
plan-first → verify → subagent-driven dev.

## Purpose

Make the most complex differentiable object in progenax — the multicomponent
equipartition cluster `MultiComponentCluster.from_imf` (Engine A) — **faster and
more correct** under autodiff, so the B2 inference demos (Tasks 4–6) and future
ecosystem cluster-inference run at SoTA speed without sacrificing accuracy.

**Profiling + prototype evidence (measured, fresh):** the B2 joint MLE
(4 inits × 600 Adam steps) takes ~50 min at full accuracy. The dominant cost is
`from_imf` → `find_alpha_for_masses`: a fixed `n_iter=30` eigenvalue **fixed-point
scan**, each step a coupled N_COMP-component Poisson BVP ODE, **unrolled and
differentiated**. A benchmarked hand-rolled IFT prototype (preserved at
`docs/plans/_ift_prototype_reference.py`) cuts one `value_and_grad` of the solve
from **716 ms → 249 ms (2.9×)** with gradients exact to **3.3e-7 vs central
finite differences**. The full-accuracy reference recovery (the gold anchor) is
α̂=2.2931±0.0041, δ̂=0.3972±0.0344, Ŵ₀=4.9900±0.0209 (all 3σ PASS).

## The optimistix detour (recorded so we don't repeat it)

We first chose `optimistix.fixed_point(..., adjoint=ImplicitAdjoint())`.
Empirically it **does not run** (its adjoint builds `(I−∂map/∂α)` by **JVP**,
which the inner diffrax ODE's reverse-only `custom_vjp` rejects:
`can't apply jvp to a custom_vjp`), and the only fix —
`diffrax.DirectAdjoint` — makes it **slower than unrolling** (0.76–0.90× at the
n_iter we use; it only wins past n_iter≈54). **Rejected.** The win comes from
building the implicit operator by **reverse-mode `vjp`** instead, which uses the
ODE's efficient checkpointed adjoint — a hand-rolled `jax.custom_vjp`, no
optimistix/lineax dependency.

## Why IFT now (not a past oversight)

CLAUDE.md doctrine is "fixed-length `lax.scan`, never `while_loop`,
differentiable by unrolling." Unrolling is the robust always-correct default;
`from_imf`'s original use (build one IC / a validation check) made 30 unrolled
iters trivial. The bottleneck exists only because this demo puts `from_imf`
inside a 2400-eval MLE — a new usage. This is the profile-justified moment
(astro-code-dev: Correctness > Performance; profile before optimizing).

**Two wins, not one:** (a) **speed** — the implicit backward is ~46 ms and
**flat in n_iter** (vs unrolled ~400 ms that scales with the loop); (b)
**robustness** — the forward becomes an **adaptive `while_loop` to a residual
tolerance**, so it always converges (9–12 iters across the whole (α,δ,W₀) box,
vs a fixed 30 that is both wasteful and brittle in hard corners). Note: at our
converged residual the current unrolled gradient is *already* exact to ~1e-8, so
IFT's correctness gain is marginal — the real prizes are speed + robustness.

## Blast radius (mapped against the code)

- `find_alpha_for_masses` has **one** src caller: `from_imf`
  (`multicomponent.py:379`). `from_imf` has **one** released-core test
  (`test_multicomponent.py:296`) — **forward-only** (residual<2e-3, shapes,
  equipartition ordering; no gradient).
- **Released-core DOES differentiate `find_alpha` directly** (corrected after
  v2 verification): `test_limepy_multimass.py::test_differentiable_in_targets_and_delta`
  (lines 221–241) takes `jax.grad` w.r.t. (M_j, δ) through
  `find_alpha_for_masses` **isotropic** (`ra_hat=None`, n_comp=3). This is a
  *second* gradient consumer — and a benefit: the 1163 invariant now exercises
  the new IFT gradient path (H2 Step 4 runs this test). The refactor MUST keep
  it green (it crashes the naive single-solver `-None` backward → see the
  two-solver requirement below).
- The diffrax ODE *inside* keeps its own `custom_vjp` (we use its efficient
  reverse). Only the **outer eigenvalue scan** changes.
- **Containment:** the refactored forward converges at least as tight as the old
  (residual ≤ old), so forward values are unchanged to ~residual → the 1163
  tests stay green; gradients change only on paths through `from_imf`, whose sole
  gradient consumer is the B2 demo (validated against the reference).

## Approach

### Part 1 — hand-rolled `jax.custom_vjp` on `find_alpha_for_masses` (released-core)

Keep the signature and the Gieles & Zocchi 2015 §4.1 √-update map. Reference
implementations: `docs/plans/_ift_prototype_reference.py` (Prototype 2 —
benchmarked) and `docs/plans/_ift_realsig_reference.py` (the v2-verified
real-signature fix). Structure:

**Two `custom_vjp` solvers, dispatched on `ra_hat is None` (v2 verification —
required).** A single solver crashes: with `ra_hat=None` the backward computes
`−gra` where `gra=None` → `TypeError: bad operand for unary -`. So:
- **Isotropic** (`ra_hat is None`, the demo + the released grad test): `ra_hat,
  eta` are NONDIFF (closed over); differentiate `(m_j, M_j, W0, g, delta)`;
  backward returns a 5-tuple. (Rejected alternative: a traced sentinel can't hit
  the Python `is None` branch in `_realized_fractions`, so it would route the
  isotropic case through the slow/wrong anisotropic table path.)
- **Anisotropic** (`ra_hat` finite): differentiate
  `(m_j, M_j, W0, g, delta, ra_hat, eta)`; 7-tuple backward.
Share the `while_loop` body via the module-level `_alpha_map`/`_alpha_residual`;
only the thin `fwd`/`bwd`/`defvjp` differ. Both verified: grad vs FD ≤2.2e-7.

- **Forward (`fwd`, not differentiated):** adaptive `jax.lax.while_loop` running
  `α ← normalize(α·√(f_target/f_real(α,θ)))` until
  `max_j|f_real−f_target| < tol` (tol=1e-6) or a `max_iter` cap (=`n_iter`,
  default 30). Legal because `custom_vjp.fwd` is never differentiated — the
  implicit rule supplies the gradient (this is the standard jaxopt/optimistix
  internal pattern). Save `(α*, differentiable θ)`.
- **Backward (`bwd`, reverse-mode implicit VJP):** with the **fixed-point map
  residual** `R(α,θ) = α − normalize(α·√(f_target/f_real(α,θ)))` (R(α*,θ)=0):
  ```
  _, vjp_alpha = jax.vjp(lambda a: R(a, θ), α*)
  J_alpha = vmap(lambda e: vjp_alpha(e)[0])(eye(n))   # 4x4, reverse-mode (efficient ODE adjoint)
  w = jnp.linalg.lstsq(J_alpha.T, alpha_bar, rcond=None)[0]
  _, vjp_theta = jax.vjp(lambda th: R(α*, th), θ)
  theta_bar = tree_map(jnp.negative, vjp_theta(w))
  ```
- **Critical correctness details (from the benchmark):**
  - Use **R (the √-map residual)**, NOT `f_real−f_target`: J_α(R) has condition
    number **2.6** with a *benign* Σ=0 null direction (the simplex tangent), so
    `lstsq` min-norm is exact; `f_real−f_target` has condition **1e16** → wrong
    gradients.
  - **`jnp.linalg.lstsq(rcond=None)`** for the 4×4 solve — correct and finite at
    box edges (pinv/Tikhonov also work; lstsq is cheapest).
  - **Backward accuracy is set by forward convergence** (it linearizes at the
    returned α*): forward **tol ≤ 1e-5** required; use **1e-6** for margin
    (gives 3.3e-7 vs FD).
  - Thread the differentiable params (m_j, M_j→f_target, W0, g, delta, [ra_hat,
    eta]) as **explicit args** to the `vjp` calls (clean reverse-mode); statics
    (xi_max, n_points, aniso_method, tol, max_iter) via closure / nondiff.
- Return `(alpha_final, residual)` unchanged (residual = last forward iterate's
  `max|f_real−f_target|`, a diagnostic — carry it out of the forward to avoid an
  extra ODE solve; it gets a zero cotangent).

### Part 2 — adaptive forward subsumes the `n_iter` trim (released-core)

The adaptive `while_loop` converges in ~9–12 iters everywhere (vs fixed 30), so
the "n_iter trim" cheap win is automatic and *robust* (it always hits tol). The
`n_iter` arg becomes the safety cap (default 30).

### Part 3 — B2 demo optimizer right-size (demo-level, no physics cost)

`mle_adam`: 4×600 → **3 inits × ~300 steps** (plateau-verified; ≥3 inits keeps
the secondary-min detection the reference exhibited). Demo-side only.

## Validation gates (the contract — never weakened)

1. **Forward consistency:** new α* is a valid fixed point (`residual<2e-3`) and
   agrees with the old fixed-30 solve to `<1e-4`; a stored regression pin on the
   new α* (rtol 1e-8 self-consistency); the **1163** released-core suite green.
2. **Gradient correct:** new TDD test — implicit grad of `find_alpha`/`from_imf`
   w.r.t. (M_j, δ, W₀) matches **central finite differences** to `<1e-5`
   (measured 3.3e-7), and matches the old unrolled grad to ~residual. This is the
   gate that guards the `lstsq`/R-residual choice.
3. **Demo recovery unchanged:** re-run `demo_delta_recovery`; θ̂ within Fisher σ
   of the reference; ALL PASS; report new wall-clock (target ≈3× per-eval ×
   optimizer reduction).
4. **Robustness:** adaptive forward converges (`residual<2e-3`) at the box
   corners {(1.9,0.1,4), (2.3,0.4,5), (2.7,0.6,7)} in ≤ cap iters.

## Risks & mitigations

- **`while_loop` in `fwd`:** legal (undifferentiated; jit/vmap-safe) but MUST
  stay inside the `custom_vjp` — a bare `while_loop` breaks `jax.grad`. Gate 2
  (grad-match) and the 1163 suite catch any wiring error.
- **Forward-tol vs backward accuracy:** tol=1e-6 (margin under the 1e-5 gate).
- **Rank-deficient solve:** R-residual + `lstsq` (cond 2.6) — gate 2 proves it;
  pinv is the drop-in fallback.
- **Released-core change:** TDD (characterization-first) + 1163 invariant +
  grad-match; blast radius contained (above).
- The δ→1 Spitzer ODE crash is orthogonal (inner-ODE max_steps), guarded by the
  demo's `DELTA_BOX=(0,0.7)`.

## Out of scope

Engine B / King-tidal-radius IFT (future candidates, noted); changing the
physics (N_COMP=4, the √-map); the B2 science beyond re-validating Task 4
(Tasks 5–9 resume after this lands and the demo re-validates vs the reference).
