# Multicomponent Engine A — Implicit-Function-Theorem Hardening + B2 Speed — Design

**Date:** 2026-06-10 (brainstormed with Anna; astro-code-dev + brainstorming skills)
**Status:** APPROVED design (Anna: optimistix, fold in cheap wins, plan-first → verify → subagent-driven dev)

## Purpose

Make the most complex differentiable object in progenax — the multicomponent
equipartition cluster `MultiComponentCluster.from_imf` (Engine A) — **fast and
more correct** under autodiff, so the B2 inference demos (Tasks 4–6) and any
future ecosystem cluster-inference run at SoTA speed without sacrificing
accuracy.

**Profiling justification (measured, not assumed):** the B2 joint MLE
(4 inits × 600 Adam steps = 2400 grad evals) takes ~50 min wall-clock at full
accuracy. The dominant cost is `from_imf` → `find_alpha_for_masses`: a fixed
`n_iter=30` eigenvalue **fixed-point scan**, each step solving the coupled
N_COMP-component Poisson BVP ODE, **unrolled and differentiated** (≈30 ODE
solves forward + ≈30 in the backward pass per eval). The full-accuracy
reference recovery (the gold anchor for this work) is:

| param | truth | θ̂ (reference) | σ̂ | pull |
|---|---|---|---|---|
| α | 2.3 | 2.2931 | 0.0041 | −1.68σ |
| δ | 0.4 | 0.3972 | 0.0344 | −0.08σ |
| W₀ | 5.0 | 4.9900 | 0.0209 | −0.48σ |

All 3σ gates PASS; 3/4 inits hit the global min (−4511.130), 1 found a shallow
secondary (−4500.97). **Correctness is established; only speed is the problem.**

## Why IFT now (not a past oversight)

The codebase doctrine (CLAUDE.md) is "fixed-length `lax.scan`, never
`while_loop`, differentiable by unrolling." Unrolling is the robust,
always-correct default — autodiff just works. IFT is an optimization needing
care (convergence precondition; a rank-deficient linear solve from the Σα=1
constraint). `from_imf`'s original use was building **one** cluster IC or a
validation check (a few grad evals); the bottleneck only exists because this
demo puts it inside a 2400-eval MLE — a new usage pattern. This is the
profile-justified moment to optimize (astro-code-dev: Correctness > Performance;
profile before optimizing).

**IFT is also a correctness upgrade, not only speed.** The current unrolled
gradient is the gradient of the *30-step estimator* of α\*, carrying a small
convergence-tail error. IFT returns the **exact gradient at the true fixed
point**, independent of how many forward iterations got us there — strictly
better, and it decouples gradient accuracy from `n_iter` (which is what makes
trimming `n_iter` safe).

## Blast radius (mapped against the code)

- `find_alpha_for_masses` has **one** src caller: `from_imf`
  (`multicomponent.py:379`). `from_imf` has **one** released-core test
  (`test_multicomponent.py:296`, `test_constructs_and_hits_masses`) which is
  **forward-only** (checks `residual<2e-3`, shapes, equipartition ordering — no
  gradient). The released-core gradient tests touching `cluster` differentiate
  **Engine B** (`from_density_profiles`), **not** `from_imf`.
- The diffrax ODE *inside* already carries its own `custom_vjp` (diffrax
  adjoint). Only the **outer eigenvalue scan** is unrolled — that is the IFT
  target.
- **Containment:** if the refactor keeps the forward iterates identical
  (same √-map), **forward values are bit-identical** → all 1163 tests are
  value-safe by construction; **gradients change only on paths through
  `from_imf`**, whose sole gradient consumer is the B2 demo (validated against
  the reference).

## Peer instances (enumerated; only one true peer)

Of every iterative solve in `src/`, classified by *expensive AND on a
differentiated path*:
- **`find_alpha_for_masses`** — the standout (outer fixed point over an inner
  ODE; the only "solve within a solve"). **This design's target.**
- Engine B truncation radius (`density_poisson.py:123`, 80-step bisection) and
  King tidal radius (`king.py:263` — a comment already flags "implicit
  function…") — single root-finds, cheaper, not in an inference hot loop yet.
  **Noted as future IFT candidates; out of scope here.**
- IMF/Kepler Newton solvers (`base.py`, `chabrier.py`, `kepler.py`) — cheap
  scalar Newtons; unrolling is fine, IFT not worth it.

So `from_imf` is uniquely worth hardening — and it is the deepest differentiable
model in progenax (nested solve), versus single-ODE King/Michie, quadrature+Abel
Engine B, or closed-form Plummer/EFF/IMF.

## Approach

### Part 1 — optimistix IFT refactor of `find_alpha_for_masses` (released-core)

The fixed-point map (Gieles & Zocchi 2015 §4.1, the stabilized √-update) is
kept verbatim:

```
map(α, args) = normalize( α · sqrt(f_target / f_real(α, args)) )
```

Refactor the `lax.scan` to:

```python
sol = optx.fixed_point(
    map_fn, optx.FixedPointIteration(rtol=..., atol=...),
    y0=f_target, args=(m_j, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method),
    max_steps=n_iter,
    adjoint=optx.ImplicitAdjoint(linear_solver=lineax.AutoLinearSolver(well_posed=False)),
    throw=False,
)
alpha_final = sol.value
```

- **`FixedPointIteration` + same map → forward iterates identical** to the
  current scan (lowest risk; forward bit-identical at equal step count).
- **`ImplicitAdjoint` → exact fixed-point gradient** (the win, both speed and
  accuracy). Backward = one (rank-deficient) linear solve + ODE-linearizations
  instead of unrolling `n_iter`.
- **`AutoLinearSolver(well_posed=False)`** (lineax) handles the Σα=1 null
  direction in `(I − ∂map/∂α)` — the one fiddly spot; the gradient-match gate
  (below) is what proves it correct.
- `residual` is recomputed from `f_real(alpha_final) − f_target` exactly as
  today (a reported diagnostic, never branched on). Return signature
  `(alpha_final, residual)` unchanged.
- `throw=False` to keep it jit/scan-safe (no Python-side error on
  non-convergence; the residual diagnostic surfaces it). The δ→1 Spitzer crash
  is orthogonal (the inner ODE max_steps), already guarded by the demo's
  `DELTA_BOX`.

### Part 2 — `n_iter` (forward-convergence) trim (released-core, profiled)

With IFT, the gradient is `n_iter`-independent (exact at the fixed point), so
`n_iter` only needs to converge the **forward** residual `< 2e-3`. Profile
`residual` vs `n_iter` across the (α, δ, W₀) box and pick the smallest safe
value (precedent: `test_limepy_multimass.py:229` already exercises `n_iter=15`).
Keep `from_imf`'s default conservative; the demo may pass a profiled value.

### Part 3 — demo optimizer config (demo-level, no physics-accuracy cost)

`mle_adam`: reduce from 4×600 to a profiled `(n_inits, n_steps)` that (a) keeps
the plateau gate (last-10% improvement < 1%), and (b) retains enough dispersed
inits to find the global min reliably (the reference showed 1/4 inits hit a
shallow secondary → keep ≥3 inits incl. the near-truth z₀=0). Target: pin after
measuring; report wall-clock.

## Validation gates (the contract — never weakened)

1. **Forward value-safe:** new regression test pins `find_alpha_for_masses` α\*
   to the pre-refactor value (`rtol≤1e-8`); the full **1163** released-core
   suite stays green.
2. **Gradient correct:** new TDD test — implicit grad of `find_alpha`/`from_imf`
   w.r.t. (M_j, δ, W₀) matches **central finite differences** to ~1e-5, AND is
   consistent with the old unrolled grad to ~residual. This is the gate that
   catches a wrong rank-deficient solve.
3. **Demo recovery unchanged:** re-run `demo_delta_recovery`; θ̂ matches the
   reference (2.2931, 0.3972, 4.9900) within Fisher σ; ALL PASS; report new
   wall-clock (target ≫ 5× faster).
4. **`n_iter` choice:** residual `< 2e-3` across the parameter box (profile
   table in the completion doc).

## Risks & mitigations

- **Rank-deficient adjoint solve (Σα=1):** `lineax.AutoLinearSolver(well_posed=
  False)`; gate 2 proves correctness. Fallback: hand-rolled `custom_vjp` with an
  explicit tangent-space (simplex) projection if optimistix's solve is
  ill-behaved.
- **optimistix released-core dependency:** Anna-approved; same ecosystem as the
  existing equinox/diffrax deps; add to `pyproject` core deps + lock.
- **Forward-trajectory drift:** `FixedPointIteration` + same map keeps it
  identical; if a faster solver (`Chord`/`LevenbergMarquardt` root_find) is
  explored for forward speedup, re-run gates 1–3.
- **Convergence precondition:** IFT validity requires α\* converged; the
  residual gate (`<2e-3`) is the precondition, enforced by `n_iter`/tol.

## Out of scope

- Engine B / King tidal-radius IFT (future candidates).
- Changing the physics (N_COMP stays 4 — a modeling-fidelity choice, not a perf
  knob; reducing it weakens the δ signal).
- The B2 demo science (Tasks 5–9) — this hardening unblocks them but is its own
  arc; after it lands and the demo re-validates against the reference, resume
  the B2 task sequence.
