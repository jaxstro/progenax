# Differentiable mass-segregation observables — design

**Date:** 2026-06-09
**Status:** approved (brainstorm complete), implementation starting
**Author:** Anna Rosen + Claude (brainstorming skill)

## Motivation

progenax can *measure* mass segregation (`compute_lambda_msr`, Allison et al. 2009,
SciPy-MST) and can *generate* segregated clusters with a differentiable knob
(`lambda_seg` / `MassSegregationLayer`). But it has **no differentiable segregation
*observable*** — a function `f(positions, masses) -> scalar` whose gradient is usable.
That gap blocks gradient-based / HMC inference of segregation directly from a forward
model; today only SBI is clean.

Crucial distinction (this confused the `mass-segregation.md` prose):

- `lambda_seg` is a **forward-model input parameter** — differentiable because the
  generator is built from smooth blends. `∂(model)/∂lambda_seg` works.
- A differentiable **observable** is `∂f/∂(positions, masses)` on arbitrary inputs.
  This is what we are building. It does **not** exist yet.

Anchor: **mimic what observers actually infer.** Observers report segregation through
several distinct estimators, on **projected 2D** positions and a **mass proxy**
(luminosity → mass). We mirror the most-reported estimators and keep the inputs
observer-faithful.

## Scope

**Milestone (A) — this work:** parameter-space gradients
(`∂observable/∂θ` through the differentiable forward model — the HMC-from-forward-model
use case). Any observable that avoids `argsort`/hard-`min` is automatically clean in
data-space too, so (A) does not preclude (B).

**Three observables, one shared weighting kernel, plus a head-to-head comparison.**
Holding the mass-weighting identical across all three makes the comparison a controlled
experiment: the observables then differ *only* in their spatial statistic.

**Inputs:** 2D-projected positions + **true masses** now; `project_to_2d=False` flag for
3D theory mode. 2D-vs-3D is itself a study axis (see Research questions).

### Deferred (YAGNI — explicitly out of this milestone)

- **(B) data-space inference demo** (per-star position likelihood / optimize a
  configuration directly) — follow-up.
- **Noisy mass proxy** (luminosity scatter, completeness) — data-realism layer,
  follow-up.
- **Learned 2D→3D deprojection mapping** — research thread (see below).
- No new SBI; **no changes to `compute_lambda_msr`** — it stays the validation oracle.

## Shared kernel — the "weighted Λ_MSR"

Every star carries a smooth weight (sigmoid soft mass-cut):

```
w_i = sigmoid((m_i - m_cut) / tau),   W = sum_i w_i
```

- Mirrors the observer's choice of a *massive bin* defined by a mass/luminosity cut
  `m_cut`; the hyperparameter is an interpretable observational choice, not a nuisance.
- **Hard limit:** as `tau -> 0`, `w_i -> 1[m_i > m_cut]`, recovering the exact
  "massive subset" → the exact estimator. This is the central validation route.
- Robustness variant: power-law weight `w_i ∝ m_i^p` (drop-in alternate kernel).

## The three observables

Positions projected to 2D first if `project_to_2d`. All scalars, smooth in positions
and `m_cut`, fixed-shape, JIT/grad/vmap-safe (no `argsort`, no un-softened `min`).

### 1. Soft Λ_MSR (MST-ratio surrogate)

Softens both discrete ops in `compute_lambda_msr`:

- **MST length** → nearest-neighbour estimator (the `q_approx` recipe), with a
  **softmin** nearest-neighbour distance (temperature `beta`) replacing hard `min`:
  `d_i^1NN_soft = softmin_j(d_ij)`.
- **Ratio vs random subsets** → **closed-form expectation** instead of Monte-Carlo:
  the "random" baseline is the *unweighted* (all-`w=1`) mean NN-length.

```
Lambda_soft = <d_1NN>_all  /  ( sum_i w_i d_i_1NN / sum_i w_i )
```

Differentiable analogue of `<L_random> / L_massive`; no sampling; → exact Λ_MSR as
`tau, beta -> 0`.

### 2. Radial concentration (no graph, cleanest gradient)

```
C = ( sum_i w_i r_i / W )  /  ( sum_i r_i / N ),   r_i = |x_i - xbar_w|
```

Mass-weighted mean radius over unweighted mean radius. `C < 1` ⇒ massive stars more
central ⇒ segregated. Optional half-mass-radius-ratio variant.

### 3. Soft Σ–m (mass-in-dense-regions; Maschberger–Clarke 2011)

```
S = corr_i( w_i, log Sigma_i ),   Sigma_i = (k-1) / (pi (r_ik_soft)^2),  k = 6
```

`r_ik_soft` = softmin-k kNN radius. `S > 0` ⇒ massive stars in denser regions.

## Public API — `src/progenax/diagnostics/segregation_approx.py`

Mirrors `q_approx.py` (surrogate + calibration):

- `soft_mass_weights(masses, m_cut, tau)` — shared kernel.
- `lambda_msr_approx(positions, masses, *, m_cut, tau, beta, project_to_2d=True, calibration=1.0)`
- `radial_concentration_approx(positions, masses, *, m_cut, tau, project_to_2d=True, calibration=1.0)`
- `sigma_m_approx(positions, masses, *, m_cut, tau, k=6, beta, project_to_2d=True, calibration=1.0)`
- `calibrate_segregation_approx(...)` — fit each surrogate's multiplicative calibration
  vs its exact oracle, as `calibrate_q_approx` does.

JAX-native (no scipy in observable code; scipy stays in the validation script/tests as
the oracle). Released-core (pure JAX).

## Validation tier — `tests/validation/test_segregation_approx_physics.py`

Assert against external/exact oracles, never self-consistency.

- **Oracle 1 — hard limit (central):** as `tau, beta -> 0`, each soft observable
  converges to its exact non-diff counterpart on the *same* configuration
  (`compute_lambda_msr`; exact NumPy radial ratio; SciPy-cKDTree Σ–m). Relative error
  decreases monotonically in `tau` and is below tolerance at the smallest `tau`.
- **Oracle 2 — monotonicity/sign:** on an `energy_sorted_segregation` strength sweep,
  all three move monotonically in the segregation direction (Λ↑, C↓, S↑) and
  rank-correlate (Spearman) with exact Λ_MSR.
- **Oracle 3 — hand-constructed regimes** (reuse §1 fixtures): unsegregated → null;
  maximally segregated → strong; inverse → opposite sign.
- **Oracle 4 — differentiability:** autodiff `∂/∂m_cut` and `∂/∂x` finite + match
  central finite-difference (float64, rel < 1e-5); JIT + vmap; no-NaN on degenerate
  inputs (coincident points, equal masses).

~12–15 validation tests + unit tests in `tests/unit/diagnostics/`.

## Figures — `scripts/validate_segregation_approx.py` (5, PASS/FAIL per panel)

1. **Hard-limit convergence (headline):** all three vs `tau` (and `beta`) → exact
   oracle as `tau -> 0` (log-log error). Correctness figure.
2. **Segregation response curves:** the three (normalized) vs segregation strength on one
   axis, exact Λ_MSR overlaid. Sensitivity / dynamic range.
3. **Fisher-information identifiability:** `I(theta) = (d<obs>/dtheta)^2 / Var` per
   observable w.r.t. segregation strength. Higher = more inferable → **ranks the three
   for HMC use.** The differentiable payoff.
4. **2D vs 3D bias panel:** each observable 2D-projected vs 3D on the same clusters;
   bias + 2D/3D **Fisher ratio** = fraction of segregation signal surviving projection.
   Annotated with the open research question.
5. **Differentiability / gradient validation:** autodiff vs finite-difference for
   `∂obs/∂m_cut` (all three); plus a toy gradient-descent recovery of segregation
   strength through one observable (end-to-end "works for inference", mirroring
   `recover_lambda_seg_via_gradient_descent`).

## Documentation

Extend `docs/website/50-validation/mass-segregation.md` (keep the segregation story in
one place):

- New **§5 Differentiable segregation observables** with real **Measured** column +
  the 5 figures.
- Fix **unicode → KaTeX** math throughout the existing page (`$\Lambda_{\mathrm{MSR}}$`,
  `$\Sigma$`, `$\rho$`, …).
- Fix the misleading prose conflating `lambda_seg` (a differentiable *forward-model
  parameter*) with a differentiable *observable* — state the distinction explicitly.
- **2D-vs-3D research-question admonition** (below).
- Dashboard row in `index.md` (✅ figures, date) + `audit-report.md` completeness/roadmap.

## Research questions (recorded, not in build scope)

- **2D vs 3D segregation mapping.** Because the observables are differentiable, quantify
  *how much segregation information projection destroys* via the 2D/3D **Fisher ratio**
  per observable. Then: can a small **learned/fit deprojection correction**
  `Lambda_3D ≈ g(Lambda_2D, cluster shape)` recover the 3D value? A candidate "better
  substructure mapping" between 2D and 3D methods. Follow-up thread.
- **(B) data-space inference** + **noisy mass proxy** — the observer-realism layer.

## TDD order (RED → GREEN per step)

1. Shared `soft_mass_weights` kernel + hard-limit unit test.
2. `radial_concentration_approx` (simplest — proves the pattern). **← pause for sign-off.**
3. `lambda_msr_approx` (soft MST ratio).
4. `sigma_m_approx` (soft kNN Σ–m).
5. `calibrate_segregation_approx` + `__init__` exports.
6. Validation tier (Oracles 1–4).
7. Comparison/Fisher figure script.
8. Extend `mass-segregation.md` + dashboard + audit-report; `myst build` verify.

## Verification gate before "done"

- Validation + unit tests green; **released-core invariant still green**.
- `validate_segregation_approx.py` prints 5× PASS; 10 files (PNG+PDF) land.
- `myst build --html` clean; all figures in `_build/site/public/` **and** referenced in
  the page content JSON.
- Real Measured numbers on the page (no placeholders).
- Commit per logical step; **push only when Anna says**.
