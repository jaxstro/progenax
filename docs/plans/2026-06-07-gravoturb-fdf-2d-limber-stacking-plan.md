# 2D Limber + Population-Stacking Inference — Implementation & Verification Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> **Supersedes** the A3–E build tasks in `2026-06-07-gravoturb-fdf-2d-projection-native-inference.md`
> (A1/A2 there are DONE and reused). Rationale + evolution: design doc §12 +
> `2026-06-07-gravoturb-fdf-2d-projection-native-inference-design.md`.

> **β-HEADLINE REFRAME (design §13, Anna-approved 2026-06-07).** β is THE deliverable: a
> differentiable, SBC-calibrated, physically-parameterized **successor to Q/MST** (same substructure
> content, made rigorous). **ℳ is secondary/forecast-grade** (cosmic-variance-limited per cluster,
> reported relative/population), **α is depth-gated** (not in the 2-D fit). **VALIDATED 2026-06-07:**
> the rank-G angular-clustering **slope→β is box-stable, monotonic, low-scatter** (−2.30/−2.81/−3.35
> for β=2.5/3.0/3.5; ±0.05–0.07 at n=96) — unlike the amplitude. Remaining method-validation before
> the full build: a minimal end-to-end **β-recovery** (unbiased) + a realistic **σ(β) forecast**.

**Goal:** A fast, SBC-calibrated, differentiable inference of the natal turbulence **spectral slope β**
(headline; the calibrated Q/MST successor) — with **amplitude ℳ** as a forecast-grade secondary and
**α depth-gated** — from the *projected star catalogue of a population of young clusters*, realised as
galaxy-angular-clustering-style inference robust to the fat tail and to finite-volume cosmic variance.

**Architecture (the two pillars, each doing a distinct job):**
1. **Analytic-2D Limber prediction (SPEED).** Predict the projected (angular) statistic from a *radial*
   3-D correlation + the Limber LOS integral (`limber_project_radial`, already built) + a single 2-D
   FFT — **O(n² log n)/eval**, never building a 3-D grid per evaluation. (The current 3-D-grid
   prediction is O(n³)/eval — measured 37 ms at 128³ — which would make a 128³ SBC ~1.5 h and negate
   the pivot.)
2. **Population stacking (ACCURACY).** The +0.5→+22% analytic-vs-finite gap is a **finite-volume /
   cosmic-variance bias** (verified: it shrinks 1.68→1.09 as the box grows 32→96; it is *not* shot
   noise or grid resolution). Stacking K clouds (data-vector stacking) gives the volume that makes the
   ensemble prediction match the data — and it *is* the science (β across a population).

**Statistic:** the **angular power spectrum (band-powers) of the rank-Gaussianised projected star
map**, stacked over the population. Slope→β, amplitude→ℳ (via σ_s²); rank-Gaussianisation
(Neyrinck+2011) tames the 1-pt tail; stacking tames cosmic variance. **α is depth-gated** — its POT
machinery is retained for the future depth-resolved mode and as a held-out tail stress-test, but is
**not** in the 2-D headline fit (verified α-washout).

**Tech Stack:** JAX (float64, jit/grad), Equinox, jaxtyping, pytest, numpy (data/oracle side),
blackjax NUTS. Experimental `gravoturb_fdf` (repo-only, `PYTHONPATH=src:src/experimental`).

**Non-negotiables (every task):** No hacks / no test-weakening — STOP & report to Anna if a principled
construction can't pass (this session's discipline). **Verify before build.** JAX-native +
differentiable. **SBC-valid**: data-derived quantities (rank-G map, jackknife C, depth, cell scale)
identical in generation + inference; covariances truth-independent. Ground primary-source/formula
claims in the actual PDF (no-assumptions). **HITL: Anna approves each task diff before commit.**
Released-core stays **814**.

**Run commands:**
- Experimental unit: `PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental/unit -q`
- Released-core gate (814): `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow"`
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. No `git add -A`; no push.

---

## Phase V — Verify-first gates (RUN and report to Anna BEFORE any build)

These are decision gates, not features. Each prints a table and a PASS/FAIL; **on FAIL, STOP and report
— do not build on a falsified premise** (cf. Philosophy A, which a verify-first check correctly killed).

### Task V1: Analytic-2D Limber prediction — prototype, time, and validate vs exact projection

**Files:** scratch script `src/experimental/gravoturb_fdf/validation/_v1_limber_speed_accuracy.py`
(throwaway; promote later if it passes). Reuse: `theory/projection.py::{gaussian_correlation_grid,
limber_project_radial, limber_project_slab, box_window_sq_grid}`, `theory/gaussianization`,
`theory/bm19::sigma_s_squared`.

**Step 1 — Build the analytic-2D-Limber predictor (radial → Limber → 2D):**
compute ξ_s(r) on a 1-D radial grid (from σ_s²·ρ_g(r); get ρ_g(r) radially via a 1-D profile of the
spectrum), lognormal map ξ_ρ(r)=expm1(ξ_s(r)), Limber-project with `limber_project_radial` to a 2-D
w(r_perp) grid, 2-D FFT → band-powers / cell variance.

**Step 2 — Timing:** time the analytic-2D predictor per eval at n=64/96/128 vs the current 3-D-grid
predictor (the §timing harness). **PASS-A:** analytic-2D is ≳10× faster at 128 and scales ~O(n²).

**Step 3 — Accuracy vs the EXACT discrete projection:** compare the analytic-2D (continuous Limber)
band-powers to the exact discrete projection (`limber_project_slab` on the full 3-D grid) at depth ∈
{n/2, n}. **PASS-B:** max-abs agreement < 5% (controls the flat-sky/continuum approximation). If it
fails for compact/shallow depth, the fallback (documented) is to keep the exact discrete LOS sum but
**precompute it once per eval in 2-D** (still O(n² log n) since the 3-D grid build, not the sum, was
the cost) — note which path passes.

**Step 4 — Report** the timing + accuracy tables to Anna. GATE: both PASS → proceed; else STOP.

### Task V2: Population stacking closes the cosmic-variance gap

**Files:** scratch `src/experimental/gravoturb_fdf/validation/_v2_stacking_gap.py`. Reuse the field +
projection chain (`gaussian_random_field`, the lognormal map, `project_counts_los`,
`measure_angular_bandpowers_2d` / rank-G).

**Step 1 — Stacked data vector:** for K ∈ {1,5,10,20} clouds (n=64, full-LOS), measure the rank-G
angular band-powers per cloud and **average** (data-vector stack). Compare to the analytic ensemble
band-powers (from V1's predictor). Report the gap vs K, per band and for the amplitude.

**Step 2 — GATE:** the analytic-vs-stacked gap falls toward **few-% by K≈10–20** (cosmic variance
↓ ~1/√K on scatter; bias ↓ with total volume). Report the K (or single-cloud volume) needed for
<5%. If stacking does NOT close it → STOP (the premise that this is finite-volume is wrong; reopen).

**Step 3 — Report** to Anna: the minimum K / volume for a calibrated engine. This sets the SBC mock's
stack size.

---

## Phase A — the analytic-2D projection layer (build, after V1/V2 pass)

### Task A-new1: `angular_bandpowers_2d_limber` (predicted, O(n²), differentiable)

Promote V1's predictor to `inference/covariance.py`. Signature
`angular_bandpowers_2d_limber(n_perp, depth, beta, mach, b, alpha, k_edges, ...)`; returns predicted
rank-G-space angular band-powers (slope→β, amplitude→ℳ via σ_s²). **Tests:** (unit) shape +
differentiable in (β, mach, depth); (slow oracle) matches the **stacked** measured band-powers (V2's
K) to <5% across β∈{2.5,3,3.5} and ℳ∈{4,8,16}. RED→GREEN→commit
`feat(gravoturb_fdf): angular_bandpowers_2d_limber (analytic 2D Limber predictor)`.

### Task A-new2: `rank_gaussianize_2d` + measured rank-G angular band-powers

**Files:** `validation/measure.py`. `rank_gaussianize_2d(map2d)` (Neyrinck+2011 Eq.1 rank→Gaussian,
shot-tolerant, N=0-safe) + `measure_rankG_angular_bandpowers_2d`. Deterministic data-side map
(SBC-valid). **Tests:** unit Gaussian-marginal output; matches the predicted on a lognormal mock.
RED→GREEN→commit `feat(gravoturb_fdf): rank_gaussianize_2d + measured rank-G angular band-powers`.

---

## Phase B — likelihood, stacking, covariance, nuisances

### Task B1: `jackknife_covariance` (data-derived, truth-independent)
`inference/covariance.py`; delete-1 sky-patch jackknife on the (stacked) data vector. PD,
deterministic. RED→GREEN→commit `feat(gravoturb_fdf): jackknife_covariance (SBC-valid)`.

### Task B2: `stacked_bandpower_loglike_2d` (β + ℳ; the headline block)
`inference/likelihood.py`; Gaussian on the stacked rank-G angular band-powers vs
`angular_bandpowers_2d_limber`, jackknife (Hartlap) precision; differentiable in (β, mach, depth).
**Tests:** peaks at truth on noiseless stacked data; finite grad. RED→GREEN→commit
`feat(gravoturb_fdf): stacked_bandpower_loglike_2d (beta+mach headline block)`.

### Task B3: depth/aspect-ratio nuisance + prior
`inference/priors.py`; add the aspect-ratio nuisance (physical prior ~O(1); **confirm form with
Anna**) entering the Limber depth; sample/logpdf/Jacobian. RED→GREEN→commit
`feat(gravoturb_fdf): aspect-ratio depth nuisance + prior`.

### Task B4: POT α retained (depth-gated; measurement + stress-test only)
Confirm `tail_exceedance_loglike` unchanged; add a measured projected-POT helper used ONLY in the
held-out tail stress-test (not the headline fit). RED→GREEN→commit
`feat(gravoturb_fdf): projected POT measurement (alpha depth-gated, stress-test only)`.

---

## Phase C — SBC driver + acceptance gates

### Task C1: `_build_mock_2d_stacked`
`inference/sbc.py` (or `sbc_2d.py`); generate **K clouds** from θ, project (full-LOS), rank-G, measure
+ stack band-powers; jackknife C from the stacked data. Unit test the bundle. Commit.

### Task C2: `build_logdensity_2d`
Compose `stacked_bandpower_loglike_2d` + prior (incl. aspect nuisance) + Jacobian. Finite +
differentiable in z=(mach, β, aspect). Commit.

### Task C3: `sbc_ranks_2d`
Generalise `sbc_ranks` to the stacked-2D mock/logdensity. Smoke test (n_trials=2) no-NaN. Commit.

### Task C4: acceptance gates
- **AC20-2D** (oracle): predicted vs **stacked** finite-field band-powers, <5% across (ℳ, β). (The
  cosmic-variance cure made concrete.)
- **AC15-2D** (forecast; mature `projection_fisher_spike.py`): σ(β), σ(ℳ|aspect), the α-degradation
  curve, 1/√(K·V) scaling.
- **AC16-2D** (recovery): (ℳ, β) cover truth on stacked mock; α reported as a limit (depth-gated).
- **AC18-2D** (SBC): rank-uniformity passes for **β and ℳ** (integer-aware χ²; the C1 helpers). STOP &
  report the rank histogram on failure — do not tune.
Each its own commit; wire into `validation/acceptance.py::main()`.

### Task C5: invariants + docs
Released-core **814** held; full experimental suite green (record count); update design doc + README +
VALIDATION_SUMMARY; honest-scope statement (population-β headline; per-single-cloud ℳ cosmic-variance-
limited; α depth-gated). Update memories. Commit.

---

## Final verification gate (definition of done)

- [ ] V1 (Limber speed ≳10× + <5% vs exact) and V2 (stacking closes gap by K≲20) PASSED & reported.
- [ ] AC20-2D oracle <5% across (ℳ, β) on stacked data; AC18-2D SBC passes for β & ℳ; AC16-2D covers;
      AC15-2D forecast reproduces the spike + 1/√(K·V).
- [ ] Full experimental suite green; **released-core 814 invariant**.
- [ ] α confirmed depth-gated (held-out tail stress-test characterises the residual bias).
- [ ] Design/plan/memories updated; honest scope documented.

## Honest scope / limitations (carry into docs)

- **Headline = population/relative β** for *young, dynamically-unrelaxed* clusters (natal substructure
  decays in a few crossing times — β-vs-age *is* the clock).
- **Per single small cloud, ℳ (amplitude) is cosmic-variance-limited** — fundamental (not stars/grid);
  precision is a *population* statement (1/√(K·V)).
- **α is unmeasurable from 2-D star positions** (projection washes the tail) — depth-gated to the
  future 3-D-star mode; retained as a forward-model fiducial + held-out stress-test.
- **Approximations:** Limber/flat-sky (validated vs exact discrete, V1); shared-θ-within-bin stacking
  (hierarchical = future extension); BM19 phase-random GRF (genuine filaments = the held-out 3-pt null
  test).
