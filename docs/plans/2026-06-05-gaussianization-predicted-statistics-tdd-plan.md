# gravoturb_fdf Differentiable Predicted-Statistics & Inference — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> superpowers:subagent-driven-development) to implement this task-by-task. Each task is
> RED→GREEN TDD, uv-verified, committed. **Phase 0 is BLOCKING** — no statistics code until the
> Gaussianization PDFs are obtained and their formulae verified (per
> [[no-assumptions-verify-against-pdfs]]). HITL-gate at every phase boundary
> ([[hitl-approve-everything]]).

**Goal:** Build a differentiable, physics-direct predicted-statistics layer for
`gravoturb_fdf` (log-density 2-pt `ξ_s` via Gaussianization + smoothing/Limber projection +
counts-in-cells), validated against the realization simulator as ground-truth oracle, then a
Fisher-forecast demo and (Milestone 2) an HMC inference of θ=(ℳ,b,α,β) on mocks.

**Architecture:** Differentiate the *predicted summary statistic*, not the stochastic
simulator (cosmology playbook). `ξ_s(r;θ)` from the Coles & Jones / Szapudi–Pan Hermite
series over the BM19 copula map `T=F⁻¹∘Φ` (β enters analytically through `ρ_g(r;β)`); CIC
moments from `σ²_N = N̄ + N̄²ξ̄(R)`; everything regularized at a data-set smoothing scale `R`
(no hard density cap). Likelihood is incremental hybrid: Gaussian on {ξ_s, CIC moments} for
the Fisher forecast → exact compound-Poisson 1-pt block for HMC. Inference engine: blackjax.

**Tech Stack:** JAX (`jnp`, `jax.grad`, `jax.scipy.special`), Equinox patterns, jaxtyping;
numpy/scipy permitted ONLY in oracle-measurement/validation code; blackjax (NUTS) for HMC;
pytest (`@pytest.mark.experimental`).

---

## Context — why this change

The clean-room rewrite (PR #5) left a **prototype** `q_surrogate` (7-param, mean-only, 64³,
α-frozen linear emulator) that Anna correctly rejected as the inference interface: it makes a
*fitted emulator* differentiable, not the *generator*; it is mean-only and fit on a narrow
box. This phase replaces it with a **versatile differentiable-observables toolkit** — the
foundation on which calibrated inference of molecular-cloud parameters from LSST-era cluster
substructure is built. Design doc:
`docs/plans/2026-06-05-gravoturb-fdf-differentiable-inference-design.md` (§10 Open Questions
now all closed — see "Decisions locked" below).

## Decisions locked (closed §10 open questions, 2026-06-05 brainstorm)

| # | Decision | Consequence for this plan |
|---|----------|---------------------------|
| #1 | **Smoothing-scale `R`** (data-set: cell = 2-pt window = resolution) is the primary fat-tail regularizer. No hard cap. Physical `s_max` is OPTIONAL/deferred (absolute-calibration studies only). | All linear-density moments evaluated at scale `R` ⇒ finite. Oracle uses `R`=grid cell ⇒ free convergence test. `ξ_s` (log-space) needs no cutoff regardless. |
| #5 | **Hybrid likelihood, built incrementally.** M1: Gaussian on {N̄, σ²_N(R), ξ_s(r)} (all the Fisher forecast needs). M2: 2-pt Gaussian `ξ_s` (β) + 1-pt exact compound-Poisson count likelihood (ℳ,b,α). | Predicted-stats module emits the smoothed PDF `p_R` so both moments AND the count distribution fall out — not boxed in. |
| #3 | **β analytic** through `ξ_g(r;β)=FT[k^{-β}]`; full autodiff in (ℳ,b,α,β). Paired finite-difference (CRN, reusing `q_vs_fsub`) is an **AC validation check only** (mirrors existing AC9). NO soft-sort. | `theory/projection.py` provides analytic differentiable `ρ_g(r;β)`. |
| #4 | **Analytic covariance** (Gaussian-field + Poisson shot-noise), held FIXED at fiducial θ in HMC (no log\|C(θ)\| bias term); θ-dependent form used in Fisher. **Mock cross-check** against realization covariance. | `inference/covariance.py` analytic; `validation/` mock-cov cross-check AC. |
| #7 | **blackjax (NUTS)** — hand-written differentiable `logdensity_fn`. numpyro deferred to the hierarchical population model. SBI stays out (optional later cross-check). | `inference/hmc.py` thin blackjax driver. |
| #8 | **Split by concern**: `theory/{gaussianization,projection,cic}.py` + new `inference/{covariance,likelihood,fisher,hmc}.py`. | Honors 300/500-LOC limits from the start. |
| #2 | **CIC primary** stellar observable; angular `w(θ)` deferred (optional later). | No `w(θ)` code in M1/M2. |
| #11 | **3-pt deferred to a later phase** (validation/diagnostic-only per §6b). First 3-pt AC = analytic marginal-induced vs realization-measured null test (must PASS since the simulator is phase-random). | Out of scope here; stub note only. |

## Rules in force (every task)

- **Verify against held PDFs**; never assert a formula from memory ([[no-assumptions-verify-against-pdfs]]).
- **JAX-native core**; numpy/scipy ONLY in oracle-measurement & validation paths.
- **Evidence-before-done** ([[verification-before-completion]]): paste fresh command output.
- **Single-env uv verification:** `env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental -q`
  (experimental needs `PYTHONPATH=src:src/experimental`; confirm conftest sets it — see Task 1.0).
- **HITL at phase boundaries**; surface API/physics decisions before acting.
- **NO git push / NO PR #5 merge** without Anna's explicit go.
- **Commit each task** autonomously (TDD + uv-verified), per the clean-room cadence.

## Reuse (do NOT reinvent — found in code survey)

- `theory/pdf.py`: `bm19_icdf_analytic(u,ℳ,b,α)`, `bm19_icdf(u,…)`, `bm19_mass_cdf(s,…)`,
  `bm19_mean_density(ℳ,b,α)=⟨e^s⟩`, `bm19_volume_pdf`, `bm19_volume_tail_fraction` — all
  differentiable in (ℳ,b,α). **The copula map `T` is `bm19_icdf(Φ(g)) − log⟨e^s⟩`.**
- `theory/bm19.py`: `sigma_s_squared(ℳ,b)=ln(1+b²ℳ²)`, `transition_density(α,σ_s²)`.
- `field/field.py`: `gaussian_random_field(shape,β,key)` (P(k)∝k^{-β}, the oracle field),
  `rank_copula_field`, `mass_conserving_copula_field` (physical mocks).
- `field/pipeline.py`: `build_fdf_field`, `cloud_to_stars` (mock stars for CIC oracle).
- `validation/calibration.py`: `q_vs_fsub` **CRN/paired-realization pattern** (build ONE field,
  reuse across a swept scalar) — reuse verbatim for the β finite-difference AC.
- `validation/acceptance.py`: `_header`/`_row` PASS/FAIL printers, `{"passed": bool, …}` return
  contract — the template for new ACs.
- `tests/experimental/unit/test_grads.py`: `_central_fd` finite-difference helper + the
  `parametrize → assert pytest.approx(rel=1e-4)` grad-test idiom.

---

## Phase 0 — References & grounding (BLOCKING; no statistics code before this passes)

### Task 0.1: Obtain Gaussianization reference PDFs
**Files:** add to `docs/core-papers/` (PDFs) + per-paper notes where the existing paper notes live.
- Coles & Jones (1991) MNRAS 248, 1 — lognormal model for large-scale structure.
- Szapudi & Pan (2004) ApJ 601, 697 — monotone-transform 2-pt: `ξ_Y(r)=Σ c_n²/n! · ξ_g(r)^n`.
- Carron & Szapudi (2013/2014) — Gaussianization information content (secondary).
- FK10 (Federrath+2010, already held) §3.5 — projected (column) PDF / Limber context.

**Step 1:** Search + fetch (WebFetch/Context7 or ADS). **Step 2:** Save PDFs. **Step 3:** Write
per-paper notes capturing the EXACT formula, Hermite convention (probabilists `He_n` vs
physicists `H_n`), and the variance identity `ξ_Y(0)=Σ_{n≥1} c_n²/n!`. **Commit.**

### Task 0.2: Verification memo of the formulae to be coded
**File:** append a "Phase-0 verification" section to the design doc (or a new
`docs/plans/2026-06-05-gaussianization-formula-verification.md`).
Record, with page/equation citations:
1. `c_n = ⟨T(g) He_n(g)⟩`, `g~N(0,1)`, probabilists' Hermite (weight `φ(g)`).
2. `ξ_s(r) = Σ_{n≥1} (c_n²/n!) ρ_g(r)^n`, with `ρ_g(0)=1` (normalized Gaussian correlation).
3. Variance check `Var[T] = Σ_{n≥1} c_n²/n! = σ_s²` in the lognormal limit.
4. Generating-function anchor for tests: `⟨e^{σg} He_n(g)⟩ = σ^n e^{σ²/2}` ⇒
   `Var[e^{σg}] = e^{σ²}(e^{σ²}−1)`.
5. Limber kernel `w(r_⊥)=∫ ξ_3D(√(r_⊥²+ℓ²)) dℓ` and the column-PDF-narrowing statement (FK10 §3.5).
**Gate:** Anna reviews this memo before Phase 1. **Commit.**

---

## Phase 1 — `theory/gaussianization.py`: predicted log-density 2-pt `ξ_s`

### Task 1.0: Confirm experimental test wiring
**Files:** read `tests/experimental/conftest.py` (or nearest). Confirm `sys.path`/`PYTHONPATH`
makes `gravoturb_fdf` importable. If a new `inference/` package needs an `__init__.py`, note it.
No code; just verify the harness. (If conftest doesn't inject the path, the run command must.)

### Task 1.1: The copula map `T(g)` (log-density)
**Files:** Create `src/experimental/gravoturb_fdf/theory/gaussianization.py`;
Test `tests/experimental/unit/test_gaussianization.py`.

**Step 1 — failing test:** `s_of_g` maps a unit Gaussian sample to the BM19 marginal.
```python
def test_s_of_g_recovers_bm19_marginal():
    import jax, jax.numpy as jnp
    from gravoturb_fdf.theory.gaussianization import s_of_g
    from gravoturb_fdf.theory.pdf import bm19_mean_density
    g = jax.random.normal(jax.random.PRNGKey(0), (200_000,))
    s = s_of_g(g, mach=5.0, b=0.4, alpha=2.0)
    # ⟨e^s⟩ == 1 by construction (ρ0 convention)
    assert float(jnp.mean(jnp.exp(s))) == pytest.approx(1.0, rel=2e-2)
    # mean log-density ≈ −σ_s²/2 (BM19 Eq.11), variance ≈ σ_s² in the body-dominated regime
```
**Step 2:** run → FAIL (no module). **Step 3 — implement:**
`s_of_g(g,mach,b,alpha) = bm19_icdf(Phi(g), mach,b,alpha) − log(mean(exp(...)))` where
`Phi(g)=0.5*(1+erf(g/sqrt(2)))`; reuse `bm19_icdf`/`bm19_icdf_analytic`. Differentiable in
(ℳ,b,α). **Step 4:** run → PASS. **Step 5:** commit.

### Task 1.2: Hermite-coefficient quadrature `c_n(ℳ,b,α)`
**Step 1 — failing tests** (these pin the Hermite convention BEFORE any oracle work):
```python
def test_cn_linear_map_only_c1():
    # T(g)=σ g + c  ⇒  c_1=σ, c_{n≥2}=0
def test_cn_exp_map_matches_generating_function():
    # T(g)=exp(σ g)  ⇒  c_n = σ^n e^{σ²/2}; Var=Σ c_n²/n! = e^{σ²}(e^{σ²}−1)
def test_cn_bm19_lognormal_limit_variance():
    # large α (tail negligible): Σ_{n≥1} c_n²/n! ≈ σ_s²(ℳ,b)  within tol
```
**Step 3 — implement:** `hermite_coefficients(map_fn, n_max)` →`c[1..n_max]` via Gauss–Hermite
quadrature. Nodes/weights are CONSTANTS precomputed once with `numpy.polynomial.hermite_e`
(probabilists), frozen as `jnp` arrays at module load (NOT in the hot path → core stays JAX).
`c_n = Σ_i w_i · T(x_i) · He_n(x_i)` with `He_n` from the stable recurrence
`He_{n+1}=g·He_n − n·He_{n-1}`. Differentiable in (ℳ,b,α) through `T`.
Provide `bm19_hermite_coefficients(mach,b,alpha,n_max)` wrapping `s_of_g`. **Commit.**

### Task 1.3: The Gaussianization series `ξ_s(ρ_g; c_n)`
**Step 1 — failing tests:**
```python
def test_xi_s_at_rho1_equals_variance():
    # ξ_s(ρ_g=1) == Σ_{n≥1} c_n²/n!  (==Var[s])
def test_xi_s_monotone_and_zero_at_rho0():
    # ξ_s(0)=0; monotone increasing in ρ_g∈[0,1]
def test_xi_s_differentiable_in_params():
    # jax.grad of ξ_s wrt (mach,b,alpha) is finite (no NaN)
```
**Step 3 — implement:** `gaussianized_xi(rho_g, c)= Σ_{n≥1} (c_n²/n!) rho_g**n` (vectorized
over an `r`/`ρ_g` array; `n` up to `n_max`). **Commit.**

### Task 1.4: Oracle measurement utility (numpy/scipy OK — validation path)
**Files:** Create `src/experimental/gravoturb_fdf/validation/measure.py`.
Functions (operate on realization arrays; non-diff):
- `gaussian_correlation_measured(g, r_bins)` → ρ_g(r) (normalized; FFT-based estimator).
- `field_2pt_measured(s, r_bins)` → ξ_s(r) from a realization log-density field.
- `smooth_copula_field(g, mach, b, alpha)` → the **theory-consistent** smooth map
  `s = bm19_icdf(Φ(ĝ)) − shift` with `ĝ = g/std(g)` (exact-unit-variance normalize). This is
  the oracle companion that matches the Gaussianization assumptions (vs the rank/mass-conserving
  copula). Tests: marginal matches BM19; at large N it agrees with `rank_copula_field`.
**Commit.**

### Task 1.5: **Oracle AC — predicted `ξ_s` vs realization** (AC11)
**Files:** add `ac11_xi_s_vs_oracle(...)` to `validation/acceptance.py`; wrap in
`tests/experimental/validation/test_acceptance.py`.
**Logic:** ensemble of realizations (64³ dev / 128³ confirm) via `gaussian_random_field` →
`smooth_copula_field`; measure ensemble ⟨ξ_s(r)⟩ and ⟨ρ_g(r)⟩; predict
`ξ_s_pred = gaussianized_xi(ρ_g_measured, bm19_hermite_coefficients(...))`; assert agreement
within tol (e.g. ≤5% on r-bins where ρ_g≳0.05), and report Gaussianization **convergence**
(ξ_s_pred stable as `n_max` grows — closes Open Q #6). PASS/FAIL via `_row`. **Commit.**
**Gate: Phase 1 boundary — report to Anna** (predicted-vs-oracle agreement + n_max needed).

> **PHASE 1 DONE (2026-06-05, commits b2bc682→58cb526):** AC11 max rel 0.11% / median 0.09%
> (64³×8); Gaussianization converges by n_max≈8 (Open Q #6 closed). 169 experimental + 815
> released-core tests green. AC11 validated against the *smooth-copula* oracle — the physical
> rank/mass-conserving copula equivalence is Task 2.0 below (Anna's call, 2026-06-05).

---

## Phase 2 — `theory/projection.py`: analytic `ρ_g(r;β)`, smoothing@R, Limber

### Task 2.0: Rank-copula equivalence (close the map-mismatch item) — AC11b
**Files:** `ac11b_rank_copula_equivalence(...)` in `validation/acceptance.py`; test wrapper.
**Why:** AC11 used the theory-consistent `smooth_copula_field` (map `T=F⁻¹∘Φ`). The *mocks*
use the empirical-rank `rank_copula_field` / `mass_conserving_copula_field` (`field/field.py`).
At large N the empirical CDF → Φ, so the rank-copula field → the smooth map; confirm the
predicted `ξ_s` also describes the **physical simulator** fields (not just the clean oracle).
**Step 1 — failing tests:** measure ⟨ξ_s(r)⟩ from `rank_copula_field` (and
`mass_conserving_copula_field`) realizations of the same GRFs; compare to the analytic
`gaussianized_xi(ρ_g_measured, c)` prediction. Assert (a) `rank_copula` agrees within tol on
ρ_g≳0.05 bins, and (b) the discrepancy **shrinks from small N to large N** (rank→Φ convergence).
Characterize the `mass_conserving` offset (mass-averaged marginal ⇒ may differ more — report,
don't force a tight tol). **Step 3:** thresholds set from a measured exploration run (physics-
first, not arbitrary). **Commit.**

> **TASK 2.0 DONE (2026-06-05, AC11b):** both `rank_copula` AND `mass_conserving` measured
> `ξ_s` match the analytic prediction to **max ~0.4% / median ~0.2%** on ρ_g>0.05 bins, across
> N=32/64/96. **Finding (corrected the a-priori):** the discrepancy is **noise-limited, not a
> rank→Φ bias** — already sub-% at N=32 — so the planned "shrinks with N" criterion was dropped;
> the honest criterion is "both copulas agree within a few %". Mass-averaging perturbs the
> marginal (AC6 1-pt) but barely the log-density 2-pt. Map-mismatch closed.

### Task 2.1: Analytic Gaussian correlation `ρ_g(r;β)` (differentiable in β)
**Files:** Create `theory/projection.py`; tests in `test_projection.py`.
**Step 1 — failing tests:**
```python
def test_rho_g_matches_measured_from_grf():
    # analytic ρ_g(r;β,shape) vs gaussian_correlation_measured over an ensemble: ≤ few %
def test_rho_g_normalized_and_beta_grad():
    # ρ_g(0)=1; jax.grad wrt β finite & nonzero (steeper β ⇒ more large-scale power)
```
**Step 3 — implement:** discrete mode sum matching the simulator's grid:
`ρ_g(r) = Σ_k P(k)·sinc/j0(k·r) / Σ_k P(k)`, `P(k)=k^{-β}` (DC=0), to be apples-to-apples with
`gaussian_random_field`. Differentiable in β. (Keep a continuum `j0` integral variant as a
cross-check.) **Commit.**

### Task 2.2: Smoothing at scale R (window)
**Step 1 — failing tests:** smoothed variance `σ_g²(R)` decreases with R; cell-averaged
`ξ̄(R) = ⟨ξ over cell pairs⟩` matches realization smoothed to scale R.
**Step 3 — implement:** `window(kR)` (top-hat default; Gaussian option), `smoothed_variance`,
`cell_averaged_xi(rho_g_or_spectrum, R)`. This is the SINGLE `R` shared by 2-pt + CIC. **Commit.**

### Task 2.3: Limber projection (3D→2D) — verified vs FK10 §3.5
**Step 1 — failing tests:**
```python
def test_limber_projects_known_correlation():
    # analytic check on a closed-form ξ_3D (e.g. Gaussian bump) → ∫dℓ matches quadrature
def test_projected_2pt_vs_oracle():
    # project a 3D realization to 2D, measure 2-pt; compare to Limber-predicted: agreement
```
**Step 3 — implement:** `limber_project(xi_3d_fn, r_perp, depth)` =
`∫ ξ_3D(√(r_⊥²+ℓ²)) dℓ` over cluster depth (fixed-node quadrature, differentiable). Document
the depth/distance nuisance. **Note the log-vs-linear projection subtlety** (column density is a
linear-ρ integral; `ξ_s` is log) — the oracle test decides which projected quantity sources the
2D CIC; record the resolution in the design doc. **Commit.**
**Gate: Phase 2 boundary — report to Anna** (β-analytic vs measured agreement; projection choice).

---

## Phase 3 — `theory/cic.py`: counts-in-cells (M1 moments; M2 count distribution)

### Task 3.1: CIC moments (Milestone 1)
**Files:** Create `theory/cic.py`; tests `test_cic.py`.
**Step 1 — failing tests:**
```python
def test_cic_variance_formula_vs_oracle():
    # σ²_N(R) = N̄ + N̄² ξ̄(R): predict, then bin mock stars (cloud_to_stars) into 2D cells,
    # measure CIC variance over an ensemble; agreement within tol
def test_cic_moments_differentiable():
    # jax.grad of σ²_N wrt (mach,b,alpha,beta) finite
```
**Step 3 — implement:** `cic_mean(n_bar, cell)`, `cic_variance(n_bar, xi_bar)` using §4's
`σ²_N = N̄ + N̄² ξ̄(R)`; `ξ̄(R)` from Task 2.2. **Commit.**

### Task 3.2: Smoothed density PDF `p_R(ρ)` (Milestone 2 prep)
**Step 1 — failing test:** `p_R` integrates to 1; variance equals `smoothed_variance(R)`;
→ point PDF as R→0. **Step 3 — implement:** reduced-variance BM19 (lognormal body with
`σ_s²(R)` + matched tail) OR moment-matched form; document the approximation. **Commit.**

### Task 3.3: Exact compound-Poisson count distribution `P(N)` (Milestone 2)
**Step 1 — failing tests:**
```python
def test_pN_normalizes_and_mean_matches():
    # Σ_N P(N) = 1; Σ_N N P(N) = N̄
def test_pN_overdispersed_vs_oracle_histogram():
    # P(N)=∫ Poisson(N|N̄ ρ_R/⟨ρ_R⟩) p_R(ρ) dρ vs mock CIC histogram (KL/χ² within tol)
```
**Step 3 — implement:** 1-D fixed-node quadrature over `p_R`, differentiable in θ. **Commit.**
**Gate: Phase 3 boundary — report to Anna** (CIC moments + count-dist vs oracle).

---

## Phase 4 — gradient validation (AC14)

### Task 4.1: Autodiff vs finite-difference AC
**Files:** `ac14_grad_validation(...)` in `validation/acceptance.py`; test wrapper.
- (ℳ,b,α): `jax.grad` of `ξ_s(r*)` and `σ²_N(R)` vs `_central_fd` — rel ≤1e-4 (mirrors AC9).
- **β:** analytic `∂ξ_s/∂β` (autodiff) vs **paired finite-difference** using the
  `q_vs_fsub` CRN pattern (one field, β±ε) — this is the validation of the analytic β path
  (Decision #3). Report agreement. **Commit.**
**Gate: Phase 4 boundary.**

---

## Phase 5 — `inference/` Milestone 1: covariance + Gaussian likelihood + **Fisher forecast**

### Task 5.1: `inference/covariance.py` — analytic C(θ) + mock cross-check
**Files:** Create `src/experimental/gravoturb_fdf/inference/__init__.py`,
`inference/covariance.py`; tests `test_covariance.py`.
- Analytic Gaussian-field covariance for `ξ_s(r)` bins + Poisson shot-noise for CIC moments.
- `mock_covariance(...)` (numpy, validation): cov over realization mocks (Hartlap factor).
- **Test:** analytic C within ~tol of mock C on the diagonal + dominant off-diagonals. **Commit.**

### Task 5.2: `inference/likelihood.py` — Gaussian data vector (M1)
**Step 1 — failing tests:** `data_vector(θ)` concatenates {ξ_s(r_bins), N̄, σ²_N(R)};
`gaussian_loglike(data, θ, Cinv)` is a finite scalar, differentiable in θ, maximal at θ_true on
noiseless data. **Step 3 — implement.** **Commit.**

### Task 5.3: `inference/fisher.py` — **first science deliverable** (AC15)
**Step 1 — failing tests:** `fisher_matrix(θ_fid, Cinv) = Jᵀ C⁻¹ J`, `J=∂(data_vector)/∂θ`
via `jax.jacobian`; symmetric, positive-definite; marginal `σ(β)`, `σ(ℳ)` finite and shrink as
survey area / N_cells grows. **Step 3 — implement.** Add `ac15_fisher_forecast` printing the
forecast constraint table (the headline Fisher number: "how well can Rubin constrain β→ℳ").
**Commit.**
**Gate: Phase 5 boundary — deliver Fisher forecast to Anna.**

---

## Phase 6 — `inference/` Milestone 2: compound-Poisson likelihood + HMC recovery

### Task 6.1: Upgrade likelihood — hybrid 2-pt Gaussian + 1-pt compound-Poisson
**Files:** extend `inference/likelihood.py`.
- 2-pt block: Gaussian on `ξ_s(r)`. 1-pt block: `Σ_cells log P(N_cell)` from Task 3.3.
- **Test:** differentiable scalar; recovers θ_true MAP on noiseless mock better than M1 on the
  (ℳ,b,α) shape directions. **Commit.**

### Task 6.2: `inference/hmc.py` — blackjax NUTS driver
**Step 1 — failing tests:** `logdensity_fn(θ)=loglike+logprior` finite & differentiable; a short
NUTS chain on a toy Gaussian target recovers the mean. **Step 3 — implement** thin blackjax
wrapper (window adaptation + NUTS); priors with bounded support (e.g. logit transforms for
b∈[1/3,1], α>1). **Commit.**

### Task 6.3: **HMC recovery on mocks** (AC16)
**Files:** `ac16_hmc_recovery(...)`.
- Inject θ_true → generate mock (realization mocks + analytic covariance) → run NUTS →
  assert posterior 68%/95% credible intervals cover θ_true; report bias/σ per parameter; R̂≈1.
- Use a small N_chain/N_warmup for the test wrapper; full run in `main()`. **Commit.**
**Gate: Phase 6 boundary — deliver recovery results to Anna.**

---

## Phase 7 — retire `q_surrogate` (GATED on Anna's go)

### Task 7.1: Remove the prototype emulator
**Files:** delete `src/experimental/gravoturb_fdf/surrogate.py` (+`PERSISTED_COEFFS`),
`tests/experimental/unit/test_surrogate.py`; drop `fit_q_surrogate`/`surrogate_features` from
`validation/calibration.py` if unused elsewhere (grep first); update package `__init__` exports.
- Update `VALIDATION_SUMMARY.md`, `README.md`, root `CLAUDE.md`/`progenax/CLAUDE.md` test counts
  and AC list (add AC11–AC16). **Per [[hitl-approve-everything]] this deletion is a separate
  explicit gate** (the `q_surrogate` is the design's named retirement target, but confirm before
  removing). **Commit.**

---

## New acceptance criteria (summary)

| AC | Checks | Oracle |
|----|--------|--------|
| AC11 | predicted `ξ_s(r)` vs realization-measured (+ n_max convergence, Open Q #6) | `smooth_copula_field` ensemble |
| AC12 | `ρ_g(r;β)` analytic vs measured; Limber projected-2pt vs measured | GRF + projection |
| AC13 | CIC `σ²_N(R)` (and M2 `P(N)`) vs mock CIC | `cloud_to_stars` binned |
| AC14 | autodiff vs FD for (ℳ,b,α) and **β (paired CRN)** | finite-difference |
| AC15 | Fisher forecast: PD matrix, `σ(β)` shrinks with survey size | — (sanity) |
| AC16 | HMC posterior covers injected θ on mocks; R̂≈1 | injected-θ mocks |
| (later) | 3-pt null test: analytic marginal-induced == realization-measured (must PASS, phase-random sim) | deferred Phase 8 |

## Verification (run after every task; full gate at phase boundaries)

```bash
cd /Users/anna/projects/jaxstro-dev/progenax
# Experimental subsystem (needs src/experimental on path):
PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental -q
# Released-core invariant MUST stay green (815 tests) — this phase must not touch released core:
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q
# Acceptance scripts (print PASS lines):
PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
  python -m gravoturb_fdf.validation.acceptance
```
**Definition of done (per CLAUDE.md):** tests pass + acceptance script prints PASS + a short
completion note. Released-core (815) unchanged at every gate.

## Risks / watch-list

- **Map mismatch:** theory uses smooth `T=F⁻¹∘Φ`; simulator uses rank/mass-conserving copula.
  Mitigated by `smooth_copula_field` oracle companion (Task 1.4) + a large-N equivalence test.
- **Gaussianization convergence with the power-law tail:** monitor `n_max`; the lognormal-limit
  and `exp(σg)` analytic tests pin the machinery; AC11 reports convergence. If the tail needs
  large `n_max`, document and consider a tail-resummation note (do NOT silently truncate).
- **Limber log-vs-linear projection:** record which projected quantity sources the 2D CIC; let
  the oracle decide; verify against FK10 §3.5.
- **Covariance in HMC:** keep C FIXED at fiducial (no `log|C(θ)|` term) to avoid bias; the
  θ-dependent form is for Fisher only.
- **β information is grid/Nyquist-sensitive:** `ρ_g(r;β)` must be computed on the *same* discrete
  spectrum as the simulator for clean oracle agreement.
- **Scope discipline:** no soft-sort, no `w(θ)`, no 3-pt, no SBI in this plan (all deferred).
```
