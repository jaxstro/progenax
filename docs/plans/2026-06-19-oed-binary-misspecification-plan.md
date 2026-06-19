# Binary-Misspecification Robustness OED — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> superpowers:subagent-driven-development) to implement this plan task-by-task.

**Goal:** Demonstrate, in a referee-proof way, that an OED design optimized for the dynamical
mass M under the binary-free model is *biased* when the cluster has binaries — and that a
binary-aware (marginalize-f_bin) design removes the bias — using an EFF / RV-only / single-epoch
forward model and the Moe & Di Stefano σ_los-inflation kernel.

**Architecture:** Scripts-only (no `src/progenax/` change), riding the Stage-1 additive-Fisher
backbone (`scripts/_demo_oed.py`): `F = Σ_b n_eff,b · M_b`, ONE reverse-mode jacrev, ln-θ metric
(ADR-0011), c/D/A criteria, multi-start Adam, `jax.lax.map` calibration. The new forward model is
EFF-OM `project_dispersion` (σ_los channel only) plus an additive binary term
σ²_obs = σ²_cluster(M, r_a, γ, a) + f_bin·V_bin + ε², with V_bin = Var(K_orb) a build-once
population scalar from `scripts/_demo_binaries.py`.

**Tech Stack:** JAX (jax.numpy, jacrev, jit, lax.map), Equinox profiles/DFs, optax (Adam),
pytest. Reuse: `progenax.project_dispersion`, `progenax.EFFProfile`, `progenax.EFFVelocityDF`,
`progenax.kinematics.eddington`, `progenax.binaries.kepler.KeplerElements`,
`progenax.imf.binary.MoeJointOrbit`, `progenax.imf.differentiable_binary.DifferentiableBinaryFraction`,
`progenax.stellar` (ZAMS).

**Design doc:** `docs/plans/2026-06-19-oed-binary-misspecification-design.md` (read first).
**Branch:** `feat/oed-binary-misspecification`.

---

## How to run things (shared conventions)

- **Import pattern in tests** (matches `tests/unit/test_demo_oed.py`):
  ```python
  import sys, pathlib
  import progenax  # noqa: F401  -- enables float64
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
  import _demo_oed_binary as oedb  # noqa: E402
  ```
- **FAST gate:**
  `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow" -n auto`
- **Single test:** `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_demo_oed_binary.py::test_name -v`
- **@slow MC gate** is env-gated (`PROGENAX_RUN_OED_BINARY=1`) and `@pytest.mark.slow` → OUT of CI.
- **No `src/` change** → released-core coverage/staleness gate is untouched; only the
  test-backbone dashboard re-stamps for the new test module (Phase 4).
- **Commit after every green step.** Trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **HITL:** stop and report at each phase boundary; no merge/push without Anna's separate word.
- **Perf rule (`enforce-jax-performance`):** jit hot paths; build-once K_orb pool + EFF sampler;
  MC via `jax.lax.map` (never vmap over the fit tape); smoke-test peak RSS before any long run.

**Fiducial constants** live in `_demo_oed_binary.py` module scope and are **PINNED in Phase 0**
(Task 0.2). Until pinned, treat them as `TODO(phase0)`: EFF (γ_fid, a_fid, r_t_fid),
M_fid (→ σ_cluster ≈ 8–12 km/s YMC), distance d, RV error ε_RV, primary-mass scale for Moe,
f_bin_truth, the radial bins `R_BINS`.

---

## Phase 0 — Gate & provenance (de-risk before building)

### Task 0.1: Verify Moe & Di Stefano (2017) constant provenance

**Files:** Read `scripts/_demo_binaries.py`, `src/progenax/imf/binary/moe_di_stefano.py`,
`src/progenax/binaries/eccentricity.py`; the held PDF + any `docs/website/99-bibliography/per-paper/`
note for Moe & Di Stefano 2017.

**Steps:**
1. List every numeric constant/break/slope used by `MoeJointOrbit`/`MoePeriod`/
   `MoeDiStefano2017Full`/`MoeEccentricity` and by `_demo_binaries.py`'s blend kernel.
2. Cross-check each against the PDF (Tables 13/etc.) per `research-workflow:provenance-of-constants`
   and `no-assumptions-verify-against-pdfs`. Record file:line ↔ PDF table/eq.
3. If a per-paper note is missing or thin, capture the needed expansion home
   (`brain "source-note update: moe_di_stefano_2017 — ..."`). Do **not** edit `~/brain`.
4. **Output:** a short provenance table appended to the design doc (or a `.claude-work/` note).
   **Gate:** zero unsourced constants on the binary grad path. No code change.

**Commit:** `docs(oed-binary): Moe&DiStefano2017 constant-provenance audit for the binary kernel`

### Task 0.2: PIN σ_bin numerics + confirm H1 will bite

**Files:** Create `scripts/_demo_oed_binary.py` (module skeleton + fiducial constants);
extend `scripts/_demo_binaries.py` with `population_blend_variance(...)`.
Test: `tests/unit/test_demo_oed_binary.py`.

**Step 1 — failing test (V_bin sane + ratio bites):**
```python
def test_vbin_and_sigma_ratio_bites():
    # V_bin = Var(K_orb) for a Moe massive-primary population (build-once pool)
    V_bin = oedb.V_BIN                      # (km/s)^2 scalar at the fiducial primary scale
    assert jnp.isfinite(V_bin) and V_bin > 0.0
    sig_bin = jnp.sqrt(V_bin)              # km/s
    sig_cluster = oedb.sigma_cluster_ref() # km/s, mass-weighted / central, YMC fiducial
    ratio = sig_bin / sig_cluster
    # H1 needs binaries to rival cluster heat at the YMC operating point
    assert ratio > 0.5, f"sigma_bin/sigma_cluster={ratio:.2f} too small for H1 to bite"
```
**Step 2:** Run → FAIL (module/attrs missing).
**Step 3 — implement:** in `_demo_binaries.py`, `population_blend_variance(key, n_pool,
primary_mass_dist, Z)` reusing `sample_blend_velocities`/`build_korb_kernel` (build-once);
in `_demo_oed_binary.py`, set `V_BIN` (call it once at import, fixed key) and
`sigma_cluster_ref()` (from `project_dispersion(EFFProfile(...), r_a, R, M, G).sigma_los`,
converted to km/s).
**Step 4:** Run → PASS. If `ratio` is too small even for massive primaries, **lower σ_cluster**
(drop M_fid / pick a lower-mass YMC point) and re-pin — the sweep (Phase 4) will span the ratio.
**Step 5:** Commit `feat(oed-binary): pin V_bin + sigma_cluster fiducials; confirm H1 bites`.

### Task 0.3: Pre-register the discriminating experiment

**Files:** append a "Pre-registration" block to the design doc (or `.claude-work/`).

**Steps:** Run `research-workflow:discriminating-experiment-design` to lock the observable
(σ_los moments per radial bin), the signature under H1 (binary-free fit → M̂ high) vs the
binary-aware fit (M̂ unbiased), the smallest run (n_draws for the YMC point), and the
accept/reject rules **exactly** as in the design doc (H1 |bias|>2σ_forecast; H2 ≥1.3×; H3
non-monotone allocation). Commit `docs(oed-binary): pre-register H1/H2/H3 + accept-reject`.

**PHASE 0 CHECKPOINT (HITL):** report the provenance verdict + the pinned σ_bin/σ_cluster ratio.
If H1 cannot bite at any plausible YMC point → re-scope before Phase 1.

---

## Phase 1 — Bias demonstration (H1): the minimal-falsifiable slice

> If H1 fails here, the item is DESCOPED (design doc). Do not proceed to Phase 2 without
> Anna's go after the Phase-1 checkpoint.

### Task 1.1: EFF-OM RV-only cluster forward model

**Files:** `scripts/_demo_oed_binary.py`; Test: `tests/unit/test_demo_oed_binary.py`.

**Step 1 — failing test:**
```python
def test_cluster_sigma_los_matches_project_dispersion():
    th = oedb.theta_truth_clusteronly()        # (M, r_a, gamma, a)
    R = oedb.R_BINS
    sig = oedb.cluster_sigma_los(th, R, STELLAR.G)   # (K,) km/s, RV channel only
    # oracle: project_dispersion on the same EFF-OM model
    prof = oedb.eff_profile(th)
    sig_ref = oedb.kms(project_dispersion(prof, th_ra(th), R, th_M(th), STELLAR.G).sigma_los)
    assert jnp.allclose(sig, sig_ref, rtol=1e-10)
    assert jnp.all(sig > 0)
```
**Step 2:** Run → FAIL.
**Step 3 — implement** `eff_profile(theta)` (build `EFFProfile(a=..., gamma=..., r_t=...)` from
(a, γ)), `cluster_sigma_los(theta, R, G)` = `kms(project_dispersion(...).sigma_los)`, and the
θ accessors. Reuse `_demo_oed`'s unit converters.
**Step 4:** Run → PASS. **Step 5:** Commit `feat(oed-binary): EFF-OM RV-only cluster sigma_los`.

### Task 1.2: Binary-inflated observable + ONE jacrev (ln-θ)

**Step 1 — failing test:**
```python
def test_obs_variance_adds_binary_pedestal_and_jacrev_lntheta():
    th = oedb.theta_truth()                     # (M, r_a, gamma, a, f_bin)
    R = oedb.R_BINS
    s2 = oedb.predict_sigma_obs2(th, R, STELLAR.G)         # (K,) (km/s)^2
    s2_cluster = oedb.cluster_sigma_los(th[:-1], R, STELLAR.G)**2
    assert jnp.allclose(s2, s2_cluster + th[-1]*oedb.V_BIN, rtol=1e-10)
    # ONE reverse-mode jacrev, dimensionless ln-theta (ADR-0011)
    J = oedb.jacobian_lntheta(th, R, STELLAR.G)            # (K, P=5) of d sigma_los / d ln theta
    assert J.shape == (R.shape[0], 5)
    # f_bin sensitivity concentrates in the outskirts (low sigma_los)
    fcol = J[:, oedb.IDX_FBIN]
    assert jnp.abs(fcol[-1]) > jnp.abs(fcol[0])
```
**Step 2:** Run → FAIL.
**Step 3 — implement** `predict_sigma_obs2` (cluster² + f_bin·V_bin) returning σ_los (sqrt),
and `jacobian_lntheta` = `jax.jacrev(lambda lnth: sqrt(predict_sigma_obs2(exp(lnth)...)))`
scaled to ln-θ (`J*theta` idiom). Reverse-mode by policy.
**Step 4:** Run → PASS. **Step 5:** Commit `feat(oed-binary): binary-inflated observable + ln-theta jacrev`.

### Task 1.3: Additive Fisher + binary-free c-optimal-for-M design

**Step 1 — failing test:**
```python
def test_binary_free_fisher_spd_and_design_normalized():
    # binary-free theta = (M, r_a, gamma, a); priors: gamma,a tight (photometric); r_a weak; M free
    F = oedb.fisher_binary_free(oedb.uniform_design(), oedb.N_TOTAL)
    assert jnp.allclose(F, F.T) and jnp.all(jnp.linalg.eigvalsh(F) > 0)
    res = oedb.optimize_design_M(oedb.fisher_binary_free, oedb.N_TOTAL, key=jax.random.PRNGKey(0))
    assert jnp.isclose(jnp.sum(res.n_eff), oedb.N_TOTAL, rtol=1e-4)
    assert res.sigma_M_over_M > 0
```
**Step 2:** Run → FAIL.
**Step 3 — implement** `blocks_from_eps`/`fisher`-style assembly reusing `_demo_oed` helpers,
with `PRIOR_DIAG` = tight on (γ, a), weak on r_a, zero on M; `optimize_design_M` =
c-optimal on the M index via the imported multi-start Adam. **Cache the cluster jacrev once.**
**Step 4:** Run → PASS. **Step 5:** Commit `feat(oed-binary): additive Fisher + binary-free c-optimal-for-M design`.

### Task 1.4: Cross-model evaluation harness (smoke)

**Files:** `_demo_oed_binary.py`; Test: `tests/unit/test_demo_oed_binary.py`.

**Step 1 — failing test (few draws, fast):**
```python
@pytest.mark.parametrize("n_draws", [4])
def test_cross_model_pipeline_runs(n_draws):
    # generate WITH Moe binaries on a design; fit WITHOUT binaries; return bias(M_hat)/M
    out = oedb.cross_model_bias(oedb.binary_free_design(), n_draws=n_draws,
                                key=jax.random.PRNGKey(0))
    assert jnp.isfinite(out.bias_M_frac) and jnp.isfinite(out.std_M_frac)
```
**Step 2:** Run → FAIL.
**Step 3 — implement** `cross_model_bias`: build-once EFF Eddington sampler (`EFFVelocityDF`) +
K_orb pool; per draw via **`jax.lax.map`** → sample EFF particles, inject Moe binaries for the
f_bin fraction (flux-weighted blend, reuse `_demo_binaries.sample_blend_velocities`), project to
σ_los, bin by R, subsample design counts, add ε_RV → σ̂ + SE → ln-θ GN MAP fit of the
**binary-free** θ (reuse `_demo_oed_depth._fit_theta_gn` / `_demo_inference.fisher_information_gn`)
→ collect M̂. Return mean fractional bias + std. **jit the per-draw fn; build-once everything truth-fixed.**
**Step 4:** Run → PASS. **Step 5 (perf check):** smoke-test peak RSS on n_draws=4; record wall-clock.
**Commit:** `feat(oed-binary): cross-model bias harness (lax.map, build-once)`.

### Task 1.5: H1 gate — @slow calibration MC

**Step 1 — failing test (env-gated, @slow):**
```python
@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PROGENAX_RUN_OED_BINARY"), reason="env-gated MC")
def test_H1_naive_design_is_biased_beyond_forecast():
    res = oedb.run_H1(n_draws=oedb.N_DRAWS_H1, key=jax.random.PRNGKey(0))  # YMC operating point
    # naive (binary-free) design + binary-free fit: bias exceeds its own forecast sigma
    assert abs(res.bias_M_frac) > 2.0 * res.forecast_sigma_M_frac
```
**Step 2:** Run (with env var) → expected behavior validated; if it does NOT bite, **do not weaken
the test** — re-pin σ_bin/σ_cluster (Task 0.2) or report H1 falsified (descope) at the checkpoint.
**Step 3 — implement** `run_H1` (full draws at the YMC point, returns bias + forecast σ from the
binary-free Fisher). **Step 4:** Run → PASS. **Step 5:** Commit `test(oed-binary): H1 bias-beyond-forecast gate (env-gated @slow)`.

### Task 1.6: Gated CLI + bias figure

**Files:** Create `scripts/demo_oed_binary.py` (gated CLI, exit 0);
figures → `docs/website/60-science-demos/optimal-design/figures/`.

**Steps:** `--quick` smoke (exit 0, honors `--outdir` for any run-record — recall the Stage-3 CLI
fixed-path bug); a bias-bar figure (naive M̂ vs truth vs forecast σ). Inspect the figure for
correctness. Commit `feat(oed-binary): gated CLI + M-bias figure (H1)`.

**PHASE 1 CHECKPOINT (HITL):** report H1 verdict + the bias number + wall-clock/RSS.
**Stop for Anna's go before Phase 2.** Phases 2–4 below are task-level and get expanded to
bite-sized steps here, informed by the Phase-1 numbers.

---

## Phase 2 — Marginalize fix (H2/H3)  *(bite-sized; expanded 2026-06-19 at the Phase-1 checkpoint)*

Splits into a **deterministic** group (T2.1+T2.2+H2+H3 — Fisher/design only, fast tests) and an
**MC** group (T2.5 — the @slow calibration that the binary-aware *fit* removes the bias). Reuses
the existing `jacobian_lntheta` (K×5, already includes the ∂σ/∂ln f_bin column, verified Task 1.2).

### Task 2.1: Marginalized (5-param) Fisher + f_bin prior + AD-vs-FD f_bin gate
**Files:** `scripts/_demo_oed_binary.py`; Test: `tests/unit/test_demo_oed_binary.py`.
- Build-once at the FULL truth `theta_truth()` (5-param): `_J_MARG = jacobian_lntheta(...)` (K×5),
  `_SIG_MARG = predict_sigma_obs(theta_truth(), R_BINS, G)` (the **binary-inflated** σ_obs — the
  Fisher denominator `(σ²+ε²)` uses the observed dispersion). Never re-jacrev in the loop.
- `PRIOR_DIAG_MARG` (len 5): M=0 (target); r_a=1/0.5² (weak); γ,a=1/0.1² (tight photometric);
  **f_bin = weak** (e.g. 1/0.5² — a measured nuisance, data-driven via radial leverage; document).
- `fisher_marginalized(z, N_total)` → (5,5): same additive single-RV-channel form with the 5-param
  blocks + diag(PRIOR_DIAG_MARG).
- **Step 1 (failing test):** `fisher_marginalized(uniform, N)` symmetric + SPD; **AD-vs-FD on the
  f_bin column** of `jacobian_lntheta` (`rel < 1e-3`; analytic `f_bin·V_bin/(2σ_obs)` already
  verified 4e-16 in Task 1.2 — formalize as an explicit FD grad-check, `gradient-validation`).
- **Step 2–5:** implement → pass → commit `feat(oed-binary): marginalized 5-param Fisher + f_bin block`.

### Task 2.2: Binary-aware c-optimal-for-M design + H2 (OED payoff) + H3 (allocation)
- `optimize_design_M_marg(N_total, key)` → c-optimal on M (index 0) over `fisher_marginalized`.
  Report σ(M)/M **marginalized** (expected LARGER than the binary-free 4.5% — honest binary-aware
  precision) + the allocation.
- **H2 (deterministic gate):** `precision_gain = sigmaM(binary_free_design under fisher_marginalized)
  / sigmaM(binary_aware_design under fisher_marginalized)`. **Test asserts ≥ 1.3×** (pre-registered;
  if < 1.3× do NOT weaken — report as a null finding, the binary-free design was accidentally near-
  optimal for the marginalized problem).
- **H3 (deterministic gate):** the binary-aware allocation is NOT a monotone rescaling of the
  binary-free one (per-bin weight rank-order change, or KL above a documented threshold).
- **Tests:** marginalized σ(M) > binary-free σ(M) (marginalizing costs info — sanity); H2 gain;
  H3 non-monotone. Commit `feat(oed-binary): binary-aware design + H2 precision-gain + H3 allocation`.

### Task 2.5: The fix removes the bias — @slow binary-aware calibration
- `_fit_theta_marg_gn` (5-param LM-GN MAP in ln-θ; predicts `sqrt(predict_sigma_obs(θ)²)` i.e.
  `sqrt(cluster² + f_bin·V_bin + ε²)` with **f_bin free**; PRIOR_DIAG_MARG; honest realized-σ̂ SE;
  drop empty bins — same machinery as the binary-free fit, +f_bin).
- `run_fix(n_draws, key)`: cross-model MC on the **binary-aware design**, generate-with-binaries,
  **fit-with-binary-aware** → M̂. Reuse the Route-1 mock + `jax.lax.map` + build-once.
- **Step 1 (failing test, @slow + env-gated):**
```python
@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PROGENAX_RUN_OED_BINARY"), reason="env-gated MC")
def test_fix_binary_aware_fit_is_unbiased():
    res = oedb.run_fix(n_draws=oedb.N_DRAWS_H1, key=jax.random.PRNGKey(0))
    assert abs(res.bias_M_frac) < 2.0 * res.sigma_M_marg   # the fix removes the +185% bias
```
- Report: M̂ bias (should be ≈0, vs Phase-1 +185%), f_bin_hat (≈truth 0.5?), σ(M)_marg, convergence.
  Commit `test(oed-binary): binary-aware fit removes the M bias (@slow)`.
- **Step (perf):** build-once + lax.map; smoke RSS before the full run (EFF-free, should be ~2 GB).

## Phase 3 — Min-max / maximin comparison  *(expand at checkpoint)*

- **T3.1** maximin criterion: minimize worst-case marginalized (F⁻¹)_MM over f_bin ∈ [0, f_max],
  re-evaluating cached blocks on an f_bin grid (no re-jacrev); unit test it is SPD + finite.
- **T3.2** compare maximin vs marginalize designs (allocation + σ(M) + worst-case σ(M)); figure.

## Phase 4 — Sweep + docs + close-out  *(expand at checkpoint)*

- **T4.1** σ_bin/σ_cluster sweep across system mass: bias(M̂) + remedy effectiveness; sweep figure.
- **T4.2** MyST page `docs/website/60-science-demos/optimal-design/binary-robustness.md` via
  `myst-expert` (Inputs/assumptions standard; the honest caveats; `myst build` 0 warnings);
  add to the optimal-design section index.
- **T4.3** ADR(s) via `/adr` (ADR-0019+): binary σ²-inflation forward model + f_bin block;
  EFF-OM RV-only choice; marginalize-vs-maximin.
- **T4.4** Test-backbone dashboard re-stamp for the new test module
  (`scripts/build_test_dashboard.py --emit --render`, no coverage re-stamp — no `src/` change);
  FAST + FULL gate green; `numerical-method-validation` + `verification-gate` close-out;
  completion doc `.claude-work/OED_BINARY_MISSPEC_COMPLETE.md`.
- **T4.5** Final whole-arc independent `superpowers:code-reviewer` (the Phase-0.5 lesson:
  integration-level defects hide from per-task reviews).

---

## Verification (whole arc)

- Per-task independent `superpowers:code-reviewer`; final whole-arc review.
- AD-vs-FD gate (f_bin block + binary term); cross-model @slow MC (env-gated, OUT of CI).
- FAST + FULL gates green (XLA caps, `-n auto`), verified LOCALLY; dashboard fresh; `myst build` 0 warnings.
- STATUS.md `next:/blocker:/due:` updated; milestones via `brain "…"`.
- HITL at every phase boundary; **no merge and no push without Anna's separate explicit words.**
