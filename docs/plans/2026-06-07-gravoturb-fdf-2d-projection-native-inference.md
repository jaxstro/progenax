# 2D-Projection-Native Inference — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-scope the gravoturb_fdf inference to run on the *actual observable* — a 2D projected
star catalog — so all channels share one shot-noise-limited reality, SBC (AC18) passes honestly, and
runs drop from ~40 min to minutes.

**Architecture:** Generative mock → 3D rank-copula BM19 field → Poisson stars → **project along the
LOS** → 2D sky count map. Three channels measured from that one map: **β** (angular clustering,
Limber-projected linear-ρ 2-pt — primary), **ℳ** (projected `Var[log₊(N₂D)]`, de-projected via a
**depth nuisance L** — secondary), **α** (POT tail — depth-gated, kept but reported as a limit from
2D-only data). Covariance is a **data-derived jackknife** (truth-independent → SBC-valid, no log|C|
mismatch), dissolving the fixed-fiducial-over-wide-prior wall.

**Tech Stack:** JAX (float64, jit/grad), Equinox, jaxtyping, pytest, numpy (data/oracle side),
blackjax NUTS. Experimental `gravoturb_fdf` (repo-only, `PYTHONPATH=src:src/experimental`).

**Design doc:** `docs/plans/2026-06-07-gravoturb-fdf-2d-projection-native-inference-design.md`
(read §3–§7). **Gate spike:** `validation/projection_fisher_spike.py` (β survives ~1.8×, ℳ ~2.6×,
α the casualty — depth-gated).

**Non-negotiables (every task):** No hacks / no test-weakening — fix the physics. JAX-native +
differentiable. **SBC-valid**: every data-derived quantity (jackknife C, sky-cell scale, POT
threshold, L prior) identical in generation + inference; covariances truth-independent (never keyed
to θ*). Ground any primary-source/formula claim in the actual PDF (no-assumptions; ask Anna for the
PDF). **HITL: Anna approves each task's diff before commit.** Released-core stays **814**.

**Run commands (memorize):**
- Experimental unit: `PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental/unit -q`
- Single: append `-k name` or `path::test`.
- Released-core gate (must stay 814): `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow"`

**Conventions:** RED→GREEN→commit per task. Commit messages end with
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Milestone A — the 2D projection layer (predicted + measured statistics)

### Task A1: `limber_project_slab` — depth-L line-of-sight projection of a 2-pt

The existing `limber_project_grid` is the **periodic L=n_los limit** (uniform weight `n_los`). A
finite slab of depth `L` cells projects with a **triangular overlap weight**: for Σ = Σ_{z=0}^{L-1}ρ,
`Cov(Σ, Σ) = Σ_{dl=-(L-1)}^{L-1} (L − |dl|) ξ_3D(r⊥, dl)`. This makes the **depth nuisance L** enter
the predicted 2-pt analytically and differentiably (the weight is piecewise-linear in L).

**Files:**
- Modify: `src/experimental/gravoturb_fdf/theory/projection.py`
- Test: `tests/experimental/unit/test_projection.py`

**Step 1 — Write the failing tests:**
```python
def test_limber_slab_matches_periodic_at_full_depth():
    import jax.numpy as jnp
    from gravoturb_fdf.theory.projection import (
        gaussian_correlation_grid, limber_project_grid, limber_project_slab)
    xi = gaussian_correlation_grid((16, 16, 16), 3.0)
    full = limber_project_slab(xi, depth=16, los_axis=2)   # full periodic depth
    ref = limber_project_grid(xi, los_axis=2)
    assert jnp.allclose(full, ref, rtol=1e-10)

def test_limber_slab_shallower_depth_lowers_amplitude_and_is_differentiable():
    import jax, jax.numpy as jnp
    from gravoturb_fdf.theory.projection import gaussian_correlation_grid, limber_project_slab
    xi = gaussian_correlation_grid((16, 16, 16), 3.0)
    shallow = float(limber_project_slab(xi, depth=4.0, los_axis=2)[0, 0])
    deep = float(limber_project_slab(xi, depth=12.0, los_axis=2)[0, 0])
    assert 0.0 < shallow < deep                              # variance grows with depth
    g = jax.grad(lambda L: limber_project_slab(xi, depth=L, los_axis=2)[0, 0])(6.0)
    assert g == g and abs(g) > 0.0                           # finite, nonzero d/dL
```

**Step 2 — Run, expect FAIL** (ImportError).

**Step 3 — Implement** (in `projection.py`):
```python
def limber_project_slab(xi_3d, depth, los_axis: int = 2):
    r"""Depth-``L`` (cells) LOS projection of a 3-D autocovariance: the triangular-overlap sum
    ``xi_Sigma(r_perp) = sum_{dl} (L - |dl|)_+ xi_3d(r_perp, dl)`` for Sigma = sum_{z<L} rho. The
    ``L=n_los`` periodic limit equals :func:`limber_project_grid`. Differentiable in ``depth`` (the
    depth nuisance) via the grad-safe ``relu`` weight; ``depth`` may be non-integer."""
    import jax.numpy as jnp
    n = xi_3d.shape[los_axis]
    dl = jnp.fft.fftfreq(n) * n                              # signed LOS separations (cells)
    w = jnp.clip(depth - jnp.abs(dl), a_min=0.0)            # triangular overlap weight (relu)
    shape = [1, 1, 1]; shape[los_axis] = n
    return jnp.sum(xi_3d * w.reshape(shape), axis=los_axis)
```

**Step 4 — Run, expect PASS.**

**Step 5 — Commit:** `feat(gravoturb_fdf): limber_project_slab — depth-L triangular LOS projection`

---

### Task A2: 2D measurement helpers (the data side)

**Files:**
- Modify: `src/experimental/gravoturb_fdf/validation/measure.py`
- Test: `tests/experimental/unit/test_measure.py`

Add: `project_counts_los(counts3d, depth, los_axis=2)` (sum the first `depth` slices → 2D map);
`measure_angular_bandpowers_2d(map2d, k_edges)` (the 2D analog of `measured_bandpowers`, using a 2D
`|k|` grid). `measure_log_count_variance` already works on a 2D array (it is shape-agnostic) — add a
test confirming it. Keep numpy (oracle side).

**Step 1 — Failing tests:**
```python
def test_project_counts_los_sums_slices():
    import numpy as np
    from gravoturb_fdf.validation.measure import project_counts_los
    c = np.ones((8, 8, 8))
    assert np.allclose(project_counts_los(c, depth=8), 8.0)
    assert np.allclose(project_counts_los(c, depth=3), 3.0)

def test_measure_angular_bandpowers_2d_shape_and_positive():
    import numpy as np
    from gravoturb_fdf.validation.measure import measure_angular_bandpowers_2d
    rng = np.random.default_rng(0)
    bp = measure_angular_bandpowers_2d(rng.normal(size=(32, 32)), np.linspace(1.0, 8.0, 4))
    assert bp.shape == (3,) and np.all(bp >= 0.0)
```

**Step 2 — Run, expect FAIL.**

**Step 3 — Implement** (mirror `measured_bandpowers`; 2D `|k|` from `np.fft.fftfreq`). Reuse the
exact periodogram convention `|fft(f-<f>)|^2 / f.size` so predicted/measured match.

**Step 4 — Run, expect PASS.**

**Step 5 — Commit:** `feat(gravoturb_fdf): 2D measurement helpers (project_counts_los, angular bandpowers)`

---

### Task A3: predicted angular band-powers (β carrier) + **oracle gate**

Counts trace the **linear** density, so the angular clustering predicted statistic is the Limber-
projected **linear-ρ** 2-pt (`linear_hermite_coefficients`, not the log-density `c_n`). **The
normalization and the linear-vs-log-2pt sub-decision (design §4) are settled by the oracle test, NOT
assumed.**

**Files:**
- Modify: `src/experimental/gravoturb_fdf/inference/covariance.py` (new `angular_bandpowers_2d`)
- Test: `tests/experimental/unit/test_covariance.py` + an oracle in `tests/experimental/validation/`

**Step 1 — Write the predicted fn + a RED oracle test** (predicted vs measured mock band-powers
across β, max-ABSOLUTE error — per the AC12 lesson: red/projected spectra use absolute, not
ratio-on-small-denominator):
```python
@pytest.mark.slow
def test_angular_bandpowers_2d_matches_mock_across_beta():
    import numpy as np, jax
    from gravoturb_fdf.field.field import gaussian_random_field, rank_copula_field
    from gravoturb_fdf.field.sampling import sample_cic_counts
    from gravoturb_fdf.validation.measure import (project_counts_los, measure_angular_bandpowers_2d)
    from gravoturb_fdf.inference.covariance import angular_bandpowers_2d
    shape, depth, k_edges = (48, 48, 48), 48, np.linspace(1.0, 6.0, 4)
    mach, b, alpha, n_bar1 = 8.0, 0.4, 2.5, 0.4
    for beta in (2.5, 3.0, 3.5):
        meas = []
        for r in range(6):
            k = jax.random.PRNGKey(r)
            s = rank_copula_field(gaussian_random_field(shape, beta, k), mach, b, alpha)
            cnt = np.asarray(sample_cic_counts(s, n_bar1, 1, jax.random.fold_in(k, 1)))
            meas.append(measure_angular_bandpowers_2d(project_counts_los(cnt, depth), k_edges))
        measured = np.mean(meas, axis=0)
        pred = np.asarray(angular_bandpowers_2d(shape, depth, beta, mach, b, alpha, k_edges))
        assert np.max(np.abs(pred - measured)) < 0.05 * measured.max(), (beta, pred, measured)
```

**Step 2 — Run, expect FAIL** (forces us to get the projection + normalization right, not assume it).

**Step 3 — Implement** `angular_bandpowers_2d`: build `xi_rho_3d = gaussianized_xi(rho_g(β),
linear_hermite_coefficients(ℳ,b,α))`; `xi_2d = limber_project_slab(xi_rho_3d, depth)`; `P2d =
fft2(xi_2d).real`; bin by 2D `|k|`; add the shot term consistent with the measured periodogram
normalization (`+ 1/N̄₂D` on the δ-power, with N̄₂D from `depth`/cells). **Tune normalization to the
oracle; if a residual persists, STOP and report to Anna** (candidate: the linear 2-pt is mildly
tail-sensitive even projected → switch the carrier to the 2-pt of `log₊(N₂D)`, design §4, oracle-
checked). Do NOT widen the tolerance.

**Step 4 — Run, expect PASS.** Also add a fast non-slow unit test (shape, differentiable in β).

**Step 5 — Commit:** `feat(gravoturb_fdf): angular_bandpowers_2d (Limber-projected beta carrier) + oracle`

---

### Task A4: predicted projected log-count variance (ℳ carrier, depth L) + **2D oracle (AC20-2D)**

The 2D analog of `predict_log_count_variance`: the projected count variance `σ²_{N,2D} = N̄₂D +
N̄₂D²·ξ̄_ρ,2D(R; L)`, fed through the `log₊` transform of the projected P(N). Preserves the cured
tail-robustness. ℳ and **L** both enter (the de-projection); the oracle gate must be flat across the
ℳ prior **at fixed L**.

**Files:**
- Modify: `src/experimental/gravoturb_fdf/theory/cic.py` (new `predict_log_count_variance_2d`)
- Test: `tests/experimental/unit/test_cic.py` + `tests/experimental/validation/test_acceptance.py`

**Step 1 — Write the 2D oracle (RED), mirroring AC20** (predicted vs finite-field 2D
`Var[log₊(N₂D)]`, signed relative residual flat across ℳ∈{4,6,8,12,16,20}, `rel_tol=0.06`). Use the
same `rank_copula_field → sample_cic_counts → project_counts_los → measure_log_count_variance` chain
as generation (SBC-valid identical statistic).

**Step 2 — Run, expect FAIL.**

**Step 3 — Implement** `predict_log_count_variance_2d` reusing `cell_averaged_xi_rho` + Limber
(`limber_project_slab`) for `ξ̄_ρ,2D(R; L)`, then the `log₊` variance of the projected compound-
Poisson P(N) (reuse the `count_distribution` → `log_plus` machinery with the projected `xi_bar`).
**Make it pass by getting the physics right (projection + N̄₂D bookkeeping), NOT by widening 6%.** If
a real high-ℳ residual persists → STOP, report (design §12 fallbacks).

**Step 4 — Run, expect PASS.** Add a fast unit test: monotone in ℳ, differentiable in (ℳ, depth).

**Step 5 — Commit:** `feat(gravoturb_fdf): predict_log_count_variance_2d + AC20-2D oracle (depth-aware)`

---

## Milestone B — covariance + likelihood blocks

### Task B1: `jackknife_covariance` — data-derived, truth-independent

**Files:** Modify `inference/covariance.py`; Test `tests/experimental/unit/test_covariance.py`.

Delete-d-patch jackknife over sky sub-patches of the *observed* 2D data vector → covariance. It is a
pure function of the data (truth-independent → SBC-valid) and captures non-Gaussian + shot-noise +
cross-block structure with **no log|C| term**.

**Step 1 — Failing test:** PD, deterministic (same data → same C), scales sensibly with n_patches.
**Step 2 — FAIL. Step 3 — Implement** (numpy delete-1 jackknife: `C = (n-1)/n Σ (d_i − d̄)(·)ᵀ`).
**Step 4 — PASS. Step 5 — Commit:** `feat(gravoturb_fdf): jackknife_covariance (data-derived, SBC-valid)`

---

### Task B2: `angular_clustering_loglike_2d` (β) — Gaussian on band-powers w/ jackknife precision

**Files:** Modify `inference/likelihood.py`; Test `tests/experimental/unit/test_inference.py`.
Peaks at the injected β on noiseless data; differentiable in β. `-1/2 r^T Cinv r` with the jackknife
precision (Hartlap-corrected). **Step 1** test peaks-at-truth + finite grad. **Steps 2–4** RED→GREEN.
**Step 5 — Commit:** `feat(gravoturb_fdf): angular_clustering_loglike_2d (beta block)`

---

### Task B3: `log_count_variance_loglike_2d` (ℳ + depth L)

**Files:** Modify `inference/likelihood.py`; Test `test_inference.py`. Gaussian on the projected
`Var[log₊(N₂D)]` vs `predict_log_count_variance_2d`, differentiable in (ℳ, depth). Peaks at injected
ℳ at fixed L; the (ℳ, L) ridge is the expected degeneracy (document; L marginalized by the prior).
RED→GREEN→**Commit:** `feat(gravoturb_fdf): log_count_variance_loglike_2d (mach + depth L)`

---

### Task B4: 2D POT wrapper (α — depth-gated, reuses `tail_exceedance_loglike`)

**Files:** Modify `validation/measure.py` (a `measure_exceedances` call on the projected map) +
confirm `tail_exceedance_loglike` is unchanged. Test: the projected exceedance histogram is valid
(monotone edges; empty-tail → 0 contribution, the existing finite-field branch). No new physics —
α's machinery carries over; its *information* is gated by the (sparse) projected N_tail.
**Commit:** `feat(gravoturb_fdf): 2D projected POT measurement (alpha depth-gated)`

---

## Milestone C — depth nuisance prior

### Task C1: extend the prior with L

**Files:** Modify `inference/priors.py` (add an `L` nuisance to a `BM19Prior2D`, or compose); Test
`tests/experimental/unit/test_priors.py`. Physical prior on the LOS depth (e.g. LogUniform around the
transverse size — aspect ratio ~O(1); the exact form is a science choice — **confirm with Anna**).
Sample + logpdf + the reparam/Jacobian entry for L. RED→GREEN→**Commit:**
`feat(gravoturb_fdf): depth-nuisance L prior for 2D inference`

---

## Milestone D — 2D SBC driver

### Task D1: `_build_mock_2d`

**Files:** Modify (or add `inference/sbc_2d.py`); Test `test_sbc.py`. Mirror `_build_mock` but add the
**project step** (`project_counts_los`) and measure the 2D data vector (angular band-powers, projected
`Var[log₊]`, projected exceedances). Build the **jackknife C from the trial's own data** (truth-
independent). RED (data bundle keys) → GREEN → **Commit:** `feat(gravoturb_fdf): _build_mock_2d (projected observable)`

### Task D2: `build_logdensity_2d`

**Files:** same. Compose: `tail_exceedance_loglike` (α) + `log_count_variance_loglike_2d` (ℳ,L) +
`angular_clustering_loglike_2d` (β) + prior (incl. L) + Jacobian, using the **jackknife precision**.
RED: logdensity finite + differentiable in z=(ℳ,α,β,L). GREEN. **Commit:**
`feat(gravoturb_fdf): build_logdensity_2d (3 channels + depth, jackknife C)`

### Task D3: `sbc_ranks_2d`

**Files:** same. Generalize `sbc_ranks` to the 2D mock/logdensity (4 params incl. L). Smoke test:
`n_trials=2, n_warmup=40, n_samples=40` runs without NaN. **Commit:**
`feat(gravoturb_fdf): sbc_ranks_2d driver`

---

## Milestone E — acceptance gates (the definition of done)

### Task E1: AC15-2D — mature the spike into a forecast AC
**Files:** `validation/acceptance.py` (`ac15_2d_projection_forecast`), fed by the spike's estimator.
Reports σ(β), σ(ℳ|L), the α-degradation curve, the 1/√V scaling, the 2D/3D ratios. PASS = Fisher PD,
errors finite, σ∝1/√V, β the best-constrained. **Commit:** `feat(gravoturb_fdf): AC15-2D projected forecast (matured spike)`

### Task E2: AC16-2D — joint recovery on a 2D mock
**Files:** `acceptance.py` (`ac16_2d_hmc_recovery`). (ℳ, β) cover truth within nσ; **α reported as a
limit** (depth-gated — assert the posterior ≈ prior, documenting the honest scope); L marginalized.
**Commit:** `feat(gravoturb_fdf): AC16-2D recovery (mach,beta cover; alpha limit)`

### Task E3: AC18-2D — SBC rank-uniformity (the integration gate)
**Files:** `acceptance.py` (`ac18_2d_sbc_rank_uniformity`) via `sbc_ranks_2d`. β and ℳ rank-uniform
(p>0.05); α per its gated scope (uniform under the prior it's effectively sampling). Integer-aware
χ² (the C1 helpers). If a param fails → **STOP, report the rank histogram to Anna** (do not tune
thresholds). **Commit:** `feat(gravoturb_fdf): AC18-2D SBC passes (beta,mach uniform)`

### Task E4: invariants + suite
**Files:** wire new ACs into `main()`. Run full experimental suite (record new count) + released-core
gate **814**. **Commit:** `test(gravoturb_fdf): wire AC15/16/18-2D into acceptance; suite green`

---

## Final verification gate (definition of done)

- [ ] A3 angular-bandpower oracle + A4 AC20-2D oracle pass (predicted vs finite-field 2D, ~few-%).
- [ ] AC18-2D rank-uniformity passes for **β and ℳ**; α per gated scope.
- [ ] AC16-2D: (ℳ, β) cover; α reported as a limit; L marginalized.
- [ ] AC15-2D forecast reproduces the spike (β best, σ∝1/√V).
- [ ] Full experimental suite green; **released-core 814 invariant**.
- [ ] Memories updated; design/handoff cross-referenced. Deferred-after items (count_loglike
      retirement, AC19 HMC convergence, figure-gallery orchestrator) re-queued.

## Out of scope (deferred, per the brief)

The depth-resolved 3D-star α mode; LOS velocities; retiring `count_loglike`; AC19 (HMC convergence);
the figure-gallery orchestrator — picked up after the 2D inference lands.
