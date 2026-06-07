# Count-model tail-robustness (log-space σ_s² channel) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the tail-fragile linear counts-in-cells over-dispersion ℳ-channel with a tail-robust
**log-count-variance** channel, so SBC (AC18) accepts ℳ rank-uniformity while α/β stay green.

**Architecture:** ℳ enters only through σ_s² = ln(1+(bℳ)²) (a log-density variance). We constrain σ_s²
from `Var[log₊(N)]` of the counts (Neyrinck+2011 Eq 2 transform) — predicted analytically from the
**existing** PLN `count_distribution` P(N), measured the same way on data, compared with a Gaussian
likelihood. The `log₊` transform compresses the fat tail so the variance *converges* (unlike raw
`Var(N)` ∝ ⟨e^{2s}⟩, which diverges for α≤2 and over-predicts on finite fields). ℳ is reported as a
derived quantity (b fixed). α (POT) and β (Gaussianization 2-pt) are untouched.

**Tech Stack:** JAX (float64, jit/grad), Equinox, jaxtyping, pytest, numpy (data/oracle side only),
blackjax NUTS. Experimental subsystem `gravoturb_fdf` (repo-only, `PYTHONPATH=src:src/experimental`).

**Design doc:** `docs/plans/2026-06-06-gravoturb-fdf-count-model-tail-robustness-design.md` (read §3–§6, §9).

**Resolved open decisions (were §8 of the design doc):**
1. **Data estimator = `log₊`** (Neyrinck+2011 Eq 2): `A = ln(1+δ)` for δ>0 else `δ`, with `δ = N/N̄−1`
   (so N=0 → −1; N≤N̄ uses linear δ → no `log 0`). Used identically in prediction and measurement.
2. **Shot noise = modeled exactly** by computing `Var[log₊]` from the Poisson-Lognormal `P(N)` — no
   separate analytic shot term needed (the Poisson mixture *is* the shot noise).
3. **Cell scale = a fixed, data-independent value** (start single-scale; reuse a `cell_sizes` entry),
   so it is **SBC-trivially-valid** (no data-derived cutoff). Multi-scale ladder is a later stretch.

**Run commands (memorize):**
- Experimental unit: `PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental/unit -q`
- Single test: append `-k name` or `path::test`.
- Released-core gate (must stay 814): `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow"`

**Conventions:** RED→GREEN→commit per task. Commit messages end with
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. **HITL: Anna approves each
task's diff before commit** (per `executing-plans` batch checkpoints).

---

### Task 1: `log₊` transform helper

**Files:**
- Modify: `src/experimental/gravoturb_fdf/theory/cic.py` (add `log_plus` near the top, after imports)
- Test: `tests/experimental/unit/test_cic.py`

**Step 1 — Write the failing test**
```python
def test_log_plus_neyrinck_eq2():
    import jax.numpy as jnp
    from gravoturb_fdf.theory.cic import log_plus
    n_bar = 4.0
    n = jnp.array([0.0, 2.0, 4.0, 8.0, 100.0])
    A = log_plus(n, n_bar)
    # delta = n/n_bar - 1 = [-1, -0.5, 0, 1, 24]
    # log_+ : delta>0 -> ln(1+delta)=ln(n/n_bar); else delta
    assert float(A[0]) == -1.0                          # N=0 -> delta=-1 (N=0-safe)
    assert float(A[1]) == jnp.float32(-0.5) or abs(float(A[1]) + 0.5) < 1e-12   # N<N_bar -> linear delta
    assert abs(float(A[2]) - 0.0) < 1e-12              # N=N_bar -> 0
    assert abs(float(A[3]) - jnp.log(2.0)) < 1e-12     # N>N_bar -> ln(n/n_bar)
    assert abs(float(A[4]) - jnp.log(25.0)) < 1e-12
```

**Step 2 — Run, expect FAIL** (`ImportError: cannot import name 'log_plus'`).
Run: `PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental/unit/test_cic.py -k log_plus -q`

**Step 3 — Implement** (in `cic.py`):
```python
def log_plus(n: Float[Array, " ..."], n_bar: Float[Array, ""]) -> Float[Array, " ..."]:
    r"""Modified-log transform of counts (Neyrinck, Szapudi & Szalay 2011, Eq. 2).

    ``A = ln(1+delta)`` for ``delta > 0`` else ``delta``, with ``delta = N/N_bar - 1``. Tail-
    compressing (so ``Var[A]`` converges on a fat-tailed field, unlike ``Var(N)``) and N=0-safe
    (``delta = -1`` there; the branch below ``N_bar`` uses the linear ``delta``, avoiding ``log 0``).
    Differentiable; grad-safe ``where`` guards the ``log1p`` input.
    """
    delta = n / n_bar - 1.0
    pos = delta > 0.0
    return jnp.where(pos, jnp.log1p(jnp.where(pos, delta, 0.0)), delta)
```

**Step 4 — Run, expect PASS.** (Fix the float32 typo in the test if needed: use `abs(float(A[1])+0.5)<1e-12`.)

**Step 5 — Commit:** `feat(gravoturb_fdf): log_plus count transform (Neyrinck+2011 Eq 2)`

---

### Task 2: `predict_log_count_variance` — the tail-robust predicted statistic

**Files:**
- Modify: `src/experimental/gravoturb_fdf/theory/cic.py` (after `count_distribution`)
- Test: `tests/experimental/unit/test_cic.py`

**Step 1 — Write the failing tests** (three properties: monotone in ℳ, **tail-robust** to `n_count_max`, differentiable):
```python
def test_predict_log_count_variance_monotone_and_tailrobust():
    import jax, jax.numpy as jnp
    from gravoturb_fdf.theory.cic import predict_log_count_variance
    from gravoturb_fdf.theory.projection import box_window_sq_grid
    shape, c, n_bar = (24, 24, 24), 4, 5.0
    w2 = box_window_sq_grid(shape, c)
    kw = dict(shape=shape, beta=3.0, R=float(c), b=0.4, alpha=2.5, n_s=512, w2=w2)
    v_lo = float(predict_log_count_variance(n_bar=n_bar, mach=3.0, **kw))
    v_hi = float(predict_log_count_variance(n_bar=n_bar, mach=12.0, **kw))
    assert v_hi > v_lo > 0.0                                  # grows with sigma_s^2(mach)
    # TAIL-ROBUST: insensitive to the count-grid extent (unlike raw Var(N))
    v_a = float(predict_log_count_variance(n_bar=n_bar, mach=12.0, n_count_max=80, **kw))
    v_b = float(predict_log_count_variance(n_bar=n_bar, mach=12.0, n_count_max=400, **kw))
    assert abs(v_a - v_b) / v_b < 1e-3

def test_predict_log_count_variance_differentiable():
    import jax
    from gravoturb_fdf.theory.cic import predict_log_count_variance
    from gravoturb_fdf.theory.projection import box_window_sq_grid
    shape, c = (16, 16, 16), 4
    w2 = box_window_sq_grid(shape, c)
    f = lambda m: predict_log_count_variance(5.0, shape, 3.0, float(c), m, 0.4, 2.5, n_s=256, w2=w2)
    g = float(jax.grad(f)(8.0))
    assert g == g and abs(g) > 0.0   # finite, nonzero d/dmach
```

**Step 2 — Run, expect FAIL** (ImportError).

**Step 3 — Implement** (in `cic.py`; reuses `count_distribution` + `log_plus`):
```python
def predict_log_count_variance(
    n_bar: Float[Array, ""],
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    R: Float[Array, ""],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    n_max: int = 16,
    n_quad: int = 256,
    n_s: int = 1024,
    s_min: float = -15.0,
    s_max: float = 30.0,
    window=top_hat_window,
    w2=None,
    n_count_max: int | None = None,
) -> Float[Array, ""]:
    r"""Tail-robust predicted CIC log-count variance ``Var_{P(N)}[log_plus(N)]``.

    ``P(N)`` is the Poisson-Lognormal compound-Poisson :func:`count_distribution` (shot noise
    modelled exactly by the Poisson mixture). The :func:`log_plus` transform (Neyrinck+2011 Eq 2)
    compresses the fat tail, so this variance **converges** and is insensitive to ``n_count_max``
    (the s-grid extent) -- unlike the raw count over-dispersion ``Var(N) ∝ <e^{2s}>``, which is
    tail-dominated (diverges for alpha<=2) and over-predicts on finite fields. This is the carrier
    of ``sigma_s^2 -> mach``. Differentiable in (mach, b, alpha, beta).
    """
    if n_count_max is None:
        n_count_max = int(n_bar * 8) + 30  # same convention as the SBC mock's bincount length
    n_counts = jnp.arange(n_count_max + 1, dtype=jnp.float64)
    pN = count_distribution(
        n_counts, n_bar, shape, beta, R, mach, b, alpha,
        n_max=n_max, n_quad=n_quad, n_s=n_s, s_min=s_min, s_max=s_max, window=window, w2=w2,
    )
    pN = pN / jnp.sum(pN)                       # condition on [0, n_count_max]
    A = log_plus(n_counts, n_bar)
    mean = jnp.sum(A * pN)
    return jnp.sum((A - mean) ** 2 * pN)
```

**Step 4 — Run, expect PASS.** If the tail-robust assert is tight, relax to `<3e-3` (document why).

**Step 5 — Commit:** `feat(gravoturb_fdf): predict_log_count_variance (tail-robust sigma_s^2 carrier)`

---

### Task 3: `measure_log_count_variance` — the data-side estimator

**Files:**
- Modify: `src/experimental/gravoturb_fdf/validation/measure.py`
- Test: `tests/experimental/unit/test_measure.py` (or the file holding measure tests — confirm path first)

**Step 1 — Write the failing test** (matches the same `log₊` definition; finite & non-negative):
```python
def test_measure_log_count_variance_matches_log_plus():
    import numpy as np
    from gravoturb_fdf.validation.measure import measure_log_count_variance
    rng = np.random.default_rng(0)
    n_bar = 5.0
    counts = rng.poisson(n_bar, size=(16, 16, 16))
    v = measure_log_count_variance(counts, n_bar)
    # reference: same Neyrinck Eq 2 transform, numpy
    d = counts / n_bar - 1.0
    A = np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)
    assert abs(v - float(np.var(A))) < 1e-12
    assert v >= 0.0
```

**Step 2 — Run, expect FAIL.**

**Step 3 — Implement** (numpy; data/oracle side):
```python
def measure_log_count_variance(counts, n_bar):
    r"""Measured CIC log-count variance ``Var_cells[log_plus(N_cell)]`` (Neyrinck+2011 Eq 2).

    The data-side counterpart of :func:`gravoturb_fdf.theory.cic.predict_log_count_variance`;
    uses the identical ``log_plus`` transform so the statistic is consistent in generation and
    inference (SBC-valid). ``counts`` is an integer count grid; ``n_bar`` the mean count per cell.
    """
    import numpy as np
    counts = np.asarray(counts, dtype=float)
    d = counts / n_bar - 1.0
    A = np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)
    return float(np.var(A))
```

**Step 4 — Run, expect PASS.**

**Step 5 — Commit:** `feat(gravoturb_fdf): measure_log_count_variance (data-side log_plus variance)`

---

### Task 4: **Oracle calibration — the decisive gate**

Prove the predicted statistic matches finite realizations to ~few-% **across the whole ℳ prior** —
the quantitative replacement for the design-doc §1 (+9%→+36%) table.

**Files:**
- Modify: `src/experimental/gravoturb_fdf/validation/acceptance.py` (add `ac20_log_count_variance_oracle` helper + assertion)
- Test: `tests/experimental/validation/test_acceptance.py`

**Step 1 — Write the failing test** (mean over realizations; flat residual across ℳ):
```python
@pytest.mark.slow
def test_ac20_log_count_variance_tail_robust_across_mach():
    import numpy as np, jax
    from gravoturb_fdf.field.field import gaussian_random_field, rank_copula_field
    from gravoturb_fdf.field.sampling import sample_cic_counts
    from gravoturb_fdf.theory.cic import predict_log_count_variance
    from gravoturb_fdf.theory.projection import box_window_sq_grid
    from gravoturb_fdf.validation.measure import measure_log_count_variance
    shape, c, n_bar, b, alpha, beta = (64, 64, 64), 4, 5.0, 0.4, 2.5, 3.0
    w2 = box_window_sq_grid(shape, c)
    for mach in [3.0, 5.0, 8.0, 12.0, 16.0]:
        meas = []
        for r in range(4):
            k = jax.random.PRNGKey(100 * r + int(mach))
            s = rank_copula_field(gaussian_random_field(shape, beta, k), mach, b, alpha)
            cnt = np.asarray(sample_cic_counts(s, n_bar, c, jax.random.fold_in(k, 1)))
            meas.append(measure_log_count_variance(cnt, n_bar))
        measured = float(np.mean(meas))
        pred = float(predict_log_count_variance(n_bar, shape, beta, float(c), mach, b, alpha, w2=w2, n_s=1024))
        rel = abs(pred - measured) / measured
        assert rel < 0.06, f"mach={mach}: pred={pred:.4f} meas={measured:.4f} rel={rel:.2%}"
```

**Step 2 — Run, expect FAIL initially** (it forces us to confirm the physics, not assume it).
Run: `PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental/validation/test_acceptance.py -k ac20 -q`

**Step 3 — Make it pass *by getting the physics right*, NOT by widening the tolerance.** Expected: the
`log₊` variance already matches (log compresses the tail). If a residual remains, the legitimate levers
(in priority order, each documented): (a) confirm `w2`/`R`/`n_bar` match generation exactly; (b) raise
`n_s`/`n_count_max` (must be a no-op if truly tail-robust — a *diagnostic*, not a fix); (c) if a real
bias persists at high ℳ, **stop and report to Anna** — fall back per design §12 (robust quantile of
`log₊`, or finite-N truncation). **Do not weaken the 6% gate.**

**Step 4 — Add the `ac20_*` helper** to `acceptance.py` (prints the expected-vs-measured table, returns
pass/fail) mirroring existing AC helpers; wire into `main()` AC list.

**Step 5 — Commit:** `feat(gravoturb_fdf): AC20 log-count-variance oracle — tail-robust across mach`

---

### Task 5: `log_count_variance_loglike` — the Gaussian inference block

**Files:**
- Modify: `src/experimental/gravoturb_fdf/inference/likelihood.py`
- Test: `tests/experimental/unit/test_inference.py`

**Step 1 — Write the failing test** (peaks at truth on noiseless data; differentiable):
```python
def test_log_count_variance_loglike_peaks_at_truth():
    import jax, jax.numpy as jnp
    from gravoturb_fdf.theory.cic import predict_log_count_variance
    from gravoturb_fdf.theory.projection import box_window_sq_grid
    from gravoturb_fdf.inference.likelihood import log_count_variance_loglike
    shape, c, n_bar = (24, 24, 24), 4, 5.0
    w2 = box_window_sq_grid(shape, c)
    truth = jnp.array([8.0, 0.4, 2.5, 3.0])
    # noiseless "measured" = prediction at truth
    meas = float(predict_log_count_variance(n_bar, shape, 3.0, float(c), 8.0, 0.4, 2.5, w2=w2))
    ll = lambda m: float(log_count_variance_loglike(meas, jnp.array([m,0.4,2.5,3.0]), shape, c, n_bar, var_v=1e-3))
    assert ll(8.0) > ll(5.0) and ll(8.0) > ll(12.0)     # peaks at the injected mach
    g = float(jax.grad(lambda m: log_count_variance_loglike(meas, jnp.array([m,0.4,2.5,3.0]), shape, c, n_bar, var_v=1e-3))(8.0))
    assert g == g
```

**Step 2 — Run, expect FAIL.**

**Step 3 — Implement** (mirrors `count_loglike`'s cubic-cell window; `var_v` = fixed estimator variance):
```python
def log_count_variance_loglike(
    measured_v: Float[Array, ""],
    theta: Float[Array, " 4"],
    shape: tuple[int, int, int],
    cell_size: int,
    n_bar: Float[Array, ""],
    var_v: Float[Array, ""],
    n_max: int = 14,
    n_quad: int = 256,
    n_s: int = 1024,
    s_max: float = 40.0,
) -> Float[Array, ""]:
    r"""Gaussian log-likelihood on the tail-robust CIC log-count variance (the sigma_s^2 -> mach block).

    Compares the measured ``Var_cells[log_plus(N)]`` to the analytic
    :func:`~gravoturb_fdf.theory.cic.predict_log_count_variance` at ``theta``; ``var_v`` is the fixed
    (fiducial-mock) estimator variance (Decision #4 mock-precision pattern, no log|C| term). Replaces
    ``count_loglike`` in the inference path: tail-robust, so it does not bias mach high. Differentiable
    in ``theta = (mach, b, alpha, beta)`` (carries mach; alpha/beta enter only weakly via P(N)/R).
    """
    mach, b, alpha, beta = theta
    pred = predict_log_count_variance(
        n_bar, shape, beta, float(cell_size), mach, b, alpha,
        n_max=n_max, n_quad=n_quad, n_s=n_s, s_max=s_max,
        w2=box_window_sq_grid(shape, cell_size),
    )
    return -0.5 * (pred - measured_v) ** 2 / var_v
```

**Step 4 — Run, expect PASS.**

**Step 5 — Commit:** `feat(gravoturb_fdf): log_count_variance_loglike (Gaussian sigma_s^2 block)`

---

### Task 6: `var_v` estimator-variance helper (fixed fiducial noise model)

**Files:**
- Modify: `src/experimental/gravoturb_fdf/validation/measure.py` (oracle side; numpy + jax sampling)
- Test: `tests/experimental/unit/test_measure.py`

**Step 1 — Write the failing test** (positive, finite, ~scales like 1/n_real):
```python
def test_log_count_variance_estimator_var_positive():
    import jax
    from gravoturb_fdf.validation.measure import estimate_log_count_variance_var
    vv = estimate_log_count_variance_var(
        mach=8.0, b=0.4, alpha=2.5, beta=3.0, shape=(24,24,24),
        cell_size=4, n_bar=5.0, n_real=8, key=jax.random.PRNGKey(0))
    assert vv > 0.0 and vv == vv
```

**Step 2 — Run, expect FAIL.**

**Step 3 — Implement** (small fiducial mock ensemble → variance of the measured statistic):
```python
def estimate_log_count_variance_var(mach, b, alpha, beta, shape, cell_size, n_bar, n_real, key):
    r"""Fixed (fiducial) variance of the ``measure_log_count_variance`` estimator from an ``n_real``
    mock ensemble at theta_fid. Used as ``var_v`` in :func:`log_count_variance_loglike` (mock-precision
    pattern; computed ONCE per inference, not per NUTS step -> SBC-valid as a fixed constant)."""
    import numpy as np, jax
    from gravoturb_fdf.field.field import gaussian_random_field, rank_copula_field
    from gravoturb_fdf.field.sampling import sample_cic_counts
    vals = []
    for r in range(n_real):
        k = jax.random.fold_in(key, r)
        s = rank_copula_field(gaussian_random_field(shape, beta, k), mach, b, alpha)
        cnt = np.asarray(sample_cic_counts(s, n_bar, cell_size, jax.random.fold_in(k, 1)))
        vals.append(measure_log_count_variance(cnt, n_bar))
    return float(np.var(vals, ddof=1))
```

**Step 4 — Run, expect PASS.**

**Step 5 — Commit:** `feat(gravoturb_fdf): fiducial estimator-variance for the log-count-variance block`

---

### Task 7: Wire the new block into the SBC driver (replace `count_loglike`)

**Files:**
- Modify: `src/experimental/gravoturb_fdf/inference/sbc.py` (`build_logdensity` + `_build_mock`)
- Test: `tests/experimental/unit/test_sbc.py` (or wherever the SBC-driver unit tests live — confirm path)

**Step 1 — Write the failing test** (the logdensity is finite + differentiable with the new block):
```python
def test_build_logdensity_uses_log_count_variance_block():
    import jax, jax.numpy as jnp
    # build a tiny mock bundle with the new keys, then assert logdensity(z) is finite and has grad
    # (mirror the existing SBC unit test that exercises build_logdensity; swap count_hists -> meas_v/var_v)
    ...
```

**Step 2 — Run, expect FAIL.**

**Step 3 — Implement.** In `_build_mock`: after generating the stellar counts, compute
`meas_v = measure_log_count_variance(cnt, nb)` and `var_v = estimate_log_count_variance_var(theta*_fid...)`
per cell scale; put `"log_count_vars"` (tuple), `"log_count_var_vars"` (tuple), `"n_bars"`, `"cell_sizes"`
into `data` (drop `count_hists`). In `build_logdensity`: replace the `count_loglike` loop with
```python
for c, mv, vv, nb in zip(cell_sizes, meas_vs, var_vs, n_bars):
    ll = ll + log_count_variance_loglike(mv, theta4, shape, c, nb, vv, n_max=n_max, n_s=n_s)
```
Keep the POT `tail_exceedance_loglike` and `prior.logpdf` + `log_jacobian` exactly as-is.

**Step 4 — Run, expect PASS.** Then a fast smoke SBC: `sbc_ranks(..., n_trials=2, n_warmup=40, n_samples=40)`
runs without NaN.

**Step 5 — Commit:** `feat(gravoturb_fdf): SBC driver uses tail-robust log-count-variance block`

---

### Task 8: **AC18 — flip the ℳ xfail to PASS** (the integration gate)

**Files:**
- Modify: `src/experimental/gravoturb_fdf/validation/acceptance.py::ac18_sbc_rank_uniformity` (remove the
  ℳ xfail marker / expectation once green)
- Test: `tests/experimental/validation/test_acceptance.py::...ac18...`

**Step 1 — Run AC18 at the test config** (n_trials=30, shape=(24)³, density (64)³, warmup 120, samples 200, thin 4):
`PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental/validation -k ac18 -q`
Expected now: **ℳ p > 0.05** (uniform), α/β still pass.

**Step 2 — Remove the xfail** (the e8d0222 expectation) so AC18 asserts all three params uniform.

**Step 3 — Run again, expect PASS.** If ℳ still fails: **STOP, report to Anna** with the rank histogram
(do not tune thresholds). Likely culprits to investigate first: `var_v` mis-scaled (too small → over-tight
→ ∪/∩, not slope), or a gen/inference statistic mismatch (cell scale / `n_bar` / `w2`).

**Step 4 — Confirm AC16 still green** (single-point recovery): `... pytest -k ac16 -q`.

**Step 5 — Commit:** `feat(gravoturb_fdf): AC18-M passes — tail-robust count model (xfail removed)`

---

### Task 9: Retire `count_loglike` from the inference path + docs sweep

**Files:**
- Modify: `inference/likelihood.py` (delete `count_loglike`; keep `theory/cic.py::count_distribution` as
  a labelled diagnostic), `README.md`, `VALIDATION_SUMMARY.md`, the design/handoff cross-refs, top-level
  `CLAUDE.md` test counts.
- Test: update/remove `count_loglike` unit tests; add a one-line note that `count_distribution` is
  diagnostic-only.

**Step 1 — Grep** `count_loglike` usages: `grep -rn count_loglike src tests`. Confirm only tests + (now
removed) sbc reference remain.

**Step 2 — Remove** `count_loglike` + its tests; update docstrings/README to point ℳ → the log-count-variance block.

**Step 3 — Run the full experimental suite:**
`PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental -q`
Expected: green (record the new count).

**Step 4 — Released-core gate:** `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow"` → **814** unchanged.

**Step 5 — Commit:** `refactor(gravoturb_fdf): retire count_loglike from inference (count_distribution -> diagnostic)`

---

### Task 10 (stretch, optional): σ(ℳ)-vs-N_star forecast + multi-scale ladder

Honest-scope deliverable mirroring AC17 (α): forecast the shot-noise-limited σ(ℳ) vs number of stars,
and (optionally) extend the block to a small fixed cell-scale ladder with a mock covariance (Hartlap).
**Defer unless Anna asks** — Tasks 1–9 deliver the calibrated fix.

---

## Final verification gate (definition of done)

- [ ] AC20 oracle: `|pred−meas|/meas < 6%` for ℳ ∈ {3,5,8,12,16} (Task 4).
- [ ] AC18-ℳ rank-uniformity **passes** (p>0.05); α/β still pass (Task 8).
- [ ] AC16 single-point recovery still green.
- [ ] Full experimental suite green; **released-core 814 invariant**.
- [ ] `count_loglike` retired; docs updated; design/handoff cross-referenced.
- [ ] Memories updated (`gravoturb-fdf-count-model-tail-robustness`, `gravoturb-fdf-sbc-figures-arc`):
      bug fixed → resume deferred Task 8 (AC19 HMC) + Task 9 (figures orchestrator).
