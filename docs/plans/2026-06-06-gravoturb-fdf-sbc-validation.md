# gravoturb_fdf Workstream ① — SBC + HMC Diagnostics + Figures Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the gravoturb_fdf differentiable-inference layer from single-θ injection–recovery (AC16) to a *calibrated, convergence-diagnosed* inference engine, with publication figures — via Simulation-Based Calibration (SBC), HMC convergence diagnostics (R̂/ESS/divergences), and reusable jaxstroviz figure functions.

**Architecture:** Three new `inference/` modules (`priors.py`, `sbc.py`, `diagnostics.py`) + an extended multi-chain `run_nuts`, two new acceptance criteria (AC18 SBC rank-uniformity, AC19 HMC convergence) wired into `validation/acceptance.py`, and one genuinely-new figure type (`plot_sbc_rank_histogram`) added to **jaxstroviz** (DRY). Everything is experimental-only; the released-core 815-test invariant must stay green. The mock is still drawn from BM19, so **SBC validates calibration *under the assumed model*, not misspecification** — that honest boundary is stated in every deliverable.

**Tech Stack:** JAX (float64) + blackjax (NUTS, per-step `info`) for the engine; numpy + scipy.stats (χ²/KS) on the validation/oracle side; arviz for R̂/ESS; matplotlib via jaxstroviz for figures. uv for everything.

**Branch:** `gravoturb-fdf-sbc-validation` (off `main` @ 006ccdc).

---

## Rules in force (non-negotiable)

- **HITL** — Anna approves at every task/fork; **no git push / no PR without her explicit go**.
- **Strict TDD** — write the failing test, *run it and watch it fail* (paste output), then minimal GREEN, then refactor. Commit each task.
- **Verify formulae against held PDFs** — never assert a primary-source fact from memory. The SBC rank statistic + uniformity test must be grounded in the actual **Talts et al. (2018)** PDF (Task 0). R̂/ESS conventions grounded in **Vehtari et al. (2021)** or the arviz docs.
- **JAX-native core** — `jax.numpy`, `lax.scan`, no `while_loop`. numpy/scipy/matplotlib/arviz **only** on the validation / figure / diagnostics side.
- **Evidence-before-done** — paste fresh `uv` command output for any pass/works claim.
- **Released-core invariant** — after every task: `pytest tests/unit tests/integration tests/validation -m "not slow"` stays green (815).

## Honest scope (state in the paper + every docstring)

A green SBC means the posterior is **self-consistent and calibrated under the BM19 generative model**. It does **not** test that real turbulent clouds follow that model — that is workstream ③ (external/misspecification validation). AC16's "injection–recovery, not real data" caveat **remains**.

## Verification commands (uv only)

```bash
# experimental suite (incl. new tests)
PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental -q
# released-core INVARIANT — must stay green (815)
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -m "not slow" -q
# acceptance driver — must print AC1–AC19
PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -m gravoturb_fdf.validation.acceptance
# jaxstroviz figure functions (in the jaxstroviz repo)
cd ~/projects/jaxstro-dev/jaxstroviz && env -u VIRTUAL_ENV uv run --no-sync pytest tests/ -q
```

## Verified API facts (from 2026-06-06 codebase exploration — cite, don't re-derive)

- `hmc.py:38` `run_nuts(logdensity_fn, init_position, key, n_warmup=500, n_samples=1000)` → `(n_samples, *pos.shape)`, **single chain**, discards `kernel.step` `info`.
- `hmc.py:25` `to_constrained(z)` = `[exp(z0), 1+exp(z1), exp(z2)]` → (M, α, β); `hmc.py:19` `to_unconstrained`; `hmc.py:30` `log_jacobian(z)=sum(z)`.
- `likelihood.py:121` `tail_exceedance_loglike(exc_counts, exc_edges, theta, s_thr, s_max, floor=1e-300)`; `likelihood.py:164` `pot_validity_barrier(theta, s_thr, eps=0.1)`; `likelihood.py:65` `count_loglike(count_hist, theta, shape, cell_size, n_bar, n_max=14, n_quad=256, n_s=1024, s_max=40.0, floor=1e-300)`.
- `bm19.py:19` `sigma_s_squared(mach, b)=ln(1+(b·mach)²)`; `bm19.py:38` `transition_density(alpha, sigma_s_sq)=(alpha-0.5)·sigma_s_sq`. ⟹ **POT-valid bound:** `α ≤ 0.5 + s_thr/ln(1+(b·M)²)`.
- `field.py:90` `rank_copula_field(g, mach, b, alpha)`; `field.py:32` `gaussian_random_field(shape, beta, key)`; `sampling.py:24` `sample_cic_counts(s, n_bar, cell_size, key)`; `measure.py:103` `measure_exceedances(s_field, s_thr, n_bins=20)→(exc_counts, exc_edges, s_max, n_tail)`.
- ac16 logdensity (acceptance.py:606–613): `tail_exceedance_loglike + Σ count_loglike + pot_validity_barrier + log_jacobian`. theta = `[M, b, α, β]` with **b fixed**.
- ACs return `dict` with `"passed": bool`; `main()` (acceptance.py:764) collects a `results` dict; tests (test_acceptance.py) carry `pytestmark=[experimental, validation]`, HMC ones `@pytest.mark.slow`, reduced config in test / full in `main()`.
- jaxstroviz: `analysis/inference.py` dict-returning (`compute_parameter_recovery`, `compute_coverage_probability`); `plots/inference.py` axes-first `plot_X(ax, …)->None` (`plot_posterior_1d/2d`, `plot_parameter_recovery`, `plot_recovery_correlation`, `plot_inference_summary`); `styles`: `newfig`, `gridfig`, `savefig`, `set_paper`, `to_numpy`, `PALETTE`; tests use `matplotlib.use("Agg")`. **No** SBC rank histogram exists. Deps: matplotlib yes, **arviz no**, no `[viz]` extra.

---

## Task 0: Dependencies + Talts (2018) PDF grounding (GATING — needs Anna)

**Files:**
- Modify: `pyproject.toml` (the `[project.optional-dependencies] experimental` block)
- Regenerate: `uv.lock`
- Create: `docs/website/99-bibliography/per-paper/talts-2018.md`
- Modify: `docs/website/references.bib` (add `Talts2018`, and `Vehtari2021` if R̂ is cited)

**Step 1 — HITL gate (Anna).** Ask Anna to drop the **Talts et al. (2018)** PDF ("Validating Bayesian Inference Algorithms with Simulation-Based Calibration", arXiv:1804.06788) into `docs/core-papers/`. Do **not** implement the SBC rank statistic from memory.

**Step 2 — Verify against the held PDF and record.** Read the PDF; confirm and quote:
- the **rank statistic**: for posterior draws `{θ_l}_{l=1..L}` and prior draw `θ*`, the rank `r = Σ_l 1[θ_l < θ*] ∈ {0,…,L}`; under calibration `r ~ DiscreteUniform{0,…,L}`.
- the **thinning** recommendation (draws thinned to ~independent to avoid autocorrelation artifacts in the histogram).
- the recommended **number of SBC trials** and **bin count** (bins should divide `L+1` evenly), and the **uniformity test** they discuss (χ² of the rank histogram vs uniform; KS as an alternative).

Write `talts-2018.md` (per the paper-grounding workflow: a "Verified" admonition line citing the sections checked) and add the bib entries.

**Step 3 — Add deps + lock.** Edit `pyproject.toml`:
```toml
[project.optional-dependencies]
experimental = [
    "blackjax>=1.2,<2",
    "optax>=0.2",
    "arviz>=0.18",      # R-hat, bulk/tail-ESS, BFMI (Task 3)
    "scipy>=1.11",      # chi2/KS uniformity test (Task 6/AC18); CPU-only oracle side
]
```
(matplotlib stays a jaxstroviz concern — gravoturb_fdf figures call jaxstroviz, which already pins matplotlib. Do **not** add matplotlib to progenax core.)

Run: `env -u VIRTUAL_ENV uv lock` then `env -u VIRTUAL_ENV uv lock --check` (expect clean).

**Step 4 — Verify install.** `env -u VIRTUAL_ENV uv pip install -e ".[dev,experimental]"` then `env -u VIRTUAL_ENV uv run --no-sync python -c "import arviz, scipy; print(arviz.__version__, scipy.__version__)"`.

**Step 5 — Commit.**
```bash
git add pyproject.toml uv.lock docs/website/99-bibliography/per-paper/talts-2018.md docs/website/references.bib
git commit -m "deps+docs(gravoturb_fdf): arviz/scipy extras + PDF-grounded Talts 2018 SBC note"
```

> **Gate:** do not start Task 6 (SBC driver) until Task 0's Talts verification is committed.

---

## Task 1: Explicit proper priors — `inference/priors.py`

**Why:** SBC requires drawing θ from a *proper* prior; today's HMC has only the bounded reparam + `log_jacobian` (implicitly flat-in-θ = improper). Proper priors are themselves a hardening step, and the α prior must respect the POT-valid bound.

**Files:**
- Create: `src/experimental/gravoturb_fdf/inference/priors.py`
- Test: `tests/experimental/unit/test_priors.py`
- Modify (export): `src/experimental/gravoturb_fdf/inference/__init__.py`

**Design (Option B — Anna-approved 2026-06-06):** free params `(M, α, β)`, `b` fixed (NOT a prior field — `b` is a likelihood constant). The prior is **unconditional**; POT-validity is handled *per trial* by the SBC driver (Task 6), which sets `s_thr = s_t(θ*)+margin` and relies on the existing `pot_validity_barrier`, exactly as AC16 does. So the prior carries **no** `s_thr` / `α_hi(M)` coupling (this avoids the high-Mach α-range collapse and matches the existing engine).
- `M ~ LogUniform[m_lo, m_hi]`  (default 2, 20)
- `α ~ Uniform[α_lo, α_hi]`  (default 1.1, 4.0 — physical BM19 tail slopes; α>1 for mass-integral convergence)
- `β ~ LogUniform[β_lo, β_hi]`  (default 2, 11/3 ≈ 3.667)

**Step 1 — Write the failing test** (`tests/experimental/unit/test_priors.py`):
```python
import jax, jax.numpy as jnp, pytest
from gravoturb_fdf.inference.priors import BM19Prior

pytestmark = pytest.mark.experimental

def _prior(**kw):
    return BM19Prior(m_range=(2.0, 20.0), alpha_range=(1.1, 4.0),
                     beta_range=(2.0, 11/3), **kw)

def test_sample_within_support():
    pr = _prior()
    keys = jax.random.split(jax.random.PRNGKey(0), 2000)
    thetas = jax.vmap(pr.sample)(keys)          # (2000, 3) = (M, alpha, beta)
    M, alpha, beta = thetas[:, 0], thetas[:, 1], thetas[:, 2]
    assert jnp.all((M >= 2.0) & (M <= 20.0))
    assert jnp.all((alpha >= 1.1) & (alpha <= 4.0))
    assert jnp.all((beta >= 2.0) & (beta <= 11/3 + 1e-9))

def test_logdensity_finite_inside_minus_inf_outside():
    pr = _prior()
    th_in = jnp.array([5.0, 2.0, 3.0])           # (M, alpha, beta) all in range
    assert jnp.isfinite(pr.logpdf(th_in))
    for bad in (jnp.array([1.0, 2.0, 3.0]),      # M below range
                jnp.array([5.0, 0.9, 3.0]),      # alpha below range
                jnp.array([5.0, 4.5, 3.0]),      # alpha above range
                jnp.array([5.0, 2.0, 5.0])):     # beta above range
        assert pr.logpdf(bad) == -jnp.inf

def test_logpdf_grad_finite_inside():
    pr = _prior()
    g = jax.grad(lambda th: pr.logpdf(th))(jnp.array([5.0, 2.0, 3.0]))
    assert jnp.all(jnp.isfinite(g))

def test_sampled_M_loguniform_smoke():
    # log-uniform M => CDF values ~ Uniform(0,1) (sanity for the inverse-CDF sampler)
    pr = _prior()
    keys = jax.random.split(jax.random.PRNGKey(1), 5000)
    M = jax.vmap(pr.sample)(keys)[:, 0]
    u = (jnp.log(M) - jnp.log(2.0)) / (jnp.log(20.0) - jnp.log(2.0))
    us = jnp.sort(u); emp = jnp.arange(1, us.size + 1) / us.size
    assert jnp.max(jnp.abs(emp - us)) < 0.05
```

**Step 2 — Run, watch it fail:** `PYTHONPATH=src:src/experimental … pytest tests/experimental/unit/test_priors.py -q` → ImportError / FAIL.

**Step 3 — Implement** `priors.py` as an `eqx.Module` `BM19Prior` (fields: `m_range`, `alpha_range`, `beta_range`) with `sample(key)->(M,α,β)` (inverse-CDF: log-uniform for M and β; uniform for α) and `logpdf(theta)->scalar` (sum of per-param log-densities; `-inf` outside the box support). Use the **double-`where` trick** so `logpdf` is finite and differentiable *inside* the support (the `-inf` branch is for out-of-support points only; HMC stays inside via the per-trial `pot_validity_barrier`). **No `b` / `s_thr` / `α_hi(M)`** — the prior is unconditional (Option B); per-trial POT-validity lives in the SBC driver (Task 6).

**Step 4 — Run, watch it pass.** Then released-core invariant check.

**Step 5 — Commit:** `git commit -m "feat(gravoturb_fdf): proper (M,alpha,beta) priors, POT-valid alpha bound (SBC Task 1)"`

---

## Task 2: Multi-chain NUTS with per-step diagnostics — extend `hmc.py`

**Why:** AC19 needs ≥4 chains and per-step `info` (divergences, tree depth, energy for BFMI). Current `run_nuts` runs one chain and throws `info` away. Add a sibling that keeps both — do **not** break `run_nuts` (ac16 depends on it).

**Files:**
- Modify: `src/experimental/gravoturb_fdf/inference/hmc.py` (add `run_nuts_diagnostic`)
- Test: `tests/experimental/unit/test_inference.py` (append)

**Step 1 — Failing test** (append to `test_inference.py`):
```python
def test_run_nuts_diagnostic_shapes_and_recovers_gaussian():
    import blackjax  # noqa
    from gravoturb_fdf.inference.hmc import run_nuts_diagnostic
    # 2-D unit Gaussian target
    logdensity = lambda x: -0.5 * jnp.sum(x ** 2)
    out = run_nuts_diagnostic(
        logdensity, jnp.zeros(2), jax.random.PRNGKey(0),
        n_warmup=300, n_samples=400, n_chains=4)
    assert out["positions"].shape == (4, 400, 2)        # (chains, samples, dim)
    assert out["divergences"].shape == (4, 400)         # bool per step
    assert out["tree_depth"].shape == (4, 400)
    assert out["energy"].shape == (4, 400)
    m = out["positions"].reshape(-1, 2).mean(0)
    s = out["positions"].reshape(-1, 2).std(0)
    assert jnp.all(jnp.abs(m) < 0.15) and jnp.all(jnp.abs(s - 1.0) < 0.15)
    assert out["divergences"].mean() < 0.02             # healthy target → few divergences
```

**Step 2 — Run, watch fail** (ImportError on `run_nuts_diagnostic`).

**Step 3 — Implement** `run_nuts_diagnostic(logdensity_fn, init_position, key, n_warmup, n_samples, n_chains=4)`:
- split `key` into `n_chains`; **disperse** inits (e.g. `init + 0.5*N(0,1)` per chain, distinct fold-in).
- per chain: `window_adaptation` warmup → scan the NUTS kernel keeping `info`; collect `state.position`, `info.is_divergent`, `info.num_trajectory_expansions` (tree depth proxy), `info.energy`.
- `jax.vmap` the per-chain function over the `n_chains` keys/inits → stack to `(n_chains, n_samples, …)`.
- Return a dict `{"positions", "divergences", "tree_depth", "energy", "step_size", "max_tree_depth"}` (read the exact blackjax 1.x `info` field names against the installed version; the test pins the contract).

**Step 4 — Run, watch pass.** Released-core check.

**Step 5 — Commit:** `git commit -m "feat(gravoturb_fdf): run_nuts_diagnostic — multi-chain NUTS + per-step info (AC19 Task 2)"`

---

## Task 3: HMC convergence diagnostics — `inference/diagnostics.py`

**Why:** turn the multi-chain output into R̂, bulk/tail-ESS, divergence rate, BFMI, max-tree-depth saturation (a thin arviz wrapper; numpy/arviz side, not JAX-hot).

**Files:**
- Create: `src/experimental/gravoturb_fdf/inference/diagnostics.py`
- Test: `tests/experimental/unit/test_diagnostics.py`

**Step 1 — Failing test:**
```python
import numpy as np, pytest
from gravoturb_fdf.inference.diagnostics import compute_hmc_diagnostics
pytestmark = pytest.mark.experimental

def test_diagnostics_on_iid_gaussian_chains():
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((4, 1000, 3))          # 4 well-mixed chains
    diag = compute_hmc_diagnostics(
        positions, divergences=np.zeros((4, 1000), bool),
        energy=rng.standard_normal((4, 1000)),
        tree_depth=np.full((4, 1000), 5), max_tree_depth=10,
        param_names=["M", "alpha", "beta"])
    assert np.all(diag["r_hat"] < 1.01)                    # iid → R-hat ~ 1
    assert np.all(diag["ess_bulk"] > 400)
    assert np.all(diag["ess_tail"] > 400)
    assert diag["divergence_rate"] == 0.0
    assert 0.3 < diag["bfmi"] < 3.0
    assert diag["max_tree_depth_saturation"] == 0.0        # depth 5 < 10
    assert diag["passed"] is True

def test_diagnostics_flags_stuck_chains():
    positions = np.concatenate([np.zeros((2, 500, 1)), np.ones((2, 500, 1))], axis=0)
    diag = compute_hmc_diagnostics(positions, param_names=["x"])
    assert np.any(diag["r_hat"] > 1.1)                     # disagreeing chains
    assert diag["passed"] is False
```

**Step 2 — Run, watch fail.**

**Step 3 — Implement** `compute_hmc_diagnostics(positions, *, divergences=None, energy=None, tree_depth=None, max_tree_depth=None, param_names=None, r_hat_max=1.01, ess_min=400.0)`:
- build `az.InferenceData` (`posterior` from `positions`, `sample_stats` from `energy`/`diverging`).
- `az.rhat`, `az.ess(method="bulk")`, `az.ess(method="tail")`; `divergence_rate = mean(divergences)`; `bfmi = az.bfmi(energy)` (or the E-BFMI formula); `max_tree_depth_saturation = mean(tree_depth >= max_tree_depth)`.
- `passed = all(r_hat < r_hat_max) and all(ess_* > ess_min) and divergence_rate < 0.01 and saturation < 0.01`.
- Return a dict (per AC convention). Ground R̂/ESS thresholds against arviz docs / Vehtari 2021 in the docstring.

**Step 4 — Pass + released-core check. Step 5 — Commit:** `git commit -m "feat(gravoturb_fdf): HMC convergence diagnostics (R-hat/ESS/divergences/BFMI) via arviz (AC19 Task 3)"`

---

## Task 4: jaxstroviz analysis — `compute_sbc_rank_histogram`

**Repo:** `~/projects/jaxstro-dev/jaxstroviz` (separate package; its own commits). DRY: reuse existing `compute_coverage_probability`; add only the new SBC rank computation.

**Files:**
- Modify: `src/jaxstroviz/analysis/inference.py` (+ `__all__`)
- Test: `tests/unit/analysis/test_inference.py` (append)

**Step 1 — Failing test:**
```python
def test_compute_sbc_rank_histogram_uniform_for_calibrated():
    import numpy as np
    from jaxstroviz.analysis.inference import compute_sbc_rank_histogram
    rng = np.random.default_rng(0)
    K, L, P = 400, 100, 2
    # calibrated: theta_true and posterior draws from the SAME N(0,1) → rank ~ Uniform
    thetas_true = rng.standard_normal((K, P))
    posterior = rng.standard_normal((K, L, P))
    out = compute_sbc_rank_histogram(thetas_true, posterior, n_bins=20,
                                     param_names=["a", "b"])
    assert out["ranks"].shape == (K, P)
    assert out["hist"].shape == (P, 20)
    # chi^2 uniformity p-value should be unsurprising (not reject) for calibrated data
    assert np.all(np.asarray(out["p_value"]) > 0.01)
```

**Step 2 — Run, watch fail** (in the jaxstroviz repo).

**Step 3 — Implement** `compute_sbc_rank_histogram(thetas_true, posterior_samples, *, n_bins=None, param_names=None)`:
- `ranks[k,p] = sum_l (posterior[k,l,p] < thetas_true[k,p])` ∈ {0..L} — the Talts statistic (Task 0-verified).
- histogram each param's ranks into `n_bins` (default a divisor of `L+1`); χ² of counts vs expected `K/n_bins` → `p_value` per param via `scipy.stats.chisquare`.
- Return dict `{"ranks", "hist", "bin_edges", "p_value", "n_trials", "n_draws", "param_names"}`. Document the calibration interpretation + the under-model-only caveat.

**Step 4 — Pass. Step 5 — Commit (jaxstroviz):** `git commit -m "feat(analysis): compute_sbc_rank_histogram (SBC rank statistic, Talts 2018)"`

---

## Task 5: jaxstroviz plot — `plot_sbc_rank_histogram`

**Files:**
- Modify: `src/jaxstroviz/plots/inference.py` (+ `__all__`)
- Test: `tests/unit/plots/test_inference.py` (append)

**Step 1 — Failing test** (axes-first convention, Agg backend):
```python
def test_plot_sbc_rank_histogram_draws_bars_and_band():
    import numpy as np, matplotlib.pyplot as plt
    from jaxstroviz.plots.inference import plot_sbc_rank_histogram
    ranks = np.random.default_rng(0).integers(0, 101, size=400)
    fig, ax = plt.subplots()
    plot_sbc_rank_histogram(ax, ranks, n_draws=100, param_name="alpha",
                            n_bins=20, show_expected_band=True)
    assert len(ax.patches) >= 20            # histogram bars
    assert ax.get_xlabel() != ""            # labeled
    plt.close(fig)
```

**Step 2 — Run, watch fail.**

**Step 3 — Implement** `plot_sbc_rank_histogram(ax, ranks, *, n_draws, param_name=None, n_bins=20, color=None, show_expected_band=True, **kwargs) -> None`: histogram of `ranks` into `n_bins`; horizontal **uniform expectation band** (binomial 99% interval around `K/n_bins`) as a shaded `axhspan`/`fill_between`; label axes ("Rank statistic" / "Count"); use `PALETTE`/`to_numpy`. Axes-first, returns `None`.

**Step 4 — Pass. Step 5 — Commit (jaxstroviz):** `git commit -m "feat(plots): plot_sbc_rank_histogram with uniform expectation band"`

> After Tasks 4–5: `cd ~/projects/jaxstro-dev/jaxstroviz && uv run --no-sync pytest tests/ -q` green; reinstall jaxstroviz into progenax's env if path-pinned so the new funcs import.

---

## Task 6: SBC driver — `inference/sbc.py` (GATED on Task 0)

**Why:** the calibration loop. Per trial: θ*~prior → simulate the mock (rank-copula field → POT exceedances + CIC counts) → build the **prior-aware** logdensity → `run_nuts` → rank θ* among posterior draws. Calibrated ⇔ ranks ~ Uniform.

**Files:**
- Create: `src/experimental/gravoturb_fdf/inference/sbc.py`
- Test: `tests/experimental/unit/test_sbc.py`

**Key composition (prior-aware logdensity — the careful bit):**
```
logdensity(z) = tail_exceedance_loglike(...) + Σ count_loglike(...)
              + prior.logpdf(theta(z))        # NEW: proper prior (replaces flat-in-theta)
              + pot_validity_barrier(theta, s_thr)
              + log_jacobian(z)               # KEEP: reparam Jacobian is still required
```
Factor this into a shared `build_logdensity(prior, data, b, s_thr, s_max, shape, cell_sizes, ...)` used by both SBC and (later, carefully) AC16. **theta assembly:** `(M, α, β) = to_constrained(z)`; the likelihood/barrier take the 4-vector `[M, b, α, β]` (b fixed), while `prior.logpdf` takes the 3 free params `[M, α, β]`.

**Step 1 — Failing test** (small, fast — a reduced "is the machinery wired" check; the heavy uniformity assertion lives in AC18/Task 7):
```python
import jax, jax.numpy as jnp, pytest
from gravoturb_fdf.inference.sbc import sbc_ranks
from gravoturb_fdf.inference.priors import BM19Prior
pytestmark = [pytest.mark.experimental, pytest.mark.slow]

def test_sbc_ranks_shape_and_support():
    pr = BM19Prior()                              # unconditional (M, alpha, beta)
    out = sbc_ranks(pr, key=jax.random.PRNGKey(0), n_trials=6,
                    b=0.4, s_thr_margin=0.75,     # per-trial s_thr = s_t(theta*)+margin
                    shape=(24, 24, 24), density_shape=(48, 48, 48),
                    n_warmup=80, n_samples=120, n_thin=4,
                    cell_sizes=(2, 4), n_stars=4.0e4)
    assert out["ranks"].shape == (6, 3)                 # (trials, params=M,alpha,beta)
    L = out["n_draws"]                                  # thinned draws
    assert jnp.all((out["ranks"] >= 0) & (out["ranks"] <= L))
    assert out["param_names"] == ["M", "alpha", "beta"]
```

**Step 2 — Run, watch fail.**

**Step 3 — Implement** `sbc_ranks(prior, key, n_trials, b, s_thr_margin, shape, density_shape, n_warmup, n_samples, n_thin, cell_sizes, n_stars) -> dict`:
- loop trials (python loop OK — oracle/validation side; each trial is a full NUTS run): `θ* = prior.sample(k)` → `(M*, α*, β*)`; **set the per-trial POT threshold (Option B)** `s_thr = float(transition_density(α*, sigma_s_squared(M*, b))) + s_thr_margin`; realize gas field `rank_copula_field(gaussian_random_field(density_shape, β*, …), M*, b, α*)`; `measure_exceedances(s_field, s_thr)` → `(exc_counts, exc_edges, s_max, n_tail)`; `s_lo` field + `sample_cic_counts` per cell size; `build_logdensity(prior, …, b, s_thr, s_max, …)`; `run_nuts`; **thin** posterior to ~independent (`[::n_thin]`); `rank = Σ_l (θ_l < θ*)` per free param (M, α, β).
- Return `{"ranks", "n_draws", "n_trials", "param_names", "thetas_true"}`.
- Parallelize trials across devices later if needed; for now a python loop with fold-in keys (deterministic, resumable).

**Step 4 — Run, watch pass** (slow). Released-core check (unaffected). **Step 5 — Commit:** `git commit -m "feat(gravoturb_fdf): SBC driver sbc_ranks + prior-aware build_logdensity (Task 6)"`

---

## Task 7: AC18 — SBC rank uniformity (acceptance + test + wire main)

**Files:**
- Modify: `src/experimental/gravoturb_fdf/validation/acceptance.py` (add `ac18_sbc_rank_uniformity`, wire into `main`)
- Test: `tests/experimental/validation/test_acceptance.py` (append)

**Step 1 — Failing test:**
```python
@pytest.mark.slow
def test_ac18_sbc_rank_uniformity():
    res = acceptance.ac18_sbc_rank_uniformity(
        n_trials=30, shape=(24,)*3, density_shape=(64,)*3,
        n_warmup=120, n_samples=200, n_thin=4)
    assert res["passed"]
    assert all(p > 0.05 for p in res["p_value"])        # per-param uniformity not rejected
    assert len(res["p_value"]) == 3
```

**Step 2 — Run, watch fail.**

**Step 3 — Implement** `ac18_sbc_rank_uniformity(...) -> dict`: call `sbc_ranks`, then `compute_sbc_rank_histogram` (jaxstroviz) for the χ² `p_value` per param; print via `_header`/`_row`; `passed = all(p > 0.05)`. Return `{"passed", "p_value", "n_trials", "n_draws", "ranks"}`. Wire into `main()`'s `results` dict (full config: `n_trials≈128`, `density_shape=(96,)³`, `n_warmup=300`, `n_samples=600`, slow). **State the under-model-only caveat** in the printed header + docstring.

**Step 4 — Pass. Step 5 — Commit:** `git commit -m "feat(gravoturb_fdf): AC18 SBC rank-uniformity (calibration under BM19) wired into acceptance"`

---

## Task 8: AC19 — HMC convergence (acceptance + test + wire main)

**Files:**
- Modify: `validation/acceptance.py` (add `ac19_hmc_convergence`, wire into `main`)
- Test: `tests/experimental/validation/test_acceptance.py` (append)

**Step 1 — Failing test:**
```python
@pytest.mark.slow
def test_ac19_hmc_convergence():
    res = acceptance.ac19_hmc_convergence(
        density_shape=(96,)*3, shape=(24,)*3,
        n_warmup=300, n_samples=500, n_chains=4)
    assert res["passed"]
    assert max(res["r_hat"]) < 1.01
    assert min(res["ess_bulk"]) > 400
    assert res["divergence_rate"] < 0.01
```

**Step 2 — Run, watch fail.**

**Step 3 — Implement** `ac19_hmc_convergence(...) -> dict`: build the AC16 mock + prior-aware logdensity once; `run_nuts_diagnostic` at the production config with `n_chains≥4` dispersed inits; `compute_hmc_diagnostics`; print per-param R̂/ESS + divergence rate; `passed` per thresholds (R̂<1.01, ESS_bulk/tail>400, div<1%, saturation<1%). Return the diag dict + `"passed"`. Wire into `main()`.

**Step 4 — Pass. Step 5 — Commit:** `git commit -m "feat(gravoturb_fdf): AC19 HMC convergence (R-hat/ESS/divergences, >=4 chains) wired into acceptance"`

---

## Task 9: Render figures + docs sweep + counts

**Files:**
- Create: `src/experimental/gravoturb_fdf/validation/figures.py` (compute→jaxstroviz render→save to `validation/plots/`)
- Create PNGs: `src/experimental/gravoturb_fdf/validation/plots/*.png` (or the repo's plots dir)
- Modify: `src/experimental/gravoturb_fdf/VALIDATION_SUMMARY.md`, `progenax/CLAUDE.md` (test/AC counts from a real run), `docs/website/10-theory/gravoturbulence/differentiable-inference.md` (add the SBC/diagnostics section + figures), `docs/plans/2026-06-06-trustworthiness-sbc-validation-design.md` (mark ① done).

**Figure set** (all via jaxstroviz funcs, `set_paper()` + `newfig`/`gridfig` + `savefig`): SBC rank histograms (per param, `plot_sbc_rank_histogram`) · empirical coverage curve (`compute_coverage_probability`) · recovery scatter across the prior (`plot_recovery_correlation`) · a representative (M,α,β) corner showing the broken degeneracy (`plot_posterior_2d`) · σ(α)-vs-N_tail (AC17 as a figure). gravoturb_fdf validation **calls** jaxstroviz (a `[viz]`-side dep); progenax core never imports it.

**Step 1 — Implement `figures.py`** with a `render_sbc_figures(out_dir)` that computes arrays (reusing `sbc_ranks`/`compute_hmc_diagnostics`/AC17) and renders via jaxstroviz. Guard the jaxstroviz import (`pytest.importorskip`-style / optional) so released-core never needs it.

**Step 2 — Run the acceptance driver** end-to-end; capture AC1–AC19 PASS output as evidence.
```bash
PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -m gravoturb_fdf.validation.acceptance
```

**Step 3 — Update counts** in `VALIDATION_SUMMARY.md` + `CLAUDE.md` from a real `pytest tests/experimental -q` and released-core run (don't guess numbers). Add the SBC section to the website chapter with the rendered figures and the honest-scope paragraph.

**Step 4 — Final verification (paste all):** experimental suite green, released-core 815 green, acceptance AC1–AC19 PASS, jaxstroviz tests green, MyST docs build clean.

**Step 5 — Commit:** `git commit -m "docs+figures(gravoturb_fdf): SBC/diagnostics figures, AC1-AC19 summary, website chapter, counts"`

---

## Figure Gallery — Sections A–C (brainstormed + validated with Anna 2026-06-06)

**Scope expansion (Anna):** the figures must *visually verify every forward-model + inference
component reproduces its analytic/oracle target* — the visual companion to the AC1–AC19 numeric
checks — not just the inference plots. **Decision:** the research-fresh plotters live in a new
**jaxstroviz `experimental/` tier** (mirrors progenax `src/experimental`: not in the released
jaxstroviz wheel, importable under an editable install), on a matching jaxstroviz feature branch
**`gravoturb-fdf-sbc-figures`**. This supersedes the old Tasks 4–5 (which were SBC-only).

### A. jaxstroviz `experimental/` tier
```
jaxstroviz/src/jaxstroviz/experimental/
├── __init__.py
├── analysis/  (compute: arrays/metrics in → dict out; no matplotlib)
│   ├── sbc.py        compute_sbc_rank_histogram(thetas_true, posterior, n_bins, param_names)
│   ├── spectrum.py   compute_radial_power_spectrum(field) -> (k, P_k)
│   └── residual.py   compute_overlay_residual(x, data, model) -> dict(frac_resid, ...)
└── plots/     (axes-first: plot_*(ax, …) -> None; reuse styles.figures + PALETTE; NO gravoturb_fdf import)
    ├── sbc.py        plot_sbc_rank_histogram(ax, ranks, *, n_draws, n_bins, n_trials, show_band)
    ├── pdf.py        plot_density_pdf_overlay(ax, s, hist, p_model, *, s_t); plot_density_pdf_models(ax, s, {label: p(s)}, *, s_t_map)
    ├── spectrum.py   plot_power_spectrum(ax, k, P_k, *, beta)
    ├── counts.py     plot_counts_distribution(ax, n, P_mock, P_pred, *, n_bar)
    ├── tail.py       plot_tail_exceedances(ax, edges, counts, *, alpha_hat, s_t, s_thr, s_max)
    ├── forecast.py   plot_sigma_alpha_ladder(ax, n_tail, sigma_emp, sigma_fisher, *, slope)
    └── hmc.py        plot_hmc_trace(ax, samples, *, param_name); plot_hmc_rank(ax, samples, *, n_bins)
```
- Packaging: exclude `jaxstroviz.experimental` from the wheel (mirror progenax). `scipy` added to a
  jaxstroviz `[experimental]` extra (χ²); matplotlib already core.
- Tests: `jaxstroviz/tests/experimental/` — Agg backend, assert on axes state (lines/patches/labels),
  `plt.close`. TDD per plotter.

### B. The 14-figure gallery + the validation standard
**Standard (every figure):** (a) overlay realized/oracle **vs** analytic prediction; (b) a
residual / fractional-error panel; (c) annotate the AC pass/fail + key metric (e.g. "AC11 max
rel 0.08 % < 0.5 % ✓"). The figure visually demonstrates the AC passing.

Forward model: (1) density PDF AC1/AC6; (2) GRF P(k)~k^−β; (3) ξ_s(r) + Hermite convergence AC11;
(4) Limber w(r⊥) AC12; (5) CIC P(N) **and** Var(N)=N̄+N̄²ξ̄ AC13; (6) grad AD-vs-FD AC14;
(7) Fisher σ(M,α,β)+correlation AC15. Inference: (8) POT tail truncated-exp + s_t/s_thr/s_max
AC16/17; (9) (M,α,β) corner AC16; (10) HMC traces+R̂/ESS AC19; (11) SBC rank histograms AC18;
(12) σ(α)–N_tail ladder AC17; (13) summary dashboard (PASS/FAIL tile per component).
Plus (14a) **BM19 PDF-model comparison** (pure-LN vs LN+PL across (M,α,b), s_t marked — experimental)
and (14b) **spatial-profile comparison** (Plummer/King/EFF/Michie ρ(r) — RELEASED core, reuse
released `plots/profiles.py`).

Reuse released jaxstroviz where it fits: `plot_two_point_correlation` (3), `plot_profile_validation`
(4), `plot_parameter_recovery`/`plot_posterior_2d` (9), `plot_fisher_correlation`/
`plot_marginalized_uncertainties` (7), `plots/profiles` (14b).

### C. Revised figure tasks (replace old Tasks 4–5; each RED→GREEN + code-review, jaxstroviz branch)
- **F1** — jaxstroviz `experimental/` scaffold (package, packaging-exclude, `[experimental]`+scipy,
  `tests/experimental/`) **+ SBC pair** (`analysis/sbc.compute_sbc_rank_histogram` per the
  Talts-grounded rank stat {eq}`talts-rank`; `plots/sbc.plot_sbc_rank_histogram` with the
  Binomial(N,1/B) 0.5–99.5 % band). RED tests: rank uniform for calibrated mock → χ² p>0.01;
  plot draws ≥B bars + the band.
- **F2** — forward-model plotters: `plots/pdf` (overlay + **model comparison**), `plots/spectrum`
  (+ `analysis/spectrum`), `plots/counts`, `plots/tail`. Per-plotter RED test asserts the overlay
  line + residual axis + the marked walls/transition exist.
- **F3** — inference plotters: `plots/forecast` (σ–N_tail log-log + √N slope), `plots/hmc`
  (trace + rank), `analysis/residual`. RED tests assert axes content.
- **F4** — released spatial-profile comparison: `plot_density_profile_comparison` (released
  `plots/profiles.py`) overlaying Plummer/King/EFF/Michie ρ(r); RED test in jaxstroviz released tests.

Then **Task 6** (SBC driver, now the SBC analysis/plot exist) → **7–8** (AC18/AC19) →
**Task 9** (gravoturb_fdf `validation/figures.py` orchestrator: compute arrays from AC dicts +
theory/field/measure, call the jaxstroviz plotters, save to `validation/plots/fdf/`,
import-guarded; render all 14; docs sweep; AC counts; completion doc).

## Completion (the project's Definition of Complete)

1. **Tests:** `test_priors.py`, `test_diagnostics.py`, `test_sbc.py`, appended `test_inference.py`/`test_acceptance.py` — all green (experimental); released-core 815 untouched.
2. **Validation:** AC18 (SBC uniformity p>0.05/param) + AC19 (R̂<1.01, ESS>400, div<1%) print PASS in `main()`.
3. **Figures:** SBC rank histograms + coverage + recovery + corner + σ(α)-N_tail saved under `validation/plots/`.
4. **Quantitative results:** the acceptance driver's AC18/AC19 tables (p-values, R̂, ESS).
5. **Completion doc:** `.claude-work/WS1_SBC_COMPLETE.md` — files/API, SBC results + plot refs, the under-model-only caveat, test summary, lessons, and the handoff to ② (3-pt null test).

**Nothing is pushed and no PR is opened without Anna's explicit go.**
