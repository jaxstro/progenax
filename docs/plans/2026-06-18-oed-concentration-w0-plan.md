# W₀-OED Concentration Demo — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> superpowers:subagent-driven-development) to implement this plan task-by-task.

**Goal:** A c-/D-optimal optimal-experimental-design demo headlining **W₀ = concentration** —
"where to spend a star budget (radial bins × {RV, PM_R, PM_T}) to best constrain W₀" — built on
the Stage-1 additive-Fisher backbone, for **King** (headline OM-King) and **Michie**, validated
by a real-star calibration gate + an AD-vs-FD gate on ∂σ/∂lnW₀.

**Architecture:** Scripts-only (`scripts/_demo_oed_concentration.py` core +
`scripts/demo_oed_concentration.py` CLI), reusing the `scripts/_demo_oed.py` Stage-1 helpers
(additive Fisher `F = Σ n·c·M` in the ln-θ metric, c/D/A criteria, optax optimizer) and the
Stage-2 ln-θ Gauss-Newton fitter (`scripts/_demo_oed_depth._fit_theta_gn`). The only new physics
is: (a) building King/Michie profiles from `theta=(W₀, r_a, M)` and projecting them under OM via
`project_dispersion`; (b) assembling an OM-King / OM-on-Michie-density particle sampler so the
calibration's sampler ≡ the Fisher-model. **No `src/progenax/` change** → coverage/dashboard
staleness gate is unaffected.

**Tech Stack:** JAX (`jax.jacrev`, `jax.vmap`, reverse-mode only — King/Michie custom_vjp ODEs
have no jvp rule), Equinox profiles, optax, `progenax.project_dispersion`,
`progenax.KingProfile`/`MichieProfile`, `progenax.kinematics.eddington` (+ evaluate Engine B
`MultiComponentCluster.from_density_profiles` as the preferred sampler).

**Design doc:** `docs/plans/2026-06-18-oed-concentration-w0-design.md` (read it first).

**Conventions for every task:**
- Run the FAST gate after each task:
  `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_demo_oed_concentration.py -q`
- Commit per task; stage files **explicitly by name** (never `git add -A`). Trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Never weaken a test/tolerance to pass — fix the root cause.
- Mock truth (analog of Stage-1's `MOCK`): `W0=6.0, r_a=6.0, M=1e5, r_c=1.0, d_kpc=4.0,
  eps_RV_kms=1.0, eps_PM_masyr=0.05`. `theta=(W0, r_a, M)`; index map **W0=0 (TARGET), r_a=1,
  M=2**. `r_c≡1.0` is the length unit. `R_BINS = jnp.logspace(jnp.log10(0.3), jnp.log10(12.0),
  12)` (r_c units); both King and Michie r_t at (W0=6, r_a=6) exceed ~20 r_c, so every bin is
  bound (verify in Task 2).

---

## Task 1: OM particle sampler (de-risk first — highest implementation risk)

**Why first:** the whole real-star gate rests on a sampler whose dispersion equals what
`project_dispersion(profile, r_a)` predicts (sampler ≡ Fisher-model). Prove this before anything
else; if it cannot be made to hold, the gate design must change.

**Files:**
- Create: `scripts/_demo_oed_concentration.py` (start the module; add `sample_om_cluster`)
- Test: `tests/unit/test_demo_oed_concentration.py`

**Step 1 — Investigate the cleanest sampler (read, don't guess):**
1. Read `src/progenax/cluster/` for `MultiComponentCluster.from_density_profiles` — it samples
   "prescribed Plummer/EFF/King densities, shared-Ψ Eddington/OM DFs." If it can sample a
   **single-component King (and Michie) density under OM with anisotropy radius r_a**, use it
   directly (preferred — maximal reuse).
2. Else read `src/progenax/kinematics/eff_df.py` for how `EFFVelocityDF` assembles
   `(r, ρ, dρ/dr, Ψ, dΨ/dr)` and calls `eddington_invert(..., r_a=r_a)` +
   `sample_speed_from_f_table` + `assign_om_directions`, and mirror it for King/Michie (whose
   `psi_grid`/`xi_grid` ODE tables give Ψ(r); `density(r)` gives ρ).

**Step 2 — Write the failing test** (sampler ≡ Fisher-model; this is the load-bearing check):

```python
import sys, pathlib
import jax, jax.numpy as jnp, pytest
import progenax  # noqa: F401  enables float64
from jaxstro.units import STELLAR
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import _demo_oed_concentration as oedc  # noqa: E402

@pytest.mark.parametrize("model", ["king", "michie"])
def test_om_sampler_matches_project_dispersion(model):
    """Binned LOS dispersion of the OM sampler must match project_dispersion (the Fisher
    forward model) — otherwise the calibration gate would test mismatched models."""
    key = jax.random.PRNGKey(0)
    W0, r_a, M = 6.0, 6.0, 1e5
    R, v_los, v_pm_r, v_pm_t = oedc.sample_om_cluster(model, W0, r_a, M, n_stars=200_000, key=key)
    # predicted LOS dispersion at a mid radius bin via project_dispersion
    prof = oedc.build_profile(W0, r_a, model)
    R_probe = jnp.array([2.0])
    pred = progenax.project_dispersion(prof, r_a, R_probe, M, STELLAR.G).sigma_los[0]
    # measured: stars in a thin annulus around R=2 r_c
    sel = (R > 1.6) & (R < 2.4)
    meas = jnp.std(v_los[sel], ddof=1)
    assert sel.sum() > 2000, sel.sum()
    assert jnp.abs(meas - pred) / pred < 0.05, (model, float(meas), float(pred))
```

**Step 3 — Run it, expect FAIL** (`sample_om_cluster`/`build_profile` undefined).

**Step 4 — Implement** `build_profile(W0, r_a, model)` (King via `KingProfile.from_W0_rc(W0,
1.0)`; Michie via `MichieProfile.from_W0_rc(W0, 1.0, r_a)`) and `sample_om_cluster(model, W0,
r_a, M, n_stars, key)` (Engine B preferred; else eddington assembly). Return projected per-star
`(R, v_los, v_pm_r, v_pm_t)` via the Stage-1 `project_to_sky` pattern (line of sight = +z).
Velocities in pc/Myr (STELLAR). Reuse `_demo_oed.project_to_sky` by importing it.

**Step 5 — Verify the OM DF is physical** (add an assertion/log): the OM f(Q) must be
non-negative at (W0=6, r_a=6) for both models (Merritt augmented density). If Engine B is used it
guarantees this; if raw `eddington_invert`, check `f_grid.min() > -tol`. If negative, raise r_a
until physical and record the bound in the design doc.

**Step 6 — Run, expect PASS** (both models within 5%). If Michie fails at 5%, investigate
(native-vs-OM confusion — you must sample OM-on-Michie-density, NOT `MichieVelocityDF`).

**Step 7 — Commit** `scripts/_demo_oed_concentration.py tests/unit/test_demo_oed_concentration.py`.

---

## Task 2: Forward model + ln-θ Jacobian (predict_sigma, jacobian_and_sigma)

**Files:** Modify `scripts/_demo_oed_concentration.py`; Test `tests/unit/test_demo_oed_concentration.py`.

**Step 1 — Failing tests:**

```python
def test_predict_sigma_shape_and_bins_bound():
    for model in ("king", "michie"):
        th = oedc.theta_truth()                       # (3,) = (W0, r_a, M)
        sig = oedc.predict_sigma(th, oedc.R_BINS, STELLAR.G, model)
        assert sig.shape == (3, oedc.R_BINS.shape[0])
        assert jnp.all(jnp.isfinite(sig)) and jnp.all(sig > 0)
        prof = oedc.build_profile(th[0], th[1], model)
        assert float(prof.r_t) > float(oedc.R_BINS[-1])   # every bin bound

def test_jacobian_lntheta_shape_and_W0_column_nonzero():
    for model in ("king", "michie"):
        th = oedc.theta_truth()
        J, sig = oedc.jacobian_and_sigma(th, oedc.R_BINS, STELLAR.G, model)
        K = oedc.R_BINS.shape[0]
        assert J.shape == (3, K, 3)                    # channel, bin, param
        assert jnp.all(jnp.isfinite(J))
        assert jnp.any(jnp.abs(J[:, :, 0]) > 0)        # W0 column carries signal
```

**Step 2 — Run, expect FAIL.**

**Step 3 — Implement** (mirror `_demo_oed.predict_sigma` / `jacobian_and_sigma`, add `model`
arg; index map W0=0, r_a=1, M=2; ln-θ scaling `J * theta[None,None,:]`). `theta_truth()` returns
`jnp.array([6.0, 6.0, 1e5])`. Define module constants `MOCK`, `R_BINS`, `EPS` (per-channel
errors via the Stage-1 `kms_to_pcMyr` / `pm_masyr_to_kms` conversions — import them from
`_demo_oed`).

**Step 4 — Run, expect PASS. Step 5 — Commit.**

---

## Task 3: AD-vs-FD gate on ∂σ/∂lnW₀ (King clean; Michie Richardson-FD)

**Files:** Modify `scripts/_demo_oed_concentration.py` (no new src — just a test + maybe a
`richardson_fd` helper); Test `tests/unit/test_demo_oed_concentration.py`.

**Step 1 — Failing tests** (thresholds from the design doc; never weaken):

```python
def test_grad_sigma_W0_king_AD_vs_FD():
    """OM-King ∂σ_los/∂W0 is FD-consistent at every bin (verified ≤3.2e-4 in design)."""
    th = oedc.theta_truth()
    def f(W0):
        return oedc.predict_sigma(th.at[0].set(W0), oedc.R_BINS, STELLAR.G, "king")[0]  # los
    W0 = th[0]; h = 1e-4
    J_ad = jax.jacrev(f)(W0)
    J_fd = (f(W0 + h) - f(W0 - h)) / (2 * h)
    rel = jnp.abs(J_ad - J_fd) / (jnp.abs(J_fd) + 1e-30)
    assert jnp.all(rel < 1e-3), rel

def test_grad_sigma_W0_michie_inner_AD_vs_FD():
    """OM-Michie ∂σ/∂W0 is FD-consistent at R ≲ r_a (fixed-step FD faithful there)."""
    th = oedc.theta_truth()
    R_inner = oedc.R_BINS[oedc.R_BINS <= th[1]]      # R <= r_a
    def f(W0):
        return oedc.predict_sigma(th.at[0].set(W0), R_inner, STELLAR.G, "michie")[0]
    W0 = th[0]; h = 1e-4
    rel = jnp.abs(jax.jacrev(f)(W0) - (f(W0+h)-f(W0-h))/(2*h)) / (jnp.abs((f(W0+h)-f(W0-h))/(2*h)) + 1e-30)
    assert jnp.all(rel < 1e-3), rel

def test_grad_sigma_W0_michie_outer_richardson():
    """At outer radii Michie's r_t(W0) near-divergence makes fixed-step FD a poor proxy
    (design: 8e-3 at R=8). AD is correct: assert AD→FD CONVERGES as h↓ (Richardson), per ADR-0016."""
    th = oedc.theta_truth()
    R_outer = jnp.array([oedc.R_BINS[-1]])           # the outermost bin
    def f(W0):
        return oedc.predict_sigma(th.at[0].set(W0), R_outer, STELLAR.G, "michie")[0, 0]
    W0 = th[0]; J_ad = jax.grad(f)(W0)
    rels = []
    for h in (1e-2, 1e-3, 1e-4):
        J_fd = (f(W0 + h) - f(W0 - h)) / (2 * h)
        rels.append(float(jnp.abs(J_ad - J_fd) / (jnp.abs(J_fd) + 1e-30)))
    assert rels[-1] < rels[0] and rels[-1] < 1e-3, rels   # converges toward AD
```

**Step 2 — Run, expect FAIL** (until predict_sigma exists; if Task 2 done, the Michie-outer test
verifies the Richardson behavior — confirm `rels` is monotone-decreasing; if not, investigate
before adjusting). **Step 3 — Implement** any small helper if needed (likely none). **Step 4 —
Run, expect PASS. Step 5 — Commit.**

---

## Task 4: per-star blocks, additive Fisher, c/D/A, optimizer (mostly reuse)

**Files:** Modify `scripts/_demo_oed_concentration.py`; Test same.

The Stage-1 functions `blocks_from_eps`, `design_counts`, `completeness`, `fisher`,
`c_criterion`, `d_criterion`, `a_criterion`, `optimize_design`, `DesignResult` are
model-agnostic (they consume the per-star blocks `Mb` and a design vector). **Import and reuse
them from `_demo_oed`** — do not duplicate. Add only:
- `per_star_blocks(theta, R_bins, eps, G, model)` → calls the new `jacobian_and_sigma(...,
  model)` then `_demo_oed.blocks_from_eps`.
- `PRIOR_DIAG` for this arc: **prior on M only** (index 2 — the one param with an external
  constraint: integrated light × M/L); **zero on W0 (target) and zero on r_a** (anisotropy is
  constrained by kinematics alone). `PRIOR_DIAG = jnp.array([0.0, 0.0, 1.0/0.3**2])`.

**Step 1 — Failing tests:**

```python
def test_blocks_shape_symmetry_and_fisher_spd():
    for model in ("king", "michie"):
        th = oedc.theta_truth()
        Mb, sig = oedc.per_star_blocks(th, oedc.R_BINS, oedc.EPS, STELLAR.G, model)
        K = oedc.R_BINS.shape[0]
        assert Mb.shape == (3, K, 3, 3)
        assert jnp.allclose(Mb, jnp.swapaxes(Mb, -1, -2), atol=1e-12)
        z = jnp.zeros(3 * K)
        F = oedc.fisher(z, Mb, oedc.completeness(oedc.R_BINS), 1000.0, oedc.PRIOR_DIAG)
        evals = jnp.linalg.eigvalsh(F)
        assert jnp.all(evals > 0), evals          # SPD with M-only prior (data covers W0,r_a)

def test_c_criterion_targets_W0_and_grad_AD_vs_FD():
    for model in ("king", "michie"):
        th = oedc.theta_truth()
        Mb, _ = oedc.per_star_blocks(th, oedc.R_BINS, oedc.EPS, STELLAR.G, model)
        cb = oedc.completeness(oedc.R_BINS); K = oedc.R_BINS.shape[0]
        loss = lambda z: oedc.c_criterion(oedc.fisher(z, Mb, cb, 1000.0, oedc.PRIOR_DIAG), target=0)
        z = jax.random.normal(jax.random.PRNGKey(1), (3*K,)) * 0.5
        g_ad = jax.grad(loss)(z)
        i = 5; eps = 1e-4
        g_fd = (loss(z.at[i].add(eps)) - loss(z.at[i].add(-eps))) / (2*eps)
        assert jnp.allclose(g_ad[i], g_fd, rtol=1e-4, atol=1e-8)
```

**Note (SPD):** if `test_..._spd` fails for some generic z during optimization, the (W0,r_a)
data-block is rank-deficient for that design — document and add a *weak* r_a regularizer
(`PRIOR_DIAG[1] = 1/0.5**2`, labeled a conditioning regularizer, NOT an external constraint).
Default is M-only; only escalate with a recorded reason.

**Step 2 — FAIL. Step 3 — Implement `per_star_blocks` + `PRIOR_DIAG`; re-export reused Stage-1
symbols. Step 4 — PASS. Step 5 — Commit.**

---

## Task 5: real-star @slow calibration gate (both models)

> **SUPERSEDED by ADR-0018 → KING-ONLY.** After two host OOM crashes (the Michie MLE-MC
> reverse-mode-through-ODE fit is ~28 GB even under sequential `jax.lax.map`, and adds no new
> anisotropy physics over King), the calibration was scoped to King only; Michie's model is
> validated by the cheaper sampler-match/forward/W₀-gradient tests. The fit also became
> Levenberg–Marquardt-damped with a W₀-target convergence witness (review I1). See ADR-0018.

**Files:** Modify `scripts/_demo_oed_concentration.py` (add `calibrate_fisher_W0`); Test same.

Mirror `_demo_oed.calibrate_fisher` but: (a) draw mocks with `sample_om_cluster(model, ...)`
(Task 1); (b) fit θ in the **ln-θ Gauss-Newton metric** using
`_demo_oed_depth._fit_theta_gn` (import it) — NOT physical-Adam (Stage-2 lesson: physical-Adam
pins large-scale params; W0~6 is well-scaled but M~1e5 is not); (c) collect **lnŴ₀** and compare
`Var(lnŴ₀)` to the Fisher `(F⁻¹)_{W0,W0}` (already a fractional/ln variance in the ln-θ metric).
Reuse `_demo_oed._r_bin_edges`, `_binned_sigma_hat` (import; they bin by R and subsample design
counts — model-agnostic given the channels).

**Step 1 — Failing @slow test:**

```python
@pytest.mark.slow
@pytest.mark.parametrize("model", ["king", "michie"])
def test_W0_fisher_calibration_matches_realized_scatter(model):
    key = jax.random.PRNGKey(7)
    K = oedc.R_BINS.shape[0]
    cal = oedc.calibrate_fisher_W0(z=jnp.zeros(3*K), N_total=400.0, n_draws=48, key=key, model=model)
    band = 2.0 * (2.0 / 48) ** 0.5                 # MC band on a variance from 48 draws (~0.41)
    ratio = cal.realized_var_W0 / cal.fisher_var_W0
    assert jnp.abs(ratio - 1.0) < band, (model, ratio)
```

**Step 2 — FAIL. Step 3 — Implement `calibrate_fisher_W0` + `CalibResultW0(realized_var_W0,
fisher_var_W0)`.** Size the parent catalog so the thinnest outer bin holds ≫ its design count
(reuse `_demo_oed`'s `n_parent = max(8000, 4*N_total)` logic; raise if a bin underflows). **Step
4 — Run** (this is @slow; run explicitly:
`... pytest tests/unit/test_demo_oed_concentration.py -q -m slow`). Expect PASS within the band,
no significant bias. If biased, root-cause (fitter metric, SE formula, sampler mismatch) — do NOT
widen the band. **Step 5 — Commit.**

---

## Task 6: CLI demo + figures + the science result

**Files:** Create `scripts/demo_oed_concentration.py` (gated CLI); Modify
`scripts/_demo_oed_concentration.py` (add `optimal_allocation_summary`); Test same (smoke).

The CLI: optimize c/D/A designs for both models, render figures to a `--outdir`, print a
quantitative summary, exit 0. Figures (publication-quality, labeled): (1) W₀ optimal allocation
map (radius × channel) for King; (2) same for Michie; (3) **side-by-side W₀-vs-r_a allocation**
(load Stage-1's r_a result from `_demo_oed` to contrast — the pre-registered hypothesis test);
(4) c/D/A criterion comparison + equal-precision star-factor; (5) the `F_{W0,r_a}` off-diagonal /
correlation quantifying the degeneracy.

**Science check (the pre-registered H1):** compute, for the W₀ c-optimal design, (a) the
core-vs-outskirts weight split and (b) the channel balance (PM-fraction vs radius), and compare
to Stage-1's r_a design. Print whether the result supports **H1** (W₀ more core/balanced) or
**H0** (W₀ resembles r_a). A wrong prediction is a finding — report it honestly.

**Step 1 — Failing smoke test:**

```python
def test_cli_concentration_smoke(tmp_path):
    import importlib
    cli = importlib.import_module("demo_oed_concentration")
    rc = cli.main(["--outdir", str(tmp_path), "--n-starts", "2", "--n-steps", "60",
                   "--calib-draws", "0", "--quick"])   # dial cost down; keep gates real
    assert rc == 0
    figs = list(tmp_path.glob("*.png"))
    assert len(figs) >= 4, figs
```

**Step 2 — FAIL. Step 3 — Implement the CLI** (argparse; guard matplotlib import; gate behind
`if __name__ == "__main__"`). Mirror `scripts/demo_oed.py` / `demo_oed_dynamical_mass.py`
structure. **Step 4 — Run** the CLI for real once (`... python scripts/demo_oed_concentration.py
--outdir /tmp/oedc`), inspect all figures, confirm exit 0. **Step 5 — Commit** scripts + test.

---

## Task 7: MyST science-demo page

**Files:** Create `docs/website/60-science-demos/optimal-design/concentration.md`; Modify the
optimal-design `index.md`/TOC to link it.

Match the existing pages' "Inputs and assumptions" standard (read `anisotropy.md` and
`dynamical-mass.md`). Cover: the OED question; the OM-King + OM-on-Michie-density modeling choice
and **both honest caveats** (jeans-path-not-df_moment; informax-bound/out-of-v0.1.0); the
Fisher/criteria recap (cross-link the shared `background.md`); the pre-registered hypothesis and
the result (with the side-by-side W₀-vs-r_a figure); the validation gate (AD-vs-FD + real-star
calibration) with the numbers. Use myst-expert for syntax.

**Step 1 — Write the page. Step 2 — Build:**
`env -u VIRTUAL_ENV uv run --no-sync myst build` (or the repo's build command) — **0 warnings**.
**Step 3 — Commit** the page + TOC.

---

## Task 8: ADR-0018 + STATUS + brain + final whole-arc review

**Files:** `/adr` new record; `STATUS.md`; (brain capture).

1. `/adr` → **ADR-0018**: the OM-King + OM-on-Michie-density modeling choice for the W₀-OED demo
   and the eddington/Engine-B real-star calibration gate (with the Michie Richardson-FD
   methodology note). Status: accepted.
2. Run the **FULL gate**:
   `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto`
   — confirm released-core count unchanged + 0 failures (scripts-only; no src). Run the @slow
   concentration calibration explicitly and record the realized/Fisher ratios for both models.
3. Update `STATUS.md` (`next:`/`blocker:`/`due:`) with the arc outcome, the science finding
   (H1 vs H0), the gate numbers, and the branch state.
4. `brain "W0-OED concentration demo: <one-line outcome + H1/H0 + gate ratios>"`.
5. Dispatch an independent **superpowers:code-reviewer** for the whole-arc review (design
   adherence, no fabricated constants, honest caveats present, no weakened tolerances). Fix any
   Critical/Important before declaring complete.

**Completion doc:** `.claude-work/W0_OED_CONCENTRATION_COMPLETE.md` (files, API, gate results with
plot references, the H1/H0 finding, lessons).

---

## Done = all of:
- FAST + FULL gates green; released-core count unchanged (scripts-only).
- AD-vs-FD ∂σ/∂lnW₀: King < 1e-3 all bins; Michie < 1e-3 inner + Richardson-convergent outer.
- Real-star calibration: realized/Fisher σ(lnW₀) within 2√(2/n_draws), no bias, both models.
- CLI exits 0, ≥4 inspected figures; MyST page builds 0 warnings.
- ADR-0018 recorded; STATUS + completion doc written; whole-arc review clean.
- HITL: nothing merged/pushed without Anna's separate explicit words.
