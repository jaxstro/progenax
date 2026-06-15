# B12 Binary-Inflated Dynamical Mass Demo — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task.

**Goal:** A gated science demo (B12) showing unresolved binaries bias the virial/dynamical mass high, that a dispersion-only analysis cannot fix it (rank-1 degeneracy), and that a differentiable joint recovery from the velocity-distribution wings returns an unbiased mass — with a Fisher forecast vs (N, ε).

**Architecture:** A forward model (cluster LOS velocity DF + Moe P–q–e binaries whose unresolved RV is a ZAMS-flux-weighted blend) produces a mock `v_los` sample. The binary contamination is a precomputed σ-independent kernel `K_orb`. A binned Poisson mixture likelihood `μ_k(σ,f_b)=N[(1−f_b)𝒩_k(σ)+f_b(𝒩(σ)⊛K_orb)_k]` is fit with the existing `_demo_inference` toolkit; the Fisher quantifies how the wings break the `σ_true`–`f_b` degeneracy.

**Tech Stack:** JAX (`jax.numpy`, `jax.grad`, `jax.vmap`), `progenax.stellar.zams_luminosity`, `progenax.imf.binary.MoeJointOrbit`, `progenax.binaries.KeplerElements.to_state`, `progenax.{PlummerProfile,PlummerVelocityDF}` (or King), `scripts/_demo_inference.py` (poisson_loglike, mle_adam, poisson_fisher_information, constrained_cov, logit/expit), matplotlib.

**Design doc:** `docs/plans/2026-06-14-binary-dynamical-mass-demo-design.md` (4 ratified decisions).

**Branch:** `feat/binary-dynamical-mass-demo` (already cut). **Demo only — `scripts/` + `docs/` + demo-harness `tests/`; NO `src/progenax/` change → released-core gate unaffected.** All LOCAL; nothing pushed/merged without Anna.

**Conventions:** `import progenax` first (float64). Run tests with `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest <args> -v`. Matplotlib `Agg`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Phase P1 — Demo helpers (TDD)

New file `scripts/_demo_binaries.py` holding the three reusable pieces, with unit tests in `tests/demos/test_demo_binaries.py` (demo-harness tier; NOT released-core). Use @superpowers:test-driven-development.

### Task P1.1: `project_los_velocity` — isotropic LOS projection of a 3-velocity

**Files:** Create `scripts/_demo_binaries.py`; Test `tests/demos/test_demo_binaries.py` (+ `tests/demos/__init__.py`).

**Step 1 (failing test):**
```python
import numpy as np, jax.numpy as jnp, progenax
from scripts._demo_binaries import project_los_velocity   # adjust import per how scripts are imported

def test_los_projection_is_isotropic_mean_zero():
    rng = np.random.default_rng(0)
    v = jnp.array([10.0, 0.0, 0.0])               # 10 km/s along x
    dirs = rng.normal(size=(20000, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    los = np.array([float(project_los_velocity(v, jnp.asarray(d))) for d in dirs])
    assert abs(los.mean()) < 0.3                  # isotropic -> mean ~0
    assert np.isclose(los.std(), 10.0/np.sqrt(3), rtol=0.05)   # variance/3 per axis
```
**Step 2:** run → FAIL (no `project_los_velocity`). **Step 3:** implement `project_los_velocity(vel3, los_hat) = jnp.dot(vel3, los_hat)` (LOS unit vector dotted into the velocity). **Step 4:** PASS. **Step 5:** commit `feat(demo): project_los_velocity helper`.

### Task P1.2: `build_korb_kernel` — the σ-independent flux-weighted blend kernel

The kernel = histogram of the internal blend velocity `Δ` over a large Moe pool. For each system: draw `m1` (IMF), `(P,q,e)=MoeJointOrbit.sample`, `m2=q*m1`; build `KeplerElements` at a **random orbital phase** (mean anomaly ~U(0,2π)); `to_state(M_total=m1+m2, G)` → relative `(pos, v_rel)`; component velocities `v1=(m2/M)v_rel`, `v2=-(m1/M)v_rel`; project each onto a **random isotropic LOS** `los_hat` (one per system); flux-weight `Δ=(L1*v1_los+L2*v2_los)/(L1+L2)` with `L=zams_luminosity(m,Z)`; histogram `Δ` on a fixed velocity grid → normalized density.

**Step 1 (failing tests):**
```python
def test_korb_high_q_self_cancels():
    # equal-mass (q=1) binaries: L1=L2, m1=m2 -> Delta -> 0 (centroid cancellation)
    v_grid, k = build_korb_kernel(n_pool=20000, q_fixed=1.0, Z=1e-3, seed=1)
    # spread of K_orb must be far smaller than for a low-q pool
    spread_hi = _kernel_std(v_grid, k)
    v_grid2, k2 = build_korb_kernel(n_pool=20000, q_fixed=0.2, Z=1e-3, seed=1)
    spread_lo = _kernel_std(v_grid2, k2)
    assert spread_hi < 0.3 * spread_lo            # high-q cancels, low-q (primary) does not

def test_korb_normalized_and_zero_mean():
    v_grid, k = build_korb_kernel(n_pool=50000, Z=1e-3, seed=2)
    dv = v_grid[1]-v_grid[0]
    assert np.isclose(np.sum(k)*dv, 1.0, atol=1e-3)         # normalized density
    assert abs(np.sum(k*v_grid)*dv) < 0.5                   # ~zero mean (random phase+orientation)
```
**Step 3 (implement):** the function above; `q_fixed` optional override for the test (else draw from Moe). Use `progenax.imf.binary.MoeJointOrbit` with the project defaults (mirror B4's `JOINT`), an IMF (`Maschberger`), `KeplerElements` (read `src/progenax/binaries/kepler.py` for the field names incl. the phase/anomaly field — set it from a uniform draw), `STELLAR.G`. Build on a symmetric `v_grid` (e.g. `np.linspace(-150,150,601)` km/s in code units). Provide `_kernel_std(v_grid,k)` helper. **Step 4:** PASS. **Step 5:** commit `feat(demo): build_korb_kernel (Moe+ZAMS flux-weighted blend kernel, sigma-independent)`.

> NOTE: if `KeplerElements.to_state` returns velocity in code units (pc/Myr), convert to km/s with `STELLAR` (`* STELLAR.velocity_scale` or the documented factor — check `jaxstro.units`); keep ALL velocities in ONE unit system (km/s) end-to-end.

### Task P1.3: `predict_vlos_counts` — the differentiable binned mixture model

**Step 1 (failing tests):**
```python
def test_predict_reduces_to_gaussian_at_fb0():
    v_edges = np.linspace(-30, 30, 61)
    v_grid, k = build_korb_kernel(n_pool=20000, Z=1e-3, seed=3)
    mu = predict_vlos_counts(sigma=5.0, f_b=0.0, N=1500, v_edges=v_edges,
                             korb_grid=v_grid, korb=k, eps=1.0)
    # f_b=0 -> pure Gaussian(sigma_obs=sqrt(5^2+1^2)) binned; integral = N
    assert np.isclose(mu.sum(), 1500, rtol=1e-6)
    # symmetric, peaked at center
    assert mu.argmax() in (len(mu)//2 - 1, len(mu)//2)

def test_predict_differentiable_in_sigma_and_fb():
    import jax
    v_edges = jnp.linspace(-30, 30, 61); v_grid, k = build_korb_kernel(n_pool=20000, Z=1e-3, seed=4)
    f = lambda s, fb: jnp.sum(predict_vlos_counts(s, fb, 1500, v_edges, v_grid, jnp.asarray(k), 1.0)**2)
    gs, gfb = jax.grad(f, argnums=(0,1))(5.0, 0.5)
    assert jnp.isfinite(gs) and jnp.isfinite(gfb)
```
**Step 3 (implement):** `predict_vlos_counts(sigma, f_b, N, v_edges, korb_grid, korb, eps)`:
- single density on `korb_grid`: `g = 𝒩(korb_grid; 0, sigma²+eps²)` (analytic, differentiable in sigma);
- binary density: `b = conv(𝒩(·;0,sigma²+eps²), korb)` on the grid (discrete convolution via `jnp.convolve(..., mode="same")*dv`, or FFT); differentiable in sigma;
- mixture density `p = (1-f_b)*g + f_b*b`; integrate `p` over each `[v_edges_k, v_edges_{k+1}]` (trapezoid on the grid masked per bin, or interpolate the CDF) → fractions `phi_k`; `mu_k = N * phi_k`.
- Keep everything `jnp` so `jax.grad` flows. **Step 4:** PASS. **Step 5:** commit `feat(demo): predict_vlos_counts (differentiable binned single+binary mixture)`.

### Task P1.4: `virial_mass_factor` / bias helper
Small helper `dyn_mass_ratio(sigma_obs, sigma_true) -> (sigma_obs/sigma_true)**2` (virial `M ∝ σ²`) with a test that it's 1 at equality and >1 when inflated. Commit `feat(demo): dynamical-mass bias helper`.

---

## Phase P2 — Forward model (mock data) in the demo script

**File:** Create `scripts/demo_binary_dynamical_mass.py` (the B12 CLI). Header docstring = the design's premise + gate list. Config block (mirror B4/B8 style): `SIGMA_TRUE=5.0` km/s, `Z_MET=1e-3`, `F_B_TRUE=0.5`, `N_STARS=1500`, `EPS_KMS=1.0`, `R_H`, IMF, Moe `JOINT`, `V_EDGES`, `KORB_GRID/N_POOL`, `SEED`.

### Task P2.1: `build_mock_vlos(key)` — the observed sample
Draw `N_STARS` cluster LOS velocities `v_COM ~ 𝒩(0,SIGMA_TRUE²)` (either sample the velocity DF + `project_los_velocity`, or directly `SIGMA_TRUE*normal` for the global demo — pick the DF route if cheap, else Gaussian is faithful for a global isotropic dispersion). For a fraction `F_B_TRUE`, add `Δ` (draw fresh Moe systems → blend velocity via the P1.2 machinery, NOT from the histogram). Add `𝒩(0,EPS²)` measurement noise to every star. Return `v_obs` (N,). **Verify** (a scratch assert, not a committed test): `std(v_obs) > SIGMA_TRUE` (inflation present). Commit `feat(demo): B12 forward model — binary-contaminated v_los sample`.

---

## Phase P3 — Gate 1 (bias) + Gate 2 (dispersion-only rank-1)

### Task P3.1: bias panel + gate
Compute `sigma_obs = std(v_obs)`, `M_ratio = (sigma_obs/SIGMA_TRUE)**2`; sweep `f_b∈[0,0.7]` rebuilding the sample → `M_ratio(f_b)` curve. **Gate 1:** `sigma_obs > SIGMA_TRUE` and `M_ratio(0.5) > 1.10` (report the %). Figure `validation/plots/demo_binary_dynamical_mass_bias.png` (M_ratio vs f_b). Commit.

### Task P3.2: dispersion-only degeneracy gate
Build the **dispersion-only** Fisher: the single summary `sigma_obs` as the data, predicted `sigma_obs(σ_true,f_b)=sqrt(σ_true²+f_b*Var(K_orb)+eps²)`; its 2×2 Fisher (via `jax.grad` of the scalar predicted summary, outer product) is **rank-1**. **Gate 2:** condition number `> 1e8` or smallest eigenvalue `< 1e-8 * largest`. (This formalizes "one number can't separate two parameters.") Commit `feat(demo): Gate 1 bias + Gate 2 dispersion-only rank-1`.

---

## Phase P4 — Gate 3 (joint recovery + full-rank Fisher)

### Task P4.1: joint MLE recovery
Precompute `KORB` once. Define `predict_mu(z)` = `predict_vlos_counts(expit_sigma(z[0]), expit_fb(z[1]), N_STARS, V_EDGES, KORB_GRID, KORB, EPS)` with `logit/expit` bounds (`sigma∈(0.5,30)`, `f_b∈(0,0.95)`). Data = `binned counts of v_obs over V_EDGES`. `nll = lambda z: -poisson_loglike(counts, predict_mu)(z)`; `z_hat = mle_adam(nll, z0)`; map back to `(sigma_hat, fb_hat)`. **Gate 3a:** `|sigma_hat - SIGMA_TRUE| < 3*Fisher_sigma` AND `(sigma_hat/SIGMA_TRUE)**2` consistent with 1 (unbiased M_dyn) — i.e. recovered mass bias removed. Commit.

### Task P4.2: full-rank Fisher + the constraint figure
`F = poisson_fisher_information(predict_mu, z_hat)` → `constrained_cov(F, dtheta_dz)` → `Cov(sigma,f_b)`. **Gate 3b:** `F` well-conditioned (cond `< 1e6`) — the wings broke the degeneracy. Figure `..._constraint.png`: the dispersion-only degenerate ridge (from P3.2) vs the tight full-distribution 1σ ellipse. Optional: `run_nuts` corner as a cross-check (stretch). Commit `feat(demo): Gate 3 joint recovery + full-rank Fisher + constraint figure`.

---

## Phase P5 — Gate 4 (ε-floor) + Gate 5 (null) + Gate 6 (AD-vs-FD) + N-forecast

### Task P5.1: ε-floor sweep
Sweep `EPS∈[0.2,5]` km/s: for each, recover `(sigma,f_b)` and record the residual mass bias + `fb_hat`. **Gate 4:** bias-removal degrades monotonically as ε grows; `fb_hat` decreases toward the detectable fraction (report `f_b(P<P_max(ε))` honestly). Figure `..._eps_floor.png`. Commit.

### Task P5.2: null + AD-vs-FD + N-forecast
- **Gate 5 (null):** rebuild with `F_B_TRUE=0` → `sigma_obs≈SIGMA_TRUE`, `fb_hat<0.05`.
- **Gate 6 (AD-vs-FD):** the Fisher's Jacobian `∂μ/∂z` via `jax.jacobian` vs central finite differences, max rel-err `< 1e-4` (gradient integrity, suite style).
- **N-forecast:** `σ(sigma_true)`, `σ(f_b)` vs `N∈{500,1500,5000,15000}` from the Poisson Fisher (∝ 1/√N). Figure `..._fisher_vs_N.png`. Commit `feat(demo): Gate 4 eps-floor + Gate 5 null + Gate 6 AD-vs-FD + N-forecast`.

---

## Phase P6 — Gated CLI assembly + run-record

### Task P6.1: `main()` + the 6-gate summary
Assemble `main()` running all panels, printing an expected-vs-measured table per gate with `PASS/FAIL`, a final `ALL PASS` banner, `sys.exit(0 if all_pass else 1)` (mirror `validate_tidal.py`/`demo_*` structure). Also save the headline `v_los` figure `..._distribution.png` (singles vs observed, wings annotated). Run `... python scripts/demo_binary_dynamical_mass.py` → **exit 0, ALL PASS**; capture the printed table into the completion doc. **VISUALLY INSPECT all 5 figures** (publication-style, physically correct). Commit `feat(demo): B12 gated CLI — binary-inflated dynamical mass, all gates pass`.

---

## Phase P7 — Website page + close-out

### Task P7.1: `docs/website/60-science-demos/binary-dynamical-mass.md` + nav
Use @myst:myst-expert. Sections: the premise (binary-inflated virial mass; UFD M/L relevance), the flux-weighted SB2 blend (cite {cite:t}`Tout1996` for the ZAMS L, {cite:t}`MoeDiStefano2017` for P–q–e), the rank-1 degeneracy → wings-break-it story, the 5 figures, the ε-floor honesty panel, the B4↔B12 multi-channel note. Wire into `docs/website/myst.yml` + the `60-science-demos/index.md` run list (this is B12; note no fluxax/gravax needed). `cd docs/website && make build` → 0 new warnings. Commit.

### Task P7.2: close-out
- FULL released-core gate (`pytest tests/unit tests/integration tests/validation -q -n auto`) → confirm unaffected (scripts/+docs/+demo-tests only; no src/progenax change → coverage src-fresh, registries untouched). Also run the new demo-harness tests `tests/demos/`.
- Completion doc `.claude-work/B12_BINARY_DYNAMICAL_MASS_COMPLETE.md` (gates table + run-record + figures list + lessons).
- STATUS + `brain`. **CHECKPOINT → merge `feat/binary-dynamical-mass-demo` → main on Anna's go.**

---

## Definition of Complete
- [ ] `scripts/demo_binary_dynamical_mass.py` — gated CLI, **exit 0, all 6 gates PASS**, run-record captured.
- [ ] `scripts/_demo_binaries.py` helpers + `tests/demos/test_demo_binaries.py` passing.
- [ ] 5 figures in `validation/plots/`, visually inspected (publication-quality, physically correct).
- [ ] `docs/website/60-science-demos/binary-dynamical-mass.md` + nav; `make build` 0 warnings.
- [ ] FULL released-core gate unaffected (verified); completion doc; STATUS/brain.

## Risks / watch-items
- **Units:** keep ALL velocities in km/s end-to-end; `to_state` returns code units — convert once. The single biggest bug risk.
- **K_orb tails vs bin range:** `V_EDGES` must span the wings (short-period binaries reach tens of km/s) or the Poisson likelihood loses the f_b information. Set `V_EDGES` wide (±~6σ_obs) and check tail occupancy.
- **Orbital phase + isotropic orientation** must both be randomized in `build_korb_kernel` (eccentric orbits spend more time slow at apocenter — a uniform *mean anomaly* draw, not eccentric/true anomaly, gives the correct time-weighting).
- **Convolution differentiability:** use `jnp.convolve` (differentiable) on a fixed grid; the Gaussian is analytic in σ. Avoid histogramming inside the predict (non-differentiable) — `K_orb` is histogrammed ONCE (data, not a parameter).
- Demo only — if any task tempts a `src/progenax/` change, STOP and reconsider (it would pull in the registry/coverage dance).
