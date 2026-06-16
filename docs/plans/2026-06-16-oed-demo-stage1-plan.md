# OED Demo — Phase 1, Stage 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this
> plan task-by-task (fresh subagent per task, code review between tasks).

**Goal:** A pure pre-data, c-optimal Bayesian experimental-design demo that allocates a fixed star
budget across (radius × {RV, PM_R, PM_T}) to minimize the marginal variance of the Osipkov–Merritt
anisotropy radius `r_a`, and *discovers* that proper-motion stars belong in the outskirts.

**Architecture:** Scripts-level consumer of the packaged Phase-0 `project_dispersion`. The
load-bearing idea (ADR 0004): the Fisher is **additive and linear in the design** —
`F(design) = Σ_{bin b, channel c} n_eff,{b,c} · M_{b,c}`, where each per-star block `M_{b,c}` is
**design-independent** and computed **once** via a single reverse-mode `jacrev` through
`project_dispersion`. The optimization is then pure 3×3 linear algebra. We use `jacrev` (reverse
mode) because it is the supported/tested AD path for *all* profiles and keeps the demo correct if a
King/Michie mock is ever swapped in — for those equilibrium-solver profiles forward-mode genuinely
fails (`custom_vjp` ODE with no `jvp` rule). On the **Plummer** path used here there is no ODE, so
forward-mode would in fact work; `jacrev` is the robust, not the only, choice.

**Tech Stack:** `jax` (jnp, `jacrev`, `grad`, `jit`), `optax` (Adam), `diffrax` (inside
`project_dispersion`), `matplotlib`. Zero new dependencies. No packaged `src/` changes → **no
registry burden** (this is a consumer of already-registered symbols).

**Design doc:** `docs/plans/2026-06-16-oed-demo-stage1-design.md` (read it first).

---

## Conventions used throughout

- **Run from:** `/Users/anna/projects/jaxstro-dev/progenax`, branch `feat/oed-demo-stage1`.
- **Run tests with** (float64 is auto-enabled on `import progenax`):
  ```bash
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_demo_oed.py -q
  ```
- **Parameter vector** `theta = (r_a, M, r_h)` — index 0 = `r_a` (TARGET), 1 = `M`, 2 = `r_h`
  (nuisances). `P = 3`.
- **Dimensionless metric (ADR 0011, added Task 2.5):** the Fisher is built wrt `ln theta`
  (`J → J·diag(theta_fid)`), so `F` is dimensionless (cond≈45, not 1.7e9), covariances are
  **fractional variances**, the c-headline is the **fractional precision** `σ(r_a)/r_a ≈ 12%`, and
  `PRIOR_DIAG` is fractional `[0, 1/0.3², 1/0.3²]`.
- **Channels** order `(los, pm_r, pm_t)` → index `c ∈ {0,1,2}`. Per-channel per-star error
  `eps = (eps_RV, eps_PM, eps_PM)` (both PM axes share the astrometric error).
- **Units:** `STELLAR` (M⊙, pc, Myr). `project_dispersion` returns σ in **pc/Myr**. Errors must be in
  the same units. Conversions (module constants):
  - `KMS_PER_PC_PER_MYR = 0.977792` → `sigma[pc/Myr] = sigma[km/s] / 0.977792`.
  - PM error → velocity: `v[km/s] = 4.74047 · mu[mas/yr] · d[kpc]`.
  - Working mock: `eps_RV = 1.0 km/s`, `eps_PM = 0.05 mas/yr @ d=4 kpc → 4.74047·0.05·4 ≈ 0.948 km/s`.
- **Mock constants** (module-level dict `MOCK`): `M=1e5`, `r_h=3.0`, `r_a=6.0`, `d_kpc=4.0`,
  `eps_RV_kms=1.0`, `eps_PM_masyr=0.05`, `G=STELLAR.G`.
- **Bins:** `K=12` log-spaced bin centres `R_bins = jnp.logspace(log10(0.3*r_h), log10(3*r_h), 12)`.
- **Commit messages** end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Stage explicitly** (`git add <paths>`), never `git add -A`.

---

## Task 1: Predicted observable + per-star Fisher blocks (the `jacrev`-once core)

**Files:**
- Create: `scripts/_demo_oed.py`
- Test: `tests/unit/test_demo_oed.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_demo_oed.py
import jax, jax.numpy as jnp
import progenax  # enables float64
from jaxstro.units import STELLAR
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import _demo_oed as oed


def test_predict_sigma_shape_and_units():
    th = oed.theta_truth()                      # (3,) = (r_a, M, r_h)
    sig = oed.predict_sigma(th, oed.R_BINS, STELLAR.G)   # (3, K) channels x bins
    assert sig.shape == (3, oed.R_BINS.shape[0])
    assert jnp.all(sig > 0)
    # isotropic-ish check: at small R, los ~ pm_r ~ pm_t within 30%
    inner = sig[:, 0]
    assert jnp.max(inner) / jnp.min(inner) < 1.5


def test_per_star_blocks_shape_and_symmetry():
    th = oed.theta_truth()
    Mb, sig = oed.per_star_blocks(th, oed.R_BINS, oed.EPS, STELLAR.G)
    K = oed.R_BINS.shape[0]
    assert Mb.shape == (3, K, 3, 3)             # channel, bin, P, P
    # each block is symmetric PSD rank-1: M = 2 J J^T / denom
    assert jnp.allclose(Mb, jnp.swapaxes(Mb, -1, -2), atol=1e-12)
    # diagonal entries non-negative
    assert jnp.all(jnp.diagonal(Mb, axis1=-2, axis2=-1) >= -1e-12)
```

**Step 2: Run to verify it fails** (`ImportError`/`AttributeError`).

**Step 3: Minimal implementation**

```python
# scripts/_demo_oed.py
"""Stage-1 OED demo core: additive Fisher over (radius x channel), c/D/A criteria,
optax optimizer, sky projection + calibration. Consumer of progenax.project_dispersion.
See docs/plans/2026-06-16-oed-demo-stage1-design.md."""
import jax, jax.numpy as jnp
from progenax import PlummerProfile, project_dispersion
from jaxstro.units import STELLAR

KMS_PER_PC_PER_MYR = 0.977792
def kms_to_pcMyr(v_kms): return v_kms / KMS_PER_PC_PER_MYR
def pm_masyr_to_kms(mu, d_kpc): return 4.74047 * mu * d_kpc

MOCK = dict(M=1e5, r_h=3.0, r_a=6.0, d_kpc=4.0, eps_RV_kms=1.0, eps_PM_masyr=0.05)
R_BINS = jnp.logspace(jnp.log10(0.3 * MOCK["r_h"]), jnp.log10(3.0 * MOCK["r_h"]), 12)
_eps_RV = kms_to_pcMyr(MOCK["eps_RV_kms"])
_eps_PM = kms_to_pcMyr(pm_masyr_to_kms(MOCK["eps_PM_masyr"], MOCK["d_kpc"]))
EPS = jnp.array([_eps_RV, _eps_PM, _eps_PM])      # (3,) per-channel per-star error [pc/Myr]

def theta_truth():
    return jnp.array([MOCK["r_a"], MOCK["M"], MOCK["r_h"]])

def predict_sigma(theta, R_bins, G):
    """(3, K) predicted dispersions: rows = (los, pm_r, pm_t)."""
    r_a, M, r_h = theta[0], theta[1], theta[2]
    prof = PlummerProfile(r_h=r_h)
    pd = project_dispersion(prof, r_a, R_bins, M, G)
    return jnp.stack([pd.sigma_los, pd.sigma_pm_r, pd.sigma_pm_t])   # (3, K)

def per_star_blocks(theta, R_bins, eps, G):
    """Design-INDEPENDENT per-star Fisher blocks M_{c,b} = 2 J J^T / (sigma^2 + eps_c^2).
    One reverse-mode jacrev through project_dispersion. Returns (Mb (3,K,3,3), sigma (3,K))."""
    sig = predict_sigma(theta, R_bins, G)                       # (3, K)
    J = jax.jacrev(predict_sigma)(theta, R_bins, G)             # (3, K, 3)
    denom = sig**2 + (eps[:, None])**2                          # (3, K)
    Mb = 2.0 * jnp.einsum("ckp,ckq->ckpq", J, J) / denom[..., None, None]
    return Mb, sig
```

**Step 4: Run tests, verify pass.**

**Step 5: Commit**
```bash
git add scripts/_demo_oed.py tests/unit/test_demo_oed.py
git commit -m "feat(oed): predicted observable + per-star Fisher blocks (jacrev-once core)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Design allocation, completeness, additive Fisher `F = Σ n·c·M`

**Files:** Modify `scripts/_demo_oed.py`; Test `tests/unit/test_demo_oed.py`.

**Step 1: Failing test**

```python
def test_fisher_additivity_and_linearity():
    th = oed.theta_truth()
    Mb, _ = oed.per_star_blocks(th, oed.R_BINS, oed.EPS, STELLAR.G)
    K = oed.R_BINS.shape[0]
    z = jnp.zeros(3 * K)                         # uniform softmax
    F1 = oed.fisher(z, Mb, oed.completeness(oed.R_BINS), N_total=1000.0)
    F2 = oed.fisher(z, Mb, oed.completeness(oed.R_BINS), N_total=2000.0)
    # F linear in N_total at fixed design fractions
    assert jnp.allclose(F2, 2.0 * F1, rtol=1e-10)
    assert F1.shape == (3, 3)
    assert jnp.allclose(F1, F1.T, atol=1e-10)

def test_completeness_rolls_off_outward():
    c = oed.completeness(oed.R_BINS)
    assert c[0] > c[-1]                          # core more complete than outskirts
    assert jnp.all((c > 0) & (c <= 1.0))
```

**Step 2: Verify fail.**

**Step 3: Implementation** (append to `_demo_oed.py`)

```python
def completeness(R_bins, R_turn=None, width=None):
    """Smooth faint-end roll-off (logistic in R): ~1 core -> <1 outskirts.
    Illustrative selection function, NOT a real survey curve."""
    R_turn = 2.0 * MOCK["r_h"] if R_turn is None else R_turn
    width = 0.5 * MOCK["r_h"] if width is None else width
    return 1.0 / (1.0 + jnp.exp((R_bins - R_turn) / width))

def design_counts(z, completeness_b, N_total):
    """n_eff (3,K): softmax allocation x budget x completeness."""
    K = completeness_b.shape[0]
    n = N_total * jax.nn.softmax(z).reshape(3, K)
    return n * completeness_b[None, :]

def fisher(z, Mb, completeness_b, N_total, prior_diag=None):
    """F = Sum_{c,b} n_eff,{c,b} M_{c,b}  (+ optional prior precision)."""
    n_eff = design_counts(z, completeness_b, N_total)            # (3, K)
    F = jnp.einsum("ck,ckpq->pq", n_eff, Mb)
    if prior_diag is not None:
        F = F + jnp.diag(prior_diag)
    return F
```

**Note for implementer:** `prior_diag` represents independent prior knowledge of the nuisances
(e.g. `M` from integrated light, `r_h` from photometry). Default mock prior (define as
`PRIOR_DIAG`): `sigma_prior_M = 0.3*M`, `sigma_prior_rh = 0.3*r_h`, none on `r_a`
(`prior_diag = [0.0, 1/sigma_prior_M**2, 1/sigma_prior_rh**2]`). This keeps `F` well-conditioned
without constraining the target. Document it in the docstring and the MyST page.

**Step 4–5: Test pass, commit** (`feat(oed): additive Fisher F=Σ n·c·M + completeness + prior`).

> **NOTE (ADR 0011 retrofit):** Tasks 1–2 were committed with raw-unit `theta` and the raw
> `PRIOR_DIAG = [0, 1/(0.3M)², 1/(0.3 r_h)²]`. Task 2.5 (below) converts both to the dimensionless
> metric. The text above is the as-built history; the live code after Task 2.5 differs as specified
> in Task 2.5.

---

## Task 2.5: Dimensionless (log/fiducial-scaled) Fisher (ADR 0011)

**Why:** raw-unit `F` has cond≈1.7e9 and `tr(F⁻¹)` is dominated by `M`'s absolute variance, so
A/D-optimality are not scale-invariant and the c-vs-D-vs-A contrast is meaningless. Differentiating
wrt `ln theta` fixes this (cond≈45) and makes the headline a fractional precision.

**Files:** Modify `scripts/_demo_oed.py`; Test `tests/unit/test_demo_oed.py`.

**Step 1: Failing test**

```python
def test_fisher_dimensionless_well_conditioned():
    th = oed.theta_truth()
    Mb, _ = oed.per_star_blocks(th, oed.R_BINS, oed.EPS, STELLAR.G)
    cb = oed.completeness(oed.R_BINS)
    z = jnp.zeros(3 * oed.R_BINS.shape[0])
    F = oed.fisher(z, Mb, cb, 4000.0, oed.PRIOR_DIAG)
    assert jnp.linalg.cond(F) < 1e3                       # dimensionless (was ~1.7e9 raw)
    frac_sigma_ra = jnp.linalg.inv(F)[0, 0] ** 0.5        # FRACTIONAL precision on r_a
    assert 0.01 < frac_sigma_ra < 0.5                     # ~0.12 at the working mock
```

**Step 2: Verify fail** (cond ~1.7e9 with the raw blocks).

**Step 3: Implementation** — in `per_star_blocks`, scale the Jacobian to `∂σ/∂ln θ` before forming
the blocks; redefine `PRIOR_DIAG` as fractional:

```python
def per_star_blocks(theta, R_bins, eps, G):
    sig = predict_sigma(theta, R_bins, G)                          # (3, K)
    J = jax.jacrev(predict_sigma, argnums=0)(theta, R_bins, G)     # (3, K, P) = dσ/dθ
    J = J * theta[None, None, :]                                   # -> dσ/d ln θ  (DIMENSIONLESS, ADR 0011)
    denom = sig**2 + (eps[:, None])**2
    Mb = 2.0 * jnp.einsum("ckp,ckq->ckpq", J, J) / denom[..., None, None]
    return Mb, sig

_FRAC_PRIOR = 0.3   # 30% fractional prior on each nuisance (M, r_h); none on the target r_a
PRIOR_DIAG = jnp.array([0.0, 1.0 / _FRAC_PRIOR**2, 1.0 / _FRAC_PRIOR**2])   # fractional precision
```

Document in both docstrings that the Fisher/covariance are in the **fractional (d ln θ) metric**.

**Step 4: Test pass** (incl. the unchanged additivity/linearity/shape tests — scaling `J` doesn't
break linearity in `N_total` or block symmetry).

**Step 5: Commit** (`feat(oed): dimensionless ln-θ Fisher + fractional prior (ADR 0011)`).

---

## Task 3: c / D / A criteria + AD-vs-FD gradient gate

**Files:** Modify `scripts/_demo_oed.py`; Test `tests/unit/test_demo_oed.py`.

**Step 1: Failing test**

```python
def _crits():
    th = oed.theta_truth()
    Mb, _ = oed.per_star_blocks(th, oed.R_BINS, oed.EPS, STELLAR.G)
    cb = oed.completeness(oed.R_BINS)
    return Mb, cb

def test_criteria_values_positive():
    Mb, cb = _crits()
    z = jnp.zeros(3 * oed.R_BINS.shape[0])
    F = oed.fisher(z, Mb, cb, 2000.0, oed.PRIOR_DIAG)
    assert oed.c_criterion(F) > 0                       # marginal var of r_a
    assert jnp.isfinite(oed.d_criterion(F))             # -logdet
    assert oed.a_criterion(F) > 0                       # tr F^-1

def test_criteria_grads_AD_vs_FD():
    Mb, cb = _crits()
    z = jax.random.normal(jax.random.PRNGKey(0), (3 * oed.R_BINS.shape[0],)) * 0.1
    for crit in (oed.c_criterion, oed.d_criterion, oed.a_criterion):
        loss = lambda zz: crit(oed.fisher(zz, Mb, cb, 2000.0, oed.PRIOR_DIAG))
        g_ad = jax.grad(loss)(z)
        # central FD on a few coords
        eps = 1e-5
        for i in (0, 5, 17, 31):
            zp = z.at[i].add(eps); zm = z.at[i].add(-eps)
            g_fd = (loss(zp) - loss(zm)) / (2 * eps)
            assert jnp.allclose(g_ad[i], g_fd, rtol=1e-4, atol=1e-8), (crit.__name__, i)
```

**Step 3: Implementation**

`F` is the dimensionless ln-θ Fisher (Task 2.5), so these criteria are in the **fractional metric**:
c = fractional variance of r_a; A = total fractional variance; D = log det of the dimensionless F.

```python
_TARGET = 0   # index of r_a in theta

def c_criterion(F):                       # minimize: marginal FRACTIONAL variance of r_a (ADR 0011 metric)
    return jnp.linalg.inv(F)[_TARGET, _TARGET]

def d_criterion(F):                       # minimize -logdet  (== maximize logdet, D-opt)
    return -jnp.linalg.slogdet(F)[1]

def a_criterion(F):                       # minimize total fractional variance, tr F^-1 (A-opt)
    return jnp.trace(jnp.linalg.inv(F))
```

(`PRIOR_DIAG` is the fractional nuisance prior from Task 2.5.)

**Step 4–5: Test pass, commit** (`feat(oed): c/D/A optimality criteria + AD-vs-FD grad gate`).

---

## Task 4: optax multi-start optimizer

**Files:** Modify `scripts/_demo_oed.py`; Test `tests/unit/test_demo_oed.py`.

**Step 1: Failing test**

```python
def test_optimizer_reduces_c_criterion():
    Mb, cb = _crits()
    z0 = jnp.zeros(3 * oed.R_BINS.shape[0])
    F0 = oed.fisher(z0, Mb, cb, 2000.0, oed.PRIOR_DIAG)
    res = oed.optimize_design(oed.c_criterion, Mb, cb, 2000.0,
                              key=jax.random.PRNGKey(1), n_starts=4, n_steps=300)
    Fopt = oed.fisher(res.z, Mb, cb, 2000.0, oed.PRIOR_DIAG)
    assert oed.c_criterion(Fopt) < oed.c_criterion(F0)   # design beats uniform
    assert res.trace[-1] <= res.trace[0]

def test_optimizer_allocation_normalized():
    Mb, cb = _crits()
    res = oed.optimize_design(oed.c_criterion, Mb, cb, 2000.0,
                              key=jax.random.PRNGKey(2), n_starts=2, n_steps=100)
    n = 2000.0 * jax.nn.softmax(res.z)
    assert jnp.allclose(jnp.sum(n), 2000.0, rtol=1e-6)   # budget conserved (pre-completeness)
```

**Step 3: Implementation**

```python
import optax
from typing import NamedTuple

class DesignResult(NamedTuple):
    z: jnp.ndarray
    trace: jnp.ndarray
    criterion: float

def _optimize_one(criterion_fn, z0, Mb, cb, N_total, n_steps, lr):
    opt = optax.adam(lr); state = opt.init(z0)
    loss = lambda z: criterion_fn(fisher(z, Mb, cb, N_total, PRIOR_DIAG))
    @jax.jit
    def step(carry, _):
        z, st = carry
        l, g = jax.value_and_grad(loss)(z)
        upd, st = opt.update(g, st)
        return (optax.apply_updates(z, upd), st), l
    (z, _), trace = jax.lax.scan(step, (z0, state), None, length=n_steps)
    return z, trace

def optimize_design(criterion_fn, Mb, cb, N_total, key, n_starts=8, n_steps=500, lr=0.05):
    K = cb.shape[0]
    best = None
    for s in range(n_starts):
        z0 = jax.random.normal(jax.random.fold_in(key, s), (3 * K,)) * 0.5
        z, trace = _optimize_one(criterion_fn, z0, Mb, cb, N_total, n_steps, lr)
        crit = float(criterion_fn(fisher(z, Mb, cb, N_total, PRIOR_DIAG)))
        if best is None or crit < best.criterion:
            best = DesignResult(z=z, trace=trace, criterion=crit)
    return best
```

**Step 4–5: Test pass, commit** (`feat(oed): optax multi-start design optimizer`).

---

## Task 5: Sky projection + calibration ensemble (the gate)

**Files:** Modify `scripts/_demo_oed.py`; Test `tests/unit/test_demo_oed.py`.

**Step 1: Failing test** (mark `@pytest.mark.slow` — repeated sampling)

```python
import pytest

def test_project_to_sky_components():
    pos = jnp.array([[3.0, 0.0, 1.0], [0.0, 2.0, -1.0]])
    vel = jnp.array([[0.0, 5.0, 7.0], [3.0, 0.0, -2.0]])
    R, v_los, v_pm_r, v_pm_t = oed.project_to_sky(pos, vel)
    assert jnp.allclose(R, jnp.array([3.0, 2.0]))
    assert jnp.allclose(v_los, jnp.array([7.0, -2.0]))      # = v_z
    # star 1 at phi=0: pm_r = vx, pm_t = vy
    assert jnp.allclose(v_pm_r[0], 0.0) and jnp.allclose(v_pm_t[0], 5.0)

@pytest.mark.slow
def test_fisher_calibration_matches_realized_scatter():
    """Realized Var(r_a_hat) over mock draws ~ (F^-1)_{r_a,r_a} at the uniform design."""
    cal = oed.calibrate_fisher(z=jnp.zeros(3 * oed.R_BINS.shape[0]),
                               N_total=4000.0, n_draws=64, key=jax.random.PRNGKey(7))
    # tolerance set by MC error on a variance from 64 draws (~sqrt(2/64)~18%): allow 35%
    assert jnp.abs(cal.realized_var_ra - cal.fisher_var_ra) / cal.fisher_var_ra < 0.35
```

**Step 3: Implementation** (key pieces; implementer fills the binning detail)

```python
from progenax import PlummerVelocityDF   # verify exact import path & ctor in the live module first

def project_to_sky(pos, vel):
    """LOS = z-axis. Returns (R, v_los, v_pm_r, v_pm_t)."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
    R = jnp.hypot(x, y)
    phi = jnp.arctan2(y, x)
    v_los = vz
    v_pm_r = vx * jnp.cos(phi) + vy * jnp.sin(phi)
    v_pm_t = -vx * jnp.sin(phi) + vy * jnp.cos(phi)
    return R, v_los, v_pm_r, v_pm_t

class CalibResult(NamedTuple):
    realized_var_ra: float
    fisher_var_ra: float

def _draw_mock(key, n_stars):
    """Sample OM-Plummer stars at the truth; return projected per-star (R, v_los, v_pm_r, v_pm_t)."""
    kp, kv = jax.random.split(key)
    prof = PlummerProfile(r_h=MOCK["r_h"])
    df = PlummerVelocityDF(r_h=MOCK["r_h"], anisotropy_radius=MOCK["r_a"])
    masses = jnp.ones(n_stars)
    pos = prof.sample_positions(masses, kp)
    vel = df.sample_velocities(pos, masses, kv, G=STELLAR.G)   # verify exact kwargs in live module
    return project_to_sky(pos, vel)
```

**Calibration recipe (document in code):**
1. For the design `z` and `N_total`, allocate per-(bin,channel) counts `n_eff` and draw `n_draws`
   independent mock catalogs at the truth.
2. For each draw: bin stars by `R` into `R_EDGES` (edges bracketing `R_BINS`); per bin & channel form
   the binned dispersion `sigma_hat` with measurement broadening `eps_c` added in quadrature, and SE
   `se = sqrt((sigma_hat^2)/(2 n_bin))`.
3. Build `residual_fn(theta) = (sigma_hat - predict_sigma(theta,...).flatten()) / se` and use
   `fisher_information_gn(residual_fn, theta_truth)` (jacrev Gauss-Newton Fisher). Prefer this over
   `fisher_cov`/hessian for two reasons: (a) the GN/expected Fisher equals our analytic *design*
   Fisher by construction, so the calibration is a clean apples-to-apples comparison; (b) it stays
   correct if a King/Michie mock is swapped in, where hessian's forward-mode genuinely crashes
   through the `custom_vjp` ODE. (On the Plummer path here hessian would also work — but the GN
   Fisher is the right quantity regardless.) Fit the MLE with `mle_adam` from `_demo_inference.py`;
   collect `r_a_hat`.
4. `realized_var_ra = Var(r_a_hat over draws)`; `fisher_var_ra = (inv(F_design))_{r_a,r_a}` at the
   same `z, N_total`. Return both.

**Step 4–5: Test pass, commit** (`feat(oed): sky projection + Fisher calibration ensemble (gate)`).

---

## Task 6: Gated CLI `demo_oed.py` + run-record + headline/interpretability gates

**Files:** Create `scripts/demo_oed.py`; Modify `tests/unit/test_demo_oed.py`.

**Step 1: Failing test**

```python
def test_headline_design_beats_uniform():
    Mb, cb = _crits()
    res = oed.optimize_design(oed.c_criterion, Mb, cb, 4000.0,
                              key=jax.random.PRNGKey(3), n_starts=6, n_steps=400)
    z_unif = jnp.zeros(3 * oed.R_BINS.shape[0])
    c_unif = oed.c_criterion(oed.fisher(z_unif, Mb, cb, 4000.0, oed.PRIOR_DIAG))
    # equal-precision factor = c_uniform / c_designed, BOTH AT THE SAME N (prior cancels exactly).
    # CAVEAT (Task-4 review): converting this to an "equivalent uniform star count" via c∝1/N is
    # only exact in the prior-free limit; with PRIOR_DIAG the fixed prior dilutes as N grows, so
    # c·N drifts ~18% over N=1e3..8e3. Report the fixed-N factor as the headline; flag any
    # star-count gloss as approximate (Task 6 CLI + Task 8 MyST prose).
    factor = c_unif / res.criterion
    assert factor > 1.3                       # report the actual number in the CLI/page

def test_pm_fraction_increases_outward():
    Mb, cb = _crits()
    res = oed.optimize_design(oed.c_criterion, Mb, cb, 4000.0,
                              key=jax.random.PRNGKey(4), n_starts=6, n_steps=400)
    n = (4000.0 * jax.nn.softmax(res.z)).reshape(3, oed.R_BINS.shape[0])
    pm_frac = (n[1] + n[2]) / jnp.sum(n, axis=0)      # per-bin PM fraction
    K = oed.R_BINS.shape[0]
    assert jnp.mean(pm_frac[K // 2:]) > jnp.mean(pm_frac[:K // 2])   # PMs favored outward
```

**Step 3: Implementation.** `scripts/demo_oed.py` follows the existing `scripts/demo_*.py` pattern
(read `scripts/demo_anisotropy.py` for the house style: argparse, `_plotstyle`, a `main()` that
writes figures to `docs/website/60-science-demos/figures/` and a JSON run-record). It must:
- compute `Mb` once, optimize c/D/A designs, compute the headline factor + PM-fraction trend,
- run the calibration (small `n_draws` unless `--full`),
- print a quantitative summary table (criterion values, factor, PM-fraction inner/outer),
- write `figures/demo_oed_*.png` (Task 7) and a run-record JSON,
- **exit 0** on success.

**Step 4–5: Test pass; run `uv run python scripts/demo_oed.py --quick` to confirm exit 0; commit**
(`feat(oed): gated CLI demo_oed.py + headline & interpretability gates`).

---

## Task 7: Five figures

**Files:** Modify `scripts/demo_oed.py` (figure functions); manual visual inspection.

Implement the five figures from the design doc, saved to
`docs/website/60-science-demos/figures/`:
1. `demo_oed_optpath.png` — c-criterion vs iteration, multi-start traces.
2. `demo_oed_headline.png` — optimal radial weighting + RV/PM split over `σ(r)` / `β(r)`.
3. `demo_oed_cda.png` — c-vs-D-vs-A allocations side-by-side.
4. `demo_oed_frontier.png` — precision vs budget, designed vs uniform (the N× factor). Plot the
   measured `σ(r_a)/r_a` vs N for both designs directly (do NOT extrapolate via an idealized
   `c∝1/N` star-count claim — with the nuisance prior, `c·N` drifts ~18% over N=1e3..8e3; the
   curves show the real, mildly-non-1/N frontier).
5. `demo_oed_calibration.png` — realized MLE Cov vs `F⁻¹`.

Use `scripts/_plotstyle.py`. **Step: visually inspect each PNG** before committing
(`feat(oed): Stage-1 figures`).

---

## Task 8: B14 MyST page + completion doc + STATUS/brain

**Files:**
- Create `docs/website/60-science-demos/optimal-design.md`
- Add the page to the science-demos toc (check `docs/website/60-science-demos/index.md` and
  `myst.yml`/`_toc` for how siblings like `anisotropy.md` are registered).
- Create `.claude-work/OED_DEMO_STAGE1_COMPLETE.md`
- Update `STATUS.md` (`next:`/`blocker:`/`due:`).

**MyST page must include:** the OED idea; the additive-Fisher backbone; the mock + its assumptions
("Inputs and assumptions" block matching the house standard); the five figures with captions; the
quantitative headline (N× factor, PM-outskirts trend); the c-vs-D-vs-A lesson; an **"Current scope /
planned extensions"** caveat audit naming Stage 1's boundaries (single LOS; fixed completeness;
OM-Plummer; depth/epochs deferred to Stages 2–3; rotation/flattening/tracer≠mass on the roadmap).

**Build gate:**
```bash
cd docs/website && env -u VIRTUAL_ENV uv run --no-sync myst build --html   # expect 0 warnings
```

**Step: commit** (`docs(oed): B14 optimal-design page + completion + STATUS`).

---

## Final gate (after all tasks)

1. Full released-core suite green locally (fast gate, then full gate):
   ```bash
   XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
     env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
   ```
2. `tests/unit/test_demo_oed.py` all green (incl. `@slow` calibration).
3. `scripts/demo_oed.py` exits 0 + writes run-record; five figures inspected.
4. `myst build --html` 0 warnings.
5. Completion doc written; STATUS updated; `brain` capture.
6. **Final whole-arc code review** (one reviewer over `_demo_oed.py` + `demo_oed.py`) — the Phase-0.5
   lesson: a final integration-level review catches what per-task reviews miss.
7. **Anna merge-go** → merge to local `main` → push on her word → delete branch.

**Definition of Complete** (design doc): N× factor reported; PM-fraction-increases-outward asserted;
AD-vs-FD on c/D/A passes; calibration passes; tests + CLI + MyST + completion all green.
