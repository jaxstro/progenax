# OED Stage 2 (M_dyn depth knob) + OED section — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this
> plan task-by-task (fresh subagent per task, code review between tasks).

**Goal:** Promote the magnitude limit from Stage-1's fixed completeness to an optimisable design knob
`m_lim`, headlining the dynamical mass `M_dyn`; factor the reusable selection/photometry physics into
a shared helper; and refactor the OED demo into a proper section (theory separated from examples).

**Architecture:** A new shared `scripts/_demo_selection.py` (ZAMS→mag→m_min, photon-noise error,
IMF-detectable counts) feeds a depth layer that reuses the Stage-1 additive Fisher. The predicted
dispersions are `m_lim`-independent, so the per-star Jacobian `J` is still computed **once**; `m_lim`
enters only through a per-channel effective error `ε_eff(m_lim)` and a per-bin availability weight
`avail_b(m_lim)`. The design is `[z, m_lim]`, optimised jointly.

**Tech Stack:** `jax` (jnp, `jacrev`, `grad`, `jit`), `optax`, `progenax.stellar` (ZAMS),
`progenax.ChabrierIMF`, `diffrax` (inside `project_dispersion`). Zero new dependencies.

**Design doc:** `docs/plans/2026-06-17-oed-stage2-dynamical-mass-design.md` (read first).

---

## Conventions

- **Run from** `/Users/anna/projects/jaxstro-dev/progenax`, branch `feat/oed-stage2-dynamical-mass`.
- **Run tests:**
  ```bash
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_demo_selection.py tests/unit/test_demo_oed_depth.py -q
  ```
- **θ = (r_a, M, r_h)**, index 0/1/2. Stage 2's target is **M (index 1)**; c-criterion targets it.
- **Mock** reuses Stage-1's `MOCK` (M=1e5, r_h=3pc, r_a=6pc, d=4kpc) + new selection knobs.
- **Magnitudes are bolometric** (a documented simplification — no band/BC/extinction): with
  `M_bol_sun = 4.74`, `M_bol(m) = 4.74 − 2.5 log10(L(m)/L_sun)` and
  `m_app = M_bol(m) + 5 log10(d_pc/10)`.
- **Commit messages** end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`;
  stage files explicitly (never `git add -A`).
- TDD throughout: failing test first, see it fail, minimal impl, see it pass, commit. NEVER weaken a
  tolerance to pass — fix the root cause or report.

---

## Task 1: `_demo_selection.py` — selection/photometry physics (shared helper)

**Files:** Create `scripts/_demo_selection.py`, `tests/unit/test_demo_selection.py`.

**Step 1: Failing tests**

```python
# tests/unit/test_demo_selection.py
import jax.numpy as jnp, sys, pathlib
import progenax
from progenax import ChabrierIMF
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import _demo_selection as sel

def test_apparent_mag_and_distance_modulus():
    # the Sun (M=1) at 10 pc has m_app = M_bol_sun = 4.74
    m_app = sel.apparent_mag(jnp.array(1.0), d_pc=10.0, Z=0.02)
    assert jnp.allclose(m_app, 4.74, atol=0.05)
    # farther -> fainter (larger m_app)
    assert sel.apparent_mag(jnp.array(1.0), d_pc=4000.0) > sel.apparent_mag(jnp.array(1.0), d_pc=10.0)

def test_m_min_monotonic_in_depth():
    d = 4000.0
    # a brighter (smaller) m_lim admits only higher-mass stars -> larger m_min
    m_min_shallow = sel.m_min(m_lim=10.0, d_pc=d)
    m_min_deep    = sel.m_min(m_lim=14.0, d_pc=d)
    assert m_min_deep < m_min_shallow            # deeper reaches lower mass
    # round-trip: a star exactly at m_lim has apparent mag ~ m_lim
    assert jnp.allclose(sel.apparent_mag(m_min_deep, d_pc=d), 14.0, atol=0.1)

def test_photon_noise_grows_faint():
    eps_bright = sel.photon_noise_error(m_app=jnp.array(12.0), eps0=1.0, m_ref=12.0)
    eps_faint  = sel.photon_noise_error(m_app=jnp.array(16.0), eps0=1.0, m_ref=12.0)
    assert jnp.allclose(eps_bright, 1.0)          # at m_ref, error = eps0
    assert eps_faint > eps_bright                 # 10^{0.2*4} = 2.51x

def test_detectable_fraction_in_unit_interval_and_monotonic():
    imf = ChabrierIMF(m_min=0.08, m_max=100.0)
    f_shallow = sel.detectable_fraction(m_lim=10.0, d_pc=4000.0, imf=imf)
    f_deep    = sel.detectable_fraction(m_lim=14.0, d_pc=4000.0, imf=imf)
    assert 0.0 <= f_shallow <= 1.0 and 0.0 <= f_deep <= 1.0
    assert f_deep > f_shallow                     # deeper detects a larger IMF fraction
```

**Step 3: Implementation**

```python
# scripts/_demo_selection.py
"""Shared selection / photometry physics for the demos (magnitude limits, photon-noise errors,
IMF-detectable counts). Reusable by the OED Stage-2 demo, B4 (binary mass function), B5 (IMF), and
any future magnitude-limited demo. Bolometric magnitudes (documented simplification: no band/BC/
extinction); band-specific photometry (BCs, extinction, crowding) is a planned follow-up via the
`fluxax` package once it is finalised. All functions are jnp / differentiable."""
import jax.numpy as jnp
from progenax.stellar import zams_luminosity, inverse_zams_luminosity

M_BOL_SUN = 4.74

def abs_bol_mag(mass, Z=0.02):
    """Absolute bolometric magnitude from the Tout+1996 ZAMS L(M)."""
    L = zams_luminosity(mass, Z)                      # [L_sun]
    return M_BOL_SUN - 2.5 * jnp.log10(L)

def distance_modulus(d_pc):
    return 5.0 * jnp.log10(d_pc / 10.0)

def apparent_mag(mass, d_pc, Z=0.02):
    return abs_bol_mag(mass, Z) + distance_modulus(d_pc)

def m_min(m_lim, d_pc, Z=0.02):
    """Minimum detectable mass at limiting (apparent) magnitude m_lim and distance d_pc.
    Differentiable: m_lim -> faintest absolute mag -> L_min -> inverse ZAMS -> mass."""
    M_abs_max = m_lim - distance_modulus(d_pc)
    L_min = 10.0 ** (-0.4 * (M_abs_max - M_BOL_SUN))  # [L_sun]
    return inverse_zams_luminosity(L_min, Z)

def photon_noise_error(m_app, eps0, m_ref):
    """Per-star measurement error scaling: eps = eps0 * 10^{0.2 (m_app - m_ref)} (flux^-0.5-like)."""
    return eps0 * 10.0 ** (0.2 * (m_app - m_ref))

def detectable_fraction(m_lim, d_pc, imf, Z=0.02):
    """IMF fraction with mass >= m_min(m_lim): 1 - cdf(m_min), clamped to the IMF support."""
    mm = jnp.clip(m_min(m_lim, d_pc, Z), imf.m_min, imf.m_max)
    return 1.0 - imf.cdf(mm)
```

**Step 5: Commit** (`feat(demo): shared _demo_selection.py — mag limits, photon noise, IMF counts`).

---

## Task 2: expose `J` and `σ` from `per_star_blocks` (reuse "J once" at varying ε)

**Files:** Modify `scripts/_demo_oed.py`; Test `tests/unit/test_demo_oed_depth.py`.

The depth layer must rebuild `M_{b,c} = 2 J J^T/(σ² + ε_eff²)` at varying `ε_eff(m_lim)` **without**
re-`jacrev`-ing. Add a cheap helper and expose the raw `J`, `σ`.

**Step 1: Failing test**

```python
# tests/unit/test_demo_oed_depth.py
import jax, jax.numpy as jnp, sys, pathlib
import progenax
from jaxstro.units import STELLAR
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import _demo_oed as oed

def test_blocks_from_eps_matches_per_star_blocks():
    th = oed.theta_truth()
    J, sig = oed.jacobian_and_sigma(th, oed.R_BINS, STELLAR.G)   # NEW: (3,K,3),(3,K)
    Mb_ref, _ = oed.per_star_blocks(th, oed.R_BINS, oed.EPS, STELLAR.G)
    Mb = oed.blocks_from_eps(J, sig, oed.EPS)                    # NEW: rebuild at given eps
    assert jnp.allclose(Mb, Mb_ref, atol=1e-12)
    # different eps -> different blocks (larger eps -> smaller information)
    Mb_noisy = oed.blocks_from_eps(J, sig, 2.0 * oed.EPS)
    assert jnp.all(jnp.diagonal(Mb_noisy, axis1=-2, axis2=-1)
                   <= jnp.diagonal(Mb, axis1=-2, axis2=-1) + 1e-12)
```

**Step 3: Implementation** (refactor `per_star_blocks` to delegate):

```python
def jacobian_and_sigma(theta, R_bins, G):
    """Return (J, sigma): J = d sigma_pred / d ln theta (3,K,3), sigma (3,K). One jacrev."""
    sig = predict_sigma(theta, R_bins, G)
    J = jax.jacrev(predict_sigma, argnums=0)(theta, R_bins, G) * theta[None, None, :]
    return J, sig

def blocks_from_eps(J, sig, eps):
    """M_{c,b} = 2 J J^T / (sigma^2 + eps^2). eps broadcasts: (3,) per-channel or (3,K) per-cell."""
    eps = jnp.asarray(eps)
    eps2 = (eps[:, None] if eps.ndim == 1 else eps) ** 2
    denom = sig**2 + eps2
    return 2.0 * jnp.einsum("ckp,ckq->ckpq", J, J) / denom[..., None, None]

def per_star_blocks(theta, R_bins, eps, G):     # now a thin wrapper (backward-compatible)
    J, sig = jacobian_and_sigma(theta, R_bins, G)
    return blocks_from_eps(J, sig, eps), sig
```

**Step 5: Commit** (`refactor(oed): expose jacobian_and_sigma + blocks_from_eps (rebuild M at varying eps)`).

---

## Task 3: the depth layer — `ε_eff(m_lim)`, `avail_b(m_lim)`, `depth_fisher`

**Files:** Modify `scripts/_demo_oed.py` (or a new `scripts/_demo_oed_depth.py` if `_demo_oed.py`
nears 500 LOC — check `wc -l` first; prefer a new module to stay under the cap); Test
`tests/unit/test_demo_oed_depth.py`.

**Physics:** `ε_eff,c(m_lim)` = IMF-weighted RMS of the per-star error over detectable masses (global,
per-channel — all stars share `d`). `avail_b(m_lim)` = intrinsic per-bin star pool × global detectable
fraction. The design's effective per-cell count is **smoothly capped** by availability:
`n_eff = avail · tanh(n_design / avail)` (≈ n when n≪avail; saturates at avail when n≫avail).

**Step 1: Failing test**

```python
def test_eps_eff_rises_with_depth():
    e_shallow = oed_depth.eps_eff(m_lim=10.0)     # (3,) per channel
    e_deep    = oed_depth.eps_eff(m_lim=14.0)
    assert jnp.all(e_deep > e_shallow)            # admitting faint stars raises the mean error

def test_availability_rises_with_depth_and_is_radial():
    a_shallow = oed_depth.avail_bins(m_lim=10.0)  # (K,)
    a_deep    = oed_depth.avail_bins(m_lim=14.0)
    assert jnp.all(a_deep >= a_shallow)           # deeper detects more per bin
    assert a_shallow[0] > a_shallow[-1]           # outskirts star-starved (lower density)

def test_depth_fisher_spd_and_targets_M():
    z = jnp.zeros(3 * oed.R_BINS.shape[0])
    F = oed_depth.depth_fisher(z, m_lim=12.0, N_total=4000.0)
    assert F.shape == (3, 3) and jnp.allclose(F, F.T, atol=1e-10)
    assert jnp.all(jnp.linalg.eigvalsh(F) > 0)
    # c-criterion targeting M (index 1) is finite & positive
    assert oed.c_criterion(F, target=1) > 0
```

**Step 3: Implementation** (sketch — implementer fills the quadratures):

```python
# eps_eff: IMF-weighted RMS error over [m_min(m_lim), m_max], per channel
_IMF = ChabrierIMF(m_min=0.08, m_max=MOCK["m_max"])
def eps_eff(m_lim):
    m_grid = ...                                  # log-spaced [m_min(m_lim), m_max], differentiable lower edge
    w = imf_pdf(m_grid); w = w / w.sum()
    m_app = sel.apparent_mag(m_grid, MOCK["d_pc"])
    eps_RV = sel.photon_noise_error(m_app, EPS0_RV, M_REF)
    eps_PM = sel.photon_noise_error(m_app, EPS0_PM, M_REF)
    rms = lambda e: jnp.sqrt(jnp.sum(w * e**2))
    return jnp.array([rms(eps_RV), rms(eps_PM), rms(eps_PM)])

def avail_bins(m_lim):
    return N_FIELD_BINS * sel.detectable_fraction(m_lim, MOCK["d_pc"], _IMF)   # (K,) x scalar

def depth_fisher(z, m_lim, N_total, prior_diag=oed.PRIOR_DIAG):
    J, sig = oed.jacobian_and_sigma(oed.theta_truth(), oed.R_BINS, STELLAR.G)  # J once
    Mb = oed.blocks_from_eps(J, sig, eps_eff(m_lim))
    n_design = N_total * jax.nn.softmax(z).reshape(3, oed.R_BINS.shape[0])
    avail = avail_bins(m_lim)[None, :]
    n_eff = avail * jnp.tanh(n_design / avail)    # smooth availability cap
    return jnp.einsum("ck,ckpq->pq", n_eff, Mb) + jnp.diag(prior_diag)
```

Also generalize `oed.c_criterion(F, target=0)` to take a target index (default 0 keeps Stage 1).

**Step 5: Commit** (`feat(oed): depth layer — eps_eff/avail/depth_fisher + M-target criterion`).

---

## Task 4: joint optimiser over `[z, m_lim]` + AD-vs-FD gate

**Files:** Modify the depth module; Test `tests/unit/test_demo_oed_depth.py`.

**Step 1: Failing test** (the load-bearing differentiability gate — the new `m_lim` gradient):

```python
def test_depth_criterion_grad_AD_vs_FD():
    z = jax.random.normal(jax.random.PRNGKey(0), (3 * oed.R_BINS.shape[0],)) * 0.1
    u = jnp.array(0.3)                              # m_lim via expit into [m_lo, m_hi]
    loss = lambda zz, uu: oed.c_criterion(oed_depth.depth_fisher_u(zz, uu, 4000.0), target=1)
    g_ad = jax.grad(loss, argnums=(0, 1))(z, u)
    eps = 1e-5
    # FD on m_lim (the new dimension) and a few z coords
    g_fd_u = (loss(z, u+eps) - loss(z, u-eps)) / (2*eps)
    assert jnp.allclose(g_ad[1], g_fd_u, rtol=1e-4, atol=1e-8)
    for i in (0, 17, 31):
        zp = z.at[i].add(eps); zm = z.at[i].add(-eps)
        assert jnp.allclose(g_ad[0][i], (loss(zp,u)-loss(zm,u))/(2*eps), rtol=1e-4, atol=1e-8)

def test_joint_optimizer_beats_fixed_depth():
    res = oed_depth.optimize_depth_design(target=1, N_total=4000.0,
                                          key=jax.random.PRNGKey(1), n_starts=6, n_steps=400)
    # the jointly-optimised design beats a shallow and a very-deep fixed depth
    assert res.criterion < oed_depth.crit_at_fixed_depth(m_lim=10.0, target=1, N_total=4000.0)
    assert res.criterion < oed_depth.crit_at_fixed_depth(m_lim=16.0, target=1, N_total=4000.0)
```

**Step 3: Implementation** — `depth_fisher_u(z, u, N)` maps `u=expit^{-1}` → `m_lim ∈ [m_lo, m_hi]`;
`optimize_depth_design` runs multi-start Adam over `[z, u]` (reuse the Stage-1 `optax` scan pattern);
`crit_at_fixed_depth` evaluates the best allocation at a frozen `m_lim`. Do NOT loosen the rtol-1e-4
gate; if the `m_lim` gradient fails it, investigate (likely a non-differentiable `m_grid` lower edge —
keep it smooth).

**Step 5: Commit** (`feat(oed): joint [z, m_lim] optimiser + AD-vs-FD depth-gradient gate`).

---

## Task 5: the interior-optimum result (the headline physics)

**Files:** Modify the depth module; Test `tests/unit/test_demo_oed_depth.py`.

**Step 1: Failing test**

```python
def test_sigma_M_has_interior_optimum_in_depth():
    m_grid = jnp.linspace(M_LO, M_HI, 25)
    sigM = jnp.array([jnp.sqrt(oed_depth.crit_at_fixed_depth(m, target=1, N_total=4000.0))
                      for m in m_grid])            # sigma(M)/M vs depth
    i = int(jnp.argmin(sigM))
    assert 0 < i < len(m_grid) - 1                 # INTERIOR minimum (not at an endpoint)
    assert sigM[i] < sigM[0] and sigM[i] < sigM[-1]
```

**Step 3:** ensure `crit_at_fixed_depth` and the depth grid are exposed; pick `M_LO/M_HI` so the mock
genuinely brackets the optimum (tune in the test; if the min sits at an endpoint, the
star-pool/error normalisation needs adjusting — that is the physics to get right, NOT the test to
weaken). Report the optimal `m_lim` and `σ(M)/M` there.

**Step 5: Commit** (`feat(oed): interior-optimum-in-depth result for sigma(M_dyn)`).

---

## Task 6: magnitude-selected calibration (gate, `@slow`)

**Files:** Modify the depth module; Test `tests/unit/test_demo_oed_depth.py`.

Mirror Stage-1's `calibrate_fisher`, but draw the mock with the **magnitude selection** applied:
sample masses from the Chabrier IMF, keep stars with `m_app ≤ m_lim`, give each a magnitude-dependent
error `photon_noise_error(m_app(mass), …)`, bin, MAP-fit θ, and compare the realised `Var(M̂)/M²` to
`(F⁻¹)_{MM}`. Tolerance `2√(2/n_draws)` (principled MC band, Stage-1 precedent). `@pytest.mark.slow`.
**Do not widen the tolerance to pass** — report diagnostics if it does not close.

**Step 5: Commit** (`feat(oed): magnitude-selected calibration of the depth Fisher (gate)`).

---

## Task 7: gated CLI `demo_oed_dynamical_mass.py`

**Files:** Create `scripts/demo_oed_dynamical_mass.py`; Modify `tests/unit/test_demo_oed_depth.py`.

Follow `scripts/demo_oed.py` (Stage 1) house style: argparse (`--full`, `--seed`, `--n-total`),
compute the joint optimum, print a quantitative summary (optimal `m_lim`; σ(M)/M at optimal vs
shallow vs deep; the interior-optimum factor; calibration), write a JSON run-record, **exit 0** on
gated success (interior optimum exists; AD-vs-FD passes; calibration within band). Calibration uses a
small `n_draws` default, `--full` → 64.

**Step 5: Commit** (`feat(oed): gated CLI demo_oed_dynamical_mass.py + Stage-2 gates`).

---

## Task 8: five Stage-2 figures

**Files:** Modify `scripts/demo_oed_dynamical_mass.py`; visually inspect each PNG.

Save to `docs/website/60-science-demos/optimal-design/figures/` (the section dir, Task 9):
1. `demo_oed2_depth_optimum.png` — σ(M_dyn)/M_dyn vs `m_lim`, interior minimum annotated.
2. `demo_oed2_depth_trade.png` — detectable counts vs `m_lim` (rising) and per-star error vs `m_lim`
   (rising) → the information curve.
3. `demo_oed2_allocation.png` — optimal radial×channel allocation at the optimal depth.
4. `demo_oed2_frontier.png` — σ(M_dyn) vs star budget at the optimal depth.
5. `demo_oed2_calibration.png` — realized vs Fisher σ(M_dyn) (mag-selected mock).

Use `scripts/_plotstyle.py`. **Inspect each** before committing (`feat(oed): Stage-2 figures`).

---

## Task 9: OED section refactor + the Stage-2 page

**Files:**
- `git mv docs/website/60-science-demos/optimal-design.md docs/website/60-science-demos/optimal-design/anisotropy.md`
- `git mv docs/website/60-science-demos/figures/demo_oed_*.{png,json}` → `optimal-design/figures/`
  (update the `{figure}` paths inside the moved page).
- Create `optimal-design/index.md` — move the OED hook + the **"What OED can do with progenax"
  capability map** out of the Stage-1 page into here; add an examples index table.
- Create `optimal-design/background.md` — move the shared formalism (Fisher/CRB, additive backbone,
  dimensionless metric, c/D/A, B&M82 projection geometry) out of the Stage-1 page into here.
- Slim `optimal-design/anisotropy.md` to the Stage-1 *application*, `{ref}`-ing `background.md`.
- Create `optimal-design/dynamical-mass.md` — the Stage-2 page (depth knob, the 5 figures, M_dyn
  result, scope/caveats, how-to-run), `{ref}`-ing `background.md`. Match the docs voice (the
  `myst:docs-writing-voice` standard); cross-link the top-level `anisotropy.md` (B6 recovery demo).
  - **Document every modelling choice with its assumption + rationale + supporting evidence**
    (Anna-mandated). Each approximation gets a short "why this is OK here" paragraph, not just a
    mention. Specifically, an explicit **Assumptions & approximations** subsection covering:
    1. **Bolometric magnitudes** (no band / no bolometric correction / no extinction) — state it,
       justify it for a *pedagogical depth knob* (the headline is the *shape* of the
       information-vs-depth trade, which is set by the IMF×ZAMS supply and the photon-noise scaling,
       both band-independent to first order), and **add a `:::{note}` follow-up callout that the
       real, band-specific photometry (BCs, extinction, crowding) will come from the `fluxax`
       package once it is hardened/finalised** (next arc). Frame current numbers as illustrative.
    2. **Photon-noise error model** `ε ∝ 10^{0.2(m_app−m_ref)}` — illustrative flux⁻⁰·⁵ scaling, not a
       real survey ETC; say so, show the scaling, anchor `ε₀`/`m_ref`.
    3. **Availability soft-cap** `n_eff = avail·tanh(n_design/avail)` — why a differentiable supply
       constraint, what it represents (finite bright-star pool), its `n≪avail`/`n≫avail` limits.
    4. **`ε_eff` is per-channel global** (single cluster distance) vs **availability is radial** —
       explain the geometry that makes this exact for one distance.
    5. **Single-population, mass-follows-light** — why `σ_pred` is `m_lim`-independent (the additive
       backbone survives; `J` computed once).
    Include the supporting evidence already in hand (the interior-optimum result, the AD-vs-FD gate,
    the calibration band) as the quantitative backing for these choices.
- `docs/website/myst.yml` — nest the four pages under an "Optimal experimental design" parent.
- `docs/website/60-science-demos/index.md` — update the Batch row(s) to point at the section.

**Build gate:**
```bash
cd docs/website && env -u VIRTUAL_ENV uv run --no-sync myst build --html   # 0 content warnings
```
Verify every moved `{ref}`/`{numref}`/figure path resolves.

**Step: commit** (`docs(oed): refactor OED section (index/background/anisotropy) + Stage-2 page`).

---

## Task 10: completion + final whole-arc review + gate

1. Full released-core gate green locally:
   ```bash
   XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
     env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
   ```
   (If a new test module trips the **test-backbone dashboard freshness gate**, regenerate:
   `python scripts/build_test_dashboard.py --emit --render` and recommit — the Stage-1 precedent.)
2. `demo_oed_dynamical_mass.py --full` exits 0 + run-record; five figures inspected.
3. `myst build` 0 content warnings (whole section).
4. `.claude-work/OED_DEMO_STAGE2_COMPLETE.md`; STATUS + brain.
5. **Final whole-arc code review** (one reviewer over `_demo_selection.py` + the depth layer + the CLI
   + the section) — the Phase-0.5/Stage-1 lesson that integration review catches what per-task
   reviews miss.
6. **Anna merge-go** → local `main` → push on her word → delete branch.

**Definition of Complete** (design doc): interior optimum in `m_lim` reported; AD-vs-FD on `[z,m_lim]`
passes; mag-selected calibration within band; selection-helper tests; full gate + CLI + section build
all green.
