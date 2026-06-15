# Phase 0 — Differentiable dispersion capability (Implementation Plan)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this
> plan task-by-task (one subagent per task, independent code review between tasks, Anna HITL at the
> phase boundary). Verify LOCALLY; nothing pushed/merged without Anna's explicit go.

**Goal:** Add a packaged, differentiable `dispersion_profile(r, M, G)` method to `PlummerVelocityDF`,
`EFFVelocityDF`, and `MichieVelocityDF` returning the population velocity-dispersion profiles
(σ_r, σ_t, σ_1d, β), closing the verification-found gap (the DFs are samplers and expose no σ getter).
This is the packaged capability the OED demo (Phase 1) rides on.

**Architecture:** A new shared module `src/progenax/kinematics/dispersion.py` holds (a) the
`DispersionProfile` NamedTuple, (b) an anisotropic-Jeans radial quadrature, and (c) an f-table
second-moment kernel. Each DF gets a thin `dispersion_profile()` method. Both methods are validated
against the empirical binned dispersion from the DF's own sampler — a **3-way physics anchor** (Jeans
= f-moment = empirical), with Plummer adding a 4th analytic leg. Differentiability is gated by
AD-vs-FD on ∂σ_r/∂(r_a, M). The self-policing registries (physics, grad_audit, provenance) are updated.

**Tech Stack:** JAX (`jax.numpy`, `jax.grad`/`jacrev`, `jax.lax.scan`), Equinox modules, `jaxtyping`,
`progenax.numerics` (`cumulative_trapezoid`, `inverse_cdf_draw`), pytest 3-tier suite.

**Ratified design:** `docs/plans/2026-06-15-oed-dispersion-arc-design.md`. Phase 0 is gated separately
from Phase 1 (the OED demo): full released-core gate green + Anna merge-go before any Phase-1 code.

---

## Physics reference (the math every task implements)

**Anisotropic Jeans (Osipkov–Merritt), Binney & Tremaine (2008) §4.8.3 + Merritt (1985) Eq. 15.**
With OM anisotropy `β(r) = r²/(r² + r_a²)` the integrating factor is `(r² + r_a²)`, giving

```
ρ(r) σ_r²(r) = 1/(r² + r_a²) · ∫_r^∞ (s² + r_a²) ρ(s) · (G M(<s) / s²) ds
σ_t²(r) = σ_r²(r) · r_a² / (r_a² + r²)        # one tangential component (σ_θ² = σ_φ² = σ_t²)
σ_1d²(r) = (σ_r² + 2 σ_t²) / 3
β(r)     = 1 − σ_t² / σ_r² = r² / (r² + r_a²)  # OM identity; isotropic limit r_a→∞ ⇒ β=0
```

Isotropic (`r_a = None`): integrating factor 1, `σ_t = σ_r = σ_1d`, `β = 0`.

**Plummer closed forms** (`a` = scale radius, stored on the DF):
```
ρ(s) ∝ (1 + s²/a²)^(-5/2)              # normalization cancels in ρσ_r²/ρ
M(<s) = M · s³ / (s² + a²)^(3/2)        # M = total mass (the `M` arg)
```
Isotropic closed form (validation truth): `σ_1d²(r) = G M / (6 √(r² + a²))`.

**EFF / Michie:** ρ(s), Ψ(s), M(<s) come from the DF's **stored grids** (`r_grid`/`xi_grid`,
`Psi_grid`/`psi_grid`); `M(<s) = -s² dΨ/ds` (already used to build the table). The Jeans quadrature
runs on those grids. The **f-table second moment** integrates `v_r²` over the stored `f`-table at each
radius and must agree with Jeans to table resolution.

**β sign convention note:** the OM identity `β = r²/(r²+r_a²)` (above) is the source of truth. The
grad-audit binner `binned_sigma_beta` uses the *Binney* estimator `β = 1 − σ_t²/(2σ_r²)` with σ_t² the
**summed** two-component tangential variance; when comparing to the binner, convert:
`σ_t,sum² = 2 σ_t²` ⇒ `β_Binney = 1 − σ_t,sum²/(2σ_r²) = 1 − σ_t²/σ_r²` = the same β. Keep the
`DispersionProfile.beta` field on the OM identity and assert equality through the conversion.

---

## Task 1 — Scaffold `dispersion.py` (NamedTuple + primitive stubs)

**Files:**
- Create: `src/progenax/kinematics/dispersion.py`
- Test: `tests/unit/kinematics/test_dispersion.py`

**Step 1 — Write the failing test (module imports + NamedTuple shape):**
```python
import jax.numpy as jnp
from progenax.kinematics.dispersion import DispersionProfile, jeans_sigma_r

def test_dispersion_profile_namedtuple_fields():
    dp = DispersionProfile(
        r=jnp.zeros(3), sigma_r=jnp.zeros(3), sigma_t=jnp.zeros(3),
        sigma_1d=jnp.zeros(3), beta=jnp.zeros(3),
    )
    assert dp.sigma_r.shape == (3,)
    assert dp._fields == ("r", "sigma_r", "sigma_t", "sigma_1d", "beta")
```

**Step 2 — Run, expect ImportError/fail:**
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/kinematics/test_dispersion.py -q`
Expected: FAIL (module/`jeans_sigma_r` not defined).

**Step 3 — Minimal implementation:**
```python
"""Differentiable velocity-dispersion profiles for the self-consistent velocity DFs.

Anisotropic-Jeans (Osipkov-Merritt) quadrature + f-table second moment. See
docs/plans/2026-06-15-oed-dispersion-arc-design.md and Binney & Tremaine (2008) sec. 4.8.3,
Merritt (1985) AJ 90, 1027, Eq. 15.
"""
from typing import NamedTuple
import jax.numpy as jnp
from jaxtyping import Array, Float


class DispersionProfile(NamedTuple):
    r: Float[Array, "R"]          # radial grid the profiles are evaluated on
    sigma_r: Float[Array, "R"]    # radial velocity dispersion sigma_r(r)
    sigma_t: Float[Array, "R"]    # one-component tangential dispersion sigma_t(r)
    sigma_1d: Float[Array, "R"]   # 1-D dispersion sqrt((sigma_r^2 + 2 sigma_t^2)/3)
    beta: Float[Array, "R"]       # OM anisotropy beta(r) = 1 - sigma_t^2/sigma_r^2


def _sigma_components(sigma_r2, r, r_a):
    """Given sigma_r^2(r) and OM r_a (or None), return (sigma_r, sigma_t, sigma_1d, beta)."""
    if r_a is None:
        sigma_r = jnp.sqrt(jnp.maximum(sigma_r2, 0.0))
        return sigma_r, sigma_r, sigma_r, jnp.zeros_like(sigma_r)
    ratio = r_a**2 / (r_a**2 + r**2)             # sigma_t^2 / sigma_r^2
    sigma_r = jnp.sqrt(jnp.maximum(sigma_r2, 0.0))
    sigma_t = sigma_r * jnp.sqrt(ratio)
    sigma_1d = jnp.sqrt((sigma_r**2 + 2.0 * sigma_t**2) / 3.0)
    beta = r**2 / (r**2 + r_a**2)
    return sigma_r, sigma_t, sigma_1d, beta


def jeans_sigma_r(r, rho_grid, M_grid, s_grid, G, r_a=None):
    """Anisotropic-Jeans sigma_r^2 at radii r (interpolated from a fine s-grid solution).

    rho_grid, M_grid := rho(s), M(<s) on the ascending s_grid. Returns sigma_r^2(r).
    Reverse cumulative integral of the integrand (s^2+r_a^2) rho GM/s^2, divided by
    (r^2+r_a^2) rho(r). Differentiable (pure jnp + cumulative_trapezoid on flipped grid).
    """
    raise NotImplementedError  # filled in Task 2
```
(Add `dispersion` to `src/progenax/kinematics/__init__.py` imports only if a symbol is re-exported —
**do NOT** add `DispersionProfile` to `kinematics.__all__` or `progenax.__all__`; it is a return type,
not a public entry point, and keeping it out avoids registry churn.)

**Step 4 — Run, expect the NamedTuple test PASS** (the `jeans_sigma_r` import resolves; its body is
unreached). Command as Step 2. Expected: 1 passed.

**Step 5 — Commit:**
```bash
git add src/progenax/kinematics/dispersion.py tests/unit/kinematics/test_dispersion.py
git commit -m "feat(kinematics): scaffold DispersionProfile + jeans primitive (Phase 0 Task 1)
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2 — Plummer isotropic + OM Jeans → `PlummerVelocityDF.dispersion_profile`

**Files:**
- Modify: `src/progenax/kinematics/dispersion.py` (implement `jeans_sigma_r`)
- Modify: `src/progenax/kinematics/plummer_df.py` (add `dispersion_profile`)
- Test: `tests/unit/kinematics/test_dispersion.py`

**Step 1 — Failing tests (analytic truth):**
```python
import jax, jax.numpy as jnp
from progenax.kinematics import PlummerVelocityDF

def test_plummer_isotropic_matches_closed_form():
    G, M, r_h = 0.00449, 400.0, 1.0
    df = PlummerVelocityDF(r_h=r_h)
    r = jnp.array([0.3, 0.7, 1.0, 2.0, 3.0])
    dp = df.dispersion_profile(r, M=M, G=G)
    a = df.a
    sigma_1d_theory = jnp.sqrt(G * M / (6.0 * jnp.sqrt(r**2 + a**2)))
    assert jnp.allclose(dp.sigma_1d, sigma_1d_theory, rtol=2e-3)
    assert jnp.allclose(dp.beta, 0.0, atol=1e-10)
    assert jnp.allclose(dp.sigma_r, dp.sigma_t, rtol=1e-6)

def test_plummer_om_beta_is_om_identity():
    G, M, r_a = 0.00449, 400.0, 2.0
    df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=r_a)
    r = jnp.array([0.5, 1.0, 2.0, 4.0])
    dp = df.dispersion_profile(r, M=M, G=G)
    assert jnp.allclose(dp.beta, r**2 / (r**2 + r_a**2), rtol=1e-6)
    assert jnp.all(dp.sigma_r >= dp.sigma_t)            # radially biased
```

**Step 2 — Run, expect FAIL** (`dispersion_profile` not defined). Command as Task 1 Step 2.

**Step 3 — Implement `jeans_sigma_r` and the Plummer method.**

`jeans_sigma_r` (reverse cumulative quadrature, uniform s-grid):
```python
from progenax.numerics import cumulative_trapezoid

def jeans_sigma_r(r, rho_grid, M_grid, s_grid, G, r_a=None):
    ra2 = jnp.inf if r_a is None else r_a**2
    weight = jnp.where(jnp.isfinite(ra2), s_grid**2 + ra2, 1.0)      # (s^2+r_a^2) or 1
    integrand = weight * rho_grid * G * M_grid / jnp.maximum(s_grid**2, 1e-30)
    dx = s_grid[1] - s_grid[0]
    # reverse cumulative: I(s) = ∫_s^∞ integrand ds  (flip, cumulative, flip back)
    flipped = jnp.flip(integrand)
    I_rev = cumulative_trapezoid(flipped, dx=dx)                     # ∫ from outer edge
    I_outer_to_s = jnp.flip(I_rev)                                  # I(s_grid)
    rho_r = jnp.interp(r, s_grid, rho_grid)
    I_r = jnp.interp(r, s_grid, I_outer_to_s)
    denom = jnp.where(jnp.isfinite(ra2), (r**2 + ra2), 1.0) * jnp.maximum(rho_r, 1e-30)
    return I_r / denom
```

`PlummerVelocityDF.dispersion_profile` (inline analytic ρ, M; fine s-grid to a large outer radius):
```python
def dispersion_profile(self, r, M, G):
    """Population velocity-dispersion profiles sigma_r/sigma_t/sigma_1d/beta at radii r.

    Self-consistent with THIS DF's Plummer potential (mixed-pairing caveat: with non-Plummer
    positions these are the Plummer-self-consistent dispersions, not the position profile's).
    Differentiable in (anisotropy_radius, M, r_h). See progenax.kinematics.dispersion.
    """
    from progenax.kinematics.dispersion import jeans_sigma_r, _sigma_components, DispersionProfile
    r = jnp.atleast_1d(jnp.asarray(r))
    a = self.a
    s = jnp.linspace(1e-4 * a, 30.0 * a, 4000)            # fine radial grid
    rho = (1.0 + (s / a) ** 2) ** (-2.5)                  # unnormalized (cancels)
    M_enc = M * s**3 / (s**2 + a**2) ** 1.5
    r_a = self.anisotropy_radius
    sigma_r2 = jeans_sigma_r(r, rho, M_enc, s, G, r_a=r_a)
    sr, st, s1, beta = _sigma_components(sigma_r2, r, r_a)
    return DispersionProfile(r=r, sigma_r=sr, sigma_t=st, sigma_1d=s1, beta=beta)
```

**Step 4 — Run, expect PASS** (both tests). Command as Step 2. Expected: passed. If isotropic σ_1d is
off, check the s-grid extent (30·a) and the `M(<s)` closed form, NOT the tolerance.

**Step 5 — Commit** (`feat(kinematics): Plummer Jeans dispersion_profile (Phase 0 Task 2)` + trailer).

---

## Task 3 — Plummer 4-way anchor (analytic = Jeans = f-moment = empirical)

**Files:**
- Modify: `src/progenax/kinematics/dispersion.py` (add `ftable_sigma_r` for the analytic OM table)
- Create: `tests/validation/test_dispersion_physics.py`

**Step 1 — Failing physics test (empirical-binned, reuse `test_plummer_physics` pattern):**
```python
import jax, jax.numpy as jnp
from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF

def _empirical_sigma_at(df, r0, M, G, N=200_000, seed=0):
    pos = jnp.tile(jnp.array([r0, 0.0, 0.0]), (N, 1))
    vel = df.sample_velocities(pos, jnp.full((N,), M / N), jax.random.PRNGKey(seed), G=G)
    # radial = x-component at (r0,0,0); tangential = y,z components
    return float(jnp.std(vel[:, 0])), float(jnp.sqrt(0.5 * (jnp.var(vel[:, 1]) + jnp.var(vel[:, 2]))))

def test_plummer_om_jeans_matches_sampler():
    G, M, r_a = 0.00449, 400.0, 2.0
    df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=r_a)
    for r0 in (0.5, 1.0, 2.0):
        dp = df.dispersion_profile(jnp.array([r0]), M=M, G=G)
        emp_r, emp_t = _empirical_sigma_at(df, r0, M, G)
        assert abs(float(dp.sigma_r[0]) - emp_r) / emp_r < 0.05    # 5% MC tol
        assert abs(float(dp.sigma_t[0]) - emp_t) / emp_t < 0.05
```
(Add a sibling `test_plummer_isotropic_*` asserting the analytic σ_1d closed form, and a
`test_plummer_jeans_equals_ftable` comparing `jeans_sigma_r` to `ftable_sigma_r` for the analytic
Merritt OM table stored on the DF (`_om_E_grid`, `_om_f_grid`) to ≤ table resolution.)

**Step 2 — Run, expect FAIL** (`test_dispersion_physics.py` new). Command:
`... uv run --no-sync pytest tests/validation/test_dispersion_physics.py -q`

**Step 3 — Implement `ftable_sigma_r`** (second moment over the stored speed pdf `s²f(Ψ−s²/2)` for the
isotropic/OM split; for Plummer use the dimensionless analytic table + `sigma0² = GM/(6a)` scaling from
`plummer_df.py:191`). Make the empirical test pass by confirming Jeans is correct (it should already
pass from Task 2 — this task ADDS the sampler+f-moment legs of the anchor).

**Step 4 — Run, expect PASS** (all Plummer anchor tests). Note: validation tests are slow (large N);
mark the heaviest `@pytest.mark.slow` if > ~5 s.

**Step 5 — Commit** (`test(validation): Plummer 4-way dispersion anchor (Phase 0 Task 3)` + trailer).

---

## Task 4 — EFF: Jeans-on-stored-grids + f-moment, 3-way anchor

**Files:**
- Modify: `src/progenax/kinematics/eff_df.py` (add `dispersion_profile` using stored
  `r_grid`, `Psi_grid`, `E_grid`, `f_grid`; `M(<s) = -s² dΨ/ds`)
- Modify: `tests/validation/test_dispersion_physics.py`

**Step 1 — Failing tests:** isotropic EFF σ_r matches empirical binned to MC tol; OM EFF β tracks the
OM identity; `jeans_sigma_r` vs `ftable_sigma_r` agree to table resolution (rtol ~ 0.02).

**Step 2 — Run, expect FAIL.** **Step 3 — Implement** (`M_enc = -s² · dPsi/ds` via `jnp.gradient` on
the stored grids; feed `r_grid, rho_grid, M_enc` to `jeans_sigma_r`; `rho_grid = (1+(r/a)²)^(-γ/2)`).
**Step 4 — Run, expect PASS.** **Step 5 — Commit** (`feat(kinematics): EFF dispersion_profile + anchor (Phase 0 Task 4)`).

---

## Task 5 — Michie: Jeans + f-moment, 3-way anchor + isotropic→King limit

**Files:**
- Modify: `src/progenax/kinematics/michie_df.py` (add `dispersion_profile` using `xi_grid`,
  `psi_grid`, `mu`; physical scaling `σ² ∝ G M / (r_c μ)` per the class's own normalization)
- Modify: `tests/validation/test_dispersion_physics.py`

**Step 1 — Failing tests:** Michie σ_r matches empirical binned; β increases outward
(`beta[-1] > beta[0]`); large-`r_a` isotropic limit → β ≈ 0; Jeans vs f-moment agree to table res.
**Steps 2–4** as Task 4. **Step 5 — Commit** (`feat(kinematics): Michie dispersion_profile + anchor (Phase 0 Task 5)`).

---

## Task 6 — Differentiability (AD-vs-FD), all three DFs

**Files:**
- Modify: `tests/unit/kinematics/test_dispersion.py`

**Step 1 — Failing grad tests** (finite-difference consistency, reverse-mode only):
```python
import jax, jax.numpy as jnp
from progenax.kinematics import PlummerVelocityDF

def _sigma_r_sum(r_a):
    df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=r_a)
    return jnp.sum(df.dispersion_profile(jnp.array([0.5, 1.0, 2.0]), M=400.0, G=0.00449).sigma_r)

def test_dispersion_grad_wrt_r_a_matches_fd():
    ra0 = 2.0
    g_ad = jax.grad(_sigma_r_sum)(ra0)
    h = 1e-4 * ra0
    g_fd = (_sigma_r_sum(ra0 + h) - _sigma_r_sum(ra0 - h)) / (2 * h)
    assert abs(g_ad - g_fd) / (abs(g_fd) + 1e-12) < 1e-3
    assert abs(g_ad) > 1e-9                       # not a silent zero
```
(Add an analogous `∂σ_r/∂M` test, and EFF/Michie variants in `r_a`/`anisotropy_radius`.)

**Step 2 — Run, expect FAIL** (if any path is non-differentiable — e.g. a `jnp.gradient` edge or an
`interp` clamp — FIX the physics/numerics, never loosen `tol`). **Step 3** — fix as needed.
**Step 4 — Run, expect PASS.** **Step 5 — Commit** (`test(kinematics): AD-vs-FD dispersion grads (Phase 0 Task 6)`).

---

## Task 7 — Self-policing registry updates

**Files:**
- Modify: `tests/validation/physics_registry/manifest.py` — add one invariant line to EACH of the three
  DF dicts (`PlummerVelocityDF`, `EFFVelocityDF`, `MichieVelocityDF`) pointing at the new anchor tests,
  e.g.:
  ```python
  "OM dispersion getter sigma_r/sigma_t(r): Jeans = f-moment = empirical-binned (3/4-way anchor)":
      "tests/validation/test_dispersion_physics.py::test_plummer_om_jeans_matches_sampler",
  ```
- Modify: `tests/validation/grad_audit/manifest.py` — add `MUST_AUDIT` entries (keyed `(id, param)`):
  ```python
  ("PlummerVelocityDF.dispersion_profile", "anisotropy_radius"): "OM dispersion getter sigma_r in r_a",
  ("PlummerVelocityDF.dispersion_profile", "M"): "dispersion getter sigma_r in total mass",
  ("EFFVelocityDF.dispersion_profile", "anisotropy_radius"): "EFF Jeans dispersion in r_a",
  ("MichieVelocityDF.dispersion_profile", "r_a"): "Michie Jeans dispersion in r_a",
  ```
  (SYMBOL_CATEGORY unchanged — the DF symbols are already AUDITED.)
- Modify: `tests/validation/grad_audit/registry.py` — add the matching `Case(...)` entries (use the
  `core.py` `Case` dataclass; `direction="params->summary"`, `reduce=identity_sum`, `tol=1e-3`,
  `expect="consistent"`; mirror the existing `PlummerVelocityDF+OM.sample_velocities` case at
  lines ~1483).
- Modify: `tests/validation/provenance_registry/manifest.py` — add a PROVENANCE row:
  ```python
  "kinematics/dispersion.py::anisotropic-Jeans sigma_r quadrature (OM integrating factor r^2+r_a^2)":
      "Binney & Tremaine (2008) Galactic Dynamics sec. 4.8.3 (Jeans eqs.) + Merritt (1985) AJ 90, "
      "1027, Eq. 15 (OM sigma_r^2/sigma_t^2 = 1 + r^2/r_a^2) — derivable identity.",
  ```
  (No `ALLOWLIST_MODULES` add needed: `dispersion.py` carries no fitted coefficients, only the formula
  + numerical guards. If the coverage test flags a literal, add it to `ALLOWLIST_NON_COEFFICIENT` with a
  reason.)

**Step — Run the four coverage tests + grad audit:**
```
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest \
  tests/validation/api_coverage tests/validation/physics_registry \
  tests/validation/provenance_registry tests/validation/grad_audit -q
```
Expected: PASS (every new `MUST_AUDIT` (id,param) has a covering `Case`; every DF invariant points at a
real asserting test). If a coverage test reds, the manifest/registry is incomplete — fix it, don't skip.

**Step — Commit** (`test(registries): register dispersion physics/grad/provenance (Phase 0 Task 7)`).

---

## Task 8 — Docstrings, mixed-pairing caveat, FULL gate, completion doc

**Files:**
- Modify: the three DF `dispersion_profile` docstrings — state the **mixed-pairing caveat** (returns the
  dispersion self-consistent with the DF's own potential), the units (`M`, `G` explicit), and
  differentiability (reverse-mode; forward-mode forbidden by diffrax `custom_vjp`).
- Create: `.claude-work/PHASE0_DISPERSION_COMPLETE.md` (files, API, anchor results table, grad results,
  lessons, Phase-1 handoff).

**Step — Run the FULL released-core gate** (the merge gate):
```
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
```
Expected: ALL PASS (~1243 + the new dispersion tests). Capture the count into the completion doc.

**Step — Commit** (`docs(phase0): dispersion capability complete + run-record` + trailer), update
`STATUS.md` (`next:` Phase 1 OED demo), `brain "..."` capture, then **PAUSE at the Anna HITL
checkpoint** — present the gate output and await explicit merge-go before any Phase-1 code.

---

## Definition of Complete (Phase 0)

- [ ] `dispersion_profile(r, M, G)` on Plummer/EFF/Michie; `DispersionProfile` NamedTuple (not in `__all__`).
- [ ] Plummer 4-way anchor (analytic = Jeans = f-moment = empirical) passes.
- [ ] EFF + Michie 3-way anchor (Jeans = f-moment = empirical) passes; Michie isotropic→King limit.
- [ ] AD-vs-FD grads consistent for ∂σ_r/∂(r_a, M), all three; no silent zeros.
- [ ] Four registries updated; coverage tests green.
- [ ] FULL released-core gate green (count captured).
- [ ] Completion doc + STATUS + brain; **Anna merge-go obtained** before Phase 1.
