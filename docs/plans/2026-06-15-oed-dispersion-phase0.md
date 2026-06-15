# Phase 0 — Differentiable dispersion capability (Implementation Plan)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this
> plan task-by-task (one subagent per task, independent code review between tasks, Anna HITL at the
> phase boundary). Verify LOCALLY; nothing pushed/merged without Anna's explicit go.

**Goal:** Add a packaged, differentiable `jeans_dispersion(profile, r_a, r, M, G)` returning the
population velocity-dispersion profiles (σ_r, σ_t, σ_1d, β) for any spatial profile under Osipkov–Merritt
anisotropy — closing the verification-found gap (the DFs are samplers and expose no σ getter). This is
the packaged forward model the OED demo (Phase 1) rides on.

**Architecture (revised 2026-06-15, Anna):** A **free function** in
`src/progenax/kinematics/dispersion.py`, exported in `progenax.__all__`. The dispersion is a property
of the *(potential, anisotropy)* pair, so it lives with the **profile** (which owns ρ, M, Φ), not the
DF — this eliminates the ρ/M duplication and the mixed-pairing footgun (you pass the *actual* profile),
and decouples the forward model from the stochastic sampler. The enclosed mass `M(<s)` comes from a
**quadrature of `profile.density`** (builder-quality, no re-differentiated Ψ). The DF's stored `f`-table
second moment is a **cross-check** (Anna's "both, cross-checked": Jeans primary, f-moment gated to agree
to table resolution). Validation is a **3-way anchor** (Jeans = f-moment = empirical-binned), Plummer
4-way (+ analytic closed form). Differentiability gated by AD-vs-FD on ∂σ_r/∂(r_a, M).

**Tech Stack:** JAX (`jax.numpy`, `jax.grad`/`jacrev`), Equinox, `jaxtyping`, `progenax.numerics`
(`cumulative_trapezoid`), pytest 3-tier suite. **Zero new deps.**

**Ratified design:** `docs/plans/2026-06-15-oed-dispersion-arc-design.md`. Phase 0 is gated separately
from Phase 1: FULL released-core gate green + Anna merge-go before any Phase-1 code.

---

## Physics reference (the math every task implements)

**Anisotropic Jeans (Osipkov–Merritt), Binney & Tremaine (2008) §4.8.3 + Merritt (1985) Eq. 15.**
With `β(r)=r²/(r²+r_a²)` the integrating factor is `(r²+r_a²)`:

```
ρ(r) σ_r²(r) = 1/(r²+r_a²) · ∫_r^∞ (s²+r_a²) ρ(s) · (G M(<s)/s²) ds
σ_t²(r) = σ_r²(r) · r_a²/(r_a²+r²)            # one tangential component (σ_θ²=σ_φ²=σ_t²)
σ_1d²(r) = (σ_r² + 2σ_t²)/3
β(r)     = 1 − σ_t²/σ_r² = r²/(r²+r_a²)       # r_a=None ⇒ factor 1, σ_r=σ_t=σ_1d, β=0
```
(Re-derived from d(ρσ_r²)/dr + (2β/r)ρσ_r² = −ρGM/r²; integrating factor exp(∫2β/r dr)=r²+r_a².
The boundary term at ∞ vanishes; at r=0 the (r²+r_a²)→r_a² denominator is finite, so no singularity.)

**Enclosed mass (all profiles, no Ψ re-differentiation):**
```
M(<s) = M · cumtrap(ρ(s)·s²) / cumtrap_total(ρ(s)·s²)   # ρ = profile.density(s), normalization cancels
```
For Plummer this equals the closed form `M·s³/(s²+a²)^{3/2}` analytically. Validation truth (isotropic
Plummer): `σ_1d²(r) = GM/(6√(r²+a²))`.

**β convention vs the grad-audit binner:** `DispersionProfile.beta` uses the OM identity
`β=r²/(r²+r_a²)`. `binned_sigma_beta` uses Binney `β=1−σ_t,sum²/(2σ_r²)` with σ_t,sum²=2σ_t²; both
reduce to `1−σ_t²/σ_r²` — assert equality through that conversion.

**Integration grid extent:** `r_max = getattr(profile, "r_t", None)` for truncated profiles
(EFF/King/Michie); for Plummer (no `r_t`) use `r_max = 30·profile.a`. Query radii `r` must lie within
`[s_min, r_max]` (else `jnp.interp` clamps → wrong σ with a silent-zero gradient); assert/clip in-range.

---

## Task 1 — Scaffold `dispersion.py` (NamedTuple + free-fn signature) + exports

**Files:**
- Create: `src/progenax/kinematics/dispersion.py`
- Modify: `src/progenax/kinematics/__init__.py` (add `jeans_dispersion`, `DispersionProfile` to its `__all__`)
- Modify: `src/progenax/__init__.py` (re-export `jeans_dispersion` into `progenax.__all__`)
- Test: `tests/unit/kinematics/test_dispersion.py`

**Step 1 — Failing test:**
```python
import jax.numpy as jnp
import progenax
from progenax import jeans_dispersion
from progenax.kinematics.dispersion import DispersionProfile

def test_jeans_dispersion_exported_and_namedtuple():
    assert "jeans_dispersion" in progenax.__all__
    dp = DispersionProfile(r=jnp.zeros(3), sigma_r=jnp.zeros(3), sigma_t=jnp.zeros(3),
                           sigma_1d=jnp.zeros(3), beta=jnp.zeros(3))
    assert dp._fields == ("r", "sigma_r", "sigma_t", "sigma_1d", "beta")
```

**Step 2 — Run, expect FAIL** (import error / not in `__all__`):
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/kinematics/test_dispersion.py -q`

**Step 3 — Minimal implementation** (`dispersion.py`): the `DispersionProfile` NamedTuple, a
`_sigma_components(sigma_r2, r, r_a)` helper (returns σ_r/σ_t/σ_1d/β per the physics ref), a
`jeans_sigma_r(...)` quadrature stub raising `NotImplementedError`, and `jeans_dispersion(profile, r_a,
r, M, G)` raising `NotImplementedError`. Wire both into `kinematics/__init__.py:__all__` and re-export
`jeans_dispersion` from `src/progenax/__init__.py` (follow the existing kinematics re-export pattern;
do **not** export `DispersionProfile` at top level — it is a return type, not an entry point).

**Step 4 — Run, expect PASS** (export + NamedTuple resolve; stub bodies unreached). Command as Step 2.

**Step 5 — Commit:**
```bash
git add src/progenax/kinematics/dispersion.py src/progenax/kinematics/__init__.py \
        src/progenax/__init__.py tests/unit/kinematics/test_dispersion.py
git commit -m "feat(kinematics): scaffold jeans_dispersion + DispersionProfile (Phase 0 Task 1)
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2 — Implement `jeans_dispersion` (Plummer truth) — works for any profile

**Files:**
- Modify: `src/progenax/kinematics/dispersion.py`
- Test: `tests/unit/kinematics/test_dispersion.py`

**Step 1 — Failing tests (Plummer analytic truth + OM β identity):**
```python
import jax.numpy as jnp
from progenax import jeans_dispersion
from progenax.profiles import PlummerProfile

def test_plummer_isotropic_matches_closed_form():
    G, M = 0.00449, 400.0
    prof = PlummerProfile(r_h=1.0)
    r = jnp.array([0.3, 0.7, 1.0, 2.0])
    dp = jeans_dispersion(prof, None, r, M, G)
    sigma_1d_theory = jnp.sqrt(G * M / (6.0 * jnp.sqrt(r**2 + prof.a**2)))
    assert jnp.allclose(dp.sigma_1d, sigma_1d_theory, rtol=3e-3)
    assert jnp.allclose(dp.beta, 0.0, atol=1e-10)
    assert jnp.allclose(dp.sigma_r, dp.sigma_t, rtol=1e-6)

def test_plummer_om_beta_identity_and_radial_bias():
    G, M, r_a = 0.00449, 400.0, 2.0
    prof = PlummerProfile(r_h=1.0)
    r = jnp.array([0.5, 1.0, 2.0, 4.0])
    dp = jeans_dispersion(prof, r_a, r, M, G)
    assert jnp.allclose(dp.beta, r**2 / (r**2 + r_a**2), rtol=1e-6)
    assert jnp.all(dp.sigma_r >= dp.sigma_t)
```

**Step 2 — Run, expect FAIL** (NotImplementedError).

**Step 3 — Implement** `jeans_sigma_r` (reverse cumulative quadrature on a uniform s-grid) and
`jeans_dispersion`:
```python
from progenax.numerics import cumulative_trapezoid

def jeans_sigma_r(r, rho, M_enc, s, G, r_a=None):
    ra_finite = r_a is not None
    fac = (s**2 + r_a**2) if ra_finite else 1.0
    integrand = fac * rho * G * M_enc / jnp.maximum(s**2, 1e-30)
    dx = s[1] - s[0]
    I_rev = cumulative_trapezoid(jnp.flip(integrand), dx=dx)     # ∫ from outer edge inward
    I_of_s = jnp.flip(I_rev)                                     # I(s) = ∫_s^∞ integrand ds
    rho_r = jnp.interp(r, s, rho)
    I_r = jnp.interp(r, s, I_of_s)
    denom = ((r**2 + r_a**2) if ra_finite else 1.0) * jnp.maximum(rho_r, 1e-30)
    return I_r / denom

def jeans_dispersion(profile, r_a, r, M, G):
    """Population velocity-dispersion profiles (sigma_r/sigma_t/sigma_1d/beta) for `profile`
    under Osipkov-Merritt anisotropy r_a (None = isotropic). Forward model for kinematic
    inference; differentiable in (r_a, M, profile params). See module docstring + B&T 4.8.3."""
    r = jnp.atleast_1d(jnp.asarray(r))
    r_max = getattr(profile, "r_t", None)
    r_max = 30.0 * profile.a if r_max is None else r_max         # Plummer has no r_t
    s = jnp.linspace(1e-4 * r_max, r_max, 4000)
    rho = profile.density(s)
    cum = cumulative_trapezoid(rho * s**2, dx=s[1] - s[0])
    M_enc = M * cum / jnp.maximum(cum[-1], 1e-30)
    sigma_r2 = jeans_sigma_r(r, rho, M_enc, s, G, r_a=r_a)
    sr, st, s1, beta = _sigma_components(sigma_r2, r, r_a)
    return DispersionProfile(r=r, sigma_r=sr, sigma_t=st, sigma_1d=s1, beta=beta)
```

**Step 4 — Run, expect PASS.** If isotropic σ_1d is biased, check the s-grid extent (30·a) and the
`M_enc` normalization — NOT the tolerance.

**Step 5 — Commit** (`feat(kinematics): jeans_dispersion forward model (Phase 0 Task 2)` + trailer).

---

## Task 3 — Plummer 4-way anchor (analytic = Jeans = f-moment = empirical)

**Files:**
- Modify: `src/progenax/kinematics/dispersion.py` (add `ftable_sigma_r(E_grid, f_grid, Psi_r, r, r_a, ...)`
  — second moment of `s²f(Ψ−s²/2)` for the DF cross-check)
- Create: `tests/validation/test_dispersion_physics.py`

**Step 1 — Failing physics tests** (empirical-binned, reuse the `test_plummer_physics.py:166` pattern —
sample N≈2e5 at fixed radii, compare σ_r=std(v_x), σ_t from var(v_y),var(v_z)):
```python
def test_plummer_om_jeans_matches_sampler():
    G, M, r_a = 0.00449, 400.0, 2.0
    prof = PlummerProfile(r_h=1.0); df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=r_a)
    for r0 in (0.5, 1.0, 2.0):
        dp = jeans_dispersion(prof, r_a, jnp.array([r0]), M, G)
        emp_r, emp_t = _empirical_sigma_at(df, r0, M, G)   # helper as in design plan
        assert abs(float(dp.sigma_r[0]) - emp_r) / emp_r < 0.05
        assert abs(float(dp.sigma_t[0]) - emp_t) / emp_t < 0.05
```
Plus `test_plummer_jeans_equals_ftable` (Jeans vs `ftable_sigma_r` on the DF's analytic OM table
`_om_E_grid/_om_f_grid`, scaled by `sigma0²=GM/(6a)`) to table resolution.

**Step 2 — Run, expect FAIL.** **Step 3 — implement `ftable_sigma_r`** (makes the f-moment leg pass;
Jeans leg already passes from Task 2). **Step 4 — Run, expect PASS** (mark heaviest tests `@pytest.mark.slow`).
**Step 5 — Commit** (`test(validation): Plummer 4-way dispersion anchor (Phase 0 Task 3)`).

---

## Task 4 — EFF 3-way anchor (Jeans via EFFProfile + f-moment via EFF DF table)

**Files:** Modify `tests/validation/test_dispersion_physics.py` (+ `ftable_sigma_r` call wiring for the
EFF DF's stored `E_grid/f_grid/Psi_grid`).

**Note:** `jeans_dispersion(EFFProfile(...), r_a, ...)` **already works** from Task 2 (EFFProfile has
`density`). This task ADDS the EFF DF f-moment cross-check + the empirical anchor.

**Step 1 — Failing tests:** isotropic EFF σ_r matches empirical binned (MC tol 5%); OM EFF β tracks the
identity; `jeans_sigma_r` vs `ftable_sigma_r` agree to table resolution (rtol ~ 0.02) — **and** a
resolution-refinement check that the Jeans-vs-f-moment gap *shrinks* with grid size (the discriminator
that the residual is numerical, not a physics bug — do NOT loosen tol to pass).
**Steps 2–5** as Task 3 (`feat/test: EFF dispersion anchor (Phase 0 Task 4)`).

---

## Task 5 — Michie 3-way anchor + isotropic→King limit

**Files:** Modify `tests/validation/test_dispersion_physics.py`.

`jeans_dispersion(MichieProfile(...), r_a, ...)` works from Task 2 (MichieProfile has `density`).
**Step 1 — Failing tests:** Michie σ_r matches empirical binned; β increases outward
(`beta[-1] > beta[0]`); large-`r_a` limit → β ≈ 0; Jeans vs f-moment agree to table resolution.
**Steps 2–5** as Task 4 (`test: Michie dispersion anchor + King limit (Phase 0 Task 5)`).

---

## Task 6 — Differentiability (AD-vs-FD)

**Files:** Modify `tests/unit/kinematics/test_dispersion.py`.

**Step 1 — Failing grad tests** (reverse-mode; FD-consistent; not silent-zero):
```python
import jax
def _sig_r_sum(r_a):
    prof = PlummerProfile(r_h=1.0)
    return jnp.sum(jeans_dispersion(prof, r_a, jnp.array([0.5, 1.0, 2.0]), 400.0, 0.00449).sigma_r)

def test_grad_wrt_r_a_matches_fd():
    ra0 = 2.0; h = 1e-4 * ra0
    g_ad = jax.grad(_sig_r_sum)(ra0)
    g_fd = (_sig_r_sum(ra0 + h) - _sig_r_sum(ra0 - h)) / (2 * h)
    assert abs(g_ad - g_fd) / (abs(g_fd) + 1e-12) < 1e-3
    assert abs(g_ad) > 1e-9
```
Plus `∂/∂M` and EFF/Michie variants (and `∂/∂r_h`/`∂/∂gamma`/`∂/∂W0` through the profile params).
**Step 2 — Run, expect FAIL if any path is non-diff** (e.g. an `interp` clamp or a `where` dead branch
— FIX the numerics, never loosen `tol`). **Steps 3–5** (`test(kinematics): AD-vs-FD dispersion grads (Phase 0 Task 6)`).

---

## Task 7 — Self-policing registry updates (4 registries; `jeans_dispersion` is a NEW `__all__` symbol)

**Files:**
- `tests/validation/api_coverage/manifest.py` — add to `SYMBOL_TESTS`:
  ```python
  "jeans_dispersion": "tests/unit/kinematics/test_dispersion.py::test_plummer_isotropic_matches_closed_form",  # assert sigma_1d == GM/(6 sqrt(r^2+a^2))
  ```
- `tests/validation/physics_registry/manifest.py` — add to `EXEMPT_NON_MODEL`:
  ```python
  "jeans_dispersion": "anisotropic-Jeans dispersion forward model (sigma_r/sigma_t(r) for a profile+OM r_a); a forward-model helper, not an equilibrium model. Physics anchored in test_dispersion_physics.py.",
  ```
  Also add ONE invariant line to each DF dict (`PlummerVelocityDF`/`EFFVelocityDF`/`MichieVelocityDF`)
  for the 3/4-way anchor pointing at the new validation tests.
- `tests/validation/grad_audit/manifest.py` — add `"jeans_dispersion": AUDITED` to `SYMBOL_CATEGORY`,
  and `MUST_AUDIT` entries:
  ```python
  ("jeans_dispersion[Plummer+OM]", "r_a"): "OM Jeans dispersion sigma_r in anisotropy radius",
  ("jeans_dispersion[Plummer]", "M"): "Jeans dispersion sigma_r in total mass",
  ("jeans_dispersion[EFF+OM]", "r_a"): "EFF Jeans dispersion in r_a",
  ("jeans_dispersion[Michie+OM]", "r_a"): "Michie Jeans dispersion in r_a",
  ```
- `tests/validation/grad_audit/registry.py` — add matching `Case(...)` entries (`core.py` `Case`:
  `direction="params->summary"`, `fn=closure(theta)->sigma_r array`, `param`, `theta0`,
  `reduce=identity_sum`, `expect="consistent"`, `tol=1e-3`; mirror the `PlummerVelocityDF+OM` case ~L1483).
- `tests/validation/provenance_registry/manifest.py` — add to `PROVENANCE`:
  ```python
  "kinematics/dispersion.py::anisotropic-Jeans sigma_r quadrature (OM integrating factor r^2+r_a^2)":
      "Binney & Tremaine (2008) Galactic Dynamics sec. 4.8.3 + Merritt (1985) AJ 90, 1027, Eq. 15 "
      "(OM sigma_r^2/sigma_t^2 = 1 + r^2/r_a^2) — derivable identity.",
  ```
  (If the coverage scanner flags a numeric literal in `dispersion.py`, add it to
  `ALLOWLIST_NON_COEFFICIENT` with a reason — `4000` grid size, `30.0` Plummer extent, `1e-30` guards.)

**Step — Run the four coverage suites + grad audit:**
```
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest \
  tests/validation/api_coverage tests/validation/physics_registry \
  tests/validation/provenance_registry tests/validation/grad_audit -q
```
Expected: PASS (every new `MUST_AUDIT` (id,param) has a covering `Case`; the new `__all__` symbol is
categorized in all four). If red, the manifest is incomplete — fix it, don't skip.
**Step — Commit** (`test(registries): register jeans_dispersion physics/grad/provenance (Phase 0 Task 7)`).

---

## Task 8 — Docstrings, FULL gate, completion doc, HITL pause

**Files:**
- `dispersion.py` docstring — state units (`M`, `G` explicit), differentiability (reverse-mode;
  forward-mode forbidden by diffrax `custom_vjp`), and that it returns the **equilibrium dispersion of
  `profile` under OM `r_a`** (no mixed-pairing ambiguity — the caller supplies the profile).
- Create `.claude-work/PHASE0_DISPERSION_COMPLETE.md` (files, API, anchor results table, grad results,
  lessons, Phase-1 handoff).

**Step — FULL released-core gate (merge gate):**
```
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
```
Expected: ALL PASS (~1243 + new dispersion tests; capture the count). **Step — Commit**
(`docs(phase0): dispersion capability complete + run-record`), update `STATUS.md` (`next:` Phase 1),
`brain "..."` capture, then **PAUSE at the Anna HITL checkpoint** — present the gate output and await
explicit merge-go before any Phase-1 code.

---

## Definition of Complete (Phase 0)

- [ ] `jeans_dispersion(profile, r_a, r, M, G)` in `progenax.__all__`; `DispersionProfile` NamedTuple (not top-level).
- [ ] Plummer 4-way anchor (analytic = Jeans = f-moment = empirical) passes.
- [ ] EFF + Michie 3-way anchor passes; Jeans-vs-f-moment gap shown to shrink with resolution (numerical, not a bug); Michie isotropic→King limit.
- [ ] AD-vs-FD grads consistent for ∂σ_r/∂(r_a, M, profile params), all three; no silent zeros.
- [ ] Four registries updated (incl. the new `__all__` symbol); coverage suites green.
- [ ] FULL released-core gate green (count captured).
- [ ] Completion doc + STATUS + brain; **Anna merge-go obtained** before Phase 1.
