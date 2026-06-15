# Phase 0 — Differentiable dispersion capability (Implementation Plan)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this
> plan task-by-task (one subagent per task, independent code review between tasks, Anna HITL at the
> phase boundary). Use research-workflow:numerical-method-validation for the convergence study (Task 3).
> Verify LOCALLY; nothing pushed/merged without Anna's explicit go.

**Goal:** Add two packaged, differentiable forward models for cluster kinematics: `jeans_dispersion`
(3-D σ_r/σ_t/σ_1d/β via anisotropic Jeans) and `project_dispersion` (the OBSERVED σ_los/σ_pm,R/σ_pm,T
via Binney & Mamon 1982 line-of-sight projection). Closes the verification-found gap (DFs are samplers,
expose no σ getter) and makes the OED's "RV↔σ_los, PM↔σ_pm" claim physically honest / Gaia-ready.

**Architecture (revised 2026-06-15, Anna):** Free functions in
`src/progenax/kinematics/dispersion.py`, both exported in `progenax.__all__`. The dispersion is a
property of the *(potential, anisotropy)* pair, so it lives with the **profile** (owns ρ, M, Φ), not
the DF — eliminating ρ/M duplication and the mixed-pairing footgun, and decoupling the forward model
from the stochastic sampler. Enclosed mass `M(<s)` is a **quadrature of `profile.density`**
(builder-quality, no re-differentiated Ψ). The DF's stored `f`-table second moment is an
**isotropic-only** cross-check. Validation: a **3-way anchor** (Jeans = isotropic-f-moment =
empirical) + a tight **analytic Merritt OM-Plummer oracle** + a **quadrature convergence study**.
`project_dispersion` adds the B&M82 LOS integrals (singularity removed by `r²=R²+u²`).

**Tech Stack:** JAX (`jax.numpy`, `jax.grad`/`jacrev`, `jax.jit`), Equinox, `jaxtyping`,
`progenax.numerics` (`cumulative_trapezoid`), pytest 3-tier. **Zero new deps.**

**Ratified design:** `docs/plans/2026-06-15-oed-dispersion-arc-design.md`. Phase 0 is gated separately:
FULL released-core gate green + Anna merge-go before any Phase-1 code.

---

## Physics reference

**Anisotropic Jeans (B&T 2008 §4.8.3; Merritt 1985 Eq. 15), OM `β=r²/(r²+r_a²)`:**
```
ρ σ_r²(r) = 1/(r²+r_a²) · ∫_r^∞ (s²+r_a²) ρ(s) G M(<s)/s² ds
σ_t² = σ_r²·r_a²/(r_a²+r²);  σ_1d² = (σ_r²+2σ_t²)/3;  β = r²/(r²+r_a²)
M(<s) = M·cumtrap(ρ s²)/cumtrap_total(ρ s²)          # any profile.density, no dΨ/ds
isotropic Plummer truth: σ_1d²(r) = GM/(6√(r²+a²))
```

**Projection — Binney & Mamon (1982), ν=ρ (tracer = mass here):**
```
Σ(R)           = 2 ∫_R^∞ ρ                       r/√(r²−R²) dr
Σ σ_los²(R)    = 2 ∫_R^∞ (1 − β R²/r²)   ρ σ_r²  r/√(r²−R²) dr     # RV channel
Σ σ_pm,R²(R)   = 2 ∫_R^∞ (1 − β + β R²/r²) ρ σ_r² r/√(r²−R²) dr     # PM, on-sky radial
Σ σ_pm,T²(R)   = 2 ∫_R^∞ (1 − β)         ρ σ_r²  r/√(r²−R²) dr     # PM, on-sky tangential
```
Singularity removed by `r²=R²+u²` ⇒ `∫_R^∞ g(r) r/√(r²−R²) dr = ∫_0^{√(r_max²−R²)} g(√(R²+u²)) du`.
**Isotropic check (β=0): σ_los = σ_pm,R = σ_pm,T = σ_1d.** Anisotropy is in the ratios.

**Grid extent:** `r_max = getattr(profile,"r_t",None)`; Plummer → `30·profile.a`. Query r/R must lie in
`[s_min, r_max]` (else `jnp.interp` clamps → silent-zero gradient): assert/clip in-range.
**β-convention vs binner:** `beta` uses `r²/(r²+r_a²)`; convert to Binney `1−σ_t,sum²/(2σ_r²)` when
comparing to `binned_sigma_beta`.

---

## Task 1 — Scaffold (two NamedTuples, two free-fn stubs, exports)

**Files:** Create `src/progenax/kinematics/dispersion.py`; modify `kinematics/__init__.py` +
`src/progenax/__init__.py` (export `jeans_dispersion`, `project_dispersion`); test
`tests/unit/kinematics/test_dispersion.py`.

**Step 1 — Failing test:**
```python
import progenax
from progenax import jeans_dispersion, project_dispersion
from progenax.kinematics.dispersion import DispersionProfile, ProjectedDispersion

def test_exports_and_namedtuples():
    assert {"jeans_dispersion", "project_dispersion"} <= set(progenax.__all__)
    assert DispersionProfile._fields == ("r", "sigma_r", "sigma_t", "sigma_1d", "beta")
    assert ProjectedDispersion._fields == ("R", "sigma_los", "sigma_pm_r", "sigma_pm_t", "Sigma")
```
**Step 2 — Run, expect FAIL.** Command:
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/kinematics/test_dispersion.py -q`
**Step 3 — Implement** both NamedTuples, `_sigma_components`, and `jeans_dispersion`/`project_dispersion`
stubs (`raise NotImplementedError`); wire exports (do NOT export the NamedTuples at top level).
**Step 4 — Run, expect PASS.** **Step 5 — Commit** (`feat(kinematics): scaffold dispersion + projection (Phase 0 Task 1)` + trailer).

---

## Task 2 — `jeans_dispersion` (3-D) + invariants + domain guard + jit

**Files:** Modify `dispersion.py`; test `tests/unit/kinematics/test_dispersion.py`.

**Step 1 — Failing tests:**
```python
import jax, jax.numpy as jnp
from progenax import jeans_dispersion
from progenax.profiles import PlummerProfile

def test_plummer_isotropic_closed_form():
    prof = PlummerProfile(r_h=1.0); r = jnp.array([0.3,0.7,1.0,2.0])
    dp = jeans_dispersion(prof, None, r, M=400.0, G=0.00449)
    truth = jnp.sqrt(0.00449*400.0/(6.0*jnp.sqrt(r**2+prof.a**2)))
    assert jnp.allclose(dp.sigma_1d, truth, rtol=3e-3)
    assert jnp.allclose(dp.beta, 0.0, atol=1e-10) and jnp.allclose(dp.sigma_r, dp.sigma_t, rtol=1e-6)

def test_gm_scaling_invariants():
    prof = PlummerProfile(r_h=1.0); r = jnp.array([1.0])
    s1 = jeans_dispersion(prof, 2.0, r, 400.0, 0.00449).sigma_r
    assert jnp.allclose(jeans_dispersion(prof, 2.0, r, 800.0, 0.00449).sigma_r**2, 2*s1**2, rtol=1e-4)
    assert jnp.allclose(jeans_dispersion(prof, 2.0, r, 400.0, 2*0.00449).sigma_r**2, 2*s1**2, rtol=1e-4)

def test_r_a_domain_guard():
    import pytest
    prof = PlummerProfile(r_h=1.0)
    with pytest.raises(ValueError):           # r_a < 0.75 a is unphysical for Plummer OM
        jeans_dispersion(prof, 0.1*prof.a, jnp.array([1.0]), 400.0, 0.00449)

def test_jit_smoke():
    prof = PlummerProfile(r_h=1.0)
    f = jax.jit(lambda ra: jeans_dispersion(prof, ra, jnp.array([1.0]), 400.0, 0.00449).sigma_r)
    assert jnp.isfinite(f(2.0)).all()
```
**Step 2 — Run, expect FAIL.** **Step 3 — Implement** `jeans_sigma_r` (reverse `cumulative_trapezoid`
on `jnp.flip(integrand)`) and `jeans_dispersion` (profile.density quadrature for `M_enc`; grid extent
per `r_t`/`30a`; the `r_a < 0.75a` guard is eager for a concrete `r_a`, skipped under tracing — mirror
`plummer_df.py:128`). **Step 4 — Run, expect PASS** (fix grid extent/normalization, never the tol).
**Step 5 — Commit** (`feat(kinematics): jeans_dispersion 3-D + invariants/guard/jit (Phase 0 Task 2)`).

---

## Task 3 — Plummer anchor: analytic Merritt oracle + isotropic f-moment + convergence

**Files:** Modify `dispersion.py` (add `ftable_sigma_r_isotropic`); create
`tests/validation/test_dispersion_physics.py`.

**Step 1 — Failing tests:**
- `test_plummer_om_vs_merritt_analytic`: Jeans σ_r vs the **closed-form Merritt (1985) OM-Plummer**
  dispersion at sample radii, **rtol 1e-3** (the tight oracle — derive/encode the analytic σ_r²(r) for
  the OM Plummer DF; cite Merritt 1985 in a comment).
- `test_plummer_isotropic_jeans_equals_ftable`: `jeans_sigma_r` vs `ftable_sigma_r_isotropic` over the
  DF speed pdf `s²f(Ψ−s²/2)` (isotropic Plummer), rtol to table resolution.
- `test_jeans_quadrature_convergence`: σ_r at fixed r for n_s ∈ {500,1000,2000,4000}; assert the error
  vs the analytic oracle **falls ~4× per doubling** (trapezoid O(h²)) — the method is *verified*, not
  asserted at one resolution. (numerical-method-validation skill.)
- `test_plummer_om_jeans_matches_sampler`: empirical-binned σ_r/σ_t (sample N≈2e5 at fixed radii) within
  5% MC — the looser sampler leg of the anchor.

**Step 2 — Run, expect FAIL.** **Step 3 — implement `ftable_sigma_r_isotropic`** + encode the Merritt
analytic oracle in the test. **Step 4 — Run, expect PASS** (mark heavy tests `@pytest.mark.slow`).
**Step 5 — Commit** (`test(validation): Plummer dispersion anchor + Merritt oracle + convergence (Phase 0 Task 3)`).

---

## Task 4 — EFF + Michie 3-D anchors (Jeans vs empirical; iso f-moment; King limit)

**Files:** Modify `tests/validation/test_dispersion_physics.py`.

`jeans_dispersion(EFFProfile/MichieProfile, …)` already works from Task 2 (both expose `density`).
**Step 1 — Failing tests:** EFF + Michie isotropic σ_r vs empirical binned (5% MC); OM β tracks the
identity; isotropic Jeans-vs-f-moment to table resolution **with a resolution-refinement check that the
gap shrinks** (numerical, not a bug); Michie large-`r_a` → β≈0 (King limit). **Steps 2–5** as Task 3
(`test(validation): EFF + Michie dispersion anchors (Phase 0 Task 4)`).

---

## Task 5 — `project_dispersion` (B&M82) + isotropic limit + analytic oracle

**Files:** Modify `dispersion.py`; test `tests/validation/test_dispersion_physics.py`.

**Step 1 — Failing tests:**
```python
from progenax import project_dispersion
def test_projection_isotropic_all_equal():
    prof = PlummerProfile(r_h=1.0); R = jnp.array([0.5,1.0,2.0])
    pj = project_dispersion(prof, None, R, 400.0, 0.00449)        # beta=0
    assert jnp.allclose(pj.sigma_los, pj.sigma_pm_r, rtol=1e-3)
    assert jnp.allclose(pj.sigma_los, pj.sigma_pm_t, rtol=1e-3)
def test_projection_isotropic_plummer_los_oracle():
    # isotropic Plummer has an analytic projected sigma_los(R); assert rtol 3e-3
    ...
def test_projection_anisotropy_signature():
    prof = PlummerProfile(r_h=1.0); R = jnp.array([2.0,4.0])
    pj = project_dispersion(prof, 1.0, R, 400.0, 0.00449)         # radial bias
    assert jnp.all(pj.sigma_pm_t < pj.sigma_los)                  # tangential PM suppressed outward
```
**Step 2 — Run, expect FAIL.** **Step 3 — Implement** `project_dispersion`: a shared LOS integrator
`_los(g, R, r_max)` via `r²=R²+u²` (`u = linspace(0, √(r_max²−R²)); r=√(R²+u²); trapezoid(g(r), u)`),
vmapped over R; build `σ_r²(r)`, `β(r)`, `ρ(r)` from `jeans_dispersion`/`profile.density`; apply the
three B&M82 kernels; return `ProjectedDispersion(R, σ_los, σ_pm_r, σ_pm_t, Σ)`. **Step 4 — Run, PASS.**
**Step 5 — Commit** (`feat(kinematics): project_dispersion (B&M82) + isotropic oracle (Phase 0 Task 5)`).

---

## Task 6 — Projected empirical anchor + β-recovery

**Files:** Modify `tests/validation/test_dispersion_physics.py`.

**Step 1 — Failing tests:** sample the OM DF, **project to sky** (pick a LOS axis; on-sky radius
`R=√(y²+z²)`; σ_los from `v_x`, σ_pm,R/σ_pm,T from the in-plane velocity decomposition), bin in R, and
assert the analytic `project_dispersion` matches the empirical projected dispersions within MC tol;
plus a β-recovery check (the σ_pm,T/σ_los ratio reflects the input r_a). **Steps 2–5**
(`test(validation): projected-dispersion empirical anchor + beta recovery (Phase 0 Task 6)`).

---

## Task 7 — Differentiability (AD-vs-FD), both forward models

**Files:** Modify `tests/unit/kinematics/test_dispersion.py`.

**Step 1 — Failing grad tests** (reverse-mode; FD-consistent <1e-3; |AD|>1e-9): `∂σ_r/∂(r_a, M)` for
`jeans_dispersion`, and `∂σ_los/∂(r_a, M)`, `∂σ_pm_t/∂r_a` for `project_dispersion`; plus through the
profile params (`r_h`, `gamma`, `W0`). **Step 2 — Run, expect FAIL where non-diff** (interp clamp / a
`where` dead branch — FIX numerics, never the tol). **Steps 3–5**
(`test(kinematics): AD-vs-FD grads, jeans + projection (Phase 0 Task 7)`).

---

## Task 8 — Registries (TWO new `__all__` symbols → all four)

**Files:** as in the prior plan, but register **both** `jeans_dispersion` and `project_dispersion`:
- `api_coverage/manifest.py::SYMBOL_TESTS` — one asserting test each.
- `physics_registry/manifest.py::EXEMPT_NON_MODEL` — both ("forward-model helper, not an equilibrium
  model; physics anchored in test_dispersion_physics.py"); + one anchor invariant line per DF dict.
- `grad_audit/manifest.py` — `SYMBOL_CATEGORY`: both `AUDITED`; `MUST_AUDIT`: the (id,param) pairs for
  `jeans_dispersion[Plummer/EFF/Michie+OM]` and `project_dispersion[Plummer+OM]` in `r_a`/`M`; add the
  matching `Case(...)` in `registry.py` (`direction="params->summary"`, `reduce=identity_sum`,
  `expect="consistent"`, `tol=1e-3`).
- `provenance_registry/manifest.py::PROVENANCE` — the Jeans/Merritt row **and** a B&M82 projection row
  ("Binney & Mamon (1982) MNRAS 200, 361 — LOS projection integrals; derivable identity"). Add any
  `dispersion.py` grid/guard literals to `ALLOWLIST_NON_COEFFICIENT`.

**Step — Run the four coverage suites + grad audit** (command as prior plan). Expected PASS; if red,
the manifest is incomplete — fix, don't skip. **Step — Commit** (`test(registries): register jeans + projection (Phase 0 Task 8)`).

---

## Task 9 — Docstrings, FULL gate, completion doc, HITL pause

**Files:** `dispersion.py` docstrings (units; reverse-mode only; equilibrium dispersion of `profile`
under OM `r_a`; B&M82 cite + the LOS substitution); create `.claude-work/PHASE0_DISPERSION_COMPLETE.md`.

**Step — FULL released-core gate:**
```
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
```
Expected: ALL PASS (capture count). **Step — Commit** (`docs(phase0): dispersion + projection complete + run-record`),
update `STATUS.md` (`next:` Phase 1 OED), `brain "..."` capture, then **PAUSE at the Anna HITL
checkpoint** — present the gate output, await explicit merge-go before any Phase-1 code.

---

## Definition of Complete (Phase 0)

- [ ] `jeans_dispersion` + `project_dispersion` in `progenax.__all__`; NamedTuples (not top-level).
- [ ] Plummer: analytic Merritt oracle (rtol 1e-3) + isotropic Jeans=f-moment + quadrature O(h²) convergence + 5% sampler anchor.
- [ ] EFF + Michie 3-way anchors; Michie isotropic→King limit; Jeans-vs-f-moment gap shrinks with resolution.
- [ ] Projection: isotropic limit (3 projections equal) + analytic isotropic-Plummer σ_los oracle + empirical projected anchor + β-recovery.
- [ ] Invariants (σ²∝GM), r_a domain guard, jit smoke test pass.
- [ ] AD-vs-FD grads consistent for both forward models in (r_a, M, profile params); no silent zeros.
- [ ] Four registries updated (both new symbols); coverage suites green.
- [ ] FULL released-core gate green (count captured); completion doc + STATUS + brain; **Anna merge-go** before Phase 1.
