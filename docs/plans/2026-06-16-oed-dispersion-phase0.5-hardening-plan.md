# Phase 0.5 — Dispersion-engine SoTA hardening — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development to implement this plan
> task-by-task (one subagent per task, independent code review between tasks, Anna HITL at the marked
> checkpoints). Use research-workflow:derivation-before-implementation for D1/D2 math, and
> research-workflow:gradient-validation for every AD-vs-FD gate. Verify LOCALLY; nothing pushed/merged
> without Anna's explicit go.

**Goal:** Harden the merged Phase-0 dispersion engine to SoTA — fix caveats #2 (Plummer tail), #3
(per-R Jeans re-solve), and #1 (OM-only → general-β Jeans + exact Michie DF-moment) — and regression-gate
#4 (the upstream Michie/King `∂σ/∂W0` defect is deferred to the gradient-audit arc; see
`2026-06-16-michie-king-equilibrium-gradient-redesign-deferred.md`).

**Architecture:** All changes in `src/progenax/kinematics/dispersion.py`. One new public symbol
`df_moment_dispersion`; `jeans_dispersion` gains an optional `beta_fn=None` (OM analytic default,
bit-preserved). Refactor `project_dispersion` to tabulate one master Jeans solve and interpolate per-R.

**Tech stack:** JAX (`jnp`, `jax.grad`/`jacrev`, `jit`, `vmap`), `progenax.numerics.cumulative_trapezoid`,
pytest 3-tier. **Zero new deps.** Reverse-mode differentiable.

**Design:** `2026-06-15-oed-dispersion-phase0.5-hardening-design.md` (Q1–Q4 ratified).

**Hard constraints:** TDD (RED→GREEN, never loosen a gate). Paper-grounding before encoding the Michie
DF (D0). Doc reconciliation in the SAME task whenever an equation is verified (per-paper note + website
pages). Commit per task with the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer. Stage files
explicitly. Run tests with the project command:
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest <args>`
(abbreviated `PYTEST` below). Branch: `feat/dispersion-hardening` off `main` (shared tree).

**Task order:** A → C → B → D0(⛔HITL) → D1 → D2 → D3 → D4 → Gate(⛔HITL). A and C both touch the
master-grid helper, so A creates it and C extends its Plummer branch. B is integrator-independent.

---

## Physics reference (verified)

**General-β anisotropic Jeans** (integrating factor `f(r)=exp(2∫β(s)/s ds)`; B&T 2008 §4.8):
```
ρσ_r²(r) = (1/f(r)) ∫_r^∞ f(s) ρ(s) G M(<s)/s² ds
σ_t² = (1−β) σ_r² ;  σ_1d² = (σ_r²+2σ_t²)/3
```
OM is the special case f(r)=r²+r_a², β=r²/(r²+r_a²) — the analytic default (UNCHANGED).

**Michie DF-moment** (verified vs `michie_df.py`): dimensionless `u=v/σ`, `W(r)=interp(r/r_c, xi_grid,
psi_grid, left=W0, right=0)` clamped ≥0, `s=r/r_a`, `σ²=G M/(9 r_c μ)`:
```
f̃(u_r,u_t) = exp(−s² u_t²/2) [exp(W − ½(u_r²+u_t²)) − 1]   on u_r²+u_t² ≤ 2W
d³u = 2π u_t du_r du_t ;  polar u_r=w cosα, u_t=w sinα, α∈[0,π], w∈[0,√(2W)]  ⇒ measure ∝ w² sinα
σ_r² = σ² <u_r²> ;  σ_t² = ½ σ² <u_t²> ;  β = 1 − σ_t²/σ_r²
```
(measure derivation: d³u = 2π u_t·(w dw dα) = 2π w sinα · w dw dα; the 2π and any sign factor cancel
in the moment ratios). Validation = the sampler (truth), Tier-A Jeans consistency, zeroth-moment.

---

## Task A — tabulate-once-then-project (#3 perf)

**Files:** Modify `src/progenax/kinematics/dispersion.py`; Test `tests/unit/kinematics/test_dispersion.py`.

**Step 1 — Failing test** (equivalence + single-solve structure):
```python
def test_project_equivalence_after_tabulate():
    """project_dispersion is numerically unchanged by the tabulate-once refactor."""
    prof = PlummerProfile(r_h=1.0); R = jnp.array([0.5, 1.0, 2.0, 4.0])
    pj = project_dispersion(prof, 2.0, R, 400.0, G_STELLAR)
    # Baseline captured from main@54f5437 (pre-refactor), rtol 1e-9:
    assert jnp.allclose(pj.sigma_los,  jnp.array([_BL_LOS]),  rtol=1e-9)   # fill _BL_* in Step 2
    assert jnp.allclose(pj.sigma_pm_t, jnp.array([_BL_PMT]), rtol=1e-9)
```
**Step 2 — Capture baseline:** before editing, run current `project_dispersion` at those R and paste the
exact arrays into the test (`_BL_LOS`, `_BL_PMT`). This pins the pre-refactor values.
**Step 3 — Run, expect PASS (pre-refactor) then refactor.** Implement two private helpers and rewire:
```python
def _jeans_tables(profile, r_a, M, G, n_s):
    """R-independent master tables (s, rho, I_outward) for the anisotropic-Jeans sigma_r."""
    r_max = getattr(profile, "r_t", None)
    if r_max is None:
        r_max = 30.0 * profile.a
    s = jnp.linspace(1e-4 * r_max, r_max, n_s)
    rho = profile.density(s)
    ds = s[1] - s[0]
    cum = cumulative_trapezoid(rho * s**2, dx=ds)
    M_enc = M * cum / jnp.maximum(cum[-1], 1e-30)
    s2 = s**2
    weight = jnp.ones_like(s) if r_a is None else s2 + jnp.asarray(r_a) ** 2
    integrand = weight * rho * G * M_enc / jnp.maximum(s2, 1e-30)
    I_outward = jnp.flip(cumulative_trapezoid(jnp.flip(integrand), dx=ds))
    return s, rho, I_outward

def _sigma_r2_from_tables(r, s, rho, I_outward, r_a):
    rho_r = jnp.interp(r, s, rho)
    I_r = jnp.interp(r, s, I_outward)
    prefactor = jnp.ones_like(r) if r_a is None else 1.0 / jnp.maximum(r**2 + jnp.asarray(r_a) ** 2, 1e-30)
    return prefactor * I_r / jnp.maximum(rho_r, 1e-30)
```
Rewrite `jeans_dispersion` to call `_jeans_tables` once then `_sigma_r2_from_tables(r, ...)` (bit-identical
to the old `jeans_sigma_r` path — same operations). Keep public `jeans_sigma_r` as a thin wrapper that
builds I_outward from its `(rho, M_enc, s)` args (its callers in `test_dispersion_physics.py` are
unchanged). In `project_dispersion`, call `_jeans_tables(profile, r_a, M, G, n_s_default)` **once before**
the vmap; inside `_los_quantities(R_i)` replace `jeans_dispersion(...)` with
`_sigma_r2_from_tables(r, s, rho, I_outward, r_a)` and `beta` from `_sigma_components`/closed form;
`rho` at r via `jnp.interp(r, s, rho)`.
**Step 4 — Run** `PYTEST tests/unit/kinematics/test_dispersion.py tests/validation/test_dispersion_physics.py -q`.
Expected: ALL PASS (existing physics anchors pin correctness; new equivalence test green).
**Step 5 — Commit** `perf(kinematics): tabulate-once project_dispersion (Phase 0.5 Task A)`.

---

## Task C — algebraic compactification for Plummer tail (#2)

**Files:** Modify `dispersion.py` (`_jeans_tables` Plummer branch); Test `test_dispersion_physics.py`.

**Step 1 — Failing test** (tighten the Plummer residual; O(h²) preserved):
```python
@pytest.mark.slow
def test_plummer_isotropic_tail_machine_precision():
    """Compactified Plummer Jeans matches exact GM/(6 sqrt(r^2+a^2)) to << the old 8.6e-4 tail bias."""
    prof = PlummerProfile(r_h=1.0); r = jnp.array([2.0, 5.0, 10.0, 20.0])  # outer radii where tail bit
    dp = jeans_dispersion(prof, None, r, M=400.0, G=G)
    truth = jnp.sqrt(G*400.0/(6.0*jnp.sqrt(r**2+prof.a**2)))
    assert jnp.max(jnp.abs(dp.sigma_1d/truth - 1.0)) < 5e-5   # was ~8.6e-4 truncated at 30a
```
**Step 2 — Run, expect FAIL** (current 30a truncation gives ~8.6e-4 at outer r).
**Step 3 — Implement** the compactified grid in `_jeans_tables` for the no-`r_t` (Plummer) branch only:
```python
    r_max = getattr(profile, "r_t", None)
    if r_max is None:                       # Plummer: map [s_min, inf) -> t in [0,1) via s = a t/(1-t)
        a = profile.a
        t = jnp.linspace(_T_MIN, _T_MAX, n_s)            # _T_MIN=1e-4, _T_MAX=1-1e-6 (module consts)
        s = a * t / (1.0 - t)
        jac = a / (1.0 - t) ** 2                          # ds/dt
        rho = profile.density(s)
        # M_enc and I_outward integrate in uniform t with the Jacobian folded in:
        dt = t[1] - t[0]
        cum = cumulative_trapezoid(rho * s**2 * jac, dx=dt)
        M_enc = M * cum / jnp.maximum(cum[-1], 1e-30)
        weight = jnp.ones_like(s) if r_a is None else s**2 + jnp.asarray(r_a) ** 2
        integrand = weight * rho * G * M_enc / jnp.maximum(s**2, 1e-30) * jac
        I_outward = jnp.flip(cumulative_trapezoid(jnp.flip(integrand), dx=dt))
        return s, rho, I_outward
    # ... finite-r_t branch unchanged (uniform s grid) ...
```
(`jnp.interp(r, s, ...)` still works: `s` is monotone increasing though non-uniform.)
**Step 4 — Run** the new test + `test_jeans_quadrature_convergence` (O(h²) must still hold) + the full
dispersion suites. Expected: PASS; convergence ratios stay ~4×/doubling. If convergence regresses, fix the
Jacobian/grid (NOT the tolerance).
**Step 5 — Commit** `feat(kinematics): algebraic compactification for Plummer Jeans tail (Phase 0.5 Task C)`.

---

## Task B — regression-gate profile-param gradients (#4 re-scoped)

**Files:** Modify `tests/unit/kinematics/test_dispersion.py`. (No `src` change — see deferred note.)

**Step 1 — Add gates** using the existing `_assert_ad_fd` helper:
```python
def test_grad_jeans_eff_wrt_r_t():
    def f(r_t): return jnp.sum(jeans_dispersion(EFFProfile(a=1.0, gamma=4.0, r_t=r_t), None,
                                                jnp.array([1.0]), 400.0, G_STELLAR).sigma_r)
    _assert_ad_fd(f, 8.0, name="jeans EFF sigma_r / r_t")          # verified clean rel ~1e-8

def test_grad_jeans_eff_wrt_gamma():
    def f(g): return jnp.sum(jeans_dispersion(EFFProfile(a=1.0, gamma=g, r_t=8.0), None,
                                              jnp.array([1.0]), 400.0, G_STELLAR).sigma_r)
    _assert_ad_fd(f, 4.0, name="jeans EFF sigma_r / gamma")        # verified clean

@pytest.mark.xfail(reason="upstream Michie/King ODE-solver gradient (~5e-3 FD-inconsistent); "
                          "deferred to gradient-audit arc, see "
                          "docs/plans/2026-06-16-michie-king-equilibrium-gradient-redesign-deferred.md",
                   strict=True)
def test_grad_jeans_michie_wrt_W0_DEFERRED():
    from progenax.profiles import MichieProfile
    def f(W0): return jnp.sum(jeans_dispersion(MichieProfile.from_W0_rc(W0=W0, r_c=1.0, r_a=5.0),
                                               None, jnp.array([1.0]), 400.0, G_STELLAR).sigma_r)
    _assert_ad_fd(f, 6.0, name="jeans Michie sigma_r / W0")        # rel ~5e-3 > 1e-3 -> xfail(strict)
```
(Note: `import pytest`, `EFFProfile` already imported in the test module.)
**Step 2 — Run** `PYTEST tests/unit/kinematics/test_dispersion.py -q`. Expected: EFF gates PASS; the
Michie gate **xfails strict** (documents the limitation + auto-alerts if the gradient-audit arc later
fixes it, flipping xfail→xpass).
**Step 3 — Commit** `test(kinematics): regression-gate profile-param dispersion grads; xfail deferred Michie-W0 (Phase 0.5 Task B)`.

---

## ⛔ Task D0 — Michie paper grounding + doc reconciliation (HITL CHECKPOINT)

**Files:** Read `docs/core-papers/michie-1963.pdf`, `king1966.pdf`. Verify/correct
`docs/website/99-bibliography/per-paper/michie-1963.md`, `docs/website/10-theory/spatial-profiles/king.md`,
`docs/website/50-validation/michie-anisotropy.md`.

**Step 1 — Read the PDF** (Read tool, `michie-1963.pdf`). Confirm the DF functional form
`f ∝ exp(−J²/2 r_a²σ²)[exp(−E/σ²) − 1]`, the relative-energy/angular-momentum definitions, and the
King (1966) cutoff. Quote the exact equation(s).
**Step 2 — Reconcile docs:** verify each website/per-paper equation matches the PDF; fix any stale or
unverified form. Record what was confirmed vs corrected.
**Step 3 — ⛔ PAUSE for Anna:** present the verified DF form + the 3-leg oracle design (sampler / Tier-A
consistency / zeroth-moment) and get explicit go before D1–D4.
**Step 4 — Commit** `docs(provenance): verify Michie 1963 DF vs PDF + reconcile website (Phase 0.5 D0)`.

---

## Task D1 — general-β `jeans_dispersion` (Tier A)

**Files:** Modify `dispersion.py`; Test `test_dispersion.py`.

**Step 1 — Failing tests:**
```python
def test_beta_fn_om_equals_default():
    """beta_fn reproducing OM matches the analytic OM default (bit-preserved default path)."""
    prof = PlummerProfile(r_h=1.0); r = jnp.array([0.5, 1.0, 2.0]); r_a = 2.0
    default = jeans_dispersion(prof, r_a, r, 400.0, G_STELLAR)
    beta_om = lambda rr: rr**2 / (rr**2 + r_a**2)
    general = jeans_dispersion(prof, None, r, 400.0, G_STELLAR, beta_fn=beta_om)
    assert jnp.allclose(default.sigma_r, general.sigma_r, rtol=1e-4)
    assert jnp.allclose(default.beta,    beta_om(r),      rtol=1e-6)

def test_grad_jeans_beta_fn_wrt_M():
    beta_om = lambda rr: rr**2 / (rr**2 + 4.0)
    def f(M): return jnp.sum(jeans_dispersion(PlummerProfile(r_h=1.0), None, jnp.array([1.0]),
                                              M, G_STELLAR, beta_fn=beta_om).sigma_r)
    _assert_ad_fd(f, 400.0, name="general-beta jeans / M")
```
**Step 2 — Run, expect FAIL** (`beta_fn` kwarg absent).
**Step 3 — Implement** `beta_fn=None` in `jeans_dispersion` and a numerical-integrating-factor branch in
`_jeans_tables` (analytic OM untouched when `beta_fn is None`):
```python
# in jeans_dispersion signature: def jeans_dispersion(profile, r_a, r, M, G, n_s=4000, beta_fn=None)
# when beta_fn is not None: build f(s)=exp(2∫β/s ds) numerically, in a max-subtracted log form:
    beta_s = beta_fn(s)
    F = 2.0 * cumulative_trapezoid(beta_s / jnp.maximum(s, 1e-30), dx=ds)   # ln f(s) up to const
    Fmax = jnp.max(F)
    integrand = jnp.exp(F - Fmax) * rho * G * M_enc / jnp.maximum(s**2, 1e-30)
    I_outward = jnp.flip(cumulative_trapezoid(jnp.flip(integrand), dx=ds))
    # sigma_r²(r) = exp(-(F(r)-Fmax)) * I(r)/rho(r):  carry F in the tables for the beta_fn path
# beta(r), sigma_t, sigma_1d from beta_fn(r): sigma_t²=(1-β)σ_r²; sigma_1d²=(σ_r²+2σ_t²)/3
```
Generalize `_sigma_components` to accept an explicit `beta` array (used when `beta_fn` given); the OM/
isotropic path keeps the closed forms. Keep the `r_a < 0.75a` guard gated on `beta_fn is None`
(OM-only validity domain).
**Step 4 — Run** `PYTEST tests/unit/kinematics/test_dispersion.py -q` + the full dispersion suites.
Expected: PASS (OM default bit-preserved; new beta_fn tests green).
**Step 5 — Commit** `feat(kinematics): general-beta jeans_dispersion (Tier A, Phase 0.5 D1)`.

---

## Task D2 — `df_moment_dispersion` (Tier B exact Michie moment)

**Files:** Modify `dispersion.py` (+ export wiring); Test `test_dispersion.py`, `test_dispersion_physics.py`.

**Step 1 — Failing tests:**
```python
def test_df_moment_export_and_shapes():
    from progenax import df_moment_dispersion
    from progenax.profiles import MichieProfile
    from progenax.kinematics import MichieVelocityDF
    assert "df_moment_dispersion" in set(progenax.__all__)
    df = MichieVelocityDF(W0=6.0, r_c=1.0, r_a=5.0)
    dp = df_moment_dispersion(df, jnp.array([0.5, 1.0, 2.0]), 400.0, G_STELLAR)
    assert dp.sigma_r.shape == (3,) and jnp.all(dp.sigma_r > 0)
```
**Step 2 — Run, expect FAIL.**
**Step 3 — Implement** `df_moment_dispersion(df, r, M, G, n_w=256, n_alpha=128)` per the verified physics
reference (polar fixed-domain `[0,√(2W)]×[0,π]`, measure `∝ w² sinα`, `σ²=GM/(9 r_c μ)`,
`W=interp(r/r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0)` clamped ≥0, `s=r/df.r_a`),
returning a `DispersionProfile`. vmap the per-r 2-D moment. Wire exports: `kinematics/__init__.py` and
`progenax/__init__.py` `__all__ += ["df_moment_dispersion"]`.
**Step 4 — Run.** Expected: PASS (shapes + positivity).
**Step 5 — Commit** `feat(kinematics): df_moment_dispersion exact Michie moment (Tier B, Phase 0.5 D2)`.

---

## Task D3 — Michie all-radii anchor (3-leg oracle, kills "inner-region-only")

**Files:** `tests/validation/test_dispersion_physics.py`.

**Step 1 — Failing tests** (`@pytest.mark.slow`; mirror `test_michie_jeans_matches_sampler` style):
```python
@pytest.mark.slow
def test_df_moment_matches_sampler_all_radii():
    """Tier B (DF-moment) vs Michie sampler at ALL radii (incl. outer) within 5% MC."""
    W0, r_c, r_a, M, N = 6.0, 1.0, 5.0, 400.0, 200_000
    df = MichieVelocityDF(W0=W0, r_c=r_c, r_a=r_a)
    for r0 in (0.5, 1.0, 2.0, 4.0, 8.0):                # outer radii where OM-Jeans diverged
        positions = jnp.zeros((N, 3)).at[:, 0].set(r0); masses = jnp.full((N,), M / N)
        v = df.sample_velocities(positions, masses, jax.random.PRNGKey(int(r0*1000)+7), G=G)
        sr_emp = jnp.std(v[:, 0]); st_emp = jnp.sqrt(0.5*(jnp.var(v[:,1])+jnp.var(v[:,2])))
        dp = df_moment_dispersion(df, jnp.array([r0]), M, G)
        assert jnp.allclose(dp.sigma_r[0], sr_emp, rtol=0.05)
        assert jnp.allclose(dp.sigma_t[0], st_emp, rtol=0.05)

@pytest.mark.slow
def test_tierA_jeans_consistent_with_tierB():
    """A true equilibrium satisfies Jeans: general-beta Jeans with the DF's own beta(r) == Tier B sigma_r."""
    df = MichieVelocityDF(W0=6.0, r_c=1.0, r_a=5.0); prof = MichieProfile.from_W0_rc(W0=6.0, r_c=1.0, r_a=5.0)
    s = jnp.linspace(0.1, 0.9*float(prof.r_t), 60)
    b = df_moment_dispersion(df, s, 400.0, G).beta
    beta_fn = lambda rr: jnp.interp(rr, s, b)
    A = jeans_dispersion(prof, None, s, 400.0, G, beta_fn=beta_fn).sigma_r
    B = df_moment_dispersion(df, s, 400.0, G).sigma_r
    assert jnp.max(jnp.abs(A/B - 1.0)) < 0.02            # numerical consistency (not a fitted tol)

def test_df_moment_zeroth_moment_self_consistent():
    """∫ f d³u (Tier B normalization) tracks profile.density(r) in shape."""
    df = MichieVelocityDF(W0=6.0, r_c=1.0, r_a=5.0); prof = MichieProfile.from_W0_rc(W0=6.0, r_c=1.0, r_a=5.0)
    r = jnp.array([0.5, 1.0, 2.0, 4.0])
    # df_moment_dispersion exposes the zeroth moment via a helper or returns it; compare ratios to density
    # (implementation: assert rho_quad(r)/rho_quad(r[0]) ~ density(r)/density(r[0]) within 2%)
```
**Step 2 — Run, expect FAIL** (until D2 correct). **Step 3 — tune `n_w`/`n_alpha`** in `df_moment_dispersion`
until the sampler/consistency legs pass (numerical convergence, NOT tolerance-loosening). Add a small
zeroth-moment accessor if needed. **Step 4 — Run** the validation suite. Expected PASS.
**Step 5 — Commit** `test(validation): Michie DF-moment 3-leg anchor, correct at all radii (Phase 0.5 D3)`.

---

## Task D4 — registries (1 new symbol → all four) + dashboard re-stamp

**Files:** the four manifests + `registry.py` + regenerate dashboard + re-stamp coverage.

Register `df_moment_dispersion` consistently with the existing dispersion entries (verbatim patterns in
the recon). Concretely:
- **api_coverage/manifest.py::SYMBOL_TESTS** — `"df_moment_dispersion":
  "tests/unit/kinematics/test_dispersion.py::test_df_moment_export_and_shapes"`.
- **physics_registry/manifest.py::EXEMPT_NON_MODEL** — `"df_moment_dispersion"`: "exact Michie DF
  second-moment (sigma_r/sigma_t/beta) forward-model helper, not an equilibrium model; physics anchored
  in test_dispersion_physics.py (sampler + Tier-A consistency)."
- **grad_audit/manifest.py** — `SYMBOL_CATEGORY["df_moment_dispersion"] = AUDITED`; `MUST_AUDIT[
  ("df_moment_dispersion[Michie]", "M")] = "Michie DF-moment sigma_r in total mass"`.
- **grad_audit/registry.py** — add a closure `_df_moment_M(M)` (Michie, interior r, `.sigma_r`) + a
  `Case(id="df_moment_dispersion[Michie]", direction="params->summary", fn=_df_moment_M, param="M",
  theta0=400.0, reduce=identity_sum, expect="consistent", tol=1e-3)`. (M-gradient only — the W0 path is
  the deferred limitation; do NOT add a W0 Case here.)
- **provenance_registry/manifest.py::PROVENANCE** — a Michie-1963 row for the DF-moment kernel citing the
  D0-verified equation.
- Regenerate: `scripts/build_test_dashboard.py --emit --render`; bump any `__all__`-count ratchet in
  `tests/validation/test_dashboard_gen.py` (121 → 122). Re-measure + re-stamp coverage per the Phase-0
  dogfood playbook (separate stamp commit so `git_sha` is an ancestor with no later `src/` diff).

**Run** `PYTEST tests/validation/api_coverage tests/validation/physics_registry tests/validation/grad_audit
tests/validation/provenance_registry tests/validation/test_dashboard_fresh.py -q`. Expected PASS.
**Commit** `test(registries): register df_moment_dispersion across four + dashboard (Phase 0.5 D4)`.

---

## ⛔ Gate — FULL released-core + completion (HITL CHECKPOINT)

**Step 1 — FULL gate** (deterministic under --cov; flaky timing skips):
```
PYTEST tests/unit tests/integration tests/validation -q -n auto
```
Expected: ALL PASS (capture count; baseline was 1506 passed, 2 skipped + the new tests).
**Step 2 — Completion doc** `.claude-work/PHASE0.5_HARDENING_COMPLETE.md` (results table: A equivalence +
speedup, C residual, B gate status, D 3-leg numbers, D4 registries). Update `STATUS.md`, `brain` capture,
memory ([[oed-dispersion-arc]]).
**Step 3 — ⛔ PAUSE for Anna** — present the gate output; await explicit merge-go before any Phase-1 OED code.

---

## Definition of Complete (Phase 0.5)
- [ ] A: `project_dispersion` ~1e-9 equivalence + one-master-solve structure; projection physics green.
- [ ] C: Plummer outer-radius residual < 5e-5 (was 8.6e-4); O(h²) convergence preserved; King/EFF untouched.
- [ ] B: EFF `r_t`/`γ` regression gates clean; Michie-`W0` xfail(strict) cross-ref'ing the deferred note.
- [ ] D0: Michie 1963 DF verified vs PDF; per-paper note + website pages reconciled; ⛔ Anna go.
- [ ] D1: general-β `jeans_dispersion` (OM default bit-preserved); beta_fn AD-vs-FD clean.
- [ ] D2: `df_moment_dispersion` in `progenax.__all__`; D3: sampler (all radii) + Tier-A consistency +
      zeroth-moment all green; Michie correct at ALL radii.
- [ ] D4: four registries + dashboard re-stamp + coverage re-measure; all green.
- [ ] FULL gate green (count captured); completion doc + STATUS + brain + memory; ⛔ Anna merge-go.
