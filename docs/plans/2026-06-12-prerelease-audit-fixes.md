# Pre-Release Audit Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Execute in batches (one phase = one batch); STOP at every ⛔ CHECKPOINT and report to Anna — she is the non-negotiable human-in-the-loop and must approve before you continue. Never weaken a test to make it pass; if a fix changes expected physics, derive the new expectation analytically inside the test.

**Goal:** Fix and validate every release blocker from `docs/reviews/2026-06-11-prerelease-adversarial-audit.md` (findings R1–R10 + high-value mediums), moving progenax from Beta to Release Candidate.

**Architecture:** Five phases ordered by dependency: (0) make CI green and enforcing, (1) TDD fixes for the five confirmed science bugs, (2) JAX/API hardening, (3) documentation honesty sweep, (4) packaging/metadata. Each science fix follows strict RED→GREEN: write the failing test that encodes the *paper's* convention or a hi-res numerical reference, watch it fail against current code, then fix.

**Tech Stack:** JAX/Equinox (100% JAX-native core — never numpy/scipy in `src/progenax` except the documented `diagnostics` carve-out), pytest + pytest-xdist, uv, diffrax, GitHub Actions.

**Source of truth:** `docs/reviews/2026-06-11-prerelease-adversarial-audit.md`. Finding IDs (R1–R10, S*, J*, A*, T*, D*) refer to that report. Read its §3 and §4 before starting.

---

## Ground rules for the executing session

- **Branch:** create `fix/prerelease-audit` off `main`. Commit per task. Do NOT push or merge without Anna's explicit go (HITL).
- **Test commands** (from `progenax/CLAUDE.md`):
  ```bash
  # FAST gate (inner loop, ~4 min). Baseline before any change: 1145 passed.
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow" -n auto
  # FULL gate (phase close-out, ~9 min). Baseline: 1192 passed.
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
  # Single test during TDD:
  env -u VIRTUAL_ENV uv run --no-sync pytest <path>::<test> -v
  ```
- **Verify RED before GREEN.** Every new test must be run and shown failing against unmodified code before the fix lands. If a test passes immediately, the test is wrong — stop and rethink.
- **Units:** STELLAR (M☉, pc, Myr); G explicit everywhere; never hardcode G.
- **JAX policy:** `jnp` only in core, `lax.scan`/`fori_loop` (no `while_loop`), the double-`where` `exp_safe` pattern for removable singularities (reference implementation: `src/progenax/imf/differentiable.py:47-54`).
- After each phase: run the FULL gate, report results + diffs to Anna at the ⛔ CHECKPOINT.

---

# Phase 0 — CI: make the suite enforced again (R1, R8, T4, M5)

CI has been red on `main` since 2026-06-10: every job dies in ~25 s at `uv sync`/`uv lock --check` with `Distribution not found at: .../jaxstroviz`, because `pyproject.toml` declares `[tool.uv.sources] jaxstroviz = { path = "../jaxstroviz" }` (and `uv.lock` pins it, dragging `fluxax` transitively) but `.github/workflows/tests.yml` only checks out `progenax` + `jaxstro`. Nothing in this repo is currently enforced anywhere.

### Task 0.1: Check out the missing sibling packages in all CI jobs

**Files:**
- Modify: `.github/workflows/tests.yml` (jobs `lock-check`, `released-core`, `experimental` — every job that runs `uv lock --check` or `uv sync`)

**Step 1: Identify every sibling path dependency the lock graph needs**

```bash
grep -n "editable" uv.lock | sort -u
grep -n "path =" pyproject.toml
```
Expected: `../jaxstro`, `../jaxstroviz`, and (transitively via jaxstroviz) possibly `../fluxax`. Note exactly which appear — the checkout list below must match.

**Step 2: Add checkout steps**

In `tests.yml`, immediately after the existing `Checkout jaxstro` step in **each** of the three jobs, add (repeat per missing sibling found in Step 1):

```yaml
      - name: Checkout jaxstroviz (path dependency, sibling layout)
        uses: actions/checkout@v4
        with:
          repository: jaxstro/jaxstroviz
          path: jaxstroviz
      - name: Checkout fluxax (transitive path dependency of jaxstroviz)
        uses: actions/checkout@v4
        with:
          repository: jaxstro/fluxax
          path: fluxax
```

If a repo name differs (check with `gh repo list jaxstro`), use the real name. If `fluxax` is NOT in `uv.lock`, omit it.

**Step 3: Verify locally that the lock graph is the only problem**

```bash
env -u VIRTUAL_ENV uv lock --check
```
Expected: passes locally (siblings exist here). The CI failure is purely the missing checkout.

**Step 4: Commit, push the branch, and watch CI**

```bash
git add .github/workflows/tests.yml
git commit -m "fix(ci): check out jaxstroviz (+fluxax) siblings — unbreaks uv sync, CI red since 2026-06-10 (audit R1)"
git push -u origin fix/prerelease-audit   # pushing the FIX BRANCH to run CI is authorized for this task only
gh run watch --exit-status || gh run list --branch fix/prerelease-audit --limit 3
```
Expected: `lock-check` passes; `released-core` and `experimental` shards run actual tests. If a shard fails on a *test* (not on sync), STOP and report — that's new information (tests that haven't run in CI for days may be stale).

### Task 0.2: Make the LIMEPY reference cache a hard failure in strict mode (T4, H2)

`tests/validation/test_limepy_reference_parity.py:62-64` silently `pytest.skip`s when `validation/data/limepy_reference/*.npz` is absent — a packaging mistake would convert the strongest external-reference gate into a silent pass.

**Step 1: Write the failing test** — Create `tests/validation/test_strict_refs.py`:

```python
"""Strict-mode guard: reference caches must exist when PROGENAX_STRICT_REFS=1."""
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIMEPY_CACHE = REPO_ROOT / "validation" / "data" / "limepy_reference"


@pytest.mark.skipif(
    os.environ.get("PROGENAX_STRICT_REFS") != "1",
    reason="strict reference mode not requested",
)
def test_limepy_reference_cache_present():
    """In strict (nightly/release) mode the committed LIMEPY cache must exist.

    Guards against a checkout/packaging mistake silently disabling the
    reference-parity suite (audit T4/H2).
    """
    npz_files = sorted(LIMEPY_CACHE.glob("*.npz"))
    assert npz_files, f"no LIMEPY reference .npz under {LIMEPY_CACHE}"
```

**Step 2: Run it RED then GREEN**

```bash
PROGENAX_STRICT_REFS=1 env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_strict_refs.py -v
```
Expected: PASS (cache is committed). Now verify it actually guards: temporarily `mv validation/data/limepy_reference /tmp/`, rerun (expected FAIL), then `mv` it back and rerun (PASS). Show both outputs.

**Step 3: Convert the silent skip in the parity test to a strict-mode failure**

In `tests/validation/test_limepy_reference_parity.py`, replace the bare `pytest.skip(...)` at ~line 62-64 with:

```python
        if os.environ.get("PROGENAX_STRICT_REFS") == "1":
            pytest.fail(f"reference-LIMEPY cache absent at {cache_path} (strict mode)")
        pytest.skip(f"reference-LIMEPY cache absent at {cache_path}")
```
(add `import os` if missing).

**Step 4: Commit**

```bash
git add tests/validation/test_strict_refs.py tests/validation/test_limepy_reference_parity.py
git commit -m "test: PROGENAX_STRICT_REFS=1 turns missing reference caches into failures (audit T4)"
```

### Task 0.3: Add the nightly/release full-physics CI lane (R8)

The headline trust anchors (LIMEPY parity, multimass equilibrium, Engine A-vs-B, Engine B AD-vs-FD — 47 `@pytest.mark.slow` tests) currently run nowhere but Anna's laptop.

**Files:**
- Create: `.github/workflows/physics-validation.yml`

```yaml
name: physics-validation

on:
  schedule:
    - cron: "17 9 * * *"   # nightly
  workflow_dispatch:
  push:
    tags: ["v*"]

env:
  JAX_ENABLE_X64: "1"
  XLA_PYTHON_CLIENT_PREALLOCATE: "false"
  XLA_PYTHON_CLIENT_ALLOCATOR: platform
  OMP_NUM_THREADS: "1"
  PROGENAX_STRICT_REFS: "1"

jobs:
  slow-physics:
    runs-on: ubuntu-latest
    timeout-minutes: 90
    steps:
      # SAME checkout block as tests.yml (progenax + jaxstro + jaxstroviz [+ fluxax])
      # ... copy verbatim from tests.yml after Task 0.1 ...
      - name: Setup uv
        uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.13"
      - name: Sync from lockfile (dev + experimental)
        working-directory: progenax
        run: uv sync --locked --extra dev --extra experimental
      - name: Slow physics-validation suite (the advertised trust anchors)
        working-directory: progenax
        run: uv run --no-sync pytest tests -m slow -q
      - name: Strict reference guard
        working-directory: progenax
        run: uv run --no-sync pytest tests/validation/test_strict_refs.py -q
```

**Verify:** `gh workflow run physics-validation.yml --ref fix/prerelease-audit` then `gh run watch`. The slow suite takes ~30–60 min on the 2-core runner; if it OOMs, shard it `-m "slow"` per tier like `tests.yml` does. Commit:

```bash
git add .github/workflows/physics-validation.yml
git commit -m "ci: nightly + release-tag lane running the slow-marked headline validations (audit R8)"
```

### Task 0.4: Delete the ghost cache directories (M5)

```bash
ls src/progenax/gravoturb/ src/progenax/cluster/fdf_density/   # confirm: ONLY __pycache__/.mypy_cache debris, zero .py
git ls-files src/progenax/gravoturb src/progenax/cluster/fdf_density   # confirm: empty (nothing tracked)
rm -rf src/progenax/gravoturb src/progenax/cluster/fdf_density
```
Nothing to commit (untracked). Note in the phase report that this was done. **Do NOT touch `src/progenax/cluster/turbulence.py`** — it is live core physics consumed by the environment-IMF (audit M5 note).

### ⛔ CHECKPOINT 0
Report to Anna: CI run URLs (green or what failed), the strict-refs RED/GREEN evidence, and FAST gate output (`1145 passed` + the 1 new strict test when env var unset → skipped). **Wait for approval before Phase 1.**

---

# Phase 1 — Confirmed science bugs, strict TDD (R3, R4, R5, R6, R10)

All five were verified by execution during the audit (see report §3 + the orchestrator re-verification). Each task: failing test first.

### Task 1.1: Moe & Di Stefano F_twin normalization (R3) — **the most important fix in this plan**

**The bug:** `MoeDiStefano2017Full.pdf` (`src/progenax/imf/binary/moe_di_stefano.py:279-289`) mixes the twin block with weight `ft` against the whole q ∈ [0.1, 1] population. The paper (MD17 p.5 + Fig. 2 caption; PDF held at `docs/core-papers/Moe_2017_ApJS_230_15.pdf`) defines F_twin as the excess-twin fraction **relative to q > 0.3 companions**. Measured impact: realized paper-convention F_twin = 0.367 vs Table-13 value 0.300 at (M1=1, logP=1) — +22% twin overweight, with a matching deficit at q < 0.3.

**The math of the fix:** build the unnormalized mixture as the two-slope power law plus a twin block of mass `ft/(1−ft) · I_B` on [0.95, 1] (where `I_B` is the power-law mass on [q_break, 1]), then normalize the whole thing over [q_min, 1]. Check: excess-twin fraction among q > 0.3 = `(ft/(1−ft)·I_B) / (I_B + ft/(1−ft)·I_B)` = `ft` exactly. ✓

**Files:**
- Modify: `src/progenax/imf/binary/moe_di_stefano.py:279-289` (`pdf`; add a `_components` helper)
- Modify: `tests/unit/imf/test_moe_full.py` (the `mean(q>0.95) > 0.28` assertion at ~lines 92-100 encodes the OLD convention — replace with a derived expectation, do not just loosen)
- Test: `tests/unit/imf/test_moe_full.py` (new tests appended)

**Step 1: Write the failing tests**

Append to `tests/unit/imf/test_moe_full.py`:

```python
class TestFTwinPaperConvention:
    """MD17 p.5 / Fig.2: F_twin = excess-twin fraction of q > 0.3 companions.

    Audit finding R3: the pre-fix mixture realized F_twin = 0.367 instead of
    0.300 at (M1=1, logP=1). These tests pin the paper convention at four
    Table-13 nodes via quadrature of the pdf's mixture components.
    """

    NODES = [(1.0, 1.0), (1.0, 3.0), (3.2, 1.0), (12.0, 3.0)]  # (M1 [Msun], logP)

    def test_f_twin_excess_over_q_gt_03(self):
        md = MoeDiStefano2017Full()
        qs = jnp.linspace(md.q_min, 1.0, 200_001)
        for m1, logP in self.NODES:
            mass = jnp.asarray(m1)
            period = jnp.asarray(10.0**logP)
            ft_table = float(md.f_twin(period, mass))
            p_pl, p_twin = md._components(qs, mass, period)
            mask = qs >= 0.3
            twin_mass = float(jnp.trapezoid(jnp.where(mask, p_twin, 0.0), qs))
            total_gt03 = float(
                jnp.trapezoid(jnp.where(mask, p_pl + p_twin, 0.0), qs)
            )
            realized = twin_mass / total_gt03
            assert abs(realized - ft_table) < 2e-3, (
                f"(M1={m1}, logP={logP}): realized paper-convention F_twin "
                f"{realized:.4f} != Table 13 value {ft_table:.4f}"
            )

    def test_pdf_still_normalized(self):
        md = MoeDiStefano2017Full()
        qs = jnp.linspace(md.q_min, 1.0, 200_001)
        for m1, logP in self.NODES:
            p = jax.vmap(
                lambda q: md.pdf(q, jnp.asarray(m1), jnp.asarray(10.0**logP))
            )(qs)
            Z = float(jnp.trapezoid(p, qs))
            assert abs(Z - 1.0) < 1e-3

    def test_sample_matches_pdf_twin_fraction(self):
        """Sampled P(q >= 0.95) must match the quadrature of the FIXED pdf."""
        md = MoeDiStefano2017Full()
        n = 200_000
        key = jax.random.PRNGKey(0)
        m1 = jnp.full((n,), 1.0)
        periods = jnp.full((n,), 10.0)
        q = md.sample(key, m1, periods)
        qs = jnp.linspace(md.q_min, 1.0, 200_001)
        p = jax.vmap(lambda qq: md.pdf(qq, jnp.asarray(1.0), jnp.asarray(10.0)))(qs)
        expected = float(
            jnp.trapezoid(jnp.where(qs >= 0.95, p, 0.0), qs)
            / jnp.trapezoid(p, qs)
        )
        observed = float(jnp.mean(q >= 0.95))
        assert abs(observed - expected) < 0.01  # shot noise ~0.001 at n=2e5
```

**Step 2: Run RED**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/imf/test_moe_full.py::TestFTwinPaperConvention -v
```
Expected: `test_f_twin_excess_over_q_gt_03` FAILS — first with `AttributeError: _components` (helper doesn't exist), and after you add the helper as a pure refactor of the CURRENT (buggy) mixture it must fail with `realized 0.367 != 0.300` at the (1.0, 1.0) node. Show both failures.

**Step 3: Implement**

In `moe_di_stefano.py`, replace `pdf` (lines 279-289) with:

```python
    def _components(self, q, masses, periods):
        """Normalized (power-law, twin-excess) mixture components.

        Paper convention (MD17 p.5, Fig. 2): F_twin is the excess-twin fraction
        of q > 0.3 companions. The unnormalized twin block therefore carries
        mass ft/(1-ft) * I_B (I_B = power-law mass on [q_break, 1]), so that
        twin/(twin + I_B) = ft exactly after joint normalization.
        """
        gs, gl, C, I_A, I_B = self._two_slope(periods, masses)
        ft = self.f_twin(periods, masses)
        p_lo = jnp.power(q, gs)
        p_hi = C * jnp.power(q, gl)
        p_pl_unnorm = jnp.where(q < self.q_break, p_lo, p_hi)
        ft_safe = jnp.minimum(ft, 0.95)  # Table 13 max is ~0.3; guard 1/(1-ft)
        twin_mass = ft_safe / (1.0 - ft_safe) * I_B
        twin_unnorm = twin_mass * jnp.where(
            (q >= 0.95) & (q <= 1.0), 1.0 / 0.05, 0.0
        )
        Z_tot = I_A + I_B + twin_mass
        in_range = (q >= self.q_min) & (q <= 1.0)
        p_pl = jnp.where(in_range, p_pl_unnorm / Z_tot, 0.0)
        p_twin = jnp.where(in_range, twin_unnorm / Z_tot, 0.0)
        return p_pl, p_twin

    def pdf(self, q, masses, periods):
        """Conditional mass-ratio pdf p(q | M1, P), normalized on [q_min, 1].

        F_twin follows the paper convention: excess-twin fraction of the
        q > 0.3 population (NOT of all q > q_min companions — audit R3).
        """
        p_pl, p_twin = self._components(q, masses, periods)
        return p_pl + p_twin
```

Update the class docstring (lines 233-242) to state the q>0.3 convention explicitly.

**Step 4: Run GREEN + fix the stale assertion**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/imf/test_moe_full.py -v
```
The old `mean(q > 0.95) > 0.28` test may now fail (the corrected solar-node twin share is ≈ 0.28, right at the gate). Replace its magic number with the quadrature-derived expectation (the same pattern as `test_sample_matches_pdf_twin_fraction`). Do NOT simply lower the constant.

**Step 5: Check downstream consumers + gradients**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/imf tests/unit/binaries tests/integration/test_binary_cluster.py tests/unit/imf/test_imf_gradients.py -q
```
`MoeJointOrbit` and `MoeCompanions` consume this pdf — their distribution tests may have pinned the old convention; update each with derived expectations, documenting the R3 fix in a comment.

**Step 6: Commit**

```bash
git add src/progenax/imf/binary/moe_di_stefano.py tests/
git commit -m "fix(imf): MoeDiStefano2017Full F_twin now follows the paper's q>0.3 convention (audit R3)

Twin excess was mixed against the full q in [0.1,1] population; MD17 p.5/Fig.2
define F_twin relative to q>0.3. Realized F_twin at (M1=1, logP=1) was 0.367
for a Table-13 value of 0.300. Mixture rebuilt as PL + ft/(1-ft)*I_B twin block,
jointly normalized; pinned at four Table-13 nodes."
```

### Task 1.2: High-W0 King/Michie CDF core resolution (R4)

**The bug:** `KingProfile.__init__` (`src/progenax/profiles/king.py:359`) builds the sampling CDF on `r_grid = jnp.linspace(0.0, r_t_arr, n_grid)` with `n_grid=1000`. Since r_t/r_c grows super-exponentially with W0 (131 at W0=9, 548 at W0=12), the core gets ~8 points at W0=9 and ~2 at W0=12. Measured enclosed-mass error at 0.3 r_c: **+18.3% (W0=9), +270% (W0=12)** (production sampler vs hi-res ODE reference, 2×10⁶ samples, shot ~2%). Same pattern at `michie.py:177` and `eff.py:83`.

**The fix:** sqrt-stretched radial grid `r = r_t · u²` with `u = linspace(0, 1, n_grid)` — spacing dr ∝ √r concentrates points in the core (12+ points inside 0.3 r_c even at W0=12 with n_grid=1000) — plus the non-uniform trapezoid. Differentiability in W0/r_c is preserved (r_grid remains a smooth function of the traced `r_t_arr`).

**Files:**
- Modify: `src/progenax/profiles/king.py:358-392` (grid + cumulative-mass block in `__init__`)
- Modify: `src/progenax/profiles/michie.py:~177` (same pattern — locate the `linspace(0, r_t, ...)` CDF grid and apply identically)
- Modify: `src/progenax/profiles/eff.py:~83` (same, for uniformity; EFF is currently fine at typical r_t/a but shares the pattern)
- Test: `tests/validation/test_king_physics.py` (new class)

**Step 1: Write the failing test** — append to `tests/validation/test_king_physics.py`:

```python
class TestHighW0CoreResolution:
    """Audit R4: the linear 1000-pt CDF grid under-resolves the core at W0 >= 9.

    Reference = direct quadrature of rho_hat(psi) on a dense ODE solve
    (xi_max=600, n_points=20000) — independent of the profile's internal CDF.
    Measured pre-fix errors at 0.3 r_c: +18% (W0=9), +270% (W0=12).
    """

    @pytest.mark.parametrize("W0", [7.0, 9.0, 12.0])
    def test_sampled_core_mass_matches_dense_reference(self, W0):
        from progenax.profiles.king import (
            solve_king_profile,
            king_lowered_maxwellian_density,
        )

        xi, psi = solve_king_profile(W0, xi_max=600.0, n_points=20_000)
        rho = king_lowered_maxwellian_density(jnp.maximum(psi, 0.0))
        integ = rho * xi**2
        M = jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(0.5 * (integ[1:] + integ[:-1]) * jnp.diff(xi))]
        )
        prof = KingProfile.from_W0_rc(W0=W0, r_c=1.0)
        n = 2_000_000
        pos = prof.sample_positions(jnp.ones(n), jax.random.PRNGKey(3))
        r = jnp.linalg.norm(pos, axis=1)
        for r_probe in (0.3, 1.0, 3.0):
            m_ref = float(jnp.interp(r_probe, xi, M) / M[-1])
            m_samp = float(jnp.mean(r < r_probe))
            shot = 3.0 / (m_ref * n) ** 0.5  # 3 sigma binomial
            tol = max(0.03, shot)  # 3% grid budget or shot noise, whichever larger
            assert abs(m_samp / m_ref - 1.0) < tol, (
                f"W0={W0}, r={r_probe} r_c: sampled M(<r)/M = {m_samp:.3e} vs "
                f"reference {m_ref:.3e} (rel err {(m_samp/m_ref-1)*100:+.1f}%)"
            )
```

Mark the class `@pytest.mark.slow` if a 2M-sample draw ×3 takes > 60 s; if so, ALSO add one fast non-slow case (W0=9, r_probe=0.3 only, n=5×10⁵, tol=max(0.05, shot)) so the regression is enforced in the PR lane.

**Step 2: Run RED**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_king_physics.py::TestHighW0CoreResolution -v
```
Expected: FAIL at W0=9 (+~18% at 0.3 r_c) and W0=12 (+~270%); W0=7 passes.

**Step 3: Implement** — in `king.py.__init__`, replace lines 358-360:

```python
        # Build radial grid for CDF — sqrt-stretched (r = r_t * u^2): spacing
        # dr ∝ sqrt(r) concentrates points in the core. A LINEAR grid leaves
        # <10 points inside the core at W0 >= 9 (xi_t grows super-exponentially:
        # 131 r_c at W0=9, 548 at W0=12), giving +18%..+270% core-mass errors
        # (audit R4, measured). Smooth in r_t -> differentiable in W0/r_c.
        u_grid = jnp.linspace(0.0, 1.0, n_grid)
        r_grid = r_t_arr * u_grid**2
        xi_grid_local = r_grid / r_c_arr
```

and replace the uniform-`dr` cumulative-mass block (lines ~386-392) with the non-uniform trapezoid:

```python
        integrand = 4.0 * jnp.pi * r_grid**2 * rho_grid
        M_cum = jnp.concatenate([
            jnp.zeros(1),
            jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(r_grid)),
        ])
```

(Keep whatever normalization/interp follows; only the grid and the quadrature weights change.) Check whether `progenax.numerics` already exposes a non-uniform `cumulative_trapezoid` — if yes, call it instead of inlining (DRY).

Apply the identical transformation to the CDF-grid blocks in `michie.py` (~line 177) and `eff.py` (~line 83). Read each block first; the variable names differ.

**Step 4: Run GREEN + regression**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_king_physics.py tests/validation/test_michie_physics.py tests/validation/test_eff_physics.py tests/unit/profiles -q
```
All pass, including the existing W0=7 CDF quadrature gates (`test_cdf_quadrature.py`) — the stretch must not degrade the already-good low-W0 regime.

**Step 5: Gradient regression** (the grid change touches the differentiable path):

```bash
env -u VIRTUAL_ENV uv run --no-sync python - <<'EOF'
import jax, jax.numpy as jnp
from progenax import KingProfile
masses, key = jnp.ones(300), jax.random.PRNGKey(7)
def mean_r(W0):
    pos = KingProfile.from_W0_rc(W0=W0, r_c=1.0).sample_positions(masses, key)
    return jnp.mean(jnp.linalg.norm(pos, axis=1))
g = float(jax.grad(mean_r)(5.0)); h = 1e-3
fd = float((mean_r(5.0+h) - mean_r(5.0-h)) / (2*h))
print(f"AD={g:.8f} FD={fd:.8f} ratio={g/fd:.6f}")
assert abs(g/fd - 1) < 1e-3
EOF
```
Expected: ratio within 1e-3 of 1 (pre-fix baseline was 0.999869).

**Step 6: Commit**

```bash
git add src/progenax/profiles/king.py src/progenax/profiles/michie.py src/progenax/profiles/eff.py tests/validation/
git commit -m "fix(profiles): sqrt-stretched CDF grids resolve the core at high W0 (audit R4)

Linear 1000-pt grids left <10 points in the core at W0>=9: measured core-mass
errors +18% (W0=9) and +270% (W0=12) vs a dense ODE reference. r = r_t*u^2
grid + non-uniform trapezoid; AD-vs-FD W0 gradient regression preserved."
```

### Task 1.3: `sample_fixed_n` silent mass shortfall (R5)

**The bug:** `BaseIMF.sample_fixed_n` (`src/progenax/imf/base.py:230-281`) promises "exactly n masses summing to m_total" but the one-sided quantile stretch (`q ≤ 1`) caps the achievable total at `Σ ppf((i+0.5)/n)`; the clipped Newton solver converges to the boundary silently. Measured: `Maschberger().sample_fixed_n(key, 1000, 500.0)` → **349.0 M☉**. None of the four sampling modes (`sample_m_total`, `sample_m_total_packed`, `sample_fixed_n`) has any test.

**Scope decision (YAGNI):** guard + honest docs + tests. A two-sided stretch is a behavior redesign — defer it; note it in the phase report as a follow-up option for Anna.

**Step 1: Write the failing tests** — create `tests/unit/imf/test_sampling_modes.py`:

```python
"""Tests for the BaseIMF mass-target sampling modes (audit R5: previously zero tests)."""
import jax
import jax.numpy as jnp
import pytest

from progenax.imf import Maschberger


class TestSampleFixedN:
    def test_reachable_target_is_hit(self):
        imf = Maschberger()
        key = jax.random.PRNGKey(0)
        n = 1000
        # natural total ~ n * E[m]; pick a comfortably reachable target
        target = 250.0
        m = imf.sample_fixed_n(key, n, target)
        assert m.shape == (n,)
        assert abs(float(m.sum()) / target - 1.0) < 0.05  # ~target, not "exactly"

    def test_unreachable_target_raises(self):
        """Pre-fix behavior: silently returned 349 Msun for a 500 Msun target."""
        imf = Maschberger()
        key = jax.random.PRNGKey(0)
        with pytest.raises(ValueError, match="unreachable"):
            imf.sample_fixed_n(key, 1000, 500.0)

    def test_masses_within_imf_bounds(self):
        imf = Maschberger()
        m = imf.sample_fixed_n(jax.random.PRNGKey(1), 500, 120.0)
        assert float(m.min()) >= imf.m_min
        assert float(m.max()) <= imf.m_max


class TestSampleMTotalModes:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_sample_m_total_hits_target(self, seed):
        imf = Maschberger()
        out = imf.sample_m_total(jax.random.PRNGKey(seed), m_total=500.0)
        # read the actual return signature first and unpack accordingly;
        # assert realized total within one max-mass overshoot of the target
        ...

    def test_sample_m_total_packed_consistent_with_unpacked(self):
        ...
```

Before writing the two `sample_m_total` tests, READ `src/progenax/imf/base.py:120-228` for the actual signatures and overshoot semantics (prefix-cut, ≤1-star overshoot) and encode those semantics — not your guess.

**Step 2: Run RED** — `test_unreachable_target_raises` must FAIL (no ValueError raised; total silently 349).

**Step 3: Implement the guard** — in `sample_fixed_n` (base.py:230-251), after computing `u_base`:

```python
        u_base = (jnp.arange(n) + 0.5) / n

        # Reachability: the one-sided stretch (q <= 1) caps the total at
        # sum(ppf(u_base)) — and stratification truncates the heavy tail, so
        # this ceiling sits ~5% BELOW n*E[m]. A target above it used to return
        # a silently short total (audit R5: asked 500, got 349). Eager inputs
        # fail loudly; traced inputs cannot be checked (documented).
        try:
            m_ceiling = float(jnp.sum(self.ppf(u_base)))
            if float(m_total) > m_ceiling:
                raise ValueError(
                    f"m_total={float(m_total):.6g} Msun is unreachable for n={n}: "
                    f"the stratified-quantile ceiling is {m_ceiling:.6g} Msun. "
                    f"Increase n, lower m_total, or use sample_m_total()."
                )
        except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
            pass  # traced m_total: caller owns reachability
```

Fix the docstring: replace "exactly n masses summing to m_total" with "n masses whose total approximates m_total (stratified quantile stretch; the realized total carries the residual of the final random jitter)" and document the ceiling + traced-input caveat.

**Step 4: GREEN** — run the new file + `tests/unit/imf -q` (no regressions).

**Step 5: Commit**

```bash
git add src/progenax/imf/base.py tests/unit/imf/test_sampling_modes.py
git commit -m "fix(imf): sample_fixed_n raises on unreachable m_total instead of silently undershooting (audit R5)

Measured: target 500 Msun, n=1000 returned 349 Msun with no warning. Adds the
stratified-ceiling guard (eager inputs), honest docstring, and first-ever tests
for the mass-target sampling-mode family."
```

### Task 1.4: `compute_stellar_radii` inverted exponents (R6)

**The bug:** `src/progenax/builders.py:155-193` assigns R ∝ M^0.8 *above* 1 M☉ and M^0.57 *below* — the standard homology/empirical assignment is the opposite. 10 M☉ → 6.31 R☉ (ZAMS ≈ 4); 0.2 M☉ → 0.40 R☉ (observed ≈ 0.22). No citation anywhere; tests assert the code's own formula back at itself.

**The fix:** adopt Demircan & Kahraman (1991, Ap&SS 181, 313) main-sequence fits — `R = 1.06·M^0.945` for M < 1.66 M☉, `R = 1.33·M^0.555` for M ≥ 1.66 M☉ — keeping the brown-dwarf plateau (0.1 R☉) below 0.08 M☉. Pleasant side effects: the low-mass branch meets the BD plateau almost continuously (1.06·0.08^0.945 = 0.0975 ≈ 0.1), and the D&K joint at 1.66 M☉ has only a ~3.5% step (their own fits; document it).

> ⛔ **Mini-checkpoint inside this task:** before implementing, ask Anna whether she prefers Demircan & Kahraman (1991) (recommended: simple, cited, two-branch) or Tout et al. (1996) ZAMS (higher fidelity, rational-polynomial). The plan below assumes D&K91.

**Step 1: Write the failing test** — replace the tautological block in `tests/unit/test_builders.py:11-31` with literature-anchored values:

```python
class TestComputeStellarRadii:
    """Anchored to Demircan & Kahraman (1991) Ap&SS 181, 313 + observed radii.

    Audit R6: the previous exponents were INVERTED vs MS homology
    (M^0.8 above 1 Msun instead of below), giving 10 Msun -> 6.3 Rsun
    (ZAMS ~4) and 0.2 Msun -> 0.40 Rsun (observed ~0.22). The old tests
    asserted the code's own formula back at itself.
    """

    def test_solar(self):
        r = compute_stellar_radii(jnp.array([1.0]))
        assert abs(float(r[0]) - 1.06) < 0.01  # D&K91: 1.06 * 1^0.945

    def test_massive_star_near_zams(self):
        r = float(compute_stellar_radii(jnp.array([10.0]))[0])
        assert abs(r - 1.33 * 10.0**0.555) < 1e-6  # = 4.77 Rsun
        assert 3.0 < r < 5.5  # within ~25% of ZAMS ~4 Rsun

    def test_m_dwarf(self):
        r = float(compute_stellar_radii(jnp.array([0.2]))[0])
        assert abs(r - 1.06 * 0.2**0.945) < 1e-6  # = 0.231 Rsun
        assert 0.17 < r < 0.28  # observed ~0.22 Rsun

    def test_brown_dwarf_plateau(self):
        r = compute_stellar_radii(jnp.array([0.05, 0.02]))
        assert jnp.all(jnp.abs(r - 0.1) < 0.05)

    def test_near_continuous_at_hydrogen_burning_limit(self):
        r = compute_stellar_radii(jnp.array([0.079, 0.081]))
        assert abs(float(r[1]) / float(r[0]) - 1.0) < 0.1  # no factor-2.4 jump (audit F7)
```

**Step 2: Run RED** — `test_solar` fails (old code: 1.0^0.57 = 1.0 ≠ 1.06), `test_massive_star_near_zams` fails (6.31).

**Step 3: Implement** — replace the body of `compute_stellar_radii` (builders.py:155-193):

```python
def compute_stellar_radii(masses: Float[Array, "N"]) -> Float[Array, "N"]:
    """
    Main-sequence stellar radii from mass, in SOLAR RADII.

    Demircan & Kahraman (1991), Ap&SS 181, 313 (their Eqs. 5-6):
        R/Rsun = 1.06 (M/Msun)^0.945   for 0.08 <= M < 1.66 Msun
        R/Rsun = 1.33 (M/Msun)^0.555   for M >= 1.66 Msun
    Brown dwarfs (M < 0.08): R ~ 0.1 Rsun plateau (electron degeneracy),
    R = 0.1 (M/0.08)^0.08. The D&K branches meet at 1.66 Msun with their
    own ~3.5% fit discontinuity; the low-mass branch meets the BD plateau
    nearly continuously (1.06 * 0.08^0.945 = 0.0975).

    Used for collision radii in downstream N-body; ZAMS values, no evolution.
    """

    def radius_high_mass(m):
        return 1.33 * jnp.power(m, 0.555)

    def radius_low_mass(m):
        return 1.06 * jnp.power(m, 0.945)

    def radius_brown_dwarf(m):
        return 0.1 * jnp.power(m / 0.08, 0.08)

    return jax.vmap(
        lambda m: jax.lax.cond(
            m >= 1.66,
            radius_high_mass,
            lambda mv: jax.lax.cond(
                mv >= 0.08, radius_low_mass, radius_brown_dwarf, mv
            ),
            m,
        )
    )(masses)
```

**Step 4: GREEN + downstream check** — run `tests/unit/test_builders.py -q` and grep for other consumers of `stellar_radii` expectations: `grep -rn "stellar_radii\|compute_stellar_radii" tests/ src/`.

**Step 5: Commit**

```bash
git add src/progenax/builders.py tests/unit/test_builders.py
git commit -m "fix(builders): stellar mass-radius relation per Demircan & Kahraman (1991) (audit R6)

Previous exponents were inverted vs MS homology (uncited): 10 Msun -> 6.3 Rsun
instead of ~4.8, 0.2 Msun -> 0.40 instead of ~0.23. Tests now anchored to the
published fit, not the implementation."
```

### Task 1.5: PowerLawIMF NaN gradient at α = 1 (R10)

**The bug:** four sites in `src/progenax/imf/power_law.py` use the bare `jnp.where(|e| < 1e-12, log_branch, (hi**e − lo**e)/e)` pattern; the untaken branch computes 0/0 and the `where` VJP propagates 0 × NaN = NaN. Confirmed: `jax.grad(mean_mass)(α=1.0)` → nan (interior values FD-exact). The correct double-`where` fix already exists at `src/progenax/imf/differentiable.py:47-54` — port it.

**Sites** (verify each by reading; line numbers drift): `segment_integral` (~88-96), `_cdf_unnorm` (~200-216), `ppf` (~246-252), `mean_mass` (~276-288).

**Step 1: Write the failing test** — append to `tests/unit/imf/test_imf_gradients.py`:

```python
class TestAlphaOneGradients:
    """Audit R10: bare where(|1-a|<eps, log, pow/e) backprops 0*NaN at a=1.

    The exp_safe double-where fix exists in imf/differentiable.py:47-54;
    these tests pin its port to PowerLawIMF (4 sites).
    """

    @staticmethod
    def _imf(alpha):
        from progenax.imf import PowerLawIMF
        return PowerLawIMF(exponents=[alpha], breakpoints=[], m_min=0.1, m_max=100.0)

    def test_mean_mass_grad_finite_and_fd_correct_at_alpha_one(self):
        f = lambda a: self._imf(a).mean_mass()
        g = jax.grad(f)(1.0)
        assert jnp.isfinite(g), f"grad(mean_mass) at alpha=1 is {g}"
        h = 1e-4
        fd = (f(1.0 + h) - f(1.0 - h)) / (2 * h)
        assert abs(float(g) / float(fd) - 1.0) < 1e-4

    def test_ppf_grad_finite_at_alpha_one(self):
        f = lambda a: self._imf(a).ppf(jnp.array(0.5))
        g = jax.grad(f)(1.0)
        assert jnp.isfinite(g)
        h = 1e-4
        fd = (f(1.0 + h) - f(1.0 - h)) / (2 * h)
        assert abs(float(g) / float(fd) - 1.0) < 1e-4

    def test_sample_statistic_grad_finite_at_alpha_one(self):
        def loss(a):
            m = self._imf(a).sample(jax.random.PRNGKey(0), 500)
            return jnp.mean(jnp.log(m))
        assert jnp.isfinite(jax.grad(loss)(1.0))
```

**Step 2: Run RED** — all three FAIL with nan at α=1.0.

**Step 3: Implement** — at each of the four sites apply the pattern (shown for `segment_integral`, power_law.py:88-96):

```python
        def segment_integral(i):
            a = alphas[i]
            lo, hi = bounds[i], bounds[i + 1]
            e = 1.0 - a
            # exp_safe double-where: keep BOTH branches finite so the VJP
            # is finite at exactly alpha=1 (bare where backprops 0*NaN).
            # Same pattern as imf/differentiable.py power_integral.
            e_safe = jnp.where(jnp.abs(e) < 1e-12, 1.0, e)
            return jnp.where(
                jnp.abs(e) < 1e-12,
                cont[i] * jnp.log(hi / lo),
                cont[i] * (hi**e_safe - lo**e_safe) / e_safe,
            )
```

For `ppf`, the inverse direction also contains `(...)**(1/e)`-type expressions — substitute `e_safe` inside EVERY power/division involving `e` in the untaken branch, not just the first one you see. Read the whole function before editing.

**Step 4: GREEN** — new tests + `tests/unit/imf -q` + `tests/validation/test_imf_physics.py -q` (forward values must be bit-unchanged away from α=1; the existing FD tests at α=2.35 verify it).

**Step 5: Commit**

```bash
git add src/progenax/imf/power_law.py tests/unit/imf/test_imf_gradients.py
git commit -m "fix(imf): finite gradients at alpha=1 exactly — port exp_safe double-where to PowerLawIMF (audit R10)"
```

### ⛔ CHECKPOINT 1
Run the FULL gate. Report: per-fix RED→GREEN evidence (show the failing output and the passing output), the FULL-gate count (expect ≈ 1192 + ~15 new), and any test whose expectation you had to re-derive (especially Task 1.1 Step 4 — show the derivation). **Wait for Anna's approval.** Flag for her decision: two-sided stretch for `sample_fixed_n` (deferred), D&K91 vs Tout+1996 confirmation.

---

# Phase 2 — JAX / API hardening (J4, J5, J6, S1, S2, A1)

### Task 2.1: King traced-W0 silent domain pinning (J4)

`_auto_ode_domain` (`king.py:159-166`) falls back to the fixed (300, 2000) domain under tracing; for W0 ≳ 10 the ODE never crosses ψ=0 and `_find_tidal_radius` silently pins ξ_t to the boundary — a wrong answer exactly where the user can't see it.

**Step 1: Failing test** — append to `tests/unit/profiles/test_king.py`:

```python
def test_traced_high_w0_without_explicit_domain_raises():
    """Audit J4: traced W0>9.5 at the fixed fallback domain silently pins
    xi_t to the boundary. Concrete high W0 auto-scales (fine); traced high
    W0 must be an explicit, loud choice via xi_max."""
    import pytest
    with pytest.raises(ValueError, match="xi_max"):
        jax.jit(lambda w: KingProfile.from_W0_rc(W0=w, r_c=1.0))(12.0)


def test_traced_typical_w0_still_works():
    prof = jax.jit(lambda w: KingProfile.from_W0_rc(W0=w, r_c=1.0))(7.0)
    assert jnp.isfinite(prof.r_t)
```

Wait — a ValueError cannot depend on a *traced value*. Correct design: `from_W0_rc` cannot know the traced W0. Instead, make the **pinning detectable and loud at solve time**: have `_find_tidal_radius` (king.py:241-283) compute `is_pinned = psi_grid[-1] > 0` (a traced bool) and store it on the profile as a diagnostic field `r_t_is_pinned`; for CONCRETE W0 > 9.5 with the fallback domain this cannot happen (auto-scaling), so additionally raise eagerly in `_auto_ode_domain`'s except-branch is impossible (W0 unknown). So the testable contract is:

```python
def test_pinned_domain_is_flagged():
    """Force the failure mode concretely: explicit too-small xi_max."""
    prof = KingProfile.from_W0_rc(W0=12.0, r_c=1.0, xi_max=50.0, n_ode_points=500)
    assert bool(prof.r_t_is_pinned)

def test_healthy_solve_not_flagged():
    prof = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
    assert not bool(prof.r_t_is_pinned)

def test_concrete_pinned_solve_raises():
    """Eager construction with a domain that pins xi_t must refuse loudly."""
    import pytest
    with pytest.raises(ValueError, match="pinned|xi_max"):
        KingProfile.from_W0_rc(W0=12.0, r_c=1.0, xi_max=50.0, n_ode_points=500)
```

(The first test then becomes the traced-path behavior: flag stored, no raise — the Engine-B two-tier concrete/traced pattern already used in `cluster/eddington_engine.py:192-217`; read it and mirror it.) Drop the two draft tests above in favor of these three.

**Step 2-4:** RED → implement (`r_t_is_pinned: Bool[Array, ""]` eqx field; eager `ValueError` when `jnp.ndim`-concrete and pinned; mirror the Engine-B `_is_concrete` guard) → GREEN → commit:

```bash
git commit -m "fix(profiles): King r_t boundary-pinning is now flagged (traced) or refused (eager) (audit J4)"
```

### Task 2.2: `virial_scale` T=0 NaN + dedupe with its sibling (J5)

`builders.py:256-258` divides by `Q_current` with no guard → cold input (v=0) yields NaN velocities silently; the near-duplicate `dynamics/virial.py:251 rescale_velocities_to_virial` has a different guard. Rescaling from T=0 is mathematically impossible — refuse loudly.

**Step 1: Failing test** (`tests/unit/test_builders.py`):

```python
def test_virial_scale_zero_velocities_raises():
    """Audit J5: cold input used to return all-NaN velocities silently."""
    import pytest
    pos = jax.random.normal(jax.random.PRNGKey(0), (50, 3))
    vel = jnp.zeros((50, 3))
    m = jnp.ones(50)
    with pytest.raises(ValueError, match="zero kinetic"):
        virial_scale(pos, vel, m, Q_target=0.5, G=0.00449)
```

**Step 2-4:** RED → implement an eager concrete-input check (`try: float(T) ... except ConcretizationTypeError: pass`, raising `ValueError("cannot rescale from zero kinetic energy (T=0): velocities are all zero")`; for traced inputs keep a `jnp.where(T > 0, scale, jnp.nan)` — NaN is then the honest traced sentinel, document it) → GREEN. Then reconcile with `rescale_velocities_to_virial`: make `virial_scale` delegate to it or vice versa so there is ONE implementation (DRY; read both first — they differ in COM handling, keep the semantics of each public signature). Commit:

```bash
git commit -m "fix(builders): virial_scale refuses T=0 input; dedupe with rescale_velocities_to_virial (audit J5)"
```

### Task 2.3: `q_approx` static-N branch (J6)

`diagnostics/q_approx.py:270-275` (and `_compute_s_bar_subsampled:239`) use `lax.cond` on a *Python int* N — both branches compile every shape (~0.4 s wasted). Replace with Python `if N > 1000:` — identical semantics, half the compile.

No new physics test needed (pure performance); the regression is the existing `tests/unit/substructure` + `tests/unit/diagnostics` suites passing unchanged, plus:

```python
def test_auto_dispatch_small_n_equals_naive():
    pos = jax.random.normal(jax.random.PRNGKey(0), (200, 3))
    assert jnp.allclose(q_approx(pos, method="auto"), q_approx(pos, method="naive"))

def test_auto_dispatch_large_n_equals_fast():
    pos = jax.random.normal(jax.random.PRNGKey(0), (1500, 3))
    assert jnp.allclose(q_approx(pos, method="auto"), q_approx(pos, method="fast"))
```

Commit: `perf(diagnostics): Python-if dispatch on static N in q_approx — avoids compiling both branches (audit J6)`.

### Task 2.4: King r_t consistency guard (S1, A3, M4)

`KingProfile`'s direct constructor accepts an arbitrary `r_t` inconsistent with c(W0) — `KingProfile(W0=7, r_c=1, r_t=10)` (the README's own example!) silently builds a non-self-consistent, non-equilibrium model. `KingVelocityDF.r_t` (`king_df.py:101,123`) is stored and never used.

**Design (confirm at checkpoint if unsure):** in the direct `KingProfile.__init__`, when ALL inputs are concrete, warn (`warnings.warn`) if `r_t` deviates from `r_c · ξ_t(W0)` by > 5%; `from_W0_rc` (which derives r_t) is untouched. Remove the dead `r_t` field from `KingVelocityDF` — this repo's policy is no backwards-compat shims; update every callsite (`grep -rn "KingVelocityDF(" src tests docs README.md`).

**TDD:** failing test asserting `pytest.warns(UserWarning, match="inconsistent")` for the bad triple and no warning for the consistent one; failing test asserting `KingVelocityDF(W0=7.0, r_c=1.0)` constructs (no r_t arg). RED → implement → GREEN → update callsites → commit:

```bash
git commit -m "fix(api): warn on inconsistent KingProfile r_t; drop the dead KingVelocityDF.r_t field (audit S1)"
```

### Task 2.5: Pin the EFF γ=3 sub-virial offset (S2)

The default EFF DF (γ=3, sharp truncation) is knowingly ~5-8% sub-virial (Eddington inversion of a truncated ρ with ρ(r_t)>0 — disclosed in the module docstring, but the default config is the broken case and nothing pins it, so a regression to −15% would pass).

**Step 1: Failing-by-absence test** — append to `tests/validation/test_eff_physics.py`:

```python
def test_gamma3_default_subvirial_offset_is_pinned():
    """Audit S2: gamma=3 + sharp truncation is KNOWN ~5-8% sub-virial by
    construction (f(E) cannot represent rho(r_t)>0). Pin the band so a
    regression (e.g. to -15%) cannot pass silently. This is a documented
    limitation, not a target: see eff_df.py module docstring."""
    # build the DEFAULT EFFProfile/EFFVelocityDF pair (read the current
    # defaults from eff.py / eff_df.py), sample N=20_000, seed 0,
    # compute Q = T/|V| with compute_kinetic_energy / compute_potential_energy
    # at G=STELLAR.G, and assert:
    assert 0.42 < Q < 0.49, f"gamma=3 Q drifted out of the documented band: {Q}"
```

Fill in the construction by copying the existing γ=5 equilibrium test in the same file (lines ~227-249) and switching to defaults. Measure Q at 3 seeds first, set the band as measured ± 3σ (measured-first frozen gates — the project convention), THEN freeze. Also surface the caveat at the constructor docstring level in `eff_df.py` (currently module-level only).

Commit: `test(eff): pin the documented gamma=3 sub-virial offset + constructor-level caveat (audit S2)`.

### Task 2.6: Dedupe the `VelocityDF` protocol (A1)

Two structurally identical `runtime_checkable` Protocols: `protocols.py:54-83` (canonical) and `kinematics/api.py:53-70` (drift hazard). Make `kinematics/api.py` import from `progenax.protocols`; `grep -rn "VelocityDF" src/ tests/` and fix every import that referenced the api.py copy. Regression: `tests/unit/test_protocols.py` + FAST gate. Commit: `refactor: single source of truth for the VelocityDF protocol (audit A1)`.

### ⛔ CHECKPOINT 2
FULL gate + report. Flag for Anna: the S1 design decision (warn vs raise on inconsistent r_t), and the deferred units-policy question (A2: should DF `G=None` defaults be removed for protocol-wide explicit G? — pre-1.0 is the time; recommend deciding now but it is a breaking API sweep, so it's HER call, not yours).

---

# Phase 3 — Documentation honesty sweep (R7, D3–D5, S3, S4, S6, S10, S16, T5, L1)

No physics changes in this phase; the verification is (a) every documented symbol imports, (b) every documented command runs, (c) the docs-example smoke test passes, (d) myst build clean.

### Task 3.1: Reconcile the public API with the docs (R7 part 1)

**Decision (recommended; confirm at checkpoint):** export the IMF/binary-statistics symbols from `progenax.__init__` — the docs already promise them and they are stable.

**Step 1: Failing test** — create `tests/unit/test_documented_api.py`:

```python
"""Every symbol the docs promise must import from the package root (audit R7).

IGIMF and EnvironmentIMF are deliberately ABSENT: they never existed
(the environment-dependent IMF is the functional BirthEnvironment +
env_to_imf_params API); the docs are being fixed to stop advertising them.
"""
import pytest

DOCUMENTED_ROOT_SYMBOLS = [
    "PowerLawIMF", "ChabrierIMF", "Maschberger", "TruncatedIMF", "BinaryIMF",
    "FlatMassRatio", "PowerLawMassRatio", "TwinPeakedMassRatio",
    "MoeDiStefano2017", "MoeDiStefano2017Full", "MoePeriod", "MoeJointOrbit",
    "ConstantBinaryFraction", "MassDependentBinaryFraction",
    "UniformSphereProfile",  # audit L1: only profile missing from the root
]

@pytest.mark.parametrize("name", DOCUMENTED_ROOT_SYMBOLS)
def test_symbol_importable_from_root(name):
    import progenax
    assert hasattr(progenax, name), f"progenax.{name} promised by docs, not exported"

def test_phantom_classes_stay_dead():
    import progenax
    for phantom in ("IGIMF", "EnvironmentIMF"):
        assert not hasattr(progenax, phantom)
```

**Step 2-4:** RED (15 failures) → add the imports + `__all__` entries to `src/progenax/__init__.py` (group under an `# IMF / binary statistics` comment matching the existing style) → GREEN.

**Step 5: Purge the phantoms from docs.** `grep -rn "IGIMF\|EnvironmentIMF" CLAUDE.md ../CLAUDE.md README.md docs/ src/` — replace every class-like mention with the real API ("environment-dependent IMF: `BirthEnvironment` + `env_to_imf_params()` (Marks+2012/Jeřábková+2018 α₃ relations)"). The audit (§4.3) is explicit: never use the word "IGIMF" for this machinery — it is not galaxy-wide IGIMF integration.

Commit: `fix(api+docs): export the documented IMF symbols; kill the phantom IGIMF/EnvironmentIMF claims (audit R7)`.

### Task 3.2: README rewrite (D3, H1, H2)

Rewrite `README.md` against reality. Requirements checklist (verify each with a command, not by eye):

- [ ] Quickstart uses ONLY current API — the `TwoComponentConfig`/`generate_two_component_cluster` block (lines ~154-175) is replaced by a `MultiComponentCluster` example (`from_components` or `from_density_profiles`; copy a working invocation from `tests/unit/cluster/`) plus the basic `build_spatial_ic` example.
- [ ] `MultiComponentCluster`, `MichieProfile`/`MichieVelocityDF`, `LIMEPYVelocityDF` appear in the feature tables.
- [ ] `apply_osipkov_merritt()` (line ~34) replaced by the real `anisotropy_radius` constructor argument.
- [ ] The King example `KingVelocityDF(W0=7.0, r_c=1.0, r_t=10.0)` (line ~320) fixed to the post-Task-2.4 signature.
- [ ] Installation: uv-first, sibling-checkout layout, only the extras that exist (`dev`, `experimental`, plus `diagnostics` after Task 4.2); the conda block deleted; an honest "not on PyPI yet — requires the jaxstro sibling checkout" banner.
- [ ] Stale metrics removed or regenerated (LOC/test counts — quote the FULL-gate number from CHECKPOINT 1, or drop counts entirely; audit T5/L3: three documents currently give three different numbers).
- [ ] "3 runtime-checkable protocols" → 9 (or just "protocol-based", uncounted).
- [ ] License section matches the actual LICENSE file (Task 4.1).
- [ ] No PRNG key reuse in any example (the installation smoke test currently passes the same key twice — split keys in every snippet).

**Verification — the docs-example smoke test (audit test-matrix item 9).** Create `tests/integration/test_readme_examples.py`:

```python
"""Execute every python code block in README.md (audit: would have caught R7).

Skips blocks marked `# doctest: +SKIP` (e.g. GPU-only). Each block runs in a
fresh namespace with a 120 s budget.
"""
import re
from pathlib import Path

import pytest

README = (Path(__file__).resolve().parents[2] / "README.md").read_text()
BLOCKS = [
    b for b in re.findall(r"```python\n(.*?)```", README, re.DOTALL)
    if "+SKIP" not in b
]

@pytest.mark.parametrize("i", range(len(BLOCKS)))
def test_readme_block_executes(i):
    exec(compile(BLOCKS[i], f"README.md:block{i}", "exec"), {})
```

Run it RED against the CURRENT README first (it must fail on `TwoComponentConfig` — proof the harness works), then rewrite, then GREEN. Same treatment for `docs/website/00-getting-started/installation.md` code blocks if cheap; at minimum fix the `[all]`/`[io]`/`[viz]`/`[ml]` phantom extras and the monorepo-vs-split-repo clone instructions (H2), and the reused-key smoke test.

Commit: `docs: README rewritten against the current API, with an executable-examples smoke test (audit R7/D3/D4)`.

### Task 3.3: Kill the `progenax-legacy` fabrication (D5/H4)

`docs/website/20-architecture/ic-redesign-history.md:110` claims `pip install progenax-legacy` works — it does not exist on PyPI. Rewrite that paragraph to: the pre-redesign code is preserved at the repo tag/path (find the real one: `git tag` / `ls ../legacy/`), no install claim. Then `grep -rn "progenax-legacy" docs/ README.md` → zero hits. Verify the website still builds: `env -u VIRTUAL_ENV uv run --no-sync myst build 2>&1 | tail -3` (expect 0 warnings, ~151 pages). Commit.

### Task 3.4: Equilibrium-caveat docstrings (S3, S4, S6, S10, S16, S15)

Pure docstring/comment fixes — one commit, no test changes, but each claim must be *accurate*, so read the audit entry + the function before writing:

- `kinematics/rotation.py` (both transforms + module): add "kinematic overlay — injects kinetic energy and L_z; the output is NOT a stationary equilibrium (Q > 0.5)"; remove the misleading `target_Q` rescale suggestion in `kinematics/api.py:199-235` (an isotropic rescale does not restore stationarity); fix the stale zero-axis NaN comment (`rotation.py:49-50`) — behavior is a silent no-op; add an eager ValueError for a concrete zero axis (this one IS testable: 2-line test).
- `tidal.py:apply_tidal_truncation`: paragraph on the survivor set being super-virial w.r.t. its own reduced potential (some stars formally unbound near r_t); recommend `rescale_velocities_to_virial` or King/LIMEPY r_t-consistent models.
- `cluster/mass_segregation.py:energy_sorted_segregation`: the segregated mass-weighted density no longer matches the parent profile → the self-consistent potential differs; callers should finalize with a global virial rescale (as the validation suite already does).
- `binaries/diagnostics.py:222-223`: "inflated" → "deflated" (measured Q_resolved 0.31 < 0.5); add the one-line note that `softening` does not soften `E_internal`.
- `dynamics/virial.py:201-203`: per-group W_j is origin-dependent for subgroups; document "positions must be pre-centered" (or center internally — your call, but if you center, run the cluster validation tests).

Commit: `docs(src): honest equilibrium caveats — rotation overlays, tidal truncation, segregation, energy-budget wording (audit S3/S4/S6/S10/S16)`.

### Task 3.5: Reconcile the three stale test-count claims (T5/L3/D6)

`tests/README.md` says 874; `progenax/CLAUDE.md` says 1163; STATUS.md says 1192. Update `tests/README.md` and `CLAUDE.md` to the post-Phase-1 FULL-gate count (quote the actual pytest output), and where counts appear in prose, prefer "the released-core suite (see CI)" over a number that rots. Commit.

### ⛔ CHECKPOINT 3
FULL gate + `myst build` output + the README smoke test green + `grep` proofs (zero hits for `TwoComponentConfig|progenax-legacy|IGIMF` outside the audit report and dev-log history pages). **Wait for approval.**

---

# Phase 4 — Packaging & metadata (R2, R9, D1, D7, M6/H1-CI)

### Task 4.1: LICENSE + project metadata (D1/C3)

- Create `LICENSE` — MIT, copyright `2026 Anna Rosen` (⛔ confirm holder/year with Anna at the checkpoint if any doubt — the README claims MIT already).
- In `pyproject.toml [project]` add:

```toml
license = "MIT"
license-files = ["LICENSE"]
authors = [{ name = "Anna Rosen", email = "alrosen@sdsu.edu" }]
keywords = ["astronomy", "n-body", "initial-conditions", "jax", "star-clusters", "differentiable"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Astronomy",
]

[project.urls]
Repository = "https://github.com/jaxstro/progenax"
```

- `uv lock` (metadata change → lockfile refresh), then verify the wheel:

```bash
env -u VIRTUAL_ENV uv build --wheel -o /tmp/progenax_wheel_check
python3 -c "
import zipfile, glob
w = glob.glob('/tmp/progenax_wheel_check/*.whl')[0]
meta = zipfile.ZipFile(w).read([n for n in zipfile.ZipFile(w).namelist() if n.endswith('METADATA')][0]).decode()
for field in ('License', 'Author', 'Classifier', 'Project-URL'):
    assert field in meta, field
print('wheel metadata OK')
assert not any('experimental' in n or 'gravoturb' in n for n in zipfile.ZipFile(w).namelist())
print('wheel content clean (no experimental leakage)')
"
```

Commit: `chore(packaging): LICENSE (MIT) + author/classifiers/URLs in wheel metadata (audit D1)`.

### Task 4.2: `[diagnostics]` dependency story (R9/H3)

`progenax.diagnostics` imports numpy + scipy eagerly (`substructure.py:20-22`, `mass_segregation.py:18-20`, both pulled in by `diagnostics/__init__.py:41-46`), but neither is a declared dependency → `from progenax.diagnostics import compute_q_parameter` explodes in a clean install.

**Step 1: Failing test** — `tests/unit/diagnostics/test_lazy_imports.py`:

```python
import subprocess, sys

def test_diagnostics_import_without_scipy_gives_actionable_error():
    """Audit R9: in an env without scipy, importing the subpackage must not
    crash at import time; calling an exact (scipy-backed) function must raise
    an ImportError naming the [diagnostics] extra."""
    code = (
        "import sys;"
        "sys.modules['scipy'] = None; sys.modules['scipy.sparse'] = None;"
        "sys.modules['scipy.sparse.csgraph'] = None; sys.modules['scipy.spatial'] = None;"
        "sys.modules['scipy.spatial.distance'] = None;"
        "import progenax.diagnostics as d;"           # must import fine
        "from progenax.diagnostics import q_approx;"  # JAX-native: must work
        "import jax.numpy as jnp, jax;"
        "q_approx(jax.random.normal(jax.random.PRNGKey(0), (50, 3)));"
        "import pytest;"
        "exc = None\n"
        "try:\n"
        "    d.compute_q_parameter(jax.random.normal(jax.random.PRNGKey(0), (50, 3)))\n"
        "except ImportError as e:\n"
        "    exc = e\n"
        "assert exc is not None and 'diagnostics' in str(exc), exc\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
```

**Step 2-4:** RED → implement: move the numpy/scipy imports inside the functions that need them (`compute_q_parameter`, `compute_lambda_msr`, the MST builders), wrapped:

```python
    try:
        import scipy.sparse.csgraph as csgraph
    except ImportError as e:
        raise ImportError(
            "compute_q_parameter needs numpy+scipy (the exact, non-differentiable "
            "diagnostics path). Install them with: uv pip install 'progenax[diagnostics]'"
        ) from e
```

and add to `pyproject.toml`:

```toml
diagnostics = ["numpy>=1.24", "scipy>=1.11"]
```

(`uv lock` after.) → GREEN → commit: `fix(packaging): [diagnostics] extra + lazy numpy/scipy imports with actionable errors (audit R9)`.

### Task 4.3: Wheel-smoke + version-matrix CI (M6, T3, audit test-matrix items 4 & 10)

Append two jobs to `.github/workflows/tests.yml`:

```yaml
  wheel-smoke:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      # same sibling checkouts as released-core
      # ...
      - name: Setup uv
        uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.13"
      - name: Build wheel
        working-directory: progenax
        run: uv build --wheel -o dist/
      - name: Install wheel in a clean venv and import
        working-directory: progenax
        run: |
          uv venv /tmp/clean
          uv pip install --python /tmp/clean/bin/python dist/*.whl ../jaxstro
          /tmp/clean/bin/python -c "import progenax; import progenax.diagnostics; print(progenax.__name__, 'imports clean')"
```

and a nightly matrix job in `physics-validation.yml` (NOT the PR lane — keep PR latency flat): `python-version: ["3.10", "3.13"]` running the unit tier only. If 3.10 fails for an uninteresting reason (e.g. a dependency floor), the honest alternative is raising `requires-python` — flag it at the checkpoint rather than burning time.

Add `wheel-smoke` to the `tests` aggregator's `needs` + result check. Commit.

### Task 4.4: CHANGELOG + the jaxstro decision (D7, R2) — ⛔ DECISION TASK, do not implement unilaterally

- Create `CHANGELOG.md` with an `## 0.1.0 (unreleased)` section summarizing the audit-fix arc (one line per R-finding fixed, with commit refs).
- **The pip-installability blocker (R2) requires Anna's strategy decision** (audit §9): (A — recommended) publish `jaxstro` to PyPI first and pin `jaxstro>=0.1,<0.2`; (B) vendor `units`+`jaxconfig` into `progenax._vendor`; (C) git-URL dep for a GitHub-only soft launch. Present the trade-offs from the audit verbatim and STOP. Whichever she picks is a separate follow-up arc (Option A is mostly work in the jaxstro repo).

### ⛔ CHECKPOINT 4 (final)
1. FULL gate output (expect ≈ 1192 + ~25 new tests, 0 failures).
2. FAST gate output.
3. `PROGENAX_STRICT_REFS=1` slow lane locally once (`pytest tests -m slow -q`) — the trust anchors must pass with the Phase-1 physics fixes in (the LIMEPY parity and Engine A/B tests must be untouched by the CDF-grid change: Engine A samples via its own DF tables, but VERIFY, don't assume).
4. CI green on `fix/prerelease-audit` (all jobs incl. wheel-smoke), `gh run` URL.
5. Wheel metadata + content check output (Task 4.1).
6. Completion doc `.claude-work/PRERELEASE_AUDIT_FIXES_COMPLETE.md`: per-finding before/after evidence table (R1–R10 + each S/J/A/T/D handled), lessons learned, and the open Anna-decision list (jaxstro strategy, units-policy A2, two-sided stretch, D&K91-vs-Tout).
7. Update `STATUS.md` `next:`/`blocker:` lines.
8. **Merge/PR: Anna's call.** Per the project git workflow: merge to local main only on her go; ONE final PR when CI is green.

---

## Explicitly OUT of scope (deferred, listed so they aren't silently dropped)

- jaxstro PyPI publication / vendoring (Anna decision, Task 4.4).
- Units-policy A2 sweep (drop `G=None` from DF protocol) — breaking change, Anna decision.
- `q_approx` calibration sweep over (profile, N) (S5) — separate validation arc.
- Pre-paper PDF verification sweep (Moe Eqs. 17-18, CW04 Table 1, GZ15 §2.2) — requires Anna's eyes on the PDFs per project policy.
- 10-t_cross gravax stationarity runs per DF — paper arc, needs gravax.
- Two-sided quantile stretch for `sample_fixed_n` (R5 follow-up).
- Experimental `gravoturb_fdf` GRF β-gradient NaN (J3) — off the production inference path; fix opportunistically if touching that file.
