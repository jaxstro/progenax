# LIMEPY DF Tables (Phase 1.5) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> superpowers:subagent-driven-development) to implement this plan task-by-task.

**Goal:** Make anisotropic `MultiComponentCluster` construction ~5–10× faster and
large-N sampling ~5–10× faster by replacing repeated pointwise quadrature of the
smooth 2-D function ρ̂(W, p; g) with a precomputed, differentiable interpolation
table — while keeping the exact quadrature path as the always-available oracle and
proving the table reproduces it to a stated accuracy budget.

**Architecture:** A new `AnisoDensityTable` (eqx.Module) tabulates the anisotropic
LIMEPY density ρ̂(W, p) for fixed g on a (√W, asinh p) grid, built in ONE batched
call with the existing exact quadrature. One table is shared by all components of a
coupled solve (component j only shifts the arguments: W_j = rescale_j·ψ,
p_j = ξ/r̂_{a,j}). `solve_multicomponent_limepy` gains `aniso_method="table"|
"quadrature"` (default "table" once validated); the quadrature path is untouched and
remains the oracle. A second tranche tabulates the speed CDF for the sampler.
`component_virial_ratios` deliberately STAYS on exact quadrature (it is the
equilibrium proof and must remain independent of the table approximation).

**Tech Stack:** JAX (jax.numpy only in src/), Equinox, diffrax (existing solver),
pytest (+xdist). float64 is automatic on `import progenax`.

**Measured baseline (commit 3f15479, 12-core M-series, warm):** anisotropic
construction ≈ 1.2–1.8 s of which the ODE solve is 86% (~3,060 RHS-equivalent
evaluations × 382 µs per 2-component density eval); sampling 59–73 µs/star
anisotropic, 22–36 µs/star isotropic.

---

## Context for an engineer with zero progenax background

- **Repo:** `~/projects/jaxstro-dev/progenax`, branch `feat/multimass-limepy-equilibrium`.
  Do NOT push. Commit after each task with the message given in the task.
- **Run anything via uv:** prefix every python/pytest command with
  `env -u VIRTUAL_ENV uv run --no-sync `. For parallel test runs add
  `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`
  and `-n auto`.
- **JAX-native rules (CLAUDE.md, non-negotiable):** `jax.numpy` only in `src/`
  (no numpy/scipy); no Python loops over data in hot paths; `jax.lax.scan` with
  fixed iterations (never `while_loop`); everything differentiable; eqx.Module for
  stateful classes; explicit `G`/units; functions ≤100 LOC, files ≤500 LOC.
- **Never weaken an existing test to make it pass.** If a physics test fails, the
  implementation is wrong.
- **Key existing code (READ these files before coding):**
  - `src/progenax/profiles/limepy.py` — `lowered_exponential(g, x)` (E_γ(g+3/2, x),
    the lowered-exponential kernel), `_angle_integral_T(beta)` (the tangential angle
    integral T(β) = ∫₀¹ e^{−β(1−c²)} dc as a stable bounded series), and
    `_aniso_density_scalar(W, p, g)` — THE exact anisotropic density quadrature this
    plan tabulates. Signature: scalar W ≥ 0, scalar p ≥ 0, scalar g → scalar ρ̂
    (UNNORMALIZED; callers normalize by the central value).
  - `src/progenax/profiles/limepy_multimass.py` — `solve_multicomponent_limepy(
    alpha_j, rescale_j, W0, g, xi_max=300.0, n_points=2000, ra_hat_j=None)`; the
    anisotropic RHS closure `density_components` (lines ~140–155) evaluates
    `_aniso_density_scalar(rescale_j·ψ, ξ/ra_j, g)` per component per ODE step —
    this is the 86% hotspot. Also `_grid_density_components(...)` (grid version,
    same math) and the non-truncation guard at the end (keep it working).
  - `src/progenax/cluster/multicomponent.py` — `MultiComponentCluster`
    (`from_components` / `from_mass_segregation` / `from_imf`), the jitted sampler
    core `_sample_cluster_arrays` (bottom of file), and `component_virial_ratios`
    (do NOT switch it to tables).
  - `src/progenax/kinematics/limepy_df.py` — `_sample_unit_speed(key, W, g,
    n_speed)` (isotropic per-star inverse-CDF speed draw) and `_sample_speed_angle(
    key, W, p, g, n_speed, n_c)` (anisotropic speed+angle draw). Read them before
    Task 5.
- **Why (√W, asinh p) coordinates:** ρ̂ ∝ W^{g+3/2} as W→0 (power law — uniform-in-W
  grids under-resolve it; √W stretches the small-W region), and at large p the
  density decays slowly (~p⁻¹ via T(β) ≈ √(π/4β)) — asinh gives log-like resolution
  at large p without a singular node at p=0.
- **Domain coverage (no extrapolation by construction):** a coupled solve needs
  W ∈ [0, max_j(rescale_j)·W0] and p ∈ [0, xi_max / min_j(ra_hat_j)]. Build the
  table on exactly that box (with a small ×1.001 safety factor on W_max); clamp
  query points to the box edges.
- **Differentiability requirement:** gradients must flow through the TABLE BUILD
  (node values are functions of g; query args of (W0, rescale_j, ra_hat_j)), so
  `jax.grad` w.r.t. (W0, g, w_j, ra_hat_j) keeps working end-to-end. Bilinear
  interpolation via `jnp.searchsorted` + manual lerp (or `jnp.interp` composition)
  is differentiable in both the query and the node values.

---

## Tranche A — density table in the coupled Poisson solve (the 86% hotspot)

### Task 1: `AnisoDensityTable` with an accuracy oracle

**Files:**
- Create: `src/progenax/profiles/limepy_tables.py`
- Test: `tests/unit/profiles/test_limepy_tables.py` (new)

**Step 1: Write the failing tests**

```python
# tests/unit/profiles/test_limepy_tables.py
"""AnisoDensityTable: tabulated anisotropic LIMEPY density rho_hat(W, p; g).

The table must reproduce the exact quadrature (_aniso_density_scalar) to the
stated budget across its whole domain, and be differentiable in (g, queries).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


class TestAnisoDensityTable:
    def _table(self, W_max=12.0, p_max=80.0, g=1.0, n_W=512, n_p=96):
        from progenax.profiles.limepy_tables import AnisoDensityTable
        return AnisoDensityTable.build(W_max=W_max, p_max=p_max, g=g,
                                       n_W=n_W, n_p=n_p)

    def test_reproduces_exact_quadrature_on_random_points(self):
        """Max relative error <= 1e-5 against _aniso_density_scalar on 2000
        random interior points (relative to max(exact, 1e-8 * central))."""
        from progenax.profiles.limepy import _aniso_density_scalar

        tab = self._table()
        rng = np.random.default_rng(0)
        W = jnp.asarray(rng.uniform(1e-3 * 12.0, 12.0, 2000))
        p = jnp.asarray(rng.uniform(0.0, 80.0, 2000))
        approx = jax.vmap(tab.evaluate)(W, p)
        exact = jax.vmap(lambda w, pp: _aniso_density_scalar(w, pp, jnp.asarray(1.0)))(W, p)
        central = float(_aniso_density_scalar(jnp.asarray(12.0), jnp.asarray(0.0),
                                              jnp.asarray(1.0)))
        rel = np.asarray(jnp.abs(approx - exact) /
                         jnp.maximum(exact, 1e-8 * central))
        assert rel.max() <= 1e-5, f"max rel err {rel.max():.2e}"

    def test_W_zero_gives_zero_density(self):
        tab = self._table()
        assert float(tab.evaluate(jnp.asarray(0.0), jnp.asarray(3.0))) == 0.0

    def test_clamps_outside_domain(self):
        """Queries past the box edges clamp (no NaN, no extrapolation blow-up)."""
        tab = self._table()
        out = jax.vmap(tab.evaluate)(jnp.array([13.0, 5.0]), jnp.array([3.0, 100.0]))
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_isotropic_p0_matches_closed_form(self):
        """At p=0 the anisotropic density reduces to the isotropic closed form
        limepy_density_hat (an independent oracle, not the quadrature)."""
        from progenax.profiles.limepy import limepy_density_hat

        tab = self._table()
        W = jnp.linspace(0.05, 11.5, 64)
        approx = jax.vmap(lambda w: tab.evaluate(w, jnp.asarray(0.0)))(W)
        exact = limepy_density_hat(W, 1.0)
        np.testing.assert_allclose(np.asarray(approx), np.asarray(exact),
                                   rtol=5e-5, atol=1e-12)

    def test_differentiable_in_g_and_queries(self):
        """AD through the table BUILD (g) and the query (W, p); AD matches FD."""
        from progenax.profiles.limepy_tables import AnisoDensityTable

        def f(g, W, p):
            tab = AnisoDensityTable.build(W_max=10.0, p_max=20.0, g=g,
                                          n_W=128, n_p=32)
            return tab.evaluate(W, p)

        g0, W0_, p0 = 1.0, 4.0, 2.0
        dg = jax.grad(f, 0)(g0, jnp.asarray(W0_), jnp.asarray(p0))
        dW = jax.grad(f, 1)(g0, jnp.asarray(W0_), jnp.asarray(p0))
        dp = jax.grad(f, 2)(g0, jnp.asarray(W0_), jnp.asarray(p0))
        assert all(jnp.isfinite(d) for d in (dg, dW, dp))
        eps = 1e-5
        fd_g = (f(g0 + eps, jnp.asarray(W0_), jnp.asarray(p0))
                - f(g0 - eps, jnp.asarray(W0_), jnp.asarray(p0))) / (2 * eps)
        np.testing.assert_allclose(float(dg), float(fd_g), rtol=1e-4, atol=1e-8)
```

**Step 2: Run, verify they fail with ModuleNotFoundError**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/profiles/test_limepy_tables.py -q`
Expected: 5 failed, `ModuleNotFoundError: No module named 'progenax.profiles.limepy_tables'`

**Step 3: Implement `AnisoDensityTable`**

Create `src/progenax/profiles/limepy_tables.py`:

```python
"""Tabulated LIMEPY DF moments (Phase 1.5 performance layer).

AnisoDensityTable tabulates the anisotropic lowered-isothermal density
rho_hat(W, p; g) -- the 86%% hotspot of the multi-component coupled solve --
on a (sqrt(W), asinh(p)) grid, built in one batched call with the EXACT
quadrature (_aniso_density_scalar). The quadrature path remains available
everywhere as the oracle; this module must reproduce it to <= 1e-5 relative
(asserted in tests/unit/profiles/test_limepy_tables.py).

Coordinates: rho_hat ~ W^(g+3/2) as W->0 (sqrt stretches the power-law region);
rho_hat decays ~ 1/p at large p via T(beta) ~ sqrt(pi/4 beta) (asinh gives
log-like large-p resolution). Bilinear interpolation; queries clamp to the box.
Differentiable in the build inputs (g) and the queries (W, p).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from progenax.profiles.limepy import _aniso_density_scalar


class AnisoDensityTable(eqx.Module):
    """Bilinear table of rho_hat(W, p) for fixed g (one table serves all
    components of a coupled solve: component j queries (rescale_j*psi, xi/ra_j))."""

    s_nodes: Float[Array, "n_W"]   # sqrt(W) nodes, uniform on [0, sqrt(W_max)]
    q_nodes: Float[Array, "n_p"]   # asinh(p) nodes, uniform on [0, asinh(p_max)]
    values: Float[Array, "n_W n_p"]

    @classmethod
    def build(cls, W_max, p_max, g, n_W: int = 512, n_p: int = 96):
        s = jnp.linspace(0.0, jnp.sqrt(jnp.asarray(W_max) * 1.001), n_W)
        q = jnp.linspace(0.0, jnp.arcsinh(jnp.asarray(p_max)), n_p)
        W = s**2
        p = jnp.sinh(q)
        vals = jax.vmap(lambda w: jax.vmap(
            lambda pp: _aniso_density_scalar(w, pp, g))(p))(W)
        return cls(s_nodes=s, q_nodes=q, values=vals)

    def evaluate(self, W, p):
        """Bilinear interp at (W, p), clamped to the table box. W<=0 -> 0."""
        s = jnp.sqrt(jnp.maximum(W, 0.0))
        q = jnp.arcsinh(jnp.maximum(p, 0.0))
        s = jnp.clip(s, self.s_nodes[0], self.s_nodes[-1])
        q = jnp.clip(q, self.q_nodes[0], self.q_nodes[-1])
        i = jnp.clip(jnp.searchsorted(self.s_nodes, s) - 1, 0, self.s_nodes.size - 2)
        j = jnp.clip(jnp.searchsorted(self.q_nodes, q) - 1, 0, self.q_nodes.size - 2)
        ts = (s - self.s_nodes[i]) / (self.s_nodes[i + 1] - self.s_nodes[i])
        tq = (q - self.q_nodes[j]) / (self.q_nodes[j + 1] - self.q_nodes[j])
        v = ((1 - ts) * (1 - tq) * self.values[i, j]
             + ts * (1 - tq) * self.values[i + 1, j]
             + (1 - ts) * tq * self.values[i, j + 1]
             + ts * tq * self.values[i + 1, j + 1])
        return jnp.where(W > 0.0, v, 0.0)


__all__ = ["AnisoDensityTable"]
```

If the 1e-5 budget fails, increase `n_W`/`n_p` (try 768/128) — do NOT relax the
test. Record the final grid size in the commit message.

**Step 4: Run tests, verify all 5 pass**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/profiles/test_limepy_tables.py -q`
Expected: 5 passed

**Step 5: Commit**

```bash
git add src/progenax/profiles/limepy_tables.py tests/unit/profiles/test_limepy_tables.py
git commit -m "feat(profiles): AnisoDensityTable - tabulated rho_hat(W,p;g) with 1e-5 oracle budget (Phase 1.5 Task 1)"
```

---

### Task 2: table-backed anisotropic RHS in `solve_multicomponent_limepy`

**Files:**
- Modify: `src/progenax/profiles/limepy_multimass.py` (the anisotropic branch of
  `solve_multicomponent_limepy`; also re-export nothing new)
- Test: extend `tests/unit/profiles/test_limepy_tables.py`

**Step 1: Write the failing tests**

Append to `tests/unit/profiles/test_limepy_tables.py`:

```python
class TestTableBackedSolver:
    """solve_multicomponent_limepy(aniso_method='table') reproduces the exact
    quadrature solve to the stated budget and is the new default."""

    def _solve(self, method):
        from progenax.profiles.limepy_multimass import solve_multicomponent_limepy
        return solve_multicomponent_limepy(
            jnp.array([0.6, 0.4]), jnp.array([1.0, 1.6]), W0=7.0, g=1.0,
            xi_max=800.0, n_points=2000, ra_hat_j=jnp.array([10.0, 10.0]),
            aniso_method=method)

    def test_table_solve_matches_quadrature_solve(self):
        """|psi_table - psi_quad| <= 1e-4 * W0 everywhere; per-component
        densities match to 2e-4 absolute (normalized units)."""
        xi_q, psi_q, rho_q = self._solve("quadrature")
        xi_t, psi_t, rho_t = self._solve("table")
        np.testing.assert_allclose(np.asarray(xi_t), np.asarray(xi_q), rtol=0)
        assert float(jnp.max(jnp.abs(psi_t - psi_q))) <= 1e-4 * 7.0
        assert float(jnp.max(jnp.abs(rho_t - rho_q))) <= 2e-4

    def test_table_is_default_and_iso_path_untouched(self):
        """Default aniso_method is 'table'; the ISOTROPIC path is bit-identical
        to before (no table involved when ra_hat_j is None)."""
        from progenax.profiles.limepy_multimass import solve_multicomponent_limepy
        xi_d, psi_d, _ = self._solve("table")
        from inspect import signature
        assert signature(solve_multicomponent_limepy).parameters["aniso_method"].default == "table"
        xi_i, psi_i, _ = solve_multicomponent_limepy(
            jnp.array([0.6, 0.4]), jnp.array([1.0, 1.6]), W0=7.0, g=1.0,
            xi_max=300.0, n_points=2000)  # iso: no ra_hat_j
        assert bool(jnp.all(jnp.isfinite(psi_i)))

    def test_table_solve_differentiable_in_rescale_ra(self):
        from progenax.profiles.limepy_multimass import solve_multicomponent_limepy

        def metric(rescale, ra):
            xi, psi, _ = solve_multicomponent_limepy(
                jnp.array([0.5, 0.5]), rescale, 7.0, 1.0, xi_max=800.0,
                n_points=1500, ra_hat_j=ra, aniso_method="table")
            return jnp.mean(psi[:300])

        d_r = jax.grad(metric, 0)(jnp.array([1.0, 1.6]), jnp.array([10.0, 10.0]))
        d_a = jax.grad(metric, 1)(jnp.array([1.0, 1.6]), jnp.array([10.0, 10.0]))
        assert jnp.all(jnp.isfinite(d_r)) and jnp.any(jnp.abs(d_r) > 0)
        assert jnp.all(jnp.isfinite(d_a)) and jnp.any(jnp.abs(d_a) > 0)

    def test_table_solve_is_faster(self):
        """Warm table solve >= 3x faster than warm quadrature solve (the measured
        target is ~5-10x; assert a conservative 3x so the test is not flaky)."""
        import time

        def timed(method):
            self._solve(method)  # warm/compile
            t0 = time.perf_counter()
            out = self._solve(method)
            jax.block_until_ready(out[1])
            return time.perf_counter() - t0

        assert timed("quadrature") / timed("table") >= 3.0
```

**Step 2: Run, verify fail** (unexpected keyword `aniso_method`)

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/profiles/test_limepy_tables.py::TestTableBackedSolver -q`
Expected: 4 failed, `TypeError: ... unexpected keyword argument 'aniso_method'`

**Step 3: Implement**

In `solve_multicomponent_limepy`:
- Add parameter `aniso_method: str = "table"` (document: ignored when
  `ra_hat_j is None`; `"quadrature"` is the exact oracle path).
- In the anisotropic branch, when `aniso_method == "table"`:
  build ONE shared table before the solve —
  `W_max = jnp.max(rescale) * W0`, `p_max = xi_max / jnp.min(ra_j)`,
  `tab = AnisoDensityTable.build(W_max, p_max, g)` — and define
  `density_components(xi, psi)` as
  `tab.evaluate(rescale * psi, xi / ra_j)` vmapped over components, normalized by
  `rho0_j = vmap(lambda res: tab.evaluate(res * W0, 0.0))(rescale)` with the same
  `where(rho0_j > 1e-300, ...)` guard as the quadrature branch.
- Keep the quadrature branch verbatim for `aniso_method == "quadrature"`.
- Raise `ValueError` for any other string (concrete input, plain Python check).
- Keep the non-truncation guard working for both methods.
- Import `AnisoDensityTable` lazily inside the function or at module top — check
  for import cycles (`limepy_tables` imports from `limepy` only, so a top-level
  import in `limepy_multimass` is safe).
- Mind the 100-LOC function limit: if `solve_multicomponent_limepy` exceeds it,
  extract a private `_aniso_density_fn(rescale, ra_j, W0, g, xi_max, method)`
  helper returning the closure.

**Step 4: Run the new tests AND the full multimass solver file**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/profiles/test_limepy_tables.py tests/unit/profiles/test_limepy_multimass.py -q`
Expected: all pass. NOTE: `test_limepy_multimass.py` contains solver-equivalence
tests at tight tolerances (e.g. `test_mass_wrapper_matches_direct_core`,
rtol 1e-11) — these compare table-vs-table (both paths use the default), so they
must still pass. `test_single_component_recovers_single_mass_anisotropic`
compares the multi-component TABLE solve against the single-mass QUADRATURE
solve (`solve_limepy_profile`) at rtol 2e-3 — the 1e-5 table budget fits inside.
If it fails, the table accuracy is insufficient: increase the grid in Task 1,
do not touch the test.

**Step 5: Commit**

```bash
git add src/progenax/profiles/limepy_multimass.py tests/unit/profiles/test_limepy_tables.py
git commit -m "perf(profiles): table-backed anisotropic RHS in solve_multicomponent_limepy (default; quadrature oracle retained)"
```

---

### Task 3: model integration + physics regression at the model level

**Files:**
- Modify: `src/progenax/cluster/multicomponent.py` (constructors pass
  `aniso_method` through; default "table"; `component_virial_ratios` UNCHANGED on
  quadrature)
- Test: extend `tests/unit/cluster/test_multicomponent.py`

**Step 1: Write the failing test**

Append to `tests/unit/cluster/test_multicomponent.py` (class `TestFromComponents`):

```python
    @pytest.mark.slow
    def test_table_model_equilibrium_matches_quadrature_oracle(self):
        """A table-backed anisotropic model still proves Q_j = 0.5 via the
        EXACT-quadrature component_virial_ratios (oracle independence), and its
        mass CDF matches the quadrature-built model to 5e-4."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        kw = dict(alpha_j=jnp.array([0.6, 0.4]), w_j=jnp.array([1.0, 0.79]),
                  m_j=jnp.array([1.0, 4.0]), W0=7.0, g=1.0, r_c=1.0,
                  ra_hat_j=jnp.array([10.0, 10.0]), xi_max=800.0,
                  n_ode_points=3000)
        m_tab = MultiComponentCluster.from_components(**kw)  # default: table
        m_quad = MultiComponentCluster.from_components(**kw, aniso_method="quadrature")
        Qj = np.asarray(m_tab.component_virial_ratios())
        np.testing.assert_allclose(Qj, 0.5, atol=3e-3, err_msg=f"table Q_j={Qj}")
        np.testing.assert_allclose(np.asarray(m_tab._cdf_j),
                                   np.asarray(m_quad._cdf_j), atol=5e-4)
```

**Step 2: Run, verify fail** (`from_components` has no `aniso_method`)

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest "tests/unit/cluster/test_multicomponent.py::TestFromComponents::test_table_model_equilibrium_matches_quadrature_oracle" -q`
Expected: FAIL, `TypeError: ... unexpected keyword argument 'aniso_method'`

**Step 3: Implement**

- `from_components` / `from_mass_segregation` / `from_imf` gain
  `aniso_method: str = "table"`, passed to `solve_multicomponent_limepy` AND
  stored nowhere (it is a construction choice, not model state).
- `__init__`'s `dens(...)` closure (used for the r-grid CDF) must use the SAME
  method as the solve. Simplest correct approach: thread `aniso_method` into
  `__init__` and, for "table", build the same shared table (same W_max/p_max
  formula) for the CDF grid evaluation — or better, reuse `rho_on_xi` forwarding
  (already in place) for the ξ-grid and build one table for the r-grid call.
  Either way `component_virial_ratios` keeps calling the QUADRATURE grid function
  (`_grid_density_components`) — do not change it.
- `from_imf` note: `find_alpha_for_masses` (the eigenvalue iteration) still uses
  the quadrature path internally — leave it; its speedup is Tranche C / deferred.
  [SUPERSEDED at Task 2 (2968fb0): aniso_method threaded through the mass path;
  eigenvalue loop defaults to "table" for self-consistency, gated by an aniso
  realized-mass test]

**Step 4: Run model tests (fast gate for this file) + the slow oracle test**

Run: `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/cluster/test_multicomponent.py -q -n auto`
Expected: all pass (now +1 test).

**Step 5: Commit**

```bash
git add src/progenax/cluster/multicomponent.py tests/unit/cluster/test_multicomponent.py
git commit -m "perf(cluster): MultiComponentCluster on the table-backed solve; Q_j oracle stays exact-quadrature"
```

---

### Task 4: Tranche-A wrap-up — physics suite, benchmark, validation script

**Files:**
- Create: `scripts/validate_df_tables.py`
- No src changes expected.

**Step 1: Run the multimass physics validation + both validate scripts**

Run:
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_multimass_equilibrium_physics.py -q -n auto
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_multimass_anisotropy.py
```
Expected: tests pass; script prints `ANISOTROPIC SAMPLER VALIDATION PASS`.
These exercise the table path end-to-end (β(r) ≡ DF quadrature, global Q = 0.5).
Any failure = table accuracy problem; fix in Task 1 grid, never in tolerances.

**Step 2: Write `scripts/validate_df_tables.py`**

A compact CLI script (follow the style of `scripts/validate_multimass_*.py`,
use `scripts/_plotstyle.py`) that prints a PASS/FAIL table:
(a) table-vs-quadrature max rel density error over the domain (budget 1e-5);
(b) ψ(ξ) max abs deviation table-vs-quadrature solve (budget 1e-4·W0);
(c) warm solve speedup factor (report measured; PASS if ≥ 3×);
(d) AD-vs-FD gradient check through the table solve in (w_j, r_a) (rtol 1e-3).
One 4-panel figure to `validation/plots/df_tables.{png,pdf}`.

**Step 3: Run it**

Run: `env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_df_tables.py`
Expected: exit 0, all PASS lines.

**Step 4: Commit**

```bash
git add scripts/validate_df_tables.py
git commit -m "validation(profiles): DF-table accuracy/speedup/gradient validation script (PASS)"
```

---

## Tranche B — speed-CDF tables in the sampler (the per-star hotspot)

### Task 5: isotropic speed table

**Files:**
- Modify: `src/progenax/profiles/limepy_tables.py` (add `SpeedCDFTable`)
- Modify: `src/progenax/cluster/multicomponent.py` (`_sample_cluster_arrays` iso
  branch)
- Test: extend `tests/unit/profiles/test_limepy_tables.py` + one sampler test in
  `tests/unit/cluster/test_multicomponent.py`

**Step 1: READ `src/progenax/kinematics/limepy_df.py`** — understand
`_sample_unit_speed(key, W, g, n_speed)`: it builds, per star, a u-grid on
[0, √(2W)], weights u²E_γ(g, W−u²/2), forms the CDF, and inverse-CDF draws one u.

**Step 2: Write the failing tests**

```python
class TestSpeedCDFTable:
    def test_sampled_moments_match_exact_sampler(self):
        """Speeds drawn via the (W, u) CDF table reproduce the exact per-star
        sampler's distribution: mean and rms u in 8 W-bins agree to 1%%, and a
        two-sample KS statistic over 20k draws < 0.02."""
        ...  # build table for g=1, W_max=10; draw 20k speeds at random W via both
             # paths (same key strategy not required - compare DISTRIBUTIONS);
             # use scipy-free KS: max |ECDF1 - ECDF2| on a merged grid (numpy in
             # tests is allowed).

    def test_differentiable_through_table_draw(self):
        """grad of mean sampled speed w.r.t. a velocity-scale multiplier is
        finite and nonzero through the table draw."""
        ...
```

(Write these concretely — the subagent fills in the bodies following the patterns
of Task 1; the acceptance numbers above are the contract.)

**Step 3–4: Implement `SpeedCDFTable` + wire into the ISO branch of
`_sample_cluster_arrays`**

- Table: u-CDFs on a √W-grid (n_W=256) × u-grid (n_u=256 per W, on [0, √(2W)]
  in NORMALIZED u/√(2W) ∈ [0,1] coordinates so one rectangular grid serves all W).
  Per-star draw: locate W bin, lerp the two neighboring inverse-CDF rows at the
  star's uniform variate (2 × `jnp.interp` + one lerp).
  Build at model construction ONLY when needed → build lazily inside
  `_sample_cluster_arrays`? NO — jit. Build in `MultiComponentCluster.__init__`
  and store as fields (`_speed_table_*`), or build inside the jitted core (it
  will be constant-folded per model — acceptable; measure). Choose the simpler:
  build inside `_sample_cluster_arrays` from model fields (W_max = max(rescale)
  ·W0), so the model PyTree is unchanged.
- The sampled VALUES change (different draw path) — statistical tests only; do
  NOT compare to old golden samples.
- The dispersion-vs-analytic and equilibrium tests in
  `tests/unit/cluster/test_multicomponent.py` + `tests/validation/...` are the
  physics gate: run them.

**Step 5: Benchmark + commit**

Run the N-sweep (1k/10k/100k iso) with the bench pattern from
`/tmp/progenax_bench/scaling_study.py`; require ≥ 3× at N=100k (target ~5–10×).

```bash
git commit -m "perf(cluster): isotropic speed draws via (W,u) CDF table (statistical oracles pass; Nx speedup at 1e5 stars)"
```

### Task 6: anisotropic speed-angle table

Same pattern as Task 5 for `_sample_speed_angle`: tabulate the speed MARGINAL
CDF on a (√W, asinh p, u/√(2W)) grid (256×48×192 — measure build cost; it must
amortize within one 20k-star draw) and keep the angular conditional
(cos θ | u, p) EXACT (it is cheap: exp arithmetic, no special functions).
Acceptance: the `test_aniso_sampled_beta_matches_analytic` and
`test_aniso_global_virial_is_half` tests pass unchanged; β(r)-vs-DF deviation
budget unchanged (<0.06); ≥ 3× at N=100k aniso.

Commit: `perf(cluster): anisotropic speed marginal via 3-D CDF table; exact angular conditional retained`

---

## Task 7: close-out

1. Full released-core gate:
   `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto`
   Expected: all pass (count will be ≈ 1065 + new table tests). Record the count.
2. Re-run `/tmp/progenax_bench/scaling_study.py` sections; record before/after
   table in the completion doc.
3. Write `.claude-work/TASK_1.5_DF_TABLES_COMPLETE.md` (implementation summary,
   measured speedups, accuracy evidence, lessons).
4. Update `STATUS.md` `next:` line (Phase 1.5 done → resume Phase 1c) and the
   CLAUDE.md suite timing if it changed materially.
5. Commit: `docs(status): Phase 1.5 DF tables done; resume at Phase 1c`.
   DO NOT PUSH.

## Honest scope / risks

- The table changes anisotropic RESULTS within the stated budgets (1e-5 density,
  1e-4·W0 potential) — never silently: the quadrature oracle stays selectable and
  every budget is asserted in tests.
- `find_alpha_for_masses` (from_imf eigenvalue loop) stays on quadrature in this
  phase (Tranche C if ever needed).
- `component_virial_ratios` and the single-mass `solve_limepy_profile` /
  `LIMEPYProfile` are OUT OF SCOPE (oracles).
- If the Task 5/6 statistical gates prove flaky, stop Tranche B, keep Tranche A
  (the dominant win), and report — do not loosen gates.

## CURRENT STATE (post-Task-1 review)

Task 1 landed with several deviations from the plan's reference snippets —
Tasks 2–6 implementers should copy patterns from
`src/progenax/profiles/limepy_tables.py`, **not** from the plan's snippets:

- **Cubic Lagrange interpolation** (tensor-product 4-point, O(h^4)), not
  bilinear — bilinear measured O(h^2) with a large constant and cannot reach
  the 1e-5 budget even at 3072x512 nodes.
- **`jax.lax.map` build** over sqrt(W) rows — the double-vmap build OOMs
  (materializes the full (n_W, n_p, n_quad) quadrature intermediate).
- **sqrt(2*pi) normalization fix** to the p=0 closed-form test (the isotropic
  oracle is sqrt(2 pi) * limepy_density_hat).
- **W<=0 gradient guard**: `s = jnp.sqrt(jnp.maximum(W, 1e-12))` mirroring
  `_aniso_density_scalar` — sqrt-at-0 has a NaN cotangent under jax.grad and
  the final where() masks only the primal.
- **>=0 clamp**: `v = jnp.maximum(v, 0.0)` — cubic overshoot can return tiny
  negative densities (O(1e-15·central)) in the W->0 corner; rho_hat >= 0 is a
  contract (Tranche B builds CDFs from it).

### Tasks 2–4 landing notes (2026-06-09)

- **Task 2 (18b82a0 + review fix 2968fb0):** `aniso_method="table"` is the
  default in `solve_multicomponent_limepy`; the table branch lives in
  `_aniso_density_fn` and returns an **eqx.Module RHS (`_TableRHS`)** so
  diffrax's internal jit treats table/params as dynamic leaves (a fresh-closure
  RHS re-compiles every solve, ~0.4 s). The SOLVER table grid is 160×40
  (`_TAB_N_W`/`_TAB_N_P`), sized to the solve-level budget (measured max|dpsi|
  8.9e-5 vs 7e-4; warm solve 0.15 s vs 0.78 s quadrature). The review fix also
  threaded `aniso_method` through the mass path (`solve_multimass_limepy`,
  `_realized_fractions`, `find_alpha_for_masses` — the eigenvalue loop defaults
  to "table" for self-consistency with the model actually built).
- **Task 3 (173d2ab):** constructors gained `aniso_method` (a construction
  choice, not model state); the r-grid mass CDF uses the SAME method as the
  solve via a `dens_fn` forwarded into `__init__`; `component_virial_ratios`
  stays on exact quadrature (the equilibrium oracle).
- **Task 4 PART 1 dedup:** Task 3 originally rebuilt a bit-identical table for
  the CDF r-grid (measured ~0.4 s pure waste per anisotropic construction at
  the default n_ode_points=2000). Now `_solver_table(...)` in
  `limepy_multimass.py` is the single source of the box formula
  (W_max = max(rescale)·W0, p_max = max(xi_max/min(ra_hat_j), 1e-3));
  `MultiComponentCluster._shared_table_and_dens_fn` builds the table ONCE and
  passes it to BOTH the solve (`solve_multicomponent_limepy(...,
  aniso_table=tab)`) and the CDF density source
  (`_aniso_density_fn(..., table=tab)`). Direct solve calls without a table
  build-if-None (bit-identical, asserted by
  `test_explicit_table_matches_internal_build`); an unknown `aniso_method`
  raises instead of silently falling back to quadrature. Measured warm aniso
  construction (n_ode=1000, n_grid=500): 308 ms → 177 ms, bit-identical
  psi_grid/_cdf_j to the pre-refactor 173d2ab values.
