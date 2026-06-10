# Batch A: Memory Fixes + Consolidation + Docs Close-Out — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the two measured OOM drivers (O(N²) eager virial kernels; eager per-star
anisotropic DF quadrature), consolidate the four code-review duplication findings, and
close out the cluster-arc docs (science-capabilities page, 1101→1114, stale-limitation
removal).

**Architecture:** Blocked row-scan pairwise kernels (`lax.scan` over row blocks, vmap-free
accumulator — the fluxax `_accumulate_stars_chunked` pattern, gravax static-chunk
convention) replace the dense `(N,N,3)` kernels in `dynamics/virial.py`. The standalone
velocity DFs (`LIMEPYVelocityDF`, `MichieVelocityDF`, `KingVelocityDF`) route speed draws
through the already-validated `SpeedCDFTable`/`AnisoSpeedCDFTable` (the
`MultiComponentCluster` pattern), keeping the exact quadrature as a selectable oracle
(`speed_method="quadrature"`), mirroring the existing `aniso_method` precedent. A new
`progenax.numerics` module hosts ONE `cumulative_trapezoid` + ONE `inverse_cdf_draw`,
consolidating both the cumtrap×5 and speed-kernel×8 review findings. Engine A state is
grouped into `_EngineAState` (killing the NaN-sentinel union).

**Tech stack:** JAX (`lax.scan`, `vmap`), Equinox modules, pytest, myst (docs build).

**Measured baseline (2026-06-10 probe, 68.7 GB machine, peak RSS):**

| Stage | N | Peak |
|---|---|---|
| `compute_potential_energy` | 8,000 | 6.57 GB |
| `compute_potential_energy` | 20,000 | 32.77 GB (→ ~73 GB at N=30k = OOM) |
| standalone aniso `LIMEPYVelocityDF` | 20,000 | 10.87 GB |
| `MultiComponentCluster` samplers (iso/aniso/B) | 100,000 | 1.3–2.5 GB (fine) |

**Verification gates (run from repo root):**

```bash
# FAST GATE (inner loop, ~4 min)
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow" -n auto
# FULL GATE (per task-group commit, ~6 min) — drop the -m filter
```

**Git:** work on branch `fix/memory-and-consolidation` off `main` (`git checkout -b
fix/memory-and-consolidation`). Commit per task. NO push, NO merge to main without
Anna's explicit go (HITL).

**Hard rules (CLAUDE.md):** JAX-native only; no test weakening; every kernel change
proven against the existing implementation's numbers BEFORE replacing it; functions
≤100 LOC; files ≤500 LOC.

---

## Task 1: `progenax.numerics` — shared `cumulative_trapezoid` + `inverse_cdf_draw`

These two helpers consolidate BOTH review findings (cumtrap ×5, speed-CDF kernel ×8).
This task only CREATES the module with TDD; call-site migration is Tasks 8–9 (so the
memory fixes land first and each migration diffs against a green baseline).

**Files:**
- Create: `src/progenax/numerics.py`
- Test: `tests/unit/test_numerics.py` (new)

**Step 1: Write the failing tests**

```python
"""Tests for progenax.numerics — the shared trapezoid/inverse-CDF primitives.

These helpers must be BIT-IDENTICAL to the inline patterns they replace
(same op order: pairwise average -> cumsum -> concat zero), because five
Poisson passes and eight speed-CDF kernels migrate onto them.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.numerics import cumulative_trapezoid, inverse_cdf_draw


class TestCumulativeTrapezoid:
    def test_matches_inline_pattern_bit_identical(self):
        """Same ops as the inline kernel => bit-identical, not just close."""
        y = jnp.sin(jnp.linspace(0.0, 3.0, 257)) + 1.1
        dr = 3.0 / 256
        inline = jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(0.5 * (y[1:] + y[:-1]) * dr)])
        ours = cumulative_trapezoid(y, dx=dr)
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(inline))

    def test_linear_function_exact(self):
        """Trapezoid is exact for linear integrands: int_0^x t dt = x^2/2."""
        x = jnp.linspace(0.0, 2.0, 101)
        out = cumulative_trapezoid(x, dx=float(x[1] - x[0]))
        np.testing.assert_allclose(np.asarray(out), np.asarray(x**2 / 2),
                                   rtol=0, atol=1e-14)

    def test_axis_minus_one_on_2d_rows(self):
        """Row-wise (the SpeedCDFTable / multicomponent axis=1 pattern)."""
        y = jnp.arange(12.0).reshape(3, 4) + 1.0
        dr = 0.5
        inline = jnp.concatenate(
            [jnp.zeros((3, 1)),
             jnp.cumsum(0.5 * (y[:, 1:] + y[:, :-1]), axis=1) * dr], axis=1)
        ours = cumulative_trapezoid(y, dx=dr, axis=-1)
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(inline))

    def test_differentiable(self):
        g = jax.grad(lambda a: cumulative_trapezoid(a * jnp.ones(8), dx=0.1)[-1])(2.0)
        assert jnp.isfinite(g) and g > 0


class TestInverseCdfDraw:
    def test_matches_king_sample_unit_speed_pattern(self):
        """Reproduces the king_df._sample_unit_speed CDF+interp chain exactly."""
        W = 4.0
        n_u = 256
        u_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * W), n_u)
        wgt = jnp.maximum(u_grid**2 * (jnp.exp(W - u_grid**2 / 2.0) - 1.0), 0.0)
        du = u_grid[1] - u_grid[0]
        cdf = jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(0.5 * (wgt[1:] + wgt[:-1])) * du])
        cdf = cdf / (cdf[-1] + 1e-30)
        unif = jnp.asarray(0.37)
        expected = jnp.interp(unif, cdf, u_grid)
        ours = inverse_cdf_draw(wgt, u_grid, unif)
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(expected))

    def test_uniform_weight_is_identity(self):
        """Flat weight => draw is linear in unif (CDF of uniform)."""
        grid = jnp.linspace(0.0, 1.0, 200)
        out = inverse_cdf_draw(jnp.ones(200), grid, jnp.asarray(0.25))
        np.testing.assert_allclose(float(out), 0.25, atol=1e-3)

    def test_differentiable_in_weight_parameter(self):
        def f(scale):
            grid = jnp.linspace(0.0, 1.0, 64)
            return inverse_cdf_draw(jnp.exp(-scale * grid), grid, jnp.asarray(0.5))
        g = jax.grad(f)(1.0)
        assert jnp.isfinite(g)
```

**Step 2: Run the tests — verify they FAIL**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_numerics.py -q
```
Expected: `ModuleNotFoundError: No module named 'progenax.numerics'`

**Step 3: Implement `src/progenax/numerics.py`**

```python
"""Shared numerical primitives (single source of truth).

Consolidates two patterns the 2026-06-10 code review found duplicated:
the cumulative-trapezoid pass (5 inline copies: Poisson integrations in
``profiles/density_poisson.py``, ``profiles/api.py``, ``kinematics/eff_df.py``,
``cluster/multicomponent.py`` x2) and the trapezoid-CDF inverse-draw kernel
(8 speed/angle sampling sites). The op order (pairwise average -> cumsum ->
leading zero) is EXACTLY the inline pattern, so migrated call sites are
bit-identical. Fully differentiable; no data-dependent shapes.
"""
import jax.numpy as jnp
from jaxtyping import Array, Float


def cumulative_trapezoid(
    y: Float[Array, "... n"],
    dx: float,
    axis: int = -1,
) -> Float[Array, "... n"]:
    """Cumulative trapezoid integral with a leading zero, uniform spacing.

    out[..., k] = sum_{i<k} 0.5 * (y[..., i] + y[..., i+1]) * dx, out[..., 0] = 0.
    Same length as ``y`` along ``axis``.
    """
    y = jnp.moveaxis(y, axis, -1)
    inner = jnp.cumsum(0.5 * (y[..., 1:] + y[..., :-1]), axis=-1) * dx
    zero = jnp.zeros(y.shape[:-1] + (1,), dtype=inner.dtype)
    return jnp.moveaxis(jnp.concatenate([zero, inner], axis=-1), -1, axis)


def inverse_cdf_draw(
    weight: Float[Array, "n"],
    grid: Float[Array, "n"],
    unif: Float[Array, ""],
    reg: float = 1e-30,
) -> Float[Array, ""]:
    """Differentiable inverse-CDF draw from an unnormalized weight on a uniform grid.

    Builds the trapezoid CDF of ``weight`` over ``grid`` (uniform spacing inferred
    from the first cell), normalizes with the ``+reg`` guard the sampling kernels
    use (a zero total weight then draws grid[0] instead of NaN), and interpolates
    the quantile. Scalar draw; ``jax.vmap`` over stars.
    """
    dx = grid[1] - grid[0]
    cdf = cumulative_trapezoid(weight, dx=dx)
    cdf = cdf / (cdf[-1] + reg)
    return jnp.interp(unif, cdf, grid)
```

Note for the engineer: `dx` enters `cumulative_trapezoid` *inside* vs *outside* the
cumsum differs between inline sites (`cumsum(... * dr)` vs `cumsum(...) * du`) — for a
scalar uniform `dx` these are bit-identical in IEEE float64 only when `dx` multiplies
each term the same way. The implementation above multiplies AFTER the cumsum
(`cumsum(pairs) * dx`), matching the majority pattern (king_df, limepy_df, tables,
multicomponent). The bit-identity test for the `density_poisson` pattern
(`cumsum(pairs * dr)`) is checked at migration time (Task 8 Step 2) — if it differs
beyond an ulp, keep that site's order by passing pre-multiplied weights and `dx=1.0`.

**Step 4: Run the tests — verify they PASS**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_numerics.py -q
```
Expected: all pass.

**Step 5: Commit**

```bash
git add src/progenax/numerics.py tests/unit/test_numerics.py
git commit -m "feat(numerics): shared cumulative_trapezoid + inverse_cdf_draw (consolidation target for cumtrap x5 + speed-CDF x8)"
```

---

## Task 2: Blocked `compute_potential_energy` (the OOM driver)

Replace the dense `(N,N,3)` kernel with a `lax.scan` over row blocks. Peak transient
drops from `24·N²` bytes to `24·block·N` (block=256, N=20k: 123 MB vs 32.8 GB measured).
Physics identical; float64 summation order changes only across blocks (existing
tolerances: `atol=1e-12` at N=2 — single block, exact; `rtol=1e-6` Clausius at N=300).

**Files:**
- Modify: `src/progenax/dynamics/virial.py:24-49` (`compute_potential_energy`)
- Test: `tests/unit/dynamics/test_virial.py` (add class), existing tests are the regression net

**Step 1: Write the failing tests** (append to `tests/unit/dynamics/test_virial.py`,
matching its house style):

```python
class TestBlockedPotentialEnergy:
    """The blocked row-scan kernel must reproduce the dense pair sum exactly
    (same pair set, same per-pair arithmetic) at every N/block alignment."""

    def _dense_reference(self, positions, masses, G, softening=0.0):
        """Independent O(N^2) oracle: explicit double loop over i<j in numpy."""
        pos = np.asarray(positions); m = np.asarray(masses)
        V = 0.0
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                r = np.sqrt(np.sum((pos[i] - pos[j]) ** 2) + softening**2)
                V -= m[i] * m[j] / r
        return float(G) * V if False else float(G * V)  # noqa: keep -G*sum form

    def test_matches_dense_oracle_block_not_dividing_n(self):
        """N=37 with block_size=16 (padding path) vs the explicit pair sum."""
        key = jax.random.PRNGKey(3)
        pos = jax.random.normal(key, (37, 3))
        m = jnp.abs(jax.random.normal(jax.random.PRNGKey(4), (37,))) + 0.1
        V = compute_potential_energy(pos, m, G=2.5, block_size=16)
        V_ref = self._dense_reference(pos, m, G=2.5)
        np.testing.assert_allclose(float(V), V_ref, rtol=1e-13)

    def test_block_size_invariance(self):
        """Result independent of block size to float64 reassociation level."""
        key = jax.random.PRNGKey(5)
        pos = jax.random.normal(key, (100, 3))
        m = jnp.ones(100)
        vals = [float(compute_potential_energy(pos, m, G=1.0, block_size=b))
                for b in (7, 32, 100, 1024)]
        np.testing.assert_allclose(vals, vals[0], rtol=1e-13)

    def test_softened_matches_dense_oracle(self):
        key = jax.random.PRNGKey(6)
        pos = jax.random.normal(key, (25, 3))
        m = jnp.ones(25)
        V = compute_potential_energy(pos, m, G=1.0, softening=0.05, block_size=8)
        V_ref = self._dense_reference(pos, m, G=1.0, softening=0.05)
        np.testing.assert_allclose(float(V), V_ref, rtol=1e-13)

    def test_grad_finite_at_zero_softening_with_padding(self):
        """The where-before-sqrt guard must survive blocking + padding
        (a padded zero-row coincident with a real star at the origin is the
        gradient trap)."""
        pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        m = jnp.ones(3)
        g = jax.grad(lambda p: compute_potential_energy(p, m, G=1.0,
                                                        block_size=2))(pos)
        assert bool(jnp.all(jnp.isfinite(g)))

    def test_memory_bounded_smoke_n20000(self):
        """N=20k ran at 32.8 GB peak dense (measured 2026-06-10); blocked must
        run comfortably — correctness assert only, the RSS evidence lives in
        scripts/profile_cluster_memory.py."""
        key = jax.random.PRNGKey(7)
        pos = jax.random.normal(key, (20000, 3)) * 5.0
        m = jnp.ones(20000)
        V = compute_potential_energy(pos, m, G=1.0)
        assert bool(jnp.isfinite(V)) and float(V) < 0
```

(Imports at top of file already include `jax`, `jnp`, `np`,
`compute_potential_energy` — verify, add if missing.)

**Step 2: Run — verify the new tests FAIL**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/dynamics/test_virial.py -q
```
Expected: `TypeError: compute_potential_energy() got an unexpected keyword argument
'block_size'` (and the N=20k test would OOM-risk under the dense kernel — it is the
last test so the TypeErrors surface first).

**Step 3: Implement the blocked kernel** (replace `compute_potential_energy` in
`src/progenax/dynamics/virial.py`; keep the module docstring and all other functions):

```python
def _pad_rows(arr, block):
    """Pad axis 0 to a multiple of ``block`` with zeros (static shapes)."""
    pad = (-arr.shape[0]) % block
    if pad == 0:
        return arr
    widths = ((0, pad),) + ((0, 0),) * (arr.ndim - 1)
    return jnp.pad(arr, widths)


def compute_potential_energy(
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    softening: float = 0.0,
    block_size: int = 256,
) -> Float[Array, ""]:
    """Total potential energy V = -G * sum_{i<j} m_i m_j / r_ij (Plummer-softened).

    Blocked row-scan (``lax.scan`` over row blocks of ``block_size`` stars vs ALL
    columns): peak transient memory is O(block_size * N), not O(N^2) — the dense
    kernel measured 32.8 GB at N = 2e4 (2026-06-10); blocked at the default 256
    it is ~0.12 GB. Identical pair set and per-pair arithmetic; only float64
    summation ORDER changes across blocks (re-association at the 1e-15 relative
    level). ``block_size`` is a Python int and must be static under jax.jit.

    Differentiable at ``softening=0``: the i<j mask feeds excluded entries
    (diagonal, lower triangle, padded rows) a safe value *before* ``sqrt`` so no
    masked-out ``sqrt(0)`` cotangent can NaN-poison the gradient. This is the
    single canonical energy implementation; ``progenax.builders`` re-exports it.
    """
    N = positions.shape[0]
    block = int(min(block_size, N))
    pos_b = _pad_rows(positions, block).reshape(-1, block, 3)
    m_b = _pad_rows(masses, block).reshape(-1, block)
    idx_b = jnp.arange(pos_b.shape[0] * block).reshape(-1, block)
    col = jnp.arange(N)

    def body(acc, blk):
        pb, mb, ib = blk
        diff = pb[:, None, :] - positions[None, :, :]          # (block, N, 3)
        r2 = jnp.sum(diff**2, axis=2)                          # (block, N)
        upper = ib[:, None] < col[None, :]                     # i<j; padded rows all-False
        r_soft = jnp.sqrt(jnp.where(upper, r2 + softening**2, 1.0))
        pair = jnp.where(upper, (mb[:, None] * masses[None, :]) / r_soft, 0.0)
        return acc + jnp.sum(pair), None

    V, _ = jax.lax.scan(body, jnp.zeros((), dtype=positions.dtype),
                        (pos_b, m_b, idx_b))
    return -G * V
```

Notes for the engineer:
- `upper` already excludes the diagonal (i<j is false at i==j) AND every padded row
  (its global index ≥ N can never be < a column index < N), so ONE mask replaces the
  old double-`where` — and it still sits BEFORE the `sqrt` (the gradient guard).
- Do NOT use the dot-product identity (`|x_i|²+|x_j|²−2x_i·x_j`): it suffers
  catastrophic cancellation for close pairs, which dominate 1/r. The difference-based
  form is what the dense kernel used; correctness > the extra 3× memory.
- Coincident real pairs (r=0 at softening=0) were ±inf in the dense kernel too —
  behavior unchanged.

**Step 4: Run the dynamics tests + fast gate**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/dynamics/ tests/unit/test_numerics.py -q
```
Expected: all pass, including the pre-existing `test_two_body_analytic` (atol=1e-12),
`test_grad_fd_match_softened`, and `test_energy_consolidation.py`. Then:

```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow" -n auto
```
Expected: full pass (callers: `builders.virial_scale`, `compute_virial_ratio`,
`binaries/diagnostics.binary_energy_budget`, validation suites — all tolerance-based).

**Step 5: Commit**

```bash
git add src/progenax/dynamics/virial.py tests/unit/dynamics/test_virial.py
git commit -m "perf(dynamics)!: blocked row-scan potential energy - O(block*N) memory (32.8 GB -> ~0.1 GB at N=2e4)"
```

---

## Task 3: Blocked `_accelerations` (per-group virial path)

Same pattern for the second `(N,N,3)` kernel. `per_group_virial_ratio` (used 20× per
validation run) inherits the fix.

> **Amendment (2026-06-10, sanctioned by Task 2 review):** both blocked kernels hoist
> the per-block computation into a `@jax.checkpoint` inner function — without it the
> scan vjp stacks per-block residuals and the backward pass is O(N²) again — and add
> an `N == 0` guard (`(-0) % 0` would raise). Applied in commits `bd758ae` (Task 2)
> and `60622d3` (Task 3); gradient parity vs dense verified to ≤2e-14.

**Files:**
- Modify: `src/progenax/dynamics/virial.py:100-117` (`_accelerations`)
- Test: `tests/unit/dynamics/test_group_virial.py` (add class)

**Step 1: Write the failing tests** (append to `tests/unit/dynamics/test_group_virial.py`):

```python
class TestBlockedAccelerations:
    def test_matches_dense_oracle_with_padding(self):
        """N=37, block=16 vs an explicit numpy double loop."""
        from progenax.dynamics.virial import _accelerations
        key = jax.random.PRNGKey(11)
        pos = jax.random.normal(key, (37, 3))
        m = jnp.abs(jax.random.normal(jax.random.PRNGKey(12), (37,))) + 0.1
        a = np.asarray(_accelerations(pos, m, G=1.7, block_size=16))
        p = np.asarray(pos); mm = np.asarray(m)
        a_ref = np.zeros_like(p)
        for i in range(37):
            for k in range(37):
                if i == k:
                    continue
                d = p[i] - p[k]
                a_ref[i] -= 1.7 * mm[k] * d / np.sum(d**2) ** 1.5
        np.testing.assert_allclose(a, a_ref, rtol=1e-12, atol=1e-13)

    def test_clausius_identity_survives_blocking(self):
        """sum_i m_i r_i . a_i == V — ties Task 2 and Task 3 together and is the
        physics contract per_group_virial_ratio depends on (existing
        test_single_group_reproduces_global_virial re-checks at N=300)."""
        key = jax.random.PRNGKey(13)
        pos = jax.random.normal(key, (123, 3))
        m = jnp.ones(123)
        from progenax.dynamics.virial import _accelerations
        a = _accelerations(pos, m, G=1.0, block_size=32)
        W = float(jnp.sum(m * jnp.sum(pos * a, axis=1)))
        V = float(compute_potential_energy(pos, m, G=1.0, block_size=32))
        np.testing.assert_allclose(W, V, rtol=1e-10)

    def test_grad_finite_at_zero_softening(self):
        from progenax.dynamics.virial import _accelerations
        pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.5, 0.0]])
        m = jnp.ones(3)
        g = jax.grad(lambda p: jnp.sum(_accelerations(p, m, G=1.0,
                                                      block_size=2) ** 2))(pos)
        assert bool(jnp.all(jnp.isfinite(g)))
```

**Step 2: Run — verify FAIL** (`block_size` TypeError):

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/dynamics/test_group_virial.py -q
```

**Step 3: Implement** (replace `_accelerations` in `virial.py`):

```python
def _accelerations(
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    softening: float = 0.0,
    block_size: int = 256,
) -> Float[Array, "N 3"]:
    """Direct-summation accelerations a_i = -G sum_k m_k (r_i - r_k) / |r_ik|^3.

    Blocked row-scan: O(block_size * N) transient memory (see
    compute_potential_energy). Plummer-softened; differentiable at softening=0
    (the interaction mask feeds excluded entries a safe value before the
    inverse-cube — diagonal AND padded rows, so no masked inf can NaN-poison
    the vjp through the discarded pad slice).
    """
    N = positions.shape[0]
    block = int(min(block_size, N))
    pos_b = _pad_rows(positions, block).reshape(-1, block, 3)
    idx_b = jnp.arange(pos_b.shape[0] * block).reshape(-1, block)
    col = jnp.arange(N)

    def body(_, blk):
        pb, ib = blk
        diff = pb[:, None, :] - positions[None, :, :]            # (block, N, 3)
        r2 = jnp.sum(diff**2, axis=2)
        interact = (ib[:, None] != col[None, :]) & (ib[:, None] < N)
        inv_r3 = jnp.where(interact, 1.0, 0.0) * jnp.where(
            interact, r2 + softening**2, 1.0) ** (-1.5)
        a_blk = -G * jnp.sum(masses[None, :, None] * diff * inv_r3[:, :, None],
                             axis=1)
        return None, a_blk

    _, a = jax.lax.scan(body, None, (pos_b, idx_b))
    return a.reshape(-1, 3)[:N]
```

(The `(ib < N)` term zeroes entire padded rows so a padded origin-row coincident with
a real star cannot create `inf` that NaN-poisons the vjp of the discarded slice.)

**Step 4: Run dynamics tests + fast gate** — all pass, especially
`test_single_group_reproduces_global_virial` (rtol=1e-6, the Clausius contract) and
the multimass validation suite (`tests/validation/test_multimass_equilibrium_physics.py`).

**Step 5: Commit**

```bash
git add src/progenax/dynamics/virial.py tests/unit/dynamics/test_group_virial.py
git commit -m "perf(dynamics): blocked row-scan accelerations - per_group_virial_ratio now O(block*N) memory"
```

---

## Task 4: Repeatable memory-evidence script

Turn the ad-hoc probe into a permanent validation artifact with pass/fail criteria
(definition-of-complete requires quantitative evidence).

**Files:**
- Create: `scripts/profile_cluster_memory.py`

**Step 1: Implement** — port `/tmp/progenax_mem_probe.py` (subprocess-per-stage,
`resource.ru_maxrss`, Darwin=bytes / Linux=KB guard) with these stages and gates:

| Stage | N | PASS gate (peak RSS) |
|---|---|---|
| import progenax | — | < 1 GB |
| Engine A iso `sample_cluster` | 1e5 | < 3 GB |
| Engine A aniso `sample_cluster` (`ra_hat_j=[10,10]`) | 1e5 | < 4 GB |
| Engine B halo+core (`Plummer(2.0)+EFF(0.8,γ5,rt9)`, fracs `[0.6,0.4]`) | 1e5 | < 4 GB |
| `compute_potential_energy` | 20,000 | **< 2 GB** (was 32.77 GB) |
| `per_group_virial_ratio` (4 groups) | 20,000 | < 2 GB |
| standalone aniso `LIMEPYVelocityDF` (`r_a=10`) | 20,000 | < 3 GB after Task 5 (run with `--allow-fail` until then) |

Print an expected-vs-measured table and exit nonzero on gate failure. Keep it
standalone-CLI in the house style of `scripts/validate_*.py`.

**Step 2: Run**

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/profile_cluster_memory.py --allow-fail limepy_df_aniso
```
Expected: virial stages now PASS far under their gates; record the printed table in
the commit message.

**Step 3: Commit**

```bash
git add scripts/profile_cluster_memory.py
git commit -m "validation(memory): staged peak-RSS profile script with pass gates (virial N=2e4: 32.8 GB -> measured <2 GB)"
```

---

## Task 5: Route `LIMEPYVelocityDF` through the speed-CDF tables

The standalone aniso path measured 10.87 GB at N=2e4 (eager vmap of a 256-point grid ×
91-term Poisson sum per star). The cure already exists and is oracle-validated:
`AnisoSpeedCDFTable` / `SpeedCDFTable` (used by `MultiComponentCluster`). Mirror
`cluster/sampling.py:69-108` exactly; keep the quadrature as the selectable oracle.

**Files:**
- Modify: `src/progenax/kinematics/limepy_df.py` (class fields + `sample_velocities`)
- Test: `tests/unit/kinematics/test_limepy_df.py` (add class)

**Step 1: Write the failing tests:**

```python
class TestLimepyTableRouting:
    """speed_method='table' (default) must agree with the exact quadrature
    oracle (speed_method='quadrature') distributionally and in moments —
    the same contract AnisoSpeedCDFTable passed against the DF quadrature
    (tests/unit/profiles/test_limepy_tables.py: moments to 1.5%)."""

    def _two_dfs(self, r_a):
        kw = dict(W0=5.0, g=1.0, r_c=1.0, r_a=r_a)
        return (LIMEPYVelocityDF(**kw),                       # default: table
                LIMEPYVelocityDF(**kw, speed_method="quadrature"))

    def _speeds(self, df, n=30000, seed=0):
        prof = LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0)
        masses = jnp.ones(n)
        pos = prof.sample_positions(masses, jax.random.PRNGKey(seed))
        vel = df.sample_velocities(pos, masses, jax.random.PRNGKey(seed + 1),
                                   G=1.0)
        return np.asarray(jnp.linalg.norm(vel, axis=1))

    @pytest.mark.parametrize("r_a", [None, 4.0])
    def test_speed_moments_match_quadrature_oracle(self, r_a):
        df_t, df_q = self._two_dfs(r_a)
        s_t, s_q = self._speeds(df_t), self._speeds(df_q)
        assert abs(s_t.mean() / s_q.mean() - 1.0) < 0.02
        assert abs((s_t**2).mean() / (s_q**2).mean() - 1.0) < 0.03

    @pytest.mark.parametrize("r_a", [None, 4.0])
    def test_speed_distribution_ks(self, r_a):
        from scipy.stats import ks_2samp
        df_t, df_q = self._two_dfs(r_a)
        D = ks_2samp(self._speeds(df_t), self._speeds(df_q)).statistic
        assert D < 0.02

    def test_aniso_beta_profile_preserved(self):
        """The table path must keep the validated beta(r): the angular
        conditional stays EXACT, so only the speed marginal changed."""
        df_t, df_q = self._two_dfs(4.0)
        # reuse the existing _beta_profile helper of this test file
        beta_t = _beta_profile(df_t, n=60000, seed=2)
        beta_q = _beta_profile(df_q, n=60000, seed=2)
        np.testing.assert_allclose(beta_t, beta_q, atol=0.06)

    def test_table_default_and_quadrature_static(self):
        df = LIMEPYVelocityDF(W0=5.0, g=1.0, r_c=1.0)
        assert df.speed_method == "table"

    def test_differentiable_in_g_through_table(self):
        def mean_ke(g):
            df = LIMEPYVelocityDF(W0=5.0, g=g, r_c=1.0, r_a=4.0)
            prof_pos = jnp.array([[0.5, 0.0, 0.0]] * 64)
            v = df.sample_velocities(prof_pos, jnp.ones(64),
                                     jax.random.PRNGKey(0), G=1.0)
            return jnp.mean(jnp.sum(v**2, axis=1))
        g = jax.grad(mean_ke)(1.0)
        assert jnp.isfinite(g) and g != 0.0
```

(If `_beta_profile` does not exist as a helper, factor it out of
`test_radial_anisotropy_increases_outward` first — do not duplicate it.)

**Step 2: Run — verify FAIL** (`speed_method` unexpected kwarg):

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/kinematics/test_limepy_df.py -q
```

**Step 3: Implement.** In `limepy_df.py`:

1. Import the tables: `from progenax.profiles.limepy_tables import AnisoSpeedCDFTable, SpeedCDFTable`
   (no circular import: `limepy_tables` depends only on `profiles.limepy`).
2. Add a static field + init param:

```python
    speed_method: str = eqx.field(static=True)   # field list

    # __init__ signature gains: speed_method: str = "table"
    if speed_method not in ("table", "quadrature"):
        raise ValueError(f"speed_method must be 'table' or 'quadrature', got {speed_method!r}")
    self.speed_method = speed_method
```

3. In `sample_velocities`, branch each path (the quadrature branches are the EXISTING
   code, moved verbatim — they remain the oracle):

```python
        if self.is_aniso:
            p_i = radii / self.r_a
            if self.speed_method == "table":
                # Mirror of cluster/sampling.py:69-97: the speed MARGINAL comes
                # from one precomputed 3-D CDF table; the angular conditional
                # cos(theta)|u stays EXACT (_sample_costheta_given_u). Box
                # covers every star: W <= W0, p <= r_t/r_a (radii <= r_t).
                p_box = jnp.maximum(self.r_t / self.r_a, 1e-3)
                table = AnisoSpeedCDFTable.build(self.W0, p_box, self.g)
                ku_kc = jax.vmap(jax.random.split)(speed_keys)
                unif = jax.vmap(lambda kk: jax.random.uniform(kk))(ku_kc[:, 0])
                u_sp = jax.vmap(table.inverse)(W, p_i, unif)
                cos_t = jax.vmap(
                    lambda kk, uu, pp: _sample_costheta_given_u(kk, uu, pp, _N_C)
                )(ku_kc[:, 1], u_sp, p_i)
                u_r = u_sp * cos_t
                u_t = u_sp * jnp.sqrt(jnp.maximum(1.0 - cos_t**2, 0.0))
            else:
                u_r, u_t = jax.vmap(
                    lambda k, w, sl: _sample_speed_angle(k, w, sl, self.g,
                                                         _N_SPEED_GRID, _N_C)
                )(speed_keys, W, p_i)
            ...  # existing v_r/v_t -> r_hat/t_hat assembly, unchanged

        else:
            if self.speed_method == "table":
                table = SpeedCDFTable.build(self.W0, self.g)
                unif = jax.vmap(lambda kk: jax.random.uniform(kk))(speed_keys)
                u = jax.vmap(table.inverse)(W, unif)
            else:
                u = jax.vmap(lambda k, w: _sample_unit_speed(k, w, self.g,
                                                             _N_SPEED_GRID))(speed_keys, W)
            ...  # existing isotropic assembly, unchanged
```

4. Update the class docstring: table-backed by default (memory O(N·n_x) not
   O(N·256·91); measured 10.87 GB → see Task 4 gate), `speed_method="quadrature"`
   retains the exact per-star quadrature as the oracle; small-N note (aniso table
   build ~160 ms dominates below ~3k stars — batch repeated small draws or use the
   quadrature oracle there).

**Step 4: Run the kinematics tests + fast gate** — the entire existing
`test_limepy_df.py` (equilibrium Q=0.5±0.04, bound checks, g=1≡King/Michie, β
profiles) now runs on the TABLE path and is the real regression net. Then run Task 4's
script WITHOUT `--allow-fail`: the `limepy_df_aniso_20000` gate (<3 GB) must PASS.

**Step 5: Commit**

```bash
git add src/progenax/kinematics/limepy_df.py tests/unit/kinematics/test_limepy_df.py
git commit -m "perf(kinematics): LIMEPYVelocityDF speed draws via CDF tables (10.9 GB -> <3 GB at N=2e4); quadrature retained as oracle"
```

---

## Task 6: Route `MichieVelocityDF` and `KingVelocityDF` the same way

Both are exact special cases of the LIMEPY kernel (verified analytically AND by the
existing cross-tests): King's lowering `e^x − 1 = E_γ(g=1, x)` exactly, Michie = LIMEPY
g=1 anisotropic (`test_g1_velocity_scale_matches_king`,
`test_g1_aniso_matches_michie_velocity_df` are the standing proofs).

**Files:**
- Modify: `src/progenax/kinematics/king_df.py` (`KingVelocityDF`),
  `src/progenax/kinematics/michie_df.py` (`MichieVelocityDF`)
- Test: `tests/unit/kinematics/test_king_df.py`, `tests/unit/kinematics/test_michie_df.py`

**Steps (same TDD rhythm as Task 5, one DF at a time — King first, commit, then Michie):**

1. Failing tests: a `Test<DF>TableRouting` class per file with the
   moments-vs-quadrature (2%/3%), KS (<0.02), and `speed_method` default assertions
   (copy the Task 5 shapes; for Michie also the β-profile preservation test at
   atol=0.06 reusing its existing β helper).
2. Implement: add `speed_method: str = eqx.field(static=True)` with the same
   validation; route:
   - King iso: `SpeedCDFTable.build(self.W0, jnp.asarray(1.0))` — g=1 EXACTLY
     reproduces `exp(W − u²/2) − 1`. Add one extra test asserting the identity at
     the weight level:
     `lowered_exponential(1.0, x) == exp(x) - 1` to rtol 1e-12 on a grid (guards the
     g=1 reduction the routing relies on).
   - Michie aniso: `AnisoSpeedCDFTable.build(self.W0, p_box, jnp.asarray(1.0))` with
     `p_box = jnp.maximum(self.r_c * _find_tidal_radius(self.xi_grid, self.psi_grid) / self.r_a, 1e-3)`
     (Michie has no `r_t` field — import `_find_tidal_radius` from where
     `limepy_df.py` gets it). The angular conditional: reuse
     `_sample_costheta_given_u` (verify Michie's current `_sample_ur_ut` conditional
     weight is `exp(−βu(1−c²))` — it is the same Michie/OM factor; if the code
     differs in any way beyond notation, STOP and report to Anna instead of forcing
     the reuse).
3. Existing physics tests (β rises outward, isotropic-at-large-r_a, Q=0.5±0.08,
   grad-vs-FD) re-run on the table path — they are the regression net.
4. Fast gate after each DF; commits:

```bash
git commit -m "perf(kinematics): KingVelocityDF speed draws via SpeedCDFTable(g=1) - exact King reduction guarded by identity test"
git commit -m "perf(kinematics): MichieVelocityDF speed draws via AnisoSpeedCDFTable(g=1); quadrature oracle retained"
```

---

## Task 7: Validation-script hygiene

**Files:**
- Modify: `scripts/validate_multicomponent_eddington.py`
- Modify: `scripts/validate_multimass_equilibrium.py`

**Steps:**

1. `validate_multicomponent_eddington.py`: delete the local numpy `_chunked_potential`
   (line ~77) and `_chunked_accelerations` (line ~89) mirrors — call
   `progenax.compute_potential_energy` / `progenax.dynamics.virial._accelerations`
   (now memory-bounded; that was the only reason the numpy mirrors existed). Add
   `del icA, icB` (etc.) at the end of each section so sections don't stack arrays.
2. `validate_multimass_equilibrium.py`: per seed it runs BOTH `per_group_virial_ratio`
   AND `compute_virial_ratio` — two full pairwise passes. Compute the global Q from
   the per-group pass instead (Clausius: a single all-ones group reproduces the
   global Q exactly — documented contract of `per_group_virial_ratio`), or simply
   keep both now that each is cheap; prefer the dedup (one pass) and say so in a
   one-line comment.
3. Re-run BOTH scripts end-to-end; both must print ALL PASS with numbers consistent
   with their validation pages (Q values unchanged at the quoted tolerances).

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_multicomponent_eddington.py
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_multimass_equilibrium.py
git add scripts/validate_multicomponent_eddington.py scripts/validate_multimass_equilibrium.py
git commit -m "refactor(scripts): validation scripts on the blocked library kernels; drop numpy mirrors + duplicate N^2 pass"
```

---

## Task 8: Migrate the 5 cumtrap sites onto `numerics.cumulative_trapezoid`

**Files (from the audited inventory):**
- Modify: `src/progenax/kinematics/eff_df.py:54` (named `cumtrap`)
- Modify: `src/progenax/profiles/api.py:285` (named `_cumtrap`)
- Modify: `src/progenax/profiles/density_poisson.py:295` (named `cumtrap`) and the
  King-branch inline at `density_poisson.py:175-191`
- Modify: `src/progenax/cluster/multicomponent.py:214` (axis=1 inline), `:391` (scalar inline)

**Steps:**

1. For each site, capture a before-value: run the narrowest existing test file
   touching it (`tests/unit/kinematics/test_eff_df*.py` equivalents,
   `tests/unit/profiles/test_density_poisson*.py`, `tests/unit/cluster/test_multicomponent.py`)
   and confirm green BEFORE the change.
2. Replace the inline pattern with `cumulative_trapezoid(y, dx=dr)` (or
   `axis=-1` row form). WATCH the dx-placement bit-identity note from Task 1: the
   `density_poisson.py` pattern multiplies dr inside the cumsum — verify its tests
   still pass at their stated tight budgets (the solve budget is |Δψ| ≤ 1.93e-4;
   re-run `scripts/validate_df_tables.py` if any profile test moves).
3. Delete the now-unused local `cumtrap`/`_cumtrap` defs.
4. Fast gate; commit:

```bash
git commit -m "refactor(numerics): all 5 Poisson cumtrap passes on the shared cumulative_trapezoid (review consolidation 2/4)"
```

---

## Task 9: Migrate the quadrature draw sites onto `numerics.inverse_cdf_draw`

The table paths (Task 5–6 defaults) don't use this kernel; this consolidates the
remaining ORACLE quadrature paths so the pattern exists exactly once.

**Files:**
- Modify: `src/progenax/kinematics/king_df.py` (`_sample_unit_speed`)
- Modify: `src/progenax/kinematics/limepy_df.py` (`_sample_unit_speed`,
  `_sample_speed_angle` speed step, `_sample_costheta_given_u`)
- Modify: `src/progenax/kinematics/michie_df.py` (`_sample_ur_ut` both steps)
- Modify: `src/progenax/kinematics/eddington.py:122-128` (`sample_speed_from_f_table`)
- Leave alone: the table `build()` internals (reviewed, row-normalized differently)
  and the n_w=400 oracle in `cluster/eddington_engine.py` (different normalization
  semantics; it keeps its 400-point grid — documented as the deliberate oracle).

**Steps:**

1. Tests first: the Task 1 `test_matches_king_sample_unit_speed_pattern` already pins
   bit-identity for the shared chain; for each migrated site run its narrowest test
   file before AND after — identical pass set is the gate. Where a seed-pinned
   numeric exists (e.g. table-vs-quadrature moment tests), those pin distributional
   identity.
2. Each site becomes: build weight on its grid (physics unchanged, stays in place) →
   `u = inverse_cdf_draw(weight, grid, jax.random.uniform(key))` → existing
   bound-guard `where(W > 1e-6, u, 0.0)`.
3. Fast gate; commit:

```bash
git commit -m "refactor(kinematics): all quadrature inverse-CDF draw sites on numerics.inverse_cdf_draw (review consolidation 3/4)"
```

---

## Task 10: `_EngineAState` — kill the NaN-sentinel union

**Files:**
- Modify: `src/progenax/cluster/multicomponent.py` (class fields → grouped state +
  delegating properties)
- Modify: `src/progenax/cluster/eddington_engine.py:396-403` (`assemble_engine_b_fields`
  no longer emits NaN tripwires)
- Modify: `src/progenax/cluster/sampling.py` (reads via the same public names — should
  need no change if properties preserve names; verify)
- Test: `tests/unit/cluster/test_multicomponent.py` (add class)

**Step 1: Failing tests:**

```python
class TestEngineStateGrouping:
    def test_engine_a_fields_grouped(self):
        m = MultiComponentCluster.from_components(
            alpha_j=jnp.array([0.6, 0.4]), w_j=jnp.array([1.0, 0.8]),
            m_j=jnp.array([0.5, 1.0]), W0=5.0, g=1.0, r_c=1.0)
        assert m.engine_a is not None and m.engine_b is None
        assert float(m.W0) == 5.0          # delegating property

    def test_engine_b_has_no_nan_tripwires(self):
        m = _make_engine_b_model()         # reuse the file's existing helper
        assert m.engine_b is not None and m.engine_a is None

    def test_engine_b_a_only_access_raises_informatively(self):
        m = _make_engine_b_model()
        with pytest.raises(AttributeError, match="Engine A"):
            _ = m.W0

    def test_engine_a_jit_and_grad_still_flow(self):
        """Grouping must not break the pytree: grad through r_c via the state."""
        def f(r_c):
            m = MultiComponentCluster.from_components(
                alpha_j=jnp.array([1.0]), w_j=jnp.array([1.0]),
                m_j=jnp.array([1.0]), W0=5.0, g=1.0, r_c=r_c)
            return jnp.sum(m.total_density(jnp.linspace(0.1, 2.0, 16)))
        assert jnp.isfinite(jax.grad(f)(1.0))
```

**Step 2: Run — verify FAIL** (`engine_a` attribute does not exist).

**Step 3: Implement:**

1. New `_EngineAState(eqx.Module)` in `multicomponent.py` holding the A-only leaves:
   `W0, g, r_c, mu_tot, alpha_j, w_j, ra_hat_j, xi_grid, psi_grid, residual`.
2. `MultiComponentCluster` fields become: shared leaves (`r_t, m_j, N_frac_j,
   _r_grid, _cdf_j`), statics (`is_aniso, engine`), and
   `engine_a: Optional[_EngineAState] = None`, `engine_b: Optional[_EngineBState] = None`.
3. Public access preserved via delegating properties (one per old field name):

```python
    @property
    def W0(self):
        if self.engine_a is None:
            raise AttributeError(
                "W0 is an Engine A (lowered-isothermal) quantity; this model was "
                "built by from_density_profiles (Engine B). Use the profile "
                "parameters instead.")
        return self.engine_a.W0
```

   (Property access happens at Python/trace time — `engine` is static, so raising is
   jit-safe. This REPLACES the NaN-poison semantics with loud, named failures; the
   fix-batch dispatch logic for `total_density`/`rescale_j` keeps its engine branch
   but now reads through `engine_a`.)
4. Update both constructors and `assemble_engine_b_fields` to build the grouped
   states; DELETE the `nan/nan_j/nan_2` tripwire block.
5. Sweep readers (`sampling.py`, `eddington_engine.py`, tests, scripts) — with
   delegating properties most need no edits; fix any direct `_fields` construction.

**Step 4: FULL gate** (this touches the cluster core — run the slow multimass tests too):

```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
```

**Step 5: Commit**

```bash
git commit -m "refactor(cluster)!: _EngineAState grouping replaces NaN-sentinel union; A-only access on B models raises by name (review consolidation 4/4)"
```

---

## Task 11: King density/derivative structure

Light task: the split at `density_poisson.py:170-193` is CORRECT physics (Poisson
identity avoids differentiating interpolated data — the dccedbe staircase fix). Extract
it into one named helper `_king_density_and_dW(profile, r)` returning `(rho, drho)`
with the rationale in its docstring (cite King's Poisson identity + the staircase bug
it fixed), reusing `cumulative_trapezoid` from Task 8. No behavior change: the Engine B
King A-vs-B anchor test and `f_min ≥ -1e-3` realizability tests are the gates.

```bash
git commit -m "refactor(profiles): King rho+dW/dr unified in one Poisson-identity helper (staircase-fix rationale documented)"
```

---

## Task 12: science-capabilities.md — fix and wire in

**Files:**
- Modify: `docs/website/00-getting-started/science-capabilities.md`
- Modify: `docs/website/myst.yml` (TOC)

**Steps (all four claim-check findings):**

1. **TOC**: in `myst.yml`, add `- 00-getting-started/science-capabilities.md`
   immediately after `- 00-getting-started/index.md`.
2. **Test count**: replace "**1101 tests** (833 unit / 34 integration / 234
   physics-validation)" with the POST-BATCH counts — re-derive at the end of the
   batch (Task 13 Step 1), do not hardcode now.
3. **Plummer Q**: `0.502` → `0.5026` (matches `plummer-equilibrium.md`).
4. **Differentiability claim**: narrow the opening parameter list to what the cited
   pages validate. Replace
   "(half-mass radius, King concentration $W_0$, truncation shape $g$, equipartition
   exponent $\delta$, anisotropy radius $r_a$, IMF slope, binary fraction)"
   with
   "(half-mass radius, King concentration $W_0$, truncation shape $g$, equipartition
   exponent $\delta$, anisotropy radius $r_a$, IMF slope)"
   and add the AD-vs-FD evidence link `(../50-validation/multimass-equilibrium.md)`
   alongside the differentiability-rules link.
5. **Evidence-link nit**: in the binary table row "Kepler III exact to machine
   precision; orbital energy 4.2e-16", add `([evidence](../50-validation/binary-imf.md))`.
6. Precision nit: "$10^{-11}$–$10^{-3}$" → "$10^{-11}$–$5.6\times 10^{-3}$".
7. Build-verify:

```bash
cd docs/website && myst build > /tmp/myst.log 2>&1; echo "exit $?"; grep -ic warning /tmp/myst.log
```
Expected: exit 0, zero warnings, page count 143 (was 142 — the new page).

```bash
git add docs/website/00-getting-started/science-capabilities.md docs/website/myst.yml
git commit -m "docs(website): science-capabilities page - claim-checked vs validation ledger, wired into TOC"
```

---

## Task 13: Docs metrics refresh — update numbers, REMOVE stale limitations

Anna's explicit instruction: update metrics/results and remove stale ones — do NOT
catalog failed experiments or superseded numbers in the docs.

**Files:**
- Modify: `docs/website/50-validation/index.md` (dashboard test counts)
- Modify: `docs/website/00-getting-started/whats-new.md` (test counts; add a short
  dated entry for this batch: blocked kernels + table-routed DFs + grouped state)
- Modify: `docs/website/10-theory/populations/eddington-engine.md` (limitations
  admonition: DELETE every item this batch fixed — Engine-A-only accessor poison
  (now named errors), sampler scale thresholds (fixed in 0dd1cd9), any consolidation
  notes — keep ONLY limitations that still exist)
- Modify: `docs/website/50-validation/multimass-equilibrium.md` and/or the page
  section quoting sampler performance: add the measured memory row (peak RSS,
  from `scripts/profile_cluster_memory.py` output) next to the existing speedup
  numbers; replace any stale memory caveats
- Modify: `STATUS.md` (collapse the done arc; `next:` = Batch B cross-engine figures
  + science demos; mention the memory fix with its headline number)

**Steps:**

1. Re-derive the test counts (single source for every page):

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit -q --co | tail -1
env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration -q --co | tail -1
env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation -q --co | tail -1
```

2. Apply the counts to `science-capabilities.md` (Task 12 placeholder),
   `50-validation/index.md`, `whats-new.md`.
3. Sweep for any remaining "1101" / "32.8 GB"-class stale numbers:

```bash
grep -rn "1101" docs/website --include="*.md" | grep -v 90-development-log
```
4. Build-verify (exit 0, zero warnings); full myst page-count noted.
5. Commit:

```bash
git commit -m "docs(website): metrics refresh post-batch - test counts, memory results, stale limitations removed"
```

---

## Task 14: Close-out — full evidence run

**Steps:**

1. FULL gate (all tests, no marker filter) — record the exact `N passed` line.
2. `scripts/profile_cluster_memory.py` — ALL gates pass, no `--allow-fail`; capture
   the table.
3. Both validation scripts from Task 7 — ALL PASS.
4. `myst build` — exit 0, zero warnings, 143 pages.
5. Write `.claude-work/BATCH_A_MEMORY_CONSOLIDATION_COMPLETE.md`: implementation
   summary, the before/after memory table (32.77→measured, 10.87→measured),
   consolidation map (which 13 call sites moved onto `numerics`), test-count change,
   lessons (the dx-placement bit-identity, the padded-row vjp trap), and the Batch B
   handoff.
6. Final commit; then STOP and report to Anna with the evidence tables. Merge to
   `main` and push ONLY on her explicit go.

```bash
git add .claude-work/BATCH_A_MEMORY_CONSOLIDATION_COMPLETE.md STATUS.md
git commit -m "docs(status): Batch A complete - memory fixes measured, consolidations landed, docs current"
```

---

## Out of scope (explicitly deferred, do not drift into)

- `binaries/diagnostics.find_bound_pairs`/`find_bound_multiples` O(N²) (documented
  "N ≲ a few×10³" — a future batch if survey-scale binary diagnostics are needed).
- `diagnostics/segregation_approx.py` + `q_approx` O(N²) (same: documented regime).
- `eddington.py` per-star `(N,256)` chains under jit (~1 GB at 1e5 — matters only at
  N≥1e6; revisit with a prebuilt Engine B speed-CDF table if survey-scale Engine B
  sampling becomes a use case).
- float32 sampling tables (would halve table/draw memory; needs Anna's sign-off
  against the float64-everywhere convention — raise separately if ever needed).
- Batch B (cross-engine figures + science demos) — separate brainstorm + plan.
