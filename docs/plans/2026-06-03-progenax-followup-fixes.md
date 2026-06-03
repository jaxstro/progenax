# Progenax Follow-Up Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Resolve the 2 🔴 Critical, 2 🟠 Major, and ~10 🟡 Minor findings from the
2026-06-03 follow-up audit (`docs/website/90-development-log/code-reviews.md`), taking the
package from A− (90/100) to a clean A.

**Architecture:** Each fix is driven by a failing-then-passing test (TDD). The two Criticals
are "untested twins" — a correct fix already exists in the repo for a sibling call site; we
port it to the path the audit found unfixed. JAX-native throughout (`jax.numpy`, no
numpy/scipy in core, float64, differentiability preserved).

**Tech Stack:** JAX 0.10.1 (float64), Equinox, jaxtyping, pytest (+pytest-cov), uv.

**Branch:** `hardening/followup-2026-06` (off `main` @ `644c28f`). **One PR at the end.**

---

## Operating model (enforced every batch)

- **TDD per fix:** write the RED test → run it → confirm it fails *for the right reason* →
  minimal fix → GREEN. **Never** weaken a test or tolerance to make it pass.
- **Per-batch gate:** after each batch, STOP, report the diff + test evidence, and wait for
  Anna's explicit go before the next batch.
- **Run commands from the repo root** with the project venv:
  `cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync <cmd>`
  (cwd does not persist between shell calls; always prefix the `cd`).
- Apply `research-workflow:gradient-validation` (FD-vs-autodiff, double-`where`) and
  `numerical-method-validation` where relevant.

---

## Task 0: Branch + commit the audit record

**Files:**
- Already modified (uncommitted): `docs/website/90-development-log/code-reviews.md` (the
  2026-06-03 follow-up section + summary-table row).

**Step 1: Create the branch (carries the uncommitted edit)**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git checkout -b hardening/followup-2026-06
```

**Step 2: Commit only the audit record**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git add docs/website/90-development-log/code-reviews.md && git commit -m "$(cat <<'EOF'
docs(review): add 2026-06-03 follow-up audit (A-, 2 launch-blockers)

Five-lane post-hardening regression + new-code audit of 22ad6ad..644c28f.
All hardening claims verified true; surfaced two Critical "untested twins"
(bm19 default sampler still OOMs via random.categorical at fdf_tail.py:382;
build_spatial_ic crashes under jax.grad via float(softening) at builders.py:258)
plus 2 Major and ~10 Minor. Grade A- (90/100).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Step 3: Verify clean tree + on branch**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git status -sb && git log --oneline -1
```
Expected: `## hardening/followup-2026-06`, working tree clean (ignored `.claude/` aside).

---

# BATCH A — the 2 🔴 Critical launch-blockers

## Task 1: `build_spatial_ic` must be differentiable w.r.t. `r_h` (CR-FU-2)

**Root cause:** `builders.py:258` `softening = float(softening)` concretizes a tracer
(`softening = softening_factor · profile.characteristic_radius()/N^{1/3}`), so
`jax.grad` through the flagship public IC builder — the exact CLAUDE.md "fully
differentiable" example — raises `ConcretizationTypeError`. The existing grad tests miss it
because they call `sample_positions`/`sample_velocities` directly, never `build_spatial_ic`.

**Files:**
- Test: `tests/integration/test_jax_compatibility.py` (add one test)
- Modify: `src/progenax/builders.py:258`

**Step 1: Write the failing test**

```python
def test_build_spatial_ic_differentiable_wrt_r_h():
    """jax.grad through the public build_spatial_ic (the CLAUDE.md 'fully
    differentiable' example) must return a finite, correct gradient (audit CR-FU-2)."""
    import jax
    import jax.numpy as jnp
    from jaxstro.units import STELLAR
    from progenax import PlummerProfile, PlummerVelocityDF, build_spatial_ic

    def loss(r_h):
        masses = jnp.ones(64)
        ic = build_spatial_ic(
            PlummerProfile(r_h=r_h), masses, PlummerVelocityDF(r_h=r_h),
            key=jax.random.PRNGKey(0), G=STELLAR.G,
        )
        return jnp.mean(jnp.linalg.norm(ic.positions, axis=1))

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g), f"grad is {g}, expected finite"
    # mean radius scales ~linearly with r_h -> positive, O(1) sensitivity
    fd = (loss(1.0 + 1e-4) - loss(1.0 - 1e-4)) / 2e-4
    assert abs(g - fd) / (abs(g) + abs(fd) + 1e-30) < 1e-4, f"grad {g} vs FD {fd}"
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration/test_jax_compatibility.py::test_build_spatial_ic_differentiable_wrt_r_h -q
```
Expected: FAIL — `ConcretizationTypeError` at `builders.py:258`.

**Step 3: Minimal fix** — `src/progenax/builders.py:258`, replace:

```python
    softening = float(softening)
```
with:

```python
    # Keep softening as a JAX scalar so build_spatial_ic stays differentiable
    # (float() concretized a tracer -> broke jax.grad wrt r_h; audit CR-FU-2).
    # It is only stored on ICResult and passed to virial_scale, both array-safe.
    softening = jnp.asarray(softening)
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration/test_jax_compatibility.py::test_build_spatial_ic_differentiable_wrt_r_h -q
```
Expected: PASS.

**Step 5: Guard against regression in the broader suite (softening is now a 0-d array)**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/ -k "build_spatial_ic or builder or ICResult or virial" -q
```
Expected: PASS (no consumer assumed a Python float).

**Step 6: Commit**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git add tests/integration/test_jax_compatibility.py src/progenax/builders.py && git commit -m "$(cat <<'EOF'
fix(builders): keep softening a JAX scalar so build_spatial_ic is differentiable (CR-FU-2)

float(softening) concretized a tracer, crashing jax.grad through the flagship
public IC builder (the CLAUDE.md differentiable example). Now jnp.asarray;
added a grad-through-build_spatial_ic regression test the old grad tests skipped.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `mode="bm19"` tail sampler must not materialize an (N×n_cells) array (CR-FU-1)

**Root cause:** `sample_from_pmf` (`fdf_tail.py:382`) uses
`random.categorical(key, log_pmf, shape=(n_samples,))` — Gumbel-max, which materializes an
`(n_samples, n_cells)` array (~10 GB at 5000×64³, called twice → ~26 GB OOM). The CI memory
fix (`c5d2061`) ported `cumsum`+`searchsorted` only to the `pn11_legacy` twin in
`fdf_density/sampling.py`; the default/recommended `mode="bm19"` path still hits this.

**Files:**
- Test: `tests/unit/cluster/test_tail_sampling.py` (add two tests)
- Modify: `src/progenax/cluster/fdf_tail.py:355-382` (`sample_from_pmf`)

**Step 1: Write the failing test (memory-property, RAM-independent) + a faithfulness guard**

```python
class TestSampleFromPmfMemory:
    """sample_from_pmf must use O(n_cells + n_samples) inverse-CDF sampling, not the
    O(n_samples * n_cells) Gumbel-max categorical that OOMs the default bm19 path (CR-FU-1)."""

    def test_no_quadratic_materialization(self):
        import numpy as np
        import jax
        import jax.numpy as jnp
        from progenax.cluster.fdf_tail import sample_from_pmf

        n_cells, n_samples = 4096, 512
        pmf = jnp.ones(n_cells) / n_cells
        jaxpr = jax.make_jaxpr(lambda k: sample_from_pmf(k, pmf, n_samples))(
            jax.random.PRNGKey(0)
        )
        max_elems = 0
        for eqn in jaxpr.eqns:
            for v in eqn.outvars:
                aval = getattr(v, "aval", None)
                shape = getattr(aval, "shape", None)
                if shape is not None:
                    max_elems = max(max_elems, int(np.prod(shape)) if shape else 1)
        assert max_elems < n_cells * n_samples, (
            f"sample_from_pmf materializes a {max_elems}-element array "
            f">= n_cells*n_samples={n_cells * n_samples} (Gumbel-max OOM, CR-FU-1)"
        )

    def test_distribution_faithful(self):
        """The inverse-CDF sampler must reproduce the PMF (same statistics as categorical)."""
        import numpy as np
        import jax
        import jax.numpy as jnp
        from progenax.cluster.fdf_tail import sample_from_pmf

        n_cells, n_samples = 64, 400_000
        rng = np.random.default_rng(0)
        p = rng.random(n_cells) ** 2
        p = jnp.asarray(p / p.sum())
        idx = np.asarray(sample_from_pmf(jax.random.PRNGKey(1), p, n_samples))
        freq = np.bincount(idx, minlength=n_cells) / n_samples
        assert np.max(np.abs(freq - np.asarray(p))) < 5e-3, "empirical freq != PMF"
        assert idx.min() >= 0 and idx.max() < n_cells, "indices out of range"
```

**Step 2: Run to verify both fail/behave correctly**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/cluster/test_tail_sampling.py::TestSampleFromPmfMemory -q
```
Expected: `test_no_quadratic_materialization` FAILS (categorical jaxpr has a 512×4096
intermediate); `test_distribution_faithful` PASSES (categorical is already faithful — it is
the regression guard for the fix).

**Step 3: Minimal fix** — `src/progenax/cluster/fdf_tail.py`, replace the body of
`sample_from_pmf` (lines 381-382) with the inverse-CDF form (mirrors the existing
`fdf_density/sampling.py` pn11_legacy fix):

```python
    # Inverse-CDF sampling: O(n_cells + n_samples) memory. (Was random.categorical,
    # i.e. Gumbel-max, which materialized an (n_samples, n_cells) array and OOM'd the
    # default bm19 path at production scale -- audit CR-FU-1.)
    cdf = jnp.cumsum(pmf)
    cdf = cdf / cdf[-1]  # normalize (pmf may not sum to exactly 1)
    u = random.uniform(key, (n_samples,))
    idx = jnp.searchsorted(cdf, u, side="right")
    return jnp.clip(idx, 0, pmf.shape[0] - 1).astype(jnp.int32)
```
(Update the `Notes` in the docstring: it is now memory-efficient inverse-CDF sampling.)

**Step 4: Run to verify GREEN**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/cluster/test_tail_sampling.py -q
```
Expected: PASS (both new tests + the existing `TestMassFractionSplit`).

**Step 5: Confirm the bm19 field-tail validation still holds (statistics unchanged)**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_bm19_field_tail.py tests/unit/cluster/ -q
```
Expected: PASS.

**Step 6: Commit**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git add tests/unit/cluster/test_tail_sampling.py src/progenax/cluster/fdf_tail.py && git commit -m "$(cat <<'EOF'
fix(fdf): inverse-CDF in sample_from_pmf so the default bm19 path doesn't OOM (CR-FU-1)

The CI memory fix patched only the pn11_legacy twin; the default/recommended
mode='bm19' path still hit random.categorical (Gumbel-max, ~26GB at 5000x64^3).
Ported the same cumsum+searchsorted inverse-CDF. Tests: jaxpr no-quadratic-
materialization guard + distribution-faithfulness vs the PMF.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

### GATE A — STOP and report
Report both RED→GREEN transitions, the diffs, and a full `pytest tests/ -m "not slow" -q`
run. Wait for Anna's go before Batch B.

---

# BATCH B — the 2 🟠 Major

## Task 3: `compute_potential_energy(softening=0.0)` must have a finite gradient

**Root cause:** `builders.py:132-135` does `r_soft = sqrt(r² + softening²)` then masks the
diagonal with `where(eye, inf, ·)` *after* the sqrt. At the default `softening=0`, the
diagonal `sqrt(0)` has an infinite derivative; `0·inf = nan` survives the later `where`.

**Files:**
- Test: `tests/integration/test_jax_compatibility.py` (add)
- Modify: `src/progenax/builders.py:127-141`

**Step 1: Write the failing test**

```python
def test_compute_potential_energy_grad_finite_at_default_softening():
    """grad of the public compute_potential_energy at the default softening=0 (the
    CLAUDE.md C1 example form) must be finite and FD-correct (double-where; audit 🟠)."""
    import jax
    import jax.numpy as jnp
    from jaxstro.units import STELLAR
    from progenax import compute_potential_energy

    pos = jax.random.normal(jax.random.PRNGKey(1), (16, 3))
    m = jnp.ones(16)
    f = lambda p: compute_potential_energy(p, m, G=STELLAR.G)  # softening=0 default
    g = jax.grad(f)(pos)
    assert jnp.all(jnp.isfinite(g)), "grad not finite at softening=0"
    # FD check on a random direction
    v = jax.random.normal(jax.random.PRNGKey(2), pos.shape)
    v = v / jnp.linalg.norm(v)
    fd = (f(pos + 1e-5 * v) - f(pos - 1e-5 * v)) / 2e-5
    ad = jnp.sum(g * v)
    assert abs(ad - fd) / (abs(ad) + abs(fd) + 1e-30) < 1e-5, f"ad {ad} vs fd {fd}"
```

**Step 2: Run to verify it fails**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration/test_jax_compatibility.py::test_compute_potential_energy_grad_finite_at_default_softening -q
```
Expected: FAIL — assertion "grad not finite" (`max|g| = nan`).

**Step 3: Minimal fix** — `src/progenax/builders.py:127-141`, replace lines 131-135:

```python
    r_squared = jnp.sum(diff**2, axis=2)  # (N, N)
    r_soft = jnp.sqrt(r_squared + softening**2)  # Plummer softening

    # Avoid self-interaction
    r_soft = jnp.where(jnp.eye(N, dtype=bool), jnp.inf, r_soft)
```
with the double-`where` (mask the diagonal *before* the sqrt so its derivative is never
evaluated at 0):

```python
    r_squared = jnp.sum(diff**2, axis=2)  # (N, N)
    eye = jnp.eye(N, dtype=bool)
    # Double-where: feed the diagonal a safe positive value BEFORE sqrt (else the
    # diagonal sqrt(0) derivative is inf and 0*inf=nan survives a later where), then
    # set the diagonal to inf so the i<j sum drops it.
    r_squared_safe = jnp.where(eye, 1.0, r_squared + softening**2)
    r_soft = jnp.where(eye, jnp.inf, jnp.sqrt(r_squared_safe))
```

**Step 4: Run to verify GREEN (value unchanged, grad finite + correct)**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration/test_jax_compatibility.py::test_compute_potential_energy_grad_finite_at_default_softening tests/ -k "potential_energy or virial or energy" -q
```
Expected: PASS (the forward energy is identical — the diagonal never entered the `i<j` sum).

**Step 5: Commit**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git add tests/integration/test_jax_compatibility.py src/progenax/builders.py && git commit -m "$(cat <<'EOF'
fix(builders): double-where in compute_potential_energy for finite grad at softening=0

The diagonal sqrt(0) derivative is inf; 0*inf=nan survived the post-sqrt where,
so jax.grad NaN'd at the default softening=0 (the CLAUDE.md C1 example form).
Mask the diagonal to a safe value before sqrt. Forward value unchanged (i<j sum).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: De-flake `test_phi_copula_reproduces_the_bug`

**Root cause:** `tests/validation/test_bm19_field_tail.py:63` asserts
`vals.std() > 0.2*f_dense`, which fails on ~half of seed-blocks (on some seeds the Φ-copula
undersamples so hard the tail nearly vanishes, shrinking the scatter). The defining,
seed-robust symptom is the **undersampled mean**. This is *not* weakening the test: the
bug-reproduction assertion is kept and the over-tight scatter bound is replaced with a
robust relative comparison to the rank copula.

**Files:**
- Modify: `tests/validation/test_bm19_field_tail.py:57-64`

**Step 1: Demonstrate the flakiness (RED evidence)**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync python -c "
import warnings, numpy as np, jax
import progenax
from progenax.gravoturb import bm19_pipeline
from progenax.gravoturb.bm19_model import f_dense_bm19_full
from progenax.cluster.fdf_density import init_bm19_density_field
from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19
r = bm19_pipeline(2.0, 2.0); ss, st, al = float(r.sigma_s_sq), float(r.s_t), 2.0
fd = float(f_dense_bm19_full(ss, st, al))
for start in (0, 12, 16, 20):
    vals=[]
    for s in range(start, start+4):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fld = init_bm19_density_field(jax.random.PRNGKey(s), ss, st, al, grid_size=48, beta=4.0, copula='phi')
        vals.append(float(compute_tail_pmfs_bm19(fld.rho_grid, st).f_tail_actual))
    vals=np.array(vals)
    print(f'seeds {start}-{start+3}: mean {vals.mean():.4f} std {vals.std():.4f}  std>0.2*fd? {vals.std()>0.2*fd}  mean<0.85*fd? {vals.mean()<0.85*fd}')
"
```
Expected: the `std>0.2*fd` column flips to `False` on some blocks while `mean<0.85*fd`
stays `True` everywhere — confirming the mean is the robust symptom.

**Step 2: Replace the fragile assertion** — `tests/validation/test_bm19_field_tail.py:57-64`:

```python
    def test_phi_copula_reproduces_the_bug(self):
        """Regression guard: the legacy phi copula undersamples the dense tail at beta=4
        (the audit-M3 symptom) -- which is *why* rank is the default. The undersampled
        MEAN is the seed-robust signature; the rank copula reproduces f_dense."""
        vals, f_dense = _measure_f_tail(seeds=8, grid_size=48, beta=4.0, copula="phi")
        rank_vals, _ = _measure_f_tail(seeds=8, grid_size=48, beta=4.0, copula="rank")
        assert vals.mean() < 0.85 * f_dense, (
            f"phi f_tail mean {vals.mean():.4f} not undersampling f_dense {f_dense:.4f}"
        )
        assert vals.mean() < rank_vals.mean() - 0.1 * f_dense, (
            f"phi ({vals.mean():.4f}) not materially worse than rank "
            f"({rank_vals.mean():.4f}); bug not reproduced"
        )
```

**Step 3: Run across the same seed regimes to confirm robustness**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_bm19_field_tail.py -q
```
Expected: PASS (and stable across runs).

**Step 4: Commit**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git add tests/validation/test_bm19_field_tail.py && git commit -m "$(cat <<'EOF'
test(bm19): de-flake the phi-copula bug-reproduction guard (robust mean signature)

The std>0.2*f_dense scatter bound failed on ~half of seed-blocks. Keep the
bug-reproduction (undersampled mean) and contrast phi vs rank on the same seeds
-- a more discriminating, seed-robust assertion. Not a weakening.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

### GATE B — STOP and report. Wait for go before Batch C.

---

# BATCH C — code Minors

## Task 5: `init_bm19_density_field` resolution guard must not break `jax.grad`/`jit`

**Root cause:** `field_init.py:284` `float(...)` + the `if expected_tail_cells < 5.0` warn
are host-side ops that raise `ConcretizationTypeError` under tracing — defeating the M3
design-doc claim that param gradients flow through the CDF table.

**Files:**
- Test: `tests/integration/test_jax_compatibility.py` (add)
- Modify: `src/progenax/cluster/fdf_density/field_init.py:280-297`

**Step 1: Write the failing test**

```python
def test_init_bm19_density_field_differentiable_in_params():
    """The resolution guard must be skipped under tracing so init_bm19_density_field
    is differentiable in its BM19 params (M3 design-doc claim; audit minor)."""
    import jax
    import jax.numpy as jnp
    from progenax.cluster.fdf_density import init_bm19_density_field

    def summary(sigma_s_sq):
        s_t = (2.0 - 0.5) * sigma_s_sq
        fld = init_bm19_density_field(
            jax.random.PRNGKey(2), sigma_s_sq, s_t, 2.0, grid_size=16
        )
        return jnp.sum(fld.rho_grid)

    g = jax.grad(summary)(1.0)
    assert jnp.isfinite(g), f"grad is {g}, expected finite"
```

**Step 2: Run to verify it fails**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration/test_jax_compatibility.py::test_init_bm19_density_field_differentiable_in_params -q
```
Expected: FAIL — `ConcretizationTypeError` at `field_init.py:284`.

**Step 3: Minimal fix** — `field_init.py`, wrap the guard block (lines 284-297) so it runs
only on concrete inputs:

```python
    # Resolution guard (audit M3) -- host-side check on CONCRETE inputs only; under
    # jax.grad/jit the BM19 param gradients flow through the CDF table (rank copula
    # is grad-safe), so skip the float()/warn when tracing.
    try:
        tail_prob = float(jnp.clip(1.0 - jnp.interp(s_t, s_grid, F_grid), 0.0, 1.0))
    except jax.errors.ConcretizationTypeError:
        tail_prob = None
    if tail_prob is not None:
        expected_tail_cells = tail_prob * (Nx * Ny * Nz)
        if expected_tail_cells < 5.0:
            import warnings

            need = 5.0 / max(tail_prob, 1e-30)
            warnings.warn(
                f"BM19 dense tail under-resolved at grid_size={grid_size}: only "
                f"~{expected_tail_cells:.1f} cells expected above s_t={s_t:.2f} "
                f"(tail probability {tail_prob:.2e}). Realized f_tail will read low even "
                f"with the rank copula; increase grid_size (need N^3 >~ {need:.0e}).",
                UserWarning,
                stacklevel=2,
            )
```
(Add `import jax` at the top of `field_init.py` if not already imported.)

**Step 4: Run GREEN + confirm the concrete-path warn still fires**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration/test_jax_compatibility.py::test_init_bm19_density_field_differentiable_in_params tests/validation/test_bm19_field_tail.py -q
```
Expected: PASS (incl. `test_resolution_guard_warns_on_unresolved_tail` — concrete calls
still warn).

**Step 5: Commit**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git add tests/integration/test_jax_compatibility.py src/progenax/cluster/fdf_density/field_init.py && git commit -m "fix(fdf): run the BM19 resolution guard only on concrete inputs (keeps init_bm19 differentiable)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Export `energy_sorted_segregation` (doc overclaim → make it true)

**Root cause:** `CLAUDE.md:219` lists `energy_sorted_segregation()` under "exported from
`progenax.__init__`", but it is not in `progenax.__all__` (only reachable as
`progenax.cluster.energy_sorted_segregation`). It is a genuine public utility; the correct
fix is to export it (additive, not a compat shim).

**Files:**
- Test: `tests/unit/test_public_api.py` (create)
- Modify: `src/progenax/__init__.py` (add import + `__all__` entry near the other cluster exports)

**Step 1: Write the failing test**

```python
# tests/unit/test_public_api.py
"""Public-API surface guards (exports resolve; doc-advertised symbols exist)."""
import progenax


def test_energy_sorted_segregation_is_top_level_export():
    assert "energy_sorted_segregation" in progenax.__all__
    assert hasattr(progenax, "energy_sorted_segregation")
```

**Step 2: Run to verify it fails**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_public_api.py::test_energy_sorted_segregation_is_top_level_export -q
```
Expected: FAIL (`assert 'energy_sorted_segregation' in __all__`).

**Step 3: Minimal fix** — in `src/progenax/__init__.py`, add the import alongside the other
`from .cluster...` imports and add `"energy_sorted_segregation"` to `__all__`:

```python
from .cluster import energy_sorted_segregation
```
(If `progenax.cluster` does not re-export it, use
`from .cluster.mass_segregation import energy_sorted_segregation`. Verify with
`python -c "from progenax.cluster import energy_sorted_segregation"` first.)

**Step 4: Run GREEN**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_public_api.py -q && env -u VIRTUAL_ENV uv run --no-sync python -c "import progenax; print('OK', 'energy_sorted_segregation' in progenax.__all__)"
```
Expected: PASS, `OK True`.

**Step 5: Commit**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git add tests/unit/test_public_api.py src/progenax/__init__.py && git commit -m "feat(api): export energy_sorted_segregation at top level (matches docs)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Cover `profiles/api.py` (37% → exercise King/EFF dispatch + potentials)

**Files:**
- Test: `tests/unit/profiles/test_profile_api.py` (create)

**Step 1: Write tests exercising the uncovered branches (lines 101-117, 241-302)**

```python
# tests/unit/profiles/test_profile_api.py
"""Functional profile API: factory + sampling + analytic potentials for all 3 profiles."""
import jax
import jax.numpy as jnp
import pytest
from jaxstro.units import STELLAR
from progenax.profiles.api import make_profile, sample_density_profile, compute_profile_potential
from progenax.profiles.plummer import PlummerProfile
from progenax.profiles.king import KingProfile
from progenax.profiles.eff import EFFProfile


@pytest.mark.parametrize("name,cls,kw", [
    ("plummer", PlummerProfile, {}),
    ("king", KingProfile, {"W0": 7.0}),
    ("eff", EFFProfile, {"gamma": 3.0, "r_t": 15.0}),
])
def test_make_profile_dispatch(name, cls, kw):
    p = make_profile(name, R_half=1.0, **kw)
    assert isinstance(p, cls)


def test_make_profile_unknown_raises():
    with pytest.raises(ValueError, match="Unknown profile"):
        make_profile("hernquist", R_half=1.0)


@pytest.mark.parametrize("name,kw", [
    ("plummer", {}), ("king", {"W0": 7.0}), ("eff", {"gamma": 3.0, "r_t": 15.0}),
])
def test_sample_density_profile_shape(name, kw):
    pos = sample_density_profile(jax.random.PRNGKey(0), 200, name, R_half=1.0, **kw)
    assert pos.shape == (200, 3) and jnp.all(jnp.isfinite(pos))


@pytest.mark.parametrize("name,kw", [
    ("plummer", {}), ("king", {"W0": 7.0}),
    ("eff", {"gamma": 3.0, "r_t": 15.0}), ("eff", {"gamma": 4.0, "r_t": 15.0}),
])
def test_compute_profile_potential_negative_and_finite(name, kw):
    pos = jnp.array([[0.5, 0.0, 0.0], [0.0, 1.5, 0.0], [2.0, 0.0, 0.0]])
    phi = compute_profile_potential(pos, name, M_total=1000.0, R_half=1.0, G=STELLAR.G, **kw)
    assert phi.shape == (3,)
    assert jnp.all(phi < 0) and jnp.all(jnp.isfinite(phi)), f"{name}: {phi}"


def test_compute_profile_potential_unknown_raises():
    with pytest.raises(ValueError, match="Unknown profile"):
        compute_profile_potential(jnp.zeros((2, 3)), "hernquist", M_total=1.0, R_half=1.0, G=STELLAR.G)
```

**Step 2: Run to verify they pass + lift coverage** (these are new tests on existing code —
they go green on write; the goal is to exercise the previously-uncovered dispatch branches)

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/profiles/test_profile_api.py -q --cov=progenax.profiles.api --cov-report=term-missing
```
Expected: PASS; `profiles/api.py` coverage well above 37% (King/EFF branches + ValueErrors hit).

**Step 3: Commit**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git add tests/unit/profiles/test_profile_api.py && git commit -m "test(profiles): cover the functional profile API (factory, sampling, potentials)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Pin `c(W₀)` to King (1966) Table II (guard the M6 flagship fix)

**Files:**
- Test: `tests/validation/test_king_physics.py` (add)

**Step 1: Write the regression guard (existing-correct behavior; goes green on write)**

```python
def test_concentration_matches_king1966_table_ii():
    """c(W0)=log10(r_t/r_c) must match King (1966) Table II to <=0.02 (audit M6 guard).
    Reference c: W0=3 -> 0.67, W0=7 -> 1.53, W0=9 -> 2.12 (King 1966; B&T 2008)."""
    import jax.numpy as jnp
    from progenax import KingProfile
    ref = {3.0: 0.67, 7.0: 1.53, 9.0: 2.12}
    for w0, c_ref in ref.items():
        p = KingProfile.from_W0_rc(W0=w0, r_c=1.0)
        c = float(jnp.log10(p.r_t / p.r_c))
        assert abs(c - c_ref) <= 0.02, f"W0={w0}: c={c:.3f} vs King Table II {c_ref} (>0.02)"
```

**Step 2: Run to verify it passes**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_king_physics.py::test_concentration_matches_king1966_table_ii -q
```
Expected: PASS (Lane 1 measured 0.681 / 1.529 / 2.119). If the API differs, confirm
`KingProfile.from_W0_rc(W0=..., r_c=...)` and `.r_t`/`.r_c` attributes first.

**Step 3: Commit**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git add tests/validation/test_king_physics.py && git commit -m "test(king): pin c(W0) to King (1966) Table II (<=0.02) -- guards the M6 fix

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### GATE C — STOP and report. Wait for go before Batch D.

---

# BATCH D — doc Minors + ticket + closeout

## Task 9: Fix the doc defects (phantom ref, stale counts, docstring)

**Files (edits):**
- `README.md:86` — `` `harmonic_oscillator_1d()` `` → `` `harmonic_oscillator()` ``
- `CLAUDE.md:217` — `harmonic_oscillator_1d()` → `harmonic_oscillator()`
- `tests/README.md:150` — `($c \approx 0.8$)` → `($c \approx 0.67$, King 1966 Table II)`;
  remove/replace the removed-`king_K_function` test rows (lines 129, 158-159) with the current
  King-density tests; update `391 tests` (line 5) → the current count.
- `docs/website/90-development-log/code-reviews.md:600` — `suite collects **848**` →
  `suite collects **855**`.
- `README.md` / `CLAUDE.md` LOC count (`18,936`) → `18,945` if present.
- `src/progenax/cluster/fdf_tail.py` `compute_tail_pmfs_bm19` docstring — soften
  "f_tail_actual ≈ f_dense by construction" to note it reads ~0.96·f_dense at the default
  sigmoid κ.

**Step 1: Apply the edits** (use Edit per file; verify each old-string is unique).

**Step 2: Verify no phantom refs remain**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && grep -rn "harmonic_oscillator_1d\|c \\\\approx 0.8\|collects \*\*848" CLAUDE.md README.md tests/README.md docs/ || echo "NONE FOUND (good)"
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync python -c "import progenax; assert hasattr(progenax,'harmonic_oscillator'); print('harmonic_oscillator exists:', 'harmonic_oscillator' in progenax.__all__)"
```
Expected: `NONE FOUND (good)`; `harmonic_oscillator exists: True`.

**Step 3: Add a doc-symbol guard test** — `tests/unit/test_public_api.py` (append):

```python
def test_doc_advertised_analytical_symbols_exist():
    """Guard against phantom doc refs: every analytical name advertised in the docs
    must resolve (audit found harmonic_oscillator_1d, a 4th phantom)."""
    import progenax
    advertised = [
        "two_body_kepler", "three_body_figure_eight", "earth_sun_2body",
        "solar_system_inner_4", "solar_system_full", "harmonic_oscillator",
    ]
    missing = [n for n in advertised if not hasattr(progenax, n)]
    assert not missing, f"doc-advertised symbols missing from progenax: {missing}"
```

**Step 4: Run**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_public_api.py -q
```
Expected: PASS.

**Step 5: Commit**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git add -A && git commit -m "docs: fix harmonic_oscillator_1d phantom, stale c/test/count refs; add doc-symbol guard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Extend the file-length ticket + mark the follow-up findings resolved

**Files:**
- Modify: `docs/notes/2026-06-02-file-length-followup-ticket.md` — add a section noting the
  9 functions >100 LOC in the split files (`sample_positions_tail` 231, `init_bm19_density_field`
  157, `two_body_kepler` 145, `generate_fractal_ic_density` 140, `env_to_imf_params` 137,
  `init_turbulent_density_field` 131, `solar_system_inner_4` 125, `three_body_figure_eight` 111,
  `solar_system_full` 103) as a function-length follow-up.
- Modify: `docs/website/90-development-log/code-reviews.md` — append a short **Resolution
  status** subsection to the 2026-06-03 section: CR-FU-1 ✅ (Task 2), CR-FU-2 ✅ (Task 1),
  🟠 PE NaN-grad ✅ (Task 3), 🟠 flaky test ✅ (Task 4), and the minors ✅ (Tasks 5-9), with
  commit refs filled in after the fact.

**Step 1: Apply the edits.**

**Step 2: Commit**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git add docs/ && git commit -m "docs(review): mark 2026-06-03 follow-up findings resolved + extend file-length ticket

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### GATE D — STOP and report. Wait for go before opening the PR.

---

## Final: open the PR

**Step 1: Full suite (the launch gate)**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && env -u VIRTUAL_ENV uv run --no-sync pytest tests/ -m "not slow" -q
```
Expected: all pass (≥ the prior 854 + the new tests), 0 collection errors, 0 unknown-mark warnings.

**Step 2: Push + PR**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax && git push -u origin hardening/followup-2026-06
```
Then `gh pr create` titled `Follow-up audit fixes: clear the 2 launch-blockers + Major/Minor`
with a summary of the two Criticals, the two Majors, and the minors, and a test plan.

---

## Verification checklist (Definition of Done)

- [ ] Both 🔴 closed with RED→GREEN evidence (build_spatial_ic grad finite; sample_from_pmf no
      quadratic materialization + distribution-faithful).
- [ ] Both 🟠 closed (PE grad finite + FD-correct at softening=0; M3-guard test robust across seeds).
- [ ] Minors closed (init_bm19 grad-safe; energy_sorted_segregation exported; profiles/api.py
      coverage up; c(W₀) pinned; doc phantom/counts fixed; ticket extended).
- [ ] `pytest tests/ -m "not slow"` fully green; no weakened tolerances.
- [ ] One PR `hardening/followup-2026-06 → main`; per-batch commits; gates honored.
