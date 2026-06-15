# ZAMS Stellar Relations Migration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task.

**Goal:** Internalize the Tout+1996 ZAMS family (M→L,R,T_eff,log g + inverse L→M) into `src/progenax/stellar.py`, decoupling the B4 demo from the private `fluxax` sibling, with full registry + validation coverage.

**Architecture:** Five pure, array-aware, differentiable functions, refactored from `fluxax/photometry/stellar_physics.py` to progenax idiom (jaxtyping, caller-jittable, array-in/array-out). Coefficients are named frozen constants verified cell-by-cell against the held Tout+1996 PDF. Additive only — `compute_stellar_radii` (D&K91 collision radii) is untouched. Five new `__all__` symbols flow through all four test-backbone registries.

**Tech Stack:** JAX (`jax.numpy`, `jax.lax.scan`, `jax.grad`, `jax.vmap`), jaxtyping, `jaxstro.constants`, pytest, the four `tests/validation/*_registry|*_coverage` manifests, `scripts/build_test_dashboard.py`.

**Design doc:** `docs/plans/2026-06-14-zams-stellar-migration-design.md` (5 ratified decisions).

**Branch:** `feat/zams-stellar` (already cut, P0 done). All work LOCAL; nothing pushed/merged without Anna.

**Conventions:** progenax auto-enables float64 on import. Run pytest with
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest …`. Every commit message ends with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Phase P1 — Verify Tout coefficients vs the held PDF (paper-grounding first)

> **REQUIRED SUB-SKILL:** Use the paper-grounding workflow — read the actual PDF, never assert coefficients from memory or from fluxax alone.

### Task P1.1: Verify Table 1 (L) and Table 2 (R) cell-by-cell

**Files:**
- Read: `/Users/anna/Desktop/Tout1996-ZAMS-Formulae.pdf` (held; fallback copies in `~/Dropbox/.../Tout1996_ZAMS.pdf`)
- Compare against: `/Users/anna/projects/jaxstro-dev/fluxax/src/fluxax/photometry/stellar_physics.py` (lines 67–217 — the `_TOUT` coefficient blocks)
- Create: `docs/core-papers/tout1996_zams_coefficients_verified.md`

**Step 1:** Read Tout+1996 §Tables (the L(M,Z) coefficients α,β,γ,δ,ε,ζ,η and the R(M,Z) coefficients θ,ι,κ,λ,μ,ν,ξ,ο,π). Each (except scalar ν) is a degree-4 polynomial in `log10(Z/Z☉)` with 5 sub-coefficients.

**Step 2:** Tabulate the PDF values next to fluxax's transcribed values. Mark each cell MATCH / MISMATCH. fluxax's values are the candidate; the **PDF is ground truth**. If any cell mismatches, the PDF wins — record the correction.

**Step 3:** Write `tout1996_zams_coefficients_verified.md` mirroring `docs/core-papers/startrax_hurley2000_coefficients_verified.md`: a header citing Tout+1996 (MNRAS 281, 257), the full verified Table 1 + Table 2 with a per-cell MATCH ledger, the Sun-anchor sanity values (L≈0.69 L☉, R≈0.89 R☉, T_eff≈5670 K), and a note on the ~5% MS accuracy.

**Step 4: Commit**
```bash
git add docs/core-papers/tout1996_zams_coefficients_verified.md
git commit -m "docs(provenance): Tout+1996 ZAMS Table 1/2 coefficients verified cell-by-cell vs held PDF"
```

**Report:** the MATCH ledger, and ANY mismatch found (these become corrections in P2, not fluxax copies).

---

## Phase P2 — Implement `stellar.py` + unit tests (TDD)

> **REQUIRED SUB-SKILL:** Use test-driven-development (RED→GREEN per function).

### Task P2.1: Module skeleton + coefficient constants

**Files:**
- Create: `src/progenax/stellar.py`

**Step 1:** Write the module header + the PDF-verified coefficient constants (use the **P1-verified** numbers, not a blind fluxax copy). Forward functions are elementwise over `mass`; `Z` is scalar (default solar).

```python
"""Tout (1996) ZAMS stellar-structure relations (M -> L, R, T_eff, log g) + inverse.

A differentiable, metallicity-dependent placeholder for the eventual ``startrax``
stellar tracks. Distinct from :func:`progenax.compute_stellar_radii` (Demircan &
Kahraman 1991 empirical *collision* radii for N-body); these are *photometric* ZAMS
relations for CMD / mass-function science.

Reference: Tout et al. (1996), MNRAS 281, 257 (Tables 1 & 2). Coefficients verified
cell-by-cell vs the held PDF — see docs/core-papers/tout1996_zams_coefficients_verified.md.
Valid: 0.1 <= M/Msun <= ~125, 1e-4 <= Z <= 0.03, ~5% MS accuracy.
"""
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jaxstro.constants import LSUN_ERG_S, RSUN_CM, SIGMA_SB, G_CGS, MSUN_G

__all__ = [
    "zams_luminosity",
    "zams_radius",
    "zams_effective_temperature",
    "zams_surface_gravity",
    "inverse_zams_luminosity",
]

_Z_SUN = 0.02  # Tout+1996 reference solar metallicity

# Tout+1996 Table 1 — L(M,Z). Row r = degree-4 polynomial [c0..c4] in log10(Z/Zsun)
# for coefficient (alpha, beta, gamma, delta, epsilon, zeta, eta). PDF-verified (P1).
_TOUT_L_COEFFS = jnp.array([
    [ 0.39704170, -0.32913574,  0.34776688,  0.37470851,  0.09011915],  # alpha
    [ 8.52762600,-24.41225973, 56.43597107, 37.06152575,  5.45624060],  # beta
    [ 0.00025546, -0.00123461, -0.00023246,  0.00045519,  0.00016176],  # gamma
    [ 5.43288900, -8.62157806, 13.44202049, 14.51584135,  3.39793084],  # delta
    [ 5.56357900,-10.32345224, 19.44322980, 18.97361347,  4.16903097],  # epsilon
    [ 0.78866060, -2.90870942,  6.54713531,  4.05606657,  0.53287322],  # zeta
    [ 0.00586685, -0.01704237,  0.03872348,  0.02570041,  0.00383376],  # eta
])

# Tout+1996 Table 2 — R(M,Z). Coefficients (theta, iota, kappa, lambda, mu, xi,
# omicron, pi); nu is a Z-independent scalar. PDF-verified (P1).
_TOUT_R_COEFFS = jnp.array([
    [ 1.71535900,  0.62246212, -0.92557761, -1.16996966, -0.30631491],  # theta
    [ 6.59778800, -0.42450044,-12.13339427,-10.73509484, -2.51487077],  # iota
    [10.08855000, -7.11727086,-31.67119479,-24.24848322, -5.33608972],  # kappa
    [ 1.01249500,  0.32699690, -0.00923418, -0.03876858, -0.00412750],  # lambda
    [ 0.07490166,  0.02410413,  0.07233664,  0.03040467,  0.00197741],  # mu
    [ 3.08223400,  0.94472050, -2.15200882, -2.49219496, -0.63848738],  # xi
    [17.84778000, -7.45345690,-48.96066856,-40.05386135, -9.09331816],  # omicron
    [ 0.00022582, -0.00186899,  0.00388783,  0.00142402, -0.00007671],  # pi
])
_TOUT_R_NU = 0.01077422  # Table 2, Z-independent

_INVERSE_NEWTON_ITERS = 20  # quadratic convergence; ~6-8 reach machine precision


def _metallicity_coeffs(coeff_matrix: Float[Array, "n 5"], Z: float) -> Float[Array, "n"]:
    """Evaluate each row's degree-4 log10(Z/Zsun) polynomial -> per-coefficient scalars."""
    log_Z = jnp.log10(jnp.clip(Z, 1e-4, 0.03) / _Z_SUN)
    basis = log_Z ** jnp.arange(5)            # (5,)
    return coeff_matrix @ basis               # (n,)
```

**Step 2: Commit** (skeleton compiles).
```bash
git add src/progenax/stellar.py && git commit -m "feat(stellar): Tout+1996 coefficient constants + metallicity helper (PDF-verified)"
```

### Task P2.2: `zams_luminosity` (TDD)

**Files:** Modify `src/progenax/stellar.py`; Create `tests/unit/stellar/__init__.py` + `tests/unit/stellar/test_zams.py`.

**Step 1: Write failing test** (`tests/unit/stellar/test_zams.py`):
```python
import jax, jax.numpy as jnp, pytest
import progenax  # float64 on
from progenax.stellar import zams_luminosity

class TestZAMSLuminosity:
    def test_sun_anchor(self):
        # Tout+1996 Sun ZAMS L ~ 0.69 Lsun (verified P1)
        assert zams_luminosity(jnp.array(1.0)) == pytest.approx(0.69, abs=0.05)
    def test_broadcasts_over_array(self):
        L = zams_luminosity(jnp.array([0.5, 1.0, 10.0]))
        assert L.shape == (3,)
        assert jnp.all(L[1:] > L[:-1])  # monotone increasing
    def test_differentiable(self):
        g = jax.grad(lambda m: zams_luminosity(m))(jnp.array(1.0))
        assert jnp.isfinite(g) and g > 0
```

**Step 2: Run → FAIL** (`zams_luminosity` body missing).
Run: `... pytest tests/unit/stellar/test_zams.py::TestZAMSLuminosity -v`  Expected: FAIL.

**Step 3: Implement** (append to `stellar.py`):
```python
def zams_luminosity(mass: Float[Array, "..."], Z: float = 0.02) -> Float[Array, "..."]:
    """ZAMS luminosity [L_sun] from mass [M_sun] and metallicity Z. Tout+1996 Table 1."""
    M = jnp.asarray(mass, float)
    a, b, g, d, e, z, h = _metallicity_coeffs(_TOUT_L_COEFFS, Z)
    num = a * M**5.5 + b * M**11
    den = g + M**3 + d * M**5 + e * M**7 + z * M**8 + h * M**9.5
    return num / jnp.maximum(den, 1e-10)
```

**Step 4: Run → PASS.**  **Step 5: Commit** `feat(stellar): zams_luminosity (Tout+1996 Table 1, array-aware, differentiable)`.

### Task P2.3: `zams_radius` (TDD)
Same pattern. Test: Sun → R≈0.89 R☉ (abs 0.05); array broadcast + (R rises then is non-trivial — assert finite/positive, R(10)≈ a few R☉); differentiable.
```python
def zams_radius(mass, Z=0.02):
    M = jnp.asarray(mass, float)
    th, io, ka, la, mu, xi, om, pi_ = _metallicity_coeffs(_TOUT_R_COEFFS, Z)
    num = th*M**2.5 + io*M**6.5 + ka*M**11 + la*M**19 + mu*M**19.5
    den = _TOUT_R_NU + xi*M**2 + om*M**8.5 + M**18.5 + pi_*M**19.5
    return num / jnp.maximum(den, 1e-10)
```
Commit `feat(stellar): zams_radius (Tout+1996 Table 2)`.

### Task P2.4: `zams_effective_temperature` + `zams_surface_gravity` (TDD)
Tests: Sun T_eff≈5670 K (abs 150); Sun log g≈4.5 (abs 0.1); both differentiable + array-aware.
```python
def zams_effective_temperature(mass, Z=0.02):
    L = zams_luminosity(mass, Z) * LSUN_ERG_S
    R = zams_radius(mass, Z) * RSUN_CM
    return (L / (4.0 * jnp.pi * R**2 * SIGMA_SB)) ** 0.25

def zams_surface_gravity(mass, Z=0.02):
    M_cgs = jnp.asarray(mass, float) * MSUN_G
    R_cgs = zams_radius(mass, Z) * RSUN_CM
    return jnp.log10(G_CGS * M_cgs / R_cgs**2)
```
Commit `feat(stellar): zams_effective_temperature + zams_surface_gravity (Stefan-Boltzmann / g=GM/R^2)`.

### Task P2.5: `inverse_zams_luminosity` (TDD — the differentiable Newton invert)

**Step 1: Write failing tests:**
```python
class TestInverseZAMSLuminosity:
    def test_round_trip(self):
        from progenax.stellar import inverse_zams_luminosity, zams_luminosity
        m = jnp.array([0.5, 1.0, 5.0, 20.0])
        m_rec = inverse_zams_luminosity(zams_luminosity(m))
        assert jnp.allclose(m_rec, m, rtol=1e-5)
    def test_differentiable(self):
        from progenax.stellar import inverse_zams_luminosity
        g = jax.grad(lambda L: inverse_zams_luminosity(L)[0])(jnp.array([100.0]))
        assert jnp.isfinite(g)
```

**Step 3: Implement** (scalar Newton core + internal vmap; the scalar `zams_luminosity` IS scalar-output so `jax.grad` works directly):
```python
def _inverse_scalar(L_target, Z):
    L_safe = jnp.clip(L_target, 1e-15, 1e8)
    m0 = jnp.clip(jnp.where(L_safe < 1e-3, L_safe**(1/5.5), L_safe**(1/3.5)), 0.005, 125.0)
    dLdm = jax.grad(lambda m: zams_luminosity(m, Z))
    def step(m, _):
        resid = zams_luminosity(m, Z) - L_target
        slope = dLdm(m)
        m_new = m - resid / jnp.where(jnp.abs(slope) > 1e-30, slope, 1e-30)
        return jnp.clip(m_new, 0.005, 150.0), None
    m_final, _ = jax.lax.scan(step, m0, None, length=_INVERSE_NEWTON_ITERS)
    return m_final

def inverse_zams_luminosity(L_target: Float[Array, "..."], Z: float = 0.02) -> Float[Array, "..."]:
    """Invert L(M,Z): find M [M_sun] s.t. zams_luminosity(M,Z)=L_target. Differentiable."""
    L = jnp.atleast_1d(jnp.asarray(L_target, float))
    out = jax.vmap(lambda Lt: _inverse_scalar(Lt, Z))(L)
    return out if jnp.ndim(L_target) else out[0]
```

**Step 4: PASS.**  **Step 5: Commit** `feat(stellar): inverse_zams_luminosity (differentiable Newton/scan invert)`.

### Task P2.6: Wire into the public API + cross-reference `compute_stellar_radii`

**Files:** Modify `src/progenax/__init__.py` (import the 5 symbols + add to `__all__`); modify `src/progenax/builders.py` (the `compute_stellar_radii` docstring) + `src/progenax/stellar.py` (module docstring) with reciprocal one-line cross-references.

**Step 1:** Add to `progenax/__init__.py` import block + `__all__` (mirror the existing alphabetised export style).
**Step 2: Test** — `python -c "import progenax; print(progenax.zams_luminosity, progenax.inverse_zams_luminosity)"`.
**Step 3:** Add to `compute_stellar_radii` docstring: "See :func:`progenax.zams_radius` for Tout+1996 *photometric* ZAMS radii (this function is the D&K91 *collision* radius)."
**Step 4: Commit** `feat(stellar): export ZAMS relations from progenax + cross-ref compute_stellar_radii`.

### Task P2.7: Extra unit tests (valid-range, Z-direction, vmap, jit)
Add tests: Z-dependence direction (lower Z → different L at fixed M, finite), `jax.jit(zams_luminosity)` compiles, `jax.vmap` over Z, brown-dwarf/low-mass returns finite positive, high-mass (100 M☉) finite. Commit `test(stellar): valid-range, metallicity, jit/vmap coverage`.

---

## Phase P3 — Register across the 4 registries + dashboard regen

> The five new `__all__` symbols MUST be registered or the test-backbone gate reds. Mirror the EXACT existing literal format in each manifest (open the file, copy a sibling entry's shape).

### Task P3.1: API-coverage registry
**File:** `tests/validation/api_coverage/manifest.py` — add 5 entries to `SYMBOL_TESTS` (symbol → `"tests/unit/stellar/test_zams.py::Class::test"` + a `# assert …` comment naming the asserting check). Point each at the P2 asserting test (e.g. `"zams_luminosity": "tests/unit/stellar/test_zams.py::TestZAMSLuminosity::test_sun_anchor",  # assert L(1 Msun)~0.69`). `UNTESTED` stays `{}`.
**Run:** `... pytest tests/validation/api_coverage/ -q` → PASS (114→119 symbols mapped). Commit.

### Task P3.2: Physics-validation registry
**File:** `tests/validation/physics_registry/manifest.py` — add 5 entries to `EXEMPT_NON_MODEL` (symbol → reason `"Stellar mass-relation (Tout+1996), not an equilibrium model; validated in test_zams_physics.py against published anchors."`). Confirm the operational `IS_MODEL` ratchet agrees (these are not `SpatialProfile`/`VelocityDF`/`IMFProtocol`/`build_*_cluster`).
**Run:** `... pytest tests/validation/physics_registry/ -q` → PASS (90→95 exempt). Commit.

### Task P3.3: Provenance registry
**File:** `tests/validation/provenance_registry/manifest.py` — (a) add `"stellar.py"` to `ALLOWLIST_MODULES`; (b) add 2 `PROVENANCE` entries: `"stellar.py::_TOUT_L_COEFFS (Tout+1996 Table 1, L(M,Z))"` and `"stellar.py::_TOUT_R_COEFFS + _TOUT_R_NU (Tout+1996 Table 2, R(M,Z))"` → citations referencing `tout1996_zams_coefficients_verified.md`; (c) add `_INVERSE_NEWTON_ITERS`, the `1e-10`/`1e-15`/`1e-30` guards, `0.005/125/150` clip bounds, and the `5.5/11/…` Tout *exponents* to `ALLOWLIST_NON_COEFFICIENT` for `stellar.py` (numerical-method / formula-structure literals, NOT citable coefficients) — each with a reason. `UNPROVENANCED` stays `{}`.
**Run:** `... pytest tests/validation/provenance_registry/ -q` → PASS (29→31 constants, 0 unprovenanced). Commit. (If the new-literal scanner flags an exponent/guard, add it to `ALLOWLIST_NON_COEFFICIENT` with a reason — do NOT fabricate a citation.)

### Task P3.4: Differentiability grad-audit registry
**Files:** `tests/validation/grad_audit/manifest.py` (+5 `SYMBOL_CATEGORY` = `AUDITED`; +5 `MUST_AUDIT` `(id, param)` rows), `tests/validation/grad_audit/registry.py` (+5 Cases — mirror an existing `_eff_*`/`_build_spatial_ic_*` probe: `_zams_luminosity_mass`, `_zams_radius_mass`, `_zams_teff_mass`, `_zams_logg_mass`, `_inverse_zams_L`; the inverse case audits ∂M/∂L through the scan).
**Step:** regenerate the JSON: `... python scripts/audit_gradients.py` (writes `validation/data/grad_audit_results.json`).
**Run:** `... pytest tests/validation/grad_audit/ -q` → PASS (registry grows by 5, 0 hazards; staleness gate green). Commit `feat(grad-audit): 5 ZAMS AD-vs-FD cases (incl. Newton-inverse dM/dL)`.

### Task P3.5: Regenerate the dashboard
**Step:** `... python scripts/build_test_dashboard.py --emit --render`; then `... pytest tests/validation/test_dashboard_gen.py tests/validation/test_dashboard_fresh.py -q`. (Note: coverage will now be src-STALE because `stellar.py` is new — that is expected and is refreshed in P6; the staleness gate's src-based check will flag it. If the gate hard-fails on src-staleness here, defer the JSON coverage block until P6's `--cov` and re-emit then.) Commit the regenerated `test_dashboard.json` + page.

---

## Phase P4 — Validation tier (DoD scaffolding)

### Task P4.1: `tests/validation/test_zams_physics.py`
Assert against Tout published anchors (from P1): Sun L/R/T_eff/log g within the paper's stated tolerance; monotonic L(M), positive R(M); inverse round-trip rtol 1e-5 over M∈[0.1,100]; the ~5% accuracy claim at 2–3 mass points if the PDF gives comparison data. Run → PASS. Commit.

### Task P4.2: `scripts/validate_zams.py` + plots
Standalone CLI (mirror `scripts/validate_tidal.py` structure): print an expected-vs-measured table (Sun + a mass grid), pass/fail with thresholds, and save plots to `validation/plots/zams_*.png` — L–M, R–M, T_eff–M, an HR diagram (log L vs log T_eff), and inverse-residual. Exit 0 on all-pass. Run it. Commit (PNGs are gitignored; the script regenerates deterministically).

### Task P4.3: `docs/website/50-validation/zams-relations.md` + nav
New MyST page (use the `myst-expert` skill): the relations, the Tout citation, the verified-coefficients pointer, the anchor table, embedded figures. Wire into `docs/website/myst.yml` under Validation. `cd docs/website && make build` → 0 new warnings. Commit.

---

## Phase P5 — Decouple the B4 demo + docs

### Task P5.1: Rewire the demo
**File:** `scripts/demo_binary_mass_function.py` — replace `from fluxax.photometry import inverse_zams_luminosity, zams_luminosity` (lines ~81) with `from progenax.stellar import inverse_zams_luminosity, zams_luminosity`; delete the `importorskip`/try-except guard + the `uv pip install -e ../fluxax` instructions (lines ~23–25, ~81–85). Run the demo (`... python scripts/demo_binary_mass_function.py`) → exit 0, no fluxax needed. Commit `refactor(demo): B4 uses progenax.stellar ZAMS — fluxax dependency dropped`.

### Task P5.2: Docs decoupling + ecosystem boundary
**Files:** `docs/website/60-science-demos/binary-mass-function.md`, `docs/website/60-science-demos/index.md` (drop "needs fluxax" / install line), `pyproject.toml` (update the fluxax NOTE comment → "ZAMS internalized into progenax.stellar; fluxax no longer needed for B4"), and (light touch) `00-getting-started/index.md` / `units-policy.md` if they imply fluxax is required for ZAMS. Add the honest boundary note: progenax self-contains ZAMS *structure* (startrax placeholder); fluxax remains the *photometry* package. `make build` → 0 warnings. Commit.

---

## Phase P6 — Close-out

### Task P6.1: Coverage refresh (src changed → mandatory)
`stellar.py` is new src, so coverage is genuinely stale. Run the FULL suite with cov (~14 min, doubles as the FULL gate):
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync \
  pytest tests/unit tests/integration tests/validation --cov=progenax --cov-report=json:/tmp/zams_cov.json -q -n auto
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_test_dashboard.py --stamp-coverage /tmp/zams_cov.json
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_test_dashboard.py --emit --render
```
Confirm: FULL suite green; new `stellar.py` line-cov pulls total to ≥ 90 floor; `registries_full` stays True. Run `... pytest tests/validation/test_dashboard_fresh.py -q` → PASS. Commit the refreshed `coverage.json` + `test_dashboard.json` + page.

### Task P6.2: Completion doc + STATUS + brain
Write `.claude-work/TASK_zams_migration_COMPLETE.md` (gitignored local artifact): files, API, the PDF-verification ledger, registry deltas (api 114→119, physics 95 exempt, provenance 29→31, grad-audit +5), B4-decoupling evidence, final gate numbers. Update `STATUS.md` `next:` line. `brain "…"`. Commit STATUS.

### Task P6.3: CHECKPOINT → Anna
Present the release-gate status + B4-decoupled evidence. **Merge `feat/zams-stellar` → local main on Anna's explicit go (fast-forward; push only on her word).**

---

## Definition of Complete (per CLAUDE.md)
- [ ] `stellar.py` — 5 functions, PDF-verified coefficients, array-aware, differentiable.
- [ ] Unit tests (`tests/unit/stellar/`) + validation tests (`tests/validation/test_zams_physics.py`) 100% passing.
- [ ] `scripts/validate_zams.py` + `validation/plots/zams_*.png` (expected-vs-measured, pass/fail).
- [ ] All 4 registries updated; `registries_full` True; dashboard fresh; line-cov ≥ 90.
- [ ] B4 demo runs with NO fluxax; docs decoupled.
- [ ] `tout1996_zams_coefficients_verified.md` + `50-validation/zams-relations.md`.
- [ ] FULL gate green; completion doc; STATUS/brain updated.
