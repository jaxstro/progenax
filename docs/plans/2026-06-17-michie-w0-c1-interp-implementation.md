# Michie-W₀ C¹ Back-Interpolation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task. Each step is one action; RED→GREEN→commit. Anna is HITL — checkpoint at the
> marked points.

**Goal:** Make `jeans_dispersion` / `project_dispersion` give FD-consistent `∂σ/∂(profile param)`
for all truncated models (King/Michie/EFF/Plummer) by replacing the C⁰ piecewise-linear
back-interpolation in `_sigma_r2_from_tables` with a monotone **C¹ (PCHIP)** interpolant, and flip
the Michie-W₀ gradient xfail to a clean gate.

**Architecture:** Add one module-level differentiable PCHIP helper to
`src/progenax/kinematics/dispersion.py`; route the three `jnp.interp` calls in
`_sigma_r2_from_tables` (`rho`, `I_outward`, `F_shifted`) through it. Nothing else changes — the
grid build, the prefactors, `project_dispersion`'s integration, `df_moment_dispersion`, and
`ftable_sigma_r_isotropic` are untouched. The high-W₀ near-divergence residual is **not** a code
defect (validated: AD correct, FD→AD as h→0) — handled by gate methodology (keep Michie gate at
W₀=6), not code.

**Tech Stack:** JAX (`jnp`, reverse-mode `jax.grad`), the project's `_assert_ad_fd` AD-vs-FD gate.

**Design provenance:** `docs/plans/2026-06-17-michie-w0-tidal-crossing-gradient-design.md`
(corrected); ADR-0016 (corrected); supersedes ADR-0009 for the W₀=6 gate.

**Prototype-validated numbers (must reproduce):** PCHIP gives Michie W₀=6 grad rel
5.07e-3 → **3.48e-4**; Plummer isotropic 1.55e-6 → 1.52e-6; Dejonghe LOS oracle 2.6e-6 → 1.89e-6.

---

## Branch

`feat/michie-w0-tidal-crossing-gradient` (already off the merged main; carries the corrected
design doc). All work here. Verify locally per batch; no push/merge without Anna's separate word.

Gate commands (from `progenax/`):
```bash
# fast dispersion + grad-audit slice (inner loop):
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest \
  tests/unit/kinematics/test_dispersion.py tests/validation/test_dispersion_physics.py -q
# full released-core fast gate (phase gate):
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow" -n auto
```

---

### Task 1: Differentiable PCHIP helper

**Files:**
- Modify: `src/progenax/kinematics/dispersion.py` (add `_pchip_interp`, after `_safe_sqrt`)
- Test: `tests/unit/kinematics/test_dispersion.py` (new `TestPchipInterp`)

**Step 1 — Write failing tests.** A monotone C¹ interpolant must: (a) reproduce node values
exactly; (b) be exact on a linear function (any interpolant is); (c) be C¹ — its finite-difference
derivative has no jump across an interior node (the property linear interp lacks); (d) be
reverse-mode differentiable w.r.t. the data `y`.

```python
class TestPchipInterp:
    def test_passes_through_nodes(self):
        x = jnp.linspace(0.5, 4.0, 12); y = jnp.sin(x)
        from progenax.kinematics.dispersion import _pchip_interp
        assert jnp.allclose(_pchip_interp(x, x, y), y, atol=1e-12)

    def test_exact_on_linear(self):
        from progenax.kinematics.dispersion import _pchip_interp
        x = jnp.linspace(0.0, 10.0, 9); y = 3.0 * x - 1.0
        xq = jnp.linspace(0.3, 9.7, 50)
        assert jnp.allclose(_pchip_interp(xq, x, y), 3.0 * xq - 1.0, atol=1e-10)

    def test_c1_no_slope_jump_across_node(self):
        # central FD of the interpolant straddling an interior node must agree
        # left vs right (C1); linear interp would show a jump here.
        from progenax.kinematics.dispersion import _pchip_interp
        x = jnp.linspace(0.0, 6.0, 13); y = jnp.exp(-x)        # smooth, monotone
        node = x[6]; e = 1e-3
        d_left  = (_pchip_interp(jnp.array([node - e]), x, y) - _pchip_interp(jnp.array([node - 2*e]), x, y)) / e
        d_right = (_pchip_interp(jnp.array([node + 2*e]), x, y) - _pchip_interp(jnp.array([node + e]), x, y)) / e
        assert abs(float(d_left[0] - d_right[0])) < 1e-2 * abs(float(d_right[0]))

    def test_differentiable_in_data(self):
        from progenax.kinematics.dispersion import _pchip_interp
        x = jnp.linspace(0.5, 4.0, 10)
        def loss(scale):
            return jnp.sum(_pchip_interp(jnp.array([1.3, 2.7]), x, scale * jnp.cos(x)))
        g = float(jax.grad(loss)(1.0)); assert abs(g) > 1e-9 and jnp.isfinite(g)
```

**Step 2 — Run, verify RED** (`_pchip_interp` undefined): `pytest tests/unit/kinematics/test_dispersion.py::TestPchipInterp -q` → ImportError/fail.

**Step 3 — Implement** (validated Fritsch-Carlson; monotone `x` grid, possibly non-uniform — works
for both the King/EFF uniform `s` and the Plummer compactified `s`):

```python
def _pchip_interp(xq, x, y):
    """Monotone C¹ (PCHIP / Fritsch-Carlson) interpolation; reverse-mode differentiable.

    Replaces a piecewise-linear ``jnp.interp`` where a C⁰ slope-jump at a node would,
    on a grid whose nodes move with a differentiated parameter (e.g. r_t(W₀)), inject a
    bracket-crossing kink into the parameter gradient. C¹ removes that jump. The abscissa
    coordinate is irrelevant (``jnp.interp`` is scale-equivariant); only smoothness matters.
    ``x`` must be monotone increasing (the master ``s``-grid is).
    """
    n = x.shape[0]
    h = jnp.diff(x)
    d = jnp.diff(y) / h  # secant slopes
    # interior Fritsch-Carlson weighted-harmonic-mean slopes (0 at a sign change / extremum)
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    dprev, dnext = d[:-1], d[1:]
    same = dprev * dnext > 0.0
    denom = w1 / jnp.where(dprev == 0.0, 1.0, dprev) + w2 / jnp.where(dnext == 0.0, 1.0, dnext)
    m_int = jnp.where(same, (w1 + w2) / jnp.where(same, denom, 1.0), 0.0)
    m = jnp.concatenate([d[:1], m_int, d[-1:]])  # one-sided endpoint slopes
    # Hermite evaluation on the bracketing interval
    idx = jnp.clip(jnp.searchsorted(x, xq, side="right") - 1, 0, n - 2)
    x0 = x[idx]; x1 = x[idx + 1]; y0 = y[idx]; y1 = y[idx + 1]; m0 = m[idx]; m1 = m[idx + 1]
    hh = x1 - x0
    t = (xq - x0) / hh
    t2 = t * t; t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * hh * m0 + h01 * y1 + h11 * hh * m1
```

**Step 4 — Run, verify GREEN.** **Step 5 — Commit** (`feat(dispersion): add differentiable PCHIP
C¹ interpolation helper`).

---

### Task 2: Route `_sigma_r2_from_tables` through PCHIP + flip the Michie-W₀ gate

**Files:**
- Modify: `src/progenax/kinematics/dispersion.py` (`_sigma_r2_from_tables`, lines ~321–324)
- Modify: `tests/unit/kinematics/test_dispersion.py` (un-xfail `test_grad_jeans_michie_wrt_W0_DEFERRED`)

**Step 1 — RED: un-xfail the gate.** Remove the `@pytest.mark.xfail(...)` decorator from
`test_grad_jeans_michie_wrt_W0_DEFERRED`, rename it `test_grad_jeans_michie_wrt_W0`, drop
"DEFERRED" from the docstring, update the trailing comment to `# measured rel ~3.5e-4 < 1e-3`. Run
→ it FAILS (current linear interp gives 5.07e-3).

**Step 2 — GREEN: swap the three interps.** In `_sigma_r2_from_tables`:
```python
    rho_r = _pchip_interp(r, s, rho)
    I_r = _pchip_interp(r, s, I_outward)
    if F_shifted is not None:
        prefactor = jnp.exp(-_pchip_interp(r, s, F_shifted))
```
(leave the `r_a`/isotropic prefactor branches unchanged.) Update the function docstring: the
back-interpolation is now C¹ (PCHIP), replacing the C⁰ `jnp.interp` whose node-crossing slope jump
kinked `∂σ/∂(param)`; cite ADR-0016.

**Step 3 — Run** the dispersion gate slice → Michie-W₀ test GREEN; the King-W₀ and EFF gradient
gates stay GREEN.

**Step 4 — Commit** (`fix(dispersion): C¹ PCHIP back-interp removes the W₀ gradient kink (ADR-0016)`).

**🔵 HITL CHECKPOINT 1** — report the before/after gate numbers to Anna before continuing.

---

### Task 3: Re-validate every physics anchor (must pass unchanged)

**Files:** `tests/validation/test_dispersion_physics.py` (assert; add explicit guards only if missing)

**Step 1 — Run the full dispersion validation file.** Expected GREEN unchanged:
- `test_plummer_isotropic_jeans_vs_analytic` (rtol 1e-3) — prototype 1.52e-6.
- `test_plummer_om_jeans_vs_analytic` (the OM oracle, rtol 1e-3 at r_a∈{1,2,5}) — the Task-#3 oracle.
- the Dejonghe `project_dispersion` LOS oracle — prototype 1.89e-6.
- King c(W₀) Table II (`test_king_physics.py`, ≤0.002) — run it too.

**Step 2 — If any anchor moved meaningfully, STOP and report to Anna** (do not weaken a tolerance).
Expected: no change. **Step 3 — Commit** only if assertions were added/tightened.

---

### Task 4: Document the near-divergence regime (methodology, not code)

**Files:**
- Modify: `src/progenax/kinematics/dispersion.py` (`jeans_dispersion` docstring "Notes")
- Test: `tests/unit/kinematics/test_dispersion.py` (one Richardson-FD regression test)

**Step 1 — RED/GREEN: a Richardson-FD test proving AD is correct near the divergence.** Add
`test_grad_jeans_michie_high_W0_ad_correct`: at W₀=7, r_a=5, assert the AD gradient matches a
*converged* central FD (sweep h, take the small-h limit / Richardson) to <1e-3, documenting that
the coarse-step inconsistency is an FD truth-proxy artifact (r_t≈545, near mass-divergence), not a
gradient defect. (Validated this session: FD→AD, 3.2e-3 @1e-3 → 2.1e-5 @1e-6.)

**Step 2 — Docstring note** in `jeans_dispersion`: beyond W₀≈7 (r_a=5) the Michie model approaches
mass-divergence (r_t→∞); the gradient remains correct but a fixed-step FD is a poor check there.

**Step 3 — Commit** (`test(dispersion): Richardson-FD proof that high-W₀ Michie gradient is correct`).

---

### Task 5: Reconcile the grad-audit gate + manifest

**Files:**
- `tests/validation/grad_audit/` (regenerate the audit JSON; `test_json_fresh.py` must pass)
- `tests/validation/grad_audit/manifest.py` (only if an entry's status legitimately changes)

**Step 1 — Regenerate** the grad-audit and run `test_json_fresh.py`. **Step 2 — Reconcile:** the
dispersion Michie-W₀ case is the `test_dispersion.py` xfail (now flipped) — confirm no manifest
`PARAM_ALLOWLIST` / known-limitation entry references it (grep `michie.*W0` in the gradient-audit
sources); the profile-level `MichieProfile.r_t/W0` MUST_AUDIT entries are unaffected (already
clean). Remove any now-stale "deferred / xfail" wording that pointed at the dispersion case.
**Step 3 — Commit** if the manifest changed.

**🔵 HITL CHECKPOINT 2** — full released-core fast gate green; report counts (xfail 1→0,
skip unchanged) before doc/status updates.

---

### Task 6: Docs, STATUS, ADR-0009 status (Anna's call)

**Files:** `STATUS.md`; `docs/website/50-validation/michie-anisotropy.md` and/or
`differentiability-audit.md` (note the W₀ gradient is now clean + the near-divergence caveat);
`docs/website/95-release/checklist.md` (close the #4 line). ADR-0009 status flip
accepted→superseded is **Anna's decision** (decided_by: user) — propose, don't flip.

**Step — Commit** (`docs: Michie-W₀ gradient clean (C¹ interp); close #4`). Capture to brain.

---

## Execution handoff
After Anna approves this plan: **subagent-driven** (fresh subagent per task, code review between
tasks, Anna HITL at the two checkpoints) is the recommended mode — shared-infra change with tight
anchors warrants per-task review. Alternatively a parallel session with executing-plans.
