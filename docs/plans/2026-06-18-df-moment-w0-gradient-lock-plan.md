# df_moment W₀-gradient lock — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> subagent-driven-development) to implement this plan task-by-task.

**Goal:** Lock the already-confirmed-correct ∂(σ)/∂W₀ gradient through
`df_moment_dispersion` with a grad-audit registry case + regression tests + ADR,
closing the `# W0 deferred` hole. **No `src/` change** (H0 confirmed — see the
companion design doc, §4).

**Architecture:** Pure test/registry/docs lock. The W₀ gradient is AD-correct
(experiment: rel-err 2e-11…4e-8 across W₀=5,6,7 and all channels). We add (1) two
unit regression tests, (2) one grad-audit registry W₀ case, (3) ADR-0017, then
regen the grad-audit JSON + dashboard so the staleness gates stay green.

**Tech Stack:** JAX (float64 via `import progenax`), pytest, the `jaxstro.testing`
grad-audit engine (`h_rel=1e-4`), the committed grad-audit JSON + test-dashboard.

**Verify discipline (every task):** XLA thread caps + `-n auto`:
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest ...`

---

### Task 1: Regression unit tests (W₀-gradient lock)

**Files:**
- Modify: `tests/unit/kinematics/test_dispersion.py` (add after
  `test_grad_jeans_michie_high_W0_ad_correct`, ~line 320)

**Step 1: Write the tests** (mirror the existing Michie jeans W₀ gates; valid FD
step = `_assert_ad_fd` default `h=1e-4·θ=6e-4` at W₀=6, measured rel ~1e-7).

```python
def test_grad_df_moment_michie_wrt_W0():
    """df_moment_dispersion sigma_r gradient w.r.t. W0 (exact Michie DF moment).

    The DF-moment path interpolates the potential W on a FIXED xi_grid (static
    linspace) — unlike the jeans path's moving r_t(W0)-endpoint s-grid (ADR-0016) —
    so there is NO node-crossing kink and this AD-vs-FD gate is clean on merit
    (measured rel ~1e-7 at the default valid step h=6e-4). See ADR-0017 and the
    2026-06-18 design doc: the STATUS "same-cause C0-kink" hypothesis was falsified
    by a pre-registered discriminating experiment.
    """
    from progenax import MichieVelocityDF, df_moment_dispersion

    def f(W0):
        df = MichieVelocityDF(W0=W0, r_c=1.0, r_a=5.0, xi_max=800.0, n_ode_points=3000)
        return jnp.sum(
            df_moment_dispersion(df, jnp.array([2.0, 13.0, 25.0]), 400.0, G_STELLAR).sigma_r
        )

    _assert_ad_fd(f, 6.0, name="df_moment Michie sigma_r / W0")  # measured rel ~1e-7


def test_grad_df_moment_michie_high_W0_ad_correct():
    """High-W0 df_moment gradient is AD-correct; coarse-FD disagreement is an FD artifact.

    At W0=7, r_a=5 the Michie model is near its mass-divergence (r_t ~ 545), so a
    single coarse central FD is an unreliable truth-proxy. This pins that the
    reverse-mode AD gradient is nonetheless CORRECT: a central FD CONVERGES to AD as
    the step shrinks. The gated test above runs in the well-truncated W0=6 regime
    (cf. test_grad_jeans_michie_high_W0_ad_correct for the jeans analogue).
    """
    from progenax import MichieVelocityDF, df_moment_dispersion

    def f(W0):
        df = MichieVelocityDF(W0=W0, r_c=1.0, r_a=5.0, xi_max=800.0, n_ode_points=3000)
        # interior radius 0.3*r_t(W0=7) ~ 163; use a fixed interior radius well inside
        return jnp.sum(df_moment_dispersion(df, jnp.array([20.0]), 400.0, G_STELLAR).sigma_r)

    W0 = 7.0
    g_ad = float(jax.grad(f)(W0))

    def rel(h):
        g_fd = float((f(W0 + h) - f(W0 - h)) / (2.0 * h))
        return abs(g_ad - g_fd) / (abs(g_fd) + 1e-12)

    assert abs(g_ad) > 1e-9
    # FD converges to AD as the step shrinks (truncation error, not a gradient defect)
    assert rel(1e-4) < rel(1e-2)
    assert rel(1e-4) < 1e-3
```

**Step 2: Run — expect PASS** (behavior already correct; these characterize/lock it):
`pytest tests/unit/kinematics/test_dispersion.py -k "df_moment_michie and W0 or df_moment_michie_high" -v`
Expected: 2 passed. (If the W₀=7 interior radius 20.0 is NOT well inside r_t, adjust;
r_t(7)≈545 so 20.0 is deep interior — safe.)

**Step 3: Commit**
```bash
git add tests/unit/kinematics/test_dispersion.py
git commit -m "test(dispersion): lock df_moment_dispersion W0 gradient (AD-correct, valid FD step)"
```

---

### Task 2: Grad-audit registry W₀ case

**Files:**
- Modify: `tests/validation/grad_audit/registry.py` (new fn ~after line 1540;
  new `Case` ~after line 2084)
- Modify: `tests/validation/grad_audit/manifest.py:120` (add W₀ description; the
  M-case line stays)
- Regen: `validation/data/` grad-audit JSON + `test_dashboard.json`

**Step 1: Add the registry function** (after `_df_moment_dispersion_M`, ~line 1541).
Radii span interior→near-r_t at W₀=6 (r_t≈27.9): `[2.0, 13.0, 25.0]`.

```python
# W0 gradient (NOT deferred — confirmed AD-correct by the 2026-06-18 discriminating
# experiment; fixed xi_grid nodes => no moving-node kink, unlike the jeans path).
# Build the DF inside so W0 flows through solve_michie_profile; fixed xi_max=800 keeps
# the ODE domain W0-consistent. radii [2,13,25] span interior->near-r_t (r_t~27.9 @ W0=6).
def _df_moment_dispersion_W0(W0):
    df = MichieVelocityDF(W0=W0, r_c=1.0, r_a=5.0, xi_max=800.0, n_ode_points=3000)
    return df_moment_dispersion(df, jnp.array([2.0, 13.0, 25.0]), 400.0, STELLAR.G).sigma_r
```

**Step 2: Add the Case** (after the existing `df_moment_dispersion[Michie]` M Case, ~line 2085):

```python
    # W0-gradient (the deferred axis, NOW audited): exact Michie DF-moment sigma_r is
    # AD-correct in W0 (fixed-node interp => no kink; 2026-06-18 experiment, ADR-0017).
    Case(id="df_moment_dispersion[Michie].W0", direction="params->summary",
         fn=_df_moment_dispersion_W0, param="W0", theta0=6.0,
         reduce=identity_sum, expect="consistent", tol=1e-3),       # measured |ratio-1|~1e-7
```

**Step 3: Add manifest description** (`manifest.py`, after line 120):
```python
    ("df_moment_dispersion[Michie].W0", "W0"): "Michie DF-moment sigma_r in W0 "
        "(fixed-node interp => AD-correct, no kink; closes the deferred W0 axis, ADR-0017)",
```
Also update the M-case comment `# ... W0 deferred` → `# ... W0 now audited (ADR-0017)`
at `registry.py:175` (SYMBOL_CATEGORY comment) and the design-ref comments.

**Step 4: Run the grad-audit — expect 0 hazards, +1 case:**
`env -u VIRTUAL_ENV uv run --no-sync python scripts/audit_gradients.py`
Expected: `N cases; 0 hazard(s).` with N = previous+1; the new row `|ratio-1|~1e-7`.

**Step 5: Run the freshness gates — expect RED (JSON now stale):**
`pytest tests/validation/grad_audit/test_json_fresh.py tests/validation/test_dashboard_fresh.py -v`
Expected: FAIL (committed JSON missing the new case).

**Step 6: Regenerate JSON + dashboard:**
```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/audit_gradients.py   # writes grad-audit JSON
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_test_dashboard.py --emit --render
```

**Step 7: Re-run freshness gates — expect GREEN:**
`pytest tests/validation/grad_audit/test_json_fresh.py tests/validation/test_dashboard_fresh.py tests/validation/test_dashboard_gen.py -v`
Expected: all pass.

**Step 8: Commit**
```bash
git add tests/validation/grad_audit/registry.py tests/validation/grad_audit/manifest.py \
        validation/data/   # the regenerated grad-audit JSON + test_dashboard.json (+ any dashboard md/png)
git commit -m "test(grad-audit): audit df_moment_dispersion W0 axis (closes deferred hole)"
```
(Stage the exact regenerated files by name — confirm with `git status` first; no `git add -A`.)

---

### Task 3: ADR-0017 + STATUS + final gate

**Step 1: Record the ADR** via `/adr` (ADR-0017): "df_moment_dispersion W0 gradient
is AD-correct (no kink) — same-cause hypothesis falsified". Body: the fixed-node vs
moving-node mechanism distinction; the pre-registered experiment + H0 result; lock-only
outcome; retained falsified premise (cf. ADR-0016).

**Step 2: Update `STATUS.md`** `next:`/`blocker:`/`due:` — record the arc: experiment
run, H0 (clean), locked, no src change; note the release-closeout half is next.

**Step 3: FULL released-core gate (phase/commit gate):**
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
```
Expected: prior pass count + 2 new unit tests, 0 failed; staleness/ratchet green.

**Step 4: Commit** ADR + STATUS:
```bash
git add .adr/ STATUS.md
git commit -m "docs(adr,status): ADR-0017 df_moment W0 AD-correct; STATUS arc update"
```

**Step 5: capture milestone** `brain "progenax: df_moment_dispersion W0 gradient verified AD-correct (no kink, same-cause hypothesis falsified); locked with grad-audit W0 case + regression tests, ADR-0017; no src change"`

---

## Definition of done
- 2 unit regression tests pass; grad-audit `0 hazard(s)`, case count +1.
- Both freshness gates green; dashboard regenerated.
- FULL gate green; ADR-0017 + STATUS committed.
- **No `src/progenax/` change.** Nothing merged/pushed without Anna's separate word.
