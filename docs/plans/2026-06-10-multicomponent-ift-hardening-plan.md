# Multicomponent IFT Hardening + B2 Speed — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> to implement this plan task-by-task (fresh subagent per task + code review).

**Goal:** Replace the unrolled 30-step eigenvalue scan in
`find_alpha_for_masses` with an optimistix implicit-function-theorem solve
(exact, cheap gradient), profile-trim `n_iter`, and right-size the B2 demo
optimizer — so the B2 joint MLE runs ≫5× faster while recovering the verified
reference θ̂ and keeping the 1163-test released-core invariant green.

**Architecture:** `find_alpha_for_masses` is a fixed-point solve
`α ← normalize(α·√(f_target/f_real(α)))` whose `f_real` solves the coupled
Poisson ODE. Today it is a `lax.scan` unrolled through the backward pass
(≈30 ODE solves each way). Refactor to `optimistix.fixed_point` with
`FixedPointIteration` (forward iterates identical) + `ImplicitAdjoint`
(backward = one rank-deficient linear solve via `lineax`, the exact
fixed-point gradient). Design doc:
`docs/plans/2026-06-10-multicomponent-ift-hardening-design.md`.

**Tech stack:** JAX, equinox, **optimistix** (new core dep; same ecosystem as
diffrax), **lineax** (transitive, for the adjoint linear solve), optax (demo MLE).

**Reference (the accuracy anchor — every gate validates against this):** at full
accuracy the B2 MLE recovers α̂=2.2931±0.0041, δ̂=0.3972±0.0344, Ŵ₀=4.9900±0.0209
(all 3σ PASS); `find_alpha_for_masses` residual `<2e-3`.

**Gates:** released-core FAST gate (1121 not-slow) + FULL gate (1163) as in
`progenax/CLAUDE.md`. **Git:** continue on branch `feat/batch-b-demos`; commit
per task; NO push/merge without Anna's go.

**Hard rules:** JAX-native; TDD; the gradient-match gate (FD ~1e-5) and the
forward-regression pin (rtol≤1e-8) are REAL — never weaken. optimistix
`throw=False` (jit/scan-safe). Do not change the physics (N_COMP=4, the √-map).

---

## Task H0: Add optimistix as a core dependency

**Files:** Modify `pyproject.toml` (core `dependencies`).

**Step 1:** Confirm optimistix + lineax import in the env (already installed at
optimistix 0.1.0):
```bash
env -u VIRTUAL_ENV uv run --no-sync python -c "import optimistix, lineax; print(optimistix.__version__, lineax.__version__)"
```
Expected: prints versions, no error.

**Step 2:** Add to `pyproject.toml` core `dependencies` (NOT an extra — it's now
released-core), pinned compatibly, e.g. `"optimistix>=0.1,<1"`. (lineax is a
transitive optimistix dep; add it explicitly too: `"lineax>=0.0.5"`, matching
the installed version — check `uv pip show lineax`.)

**Step 3:** Re-lock and verify the env still imports progenax:
```bash
env -u VIRTUAL_ENV uv run --no-sync python -c "import progenax; print('ok')"
```
Expected: `ok`.

**Step 4: Commit**
```bash
git add pyproject.toml uv.lock
git commit -m "build(deps): add optimistix + lineax as core deps (IFT solver for find_alpha_for_masses)"
```

---

## Task H1: Pin the reference behavior (regression + gradient tests, on CURRENT code)

TDD here is **characterization-first**: write tests that capture the current
(correct) forward value and gradient, confirm they pass on the *existing*
unrolled implementation, so the H2 refactor must preserve them.

**Files:** Create `tests/unit/profiles/test_find_alpha_ift.py`.

**Step 1: Write the tests.**
```python
"""Forward-regression + gradient-correctness pins for find_alpha_for_masses.

Written against the CURRENT unrolled implementation so the optimistix/IFT
refactor (H2) must preserve the forward value and match finite-difference
gradients. The IFT gradient is the EXACT fixed-point gradient; the current
unrolled gradient approximates it to ~residual, so we gate the refactored
gradient against CENTRAL FINITE DIFFERENCES (the ground truth), not against
the old unrolled gradient.
"""
import jax
import jax.numpy as jnp
import numpy as np
import progenax  # noqa: F401 (float64)
from progenax.profiles.limepy_multimass import find_alpha_for_masses, _bin_imf
from progenax.imf.smooth import Maschberger


def _alpha_of(alpha_imf, W0, g, delta, n_iter=30):
    imf = Maschberger(alpha=alpha_imf, m_min=0.1, m_max=20.0)
    m_j, M_j = _bin_imf(imf, 4, (0.1, 20.0))
    a, res = find_alpha_for_masses(m_j, M_j, W0, g, delta, n_iter=n_iter)
    return a, res, m_j, M_j


class TestForwardRegression:
    def test_alpha_sums_to_one_and_residual_small(self):
        a, res, _, _ = _alpha_of(2.3, 5.0, 1.0, 0.4)
        assert abs(float(jnp.sum(a)) - 1.0) < 1e-9
        assert float(res) < 2e-3
        assert bool(jnp.all(a > 0))

    def test_alpha_value_pinned(self):
        # Pin the converged alpha_j to a stored reference (fill in the printed
        # values on first run; this is the forward bit-identity gate for H2).
        a, _, _, _ = _alpha_of(2.3, 5.0, 1.0, 0.4)
        ref = np.array(REF_ALPHA)  # <-- set REF_ALPHA from the first run's print
        np.testing.assert_allclose(np.asarray(a), ref, rtol=1e-8, atol=1e-10)


class TestGradientMatchesFiniteDifference:
    def test_grad_wrt_alpha_imf(self):
        # d(sum |alpha_j|^2)/d(alpha_imf) via AD vs central FD. A scalar reduction
        # of alpha_j(alpha_imf) exercises the full implicit chain.
        def loss(alpha_imf):
            a, _, _, _ = _alpha_of(alpha_imf, 5.0, 1.0, 0.4)
            return jnp.sum(a ** 2)
        g_ad = float(jax.grad(loss)(2.3))
        h = 1e-4
        g_fd = (loss(2.3 + h) - loss(2.3 - h)) / (2 * h)
        assert abs(g_ad - g_fd) / (abs(g_fd) + 1e-12) < 1e-4

    def test_grad_wrt_delta_and_W0(self):
        def loss(theta):
            delta, W0 = theta
            a, _, _, _ = _alpha_of(2.3, W0, 1.0, delta)
            return jnp.sum(a ** 2)
        g_ad = np.asarray(jax.grad(loss)(jnp.array([0.4, 5.0])))
        h = 1e-4
        g_fd = []
        for i in range(2):
            e = np.zeros(2); e[i] = h
            g_fd.append(float((loss(jnp.array([0.4, 5.0]) + e)
                               - loss(jnp.array([0.4, 5.0]) - e)) / (2 * h)))
        np.testing.assert_allclose(g_ad, np.array(g_fd), rtol=2e-3, atol=1e-6)
```

**Step 2: Run, capture the printed α reference, fill `REF_ALPHA`.**
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/profiles/test_find_alpha_ift.py -q
```
First run: `test_alpha_value_pinned` fails (REF_ALPHA placeholder). Print the
actual α (add a temporary `print`), set `REF_ALPHA = [...]` (the 4 values), rerun.
Expected after filling: **all tests PASS on the current unrolled code** (this is
the behavior H2 must preserve). The FD-grad tests passing now confirms the
current gradient is already ≈correct; H2 keeps them passing with the exact
gradient.

**Step 3: Commit**
```bash
git add tests/unit/profiles/test_find_alpha_ift.py
git commit -m "test(profiles): characterization pins for find_alpha_for_masses (forward value + FD-grad) pre-IFT"
```

---

## Task H2: Refactor `find_alpha_for_masses` to optimistix IFT

**Files:** Modify `src/progenax/profiles/limepy_multimass.py`
(`find_alpha_for_masses`, ~lines 364–418; imports at top).

**Step 1: Add imports** (top of file): `import optimistix as optx`,
`import lineax as lx`.

**Step 2: Replace the scan body.** Keep the signature and return
`(alpha_final, residual)` identical. Pass ALL differentiable dependencies via
`args` (optimistix's `ImplicitAdjoint` propagates cotangents w.r.t. `args`;
closure-captured traced arrays are NOT reliably differentiated — this is the
one correctness detail to get right):

```python
def find_alpha_for_masses(m_j, M_j, W0, g, delta, n_iter=30, xi_max=300.0,
                          n_points=2000, ra_hat=None, eta=0.0, aniso_method="table"):
    """... (keep the existing docstring; add a note: solved via
    optimistix.fixed_point with ImplicitAdjoint -- forward iterates match the
    Gieles & Zocchi sqrt-update; the gradient is the EXACT fixed-point gradient
    (implicit function theorem), independent of n_iter once converged)."""
    m_j = jnp.asarray(m_j)
    M_j = jnp.asarray(M_j)
    f_target = M_j / jnp.sum(M_j)

    # Dynamic (differentiable) args; statics (xi_max, n_points, aniso_method) via closure.
    args = (m_j, f_target, jnp.asarray(W0), jnp.asarray(g), jnp.asarray(delta),
            ra_hat, jnp.asarray(eta))

    def map_fn(alpha, args):
        m_j_, f_target_, W0_, g_, delta_, ra_hat_, eta_ = args
        f_real = _realized_fractions(alpha, m_j_, W0_, g_, delta_, xi_max, n_points,
                                     ra_hat_, eta_, aniso_method)
        alpha_new = alpha * jnp.sqrt(f_target_ / (f_real + 1e-300))
        return alpha_new / jnp.sum(alpha_new)

    sol = optx.fixed_point(
        map_fn,
        optx.FixedPointIteration(rtol=1e-10, atol=1e-12),
        f_target,
        args=args,
        max_steps=n_iter,
        adjoint=optx.ImplicitAdjoint(
            linear_solver=lx.AutoLinearSolver(well_posed=False)),
        throw=False,
    )
    alpha_final = sol.value
    f_real = _realized_fractions(alpha_final, m_j, W0, g, delta, xi_max, n_points,
                                 ra_hat, eta, aniso_method)
    residual = jnp.max(jnp.abs(f_real - f_target))
    return alpha_final, residual
```

Notes for the implementer:
- `FixedPointIteration` runs the SAME √-map → forward iterates match the old
  scan; `max_steps=n_iter` caps it (tight rtol/atol so it runs to `n_iter`
  unless already converged — keeping forward parity).
- `throw=False` keeps it jit/scan-safe (no host-side raise on non-convergence;
  `residual` remains the diagnostic).
- If `AutoLinearSolver(well_posed=False)` errors or the grad-match gate fails,
  the fallback (design doc) is a hand-rolled `custom_vjp` with an explicit
  simplex tangent-space projection — STOP and report before switching.

**Step 3: Run the H1 pins — forward value + FD-grad must still pass.**
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/profiles/test_find_alpha_ift.py -q
```
Expected: **all PASS** (forward α pinned to rtol 1e-8; grads match FD). If the
forward value drifts > 1e-8, the solver/map differs from the original — STOP and
report. If the grad-match fails, the rank-deficient adjoint solve is wrong —
STOP and report (do not loosen the gate).

**Step 4: Run the existing released-core multimass tests (forward safety).**
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/profiles/test_limepy_multimass.py tests/unit/cluster/test_multicomponent.py -q
```
Expected: PASS (forward values unchanged).

**Step 5: Commit**
```bash
git add src/progenax/profiles/limepy_multimass.py
git commit -m "perf(profiles): find_alpha_for_masses via optimistix fixed_point + ImplicitAdjoint (exact O(1) backward)"
```

---

## Task H3: Profile per-eval cost + `n_iter` residual sweep

**Files:** Create `scripts/profile_find_alpha.py` (a gated profiling CLI).

**Step 1:** Write a script that, across a small (α, δ, W₀) grid spanning the B2
boxes (e.g. α∈{1.9,2.3,2.7}, δ∈{0.1,0.4,0.6}, W₀∈{4,5,7}):
- builds `from_imf` and records `residual` vs `n_iter ∈ {10,15,20,30}`;
- times one warm `jit(value_and_grad(loss))` of a small scalar reduction of
  `from_imf` BEFORE/AFTER is not possible in one run (post-refactor only) — so
  just measure the post-refactor warm grad-eval time at `n_iter ∈ {15,30}`;
- prints a table: `n_iter | max residual over grid | warm grad-eval (s)`.

**Step 2:** Pick the smallest `n_iter` whose **max residual over the grid <
2e-3** (precedent: 15 is exercised in `test_limepy_multimass.py:229`). Record it
as the recommended demo `n_iter` (do NOT change `from_imf`'s default; the demo
passes the value).

**Step 3:** Print PASS if the chosen `n_iter` keeps residual<2e-3 everywhere;
`sys.exit(1)` otherwise. Run it:
```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/profile_find_alpha.py
```

**Step 4: Commit**
```bash
git add scripts/profile_find_alpha.py
git commit -m "perf(profiles): n_iter residual + grad-eval profiling harness for find_alpha"
```

---

## Task H4: Right-size the B2 demo optimizer + re-validate against the reference

**Files:** Modify `scripts/demo_delta_recovery.py` (the `from_imf` calls to pass
the profiled `n_iter`; the MLE `n_inits`/`n_steps`).

**Step 1:** Thread the profiled `n_iter` (H3) into the `from_imf` calls inside
`predict_binned` / truth build (add an `N_ITER` module constant; pass
`n_iter=N_ITER`). Keep truth-data construction at the conservative default (or
the same N_ITER — document).

**Step 2:** Reduce `mle_adam` to a profiled `(n_inits, n_steps)` — start
3 inits × 300 steps (keep z₀=0 near-truth + ≥2 dispersed; the reference showed a
shallow secondary, so ≥3 inits). Keep the plateau gate and the "pick lowest
final negloglike" selection.

**Step 3: Run the demo end-to-end; capture wall-clock.**
```bash
time env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_delta_recovery.py
echo "exit: $?"
```
Expected: **ALL PASS, exit 0**, recovered θ̂ within Fisher σ of the reference
(α̂≈2.293, δ̂≈0.397, Ŵ₀≈4.990), wall-clock ≫5× faster than the ~50 min baseline.
If recovery drifts beyond the reference Fisher σ, STOP and report (do not adjust
gates).

**Step 4:** Update the demo docstring's recorded timings + `n_iter`/optimizer
config. **Commit**
```bash
git add scripts/demo_delta_recovery.py
git commit -m "perf(demos): B2 MLE right-sized (profiled n_iter + 3x300 inits); recovers reference theta-hat, NNx faster"
```

---

## Task H5: Full gate + memory gates + completion doc

**Step 1: FULL released-core gate (incl. slow) — expect 1163 + the new H1 tests.**
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
```
Expected: all pass (1163 + new). Record the count.

**Step 2: Memory gates.**
```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/profile_cluster_memory.py
```
Expected: all 7 PASS.

**Step 3:** Completion doc `.claude-work/MULTICOMPONENT_IFT_HARDENING_COMPLETE.md`:
the reference-vs-refactored θ̂ table, the gradient-match numbers (AD vs FD), the
n_iter residual table, before/after wall-clock, files changed, lessons. Note the
remaining future IFT candidates (Engine B truncation radius, King tidal radius).

**Step 4:** STOP — present the completion doc + evidence to Anna. Merge to local
main only on her explicit go; then resume the B2 demo arc (Tasks 5–9).

---

## Out of scope (do not drift)

Engine B / King-tidal-radius IFT (future); changing N_COMP or the √-map physics;
the B2 demo science beyond re-validating Task 4 (Tasks 5–9 resume after this).
