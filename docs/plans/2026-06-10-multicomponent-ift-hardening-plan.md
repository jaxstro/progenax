# Multicomponent IFT Hardening + B2 Speed — Implementation Plan (v2, hand-rolled custom_vjp)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> to implement this plan task-by-task (fresh subagent per task + code review).

**Goal:** Replace the unrolled 30-step eigenvalue scan in
`find_alpha_for_masses` with a hand-rolled `jax.custom_vjp` — adaptive
`while_loop` forward to a residual tolerance + reverse-mode implicit-VJP backward
— so one `value_and_grad` drops ~716 ms → ~249 ms (2.9×, flat-in-n_iter
backward) with gradients exact to <1e-6; then right-size the B2 demo optimizer.
Recover the verified reference θ̂; keep the 1163 released-core invariant green.

**Architecture:** `find_alpha_for_masses` solves the fixed point
`α ← normalize(α·√(f_target/f_real(α)))` where `f_real` solves a coupled Poisson
ODE. Today: a `lax.scan` of `n_iter=30`, unrolled+differentiated (≈30 ODE solves
each way). New: a `jax.custom_vjp` whose forward is an adaptive `while_loop`
(undifferentiated → `while_loop` legal here) and whose backward builds the 4×4
implicit Jacobian by **reverse-mode `vjp`** of the **√-map residual R**, solves
`Jᵀw=ᾱ` with `lstsq`, and returns `−vjpθ(w)`. **No optimistix/lineax.** Design
doc: `docs/plans/2026-06-10-multicomponent-ift-hardening-design.md`. **Validated
reference prototype (read it): `docs/plans/_ift_prototype_reference.py`**
(Prototype 2 = the chosen config; benchmarked 2.9×, grad 3.3e-7 vs FD).

**Tech stack:** JAX (`jax.custom_vjp`, `jax.lax.while_loop`, `jax.vjp`,
`jnp.linalg.lstsq`), optax (demo MLE). No new dependencies.

**Reference (the accuracy anchor):** B2 MLE recovers α̂=2.2931±0.0041,
δ̂=0.3972±0.0344, Ŵ₀=4.9900±0.0209 (all 3σ PASS); `find_alpha` residual <2e-3.

**Gates:** FAST gate (1121 not-slow) + FULL gate (1163) per `progenax/CLAUDE.md`.
**Git:** continue on branch `feat/batch-b-demos`; commit per task; NO push/merge
without Anna's go.

**Hard rules:** JAX-native; TDD (characterization-first); the grad-match gate
(FD <1e-5) and forward-consistency pin are REAL — never weaken. Use **R** (√-map
residual), NOT `f_real−f_target` (cond 1e16 → wrong grads). forward **tol=1e-6**.
Do not change the physics (N_COMP=4, the √-map).

---

## Task H1: Characterization pins on the CURRENT code (forward + FD-grad)

TDD here is characterization-first: capture the current correct behavior so the
H2 refactor must preserve it.

**Files:** Create `tests/unit/profiles/test_find_alpha_ift.py`.

**Step 1: Write the tests.**
```python
"""Forward + gradient characterization pins for find_alpha_for_masses.

Written against the CURRENT unrolled implementation so the custom_vjp/IFT
refactor (H2) must preserve the forward value (to ~residual) and match
finite-difference gradients. The IFT gradient is the EXACT fixed-point
gradient; we gate it against CENTRAL FINITE DIFFERENCES (the ground truth).
"""
import jax, jax.numpy as jnp, numpy as np
import progenax  # noqa: F401  (float64)
from progenax.profiles.limepy_multimass import find_alpha_for_masses, _bin_imf
from progenax.imf.smooth import Maschberger

def _alpha_of(alpha_imf, W0, g, delta):
    imf = Maschberger(alpha=alpha_imf, m_min=0.1, m_max=20.0)
    m_j, M_j = _bin_imf(imf, 4, (0.1, 20.0))
    a, res = find_alpha_for_masses(m_j, M_j, W0, g, delta)
    return a, res

class TestForwardRegression:
    def test_sums_to_one_and_converged(self):
        a, res = _alpha_of(2.3, 5.0, 1.0, 0.4)
        assert abs(float(jnp.sum(a)) - 1.0) < 1e-9
        assert float(res) < 2e-3 and bool(jnp.all(a > 0))

    def test_alpha_value_pinned(self):
        a, _ = _alpha_of(2.3, 5.0, 1.0, 0.4)
        # Fill REF_ALPHA from the first run's print; H2's adaptive solve must
        # agree with this (converged) value to <1e-4 (consistency), and the H2
        # version re-pins to its own value at rtol 1e-8 (self-regression).
        np.testing.assert_allclose(np.asarray(a), np.array(REF_ALPHA), atol=1e-4)

class TestGradientMatchesFD:
    def test_grad_alpha_imf(self):
        def loss(ai): a, _ = _alpha_of(ai, 5.0, 1.0, 0.4); return jnp.sum(a**2)
        g_ad = float(jax.grad(loss)(2.3)); h = 1e-4
        g_fd = float((loss(2.3+h)-loss(2.3-h))/(2*h))
        assert abs(g_ad-g_fd)/(abs(g_fd)+1e-12) < 1e-5

    def test_grad_delta_W0(self):
        def loss(t):
            d, W0 = t; a,_ = _alpha_of(2.3, W0, 1.0, d); return jnp.sum(a**2)
        g_ad = np.asarray(jax.grad(loss)(jnp.array([0.4,5.0]))); h=1e-4
        g_fd=[]
        for i in range(2):
            e=np.zeros(2); e[i]=h
            g_fd.append(float((loss(jnp.array([0.4,5.0])+e)-loss(jnp.array([0.4,5.0])-e))/(2*h)))
        np.testing.assert_allclose(g_ad, np.array(g_fd), rtol=1e-5, atol=1e-7)
```

**Step 2: Run; capture printed α; fill `REF_ALPHA`.**
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/profiles/test_find_alpha_ift.py -q
```
Add a temporary print of `a`, set `REF_ALPHA=[...]`, rerun → all PASS on the
current code (the FD-grad tests passing now confirms the current gradient is
already ≈exact; H2 keeps them passing).

**Step 3: Commit**
```bash
git add tests/unit/profiles/test_find_alpha_ift.py
git commit -m "test(profiles): characterization pins for find_alpha_for_masses (forward + FD-grad) pre-IFT"
```

---

## Task H2: Refactor `find_alpha_for_masses` to a hand-rolled `custom_vjp`

**Files:** Modify `src/progenax/profiles/limepy_multimass.py`
(`find_alpha_for_masses` ~364–418; helpers above it). **Read
`docs/plans/_ift_prototype_reference.py` first** — adapt its Prototype 2
(`make_proto2`) `fwd`/`bwd` to the real signature.

**Step 1:** Add a module-level residual + step helper (explicit args, no closure,
so `jax.vjp` is clean):
```python
def _alpha_map(alpha, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method):
    f_real = _realized_fractions(alpha, m_j, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
    a = alpha * jnp.sqrt(f_target / (f_real + 1e-300))
    return a / jnp.sum(a)

def _alpha_residual(alpha, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method):
    return alpha - _alpha_map(alpha, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
```

**Step 2:** Implement TWO `custom_vjp` solvers dispatched on `ra_hat is None`
(v2 verification — a single solver crashes on `−None` when `ra_hat=None`, which
is BOTH the demo path AND the released grad test
`test_limepy_multimass.py::test_differentiable_in_targets_and_delta`). **Read
`docs/plans/_ift_realsig_reference.py` — it is the verified working code
(grad vs FD ≤2.2e-7); adapt it directly.**

- **Isotropic solver** (`ra_hat is None`): `ra_hat, eta` are NONDIFF (closed
  over / not in the diff set); differentiate `(m_j, M_j, W0, g, delta)`;
  `nondiff_argnums` covers `(ra_hat? eta? xi_max, n_points, aniso_method, tol,
  max_iter)`; `fwd` saves `(a_star, m_j, M_j, W0, g, delta)`; `bwd` returns a
  **5-tuple** `(-gm,-gM,-gW,-gg,-gd)` and `R_th` closes over `ra_hat`/`eta`.
- **Anisotropic solver** (`ra_hat` finite): the 7-tuple version below.
- `find_alpha_for_masses` branches: `if ra_hat is None: a = _solve_alpha_iso(...)
  else: a = _solve_alpha_aniso(...)`.
- Share the `while_loop` body via the module-level `_alpha_map`/`_alpha_residual`
  (Step 1) so only the thin `fwd`/`bwd`/`defvjp` differ between the two.

The anisotropic solver (differentiable args `(m_j, M_j, W0, g, delta, ra_hat,
eta)`; statics via `nondiff_argnums`; forward = adaptive `while_loop` to `tol`;
backward = reverse-mode implicit VJP — R-residual, `vmap`'d `vjp` for the n×n
Jacobian (n=n_comp, general — `jnp.eye(a_star.shape[0])`), `lstsq`,
`−vjp_theta(w)`):
```python
import functools

@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11))
def _solve_alpha(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points, aniso_method, tol, max_iter):
    f_target = M_j / jnp.sum(M_j)
    def cond(s): _, it, r = s; return jnp.logical_and(it < max_iter, r > tol)
    def body(s):
        a, it, _ = s
        a_new = _alpha_map(a, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
        f_real = _realized_fractions(a_new, m_j, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
        return a_new, it + 1, jnp.max(jnp.abs(f_real - f_target))
    a_star, _, _ = jax.lax.while_loop(cond, body, (f_target, jnp.array(0), jnp.array(jnp.inf)))
    return a_star

def _solve_alpha_fwd(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points, aniso_method, tol, max_iter):
    a_star = _solve_alpha(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points, aniso_method, tol, max_iter)
    return a_star, (a_star, m_j, M_j, W0, g, delta, ra_hat, eta)

def _solve_alpha_bwd(xi_max, n_points, aniso_method, tol, max_iter, res, a_bar):
    a_star, m_j, M_j, W0, g, delta, ra_hat, eta = res
    f_target = M_j / jnp.sum(M_j)
    R_a = lambda a: _alpha_residual(a, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
    _, vjp_a = jax.vjp(R_a, a_star)
    J = jax.vmap(lambda e: vjp_a(e)[0])(jnp.eye(a_star.shape[0]))   # 4x4, reverse-mode
    w = jnp.linalg.lstsq(J.T, a_bar, rcond=None)[0]
    R_th = lambda mj, Mj, W, gg, d, rah, et: _alpha_residual(
        a_star, mj, Mj / jnp.sum(Mj), W, gg, d, xi_max, n_points, rah, et, aniso_method)
    _, vjp_th = jax.vjp(R_th, m_j, M_j, W0, g, delta, ra_hat, eta)
    gm, gM, gW, gg, gd, gra, get = vjp_th(w)
    return (-gm, -gM, -gW, -gg, -gd, -gra, -get)

_solve_alpha.defvjp(_solve_alpha_fwd, _solve_alpha_bwd)
```
Then `find_alpha_for_masses` wraps it (keep signature; `n_iter` → `max_iter`
cap; add `tol=1e-6`; recompute `residual` once for the diagnostic — or carry it
from the forward to avoid an extra solve):
```python
def find_alpha_for_masses(m_j, M_j, W0, g, delta, n_iter=30, xi_max=300.0,
                          n_points=2000, ra_hat=None, eta=0.0, aniso_method="table", tol=1e-6):
    """... (note: solved by a hand-rolled jax.custom_vjp -- adaptive while_loop
    forward to residual tol, reverse-mode implicit-VJP backward (exact
    fixed-point gradient). Forward iterates the Gieles&Zocchi sqrt-update;
    n_iter is the safety cap.)"""
    m_j = jnp.asarray(m_j); M_j = jnp.asarray(M_j)
    a = _solve_alpha(m_j, M_j, jnp.asarray(W0), jnp.asarray(g), jnp.asarray(delta),
                     ra_hat, jnp.asarray(eta), xi_max, n_points, aniso_method, tol, n_iter)
    f_real = _realized_fractions(a, m_j, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
    residual = jnp.max(jnp.abs(f_real - M_j / jnp.sum(M_j)))
    return a, residual
```
Implementer notes: `ra_hat=None` MUST use the dedicated **isotropic solver**
above (keep `ra_hat`/`eta` OUT of the differentiated set — closed over) — do NOT
use a traced sentinel (a traced array can't hit the Python `is None` branch in
`_realized_fractions`, so it would route the isotropic case through the slow,
wrong anisotropic table path; v2-verified). `docs/plans/_ift_realsig_reference.py`
is the working two-solver code — adapt it. If `lstsq` is unstable anywhere,
`jnp.linalg.pinv(J.T) @ a_bar` is the drop-in fallback — STOP and report before
switching residual definitions (do NOT use `f_real−f_target`, cond 1e16).

**Step 3: Run the H1 pins — forward (consistency) + FD-grad must pass.**
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/profiles/test_find_alpha_ift.py -q
```
Expected: PASS (α* agrees with REF_ALPHA to <1e-4; grads match FD to <1e-5). If
the grad-match fails, the bwd math/lstsq/R-residual is wrong — STOP and report
(do not loosen the gate).

**Step 4: Run the existing released-core multimass + cluster tests (forward safety).**
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/profiles/test_limepy_multimass.py tests/unit/cluster/test_multicomponent.py -q
```
Expected: PASS.

**Step 5: Commit**
```bash
git add src/progenax/profiles/limepy_multimass.py
git commit -m "perf(profiles): find_alpha_for_masses via hand-rolled custom_vjp (adaptive fwd + reverse-mode implicit backward, ~3x, exact)"
```

---

## Task H3: Profile residual/iters across the box + before/after timing

**Files:** Create `scripts/profile_find_alpha.py` (gated CLI).

**Step 1:** Across the (α,δ,W₀) box corners {(1.9,0.1,4),(2.3,0.4,5),(2.7,0.6,7)}
report: forward iters-to-converge, final residual (gate <2e-3), and warm
`jit(value_and_grad)` ms of a scalar reduction of `from_imf`. Print a table;
`sys.exit(1)` if any corner residual ≥ 2e-3.

**Step 2: Run.**
```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/profile_find_alpha.py
```
Expected: all corners converge (<2e-3) in ≤~15 iters; PASS.

**Step 3: Commit**
```bash
git add scripts/profile_find_alpha.py
git commit -m "perf(profiles): box-corner residual + grad-eval timing harness for find_alpha"
```

---

## Task H4: Right-size the B2 demo optimizer + re-validate vs reference

**Files:** Modify `scripts/demo_delta_recovery.py` (the `mle_adam`
`n_inits`/`n_steps`; `from_imf` already faster via the adaptive forward).

**Step 1:** `mle_adam`: 3 inits × ~300 steps (keep z₀=0 + 2 dispersed; keep the
plateau gate + lowest-final-loss selection). Optionally pass an explicit
`tol`/`n_iter` cap to `from_imf` if the demo wants a tighter/looser forward.

**Step 2: Run end-to-end; capture wall-clock.**
```bash
time env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_delta_recovery.py
echo "exit: $?"
```
Expected: **ALL PASS, exit 0**, θ̂ within Fisher σ of the reference
(α̂≈2.293, δ̂≈0.397, Ŵ₀≈4.990), wall-clock ≫ the ~50 min baseline (target
≤~10 min). If recovery drifts beyond the reference Fisher σ, STOP and report (do
not adjust gates).

**Step 3:** Update the demo docstring timings/config. **Commit**
```bash
git add scripts/demo_delta_recovery.py
git commit -m "perf(demos): B2 MLE right-sized (3x300 inits) on the IFT-accelerated from_imf; recovers reference theta-hat, NNx faster"
```

---

## Task H5: Full gate + memory gates + completion doc

**Step 1: FULL released-core gate (incl. slow).**
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
```
Expected: all pass (1163 + new H1 tests). Record the count.

**Step 2: Memory gates.**
```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/profile_cluster_memory.py
```
Expected: all 7 PASS.

**Step 3:** Completion doc
`.claude-work/MULTICOMPONENT_IFT_HARDENING_COMPLETE.md`: reference-vs-refactored
θ̂; grad-match numbers (AD vs FD); box residual/iters table; before/after
wall-clock; the optimistix-rejection record; files changed; lessons; remaining
future IFT candidates (Engine B truncation radius, King tidal radius). Remove the
scratch `docs/plans/_ift_prototype_reference.py` (or note it as kept-for-record).

**Step 4:** STOP — present the completion doc + evidence to Anna. Merge to local
main only on her explicit go; then resume the B2 demo arc (Tasks 5–9).

---

## Out of scope (do not drift)

Engine B / King-tidal-radius IFT (future); optimistix (rejected); changing
N_COMP or the √-map physics; the B2 science beyond re-validating Task 4.
