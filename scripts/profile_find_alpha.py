#!/usr/bin/env python
r"""Box-corner residual + grad-eval timing harness for find_alpha_for_masses.

Gated CLI evidence artifact (Task H3): across the (alpha_imf, delta, W0)
box corners {(1.9, 0.1, 4), (2.3, 0.4, 5), (2.7, 0.6, 7)} -- the corners of
the B2 self-consistent IMF+equipartition recovery box (see the
multicomponent-IFT-hardening plan) -- it reports for the now-IFT-accelerated
``find_alpha_for_masses`` (the same eigenvalue solve ``from_imf`` calls):

  * forward iters-to-converge: the number of Gieles & Zocchi (2015) sqrt-update
    steps until max_j|f_j' - f_j| < tol. The released forward is an adaptive
    ``jax.lax.while_loop`` that does NOT expose its iteration count, so we count
    iters here by re-running the SAME cond/body (``_alpha_map`` /
    ``_realized_fractions``) in a plain Python loop. This is a diagnostic
    harness (NOT src / a hot path), so a Python loop is fine; the residual it
    converges to is cross-checked against the value ``find_alpha_for_masses``
    itself returns (must match to <1e-9).

  * final residual: ``find_alpha_for_masses``'s OWN returned residual
    max_j|f_j' - f_j|. Gate: < 2e-3 per corner.

  * warm ``jit(value_and_grad)`` wall-clock (ms): a scalar reduction
    L(delta) = sum(alpha_j) + residual through the REAL jitted
    ``find_alpha_for_masses`` (the timed/grad path goes through the released
    custom_vjp solver, NOT the Python iter-counter). Compiled once and blocked
    (JAX is async) BEFORE timing; the reported number is the MEDIAN of several
    warm calls, each ``jax.block_until_ready``-ed.

We time ``find_alpha_for_masses`` directly rather than through
``MultiComponentCluster.from_imf`` -- it is the IFT-accelerated kernel under
test, and timing it directly is cleaner and avoids the post-solve ODE/grid build
(``from_imf`` calls exactly this same ``find_alpha_for_masses`` on the binned IMF,
so the kernel timing is representative).

Exits 1 iff ANY corner's residual >= 2e-3 (a real non-convergence finding --
the gate is NEVER weakened); else exits 0.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/profile_find_alpha.py
"""
import datetime
import statistics
import sys
import time

import progenax  # noqa: F401  -- enables float64 at import (jaxstro high precision)
import jax
import jax.numpy as jnp

from progenax.imf.smooth import Maschberger
from progenax.profiles.limepy_multimass import (
    _alpha_map,
    _bin_imf,
    _realized_fractions,
    find_alpha_for_masses,
)

# Box corners (alpha_imf, delta, W0) -- the corners of the B2 IMF+equipartition
# recovery box, from the multicomponent-IFT-hardening plan (Task H3 Step 1).
CORNERS = ((1.9, 0.1, 4.0), (2.3, 0.4, 5.0), (2.7, 0.6, 7.0))

RESIDUAL_GATE = 2e-3   # plan gate: a corner with residual >= this FAILs (never weakened)
N_COMP = 4             # IMF -> 4 log-spaced components (matches the B2 demo / from_imf)
M_RANGE = (0.1, 20.0)  # IMF mass range (matches scripts/demo_delta_recovery.py)
G_MODEL = 1.0          # LIMEPY g (King g=1)
TOL = 1e-6             # forward residual tolerance (find_alpha_for_masses default)
N_ITER = 30            # forward iteration safety cap (find_alpha_for_masses default)
N_WARM = 7             # warm jit(value_and_grad) calls; report the median


def _binned_imf(alpha_imf):
    """Bin Maschberger(alpha_imf) into N_COMP components -> (m_j, M_j), as from_imf does."""
    imf = Maschberger(alpha=alpha_imf, m_min=M_RANGE[0], m_max=M_RANGE[1])
    return _bin_imf(imf, N_COMP, M_RANGE)


def _count_iters(m_j, M_j, W0, delta):
    """Iters-to-converge by re-running the released forward's cond/body in Python.

    Mirrors ``_solve_alpha_iso``'s while_loop exactly: start at alpha = f_target,
    apply the sqrt-update ``_alpha_map``, recompute the realized-fraction residual,
    and count steps until max_j|f_j' - f_j| < TOL (or the N_ITER cap). Returns
    (iters, residual_at_stop); residual_at_stop is cross-checked against the
    released solver's returned residual by the caller.
    """
    f_target = M_j / jnp.sum(M_j)
    alpha = f_target
    residual = jnp.inf
    iters = 0
    while iters < N_ITER and float(residual) > TOL:
        alpha = _alpha_map(alpha, m_j, f_target, W0, G_MODEL, delta,
                           300.0, 2000, None, 0.0, "table")
        f_real = _realized_fractions(alpha, m_j, W0, G_MODEL, delta,
                                     300.0, 2000, None, 0.0, "table")
        residual = jnp.max(jnp.abs(f_real - f_target))
        iters += 1
    return iters, float(residual)


def _warm_vgrad_ms(m_j, M_j, W0):
    """Median warm ms of jit(value_and_grad) of L(delta) = sum(alpha) + residual.

    delta is the scalar differentiated argument (the iso custom_vjp solver
    differentiates delta); the timed path is the REAL jitted find_alpha_for_masses.
    Compile once + block, then time N_WARM warm calls at perturbed delta, each
    block_until_ready-ed (JAX is async). Returns (median_ms, value, grad).
    """
    def scalar_loss(delta):
        alpha, residual = find_alpha_for_masses(
            m_j, M_j, W0, G_MODEL, delta, n_iter=N_ITER, tol=TOL)
        return jnp.sum(alpha) + residual

    vgrad = jax.jit(jax.value_and_grad(scalar_loss))

    # Warm the jit: compile once and block before any timing (JAX async).
    v0, g0 = vgrad(0.30)
    jax.block_until_ready((v0, g0))

    times_ms = []
    for k in range(N_WARM):
        delta_k = 0.30 + 0.01 * k  # perturb delta so it is a genuine steady-state call
        t0 = time.perf_counter()
        v, g = vgrad(delta_k)
        jax.block_until_ready((v, g))
        times_ms.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times_ms), float(v0), float(g0)


def main() -> int:
    print("=" * 78)
    print("find_alpha_for_masses box-corner profile (residual / iters / grad-eval ms)")
    print(f"date {datetime.date.today().isoformat()}  |  platform {sys.platform}  "
          f"|  x64 {jax.config.jax_enable_x64}")
    print(f"N_COMP={N_COMP}  M_RANGE={M_RANGE}  g={G_MODEL}  tol={TOL:g}  "
          f"n_iter cap={N_ITER}  warm calls={N_WARM} (median)")
    print(f"timed path: find_alpha_for_masses directly; grad wrt delta of "
          f"L=sum(alpha)+residual  |  residual gate < {RESIDUAL_GATE:g}")
    print("=" * 78)

    rows = []
    any_fail = False
    for alpha_imf, delta, W0 in CORNERS:
        m_j, M_j = _binned_imf(alpha_imf)

        # Released solver: the residual we GATE on is its own returned residual.
        alpha, residual = find_alpha_for_masses(
            m_j, M_j, W0, G_MODEL, delta, n_iter=N_ITER, tol=TOL)
        jax.block_until_ready((alpha, residual))
        residual = float(residual)

        iters, residual_py = _count_iters(m_j, M_j, W0, delta)
        # Cross-check: the Python iter-counter must reach the SAME residual as the
        # released forward (same cond/body). Flag a mismatch loudly.
        residual_match = abs(residual_py - residual) < 1e-9

        warm_ms, _v, _g = _warm_vgrad_ms(m_j, M_j, W0)

        passed = residual < RESIDUAL_GATE
        any_fail |= not passed
        status = "PASS" if passed else "FAIL"
        rows.append((alpha_imf, delta, W0, iters, residual, warm_ms, status,
                     residual_match))

        print(f"  corner alpha={alpha_imf:.1f} delta={delta:.1f} W0={W0:.0f}: "
              f"iters={iters:>2}  residual={residual:.2e}  "
              f"warm vgrad={warm_ms:7.1f} ms  {status}"
              + ("" if residual_match else "  [WARN iter-count residual mismatch]"),
              flush=True)

    print("-" * 78)
    print(f"  {'alpha_imf':>9} {'delta':>6} {'W0':>4} {'iters':>6} "
          f"{'residual':>11} {'gate':>9} {'warm vgrad [ms]':>16} {'status':>7}")
    for alpha_imf, delta, W0, iters, residual, warm_ms, status, _m in rows:
        print(f"  {alpha_imf:>9.1f} {delta:>6.1f} {W0:>4.0f} {iters:>6d} "
              f"{residual:>11.2e} {RESIDUAL_GATE:>9.0e} {warm_ms:>16.1f} "
              f"{status:>7}")
    print("=" * 78)

    if any_fail:
        print("  FAIL: a box corner did not converge below the 2e-3 residual gate "
              "(real finding -- gate NOT weakened).")
        return 1
    print("  PASS: all box corners converge below the 2e-3 residual gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
