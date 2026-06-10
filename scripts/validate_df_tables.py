#!/usr/bin/env python
"""Validation + figure for the LIMEPY DF density tables (Phase 1.5, Tranche A).

Proves the AnisoDensityTable performance layer reproduces the exact quadrature
oracle to the stated budgets, end to end:

  (a) table-vs-quadrature density: max relative error of AnisoDensityTable
      against _aniso_density_scalar over 2000 random domain points, at the
      Task-1 512x96 grid (budget 1e-5) AND at the solver's 160x40 grid
      (reported for honesty; the solver grid is sized to the SOLVE budget,
      not the pointwise one).
  (b) coupled-solve accuracy: max |psi_table - psi_quadrature| across the three
      Task-2 configs (budget 1e-4 * W0 each).
  (c) warm solve speedup table vs quadrature (PASS >= 3x; measured ~5x).
  (d) construction-level speedup of an anisotropic MultiComponentCluster,
      aniso_method="table" vs "quadrature" (reported; includes the Task-4
      dedup: ONE shared table for the solve AND the mass-CDF grid; PASS >= 2x).
  (e) AD-vs-FD gradient through the TABLE solve in (w_j, r_a) (rtol 1e-3) --
      the table build (box depends on rescale = w_j^-2) and the cubic
      interpolation are inside the differentiated graph.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_df_tables.py
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.profiles.limepy import _aniso_density_scalar
from progenax.profiles.limepy_multimass import (
    _TAB_N_P,
    _TAB_N_W,
    solve_multicomponent_limepy,
)
from progenax.profiles.limepy_tables import AnisoDensityTable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
W_MAX, P_MAX, G_PAR = 12.0, 80.0, 1.0       # Task-1 density-domain box
# The three Task-2 solve configs: (W0, rescale, ra_hat, xi_max, n_points)
SOLVE_CONFIGS = [
    (7.0, (1.0, 1.6), (10.0, 10.0), 800.0, 2000),
    (5.0, (1.0, 1.6), (5.0, 5.0), 800.0, 2000),
    (9.0, (1.0, 2.2), (40.0, 40.0), 5000.0, 3000),
]
ALPHA = jnp.array([0.6, 0.4])
MODEL_KW = dict(alpha_j=ALPHA, w_j=jnp.array([1.0, 0.79]),
                m_j=jnp.array([1.0, 4.0]), W0=7.0, g=1.0, r_c=1.0,
                ra_hat_j=jnp.array([10.0, 10.0]), xi_max=800.0,
                n_ode_points=2000, n_grid=1000)


def _timed(fn, n_rep=3):
    """Median wall time of fn() over n_rep warm repetitions (fn called once
    first to compile/warm; fn must block on its own outputs)."""
    fn()
    times = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def check_density_accuracy():
    """(a) Pointwise table-vs-quadrature relative error on random domain points."""
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.uniform(1e-3 * W_MAX, W_MAX, 2000))
    p = jnp.asarray(rng.uniform(0.0, P_MAX, 2000))
    exact = jax.vmap(lambda w, pp: _aniso_density_scalar(w, pp, jnp.asarray(G_PAR)))(W, p)
    central = float(_aniso_density_scalar(jnp.asarray(W_MAX), jnp.asarray(0.0),
                                          jnp.asarray(G_PAR)))
    out = {}
    for label, (n_W, n_p) in [("512x96", (512, 96)),
                              (f"{_TAB_N_W}x{_TAB_N_P}", (_TAB_N_W, _TAB_N_P))]:
        tab = AnisoDensityTable.build(W_max=W_MAX, p_max=P_MAX, g=G_PAR,
                                      n_W=n_W, n_p=n_p)
        approx = jax.vmap(tab.evaluate)(W, p)
        rel = np.asarray(jnp.abs(approx - exact) /
                         jnp.maximum(exact, 1e-8 * central))
        out[label] = (np.asarray(W), rel)
    return out


def _solve(method, W0, rescale, ra, xi_max, n_points):
    return solve_multicomponent_limepy(
        ALPHA, jnp.array(rescale), W0=W0, g=G_PAR, xi_max=xi_max,
        n_points=n_points, ra_hat_j=jnp.array(ra), aniso_method=method)


def check_solve_accuracy():
    """(b) |psi_table - psi_quad| profiles for the three Task-2 configs."""
    results = []
    for cfg in SOLVE_CONFIGS:
        xi_q, psi_q, _ = _solve("quadrature", *cfg)
        xi_t, psi_t, _ = _solve("table", *cfg)
        dpsi = np.asarray(jnp.abs(psi_t - psi_q))
        results.append((cfg, np.asarray(xi_t), dpsi))
    return results


def check_solve_speedup():
    """(c) Warm solve wall time, quadrature vs table (baseline config)."""
    cfg = SOLVE_CONFIGS[0]
    t_q = _timed(lambda: jax.block_until_ready(_solve("quadrature", *cfg)[1]))
    t_t = _timed(lambda: jax.block_until_ready(_solve("table", *cfg)[1]))
    return t_q, t_t


def check_construction_speedup():
    """(d) Warm anisotropic MultiComponentCluster construction, table vs quad."""
    def build(method):
        m = MultiComponentCluster.from_components(**MODEL_KW, aniso_method=method)
        jax.block_until_ready(m._cdf_j)
    t_q = _timed(lambda: build("quadrature"))
    t_t = _timed(lambda: build("table"))
    return t_q, t_t


def check_gradients():
    """(e) AD vs central-FD gradient of a solve metric in (w_j, r_a).

    Test point: the Task-2 config-2 well (W0=5, ra_hat=5, strong anisotropy)
    where ALL four gradient components are large enough for central FD to
    resolve at rtol 1e-3. (At the milder W0=7/ra=10 point the ra_2 component
    is ~3e-3 -- below the FD noise floor set by the solver's ~1e-8 adaptive-
    stepping micro-structure; AD there was instead verified against the
    QUADRATURE-path AD to <= 3.1e-4.) FD steps are per-block: the w response
    is truncation-limited (small step), the weaker ra response is noise-
    limited (larger step). The quadrature-AD cross-check below is FD-free and
    covers the same point as the PASS criterion.
    """
    def metric_with(method):
        def metric(w, ra):
            _, psi, _ = solve_multicomponent_limepy(
                jnp.array([0.5, 0.5]), w ** -2.0, 5.0, G_PAR, xi_max=800.0,
                n_points=1500, ra_hat_j=ra, aniso_method=method)
            return jnp.mean(psi[:300])
        return metric

    metric = metric_with("table")
    w0 = jnp.array([1.0, 0.79])
    ra0 = jnp.array([5.0, 5.0])
    ad = np.concatenate([np.asarray(jax.grad(metric, 0)(w0, ra0)),
                         np.asarray(jax.grad(metric, 1)(w0, ra0))])
    metric_q = metric_with("quadrature")
    ad_q = np.concatenate([np.asarray(jax.grad(metric_q, 0)(w0, ra0)),
                           np.asarray(jax.grad(metric_q, 1)(w0, ra0))])
    x0 = np.concatenate([np.asarray(w0), np.asarray(ra0)])
    eps_rel = [3e-4, 3e-4, 1e-3, 1e-3]  # w: truncation-limited; ra: noise-limited
    fd = np.zeros(4)
    for i in range(4):
        eps = eps_rel[i] * max(1.0, abs(x0[i]))
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = float(metric(jnp.asarray(xp[:2]), jnp.asarray(xp[2:])))
        fm = float(metric(jnp.asarray(xm[:2]), jnp.asarray(xm[2:])))
        fd[i] = (fp - fm) / (2.0 * eps)
    return ad, fd, ad_q


def main():
    print("\n" + "=" * 70)
    print("LIMEPY DF-TABLE VALIDATION (accuracy / speedup / gradients)")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import matplotlib.pyplot as plt

    checks = []  # (name, passed, detail)

    # (a) pointwise density accuracy
    dens = check_density_accuracy()
    rel_fine = dens["512x96"][1].max()
    solver_lab = f"{_TAB_N_W}x{_TAB_N_P}"
    rel_solver = dens[solver_lab][1].max()
    checks.append(("density max rel err, 512x96 grid <= 1e-5",
                   rel_fine <= 1e-5, f"{rel_fine:.2e}"))
    print(f"  (a) density max rel err: 512x96 grid {rel_fine:.2e} "
          f"(budget 1e-5); solver {solver_lab} grid {rel_solver:.2e} (report)")

    # (b) solve accuracy across the three Task-2 configs
    solves = check_solve_accuracy()
    for cfg, _, dpsi in solves:
        W0 = cfg[0]
        ok = dpsi.max() <= 1e-4 * W0
        checks.append((f"max|dpsi| <= 1e-4*W0 (W0={W0}, ra={cfg[2][0]}, "
                       f"xi_max={cfg[3]:.0f})", ok, f"{dpsi.max():.2e}"))
        print(f"  (b) W0={W0} rescale={cfg[1]} ra={cfg[2]} xi_max={cfg[3]:.0f}: "
              f"max|dpsi|={dpsi.max():.2e} (budget {1e-4 * W0:.1e})")

    # (c) warm solve speedup
    t_q, t_t = check_solve_speedup()
    sp_solve = t_q / t_t
    checks.append(("warm solve speedup >= 3x", sp_solve >= 3.0,
                   f"{sp_solve:.1f}x ({t_q * 1e3:.0f} -> {t_t * 1e3:.0f} ms)"))
    print(f"  (c) warm solve: quadrature {t_q * 1e3:.0f} ms, table {t_t * 1e3:.0f} ms "
          f"-> {sp_solve:.1f}x")

    # (d) construction-level speedup (includes the Task-4 dedup)
    tc_q, tc_t = check_construction_speedup()
    sp_con = tc_q / tc_t
    checks.append(("aniso construction speedup >= 2x (report)", sp_con >= 2.0,
                   f"{sp_con:.1f}x ({tc_q * 1e3:.0f} -> {tc_t * 1e3:.0f} ms)"))
    print(f"  (d) aniso construction: quadrature {tc_q * 1e3:.0f} ms, table "
          f"{tc_t * 1e3:.0f} ms -> {sp_con:.1f}x (one shared table: solve + CDF)")

    # (e) AD vs FD gradients (+ FD-free quadrature-AD cross-check)
    ad, fd, ad_q = check_gradients()
    rel_g = np.abs(ad - fd) / np.maximum(np.abs(fd), 1e-12)
    rel_q = np.abs(ad - ad_q) / np.maximum(np.abs(ad_q), 1e-12)
    grad_ok = bool(np.all(np.isfinite(ad)) and np.all(rel_g <= 1e-3))
    checks.append(("AD = FD through table solve (w_j, r_a), rtol 1e-3",
                   grad_ok, f"max rel diff {rel_g.max():.2e}"))
    print("  (e) gradient d<psi>/d(w1,w2,ra1,ra2) at W0=5, ra=5:")
    for lab, a, f, r in zip(["w_1", "w_2", "ra_1", "ra_2"], ad, fd, rel_g):
        print(f"      {lab:>4}: AD {a:+.6e}  FD {f:+.6e}  rel diff {r:.1e}")
    print(f"      cross-check table-AD vs quadrature-AD (FD-free): "
          f"max rel diff {rel_q.max():.2e} (report)")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
    axA, axB, axC, axD = axes.ravel()

    for label, col in [("512x96", OI["blue"]), (solver_lab, OI["orange"])]:
        Wv, rel = dens[label]
        axA.plot(Wv, np.maximum(rel, 1e-12), ".", ms=2.5, color=col, alpha=0.5,
                 label=f"{label} grid (max {rel.max():.1e})")
    axA.axhline(1e-5, color=OI["vermilion"], ls="--", lw=1.0, label="budget 1e-5 (512x96)")
    axA.set_yscale("log")
    axA.set_xlabel(r"$W$")
    axA.set_ylabel(r"relative error of $\hat\rho(W,p)$ vs quadrature")
    axA.legend(fontsize=7, loc="lower right")
    panel_label(axA, "(a)")

    cols = [OI["blue"], OI["green"], OI["purple"]]
    for (cfg, xi, dpsi), col in zip(solves, cols):
        W0 = cfg[0]
        axB.plot(xi / cfg[3], np.maximum(dpsi, 1e-12), color=col, lw=1.2,
                 label=fr"$W_0={W0:.0f}$, $\hat r_a={cfg[2][0]:.0f}$")
        axB.axhline(1e-4 * W0, color=col, ls="--", lw=0.8, alpha=0.7)
    axB.set_yscale("log")
    axB.set_xscale("log")
    axB.set_xlim(min(x[1] / c[3] for c, x, _ in solves), 1.0)
    axB.set_xlabel(r"$\xi/\xi_{\max}$")
    axB.set_ylabel(r"$|\psi_{\rm table}-\psi_{\rm quad}|$")
    axB.legend(fontsize=7, title="dashed: $10^{-4} W_0$ budgets", title_fontsize=7)
    panel_label(axB, "(b)")

    labels = ["solve\n(quad)", "solve\n(table)", "construct\n(quad)", "construct\n(table)"]
    vals = np.array([t_q, t_t, tc_q, tc_t]) * 1e3
    bars = axC.bar(labels, vals, color=[OI["vermilion"], OI["blue"]] * 2, width=0.6)
    for b, v in zip(bars, vals):
        axC.text(b.get_x() + b.get_width() / 2, v * 1.04, f"{v:.0f} ms",
                 ha="center", fontsize=7.5)
    axC.text(0.04, 0.92, f"solve {sp_solve:.1f}x", transform=axC.transAxes,
             ha="left", fontsize=8.5, color=OI["black"])
    axC.text(0.04, 0.84, f"construction {sp_con:.1f}x\n(shared table)",
             transform=axC.transAxes, ha="left", va="top", fontsize=8.5)
    axC.set_ylabel("warm wall time [ms]")
    axC.set_ylim(0, vals.max() * 1.3)
    panel_label(axC, "(c)", loc="upper right")

    x = np.arange(4)
    axD.bar(x - 0.18, ad, width=0.36, color=OI["blue"], label="AD (jax.grad)")
    axD.bar(x + 0.18, fd, width=0.36, color=OI["orange"], label="FD (central)")
    axD.set_xticks(x)
    axD.set_xticklabels([r"$w_1$", r"$w_2$", r"$\hat r_{a,1}$", r"$\hat r_{a,2}$"])
    axD.set_ylabel(r"$\partial\langle\psi\rangle/\partial\theta$")
    axD.legend(fontsize=7)
    axD.text(0.97, 0.05, f"max rel diff {rel_g.max():.1e}",
             transform=axD.transAxes, ha="right", fontsize=8)
    panel_label(axD, "(d)")

    fig.tight_layout(pad=0.6)
    save_fig(fig, OUTPUT_DIR, "df_tables")

    # ---- pass/fail summary ----
    print("-" * 70)
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= bool(ok)
        print(f"  {name:<58} {detail:>20} {'PASS' if ok else 'FAIL'}")
    print(f"  saved df_tables.{{png,pdf}}")
    print("=" * 70)
    print("  DF-TABLE VALIDATION PASS" if all_ok else "  VALIDATION FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
