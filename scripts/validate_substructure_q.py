#!/usr/bin/env python
"""
Cartwright & Whitworth (2004) Q substructure-diagnostic validation figures.

Q = m_bar / s_bar separates centrally-concentrated clusters (Q > 0.8) from
fractal/substructured ones (Q < 0.8). Five publication-quality figures anchored to
passing tests in ``tests/validation/test_substructure_q_physics.py``, each printing
expected-vs-measured.

Two implementations: the exact scipy `compute_q_parameter` (MST-based) and the
JAX-differentiable `q_approx` (kNN, for substructure inference).

Figures (-> anchoring tests):
  1. q_mbar_sbar_plane.png      CW04 (s_bar, m_bar) plane with the Q=0.8 line
  2. q_ladder.png               Q across configs vs CW04 Table 1 anchors
  3. q_approx_vs_exact.png      differentiable q_approx vs exact (honest: over-read at Q>0.8)
  4. q_approx_gradient.png      AD vs central-FD d(q_approx)/d(concentration)
  5. q_n_convergence.png        Q ~ N-independent for N > 100 (CW04)

Reference: Cartwright & Whitworth (2004), MNRAS 348, 589.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_substructure_q.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

jax.config.update("jax_enable_x64", True)

from progenax.diagnostics.q_approx import q_approx
from progenax.diagnostics.substructure import compute_q_parameter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"


# --- configuration generators (mirror the validation tests) ---
def _uniform_sphere(n, seed):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, 1, n) ** (1 / 3)
    ct = rng.uniform(-1, 1, n)
    st = np.sqrt(1 - ct**2)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * st * np.cos(ph), r * st * np.sin(ph), r * ct])


def _concentrated(n, seed):  # r^-2 number density (CW04 3D2), Q ~ 0.93
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, 1, n)
    ct = rng.uniform(-1, 1, n)
    st = np.sqrt(1 - ct**2)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * st * np.cos(ph), r * st * np.sin(ph), r * ct])


def _r1(n, seed):  # r^-1 number density (CW04 3D1), Q ~ 0.84
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, n))
    ct = rng.uniform(-1, 1, n)
    st = np.sqrt(1 - ct**2)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * st * np.cos(ph), r * st * np.sin(ph), r * ct])


def _clumpy(n, seed, k=8, spread=0.06):
    rng = np.random.default_rng(seed)
    c = rng.uniform(-1, 1, (k, 3))
    c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)
    c *= rng.uniform(0.2, 1.0, (k, 1))
    return c[rng.integers(0, k, n)] + rng.normal(0, spread, (n, 3))


def _clumpier(n, seed):
    return _clumpy(n, seed, k=5, spread=0.03)


def _m_s_q(positions):
    """(m_bar, s_bar, Q) replicating compute_q_parameter's CW04 (A=pi R^2) formula."""
    xy = positions[:, :2]
    rel = xy - xy.mean(0)
    R = np.linalg.norm(rel, axis=1).max()
    pd = pdist(xy)
    s_bar = pd.mean() / R
    L = minimum_spanning_tree(squareform(pd)).sum()
    m_bar = L / np.sqrt(len(xy) * np.pi * R**2)
    return m_bar, s_bar, m_bar / s_bar


CONFIGS = [
    ("clumpier", _clumpier, OI["purple"]),
    ("clumpy", _clumpy, OI["vermilion"]),
    ("uniform", _uniform_sphere, OI["blue"]),
    (r"$r^{-1}$", _r1, OI["green"]),
    (r"$r^{-2}$", _concentrated, OI["orange"]),
]


# ============================================================================
# Figure 1 -- the CW04 (s_bar, m_bar) plane
# ============================================================================
def fig_plane(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: CW04 (s_bar, m_bar) plane")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    sgrid = np.linspace(0.3, 1.05, 50)
    ax.plot(
        sgrid, 0.8 * sgrid, "--", color="0.4", lw=1.3, label=r"$Q=0.8$ (CW04 divide)"
    )
    ax.fill_between(sgrid, 0.8 * sgrid, 1.05, color=OI["sky"], alpha=0.10)
    ax.fill_between(sgrid, 0.0, 0.8 * sgrid, color=OI["orange"], alpha=0.10)

    ok = True
    for name, gen, col in CONFIGS:
        mb, sb, Q = np.mean([_m_s_q(gen(400, s)) for s in range(5)], axis=0)
        ax.plot(
            sb,
            mb,
            "o",
            color=col,
            ms=8,
            mec="white",
            mew=0.6,
            label=rf"{name} ($Q={Q:.2f}$)",
        )
        print(f"  {name:10s}: s_bar={sb:.3f} m_bar={mb:.3f} Q={Q:.3f}")
    ax.text(0.32, 0.97, "centrally\nconcentrated", fontsize=7.5, color="0.35", va="top")
    ax.text(0.97, 0.16, "substructured", fontsize=7.5, color="0.35", ha="right")
    ax.set_xlabel(r"$\bar{s}$ (normalized mean separation)")
    ax.set_ylabel(r"$\bar{m}$ (normalized MST length)")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.1, 1.0)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "q_mbar_sbar_plane")
    print("  saved q_mbar_sbar_plane.{png,pdf}")
    return ok


# ============================================================================
# Figure 2 -- Q ladder vs CW04 Table 1
# ============================================================================
def fig_ladder(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: Q ladder vs CW04 Table 1")
    print("=" * 60)
    cw04 = {"uniform": 0.79, r"$r^{-1}$": 0.84, r"$r^{-2}$": 0.93}  # CW04 3D0/3D1/3D2

    names, Qmean, Qstd, cols = [], [], [], []
    for name, gen, col in CONFIGS:
        Qs = np.array([compute_q_parameter(gen(400, s)) for s in range(8)])
        names.append(name)
        Qmean.append(Qs.mean())
        Qstd.append(Qs.std())
        cols.append(col)
        ref = f"  (CW04 {cw04[name]})" if name in cw04 else ""
        print(f"  {name:10s}: Q = {Qs.mean():.3f} +- {Qs.std():.3f}{ref}")
    ok = Qmean[0] < Qmean[2] < Qmean[-1]  # clumpier < uniform < r^-2

    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    xpos = np.arange(len(names))
    ax.bar(
        xpos,
        Qmean,
        yerr=Qstd,
        color=cols,
        alpha=0.85,
        capsize=3,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(0.8, color="0.3", ls="--", lw=1.3, label=r"$Q=0.8$ (CW04 divide)")
    for nm, xx in [("uniform", 2), (r"$r^{-1}$", 3), (r"$r^{-2}$", 4)]:
        ax.plot(xx, cw04[nm], "k*", ms=9, zorder=5)
    ax.plot([], [], "k*", ms=9, label="CW04 Table 1")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(r"Cartwright-Whitworth $Q$")
    ax.set_ylim(0.3, 1.0)
    ax.text(
        0.02,
        0.97,
        "concentrated",
        transform=ax.transAxes,
        fontsize=7.5,
        color="0.35",
        va="top",
    )
    ax.text(
        0.02, 0.10, "substructured", transform=ax.transAxes, fontsize=7.5, color="0.35"
    )
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "q_ladder")
    print(f"  ordering clumpier<uniform<r^-2: {ok}  -> {'PASS' if ok else 'FAIL'}")
    print("  saved q_ladder.{png,pdf}")
    return ok


# ============================================================================
# Figure 3 -- differentiable q_approx vs exact
# ============================================================================
def fig_approx_vs_exact(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: differentiable q_approx vs exact CW04 Q")
    print("=" * 60)

    ex, ap, cols, names = [], [], [], []
    for name, gen, col in CONFIGS:
        e = np.mean([compute_q_parameter(gen(400, s)) for s in range(5)])
        a = np.mean([float(q_approx(jnp.asarray(gen(400, s)))) for s in range(5)])
        ex.append(e)
        ap.append(a)
        cols.append(col)
        names.append(name)
        print(f"  {name:10s}: exact={e:.3f} q_approx={a:.3f} diff={a - e:+.3f}")
    ex, ap = np.array(ex), np.array(ap)
    # faithful where Q<=0.85 (substructure regime); over-reads above
    faithful = np.all(np.abs(ap[ex <= 0.85] - ex[ex <= 0.85]) < 0.06)
    print(
        f"  faithful (Q<=0.85, |diff|<0.06): {faithful}  -> {'PASS' if faithful else 'FAIL'}"
    )

    fig, ax = plt.subplots(figsize=(3.9, 3.7))
    lim = [0.15, 1.08]
    ax.plot(lim, lim, "-", color="0.5", lw=1.2, label="1:1")
    for e, a, c, n in zip(ex, ap, cols, names):
        ax.plot(e, a, "o", color=c, ms=8, mec="white", mew=0.6, label=n)
    ax.axvspan(0.15, 0.85, color=OI["sky"], alpha=0.08)
    ax.text(
        0.45,
        1.05,
        "faithful regime\n(substructure)",
        fontsize=7.5,
        color="0.35",
        va="top",
    )
    ax.set_xlabel(r"exact $Q$ (MST, scipy)")
    ax.set_ylabel(r"differentiable $q_{\rm approx}$ (kNN, JAX)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "q_approx_vs_exact")
    print("  saved q_approx_vs_exact.{png,pdf}")
    return faithful


# ============================================================================
# Figure 4 -- q_approx gradient (AD vs FD)
# ============================================================================
def fig_gradient(output_dir):
    print("\n" + "=" * 60)
    print("FIG 4: q_approx gradient d(q)/d(concentration), AD vs FD")
    print("=" * 60)
    rng = np.random.default_rng(3)
    dirs = rng.normal(0, 1, (500, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    u = jnp.asarray(rng.uniform(0.02, 1.0, 500))
    D = jnp.asarray(dirs)

    def q_of_p(p):
        return q_approx((u**p)[:, None] * D)

    ps = np.linspace(0.4, 1.0, 13)
    ad = np.array([float(jax.grad(q_of_p)(float(p))) for p in ps])
    fd = np.array(
        [float((q_of_p(float(p) + 5e-4) - q_of_p(float(p) - 5e-4)) / 1e-3) for p in ps]
    )
    relarr = np.abs(ad - fd) / (np.abs(ad) + np.abs(fd) + 1e-30)
    rel_med, rel_max = float(np.median(relarr)), float(np.max(relarr))
    qvals = np.array([float(q_of_p(float(p))) for p in ps])
    # q_approx is a kNN estimator: the gradient is finite, correct-sign and usable,
    # with typical (median) AD-FD agreement of a few % and worst-case cell-boundary
    # spikes -- NOT machine precision (cf. the smooth profile gradients).
    correct_sign = np.all(ad > 0)  # concentration raises Q -> positive slope
    passed = np.all(np.isfinite(ad)) and correct_sign and rel_med < 0.06
    print(
        f"  q(p): {qvals[0]:.2f} -> {qvals[-1]:.2f} (concentration raises Q; all AD>0: {correct_sign})"
    )
    print(
        f"  AD-vs-FD rel: median={rel_med:.2e}, max={rel_max:.2e} (kNN cell-boundary)  "
        f"-> {'PASS' if passed else 'FAIL'}"
    )

    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    ax.plot(ps, ad, "-", color=OI["blue"], lw=1.8, label="autodiff", zorder=2)
    ax.plot(
        ps,
        fd,
        "o",
        color=OI["vermilion"],
        ms=4.5,
        mfc="none",
        mew=1.2,
        label="finite diff",
        zorder=3,
    )
    ax.set_xlabel(r"concentration exponent $p$  ($r=u^p$)")
    ax.set_ylabel(r"$\partial\,q_{\rm approx} / \partial p$")
    ax.legend(loc="best")
    ax.text(
        0.5,
        0.05,
        rf"median rel $={rel_med:.0e}$ (kNN; finite, correct sign)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5),
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "q_approx_gradient")
    print("  saved q_approx_gradient.{png,pdf}")
    return passed


# ============================================================================
# Figure 5 -- N-independence
# ============================================================================
def fig_n_convergence(output_dir):
    print("\n" + "=" * 60)
    print("FIG 5: Q ~ N-independent for N > 100 (CW04)")
    print("=" * 60)
    Ns = [80, 150, 300, 600, 1200, 2400]
    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    ok = True
    for name, gen, col, ref in [
        ("uniform", _uniform_sphere, OI["blue"], 0.79),
        ("clumpy", _clumpy, OI["vermilion"], None),
    ]:
        mean, std = [], []
        for N in Ns:
            Qs = np.array([compute_q_parameter(gen(N, s)) for s in range(5)])
            mean.append(Qs.mean())
            std.append(Qs.std())
        mean, std = np.array(mean), np.array(std)
        ax.errorbar(
            Ns,
            mean,
            yerr=std,
            fmt="o-",
            color=col,
            ms=4.5,
            capsize=2,
            lw=1.2,
            label=name,
        )
        big = mean[np.array(Ns) > 100]
        spread = (big.max() - big.min()) / big.mean()
        ok = ok and spread < 0.25
        print(
            f"  {name:8s}: Q(N>100) spread = {spread:.1%}  "
            f"-> {'PASS' if spread < 0.25 else 'FAIL'}"
        )
    ax.axhline(0.79, color="0.5", ls=":", lw=1.0, label="CW04 3D0 (0.79)")
    ax.axvline(100, color="0.7", ls="--", lw=0.9)
    ax.text(105, 0.55, r"$N>100$", fontsize=8, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N$ (number of stars)")
    ax.set_ylabel(r"Cartwright-Whitworth $Q$")
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "q_n_convergence")
    print("  saved q_n_convergence.{png,pdf}")
    return ok


def main():
    print("\n" + "=" * 70)
    print("PROGENAX CW04 Q SUBSTRUCTURE-DIAGNOSTIC VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "Fig 1  CW04 (s_bar, m_bar) plane": fig_plane(OUTPUT_DIR),
        "Fig 2  Q ladder vs CW04 Table 1": fig_ladder(OUTPUT_DIR),
        "Fig 3  q_approx vs exact": fig_approx_vs_exact(OUTPUT_DIR),
        "Fig 4  q_approx gradient (AD vs FD)": fig_gradient(OUTPUT_DIR),
        "Fig 5  Q N-independence": fig_n_convergence(OUTPUT_DIR),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print("  ALL CW04 Q VALIDATION FIGURES PASS" if all_ok else "  SOME FIGURES FAILED")
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/q_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
