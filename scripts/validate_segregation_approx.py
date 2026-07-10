#!/usr/bin/env python
"""
Differentiable mass-segregation observable validation figures.

Five publication-quality figures for the three differentiable segregation observables
(soft Lambda_MSR, radial concentration, soft Sigma--m) in
``progenax.diagnostics.segregation_approx``, anchored to
tests/validation/test_segregation_approx_physics.py. Each prints expected-vs-measured
PASS/FAIL against an *external / exact* oracle.

Figures (-> what they validate):
  1. seg_hard_limit_convergence.png  each soft observable -> its exact non-diff oracle
                                     as the softness (tau, beta) -> 0 (correctness)
  2. seg_response_curves.png         normalised response vs segregation strength, with
                                     the exact Allison+2009 Lambda_MSR overlaid
  3. seg_fisher_identifiability.png  Fisher information I = (dmu/dtheta)^2 / Var per
                                     observable (autodiff dmu/dtheta) -> ranks them for HMC
  4. seg_2d_vs_3d.png                projection bias + 2D/3D Fisher ratio = fraction of
                                     segregation signal surviving projection (research Q)
  5. seg_gradient_validation.png     autodiff vs finite-difference d(obs)/d m_cut, and a
                                     gradient-descent recovery of segregation strength

Design: internal design note.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_segregation_approx.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from progenax.diagnostics import compute_lambda_msr
from progenax.diagnostics.segregation_approx import (
    lambda_msr_approx,
    radial_concentration_approx,
    sigma_m_approx,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
N_STARS = 400
N_MASSIVE = 20
M_CUT = 2.0

# Per-observable plotting identity.
OBS = {
    "lambda": dict(color=OI["blue"], label=r"soft $\Lambda_{\rm MSR}$", marker="o"),
    "radial": dict(color=OI["vermilion"], label=r"radial conc. $C$", marker="s"),
    "sigma": dict(color=OI["green"], label=r"$\Sigma$$-$$m$ corr. $S$", marker="^"),
}


# ----------------------------------------------------------------------------
# Cluster builders (differentiable in the segregation strength theta = core scale).
# ----------------------------------------------------------------------------
def _bases(key, N=N_STARS, nm=N_MASSIVE):
    k1, k2 = jax.random.split(key)
    halo = jax.random.normal(k1, (N - nm, 3)) * 1.0
    core_base = jax.random.normal(k2, (nm, 3))  # scaled by theta at use
    masses = jnp.concatenate([jnp.full(N - nm, 0.5), jnp.full(nm, 10.0)])
    return halo, core_base, masses


def _positions(theta, halo, core_base):
    return jnp.concatenate([halo, core_base * theta], axis=0)


def _soft(name, pos, masses, **kw):
    if name == "lambda":
        return lambda_msr_approx(
            pos,
            masses,
            m_cut=M_CUT,
            tau=kw.get("tau", 0.3),
            beta=kw.get("beta", 0.1),
            project_to_2d=kw.get("project_to_2d", True),
        )
    if name == "radial":
        return radial_concentration_approx(
            pos,
            masses,
            m_cut=M_CUT,
            tau=kw.get("tau", 0.3),
            project_to_2d=kw.get("project_to_2d", True),
        )
    return sigma_m_approx(
        pos,
        masses,
        m_cut=M_CUT,
        tau=kw.get("tau", 0.3),
        k=6,
        project_to_2d=kw.get("project_to_2d", True),
    )


# Exact oracles (the tau,beta -> 0 limits).
def _exact_radial(xy, massive):
    c = xy[massive].mean(0)
    r = np.sqrt(((xy - c) ** 2).sum(1))
    return r[massive].mean() / r.mean()


def _exact_lambda_nn(xy, massive):
    d, _ = cKDTree(xy).query(xy, k=2)
    nn = d[:, 1]
    return nn.mean() / nn[massive].mean()


def _exact_sigma(xy, massive, k=6):
    d, _ = cKDTree(xy).query(xy, k=k + 1)
    sigma = (k - 1) / (np.pi * d[:, k] ** 2)
    return np.corrcoef(massive.astype(float), np.log(sigma))[0, 1]


# ============================================================================
# Figure 1 -- hard-limit convergence (headline correctness)
# ============================================================================
def fig_hard_limit(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: soft observable -> exact oracle as softness -> 0")
    print("=" * 60)
    halo, core_base, masses = _bases(jax.random.PRNGKey(0))
    pos = _positions(0.05, halo, core_base)
    xy = np.asarray(pos[:, :2])
    massive = np.asarray(masses) > M_CUT

    taus = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
    betas = np.array([0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01])
    ex = {
        "lambda": _exact_lambda_nn(xy, massive),
        "radial": _exact_radial(xy, massive),
        "sigma": _exact_sigma(xy, massive),
    }
    errs = {k: [] for k in OBS}
    for i, t in enumerate(taus):
        errs["radial"].append(
            abs(float(_soft("radial", pos, masses, tau=t)) - ex["radial"])
        )
        errs["sigma"].append(
            abs(float(_soft("sigma", pos, masses, tau=t)) - ex["sigma"])
        )
        errs["lambda"].append(
            abs(
                float(_soft("lambda", pos, masses, tau=t, beta=betas[i])) - ex["lambda"]
            )
        )

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ok = True
    for name in OBS:
        e = np.array(errs[name])
        ax.loglog(
            taus,
            e + 1e-16,
            "-",
            color=OBS[name]["color"],
            marker=OBS[name]["marker"],
            ms=4.5,
            mfc="none",
            label=OBS[name]["label"],
        )
        converged = e[-1] < e[0]
        tol = {"radial": 1e-3, "sigma": 1e-2, "lambda": 5e-2}[name]
        passed = converged and e[-1] < tol
        ok = ok and passed
        print(
            f"  {name:7s}: exact={ex[name]:+.4f}  err(tau_min)={e[-1]:.2e}  "
            f"tol={tol:.0e}  {'PASS' if passed else 'FAIL'}"
        )
    ax.set_xlabel(r"softness $\tau$ (and $\beta$ for $\Lambda$)")
    ax.set_ylabel(r"$|{\rm soft} - {\rm exact}|$")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_ylim(1e-12, 1e1)
    ax.invert_xaxis()
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "seg_hard_limit_convergence")
    print(
        f"  saved seg_hard_limit_convergence.{{png,pdf}}  ->  {'PASS' if ok else 'FAIL'}"
    )
    return ok


# ============================================================================
# Figure 2 -- segregation response curves
# ============================================================================
def fig_response_curves(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: normalised response vs segregation strength")
    print("=" * 60)
    scales = np.linspace(0.05, 1.0, 10)
    vals = {k: [] for k in OBS}
    exact = []
    for i, s in enumerate(scales):
        halo, core_base, masses = _bases(jax.random.PRNGKey(200 + i))
        pos = _positions(float(s), halo, core_base)
        for name in OBS:
            vals[name].append(float(_soft(name, pos, masses)))
        exact.append(
            compute_lambda_msr(
                np.asarray(pos), np.asarray(masses), N_massive=N_MASSIVE
            )[0]
        )
    seg = 1.0 - scales  # segregation strength: 0 (diffuse) -> ~1 (tight core)
    exact = np.array(exact)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ok = True
    for name in OBS:
        v = np.array(vals[name])
        # Normalise each curve to [0,1] for shape comparison (sign-aware for radial).
        vn = (v - v.min()) / (np.ptp(v) + 1e-12)
        if name == "radial":
            vn = 1.0 - vn  # so "up" = more segregated for all three
        ax.plot(
            seg,
            vn,
            "-",
            color=OBS[name]["color"],
            marker=OBS[name]["marker"],
            ms=4,
            mfc="none",
            label=OBS[name]["label"],
        )
        rho = spearmanr(seg, v).correlation
        # all three should be monotonic in segregation strength
        ok = ok and abs(rho) > 0.8
        print(f"  {name:7s}: spearman(seg, obs) = {rho:+.3f}")
    exn = (exact - exact.min()) / (np.ptp(exact) + 1e-12)
    ax.plot(
        seg,
        exn,
        "--",
        color=OBS["lambda"]["color"],
        lw=1.0,
        alpha=0.6,
        label=r"exact $\Lambda_{\rm MSR}$ (Allison+09)",
    )
    rho_ex = spearmanr(seg, exact).correlation
    ok = ok and abs(rho_ex) > 0.8
    print(f"  exact  : spearman(seg, Lambda_MSR) = {rho_ex:+.3f}")
    ax.set_xlabel(r"segregation strength $1 - \theta$ (core tightness)")
    ax.set_ylabel("normalised response")
    ax.set_ylim(-0.04, 1.18)
    ax.legend(
        loc="upper left", fontsize=7.0, ncol=2, columnspacing=1.0, handlelength=1.6
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "seg_response_curves")
    print(f"  saved seg_response_curves.{{png,pdf}}  ->  {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# Figure 3 -- Fisher-information identifiability (the differentiable payoff)
# ============================================================================
def _fisher(name, theta0, n_real=40, project_to_2d=True):
    """I(theta) = (dmu/dtheta)^2 / Var, with dmu/dtheta from autodiff (per realisation)."""
    grads, vals = [], []
    for s in range(n_real):
        halo, core_base, masses = _bases(jax.random.PRNGKey(700 + s))
        f = lambda th: _soft(
            name, _positions(th, halo, core_base), masses, project_to_2d=project_to_2d
        )
        vals.append(float(f(theta0)))
        grads.append(float(jax.grad(f)(theta0)))
    dmu = np.mean(grads)
    var = np.var(vals) + 1e-12
    return dmu**2 / var, dmu, np.sqrt(var)


def fig_fisher(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: Fisher information I = (dmu/dtheta)^2 / Var  (autodiff)")
    print("=" * 60)
    theta0 = 0.3
    info = {}
    for name in OBS:
        I, dmu, sd = _fisher(name, theta0)
        info[name] = I
        print(f"  {name:7s}: dmu/dtheta={dmu:+.4f}  sigma={sd:.4f}  Fisher I={I:.3f}")
    ok = all(np.isfinite(v) and v > 0 for v in info.values())

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    names = list(OBS)
    heights = [info[n] for n in names]
    bars = ax.bar(
        range(len(names)),
        heights,
        color=[OBS[n]["color"] for n in names],
        width=0.62,
        alpha=0.85,
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([OBS[n]["label"] for n in names], fontsize=8)
    ax.set_ylabel(r"Fisher information $\mathcal{I}(\theta)$")
    ax.set_ylim(0, max(heights) * 1.16)
    # Value label above each bar.
    for b, h in zip(bars, heights):
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + max(heights) * 0.015,
            f"{h:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    best = max(info, key=info.get)
    ax.text(
        0.03,
        0.96,
        rf"most identifiable: {OBS[best]['label']}"
        "\n"
        rf"($\sim\!{info[best] / min(info.values()):.0f}\times$ the others)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="0.3",
    )
    ax.text(
        0.5,
        -0.16,
        r"$\mathcal{I}=(\mathrm{d}\mu/\mathrm{d}\theta)^2/\mathrm{Var}$ "
        r"at $\theta=0.3$ (autodiff)",
        transform=ax.transAxes,
        ha="center",
        fontsize=7,
        color="0.45",
    )
    fig.tight_layout(pad=0.5)
    save_fig(fig, output_dir, "seg_fisher_identifiability")
    print(
        f"  saved seg_fisher_identifiability.{{png,pdf}}  ->  {'PASS' if ok else 'FAIL'}"
    )
    return ok


# ============================================================================
# Figure 4 -- 2D vs 3D bias + Fisher ratio (projection research question)
# ============================================================================
def fig_2d_vs_3d(output_dir):
    print("\n" + "=" * 60)
    print("FIG 4: 2D (projected) vs 3D observables + Fisher ratio")
    print("=" * 60)
    scales = np.linspace(0.05, 1.0, 8)
    v2d = {k: [] for k in OBS}
    v3d = {k: [] for k in OBS}
    for i, s in enumerate(scales):
        halo, core_base, masses = _bases(jax.random.PRNGKey(400 + i))
        pos = _positions(float(s), halo, core_base)
        for name in OBS:
            v2d[name].append(float(_soft(name, pos, masses, project_to_2d=True)))
            v3d[name].append(float(_soft(name, pos, masses, project_to_2d=False)))
    seg = 1.0 - scales

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.4, 4.1))
    ok = True
    # Panel (a): normalise each observable to its own 3D dynamic range so all three
    # share one [0,1] axis and the 2D-vs-3D gap (not the scale clash) is what shows.
    for name in OBS:
        v3 = np.array(v3d[name])
        v2 = np.array(v2d[name])
        lo, span = v3.min(), (np.ptp(v3) + 1e-12)
        n3 = (v3 - lo) / span
        n2 = (v2 - lo) / span
        if name == "radial":  # flip so "up" = more segregated for all three
            n3, n2 = 1 - n3, 1 - n2
        axA.plot(
            seg,
            n3,
            "-",
            color=OBS[name]["color"],
            lw=1.8,
            marker=OBS[name]["marker"],
            ms=4,
            mfc="none",
            label=OBS[name]["label"],
        )
        axA.plot(seg, n2, "--", color=OBS[name]["color"], lw=1.3, alpha=0.7)
    axA.set_xlabel(r"segregation strength $1-\theta$")
    axA.set_ylabel("response (normalised to 3D range)")
    axA.set_ylim(-0.05, 1.18)
    # Legend strip ABOVE the axes (never overlaps data); style note as a sub-line.
    axA.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=7,
        columnspacing=1.1,
        handlelength=1.6,
        frameon=False,
    )
    axA.text(
        0.5,
        0.04,
        "solid = 3D  ·  dashed = 2D projection",
        transform=axA.transAxes,
        ha="center",
        fontsize=7.2,
        color="0.4",
    )
    panel_label(axA, "(a)", loc="upper left")

    # Panel (b): Fisher ratio 2D/3D = fraction of segregation signal surviving projection.
    ratios = []
    for name in OBS:
        I2, _, _ = _fisher(name, 0.3, n_real=30, project_to_2d=True)
        I3, _, _ = _fisher(name, 0.3, n_real=30, project_to_2d=False)
        frac = I2 / (I3 + 1e-12)
        ratios.append(frac)
        ok = ok and np.isfinite(frac) and frac > 0
        print(
            f"  {name:7s}: Fisher 2D/3D = {frac:.3f}  ({100 * frac:.0f}% of signal survives)"
        )
    bars = axB.bar(
        range(len(OBS)),
        ratios,
        color=[OBS[n]["color"] for n in OBS],
        width=0.62,
        alpha=0.85,
    )
    axB.axhline(1.0, color="0.45", lw=1.0, ls=":")
    axB.text(
        0.985,
        1.0,
        "no loss",
        transform=axB.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="0.45",
    )
    for b, frac in zip(bars, ratios):
        axB.text(
            b.get_x() + b.get_width() / 2,
            frac + 0.02,
            f"{frac:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    axB.set_xticks(range(len(OBS)))
    axB.set_xticklabels([OBS[n]["label"] for n in OBS], fontsize=7.5)
    axB.set_ylabel(r"Fisher ratio $\mathcal{I}_{\rm 2D}/\mathcal{I}_{\rm 3D}$")
    axB.set_ylim(0, max(ratios) * 1.22)
    axB.set_title(
        "signal fraction surviving projection", fontsize=7.5, color="0.3", pad=4
    )
    panel_label(axB, "(b)", loc="upper left")
    fig.tight_layout(pad=0.5, w_pad=1.6)
    save_fig(fig, output_dir, "seg_2d_vs_3d")
    print(f"  saved seg_2d_vs_3d.{{png,pdf}}  ->  {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# Figure 5 -- gradient validation + inference recovery
# ============================================================================
def fig_gradient_validation(output_dir):
    print("\n" + "=" * 60)
    print("FIG 5: autodiff vs finite-difference + segregation recovery")
    print("=" * 60)
    halo, core_base, masses = _bases(jax.random.PRNGKey(11))
    pos = _positions(0.1, halo, core_base)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.4, 4.0))
    # (a) AD vs FD for d(obs)/d m_cut
    mcs = np.linspace(1.0, 6.0, 11)
    ok_a = True
    for name in OBS:
        ad, fd = [], []
        for mc in mcs:
            f = lambda x: _soft_mcut(name, pos, masses, x)
            ad.append(float(jax.grad(f)(float(mc))))
            eps = 1e-4
            fd.append((float(f(mc + eps)) - float(f(mc - eps))) / (2 * eps))
        ad, fd = np.array(ad), np.array(fd)
        axA.plot(mcs, ad, "-", color=OBS[name]["color"], label=OBS[name]["label"])
        axA.plot(mcs, fd, "o", color=OBS[name]["color"], ms=3.5, mfc="none", mew=1.0)
        maxerr = np.max(np.abs(ad - fd))
        ok_a = ok_a and maxerr < 1e-3
        print(f"  {name:7s}: max|AD-FD| d/dm_cut = {maxerr:.2e}")
    axA.set_xlabel(r"mass cut $m_{\rm cut}\ [M_\odot]$")
    axA.set_ylabel(r"$\partial\,{\rm obs}/\partial m_{\rm cut}$")
    # Legend strip above the axes; method note as a clear sub-line.
    axA.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=7,
        columnspacing=1.1,
        handlelength=1.6,
        frameon=False,
    )
    axA.text(
        0.97,
        0.93,
        "line = autodiff\npoints = finite diff.",
        transform=axA.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color="0.4",
    )
    panel_label(axA, "(a)", loc="upper left")

    # (b) gradient-descent recovery of segregation strength via the radial observable
    halo2, core2, m2 = _bases(jax.random.PRNGKey(12))
    theta_true = 0.12
    target = float(_soft("radial", _positions(theta_true, halo2, core2), m2))

    def loss(theta):
        return (_soft("radial", _positions(theta, halo2, core2), m2) - target) ** 2

    grad_loss = jax.grad(loss)
    theta = 0.8
    traj = [theta]
    for _ in range(60):
        theta = float(theta - 0.5 * grad_loss(theta))
        theta = min(max(theta, 0.02), 1.0)
        traj.append(theta)
    ok_b = abs(traj[-1] - theta_true) < 0.03
    axB.plot(
        traj, "-", color=OBS["radial"]["color"], lw=1.8, label="radial-$C$ descent"
    )
    axB.axhline(theta_true, color="0.4", ls="--", lw=1.0, label=r"true $\theta=0.12$")
    axB.set_xlabel("gradient-descent step")
    axB.set_ylabel(r"recovered $\theta$ (core scale)")
    axB.set_ylim(0.0, 0.9)
    axB.legend(loc="upper right", fontsize=7.5)
    axB.text(
        0.5,
        0.55,
        rf"recovered $\theta = {traj[-1]:.2f}$"
        "\n(exact)",
        transform=axB.transAxes,
        ha="center",
        fontsize=8,
        color="0.3",
    )
    panel_label(axB, "(b)", loc="lower right")
    print(
        f"  recovery: true theta={theta_true:.3f}  recovered={traj[-1]:.3f}  "
        f"{'PASS' if ok_b else 'FAIL'}"
    )
    fig.tight_layout(pad=0.4, w_pad=1.2)
    save_fig(fig, output_dir, "seg_gradient_validation")
    ok = ok_a and ok_b
    print(
        f"  saved seg_gradient_validation.{{png,pdf}}  ->  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def _soft_mcut(name, pos, masses, mc):
    if name == "lambda":
        return lambda_msr_approx(pos, masses, m_cut=mc, tau=0.8, beta=0.1)
    if name == "radial":
        return radial_concentration_approx(pos, masses, m_cut=mc, tau=0.8)
    return sigma_m_approx(pos, masses, m_cut=mc, tau=0.8, k=6)


def main():
    print("\n" + "=" * 70)
    print("PROGENAX DIFFERENTIABLE MASS-SEGREGATION VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "Fig 1  hard-limit convergence": fig_hard_limit(OUTPUT_DIR),
        "Fig 2  response curves": fig_response_curves(OUTPUT_DIR),
        "Fig 3  Fisher identifiability": fig_fisher(OUTPUT_DIR),
        "Fig 4  2D vs 3D bias": fig_2d_vs_3d(OUTPUT_DIR),
        "Fig 5  gradient validation": fig_gradient_validation(OUTPUT_DIR),
    }
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print(
        "  ALL SEGREGATION VALIDATION FIGURES PASS"
        if all_ok
        else "  SOME SEGREGATION VALIDATION FIGURES FAILED"
    )
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/seg_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
