#!/usr/bin/env python
"""
Environment-dependent IMF validation figures (Marks+2012 / Jeřábková+2018).

Five publication-quality figures anchored to tests/validation/test_environment_physics.py
(12 tests) and tests/unit/imf/test_environment.py (77 tests). Each prints
expected-vs-measured PASS/FAIL against *published* oracles (Marks+2012 Table 1
globular-cluster alpha3, Table 4 low-mass slopes) or central finite differences.

Figures (-> what they validate):
  1. env_gc_anchors.png       predicted vs published alpha3 for 4 Marks+2012 Table 1
                              GCs (NGC 104/6341/6752/7078); NGC 7078 most top-heavy
  2. env_fundamental_plane.png  alpha3 over the (log rho_cl, [Fe/H]) plane with the
                              GCs overplotted -- density dominates metallicity ~7:1
  3. env_lowmass_slopes.png   alpha1, alpha2 vs [Fe/H] (Marks Eq. 12) vs Table 4 anchors
  4. env_marks_vs_jerabkova.png  Marks Fundamental Plane vs Jeřábková IGIMF alpha3(rho)
  5. env_gradient_validation.png  AD vs central-FD for d alpha3 / d[Fe/H], d/d log rho

Provenance: the Marks+2012 and Jeřábková+2018 paper PDFs are held in
docs/core-papers/ (Marks-IMF-mnras-2012.pdf, Jerabkova-IMF-aa-2018.pdf); the
coefficients are grounded in those papers (see the per-paper notes).

References:
    Marks et al. (2012), MNRAS 422, 2246; Jeřábková et al. (2018), A&A 620, A39.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_environment.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from progenax.imf.environment.mapping import (
    alpha3_jerabkova_rho,
    alpha3_marks_plane,
    lowmass_slopes_metallicity,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"

# Marks+2012 Table 1: (name, [Fe/H], rho_cl [1e6 Msun/pc^3], published alpha3)
GCS = [
    ("NGC 104", -0.76, 9.54, 1.34),
    ("NGC 6341", -2.28, 66.03, 1.11),
    ("NGC 6752", -1.56, 31.78, 1.27),
    ("NGC 7078", -2.16, 258.13, 0.76),
]
# Marks+2012 Table 4: [Fe/H] -> (alpha1, alpha2)
TABLE4 = [(-2.0, 0.30, 1.30), (-1.5, 0.55, 1.55), (-1.0, 0.80, 1.80),
          (0.0, 1.30, 2.30), (0.5, 1.55, 2.55)]


def _a3(rho_1e6, feh):
    return float(alpha3_marks_plane(jnp.log10(jnp.asarray(rho_1e6)), jnp.asarray(feh)))


# ============================================================================
# Figure 1 -- globular-cluster anchors (headline)
# ============================================================================
def fig_gc_anchors(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: predicted vs published alpha3 (Marks+2012 Table 1 GCs)")
    print("=" * 60)
    names = [g[0] for g in GCS]
    pub = np.array([g[3] for g in GCS])
    comp = np.array([_a3(g[2], g[1]) for g in GCS])
    dev = np.abs(comp - pub)
    ok = bool(np.all(dev < 0.20))
    most_top_heavy = names[int(np.argmin(comp))] == "NGC 7078"
    for n, p, c, d in zip(names, pub, comp, dev):
        print(f"  {n:9s} published {p:.2f}  computed {c:.3f}  |dev|={d:.3f} "
              f"(tol 0.20)  -> {'PASS' if d < 0.20 else 'FAIL'}")
    print(f"  NGC 7078 most top-heavy: {most_top_heavy}  "
          f"-> {'PASS' if most_top_heavy else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    lim = (0.5, 1.6)
    ax.plot(lim, lim, "-", color="0.6", lw=1.0, zorder=0, label="$y=x$")
    xs = np.linspace(*lim, 10)
    ax.fill_between(xs, xs - 0.20, xs + 0.20, color=OI["green"], alpha=0.10,
                    zorder=0, label=r"$\pm0.20$ (Table 1 scatter)")
    for n, p, c in zip(names, pub, comp):
        col = OI["vermilion"] if n == "NGC 7078" else OI["blue"]
        ax.plot(p, c, "o", color=col, ms=7, mec="white", mew=0.6, zorder=3)
        ax.annotate(n, (p, c), textcoords="offset points", xytext=(6, -3),
                    fontsize=6.5, color=col)
    ax.set_xlabel(r"published $\alpha_3$ (Marks+2012 Table 1)")
    ax.set_ylabel(r"predicted $\alpha_3$ (Fundamental Plane)")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.96, 0.05, "lower $\\alpha_3$ = top-heavy", transform=ax.transAxes,
            ha="right", fontsize=7, color="0.4")
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "env_gc_anchors")
    print("  saved env_gc_anchors.{png,pdf}")
    return ok and most_top_heavy


# ============================================================================
# Figure 2 -- the Fundamental Plane alpha3(log rho, [Fe/H])
# ============================================================================
def fig_fundamental_plane(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: alpha3 over the (log rho_cl, [Fe/H]) Fundamental Plane")
    print("=" * 60)
    lr = np.linspace(0.0, 3.0, 200)   # log10(rho / 1e6)
    feh = np.linspace(-2.5, 0.5, 200)
    LR, FEH = np.meshgrid(lr, feh)
    A3 = np.array(alpha3_marks_plane(jnp.asarray(LR), jnp.asarray(FEH)))

    # density vs metallicity response (1 dex each) at a representative point
    d_rho = _a3(30.0, -1.5) - _a3(300.0, -1.5)
    d_feh = _a3(30.0, -1.5) - _a3(30.0, -0.5)
    ratio = abs(d_rho) / abs(d_feh)
    dominates = ratio > 3.0
    print(f"  1-dex response: d(alpha3)/d(log rho)={d_rho:+.3f}, "
          f"d(alpha3)/d[Fe/H]={d_feh:+.3f}, ratio={ratio:.1f}  "
          f"-> {'PASS' if dominates else 'FAIL'} (density dominates)")

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    pc = ax.pcolormesh(lr, feh, A3, shading="auto", cmap="viridis", rasterized=True)
    cs = ax.contour(lr, feh, A3, levels=[0.8, 1.0, 1.2, 1.6, 2.0], colors="white",
                    linewidths=0.7)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
    for n, fe, rho, _ in GCS:
        ax.plot(np.log10(rho), fe, "o", color=OI["vermilion"], ms=6, mec="white", mew=0.8)
        ax.annotate(n.replace("NGC ", ""), (np.log10(rho), fe),
                    textcoords="offset points", xytext=(5, 2), fontsize=6,
                    color="white", fontweight="bold")
    fig.colorbar(pc, ax=ax, label=r"$\alpha_3$ (high-mass slope)")
    ax.set_xlabel(r"$\log_{10}(\rho_{\rm cl} / 10^6\,M_\odot\,{\rm pc}^{-3})$")
    ax.set_ylabel(r"[Fe/H]")
    ax.text(0.04, 0.06, rf"density $\Rightarrow$ {ratio:.0f}$\times$ metallicity",
            transform=ax.transAxes, fontsize=7, color="white")
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "env_fundamental_plane")
    print("  saved env_fundamental_plane.{png,pdf}")
    return dominates


# ============================================================================
# Figure 3 -- low-mass slopes vs metallicity
# ============================================================================
def fig_lowmass_slopes(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: low-mass slopes alpha1, alpha2 vs [Fe/H] (Marks Eq. 12)")
    print("=" * 60)
    feh = np.linspace(-2.5, 0.5, 200)
    a1 = np.array(jax.vmap(lambda f: lowmass_slopes_metallicity(f)[0])(jnp.asarray(feh)))
    a2 = np.array(jax.vmap(lambda f: lowmass_slopes_metallicity(f)[1])(jnp.asarray(feh)))

    worst = 0.0
    for fe, a1e, a2e in TABLE4:
        c1, c2 = lowmass_slopes_metallicity(jnp.asarray(fe))
        d = max(abs(float(c1) - a1e), abs(float(c2) - a2e))
        worst = max(worst, d)
        print(f"  FeH={fe:+.1f}: a1={float(c1):.3f}(pub {a1e}) "
              f"a2={float(c2):.3f}(pub {a2e})  max|dev|={d:.4f}")
    ok = worst < 0.02
    print(f"  worst Table-4 deviation {worst:.4f} (tol 0.02)  "
          f"-> {'PASS' if ok else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ax.plot(feh, a1, "-", color=OI["blue"], lw=1.8, label=r"$\alpha_1$ ($0.08$-$0.5\,M_\odot$)")
    ax.plot(feh, a2, "-", color=OI["vermilion"], lw=1.8, label=r"$\alpha_2$ ($0.5$-$1\,M_\odot$)")
    t_feh = [t[0] for t in TABLE4]
    ax.plot(t_feh, [t[1] for t in TABLE4], "o", color=OI["blue"], ms=6, mec="white",
            mew=0.6, label="Table 4 anchors", zorder=4)
    ax.plot(t_feh, [t[2] for t in TABLE4], "s", color=OI["vermilion"], ms=6, mec="white",
            mew=0.6, zorder=4)
    ax.axhline(2.3, color="0.6", ls=":", lw=1.0)
    ax.text(0.5, 2.33, "canonical $2.3$", fontsize=6.5, color="0.4")
    ax.set_xlabel(r"[Fe/H]"); ax.set_ylabel(r"low-mass slope $\alpha$")
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.96, 0.05, r"metal-poor $\Rightarrow$ bottom-light",
            transform=ax.transAxes, ha="right", fontsize=7, color="0.4")
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "env_lowmass_slopes")
    print("  saved env_lowmass_slopes.{png,pdf}")
    return ok


# ============================================================================
# Figure 4 -- Marks Fundamental Plane vs Jeřábková IGIMF
# ============================================================================
def fig_marks_vs_jerabkova(output_dir):
    print("\n" + "=" * 60)
    print("FIG 4: Marks Fundamental Plane vs Jeřábková IGIMF alpha3(rho)")
    print("=" * 60)
    lr = np.linspace(0.0, 3.0, 200)
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    max_gap = 0.0
    both_mono = True
    for feh, col in [(-2.0, OI["purple"]), (-1.0, OI["green"]), (0.0, OI["orange"])]:
        am = np.array(alpha3_marks_plane(jnp.asarray(lr), jnp.asarray(feh)))
        aj = np.array(alpha3_jerabkova_rho(jnp.asarray(lr), jnp.asarray(feh)))
        max_gap = max(max_gap, float(np.max(np.abs(am - aj))))
        # robust, validated claim: BOTH parameterizations are top-heavy with density
        both_mono &= bool(np.all(np.diff(am) <= 1e-9) and np.all(np.diff(aj) <= 1e-9))
        ax.plot(lr, am, "-", color=col, lw=1.7, label=rf"Marks [Fe/H]$={feh:+.0f}$")
        ax.plot(lr, aj, "--", color=col, lw=1.3, alpha=0.8)
    # HONEST: the two published parameterizations agree on the MECHANISM (monotone
    # top-heavy with density) but differ quantitatively in zero-point -- Jeřábková is
    # systematically the more top-heavy. We validate the robust qualitative agreement
    # and REPORT the divergence; we do not claim quantitative agreement they lack.
    passed = both_mono
    print(f"  both models monotone top-heavy with density: {both_mono}  "
          f"-> {'PASS' if both_mono else 'FAIL'}")
    print(f"  cross-model zero-point divergence (reported, not a pass/fail): "
          f"max |Marks - Jeřábková| = {max_gap:.2f} (Jeřábková more top-heavy)")

    ax.plot([], [], "-", color="0.4", lw=1.7, label="Marks plane (solid)")
    ax.plot([], [], "--", color="0.4", lw=1.3, label="Jeřábková IGIMF (dashed)")
    ax.set_xlabel(r"$\log_{10}(\rho_{\rm cl} / 10^6\,M_\odot\,{\rm pc}^{-3})$")
    ax.set_ylabel(r"$\alpha_3$")
    ax.legend(loc="lower left", fontsize=6.5, ncol=1)
    ax.text(0.96, 0.93, "same mechanism\n(top-heavy w/ density);\n"
            rf"differ $\leq{max_gap:.1f}$ in zero-point",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="0.4")
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "env_marks_vs_jerabkova")
    print("  saved env_marks_vs_jerabkova.{png,pdf}")
    return passed


# ============================================================================
# Figure 5 -- gradient validation
# ============================================================================
def _ad_fd(f, xs, h):
    ad = np.array([float(jax.grad(f)(float(x))) for x in xs])
    fd = np.array([float((f(float(x) + h) - f(float(x) - h)) / (2 * h)) for x in xs])
    rel = np.abs(ad - fd) / (np.abs(ad) + np.abs(fd) + 1e-30)
    return ad, fd, rel


def fig_gradient_validation(output_dir):
    print("\n" + "=" * 60)
    print("FIG 5: gradient validation (AD vs FD) -- alpha3 differentiability")
    print("=" * 60)
    # smooth=True gives a tanh-smoothed threshold (continuous gradients everywhere)
    specs = [
        ("FeH", r"[Fe/H]", r"$\partial\alpha_3/\partial$[Fe/H]",
         lambda fe: alpha3_marks_plane(jnp.log10(jnp.asarray(40.0)), fe, smooth=True),
         np.linspace(-2.4, 0.4, 13), 1e-4),
        ("logrho", r"$\log_{10}\rho_6$", r"$\partial\alpha_3/\partial\log\rho$",
         lambda lr: alpha3_marks_plane(lr, jnp.asarray(-1.5), smooth=True),
         np.linspace(0.2, 2.8, 13), 1e-4),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    worst = 0.0
    for ax, (key, xlab, ylab, f, xs, h), tag in zip(axes, specs, "ab"):
        ad, fd, rel = _ad_fd(f, xs, h)
        worst = max(worst, float(np.max(rel)))
        ax.plot(xs, ad, "-", color=OI["blue"], lw=1.8, label="autodiff", zorder=2)
        ax.plot(xs, fd, "o", color=OI["vermilion"], ms=4.5, mfc="none", mew=1.1,
                label="finite diff", zorder=3)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.legend(loc="best", fontsize=7.2)
        ax.text(0.5, 0.05, rf"max rel err $={np.max(rel):.0e}$", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5))
        panel_label(ax, f"({tag})", loc="upper right")
        print(f"  d alpha3/d{key}: max rel err {np.max(rel):.2e}  "
              f"-> {'DIFFERENTIABLE' if np.max(rel) < 1e-3 else 'CHECK'}")
    passed = worst < 1e-3
    print(f"  overall worst rel err {worst:.2e} (tol 1e-3)  "
          f"-> {'PASS' if passed else 'FAIL'}")
    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "env_gradient_validation")
    print("  saved env_gradient_validation.{png,pdf}")
    return passed


def main():
    print("\n" + "=" * 70)
    print("PROGENAX ENVIRONMENT-DEPENDENT IMF VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "Fig 1  GC anchors (Marks Table 1)": fig_gc_anchors(OUTPUT_DIR),
        "Fig 2  Fundamental Plane": fig_fundamental_plane(OUTPUT_DIR),
        "Fig 3  low-mass slopes (Table 4)": fig_lowmass_slopes(OUTPUT_DIR),
        "Fig 4  Marks vs Jeřábková": fig_marks_vs_jerabkova(OUTPUT_DIR),
        "Fig 5  gradient validation": fig_gradient_validation(OUTPUT_DIR),
    }
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print("  ALL ENVIRONMENT-IMF VALIDATION FIGURES PASS" if all_ok
          else "  SOME ENVIRONMENT-IMF VALIDATION FIGURES FAILED")
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/env_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
