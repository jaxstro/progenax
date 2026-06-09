#!/usr/bin/env python
"""
Tidal-truncation validation figures (Jacobi radius + apply_tidal_truncation).

Five publication-quality figures anchored to tests/validation/test_tidal_physics.py
(9 tests). Each prints expected-vs-measured PASS/FAIL against an *independent* oracle:
the inner Lagrange point L1 of the full restricted three-body problem, the tidal
force-balance condition, the Keplerian-vs-isothermal tidal-tensor factor, and the
analytic Plummer enclosed-mass profile.

Figures (-> what they validate):
  1. tidal_jacobi_vs_l1.png    King/Hill r_J = R(m/3M_g)^(1/3) vs the numerically
                               solved L1 point; the Hill limit as m/M_g -> 0
  2. tidal_force_balance.png   self-gravity G m/r^2 vs tidal+centrifugal 3 Omega^2 r
                               crossing at r_J; the rotating-frame effective potential
  3. tidal_keplerian_vs_iso.png  point-mass (factor 3) vs flat-rotation-curve (factor 2)
                               hosts -> r_J,iso / r_J,point = (3/2)^(1/3)
  4. tidal_plummer_truncation.png  bound mass vs r_t tracks the analytic Plummer M(<r);
                               spatial truncation at r_t = r_J; fill-factor regimes
  5. tidal_gradient_validation.png  AD vs analytic d r_J/d M_c (exact) and the
                               straight-through surrogate d(M_bound)/d r_t vs the
                               analytic Plummer shell mass

References:
    King (1962) AJ 67, 471; Binney & Tremaine (2008) Sec. 8.3.1; Spitzer (1987).

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_tidal.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR
from progenax.profiles import PlummerProfile
from progenax.tidal import apply_tidal_truncation, jacobi_radius, jacobi_radius_isothermal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = STELLAR.G  # pc^3 Msun^-1 Myr^-2
KMS_TO_PCMYR = 1.0227121651


def _l1_distance(m, M_g, R):
    """Distance from the cluster centre to L1 (full restricted-3-body force balance)."""
    Omega2 = G * (M_g + m) / R**3
    x_com = M_g * R / (M_g + m)
    F = lambda x: -G * m / x**2 + G * M_g / (R - x) ** 2 + Omega2 * (x - x_com)
    r_hill = R * (m / (3.0 * M_g)) ** (1.0 / 3.0)
    return scipy.optimize.brentq(F, 1e-6 * r_hill, 0.9 * R)


# ============================================================================
# Figure 1 -- Jacobi radius vs the L1 Lagrange point (headline)
# ============================================================================
def fig_jacobi_vs_l1(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: King/Hill r_J vs the L1 Lagrange point (m/M_g -> 0)")
    print("=" * 60)
    M_g, R = 1e11, 8000.0
    ratios = np.logspace(-9, -2, 25)
    rJ = np.array([float(jacobi_radius(rho * M_g, M_g, R)) for rho in ratios])
    l1 = np.array([_l1_distance(rho * M_g, M_g, R) for rho in ratios])
    rel = np.abs(rJ - l1) / l1
    realistic = rel[np.argmin(np.abs(ratios - 1e-7))]
    real_ok = realistic < 1e-2
    # the leading L1 correction to the Hill radius scales as (m/M_g)^(1/3), so the
    # relative error is a power law of slope ~1/3 (an independent scaling oracle)
    slope = float(np.polyfit(np.log10(ratios), np.log10(rel), 1)[0])
    slope_ok = abs(slope - 1.0 / 3.0) < 0.03
    print(f"  m/M_g=1e-7: |r_J - L1|/L1 = {realistic:.2e} (tol 1e-2)  "
          f"-> {'PASS' if real_ok else 'FAIL'}")
    print(f"  rel err {rel[-1]:.1e} (m/M_g=1e-2) -> {rel[0]:.1e} (m/M_g=1e-9); "
          f"log-log slope {slope:.3f} (expect 1/3 = Hill correction)  "
          f"-> {'PASS' if slope_ok else 'FAIL'}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.3, 3.1))
    axA.loglog(ratios, rJ, "-", color=OI["black"], lw=1.8, label=r"King/Hill $r_J$")
    axA.loglog(ratios, l1, "o", color=OI["vermilion"], ms=4, mfc="none", mew=1.0,
               label="L1 (restricted 3-body)")
    axA.set_xlabel(r"$m / M_{\rm g}$"); axA.set_ylabel(r"tidal radius [pc]")
    axA.legend(loc="upper left", fontsize=7.5)
    panel_label(axA, "(a)", loc="lower right")

    ref = rel[-1] * (ratios / ratios[-1]) ** (1.0 / 3.0)  # (m/M_g)^(1/3) guide
    axB.loglog(ratios, ref, "--", color="0.55", lw=1.1, label=r"$\propto(m/M_g)^{1/3}$")
    axB.loglog(ratios, rel, "o-", color=OI["blue"], ms=4, lw=1.2, label="measured")
    axB.axvline(1e-7, color="0.5", ls=":", lw=1.0)
    axB.text(1.3e-7, 3e-3, "Galactic GC", fontsize=6.5, color="0.4", rotation=90,
             va="bottom")
    axB.set_xlabel(r"$m / M_{\rm g}$")
    axB.set_ylabel(r"$|r_J - r_{\rm L1}| / r_{\rm L1}$")
    axB.legend(loc="upper left", fontsize=7)
    axB.text(0.5, 0.06, "Hill correction\n" + rf"slope $={slope:.2f}$",
             transform=axB.transAxes, ha="center", fontsize=7, color="0.4")
    panel_label(axB, "(b)", loc="lower right")

    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "tidal_jacobi_vs_l1")
    print("  saved tidal_jacobi_vs_l1.{png,pdf}")
    return real_ok and slope_ok


# ============================================================================
# Figure 2 -- tidal force balance + effective potential
# ============================================================================
def fig_force_balance(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: tidal force balance + rotating-frame effective potential")
    print("=" * 60)
    m, M_g, R = 1e6, 1e11, 8000.0
    r_J = float(jacobi_radius(m, M_g, R))
    Omega2 = G * M_g / R**3
    r = np.linspace(0.2 * r_J, 2.0 * r_J, 300)
    self_grav = G * m / r**2          # inward (cluster binding)
    tidal = 3.0 * Omega2 * r          # outward (tidal + centrifugal, point mass)
    cross = float(r[np.argmin(np.abs(self_grav - tidal))])
    bal_pass = abs(cross - r_J) / r_J < 0.02
    print(f"  self-gravity == 3 Omega^2 r at r = {cross:.2f} pc (r_J = {r_J:.2f})  "
          f"-> {'PASS' if bal_pass else 'FAIL'}")

    # effective potential along the cluster-galaxy line (L1 is its local max)
    x_com = M_g * R / (M_g + m)
    x = np.linspace(0.2 * r_J, 2.5 * r_J, 400)
    Phi = -G * m / x - G * M_g / (R - x) - 0.5 * (G * (M_g + m) / R**3) * (x - x_com) ** 2
    x_l1 = x[np.argmax(Phi)]
    l1_pass = abs(x_l1 - r_J) / r_J < 0.05
    print(f"  effective-potential max (L1) at x = {x_l1:.2f} pc (r_J = {r_J:.2f})  "
          f"-> {'PASS' if l1_pass else 'FAIL'}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.3, 3.1))
    axA.plot(r / r_J, self_grav, "-", color=OI["blue"], lw=1.8,
             label=r"self-gravity $Gm/r^2$")
    axA.plot(r / r_J, tidal, "-", color=OI["vermilion"], lw=1.8,
             label=r"tidal+centrifugal $3\Omega^2 r$")
    axA.axvline(1.0, color="0.5", ls="--", lw=1.0)
    axA.set_yscale("log")
    axA.set_xlabel(r"$r / r_J$"); axA.set_ylabel(r"radial acceleration")
    axA.legend(loc="upper right", fontsize=7)
    axA.text(1.02, axA.get_ylim()[0] * 3, r"$r_J$", fontsize=8, color="0.4")
    panel_label(axA, "(a)", loc="lower left")

    axB.plot(x / r_J, Phi, "-", color=OI["black"], lw=1.8)
    axB.plot(x_l1 / r_J, np.max(Phi), "o", color=OI["vermilion"], ms=7, mec="white",
             label=rf"L1 $\approx r_J$")
    axB.set_xlabel(r"$x / r_J$ (toward galaxy)")
    axB.set_ylabel(r"$\Phi_{\rm eff}(x)$ (rotating frame)")
    axB.legend(loc="lower center", fontsize=7.5)
    axB.text(0.5, 0.92, "escape barrier\npeaks at $r_J$", transform=axB.transAxes,
             ha="center", va="top", fontsize=7, color="0.4")
    panel_label(axB, "(b)", loc="lower left")

    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "tidal_force_balance")
    print("  saved tidal_force_balance.{png,pdf}")
    return bal_pass and l1_pass


# ============================================================================
# Figure 3 -- Keplerian vs isothermal tidal radius
# ============================================================================
def fig_keplerian_vs_iso(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: Keplerian (point-mass) vs isothermal (flat rotation) tidal radius")
    print("=" * 60)
    m = 1e4
    V = 220.0 * KMS_TO_PCMYR  # flat rotation curve, pc/Myr
    R = np.linspace(2000.0, 20000.0, 200)
    r_iso = np.array([float(jacobi_radius_isothermal(m, V, Ri, G)) for Ri in R])
    # point-mass host with the SAME Omega (=V/R) at each R: M_g(R) = Omega^2 R^3 / G
    Omega = V / R
    M_g = Omega**2 * R**3 / G
    r_point = np.array([float(jacobi_radius(m, M_g[i], R[i])) for i in range(len(R))])
    ratio = r_iso / r_point
    expected = (3.0 / 2.0) ** (1.0 / 3.0)
    ratio_ok = bool(np.allclose(ratio, expected, rtol=1e-6))
    print(f"  r_J,iso / r_J,point = {ratio.mean():.4f} (expected (3/2)^1/3 = "
          f"{expected:.4f})  -> {'PASS' if ratio_ok else 'FAIL'}")
    # Galactic GC anchor (R=8 kpc)
    iGC = np.argmin(np.abs(R - 8000.0))
    print(f"  at R=8 kpc: r_J,iso = {r_iso[iGC]:.1f} pc, r_J,point = {r_point[iGC]:.1f} pc")

    fig, ax = plt.subplots(figsize=(4.7, 3.4))
    ax.plot(R / 1000, r_iso, "-", color=OI["blue"], lw=1.9,
            label=r"isothermal (flat $v_c$, factor 2)")
    ax.plot(R / 1000, r_point, "-", color=OI["vermilion"], lw=1.9,
            label=r"point mass (Keplerian, factor 3)")
    ax.plot(8.0, r_iso[iGC], "o", color=OI["black"], ms=6, mec="white", zorder=4)
    ax.annotate("Galactic GC\n(8 kpc)", (8.0, r_iso[iGC]), textcoords="offset points",
                xytext=(6, -2), fontsize=6.5, color="0.4")
    ax.set_xlabel(r"orbital radius $R$ [kpc]"); ax.set_ylabel(r"tidal radius $r_J$ [pc]")
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.96, 0.06, rf"$r_{{\rm iso}}/r_{{\rm point}}=(3/2)^{{1/3}}={expected:.3f}$",
            transform=ax.transAxes, ha="right", fontsize=7, color="0.35")
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "tidal_keplerian_vs_iso")
    print("  saved tidal_keplerian_vs_iso.{png,pdf}")
    return ratio_ok


# ============================================================================
# Figure 4 -- Plummer truncation vs analytic enclosed mass
# ============================================================================
def fig_plummer_truncation(output_dir):
    print("\n" + "=" * 60)
    print("FIG 4: Plummer tidal truncation vs analytic enclosed mass")
    print("=" * 60)
    n, r_h = 8000, 5.0
    a = r_h * np.sqrt(2.0 ** (2.0 / 3.0) - 1.0)
    prof = PlummerProfile(r_h=r_h)
    masses = jnp.ones(n)
    pos = prof.sample_positions(masses, jax.random.PRNGKey(0))
    radii = np.asarray(jnp.linalg.norm(pos, axis=1))

    rts = np.linspace(2.0, 60.0, 40)
    bound = np.array([float(jnp.sum(apply_tidal_truncation(pos, jnp.zeros((n, 3)),
                                                           masses, rt)[2])) / n
                      for rt in rts])
    frac_an = rts**3 / (rts**2 + a**2) ** 1.5
    max_dev = float(np.max(np.abs(bound - frac_an)))
    track_ok = max_dev < 0.02
    print(f"  bound-mass fraction vs analytic Plummer M(<r): max dev = {max_dev:.3f} "
          f"(tol 0.02)  -> {'PASS' if track_ok else 'FAIL'}")

    # spatial truncation at r_t = r_J for a typical Galactic GC (fill factor ~ 0.1)
    r_J = float(jacobi_radius(1e4, 1e11, 8000.0))  # ~ 50 pc for these numbers
    fill = r_h / r_J
    _, _, mt, mask = apply_tidal_truncation(pos, jnp.zeros((n, 3)), masses, r_J)
    n_kept = int(np.sum(np.asarray(mask)))
    print(f"  r_J = {r_J:.1f} pc, fill factor r_h/r_J = {fill:.3f}; kept "
          f"{n_kept}/{n} ({100*n_kept/n:.1f} pct)")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.3, 3.1))
    axA.plot(rts / r_h, frac_an, "-", color=OI["black"], lw=1.9,
             label=r"analytic $M(<r)/M$")
    axA.plot(rts[::2] / r_h, bound[::2], "o", color=OI["vermilion"], ms=4, mfc="none",
             mew=1.0, label="truncated bound mass")
    axA.set_xlabel(r"$r_t / r_h$"); axA.set_ylabel(r"bound-mass fraction")
    axA.legend(loc="lower right", fontsize=7.5)
    panel_label(axA, "(a)", loc="upper left")

    p = np.asarray(pos)
    kept = np.asarray(mask)
    axB.scatter(p[~kept, 0], p[~kept, 1], s=2, color="0.7", label="stripped")
    axB.scatter(p[kept, 0], p[kept, 1], s=2, color=OI["blue"], label="bound")
    thr = np.linspace(0, 2 * np.pi, 100)
    axB.plot(r_J * np.cos(thr), r_J * np.sin(thr), "-", color=OI["vermilion"], lw=1.4)
    axB.set_xlim(-1.5 * r_J, 1.5 * r_J); axB.set_ylim(-1.5 * r_J, 1.5 * r_J)
    axB.set_aspect("equal")
    axB.set_xlabel(r"$x$ [pc]"); axB.set_ylabel(r"$y$ [pc]")
    axB.legend(loc="upper right", fontsize=6.5, markerscale=2)
    axB.text(0.04, 0.05, rf"$r_t=r_J={r_J:.0f}$ pc", transform=axB.transAxes,
             fontsize=7, color=OI["vermilion"])
    panel_label(axB, "(b)", loc="upper left")

    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "tidal_plummer_truncation")
    print("  saved tidal_plummer_truncation.{png,pdf}")
    return track_ok


# ============================================================================
# Figure 5 -- differentiability
# ============================================================================
def fig_gradient_validation(output_dir):
    print("\n" + "=" * 60)
    print("FIG 5: differentiability (AD vs analytic / finite difference)")
    print("=" * 60)
    M_g, R = 1e11, 8000.0

    # (a) d r_J / d M_c : AD vs exact analytic  R/(3 M_g)^(1/3) * (1/3) M_c^(-2/3)
    Mc = np.linspace(1e3, 1e6, 12)
    drj_ad = np.array([float(jax.grad(lambda mc: jacobi_radius(mc, M_g, R))(float(mc)))
                       for mc in Mc])
    drj_an = R / (3.0 * M_g) ** (1.0 / 3.0) * (1.0 / 3.0) * Mc ** (-2.0 / 3.0)
    rel_a = np.max(np.abs(drj_ad - drj_an) / np.abs(drj_an))
    a_ok = rel_a < 1e-6
    print(f"  d r_J/d M_c: AD vs analytic max rel err {rel_a:.2e} (tol 1e-6)  "
          f"-> {'PASS' if a_ok else 'FAIL'}")

    # (b) d(M_bound)/d r_t : straight-through surrogate vs analytic Plummer shell mass
    n, r_h = 12000, 5.0
    a = r_h * np.sqrt(2.0 ** (2.0 / 3.0) - 1.0)
    prof = PlummerProfile(r_h=r_h)
    masses = jnp.ones(n)
    pos = prof.sample_positions(masses, jax.random.PRNGKey(1))
    vel = jnp.zeros((n, 3))

    def bound_mass(rt):
        return jnp.sum(apply_tidal_truncation(pos, vel, masses, rt)[2])

    rts = np.linspace(4.0, 30.0, 14)
    dM_ad = np.array([float(jax.grad(bound_mass)(float(rt))) for rt in rts])
    dM_an = n * 3.0 * a**2 * rts**2 / (rts**2 + a**2) ** 2.5  # d/dr [N r^3/(r^2+a^2)^1.5]
    # mean-sense agreement (surrogate is a smoothed estimator; finite-N noise)
    rel_b = np.median(np.abs(dM_ad - dM_an) / (dM_an + 1e-9))
    b_ok = rel_b < 0.2
    print(f"  d(M_bound)/d r_t: surrogate vs analytic Plummer shell, median rel "
          f"{rel_b:.2f} (tol 0.2)  -> {'PASS' if b_ok else 'FAIL'}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axA.plot(Mc, drj_an, "-", color=OI["black"], lw=1.8, label="analytic")
    axA.plot(Mc, drj_ad, "o", color=OI["vermilion"], ms=4.5, mfc="none", mew=1.1,
             label="autodiff")
    axA.set_xlabel(r"$M_{\rm cluster}$ [$M_\odot$]")
    axA.set_ylabel(r"$\partial r_J / \partial M_{\rm cluster}$ [pc/$M_\odot$]")
    axA.legend(loc="upper right", fontsize=7.5)
    axA.text(0.5, 0.1, rf"max rel err ${rel_a:.0e}$", transform=axA.transAxes,
             ha="center", fontsize=7.5, color="0.4")
    panel_label(axA, "(a)", loc="upper left")

    axB.plot(rts / r_h, dM_an, "-", color=OI["black"], lw=1.8,
             label=r"analytic shell $dM/dr_t$")
    axB.plot(rts / r_h, dM_ad, "o", color=OI["blue"], ms=4.5, mfc="none", mew=1.1,
             label="straight-through AD")
    axB.set_xlabel(r"$r_t / r_h$")
    axB.set_ylabel(r"$\partial M_{\rm bound} / \partial r_t$")
    axB.legend(loc="upper right", fontsize=7.5)
    axB.text(0.5, 0.08, "hard cut, differentiable", transform=axB.transAxes,
             ha="center", fontsize=7.2, color="0.4")
    panel_label(axB, "(b)", loc="upper right")

    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "tidal_gradient_validation")
    print("  saved tidal_gradient_validation.{png,pdf}")
    return a_ok and b_ok


def main():
    print("\n" + "=" * 70)
    print("PROGENAX TIDAL-TRUNCATION VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "Fig 1  Jacobi radius vs L1": fig_jacobi_vs_l1(OUTPUT_DIR),
        "Fig 2  tidal force balance": fig_force_balance(OUTPUT_DIR),
        "Fig 3  Keplerian vs isothermal": fig_keplerian_vs_iso(OUTPUT_DIR),
        "Fig 4  Plummer truncation": fig_plummer_truncation(OUTPUT_DIR),
        "Fig 5  differentiability": fig_gradient_validation(OUTPUT_DIR),
    }
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print("  ALL TIDAL VALIDATION FIGURES PASS" if all_ok
          else "  SOME TIDAL VALIDATION FIGURES FAILED")
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/tidal_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
