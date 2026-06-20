#!/usr/bin/env python
"""
ZAMS (Tout+1996) validation figures: M -> L, R, T_eff, log g + the inverse L -> M.

Five publication-quality figures anchored to tests/validation/test_zams_physics.py
(34 tests). Each prints expected-vs-measured PASS/FAIL against an *independent*
oracle: the four PDF-verified solar anchors (computed cell-by-cell from the Tout+1996
Table 1/2 coefficients at M=1, Z=0.02 -- see
docs/core-papers/tout1996_zams_coefficients_verified.md), the Stefan-Boltzmann /
g=GM/R^2 closure, strict L(M) monotonicity, the paper's stated accuracy envelope, and
the inverse Newton round-trip residual.

Figures (-> what they validate):
  1. zams_luminosity_mass.png   L(M) over the fitted range at three metallicities;
                                the solar anchor L(1,0.02)=0.6977 Lsun; strict monotone
  2. zams_radius_mass.png       R(M) over the fitted range; solar anchor R=0.8882 Rsun
  3. zams_teff_mass.png         T_eff(M) via Stefan-Boltzmann; the hand-computed
                                T_eff closure overlaid (independent oracle)
  4. zams_hr_diagram.png        log L vs log T_eff (the Tout Fig. 5 view) at Z=0.02,
                                0.001, 0.0001 -- a ZAMS in the HR plane; the Sun marked
  5. zams_inverse_roundtrip.png inverse_zams_luminosity(zams_luminosity(M)) residual
                                over [0.1,100] Msun (machine-precision Newton invert)

Reference:
    Tout, Pols, Eggleton & Han (1996), MNRAS 281, 257 (Tables 1 & 2).
    Coefficients PDF-verified: docs/core-papers/tout1996_zams_coefficients_verified.md.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_zams.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.constants import LSUN_ERG_S, RSUN_CM, SIGMA_SB

import progenax  # noqa: F401  (enables float64)
from progenax.stellar import (
    inverse_zams_luminosity,
    zams_effective_temperature,
    zams_luminosity,
    zams_radius,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"

# PDF-verified solar ZAMS anchors at M=1, Z=0.02 (zeta=0 -> column-`a` algebra);
# see docs/core-papers/tout1996_zams_coefficients_verified.md.
L_SUN_ZAMS = 0.6977165691451518
R_SUN_ZAMS = 0.8882494502975121
TEFF_SUN_ZAMS = 5597.303626190019
LOGG_SUN_ZAMS = 4.540995576621913

# Metallicities for the multi-Z panels (the Tout+1996 fitted box, solar + two poorer).
Z_GRID = [
    (0.02, OI["vermilion"], r"$Z=0.02$ (solar)"),
    (0.001, OI["blue"], r"$Z=0.001$"),
    (0.0001, OI["green"], r"$Z=10^{-4}$"),
]


# ============================================================================
# Figure 1 -- L(M) over the fitted range, multi-Z, with the solar anchor
# ============================================================================
def fig_luminosity_mass(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: ZAMS luminosity L(M) vs verified solar anchor + monotonicity")
    print("=" * 60)
    M = jnp.logspace(jnp.log10(0.1), jnp.log10(100.0), 400)

    L_sun = float(zams_luminosity(jnp.array(1.0)))
    anchor_ok = abs(L_sun - L_SUN_ZAMS) / L_SUN_ZAMS < 0.03
    print(
        f"  L(1 Msun, Z=0.02) = {L_sun:.4f} Lsun (verified {L_SUN_ZAMS:.4f}, tol 3%)"
        f"  -> {'PASS' if anchor_ok else 'FAIL'}"
    )

    L_solar = zams_luminosity(M, Z=0.02)
    mono_ok = bool(jnp.all(jnp.diff(L_solar) > 0.0))
    print(
        f"  L(M) strictly monotone over [0.1,100] Msun  "
        f"-> {'PASS' if mono_ok else 'FAIL'}"
    )

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for Z, color, label in Z_GRID:
        ax.loglog(
            np.asarray(M),
            np.asarray(zams_luminosity(M, Z=Z)),
            "-",
            color=color,
            lw=1.7,
            label=label,
        )
    ax.plot(
        1.0,
        L_SUN_ZAMS,
        "*",
        color=OI["black"],
        ms=12,
        mec="white",
        mew=0.6,
        zorder=5,
        label=r"Sun (ZAMS, $0.698\,L_\odot$)",
    )
    ax.set_xlabel(r"$M\ [M_\odot]$")
    ax.set_ylabel(r"$L\ [L_\odot]$")
    ax.legend(loc="upper left", fontsize=7)
    ax.text(
        0.96,
        0.06,
        "Tout+1996 Table 1",
        transform=ax.transAxes,
        ha="right",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "zams_luminosity_mass")
    print("  saved zams_luminosity_mass.{png,pdf}")
    return anchor_ok and mono_ok


# ============================================================================
# Figure 2 -- R(M) over the fitted range, multi-Z, with the solar anchor
# ============================================================================
def fig_radius_mass(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: ZAMS radius R(M) vs verified solar anchor")
    print("=" * 60)
    M = jnp.logspace(jnp.log10(0.1), jnp.log10(100.0), 400)

    R_sun = float(zams_radius(jnp.array(1.0)))
    anchor_ok = abs(R_sun - R_SUN_ZAMS) / R_SUN_ZAMS < 0.012
    print(
        f"  R(1 Msun, Z=0.02) = {R_sun:.4f} Rsun (verified {R_SUN_ZAMS:.4f}, tol 1.2%)"
        f"  -> {'PASS' if anchor_ok else 'FAIL'}"
    )
    pos_ok = bool(jnp.all(zams_radius(M, Z=0.02) > 0.0))
    print(f"  R(M) > 0 over [0.1,100] Msun  -> {'PASS' if pos_ok else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for Z, color, label in Z_GRID:
        ax.loglog(
            np.asarray(M),
            np.asarray(zams_radius(M, Z=Z)),
            "-",
            color=color,
            lw=1.7,
            label=label,
        )
    ax.plot(
        1.0,
        R_SUN_ZAMS,
        "*",
        color=OI["black"],
        ms=12,
        mec="white",
        mew=0.6,
        zorder=5,
        label=r"Sun (ZAMS, $0.888\,R_\odot$)",
    )
    ax.set_xlabel(r"$M\ [M_\odot]$")
    ax.set_ylabel(r"$R\ [R_\odot]$")
    ax.legend(loc="upper left", fontsize=7)
    ax.text(
        0.96,
        0.06,
        "Tout+1996 Table 2",
        transform=ax.transAxes,
        ha="right",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "zams_radius_mass")
    print("  saved zams_radius_mass.{png,pdf}")
    return anchor_ok and pos_ok


# ============================================================================
# Figure 3 -- T_eff(M) via Stefan-Boltzmann + the hand-computed closure oracle
# ============================================================================
def fig_teff_mass(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: ZAMS T_eff(M) vs hand-computed Stefan-Boltzmann closure")
    print("=" * 60)
    M = jnp.logspace(jnp.log10(0.1), jnp.log10(100.0), 200)

    T_mod = np.asarray(zams_effective_temperature(M, Z=0.02))
    # independent oracle: recompute T_eff by hand in cgs from L(M), R(M)
    L_cgs = np.asarray(zams_luminosity(M, Z=0.02)) * LSUN_ERG_S
    R_cgs = np.asarray(zams_radius(M, Z=0.02)) * RSUN_CM
    T_hand = (L_cgs / (4.0 * np.pi * R_cgs**2 * SIGMA_SB)) ** 0.25
    max_rel = float(np.max(np.abs(T_mod - T_hand) / T_hand))
    closure_ok = max_rel < 1e-10
    print(
        f"  T_eff(M) vs hand Stefan-Boltzmann: max rel {max_rel:.2e} (tol 1e-10)"
        f"  -> {'PASS' if closure_ok else 'FAIL'}"
    )

    T_sun = float(zams_effective_temperature(jnp.array(1.0)))
    anchor_ok = abs(T_sun - TEFF_SUN_ZAMS) < 5.0
    print(
        f"  T_eff(1 Msun) = {T_sun:.1f} K (verified {TEFF_SUN_ZAMS:.1f} K)"
        f"  -> {'PASS' if anchor_ok else 'FAIL'}"
    )

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.loglog(
        np.asarray(M),
        T_hand,
        "-",
        color=OI["black"],
        lw=2.0,
        label=r"hand $4\pi R^2\sigma T^4=L$",
    )
    ax.loglog(
        np.asarray(M)[::6],
        T_mod[::6],
        "o",
        color=OI["vermilion"],
        ms=4,
        mfc="none",
        mew=1.0,
        label=r"$T_{\rm eff}(M)$ module",
    )
    ax.plot(
        1.0,
        TEFF_SUN_ZAMS,
        "*",
        color=OI["blue"],
        ms=12,
        mec="white",
        mew=0.6,
        zorder=5,
        label=r"Sun (ZAMS, $5597$ K)",
    )
    ax.set_xlabel(r"$M\ [M_\odot]$")
    ax.set_ylabel(r"$T_{\rm eff}\ [{\rm K}]$")
    ax.legend(loc="upper left", fontsize=7)
    ax.text(
        0.5,
        0.07,
        "Stefan-Boltzmann closure",
        transform=ax.transAxes,
        ha="center",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "zams_teff_mass")
    print("  saved zams_teff_mass.{png,pdf}")
    return closure_ok and anchor_ok


# ============================================================================
# Figure 4 -- HR diagram (log L vs log T_eff) at three metallicities (Tout Fig. 5)
# ============================================================================
def fig_hr_diagram(output_dir):
    print("\n" + "=" * 60)
    print("FIG 4: ZAMS HR diagram (log L vs log T_eff), multi-Z")
    print("=" * 60)
    M = jnp.logspace(jnp.log10(0.1), jnp.log10(100.0), 300)

    # ZAMS L slopes positively with T_eff (hot stars are luminous): the locus is
    # monotone in log T_eff -- an independent shape oracle for the HR track.
    L0 = np.asarray(zams_luminosity(M, Z=0.02))
    T0 = np.asarray(zams_effective_temperature(M, Z=0.02))
    order = np.argsort(T0)
    mono_ok = bool(np.all(np.diff(np.log10(L0[order])) > -1e-9))
    print(
        f"  ZAMS locus monotone in (log T_eff, log L)  "
        f"-> {'PASS' if mono_ok else 'FAIL'}"
    )
    # metal-poor ZAMS is hotter at fixed mass (lower opacity) -- bluer track
    T_poor = float(zams_effective_temperature(jnp.array(1.0), Z=0.0001))
    bluer_ok = T_poor > TEFF_SUN_ZAMS
    print(
        f"  T_eff(1 Msun): Z=1e-4 {T_poor:.0f} K > Z=0.02 {TEFF_SUN_ZAMS:.0f} K "
        f"(metal-poor is bluer)  -> {'PASS' if bluer_ok else 'FAIL'}"
    )

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for Z, color, label in Z_GRID:
        L = np.asarray(zams_luminosity(M, Z=Z))
        T = np.asarray(zams_effective_temperature(M, Z=Z))
        ax.plot(np.log10(T), np.log10(L), "-", color=color, lw=1.7, label=label)
    ax.plot(
        np.log10(TEFF_SUN_ZAMS),
        np.log10(L_SUN_ZAMS),
        "*",
        color=OI["black"],
        ms=13,
        mec="white",
        mew=0.6,
        zorder=5,
        label="Sun (ZAMS)",
    )
    # mass tick marks along the solar track
    for Mt in (0.3, 1.0, 3.0, 10.0, 30.0):
        Tt = float(zams_effective_temperature(jnp.array(Mt), Z=0.02))
        Lt = float(zams_luminosity(jnp.array(Mt), Z=0.02))
        ax.plot(
            np.log10(Tt), np.log10(Lt), "o", color=OI["vermilion"], ms=3.5, zorder=4
        )
        ax.annotate(
            rf"${Mt:g}\,M_\odot$",
            (np.log10(Tt), np.log10(Lt)),
            textcoords="offset points",
            xytext=(5, -2),
            fontsize=6,
            color="0.4",
        )
    ax.invert_xaxis()  # HR convention: hot/blue to the left
    ax.set_xlabel(r"$\log_{10}(T_{\rm eff}\,/\,{\rm K})$")
    ax.set_ylabel(r"$\log_{10}(L\,/\,L_\odot)$")
    ax.legend(loc="lower left", fontsize=7)
    ax.text(
        0.04,
        0.96,
        "ZAMS (Tout+1996 Fig. 5 view)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "zams_hr_diagram")
    print("  saved zams_hr_diagram.{png,pdf}")
    return mono_ok and bluer_ok


# ============================================================================
# Figure 5 -- inverse Newton round-trip residual
# ============================================================================
def fig_inverse_roundtrip(output_dir):
    print("\n" + "=" * 60)
    print("FIG 5: inverse_zams_luminosity round-trip residual (Newton/scan invert)")
    print("=" * 60)
    M = jnp.logspace(jnp.log10(0.1), jnp.log10(100.0), 120)
    M_rec = inverse_zams_luminosity(zams_luminosity(M))
    rel = np.abs(np.asarray(M_rec) - np.asarray(M)) / np.asarray(M)
    max_rel = float(np.max(rel))
    rt_ok = max_rel < 1e-5
    print(
        f"  M -> L -> M round-trip: max rel residual {max_rel:.2e} (tol 1e-5)"
        f"  -> {'PASS' if rt_ok else 'FAIL'}"
    )

    # differentiability of the invert (independent finite-difference check at L=100).
    # inverse_zams_luminosity returns a scalar for scalar input, so jax.grad applies
    # directly: dM/dL.
    L0 = 100.0
    g_ad = float(jax.grad(lambda L: inverse_zams_luminosity(L))(jnp.array(L0)))
    h = 1e-3
    g_fd = (
        float(inverse_zams_luminosity(jnp.array(L0 + h)))
        - float(inverse_zams_luminosity(jnp.array(L0 - h)))
    ) / (2 * h)
    grad_rel = abs(g_ad - g_fd) / abs(g_fd)
    grad_ok = np.isfinite(g_ad) and grad_rel < 1e-4
    print(
        f"  d M/d L at L=100: AD {g_ad:.4e} vs FD {g_fd:.4e}, rel {grad_rel:.1e}"
        f" (tol 1e-4)  -> {'PASS' if grad_ok else 'FAIL'}"
    )

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.3, 3.1))
    axA.loglog(
        np.asarray(M),
        np.asarray(M_rec),
        "-",
        color=OI["black"],
        lw=1.6,
        label="recovered",
    )
    axA.loglog(
        np.asarray(M), np.asarray(M), ":", color=OI["vermilion"], lw=1.4, label=r"$y=x$"
    )
    axA.set_xlabel(r"$M_{\rm in}\ [M_\odot]$")
    axA.set_ylabel(r"$M_{\rm rec}=L^{-1}(L(M))\ [M_\odot]$")
    axA.legend(loc="upper left", fontsize=7.5)
    panel_label(axA, "(a)", loc="lower right")

    axB.loglog(
        np.asarray(M), np.maximum(rel, 1e-18), "o-", color=OI["blue"], ms=3, lw=1.0
    )
    axB.axhline(1e-5, color="0.5", ls="--", lw=1.0)
    axB.text(0.15, 1.5e-5, "tol $10^{-5}$", fontsize=7, color="0.4")
    axB.set_xlabel(r"$M_{\rm in}\ [M_\odot]$")
    axB.set_ylabel(r"$|M_{\rm rec}-M_{\rm in}|/M_{\rm in}$")
    axB.set_ylim(1e-18, 1e-3)
    panel_label(axB, "(b)", loc="upper right")

    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "zams_inverse_roundtrip")
    print("  saved zams_inverse_roundtrip.{png,pdf}")
    return rt_ok and grad_ok


def main():
    print("\n" + "=" * 70)
    print("PROGENAX ZAMS (Tout+1996) VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "Fig 1  L(M) + solar anchor + monotone": fig_luminosity_mass(OUTPUT_DIR),
        "Fig 2  R(M) + solar anchor": fig_radius_mass(OUTPUT_DIR),
        "Fig 3  T_eff(M) Stefan-Boltzmann closure": fig_teff_mass(OUTPUT_DIR),
        "Fig 4  HR diagram (multi-Z)": fig_hr_diagram(OUTPUT_DIR),
        "Fig 5  inverse round-trip residual": fig_inverse_roundtrip(OUTPUT_DIR),
    }
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print(
        "  ALL ZAMS VALIDATION FIGURES PASS"
        if all_ok
        else "  SOME ZAMS VALIDATION FIGURES FAILED"
    )
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/zams_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
