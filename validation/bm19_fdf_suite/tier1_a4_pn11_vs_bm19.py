#!/usr/bin/env python
"""A4: PN11 vs BM19 — THEORY COMPARISON.

f_dense vs Mach for both theories, same axes.

Demonstrates "genuinely different and better constrained".

Output: a4_pn11_vs_bm19.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from progenax.gravoturb import bm19_model as bm19, pn11_model as pn11

from matplotlib.patches import Ellipse

from .helpers import (
    setup_publication_style,
    save_plot,
    COLORS,
    s_to_column_density,
    OBSERVATIONAL_ANCHORS,
)


def run_validation(
    machs: np.ndarray | None = None,
    alphas: list[float] = [1.5, 2.0, 2.5, 3.0],
    b: float = 0.4,
    Sigma: float = 100.0,  # For PN11 alpha_vir
    verbose: bool = True,
):
    """Compare PN11 and BM19 f_dense predictions.

    Parameters
    ----------
    machs : array
        Mach number range
    alphas : list
        BM19 powerlaw slopes to compare
    b : float
        Driving parameter
    Sigma : float
        Surface density for PN11 alpha_vir
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Theory predictions
    """
    if verbose:
        print("=" * 70)
        print("A4: PN11 vs BM19 THEORY COMPARISON")
        print("=" * 70)

    if machs is None:
        machs = np.linspace(5, 35, 50)

    # PN11 prediction (uses alpha_vir from Sigma)
    alpha_vir = float(pn11.alpha_vir_from_sigma(Sigma))
    if verbose:
        print(f"\nPN11 parameters: Sigma={Sigma}, alpha_vir={alpha_vir:.2f}")

    f_dense_pn11 = []
    s_crit_pn11_vals = []
    for mach in machs:
        sigma_sq = float(bm19.sigma_s_squared(mach, b))
        s_crit = float(pn11.s_crit_pn11(mach, alpha_vir))
        f_pn11 = float(pn11.f_dense_pn11(sigma_sq, s_crit))
        f_dense_pn11.append(f_pn11)
        s_crit_pn11_vals.append(s_crit)

    f_dense_pn11 = np.array(f_dense_pn11)
    s_crit_pn11_vals = np.array(s_crit_pn11_vals)

    if verbose:
        print(f"PN11 f_dense range: [{f_dense_pn11.min():.4f}, {f_dense_pn11.max():.4f}]")

    # BM19 predictions for each alpha
    bm19_results = {}
    for alpha in alphas:
        f_dense_bm19 = []
        s_t_vals = []
        for mach in machs:
            result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.6)
            f_dense_bm19.append(float(result.f_dense))
            s_t_vals.append(float(result.s_t))

        f_dense_bm19 = np.array(f_dense_bm19)
        s_t_vals = np.array(s_t_vals)

        bm19_results[alpha] = {
            "f_dense": f_dense_bm19,
            "s_t": s_t_vals,
        }

        if verbose:
            print(f"BM19 (alpha={alpha}) f_dense range: [{f_dense_bm19.min():.4f}, {f_dense_bm19.max():.4f}]")

    return {
        "machs": machs,
        "pn11": {
            "f_dense": f_dense_pn11,
            "s_crit": s_crit_pn11_vals,
            "alpha_vir": alpha_vir,
        },
        "bm19": bm19_results,
        "params": {
            "alphas": alphas,
            "b": b,
            "Sigma": Sigma,
        }
    }


def make_plot(results: dict, show: bool = False) -> str:
    """Generate PN11 vs BM19 comparison plot.

    Parameters
    ----------
    results : dict
        Output from run_validation()
    show : bool
        Display interactively

    Returns
    -------
    path : str
        Path to saved plot
    """
    setup_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    machs = results["machs"]
    alphas = results["params"]["alphas"]

    # Color palette for alphas
    alpha_colors = {1.5: "C0", 2.0: "C1", 2.5: "C2", 3.0: "C3"}

    # LEFT: f_dense vs Mach
    ax1 = axes[0]

    # PN11
    ax1.semilogy(
        machs, results["pn11"]["f_dense"],
        color=COLORS["pn11"], linewidth=3, linestyle="--",
        label=f"PN11 ($\\alpha_{{vir}}$={results['pn11']['alpha_vir']:.1f})"
    )

    # BM19 for each alpha
    for alpha in alphas:
        ax1.semilogy(
            machs, results["bm19"][alpha]["f_dense"],
            color=alpha_colors[alpha], linewidth=2,
            label=f"BM19 ($\\alpha$={alpha})"
        )

    ax1.set_xlabel("Mach Number ($\\mathcal{M}$)", fontsize=12)
    ax1.set_ylabel("$f_\\mathrm{dense}$", fontsize=12)
    ax1.set_title("Self-Gravitating Fraction: PN11 vs BM19", fontsize=14)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(machs.min(), machs.max())

    # RIGHT: s_crit/s_t vs Mach
    ax2 = axes[1]

    # PN11 s_crit
    ax2.plot(
        machs, results["pn11"]["s_crit"],
        color=COLORS["pn11"], linewidth=3, linestyle="--",
        label="PN11 $s_\\mathrm{crit}$"
    )

    # BM19 s_t for each alpha
    for alpha in alphas:
        ax2.plot(
            machs, results["bm19"][alpha]["s_t"],
            color=alpha_colors[alpha], linewidth=2,
            label=f"BM19 $s_t$ ($\\alpha$={alpha})"
        )

    ax2.set_xlabel("Mach Number ($\\mathcal{M}$)", fontsize=12)
    ax2.set_ylabel("Threshold Density ($s_\\mathrm{crit}$ or $s_t$)", fontsize=12)
    ax2.set_title("Critical Density Threshold: PN11 vs BM19", fontsize=14)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(machs.min(), machs.max())

    plt.suptitle(
        "BM19 (explicit $\\alpha$) vs PN11 (implicit $\\alpha_\\mathrm{vir}$)",
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "a4_pn11_vs_bm19")
    plt.close(fig)

    return path


def make_sigma_dependence_plot(show: bool = False) -> str:
    """Show how PN11 depends on Sigma while BM19 depends on α.

    PN11 uses alpha_vir(Sigma) so f_dense changes with Sigma.
    BM19 depends on (M, alpha, b), not Sigma - show multiple α lines.
    """
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    machs = np.linspace(5, 35, 50)
    Sigmas = [50, 100, 200, 500]
    alphas_bm19 = [1.5, 1.8, 2.0, 2.3]  # Realistic range from BM19
    b = 0.4

    # Colors for PN11 (by Sigma) - dashed lines
    colors_sigma = {50: "C0", 100: "C1", 200: "C2", 500: "C3"}

    # Colors for BM19 (by alpha) - solid lines
    colors_alpha = {1.5: "purple", 1.8: "darkred", 2.0: "darkorange", 2.3: "darkgreen"}

    # PN11 for different Sigma (dashed)
    for Sigma in Sigmas:
        alpha_vir = float(pn11.alpha_vir_from_sigma(Sigma))
        f_dense_pn11 = []
        for mach in machs:
            sigma_sq = float(bm19.sigma_s_squared(mach, b))
            s_crit = float(pn11.s_crit_pn11(mach, alpha_vir))
            f_dense_pn11.append(float(pn11.f_dense_pn11(sigma_sq, s_crit)))

        ax.semilogy(
            machs, f_dense_pn11,
            color=colors_sigma[Sigma], linewidth=2, linestyle="--",
            label=f"PN11 $\\Sigma$={Sigma}"
        )

    # BM19 for different alpha (solid lines)
    for alpha in alphas_bm19:
        f_dense_bm19 = []
        for mach in machs:
            result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.6)
            f_dense_bm19.append(float(result.f_dense))

        ax.semilogy(
            machs, f_dense_bm19,
            color=colors_alpha[alpha], linewidth=2.5, linestyle="-",
            label=f"BM19 $\\alpha$={alpha}"
        )

    ax.set_xlabel("Mach Number ($\\mathcal{M}$)", fontsize=12)
    ax.set_ylabel("$f_\\mathrm{dense}$", fontsize=12)
    ax.set_title(
        "PN11 Depends on $\\Sigma$ (dashed), BM19 Depends on $\\alpha$ (solid)\n"
        "Both models span similar range but with different physics",
        fontsize=14
    )

    # Create separate legends for clarity
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(machs.min(), machs.max())

    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "a4_pn11_vs_bm19_sigma_dependence")
    plt.close(fig)

    return path


# =============================================================================
# Real astrophysical environments (literature values)
# =============================================================================

ASTROPHYSICAL_ENVIRONMENTS = {
    "Taurus": {
        "Sigma": (20, 50),      # M☉/pc² - Heyer+2009
        "Mach": (3, 6),
        "color": "#a6cee3",     # Light blue
        "example": "Taurus, Cha I",
    },
    "Orion": {
        "Sigma": (80, 150),     # M☉/pc² - Lombardi+2014
        "Mach": (8, 15),
        "color": "#33a02c",     # Green
        "example": "Orion A/B",
    },
    "Dense clump": {
        "Sigma": (200, 500),    # M☉/pc² - Lada+2010
        "Mach": (12, 20),
        "color": "#ff7f00",     # Orange
        "example": "ρ Oph, Serpens",
    },
    "YMC": {
        "Sigma": (1000, 3000),  # M☉/pc² - Nguyen-Luong+2013
        "Mach": (20, 30),
        "color": "#e31a1c",     # Red
        "example": "W43, W51",
    },
    "Starburst": {
        "Sigma": (3000, 10000), # M☉/pc² - Leroy+2018
        "Mach": (30, 50),
        "color": "#6a3d9a",     # Purple
        "example": "Antennae, M82",
    },
}


def make_degeneracy_contour_plot(show: bool = False) -> str:
    """Show parameter degeneracies with real astrophysical environments.

    Two panels showing iso-f_dense and iso-N_H contours in (Mach, Σ) space,
    with real environments overlaid as ellipses.

    This demonstrates that different physical conditions can produce
    similar observables (degeneracy).
    """
    setup_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Parameter grids
    n_mach, n_sigma = 60, 60
    machs = np.linspace(3, 55, n_mach)
    sigmas = np.logspace(np.log10(15), np.log10(12000), n_sigma)
    Mach_grid, Sigma_grid = np.meshgrid(machs, sigmas)

    b = 0.4  # Natural mixture driving

    # Compute f_dense (PN11) over grid
    f_dense_grid = np.zeros_like(Mach_grid)
    for i in range(n_sigma):
        for j in range(n_mach):
            mach = Mach_grid[i, j]
            sigma = Sigma_grid[i, j]
            alpha_vir = float(pn11.alpha_vir_from_sigma(sigma))
            sigma_sq = float(bm19.sigma_s_squared(mach, b))
            s_crit = float(pn11.s_crit_pn11(mach, alpha_vir))
            f_dense_grid[i, j] = float(pn11.f_dense_pn11(sigma_sq, s_crit))

    # Compute N_H (BM19 at α=2.0) over grid
    alpha_bm19 = 2.0
    N_H_grid = np.zeros_like(Mach_grid)
    for i in range(n_sigma):
        for j in range(n_mach):
            mach = Mach_grid[i, j]
            sigma = Sigma_grid[i, j]
            result = bm19.bm19_pipeline(mach, b, alpha_bm19, eta_survive=0.6)
            s_t = float(result.s_t)
            N_H_grid[i, j] = s_to_column_density(s_t, sigma)

    # f_dense contour levels
    f_dense_levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    # N_H contour levels (cm^-2)
    N_H_levels = [1e22, 3e22, 1e23, 3e23, 1e24, 3e24, 1e25]

    # =========================================================================
    # Panel 1: f_dense (PN11) contours
    # =========================================================================
    ax1 = axes[0]

    # Filled contours
    cf1 = ax1.contourf(
        Mach_grid, Sigma_grid, f_dense_grid,
        levels=f_dense_levels,
        cmap="viridis",
        extend="both",
    )
    ax1.contour(
        Mach_grid, Sigma_grid, f_dense_grid,
        levels=f_dense_levels,
        colors="white",
        linewidths=0.5,
        linestyles="-",
    )

    # Add colorbar
    cbar1 = plt.colorbar(cf1, ax=ax1)
    cbar1.set_label("$f_\\mathrm{dense}$ (PN11)", fontsize=12)

    # Overlay environment ellipses
    _add_environment_ellipses(ax1)

    ax1.set_yscale("log")
    ax1.set_xlabel("Mach Number ($\\mathcal{M}$)", fontsize=12)
    ax1.set_ylabel("$\\Sigma$ [M$_\\odot$/pc$^2$]", fontsize=12)
    ax1.set_title("PN11: $f_\\mathrm{dense}$ Degeneracy in (Mach, $\\Sigma$)", fontsize=13)
    ax1.set_xlim(3, 55)
    ax1.set_ylim(15, 12000)

    # =========================================================================
    # Panel 2: N_H threshold (BM19) contours
    # =========================================================================
    ax2 = axes[1]

    # Filled contours
    cf2 = ax2.contourf(
        Mach_grid, Sigma_grid, N_H_grid,
        levels=N_H_levels,
        cmap="plasma",
        extend="both",
        norm=plt.matplotlib.colors.LogNorm(vmin=1e22, vmax=1e25),
    )
    ax2.contour(
        Mach_grid, Sigma_grid, N_H_grid,
        levels=N_H_levels,
        colors="white",
        linewidths=0.5,
        linestyles="-",
    )

    # Add colorbar
    cbar2 = plt.colorbar(cf2, ax=ax2, format="%.0e")
    cbar2.set_label("$N_H(s_t)$ [cm$^{-2}$] (BM19, $\\alpha$=2.0)", fontsize=12)

    # Lada threshold line
    lada_NH = OBSERVATIONAL_ANCHORS["lada_threshold_cm2"]
    ax2.axhline(
        y=100,  # Just a placeholder - we need to find Sigma where N_H = Lada
        color="red", linestyle="--", linewidth=2, alpha=0.0,  # Hide this
    )
    # Add annotation for Lada threshold
    ax2.annotate(
        f"Lada threshold: $N_H$ = {lada_NH:.0e} cm$^{{-2}}$",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=9, color="red", alpha=0.8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Overlay environment ellipses
    _add_environment_ellipses(ax2)

    ax2.set_yscale("log")
    ax2.set_xlabel("Mach Number ($\\mathcal{M}$)", fontsize=12)
    ax2.set_ylabel("$\\Sigma$ [M$_\\odot$/pc$^2$]", fontsize=12)
    ax2.set_title("BM19: $N_H(s_t)$ Degeneracy in (Mach, $\\Sigma$)", fontsize=13)
    ax2.set_xlim(3, 55)
    ax2.set_ylim(15, 12000)

    # Overall title
    plt.suptitle(
        "Parameter Degeneracies: Same Observable from Different Environments",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "a4_degeneracy_contours")
    plt.close(fig)

    return path


def _add_environment_ellipses(ax):
    """Add environment ellipses to an axis."""
    for name, env in ASTROPHYSICAL_ENVIRONMENTS.items():
        # Ellipse center (geometric mean for log scale)
        sigma_lo, sigma_hi = env["Sigma"]
        mach_lo, mach_hi = env["Mach"]

        # For log-scale y-axis, use geometric mean
        sigma_center = np.sqrt(sigma_lo * sigma_hi)
        mach_center = (mach_lo + mach_hi) / 2

        # Width/height (full range)
        width = mach_hi - mach_lo
        # For log scale, compute height in log space
        height_log = np.log10(sigma_hi) - np.log10(sigma_lo)

        # Create ellipse - need to handle log scale carefully
        # We'll draw a rectangle transformed to look like an ellipse region
        from matplotlib.patches import FancyBboxPatch

        # Use a rectangle for log-scale display
        rect = plt.Rectangle(
            (mach_lo, sigma_lo),
            width=mach_hi - mach_lo,
            height=sigma_hi - sigma_lo,
            facecolor=env["color"],
            edgecolor="black",
            linewidth=1.5,
            alpha=0.4,
            zorder=10,
        )
        ax.add_patch(rect)

        # Add label
        ax.annotate(
            name,
            xy=(mach_center, sigma_center),
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            zorder=11,
        )


def main():
    """Run full A4 validation."""
    results = run_validation(verbose=True)
    make_plot(results)
    make_sigma_dependence_plot()
    make_degeneracy_contour_plot()

    print("\n" + "=" * 70)
    print("A4 VALIDATION COMPLETE")
    print("=" * 70)
    print("\nKey differences:")
    print("  1. PN11 s_crit depends on alpha_vir(Sigma), Mach, phi_x")
    print("  2. BM19 s_t depends only on sigma_s^2 and alpha")
    print("  3. BM19 is better constrained (fewer free parameters)")
    print("  4. Both predict f_dense decreases with Mach")

    # Quantify difference at reference point
    mach_ref = 10.0
    alpha_vir = float(pn11.alpha_vir_from_sigma(100.0))
    sigma_sq = float(bm19.sigma_s_squared(mach_ref, 0.4))
    s_crit = float(pn11.s_crit_pn11(mach_ref, alpha_vir))
    f_pn11 = float(pn11.f_dense_pn11(sigma_sq, s_crit))
    f_bm19 = float(bm19.bm19_pipeline(mach_ref, 0.4, 2.0, 0.6).f_dense)

    print(f"\nAt M=10, Sigma=100:")
    print(f"  PN11 f_dense = {f_pn11:.4f}")
    print(f"  BM19 f_dense = {f_bm19:.4f} (alpha=2.0)")
    print(f"  Difference = {100*(f_pn11-f_bm19)/f_bm19:+.0f}%")

    return results


if __name__ == "__main__":
    main()
