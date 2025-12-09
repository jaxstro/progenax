#!/usr/bin/env python
"""
Mass segregation validation plots for progenax cluster IC generator.

Produces publication-quality figures:
1. Λ_MSR vs λ_seg with error bars
2. Mean radius of massive stars vs λ_seg
3. Radial profile comparison (λ=0 vs λ=1)
4. Snapshot projections (λ=0, 0.5, 1.0)
5. Cumulative radial distribution for massive vs all stars

Usage:
    python scripts/cluster_validation/plot_mass_segregation.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from progenax.cluster.validation import (
    generate_cluster_for_plot,
    sweep_mass_segregation_lambda,
)
from progenax.diagnostics import compute_lambda_msr


# =============================================================================
# Plot Setup
# =============================================================================

# Seaborn style
sns.set_theme(style="whitegrid", font_scale=1.2)
sns.set_palette("colorblind")

# Enable LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
})

PLOT_DIR = Path(__file__).parent.parent.parent / "validation" / "plots" / "cluster_ic" / "mass_segregation"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Figure 1: Λ_MSR vs λ_seg
# =============================================================================


def plot_lambda_msr_vs_lambda_seg(results: dict):
    """Plot Λ_MSR vs λ_seg with error bars."""
    fig, ax = plt.subplots(figsize=(7, 5))

    lambda_values = results["lambda_values"]
    msr_mean = results["lambda_msr_mean"]
    msr_std = results["lambda_msr_std"]

    ax.errorbar(
        lambda_values, msr_mean, yerr=msr_std,
        fmt="o-", capsize=5, capthick=2, markersize=10, linewidth=2,
        color=sns.color_palette()[0], ecolor=sns.color_palette()[1],
        label=r"$\Lambda_{\rm MSR}$ (N$_{\rm massive}$=10)"
    )

    ax.axhline(1.0, ls="--", color="gray", alpha=0.7, lw=1.5,
               label=r"No segregation ($\Lambda=1$)")

    ax.set_xlabel(r"$\lambda_{\rm seg}$ (segregation parameter)")
    ax.set_ylabel(r"$\Lambda_{\rm MSR}$ (mass segregation ratio)")
    ax.set_title(r"Mass Segregation: $\Lambda_{\rm MSR}$ vs $\lambda_{\rm seg}$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")

    plt.tight_layout()
    outpath = PLOT_DIR / "fig_mass_segregation_lambda_msr.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# =============================================================================
# Figure 2: Mean Radius of Massive Stars vs λ_seg
# =============================================================================


def plot_rmassive_vs_lambda_seg(results: dict):
    """Plot mean radius of massive stars vs λ_seg."""
    fig, ax = plt.subplots(figsize=(7, 5))

    lambda_values = results["lambda_values"]
    r_mean = results["r_massive_mean"]
    r_std = results["r_massive_std"]

    ax.errorbar(
        lambda_values, r_mean, yerr=r_std,
        fmt="s-", capsize=5, capthick=2, markersize=10, linewidth=2,
        color=sns.color_palette()[2], ecolor=sns.color_palette()[3],
        label=r"$\langle r \rangle$ of top 10 massive stars"
    )

    ax.set_xlabel(r"$\lambda_{\rm seg}$ (segregation parameter)")
    ax.set_ylabel(r"$\langle r_{\rm massive} \rangle$ [pc]")
    ax.set_title(r"Mass Segregation: Mean Massive Star Radius vs $\lambda_{\rm seg}$")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc="upper right")

    plt.tight_layout()
    outpath = PLOT_DIR / "fig_mass_segregation_rmassive.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# =============================================================================
# Figure 3: Radial Profile Comparison
# =============================================================================


def plot_radial_profiles(results: dict):
    """Plot radial profile comparison for λ=0 vs λ=1."""
    fig, ax = plt.subplots(figsize=(7, 5))

    bins = results["radial_bins"]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    if results["radial_hist_ref"] is not None:
        ax.step(bin_centers, results["radial_hist_ref"], where="mid",
                lw=2.5, color=sns.color_palette()[0],
                label=r"$\lambda_{\rm seg} = 0$ (unsegregated)")

    if results["radial_hist_seg"] is not None:
        ax.step(bin_centers, results["radial_hist_seg"], where="mid",
                lw=2.5, color=sns.color_palette()[1], ls="--",
                label=r"$\lambda_{\rm seg} = 1$ (segregated)")

    ax.set_xlabel(r"$r$ [pc]")
    ax.set_ylabel(r"Probability density $\rho(r)$")
    ax.set_title(r"Radial Density Profile Preservation")
    ax.legend()
    ax.set_xlim(0, bins[-1])

    plt.tight_layout()
    outpath = PLOT_DIR / "fig_mass_segregation_radial_profiles.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# =============================================================================
# Figure 4: Snapshot Projections
# =============================================================================


def plot_snapshots(key: jax.random.PRNGKey):
    """Plot x-y projections for λ=0, 0.5, 1.0."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    lambda_values = [0.0, 0.5, 1.0]
    titles = [
        r"(a) $\lambda_{\rm seg} = 0$",
        r"(b) $\lambda_{\rm seg} = 0.5$",
        r"(c) $\lambda_{\rm seg} = 1.0$",
    ]

    for ax, lam, title in zip(axes, lambda_values, titles):
        key, subkey = jax.random.split(key)

        cluster = generate_cluster_for_plot(
            subkey, lambda_seg=lam if lam > 0 else None, N_stars=5000
        )

        pos = np.array(cluster.positions)
        masses = np.array(cluster.masses)

        # Color by log mass
        log_mass = np.log10(masses)
        vmin, vmax = np.percentile(log_mass, [5, 95])

        # Size proportional to mass
        sizes = 5 + 30 * (masses / masses.max())

        sc = ax.scatter(
            pos[:, 0], pos[:, 1], c=log_mass, cmap="plasma",
            s=sizes, alpha=0.6, vmin=vmin, vmax=vmax
        )

        ax.set_xlabel(r"$x$ [pc]")
        ax.set_ylabel(r"$y$ [pc]")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, pad=0.02)
    cbar.set_label(r"$\log_{10}(M/M_\odot)$")

    fig.suptitle(r"Mass Segregation: $x$--$y$ Projections", fontsize=14, y=1.02)
    plt.tight_layout()

    outpath = PLOT_DIR / "fig_mass_segregation_snapshots.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# =============================================================================
# Figure 5: Cumulative Radial Distribution
# =============================================================================


def plot_cumulative_radius(key: jax.random.PRNGKey):
    """Plot CDF of radius for massive vs all stars at different λ_seg."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    lambda_values = [0.0, 0.5, 1.0]
    titles = [
        r"(a) $\lambda_{\rm seg} = 0$",
        r"(b) $\lambda_{\rm seg} = 0.5$",
        r"(c) $\lambda_{\rm seg} = 1.0$",
    ]
    N_massive = 20

    for ax, lam, title in zip(axes, lambda_values, titles):
        key, subkey = jax.random.split(key)

        cluster = generate_cluster_for_plot(
            subkey, lambda_seg=lam if lam > 0 else None, N_stars=5000
        )

        pos = np.array(cluster.positions)
        masses = np.array(cluster.masses)
        radii = np.linalg.norm(pos, axis=1)

        # Sort for CDF
        radii_sorted = np.sort(radii)
        cdf_all = np.arange(1, len(radii) + 1) / len(radii)

        # Massive stars
        massive_idx = np.argsort(-masses)[:N_massive]
        radii_massive = np.sort(radii[massive_idx])
        cdf_massive = np.arange(1, N_massive + 1) / N_massive

        ax.plot(radii_sorted, cdf_all, lw=2, color=sns.color_palette()[0],
                label="All stars", alpha=0.8)
        ax.plot(radii_massive, cdf_massive, lw=2.5, color=sns.color_palette()[1],
                label=f"Top {N_massive} massive", ls="--")

        ax.set_xlabel(r"$r$ [pc]")
        ax.set_ylabel(r"Cumulative fraction")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1)

    fig.suptitle(r"Cumulative Radial Distribution: Massive vs All Stars", fontsize=14, y=1.02)
    plt.tight_layout()

    outpath = PLOT_DIR / "fig_mass_segregation_cumulative_radius.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# =============================================================================
# Figure 6: Mass-Radius Scatter
# =============================================================================


def plot_mass_radius_scatter(key: jax.random.PRNGKey):
    """Plot stellar mass vs radius scatter for different λ_seg."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    lambda_values = [0.0, 0.5, 1.0]
    titles = [
        r"(a) $\lambda_{\rm seg} = 0$",
        r"(b) $\lambda_{\rm seg} = 0.5$",
        r"(c) $\lambda_{\rm seg} = 1.0$",
    ]

    for ax, lam, title in zip(axes, lambda_values, titles):
        key, subkey = jax.random.split(key)

        cluster = generate_cluster_for_plot(
            subkey, lambda_seg=lam if lam > 0 else None, N_stars=5000
        )

        pos = np.array(cluster.positions)
        masses = np.array(cluster.masses)
        radii = np.linalg.norm(pos, axis=1)

        # Scatter
        ax.scatter(radii, masses, s=3, alpha=0.3, color=sns.color_palette()[0])

        # Binned median
        bins = np.logspace(np.log10(0.1), np.log10(10), 15)
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
        bin_idx = np.digitize(radii, bins)

        median_mass = []
        for i in range(1, len(bins)):
            mask = bin_idx == i
            if np.sum(mask) > 5:
                median_mass.append(np.median(masses[mask]))
            else:
                median_mass.append(np.nan)

        ax.plot(bin_centers, median_mass, "o-", lw=2, markersize=6,
                color=sns.color_palette()[1], label="Binned median")

        ax.set_xlabel(r"$r$ [pc]")
        ax.set_ylabel(r"$M$ [$M_\odot$]")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 10)
        ax.legend(loc="upper right")

    fig.suptitle(r"Mass--Radius Relation", fontsize=14, y=1.02)
    plt.tight_layout()

    outpath = PLOT_DIR / "fig_mass_segregation_mass_radius.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Generate all mass segregation validation plots."""
    print("=" * 70)
    print("Mass Segregation Validation Plots")
    print("=" * 70)
    print()

    key = jax.random.PRNGKey(42)

    # Sweep λ_seg
    print("Running mass segregation sweep (this may take a few minutes)...")
    lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = sweep_mass_segregation_lambda(
        key, lambda_values,
        N_stars=3000,  # Reduced for speed
        n_realizations=10,
        N_massive=10,
    )

    print("\nGenerating plots...")

    # Generate plots
    plot_lambda_msr_vs_lambda_seg(results)
    plot_rmassive_vs_lambda_seg(results)
    plot_radial_profiles(results)

    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    plot_snapshots(subkey1)
    plot_cumulative_radius(subkey2)
    plot_mass_radius_scatter(subkey3)

    print()
    print(f"All plots saved to: {PLOT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
