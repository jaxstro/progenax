#!/usr/bin/env python
"""
Fractal substructure validation plots for progenax cluster IC generator.

Produces publication-quality figures:
1. Q vs D with error bars
2. σΣ/⟨Σ⟩ vs D with Küpper+11 relation overlay
3. Fractal snapshots (D = 1.6, 2.0, 2.6, 3.0)
4. Blending snapshots (λ_frac = 0, 0.5, 1.0)
5. Velocity field comparison (coherent vs incoherent)

Usage:
    python scripts/cluster_validation/plot_fractal.py
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

from progenax.cluster import (
    FractalLayer,
    SpatialStructureParams,
    generate_cluster_ic,
    generate_fractal_positions,
)
from progenax.cluster.fractal_gw_legacy import assign_velocities_and_virialize
from progenax.cluster.validation import (
    generate_cluster_for_plot,
    sweep_fractal_dimension,
)
from progenax.imf import PowerLawIMF


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

PLOT_DIR = Path(__file__).parent.parent.parent / "validation" / "plots" / "cluster_ic" / "fractal"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Figure 1: Q vs D
# =============================================================================


def plot_Q_vs_D(results: dict):
    """Plot Cartwright-Whitworth Q vs fractal dimension D."""
    fig, ax = plt.subplots(figsize=(7, 5))

    D_values = results["D_values"]
    Q_mean = results["Q_mean"]
    Q_std = results["Q_std"]

    ax.errorbar(
        D_values, Q_mean, yerr=Q_std,
        fmt="o-", capsize=5, capthick=2, markersize=10, linewidth=2,
        color=sns.color_palette()[0], ecolor=sns.color_palette()[1],
        label=r"$Q$ (Cartwright \& Whitworth)"
    )

    ax.axhline(0.8, ls="--", color="gray", alpha=0.7, lw=1.5,
               label=r"Uniform sphere ($Q \approx 0.8$)")

    ax.set_xlabel(r"Fractal dimension $D$")
    ax.set_ylabel(r"$Q$ parameter")
    ax.set_title(r"Fractal Substructure: $Q$ vs $D$")
    ax.set_xlim(1.4, 3.2)
    ax.legend(loc="lower right")

    plt.tight_layout()
    outpath = PLOT_DIR / "fig_fractal_Q_vs_D.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# =============================================================================
# Figure 2: σΣ/⟨Σ⟩ vs D
# =============================================================================


def plot_sigmaSigma_vs_D(results: dict):
    """Plot azimuthal variation σΣ/⟨Σ⟩ vs D with Küpper+11 relation."""
    fig, ax = plt.subplots(figsize=(7, 5))

    D_values = results["D_values"]
    sigma_mean = results["sigmaSigma_over_mean_mean"]
    sigma_std = results["sigmaSigma_over_mean_std"]

    # Data points
    ax.errorbar(
        D_values, sigma_mean, yerr=sigma_std,
        fmt="s-", capsize=5, capthick=2, markersize=10, linewidth=2,
        color=sns.color_palette()[2], ecolor=sns.color_palette()[3],
        label=r"Measured $\sigma_\Sigma / \langle\Sigma\rangle$"
    )

    # Küpper+11 relation: σΣ/⟨Σ⟩ ≈ -0.46 D + 1.45
    D_theory = np.linspace(1.4, 3.2, 50)
    sigma_theory = -0.46 * D_theory + 1.45

    ax.plot(D_theory, sigma_theory, ls="--", lw=2, color="gray",
            label=r"K\"upper+11: $-0.46D + 1.45$")

    ax.set_xlabel(r"Fractal dimension $D$")
    ax.set_ylabel(r"$\sigma_\Sigma / \langle\Sigma\rangle$")
    ax.set_title(r"Azimuthal Density Variation vs Fractal Dimension")
    ax.set_xlim(1.4, 3.2)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")

    plt.tight_layout()
    outpath = PLOT_DIR / "fig_fractal_sigmaSigma_vs_D.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# =============================================================================
# Figure 3: Fractal Snapshots
# =============================================================================


def plot_fractal_snapshots(key: jax.random.PRNGKey):
    """Plot x-y projections for D = 1.6, 2.0, 2.6, 3.0."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    D_values = [1.6, 2.0, 2.6, 3.0]
    titles = [
        r"(a) $D = 1.6$ (very clumpy)",
        r"(b) $D = 2.0$ (moderately clumpy)",
        r"(c) $D = 2.6$ (weakly clumpy)",
        r"(d) $D = 3.0$ (uniform)",
    ]

    for ax, D, title in zip(axes, D_values, titles):
        key, subkey = jax.random.split(key)

        cluster = generate_cluster_for_plot(
            subkey, D=D, lambda_frac=1.0, Q_vir=0.5, N_stars=5000
        )

        pos = np.array(cluster.positions)
        masses = np.array(cluster.masses)

        # Color by log mass
        log_mass = np.log10(masses)
        vmin, vmax = np.percentile(log_mass, [5, 95])

        # Size proportional to mass
        sizes = 3 + 15 * (masses / masses.max())

        ax.scatter(
            pos[:, 0], pos[:, 1], c=log_mass, cmap="viridis",
            s=sizes, alpha=0.6, vmin=vmin, vmax=vmax
        )

        ax.set_xlabel(r"$x$ [pc]")
        ax.set_ylabel(r"$y$ [pc]")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

    fig.suptitle(r"Fractal ICs: $x$--$y$ Projections at Different $D$", fontsize=14, y=1.02)
    plt.tight_layout()

    outpath = PLOT_DIR / "fig_fractal_snapshots.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# =============================================================================
# Figure 4: Blending Snapshots (λ_frac)
# =============================================================================


def plot_lambda_frac_snapshots(key: jax.random.PRNGKey):
    """Plot x-y projections for λ_frac = 0, 0.5, 1.0 at fixed D=1.6."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    lambda_frac_values = [0.0, 0.5, 1.0]
    titles = [
        r"(a) $\lambda_{\rm frac} = 0$ (smooth)",
        r"(b) $\lambda_{\rm frac} = 0.5$ (blended)",
        r"(c) $\lambda_{\rm frac} = 1.0$ (fully fractal)",
    ]

    D = 1.6

    for ax, lam_frac, title in zip(axes, lambda_frac_values, titles):
        key, subkey = jax.random.split(key)

        if lam_frac == 0.0:
            # Pure smooth profile
            cluster = generate_cluster_for_plot(
                subkey, lambda_seg=None, D=None, N_stars=5000
            )
        else:
            cluster = generate_cluster_for_plot(
                subkey, D=D, lambda_frac=lam_frac, Q_vir=0.5, N_stars=5000
            )

        pos = np.array(cluster.positions)
        masses = np.array(cluster.masses)

        # Color by log mass
        log_mass = np.log10(masses)
        vmin, vmax = np.percentile(log_mass, [5, 95])

        sizes = 3 + 15 * (masses / masses.max())

        sc = ax.scatter(
            pos[:, 0], pos[:, 1], c=log_mass, cmap="viridis",
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

    fig.suptitle(rf"Fractal Blending at $D = {D}$: Effect of $\lambda_{{\rm frac}}$",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    outpath = PLOT_DIR / "fig_fractal_lambda_snapshots.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# =============================================================================
# Figure 5: Velocity Fields
# =============================================================================


def plot_velocity_fields(key: jax.random.PRNGKey):
    """Plot velocity quiver plots for coherent vs incoherent modes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)

    # Generate fractal positions with ancestry
    N = 500
    D = 1.6
    positions, vel_frac, ancestry = generate_fractal_positions(subkey1, N, D=D)

    # Scale to physical units
    R_half = 1.0
    positions = positions * R_half * 2  # Approximate scaling

    # Random masses
    masses = jax.random.uniform(subkey2, (N,), minval=0.5, maxval=2.0)

    # Coherent velocities
    vel_coherent = assign_velocities_and_virialize(
        subkey3, positions, masses, target_Q_vir=0.3,
        ancestry=ancestry, coherent=True
    )

    # Incoherent velocities
    vel_incoherent = assign_velocities_and_virialize(
        subkey4, positions, masses, target_Q_vir=0.3,
        ancestry=ancestry, coherent=False
    )

    pos_np = np.array(positions)
    vel_coh_np = np.array(vel_coherent)
    vel_incoh_np = np.array(vel_incoherent)
    ancestry_np = np.array(ancestry)

    # Color by ancestry for visualization
    colors_coh = ancestry_np % 20  # Modulo for color cycling

    # Panel (a): Coherent velocities
    ax = axes[0]
    q = ax.quiver(
        pos_np[:, 0], pos_np[:, 1],
        vel_coh_np[:, 0], vel_coh_np[:, 1],
        colors_coh, cmap="tab20", alpha=0.7,
        scale=1.5, width=0.004
    )
    ax.set_xlabel(r"$x$ [pc]")
    ax.set_ylabel(r"$y$ [pc]")
    ax.set_title(r"(a) Coherent velocities (ancestry-based)")
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    # Panel (b): Incoherent velocities
    ax = axes[1]
    ax.quiver(
        pos_np[:, 0], pos_np[:, 1],
        vel_incoh_np[:, 0], vel_incoh_np[:, 1],
        colors_coh, cmap="tab20", alpha=0.7,
        scale=1.5, width=0.004
    )
    ax.set_xlabel(r"$x$ [pc]")
    ax.set_ylabel(r"$y$ [pc]")
    ax.set_title(r"(b) Incoherent velocities (random)")
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    fig.suptitle(rf"Velocity Field Comparison ($D = {D}$, $Q_{{\rm vir}} = 0.3$)",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    outpath = PLOT_DIR / "fig_fractal_velocity_fields.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# =============================================================================
# Quantitative Results
# =============================================================================


def print_quantitative_results(results: dict):
    """Print quantitative validation results table."""
    print()
    print("=" * 70)
    print("QUANTITATIVE VALIDATION RESULTS")
    print("=" * 70)
    print()

    D_values = results["D_values"]
    Q_mean = results["Q_mean"]
    Q_std = results["Q_std"]
    sigma_mean = results["sigmaSigma_over_mean_mean"]
    sigma_std = results["sigmaSigma_over_mean_std"]

    # Küpper+11 relation: σΣ/⟨Σ⟩ ≈ -0.46 D + 1.45
    sigma_theory = -0.46 * np.array(D_values) + 1.45

    # Q parameter expectations (Cartwright & Whitworth 2004):
    # D=3.0 (uniform) -> Q ≈ 0.8
    # D<2.0 (fractal) -> Q < 0.8
    Q_uniform_expected = 0.8

    print(f"{'D':<8} {'Q (measured)':<18} {'σΣ/⟨Σ⟩ (measured)':<22} {'σΣ/⟨Σ⟩ (theory)':<18} {'Δσ'}")
    print("-" * 70)

    all_passed = True
    for i, D in enumerate(D_values):
        Q_str = f"{Q_mean[i]:.3f} ± {Q_std[i]:.3f}"
        sigma_str = f"{sigma_mean[i]:.3f} ± {sigma_std[i]:.3f}"
        sigma_th_str = f"{sigma_theory[i]:.3f}"
        delta_sigma = sigma_mean[i] - sigma_theory[i]
        delta_str = f"{delta_sigma:+.3f}"

        # Check if within 0.3 of theory (generous for stochastic ICs)
        is_ok = abs(delta_sigma) < 0.3
        status = "✅" if is_ok else "⚠️"
        if not is_ok:
            all_passed = False

        print(f"{D:<8} {Q_str:<18} {sigma_str:<22} {sigma_th_str:<18} {delta_str} {status}")

    print("-" * 70)
    print()

    # Q parameter check
    print("Q Parameter Analysis:")
    print(f"  D=3.0 (uniform): Q = {Q_mean[-1]:.3f} ± {Q_std[-1]:.3f}, expected ≈ 0.8")
    q_uniform_ok = abs(Q_mean[-1] - Q_uniform_expected) < 0.15
    print(f"  Status: {'✅ Within 0.15 of expected' if q_uniform_ok else '⚠️ Deviation > 0.15'}")
    if not q_uniform_ok:
        all_passed = False

    print()
    print(f"  D=1.6 (clumpy): Q = {Q_mean[0]:.3f} ± {Q_std[0]:.3f}")
    q_decreases = Q_mean[0] < Q_mean[-1]
    print(f"  Status: {'✅ Q decreases with D (expected)' if q_decreases else '⚠️ Q should decrease with D'}")
    if not q_decreases:
        all_passed = False

    print()
    print("=" * 70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("⚠️ SOME VALIDATIONS NEED REVIEW")
    print("=" * 70)

    return all_passed


# =============================================================================
# Main
# =============================================================================


def main():
    """Generate all fractal validation plots."""
    print("=" * 70)
    print("Fractal Substructure Validation Plots")
    print("=" * 70)
    print()

    key = jax.random.PRNGKey(42)

    # Sweep D
    print("Running fractal dimension sweep (this may take a few minutes)...")
    D_values = [1.6, 2.0, 2.6, 3.0]
    results = sweep_fractal_dimension(
        key, D_values,
        N_stars=1000,  # Must be ≤1000 for D=1.6 with g_max=6 and replace=False
        n_realizations=10,
    )

    # Print quantitative results
    all_passed = print_quantitative_results(results)

    print("\nGenerating plots...")

    # Generate plots
    plot_Q_vs_D(results)
    plot_sigmaSigma_vs_D(results)

    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    plot_fractal_snapshots(subkey1)
    plot_lambda_frac_snapshots(subkey2)
    plot_velocity_fields(subkey3)

    print()
    print(f"All plots saved to: {PLOT_DIR}")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    main()
