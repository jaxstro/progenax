#!/usr/bin/env python
"""
Allison+09 style IC validation plot for progenax cluster IC generator.

Generates clusters with D=1.6, Q_vir=0.3, λ_frac=1.0 (rapid dynamical
mass segregation initial conditions) and verifies:
- Λ_MSR ≈ 1 (no primordial mass segregation)
- Q < 0.8 (fractal substructure present)

Usage:
    python scripts/cluster_validation/plot_allison_ic.py
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
)
from progenax.cluster.validation import generate_cluster_for_plot
from progenax.diagnostics import compute_lambda_msr, compute_q_parameter
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

PLOT_DIR = Path(__file__).parent.parent.parent / "validation" / "plots" / "cluster_ic" / "allison"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Main Plot
# =============================================================================


def plot_allison_ic(key: jax.random.PRNGKey, n_realizations: int = 20):
    """
    Generate Allison-style IC plot.

    Creates a 2-panel figure:
    - Left: x-y projection colored by mass
    - Right: Histogram of Λ_MSR over realizations

    Prints Q parameter for the shown cluster.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Allison IC parameters
    D = 1.6
    Q_vir = 0.3
    lambda_frac = 1.0
    N_stars = 1000
    N_massive = 10

    # Generate realizations and compute Λ_MSR
    print(f"\nGenerating {n_realizations} Allison-style ICs...")
    print(f"  D = {D}, Q_vir = {Q_vir}, λ_frac = {lambda_frac}, N = {N_stars}")

    lambda_msr_values = []
    Q_values = []
    imf = PowerLawIMF.kroupa()

    # For snapshot
    key, snapshot_key = jax.random.split(key)
    snapshot_cluster = None

    for i in range(n_realizations):
        key, subkey = jax.random.split(key)

        structure_params = SpatialStructureParams(
            base_profile="plummer",
            fractal=FractalLayer(D=D, lambda_frac=lambda_frac, virial_ratio=Q_vir),
        )

        cluster = generate_cluster_ic(
            key=subkey,
            N_stars=N_stars,
            M_total=float(N_stars),
            R_half=1.0,
            imf_params=imf,
            structure_params=structure_params,
        )

        pos_np = np.array(cluster.positions)
        masses_np = np.array(cluster.masses)

        # Compute Λ_MSR
        lam_msr, _ = compute_lambda_msr(
            pos_np, masses_np, N_massive=N_massive, N_random_samples=50
        )
        lambda_msr_values.append(lam_msr)

        # Compute Q
        Q = compute_q_parameter(pos_np)
        Q_values.append(Q)

        # Save first realization for snapshot
        if i == 0:
            snapshot_cluster = cluster

    lambda_msr_values = np.array(lambda_msr_values)
    Q_values = np.array(Q_values)

    # --- Panel (a): Snapshot ---
    ax = axes[0]
    pos = np.array(snapshot_cluster.positions)
    masses = np.array(snapshot_cluster.masses)

    log_mass = np.log10(masses)
    vmin, vmax = np.percentile(log_mass, [5, 95])
    sizes = 10 + 50 * (masses / masses.max())

    sc = ax.scatter(
        pos[:, 0], pos[:, 1], c=log_mass, cmap="plasma",
        s=sizes, alpha=0.7, vmin=vmin, vmax=vmax
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(r"$\log_{10}(M/M_\odot)$")

    ax.set_xlabel(r"$x$ [pc]")
    ax.set_ylabel(r"$y$ [pc]")
    ax.set_title(rf"(a) Allison IC: $D={D}$, $Q_{{\rm vir}}={Q_vir}$")
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    # Add Q annotation
    Q_snapshot = compute_q_parameter(pos)
    ax.text(0.05, 0.95, rf"$Q = {Q_snapshot:.3f}$",
            transform=ax.transAxes, fontsize=12, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # --- Panel (b): Λ_MSR histogram ---
    ax = axes[1]

    ax.hist(lambda_msr_values, bins=15, edgecolor="black", alpha=0.7,
            color=sns.color_palette()[0])
    ax.axvline(1.0, ls="--", color="red", lw=2,
               label=r"No segregation ($\Lambda_{\rm MSR} = 1$)")
    ax.axvline(np.mean(lambda_msr_values), ls="-", color="darkblue", lw=2,
               label=rf"Mean = {np.mean(lambda_msr_values):.2f}")

    ax.set_xlabel(r"$\Lambda_{\rm MSR}$")
    ax.set_ylabel("Count")
    ax.set_title(rf"(b) $\Lambda_{{\rm MSR}}$ Distribution ($N_{{\rm realizations}}={n_realizations}$)")
    ax.legend(loc="upper right")

    fig.suptitle(r"Allison+09 Style IC Validation", fontsize=14, y=1.02)
    plt.tight_layout()

    outpath = PLOT_DIR / "fig_allison_ic_snapshot.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")

    # Print summary statistics
    print()
    print("=" * 60)
    print("Allison IC Validation Summary")
    print("=" * 60)
    print(f"  D = {D}, Q_vir = {Q_vir}, λ_frac = {lambda_frac}, N = {N_stars}")
    print()
    print(f"  Λ_MSR: {np.mean(lambda_msr_values):.3f} ± {np.std(lambda_msr_values):.3f}")
    print(f"         Expected: ~1.0 (no primordial segregation)")
    print()
    print(f"  Q:     {np.mean(Q_values):.3f} ± {np.std(Q_values):.3f}")
    print(f"         Expected: < 0.8 (fractal substructure)")
    print()

    # Pass/fail
    lam_pass = 0.5 < np.mean(lambda_msr_values) < 2.0
    q_pass = np.mean(Q_values) < 0.8

    print(f"  Λ_MSR in [0.5, 2.0]: {'PASS' if lam_pass else 'FAIL'}")
    print(f"  Q < 0.8:             {'PASS' if q_pass else 'FAIL'}")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================


def main():
    """Generate Allison IC validation plot."""
    print("=" * 70)
    print("Allison+09 Style IC Validation")
    print("=" * 70)

    key = jax.random.PRNGKey(42)
    plot_allison_ic(key, n_realizations=20)

    print()
    print(f"Plot saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
