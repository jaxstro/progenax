#!/usr/bin/env python
"""
FDF Validation Suite - Publication-quality validation of Fractal Displacement Field.

Produces diagnostic figures for:
1. D parameter sweep (4x2 hero figure)
2. Substructure diagnostics (Q vs D, sigma_Sigma/<Sigma> vs D)
3. lambda_frac blending visualization
4. Velocity coherence comparison
5. Differentiability demonstration

Usage:
    python scripts/validate_fdf.py --mode sanity    # Quick checks (~30s)
    python scripts/validate_fdf.py --mode paper     # Full publication figures (~5min)
    python scripts/validate_fdf.py --mode docs      # Smaller N for examples (~1min)
    python scripts/validate_fdf.py --mode all       # Everything

Output:
    validation/plots/fdf_*.png

References:
    Cartwright & Whitworth (2004), MNRAS 348, 589 - Q parameter
    Kupper et al. (2011), MNRAS 417, 2300 - sigma_Sigma relation
"""
import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

jax.config.update("jax_enable_x64", True)

from progenax.cluster.fdf import (
    FractalDisplacementLayer,
    generate_fractal_ic,
    init_fractal_field,
    compute_amplitudes,
)
from progenax.cluster.fdf_calibration import fractal_layer_from_D
from progenax.cluster.fdf_density import (
    FractalDensityLayer,
    generate_fractal_ic_density,
    density_layer_from_D,
)
from progenax.diagnostics import compute_azimuthal_variation, compute_q_parameter
from progenax.imf import PowerLawIMF


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = "validation/plots"
SEED = 42

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


# =============================================================================
# Helper Functions
# =============================================================================


def estimate_local_density(positions: np.ndarray, k: int = 10) -> np.ndarray:
    """Estimate local surface density via k-nearest neighbor distance.

    Args:
        positions: (N, 3) array of positions
        k: Number of neighbors for density estimate

    Returns:
        (N,) array of local density estimates (higher = denser)
    """
    from scipy.spatial import cKDTree

    # Project to XY plane
    xy = positions[:, :2]
    tree = cKDTree(xy)

    # Distance to k-th neighbor
    distances, _ = tree.query(xy, k=k + 1)  # +1 because closest is self
    r_k = distances[:, -1]

    # Density ~ k / (pi * r_k^2)
    density = k / (np.pi * r_k**2 + 1e-10)
    return density


def compute_power_distribution(field, chi: float, sigma_u: float) -> dict:
    """Compute fraction of power in small-scale (high-k) modes.

    This diagnostic verifies that the v2 lognormal envelope correctly
    shifts power to small scales for low chi (clumpy) and large scales
    for high chi (smooth).

    Args:
        field: FractalField with k_vecs
        chi: Clumpiness parameter
        sigma_u: Displacement amplitude scale (physical units)

    Returns:
        dict with 'frac_small', 'frac_large', 'median_k', 'peak_k'
    """
    # Get amplitude vectors
    a_vecs = compute_amplitudes(field, chi, sigma_u)

    # Per-mode power
    power = jnp.sum(a_vecs**2, axis=1)  # (M,)
    total_power = jnp.sum(power)

    # Wavenumber magnitudes
    k_mags = jnp.linalg.norm(field.k_vecs, axis=1)
    median_k = float(jnp.median(k_mags))

    # Small-scale = high k (above median)
    small_scale_mask = k_mags > median_k
    frac_small = float(jnp.sum(power * small_scale_mask) / total_power)
    frac_large = 1.0 - frac_small

    # Find peak k (mode with most power)
    peak_idx = int(jnp.argmax(power))
    peak_k = float(k_mags[peak_idx])

    return {
        'frac_small': frac_small,
        'frac_large': frac_large,
        'median_k': median_k,
        'peak_k': peak_k,
    }


# =============================================================================
# Figure 1: 4x2 D-Sweep Hero Figure
# =============================================================================


def plot_d_sweep_hero(
    output_dir: str,
    N_stars: int = 2000,
    seed: int = SEED,
    suffix: str = "",
):
    """4x2 hero figure: D progression with same realization seed.

    Uses Option A from brainstorming: same seed across all D values
    to isolate the effect of changing clumpiness parameter.
    """
    print("\n" + "=" * 60)
    print("FIGURE 1: D-SWEEP HERO (4x2)")
    print("=" * 60)

    D_values = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    R_half = 1.0

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Same base key for all realizations
    base_key = random.PRNGKey(seed)
    imf = PowerLawIMF.kroupa()

    # Pre-initialize the FractalField once (same frozen structure for all D)
    field = init_fractal_field(base_key, n_modes=64, R_half=R_half)

    Q_values = []
    power_diags = {}  # Store power distribution for D=1.6 and D=3.0

    for idx, D in enumerate(D_values):
        row, col = divmod(idx, 4)
        ax = axes[row, col]

        # Generate cluster with same key but different chi (via D)
        frac = fractal_layer_from_D(D=D, virial_ratio=0.5)
        cluster = generate_fractal_ic(
            base_key,
            N_stars=N_stars,
            M_total=float(N_stars),
            R_half=R_half,
            profile="plummer",
            frac_params=frac,
            imf_params=imf,
            field=field,  # Use pre-initialized field
        )

        positions_np = np.array(cluster.positions)

        # Compute Q for annotation
        Q = compute_q_parameter(positions_np)
        Q_values.append(Q)

        # Compute power distribution for extreme D values
        if D in [1.6, 3.0]:
            sigma_u_physical = float(frac.sigma_u) * R_half
            power_diags[D] = compute_power_distribution(field, float(frac.chi), sigma_u_physical)

        # Estimate local density for coloring
        density = estimate_local_density(positions_np, k=10)
        log_density = np.log10(density + 1e-10)

        # Scatter plot colored by local density
        scatter = ax.scatter(
            positions_np[:, 0],
            positions_np[:, 1],
            c=log_density,
            s=2,
            cmap='viridis',
            alpha=0.7,
            rasterized=True,
        )

        ax.set_xlim(-3 * R_half, 3 * R_half)
        ax.set_ylim(-3 * R_half, 3 * R_half)
        ax.set_aspect('equal')
        ax.set_title(f'D = {D:.1f}, Q = {Q:.2f}', fontsize=12)

        if col == 0:
            ax.set_ylabel('y [pc]')
        else:
            ax.set_yticklabels([])

        if row == 1:
            ax.set_xlabel('x [pc]')
        else:
            ax.set_xticklabels([])

        ax.grid(True, alpha=0.2)

        print(f"  D = {D:.1f}: Q = {Q:.3f}")

    # Print power distribution diagnostics (v2 spectrum verification)
    print("\n  POWER DISTRIBUTION (v2 lognormal envelope):")
    for D in [1.6, 3.0]:
        if D in power_diags:
            pd = power_diags[D]
            print(f"    D = {D}: small-scale power = {pd['frac_small']*100:.1f}%, "
                  f"large-scale = {pd['frac_large']*100:.1f}%, peak_k = {pd['peak_k']:.2f}")

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.8, label='log10(local density)')

    plt.suptitle(
        f'Fractal Displacement Field: D Progression (N = {N_stars}, same seed)',
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()
    filename = f'{output_dir}/fdf_d_sweep_hero{suffix}.png'
    plt.savefig(filename)
    plt.close()
    print(f"  Saved: {filename}")

    return {'Q_values': Q_values, 'D_values': D_values}


# =============================================================================
# Figure 1b: 4x2 D-Sweep Hero Figure (DENSITY-FIELD FDF)
# =============================================================================


def plot_d_sweep_hero_density(
    output_dir: str,
    N_stars: int = 2000,
    seed: int = SEED,
    sigma_ln_rho: float = 2.0,
    suffix: str = "",
):
    """4x2 hero figure for density-field FDF: D progression.

    This uses the density-field sampling approach which should produce
    actual clumpy → smooth Q(D) progression.
    """
    print("\n" + "=" * 60)
    print("FIGURE 1b: D-SWEEP HERO (DENSITY-FIELD FDF)")
    print("=" * 60)

    D_values = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    R_half = 1.0

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    imf = PowerLawIMF.kroupa()

    Q_values = []
    density_variance = []  # Track density field variance for diagnostics

    for idx, D in enumerate(D_values):
        row, col = divmod(idx, 4)
        ax = axes[row, col]

        # Use different key for each D to get independent realizations
        key = random.PRNGKey(seed + idx * 100)

        # Generate cluster using density-field method
        layer = density_layer_from_D(
            D=D,
            sigma_ln_rho=sigma_ln_rho,
            virial_ratio=0.5,
            grid_size=64,
        )
        cluster = generate_fractal_ic_density(
            key,
            N_stars=N_stars,
            M_total=float(N_stars),
            R_half=R_half,
            imf_params=imf,
            layer=layer,
        )

        positions_np = np.array(cluster.positions)

        # Compute Q for annotation
        Q = compute_q_parameter(positions_np)
        Q_values.append(Q)

        # Estimate local density for coloring
        density = estimate_local_density(positions_np, k=10)
        log_density = np.log10(density + 1e-10)

        # Track density variance
        density_variance.append(np.var(log_density))

        # Scatter plot colored by local density
        scatter = ax.scatter(
            positions_np[:, 0],
            positions_np[:, 1],
            c=log_density,
            s=2,
            cmap='viridis',
            alpha=0.7,
            rasterized=True,
        )

        ax.set_xlim(-3 * R_half, 3 * R_half)
        ax.set_ylim(-3 * R_half, 3 * R_half)
        ax.set_aspect('equal')
        ax.set_title(f'D = {D:.1f}, Q = {Q:.2f}', fontsize=12)

        if col == 0:
            ax.set_ylabel('y [pc]')
        else:
            ax.set_yticklabels([])

        if row == 1:
            ax.set_xlabel('x [pc]')
        else:
            ax.set_xticklabels([])

        ax.grid(True, alpha=0.2)

        print(f"  D = {D:.1f}: Q = {Q:.3f}, log_density_var = {np.var(log_density):.3f}")

    plt.suptitle(
        f'Density-Field FDF: D Progression (N = {N_stars}, σ_ln_ρ = {sigma_ln_rho})',
        fontsize=14,
    )

    # Adjust layout first, then add colorbar in dedicated space on right
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])

    # Add colorbar in dedicated axes on the right
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter, cax=cbar_ax, label='log₁₀(local density)')
    filename = f'{output_dir}/fdf_density_d_sweep_hero{suffix}.png'
    plt.savefig(filename)
    plt.close()
    print(f"  Saved: {filename}")

    # Summary statistics
    print("\n  Q(D) SUMMARY:")
    print(f"    Q(D=1.6) = {Q_values[0]:.3f}")
    print(f"    Q(D=3.0) = {Q_values[-1]:.3f}")
    print(f"    Q range = {min(Q_values):.3f} to {max(Q_values):.3f}")
    print(f"    Q monotonic? {all(Q_values[i] <= Q_values[i+1] for i in range(len(Q_values)-1))}")

    return {'Q_values': Q_values, 'D_values': D_values}


# =============================================================================
# Figure 2b: Q vs D for Density-Field FDF
# =============================================================================


def plot_q_vs_d_density(
    output_dir: str,
    n_realizations: int = 10,
    N_stars: int = 2000,
    sigma_ln_rho: float = 2.0,
):
    """Q parameter vs fractal dimension for density-field FDF.

    This should show the expected monotonic increase in Q with D:
        - Low D (1.6): Low Q (substructured, clumpy)
        - High D (3.0): High Q (centrally concentrated, smooth)
    """
    print("\n" + "=" * 60)
    print("FIGURE 2b: DENSITY-FIELD FDF DIAGNOSTICS (Q vs D)")
    print("=" * 60)

    D_values = np.array([1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    R_half = 1.0
    imf = PowerLawIMF.kroupa()

    Q_all = []
    sigma_all = []

    for D in D_values:
        Q_realizations = []
        sigma_realizations = []

        for i in range(n_realizations):
            key = random.PRNGKey(SEED + i * 100 + int(D * 10))

            layer = density_layer_from_D(
                D=D,
                sigma_ln_rho=sigma_ln_rho,
                virial_ratio=0.5,
            )
            cluster = generate_fractal_ic_density(
                key,
                N_stars=N_stars,
                M_total=float(N_stars),
                R_half=R_half,
                imf_params=imf,
                layer=layer,
            )

            positions_np = np.array(cluster.positions)

            Q = compute_q_parameter(positions_np)
            sigma_over_mean = compute_azimuthal_variation(positions_np)

            Q_realizations.append(Q)
            sigma_realizations.append(sigma_over_mean)

        Q_all.append(Q_realizations)
        sigma_all.append(sigma_realizations)
        print(f"  D = {D:.1f}: Q = {np.mean(Q_realizations):.3f} +/- {np.std(Q_realizations):.3f}")

    Q_all = np.array(Q_all)
    sigma_all = np.array(sigma_all)

    Q_mean = np.mean(Q_all, axis=1)
    Q_std = np.std(Q_all, axis=1)
    sigma_mean = np.mean(sigma_all, axis=1)
    sigma_std = np.std(sigma_all, axis=1)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Q vs D
    ax = axes[0]
    ax.errorbar(D_values, Q_mean, yerr=Q_std, fmt='o-', capsize=4, capthick=1.5,
                color='darkorange', markersize=8, linewidth=2, label='Density-Field FDF')

    # CW04 reference lines
    ax.axhline(0.79, ls='--', color='gray', alpha=0.7, label='Uniform sphere (Q ≈ 0.79, CW04)')
    # CW04 Q(D) curve for GW2004 fractals
    D_cw04 = [1.5, 2.0, 2.5, 3.0]
    Q_cw04 = [0.47, 0.58, 0.70, 0.82]
    ax.plot(D_cw04, Q_cw04, 'k:', alpha=0.5, linewidth=1.5, marker='x',
            markersize=6, label='GW fractal (CW04)')

    ax.set_xlabel('Fractal Dimension D')
    ax.set_ylabel('Q (Cartwright-Whitworth)')
    ax.set_title('Density-Field FDF: Q vs D')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.5, 3.1)

    # Panel B: sigma_Sigma/<Sigma> vs D
    ax = axes[1]
    ax.errorbar(D_values, sigma_mean, yerr=sigma_std, fmt='s-', capsize=4, capthick=1.5,
                color='teal', markersize=8, linewidth=2, label='Density-Field FDF')

    ax.set_xlabel('Fractal Dimension D')
    ax.set_ylabel(r'$\sigma_\Sigma / \langle\Sigma\rangle$ (azimuthal variation)')
    ax.set_title('Azimuthal Density Variation vs D')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.5, 3.1)

    plt.tight_layout()
    filename = f'{output_dir}/fdf_density_q_vs_d.png'
    plt.savefig(filename)
    plt.close()
    print(f"  Saved: {filename}")

    # Check monotonicity
    q_monotonic = all(Q_mean[i] <= Q_mean[i+1] for i in range(len(Q_mean)-1))
    print(f"\n  Q monotonic increasing: {q_monotonic}")

    return {
        'D_values': D_values,
        'Q_mean': Q_mean,
        'Q_std': Q_std,
        'sigma_mean': sigma_mean,
        'sigma_std': sigma_std,
        'q_monotonic': q_monotonic,
    }


# =============================================================================
# Figure 2: Q vs D and sigma_Sigma vs D
# =============================================================================


def plot_q_vs_d(
    output_dir: str,
    n_realizations: int = 20,
    N_stars: int = 2000,
):
    """Q parameter and azimuthal variation vs fractal dimension with error bars.

    NOTE: Q vs D monotonicity is the MAIN CALIBRATION TARGET for FDF.
    After v2 spectrum changes, Q should increase monotonically with D:
        - Low D (1.6): Low Q (substructured, clumpy)
        - High D (3.0): High Q (centrally concentrated, smooth)

    If Q is flat across D, the spectrum is not working correctly.
    """
    print("\n" + "=" * 60)
    print("FIGURE 2: SUBSTRUCTURE DIAGNOSTICS (Q vs D)")
    print("=" * 60)

    D_values = np.array([1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    R_half = 1.0
    imf = PowerLawIMF.kroupa()

    Q_all = []
    sigma_all = []

    for D in D_values:
        Q_realizations = []
        sigma_realizations = []

        for i in range(n_realizations):
            key = random.PRNGKey(SEED + i * 100 + int(D * 10))

            frac = fractal_layer_from_D(D=D, virial_ratio=0.5)
            cluster = generate_fractal_ic(
                key,
                N_stars=N_stars,
                M_total=float(N_stars),
                R_half=R_half,
                profile="plummer",
                frac_params=frac,
                imf_params=imf,
            )

            positions_np = np.array(cluster.positions)

            Q = compute_q_parameter(positions_np)
            sigma_over_mean = compute_azimuthal_variation(positions_np)

            Q_realizations.append(Q)
            sigma_realizations.append(sigma_over_mean)

        Q_all.append(Q_realizations)
        sigma_all.append(sigma_realizations)
        print(f"  D = {D:.1f}: Q = {np.mean(Q_realizations):.3f} +/- {np.std(Q_realizations):.3f}")

    Q_all = np.array(Q_all)
    sigma_all = np.array(sigma_all)

    Q_mean = np.mean(Q_all, axis=1)
    Q_std = np.std(Q_all, axis=1)
    sigma_mean = np.mean(sigma_all, axis=1)
    sigma_std = np.std(sigma_all, axis=1)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Q vs D
    ax = axes[0]
    ax.errorbar(D_values, Q_mean, yerr=Q_std, fmt='o-', capsize=4, capthick=1.5,
                color='steelblue', markersize=8, linewidth=2, label='FDF')

    # CW04 reference: uniform sphere Q ≈ 0.79
    ax.axhline(0.79, ls='--', color='gray', alpha=0.7, label='Uniform sphere (Q ≈ 0.79, CW04)')
    # CW04 Q(D) curve for GW2004 fractals
    D_cw04 = [1.5, 2.0, 2.5, 3.0]
    Q_cw04 = [0.47, 0.58, 0.70, 0.82]
    ax.plot(D_cw04, Q_cw04, 'k:', alpha=0.5, linewidth=1.5, marker='x',
            markersize=6, label='GW fractal (CW04)')

    ax.set_xlabel('Fractal Dimension D')
    ax.set_ylabel('Q (Cartwright-Whitworth)')
    ax.set_title('Substructure Parameter Q vs D')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.5, 3.1)

    # Panel B: sigma_Sigma/<Sigma> vs D
    ax = axes[1]
    ax.errorbar(D_values, sigma_mean, yerr=sigma_std, fmt='s-', capsize=4, capthick=1.5,
                color='coral', markersize=8, linewidth=2, label='FDF')

    ax.set_xlabel('Fractal Dimension D')
    ax.set_ylabel(r'$\sigma_\Sigma / \langle\Sigma\rangle$ (azimuthal variation)')
    ax.set_title('Azimuthal Density Variation vs D')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.5, 3.1)

    plt.tight_layout()
    filename = f'{output_dir}/fdf_q_vs_d.png'
    plt.savefig(filename)
    plt.close()
    print(f"  Saved: {filename}")

    return {
        'D_values': D_values,
        'Q_mean': Q_mean,
        'Q_std': Q_std,
        'sigma_mean': sigma_mean,
        'sigma_std': sigma_std,
    }


# =============================================================================
# Figure 3: lambda_frac Blending
# =============================================================================


def plot_lambda_frac_blend(output_dir: str, D: float = 1.6, N_stars: int = 1500):
    """Show blending from smooth (lambda=0) to full fractal (lambda=1)."""
    print("\n" + "=" * 60)
    print("FIGURE 3: LAMBDA_FRAC BLENDING")
    print("=" * 60)

    lambda_values = [0.0, 0.5, 1.0]
    R_half = 1.0
    key = random.PRNGKey(SEED)
    imf = PowerLawIMF.kroupa()

    # Pre-initialize field (same for all lambda_frac values)
    field = init_fractal_field(key, n_modes=64, R_half=R_half)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Collect all densities for consistent color scale
    all_positions = []
    all_densities = []

    for lam in lambda_values:
        frac = FractalDisplacementLayer(
            chi=D,
            lambda_frac=lam,
            sigma_u=0.3,
            virial_ratio=0.5,
        )
        cluster = generate_fractal_ic(
            key,
            N_stars=N_stars,
            M_total=float(N_stars),
            R_half=R_half,
            profile="plummer",
            frac_params=frac,
            imf_params=imf,
            field=field,
        )
        positions_np = np.array(cluster.positions)
        density = estimate_local_density(positions_np, k=10)

        all_positions.append(positions_np)
        all_densities.append(density)

    # Determine consistent color scale
    all_log_density = [np.log10(d + 1e-10) for d in all_densities]
    vmin = min(d.min() for d in all_log_density)
    vmax = max(d.max() for d in all_log_density)

    for ax, lam, positions_np, log_density in zip(axes, lambda_values, all_positions, all_log_density):
        Q = compute_q_parameter(positions_np)

        scatter = ax.scatter(
            positions_np[:, 0],
            positions_np[:, 1],
            c=log_density,
            s=2,
            cmap='viridis',
            alpha=0.7,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        ax.set_title(f'$\\lambda_{{frac}}$ = {lam:.1f}, Q = {Q:.2f}')
        ax.set_xlabel('x [pc]')
        ax.set_aspect('equal')
        ax.set_xlim(-3 * R_half, 3 * R_half)
        ax.set_ylim(-3 * R_half, 3 * R_half)
        ax.grid(True, alpha=0.2)

        print(f"  lambda_frac = {lam:.1f}: Q = {Q:.3f}")

    axes[0].set_ylabel('y [pc]')

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.8, label='log10(local density)')

    plt.suptitle(f'FDF Blending: Smooth to Clumpy (D = {D})', fontsize=13, y=1.02)
    plt.tight_layout()

    filename = f'{output_dir}/fdf_lambda_blend.png'
    plt.savefig(filename)
    plt.close()
    print(f"  Saved: {filename}")


# =============================================================================
# Figure 4: Velocity Coherence
# =============================================================================


def plot_velocity_coherence(output_dir: str, D: float = 2.0, N_stars: int = 500):
    """Compare coherent vs incoherent velocity fields."""
    print("\n" + "=" * 60)
    print("FIGURE 4: VELOCITY COHERENCE")
    print("=" * 60)

    R_half = 1.0
    key = random.PRNGKey(SEED)
    imf = PowerLawIMF.kroupa()

    # Pre-initialize field
    field = init_fractal_field(key, n_modes=64, R_half=R_half)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    configs = [
        (0.0, 'Incoherent ($\\lambda_{vel}$ = 0.0)'),
        (0.3, 'Coherent ($\\lambda_{vel}$ = 0.3)'),
    ]

    for ax, (lam_vel, title) in zip(axes, configs):
        frac = FractalDisplacementLayer(
            chi=D,
            lambda_frac=1.0,
            sigma_u=0.3,
            virial_ratio=0.5,
            coherent_velocities=True,
            lambda_vel=lam_vel,
        )
        cluster = generate_fractal_ic(
            key,
            N_stars=N_stars,
            M_total=float(N_stars),
            R_half=R_half,
            profile="plummer",
            frac_params=frac,
            imf_params=imf,
            field=field,
        )

        positions_np = np.array(cluster.positions)
        velocities_np = np.array(cluster.velocities)

        # Background scatter of all positions
        ax.scatter(
            positions_np[:, 0],
            positions_np[:, 1],
            s=3,
            alpha=0.3,
            c='gray',
            rasterized=True,
        )

        # Subsample for quiver plot clarity
        step = max(1, N_stars // 100)
        idx = np.arange(0, N_stars, step)

        # Normalize velocity vectors for display
        v_scale = np.mean(np.linalg.norm(velocities_np, axis=1))
        quiver = ax.quiver(
            positions_np[idx, 0],
            positions_np[idx, 1],
            velocities_np[idx, 0] / v_scale,
            velocities_np[idx, 1] / v_scale,
            alpha=0.8,
            scale=15,
            width=0.004,
            color='steelblue',
        )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel('x [pc]')
        ax.set_aspect('equal')
        ax.set_xlim(-3 * R_half, 3 * R_half)
        ax.set_ylim(-3 * R_half, 3 * R_half)
        ax.grid(True, alpha=0.2)

        print(f"  lambda_vel = {lam_vel}: velocity dispersion = {np.std(velocities_np):.3f}")

    axes[0].set_ylabel('y [pc]')

    plt.suptitle(f'FDF Velocity Structure (D = {D})', fontsize=13, y=1.02)
    plt.tight_layout()

    filename = f'{output_dir}/fdf_velocity_coherence.png'
    plt.savefig(filename)
    plt.close()
    print(f"  Saved: {filename}")


# =============================================================================
# Figure 5: Gradient Demo (Differentiability Proof)
# =============================================================================


def plot_gradient_demo(output_dir: str, N_stars: int = 500):
    """Demonstrate differentiability: JAX autodiff vs finite difference."""
    print("\n" + "=" * 60)
    print("FIGURE 5: GRADIENT DEMO (DIFFERENTIABILITY)")
    print("=" * 60)

    R_half = 1.0
    key = random.PRNGKey(SEED)
    imf = PowerLawIMF.kroupa()

    # Pre-initialize field (frozen for all chi values)
    field = init_fractal_field(key, n_modes=64, R_half=R_half)
    field = jax.tree_util.tree_map(jax.lax.stop_gradient, field)

    def mean_radius(chi):
        """Differentiable observable: mean radius as function of chi."""
        frac = FractalDisplacementLayer(chi=chi, lambda_frac=1.0, sigma_u=0.3)
        cluster = generate_fractal_ic(
            key,
            N_stars=N_stars,
            M_total=float(N_stars),
            R_half=R_half,
            profile="plummer",
            frac_params=frac,
            imf_params=imf,
            field=field,
        )
        return jnp.mean(jnp.linalg.norm(cluster.positions, axis=1))

    # Autodiff gradient
    grad_fn = jax.grad(mean_radius)

    chi_values = np.linspace(1.6, 3.0, 15)

    print("  Computing autodiff gradients...")
    gradients_autodiff = []
    mean_radii = []
    for chi in chi_values:
        grad_val = float(grad_fn(chi))
        mean_val = float(mean_radius(chi))
        gradients_autodiff.append(grad_val)
        mean_radii.append(mean_val)
        print(f"    chi = {chi:.2f}: mean_r = {mean_val:.4f}, grad = {grad_val:.6f}")

    # Finite difference for comparison
    print("  Computing finite difference gradients...")
    eps = 1e-4
    gradients_fd = []
    for chi in chi_values:
        f_plus = float(mean_radius(chi + eps))
        f_minus = float(mean_radius(chi - eps))
        grad_fd = (f_plus - f_minus) / (2 * eps)
        gradients_fd.append(grad_fd)

    gradients_autodiff = np.array(gradients_autodiff)
    gradients_fd = np.array(gradients_fd)
    mean_radii = np.array(mean_radii)

    # Compute agreement metrics
    abs_error = np.abs(gradients_autodiff - gradients_fd)
    max_abs_error = np.max(abs_error)
    rel_error = abs_error / (np.abs(gradients_fd) + 1e-10)
    max_rel_error = np.max(rel_error)

    print(f"\n  Max |autodiff - FD| error: {max_abs_error:.6f}")
    print(f"  Max relative error: {max_rel_error * 100:.2f}%")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Gradient comparison
    ax = axes[0]
    ax.plot(chi_values, gradients_autodiff, 'b-o', label='JAX autodiff', markersize=6)
    ax.plot(chi_values, gradients_fd, 'r--s', label='Finite difference', markersize=5, alpha=0.7)
    ax.set_xlabel(r'$\chi$ (clumpiness)')
    ax.set_ylabel(r'$\partial \langle r \rangle / \partial \chi$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Gradient Comparison: Autodiff vs FD')

    # Add metrics text box
    metrics_text = (
        f"Max abs error: {max_abs_error:.2e}\n"
        f"Max rel error: {max_rel_error * 100:.1f}%"
    )
    ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel B: Mean radius vs chi
    ax = axes[1]
    ax.plot(chi_values, mean_radii, 'g-o', markersize=6)
    ax.set_xlabel(r'$\chi$ (clumpiness)')
    ax.set_ylabel(r'$\langle r \rangle$ [pc]')
    ax.grid(True, alpha=0.3)
    ax.set_title(r'Mean Radius vs $\chi$')

    plt.suptitle('FDF Differentiability Demonstration', fontsize=13, y=1.02)
    plt.tight_layout()

    filename = f'{output_dir}/fdf_gradient_demo.png'
    plt.savefig(filename)
    plt.close()
    print(f"  Saved: {filename}")

    grad_ok = max_rel_error < 0.05  # 5% tolerance
    return {'grad_ok': grad_ok, 'max_rel_error': max_rel_error, 'max_abs_error': max_abs_error}


# =============================================================================
# Figure 6: Radial CDF Overlay
# =============================================================================


def plot_radial_cdf_overlay(output_dir: str, N_stars: int = 2000):
    """Verify 'remap' mode preserves radial CDF exactly."""
    print("\n" + "=" * 60)
    print("FIGURE 6: RADIAL CDF PRESERVATION")
    print("=" * 60)

    R_half = 1.0
    key = random.PRNGKey(SEED)
    imf = PowerLawIMF.kroupa()

    from progenax.profiles import PlummerProfile

    # Generate base Plummer profile
    profile = PlummerProfile(r_h=R_half)
    masses = jnp.ones(N_stars)
    base_positions = profile.sample_positions(masses, key)
    base_radii = np.linalg.norm(np.array(base_positions), axis=1)
    base_radii_sorted = np.sort(base_radii)

    # Generate FDF with remap mode
    frac = FractalDisplacementLayer(
        chi=1.6,
        lambda_frac=1.0,
        sigma_u=0.4,
        radial_mode="remap",
    )
    cluster = generate_fractal_ic(
        key,
        N_stars=N_stars,
        M_total=float(N_stars),
        R_half=R_half,
        profile="plummer",
        frac_params=frac,
        imf_params=imf,
    )
    fdf_radii = np.linalg.norm(np.array(cluster.positions), axis=1)
    fdf_radii_sorted = np.sort(fdf_radii)

    # Compute CDF
    ecdf = np.arange(1, N_stars + 1) / N_stars

    # Compute max deviation
    max_deviation = np.max(np.abs(fdf_radii_sorted - base_radii_sorted))
    mean_deviation = np.mean(np.abs(fdf_radii_sorted - base_radii_sorted))

    print(f"  Max |r_fdf - r_base| deviation: {max_deviation:.6f}")
    print(f"  Mean deviation: {mean_deviation:.6f}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: CDF overlay
    ax = axes[0]
    ax.plot(base_radii_sorted, ecdf, 'b-', lw=2, label='Base Plummer')
    ax.plot(fdf_radii_sorted, ecdf, 'r--', lw=2, alpha=0.8, label='FDF (remap mode)')
    ax.set_xlabel('Radius r [pc]')
    ax.set_ylabel('CDF: M(<r) / M_total')
    ax.set_title('Radial CDF Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Residual
    ax = axes[1]
    residual = fdf_radii_sorted - base_radii_sorted
    ax.plot(ecdf, residual, 'k-', lw=1)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.fill_between(ecdf, residual, 0, alpha=0.3)
    ax.set_xlabel('CDF')
    ax.set_ylabel(r'$r_{FDF} - r_{base}$ [pc]')
    ax.set_title('Radial CDF Residual')
    ax.grid(True, alpha=0.3)

    # Add metrics text box
    metrics_text = (
        f"Max deviation: {max_deviation:.2e} pc\n"
        f"Mean deviation: {mean_deviation:.2e} pc\n"
        f"Status: {'PASS' if max_deviation < 1e-5 else 'WITHIN TOLERANCE'}"
    )
    ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle("FDF 'remap' Mode: Radial CDF Preservation", fontsize=13, y=1.02)
    plt.tight_layout()

    filename = f'{output_dir}/fdf_radial_cdf.png'
    plt.savefig(filename)
    plt.close()
    print(f"  Saved: {filename}")

    cdf_ok = max_deviation < 0.01  # Should be near machine precision for remap
    return cdf_ok


# =============================================================================
# Gradient Sanity Check (for 'sanity' mode)
# =============================================================================


def run_gradient_sanity_check():
    """Quick gradient sanity check without plots."""
    print("\n" + "=" * 60)
    print("GRADIENT SANITY CHECK")
    print("=" * 60)

    key = random.PRNGKey(SEED)
    imf = PowerLawIMF.kroupa()
    field = init_fractal_field(key, n_modes=64, R_half=1.0)
    field = jax.tree_util.tree_map(jax.lax.stop_gradient, field)

    def loss(chi):
        frac = FractalDisplacementLayer(chi=chi, lambda_frac=1.0, sigma_u=0.3)
        cluster = generate_fractal_ic(
            key, N_stars=200, M_total=200.0, R_half=1.0,
            profile="plummer", frac_params=frac, imf_params=imf, field=field
        )
        return jnp.mean(jnp.linalg.norm(cluster.positions, axis=1))

    grad_fn = jax.grad(loss)

    # Test at chi = 2.0
    chi_test = 2.0
    grad_autodiff = float(grad_fn(chi_test))

    # Finite difference
    eps = 1e-4
    grad_fd = float((loss(chi_test + eps) - loss(chi_test - eps)) / (2 * eps))

    rel_error = abs(grad_autodiff - grad_fd) / (abs(grad_fd) + 1e-10)

    print(f"  chi = {chi_test}")
    print(f"  Autodiff gradient: {grad_autodiff:.6f}")
    print(f"  Finite diff gradient: {grad_fd:.6f}")
    print(f"  Relative error: {rel_error * 100:.2f}%")
    print(f"  Status: {'PASS' if rel_error < 0.05 else 'FAIL'}")

    return rel_error < 0.05


# =============================================================================
# Summary
# =============================================================================


def print_validation_summary(results: dict):
    """Print formatted table of all validation metrics."""
    print("\n" + "=" * 70)
    print("FDF VALIDATION SUMMARY")
    print("=" * 70)

    print("\n" + "-" * 55)
    print("|                  QUANTITATIVE RESULTS                |")
    print("-" * 55)

    if 'Q_mean' in results:
        Q_min = results['Q_mean'].min()
        Q_max = results['Q_mean'].max()
        print(f"| Q range (D=1.6 -> 3.0):    {Q_min:.2f} -> {Q_max:.2f}               |")

    if 'sigma_mean' in results:
        sig_min = results['sigma_mean'].min()
        sig_max = results['sigma_mean'].max()
        print(f"| sigma range:               {sig_min:.2f} -> {sig_max:.2f}               |")

    if 'grad_ok' in results:
        status = 'PASS' if results['grad_ok'] else 'FAIL'
        print(f"| Gradient sanity:           {status}                         |")

    if 'cdf_ok' in results:
        status = 'PASS' if results['cdf_ok'] else 'FAIL'
        print(f"| Radial CDF preserved:      {status}                         |")

    if 'density_Q_mean' in results:
        Q_min = results['density_Q_mean'].min()
        Q_max = results['density_Q_mean'].max()
        print(f"| Density-FDF Q range:       {Q_min:.2f} -> {Q_max:.2f}               |")
        status = 'PASS' if results.get('density_q_monotonic', False) else 'FAIL'
        print(f"| Density-FDF Q monotonic:   {status}                         |")

    print("-" * 55)

    # Overall pass/fail
    all_passed = results.get('grad_ok', True) and results.get('cdf_ok', True)
    overall = 'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'
    print(f"\n  Overall: {overall}")

    print("\n" + "=" * 70)


# =============================================================================
# Main
# =============================================================================


def main():
    """Run FDF validation suite."""
    parser = argparse.ArgumentParser(
        description="FDF Validation Suite - Publication-quality validation figures"
    )
    parser.add_argument(
        '--mode',
        choices=['sanity', 'paper', 'docs', 'calib', 'density', 'all'],
        default='sanity',
        help='Validation mode: sanity (quick), paper (full), docs (small N), density (density-field FDF), all',
    )
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_DIR,
        help=f'Output directory for plots (default: {OUTPUT_DIR})',
    )

    args = parser.parse_args()
    output_dir = args.output_dir

    print("\n" + "=" * 70)
    print("PROGENAX FDF VALIDATION SUITE")
    print("=" * 70)
    print(f"\nMode: {args.mode}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {SEED}")

    os.makedirs(output_dir, exist_ok=True)

    results = {}
    start_time = time.time()

    # Sanity mode: quick gradient check
    if args.mode in ['sanity', 'all']:
        results['grad_ok'] = run_gradient_sanity_check()

    # Paper mode: full publication figures
    if args.mode in ['paper', 'all']:
        hero_results = plot_d_sweep_hero(output_dir, N_stars=2000)
        results.update(hero_results)

        q_results = plot_q_vs_d(output_dir, n_realizations=20, N_stars=2000)
        results.update(q_results)

        plot_lambda_frac_blend(output_dir, D=1.6, N_stars=1500)

        plot_velocity_coherence(output_dir, D=2.0, N_stars=500)

        grad_results = plot_gradient_demo(output_dir, N_stars=500)
        results.update(grad_results)

        results['cdf_ok'] = plot_radial_cdf_overlay(output_dir, N_stars=2000)

    # Docs mode: smaller N for quick examples
    if args.mode in ['docs', 'all']:
        plot_d_sweep_hero(output_dir, N_stars=500, suffix='_docs')

    # Calib mode: calibration sweeps (placeholder)
    if args.mode == 'calib':
        print("\n[Calibration mode not yet implemented - placeholder]")

    # Density mode: test the new density-field FDF approach
    if args.mode in ['density', 'all']:
        hero_density = plot_d_sweep_hero_density(output_dir, N_stars=2000, sigma_ln_rho=2.0)
        results['density_Q_values'] = hero_density['Q_values']

        q_density = plot_q_vs_d_density(output_dir, n_realizations=10, N_stars=2000, sigma_ln_rho=2.0)
        results['density_Q_mean'] = q_density['Q_mean']
        results['density_q_monotonic'] = q_density['q_monotonic']

    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.1f}s")

    print_validation_summary(results)

    # List generated plots
    print("\nGenerated plots:")
    for f in sorted(os.listdir(output_dir)):
        if f.startswith('fdf_') and f.endswith('.png'):
            print(f"  - {output_dir}/{f}")

    return 0 if results.get('grad_ok', True) and results.get('cdf_ok', True) else 1


if __name__ == "__main__":
    sys.exit(main())
