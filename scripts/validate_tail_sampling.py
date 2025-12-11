#!/usr/bin/env python
"""
Tail Sampling Validation Suite - Gravoturbulent dense-tail substructure.

Produces diagnostic figures for:
1. f_sub parameter sweep (hero figure showing clusters at different f_sub)
2. Q vs f_sub ensemble plot with error bars
3. Dense voxel topology visualization (local overdensity distribution)
4. Quantitative results table

Usage:
    python scripts/validate_tail_sampling.py              # Full validation (~5min)
    python scripts/validate_tail_sampling.py --quick      # Quick sanity check (~1min)

Output:
    validation/plots/tail_sampling_*.png

Physics:
    From Cartwright & Whitworth (2004):
    - Q < 0.79: substructured (fractal, multiple clumps)
    - Q > 0.79: centrally concentrated (radial profile)
    - Q ~ 0.79: uniform sphere baseline

    For our gravoturbulent model:
    - f_sub = fraction of stars from LOCAL overdensity peaks
    - f_sub up -> more stars in spatially-correlated clumps -> Q down
    - Expected: Q(0.7) < Q(0.5) < Q(0.3) < Q(0.1) ~ 0.8-0.9
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

from progenax.cluster.fdf_density import (
    FractalDensityLayer,
    TailSubstructureLayer,
    generate_fractal_ic_density,
    init_turbulent_density_field,
    sample_positions_tail,
    _gaussian_blur_3d_fft,
)
from progenax.cluster.fdf_config import (
    default_f_sub_for_cluster_type,
    f_sub_from_D,
)
from progenax.diagnostics import compute_q_parameter
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
    """Estimate local surface density via k-nearest neighbor distance."""
    from scipy.spatial import cKDTree

    xy = positions[:, :2]
    tree = cKDTree(xy)
    distances, _ = tree.query(xy, k=k + 1)
    r_k = distances[:, -1]
    density = k / (np.pi * r_k**2 + 1e-10)
    return density


# =============================================================================
# Figure 1: f_sub Sweep Hero Figure
# =============================================================================


def plot_fsub_sweep_hero(output_dir: str, N_stars: int = 2000):
    """Create 4x2 hero figure showing clusters at different f_sub values.

    Top row: XY projections colored by local density
    Bottom row: Histograms of radial distribution
    """
    print("\n" + "=" * 60)
    print("FIGURE 1: f_sub SWEEP HERO FIGURE")
    print("=" * 60)

    f_sub_values = [0.1, 0.3, 0.5, 0.7]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    imf = PowerLawIMF.kroupa()
    key = random.PRNGKey(SEED)

    Q_values = {}

    for idx, f_sub in enumerate(f_sub_values):
        key_i = random.fold_in(key, idx)

        layer = FractalDensityLayer(
            chi=2.0,
            sigma_ln_rho=2.0,
            base_profile="uniform",
        )
        tail = TailSubstructureLayer(f_sub=f_sub)

        cluster = generate_fractal_ic_density(
            key_i, N_stars=N_stars, M_total=float(N_stars),
            R_half=1.0, imf_params=imf, layer=layer, tail=tail,
        )
        positions = np.array(cluster.positions)
        Q = compute_q_parameter(positions)
        Q_values[f_sub] = Q

        # Top row: XY projection
        ax_top = axes[0, idx]
        density = estimate_local_density(positions, k=10)
        log_density = np.log10(density + 1e-10)

        scatter = ax_top.scatter(
            positions[:, 0], positions[:, 1],
            c=log_density, s=2, cmap='viridis', alpha=0.7, rasterized=True
        )
        ax_top.set_xlim(-3, 3)
        ax_top.set_ylim(-3, 3)
        ax_top.set_aspect('equal')
        ax_top.set_title(f'f_sub = {f_sub:.1f}\nQ = {Q:.2f}', fontsize=12)
        if idx == 0:
            ax_top.set_ylabel('y [pc]')
        ax_top.set_xlabel('x [pc]')

        # Bottom row: Radial histogram
        ax_bot = axes[1, idx]
        r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)
        ax_bot.hist(r, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax_bot.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='R_half')
        ax_bot.set_xlabel('r [pc]')
        if idx == 0:
            ax_bot.set_ylabel('Probability density')
            ax_bot.legend(loc='upper right')
        ax_bot.set_xlim(0, 4)

    plt.suptitle(
        'Gravoturbulent Tail Sampling: f_sub Sweep (LOCAL overdensity ranking)',
        fontsize=14
    )
    plt.tight_layout()

    filename = f'{output_dir}/tail_sampling_fsub_sweep.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")

    print("\n  Q values:")
    for f_sub, Q in Q_values.items():
        print(f"    f_sub={f_sub:.1f}: Q = {Q:.3f}")

    return Q_values


# =============================================================================
# Figure 2: Q vs f_sub Ensemble Plot
# =============================================================================


def plot_q_vs_fsub_ensemble(output_dir: str, n_realizations: int = 20, N_stars: int = 500):
    """Create Q vs f_sub plot with error bars from ensemble.

    Tests the key prediction: f_sub up -> Q down (more substructure).
    """
    print("\n" + "=" * 60)
    print("FIGURE 2: Q vs f_sub ENSEMBLE")
    print("=" * 60)

    f_sub_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    Q_means = []
    Q_stds = []

    imf = PowerLawIMF.kroupa()
    key = random.PRNGKey(SEED)

    for f_sub in f_sub_values:
        Q_values = []
        for i in range(n_realizations):
            key_i = random.fold_in(key, int(f_sub * 100) + i * 1000)

            layer = FractalDensityLayer(
                chi=2.0,
                sigma_ln_rho=2.0,
                base_profile="uniform",
            )
            tail = TailSubstructureLayer(f_sub=f_sub)

            cluster = generate_fractal_ic_density(
                key_i, N_stars=N_stars, M_total=float(N_stars),
                R_half=1.0, imf_params=imf, layer=layer, tail=tail,
            )
            Q = compute_q_parameter(np.array(cluster.positions))
            Q_values.append(Q)

        Q_means.append(np.mean(Q_values))
        Q_stds.append(np.std(Q_values))
        print(f"  f_sub={f_sub:.1f}: Q = {np.mean(Q_values):.3f} +/- {np.std(Q_values):.3f}")

    Q_means = np.array(Q_means)
    Q_stds = np.array(Q_stds)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(f_sub_values, Q_means, yerr=Q_stds, fmt='o-',
                color='steelblue', capsize=4, capthick=2, linewidth=2, markersize=8,
                label=f'FDF tail sampling (n={n_realizations})')

    # Reference lines
    ax.axhline(0.79, color='gray', linestyle='--', linewidth=1.5,
               label='Uniform sphere (Q~0.79)')
    ax.axhspan(0.0, 0.79, alpha=0.1, color='blue', label='Substructured (Q<0.79)')
    ax.axhspan(0.79, 1.5, alpha=0.1, color='orange', label='Concentrated (Q>0.79)')

    ax.set_xlabel('f_sub (dense tail fraction)', fontsize=12)
    ax.set_ylabel('Q parameter', fontsize=12)
    ax.set_title('Q vs f_sub: Higher f_sub -> More Substructure -> Lower Q', fontsize=13)
    ax.legend(loc='upper right')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.5, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f'{output_dir}/tail_sampling_q_vs_fsub.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\n  Saved: {filename}")

    # Check monotonicity
    is_monotonic = all(Q_means[i] >= Q_means[i+1] for i in range(len(Q_means)-1))
    print(f"\n  Monotonicity check: {'PASS' if is_monotonic else 'FAIL'}")

    return {'Q_means': Q_means, 'Q_stds': Q_stds, 'f_sub': f_sub_values, 'monotonic': is_monotonic}


# =============================================================================
# Figure 3: Dense Voxel Topology Visualization
# =============================================================================


def plot_dense_voxel_topology(output_dir: str):
    """Visualize where LOCAL vs GLOBAL overdensity selects voxels.

    Shows that LOCAL overdensity (our fix) distributes dense voxels across
    the volume, while GLOBAL density concentrates them at the center.
    """
    print("\n" + "=" * 60)
    print("FIGURE 3: DENSE VOXEL TOPOLOGY")
    print("=" * 60)

    key = random.PRNGKey(SEED)
    layer = FractalDensityLayer(
        chi=2.0,
        sigma_ln_rho=2.0,
        base_profile="uniform",
    )

    field = init_turbulent_density_field(key, R_half=1.0, layer=layer)
    rho = np.array(field.rho_grid)

    # Compute local overdensity
    rho_smoothed = np.array(_gaussian_blur_3d_fft(field.rho_grid, sigma_cells=5.0))
    rho_local = rho / rho_smoothed

    # Find top 10% by mass for both methods
    mass_flat = rho.ravel() / rho.sum()
    sort_idx_global = np.argsort(-rho.ravel())
    sort_idx_local = np.argsort(-rho_local.ravel())

    cum_mass_global = np.cumsum(mass_flat[sort_idx_global])
    cum_mass_local = np.cumsum(mass_flat[sort_idx_local])

    n_dense_global = np.searchsorted(cum_mass_global, 0.1)
    n_dense_local = np.searchsorted(cum_mass_local, 0.1)

    dense_idx_global = set(sort_idx_global[:n_dense_global])
    dense_idx_local = set(sort_idx_local[:n_dense_local])

    # Create masks
    Nx, Ny, Nz = rho.shape
    x_grid = np.array(field.x_grid)
    y_grid = np.array(field.y_grid)
    z_grid = np.array(field.z_grid)

    # Get 3D positions of dense voxels
    def get_dense_positions(dense_indices, Nx, Ny, Nz):
        positions = []
        for idx in dense_indices:
            i = idx // (Ny * Nz)
            j = (idx % (Ny * Nz)) // Nz
            k = idx % Nz
            positions.append([x_grid[i], y_grid[j], z_grid[k]])
        return np.array(positions)

    pos_global = get_dense_positions(dense_idx_global, Nx, Ny, Nz)
    pos_local = get_dense_positions(dense_idx_local, Nx, Ny, Nz)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Global density ranking
    ax = axes[0]
    ax.scatter(pos_global[:, 0], pos_global[:, 1], s=5, alpha=0.7, c='red')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_xlabel('x [pc]')
    ax.set_ylabel('y [pc]')
    ax.set_title(f'GLOBAL density ranking\n(top 10% mass, {len(pos_global)} voxels)\nCONCENTRATED at center',
                 fontsize=11)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

    # Right: Local overdensity ranking
    ax = axes[1]
    ax.scatter(pos_local[:, 0], pos_local[:, 1], s=5, alpha=0.7, c='blue')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_xlabel('x [pc]')
    ax.set_ylabel('y [pc]')
    ax.set_title(f'LOCAL overdensity ranking (our fix)\n(top 10% mass, {len(pos_local)} voxels)\nDISTRIBUTED across volume',
                 fontsize=11)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

    plt.suptitle('Dense Voxel Selection: Global vs Local Overdensity', fontsize=14)
    plt.tight_layout()

    filename = f'{output_dir}/tail_sampling_voxel_topology.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")

    # Compute spatial spread
    spread_global = np.std(pos_global, axis=0).mean()
    spread_local = np.std(pos_local, axis=0).mean()
    print(f"\n  Spatial spread (std):")
    print(f"    Global ranking: {spread_global:.2f} pc")
    print(f"    Local ranking:  {spread_local:.2f} pc")
    print(f"    Improvement:    {spread_local/spread_global:.1f}x more spread")

    return {'spread_global': spread_global, 'spread_local': spread_local}


# =============================================================================
# Figure 4: Cluster Type Defaults
# =============================================================================


def plot_cluster_type_defaults(output_dir: str, N_stars: int = 1500):
    """Show clusters generated with default f_sub for each cluster type."""
    print("\n" + "=" * 60)
    print("FIGURE 4: CLUSTER TYPE DEFAULTS")
    print("=" * 60)

    cluster_types = ['assoc', 'oc', 'ymc', 'gc']
    cluster_names = ['Association', 'Open Cluster', 'Young Massive', 'Globular']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    imf = PowerLawIMF.kroupa()
    key = random.PRNGKey(SEED + 100)

    for idx, (ctype, name) in enumerate(zip(cluster_types, cluster_names)):
        key_i = random.fold_in(key, idx)
        f_sub = default_f_sub_for_cluster_type(ctype)

        layer = FractalDensityLayer(
            chi=2.0,
            sigma_ln_rho=2.0,
            base_profile="uniform",
        )
        tail = TailSubstructureLayer(f_sub=f_sub)

        cluster = generate_fractal_ic_density(
            key_i, N_stars=N_stars, M_total=float(N_stars),
            R_half=1.0, imf_params=imf, layer=layer, tail=tail,
        )
        positions = np.array(cluster.positions)
        Q = compute_q_parameter(positions)

        ax = axes[idx]
        density = estimate_local_density(positions, k=10)
        log_density = np.log10(density + 1e-10)

        ax.scatter(positions[:, 0], positions[:, 1], c=log_density,
                   s=2, cmap='viridis', alpha=0.7, rasterized=True)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\nf_sub={f_sub:.2f}, Q={Q:.2f}', fontsize=11)
        ax.set_xlabel('x [pc]')
        if idx == 0:
            ax.set_ylabel('y [pc]')

        print(f"  {name}: f_sub={f_sub:.2f}, Q={Q:.3f}")

    plt.suptitle('Cluster Type Defaults: f_sub Phenomenological Values', fontsize=14)
    plt.tight_layout()

    filename = f'{output_dir}/tail_sampling_cluster_types.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\n  Saved: {filename}")


# =============================================================================
# Summary
# =============================================================================


def print_summary(results: dict):
    """Print summary of validation results."""
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if 'monotonic' in results:
        status = 'PASS' if results['monotonic'] else 'FAIL'
        print(f"  Q monotonicity (f_sub up -> Q down): {status}")

    if 'spread_local' in results and 'spread_global' in results:
        ratio = results['spread_local'] / results['spread_global']
        print(f"  Local overdensity spread improvement: {ratio:.1f}x")

    print("\n  Key insight: Using LOCAL overdensity (not global density) for")
    print("  voxel ranking produces true fractal substructure across the volume,")
    print("  not central concentration.")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Tail Sampling Validation Suite"
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick sanity check (fewer realizations)'
    )
    parser.add_argument(
        '--output-dir', default=OUTPUT_DIR,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("TAIL SAMPLING VALIDATION SUITE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    start_time = time.time()
    results = {}

    # Figure 1: Hero sweep
    N_stars = 1000 if args.quick else 2000
    hero_results = plot_fsub_sweep_hero(output_dir, N_stars=N_stars)

    # Figure 2: Q vs f_sub ensemble
    n_realizations = 5 if args.quick else 20
    N_stars_ensemble = 300 if args.quick else 500
    ensemble_results = plot_q_vs_fsub_ensemble(
        output_dir, n_realizations=n_realizations, N_stars=N_stars_ensemble
    )
    results.update(ensemble_results)

    # Figure 3: Voxel topology
    topology_results = plot_dense_voxel_topology(output_dir)
    results.update(topology_results)

    # Figure 4: Cluster type defaults
    N_stars_types = 800 if args.quick else 1500
    plot_cluster_type_defaults(output_dir, N_stars=N_stars_types)

    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.1f}s")

    print_summary(results)

    # List generated plots
    print("\nGenerated plots:")
    for f in sorted(os.listdir(output_dir)):
        if f.startswith('tail_sampling') and f.endswith('.png'):
            print(f"  - {output_dir}/{f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
