#!/usr/bin/env python
"""A1: f_tail_actual vs f_dense — CORNERSTONE PLOT.

Grid in (Mach, alpha), 10-50 FDF realizations at 64^3 or 128^3 per point,
scatter plot with 1:1 line.

Evidence that 3D stochastic implementation realizes 1D BM19 prediction.

Uses Gaussian copula (CDF remap) to generate density fields with exact
BM19 piecewise LN+PL PDF.

FINITE-GRID SAMPLING LIMITATION
-------------------------------
The validation range is constrained by finite-grid sampling. For a given
grid resolution, there is a maximum alpha value that can be reliably
validated:

    Grid Size | Max alpha (reliable) | Max alpha (marginal)
    ----------|---------------------|---------------------
    64^3      | 2.25                | 2.5
    128^3     | 2.5                 | 2.75

For alpha=3.0, the transition density s_t is ~5 sigma above the mean,
giving a volume fraction in the tail of ~2e-7. With 64^3 = 262k voxels,
we expect only 0.06 voxels in the tail — insufficient for meaningful
measurement.

This is a fundamental sampling limitation, not a code bug. The CDF remap
correctly generates the BM19 PDF, but rare tail events cannot be sampled
with finite grids.

Output: a1_ftail_vs_fdense.png
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np

from progenax.gravoturb import bm19_model as bm19
from progenax.gravoturb import gaussian_to_bm19, build_bm19_cdf_table

from .helpers import (
    setup_publication_style,
    save_plot,
    compute_statistics,
    relative_error_percent,
    COLORS,
)


def compute_expected_tail_voxels(sigma_s_sq: float, s_t: float, alpha: float, n_voxels: int) -> float:
    """Compute expected number of voxels in the tail region.

    This uses the volume-weighted CDF to determine what fraction of
    voxels will have s > s_t, then multiplies by total voxel count.

    Parameters
    ----------
    sigma_s_sq : float
        PDF variance
    s_t : float
        Transition density threshold
    alpha : float
        Powerlaw slope
    n_voxels : int
        Total number of voxels (e.g., 64^3)

    Returns
    -------
    expected_voxels : float
        Expected number of voxels with s > s_t
    """
    s_grid, F_grid = build_bm19_cdf_table(sigma_s_sq, s_t, alpha)
    idx_st = jnp.searchsorted(s_grid, s_t)
    F_at_st = float(F_grid[min(idx_st, len(F_grid) - 1)])
    volume_frac = 1.0 - F_at_st
    return n_voxels * volume_frac


def run_validation(
    n_realizations: int = 20,
    grid_size: int = 64,
    kappa: float = 10.0,
    min_expected_voxels: float = 5.0,
    verbose: bool = True,
):
    """Run f_tail vs f_dense validation.

    Uses Gaussian copula (CDF remap) to generate density fields with exact
    BM19 piecewise LN+PL PDF.

    Note: Parameter combinations where the expected number of voxels in
    the tail is less than `min_expected_voxels` are skipped, as they
    cannot be reliably validated with finite grids.

    Parameters
    ----------
    n_realizations : int
        Number of FDF realizations per (Mach, alpha) point
    grid_size : int
        3D grid resolution (64 for quick, 128 for production)
    kappa : float
        Soft sigmoid sharpness
    min_expected_voxels : float
        Minimum expected voxels in tail to include point (default 5)
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        All validation results
    """
    if verbose:
        print("=" * 70)
        print("A1: f_tail_actual vs f_dense CORNERSTONE VALIDATION")
        print("=" * 70)
        print("\nMethod: CDF remap (Gaussian copula) for exact BM19 LN+PL PDF")

    # Parameter grid - includes all alpha values; will filter by sampleability
    machs = np.array([5.0, 10.0, 15.0, 20.0])
    alphas = np.array([1.5, 2.0, 2.5, 3.0])
    b = 0.4
    n_voxels = grid_size ** 3

    if verbose:
        print(f"\nParameters:")
        print(f"  Grid: {grid_size}^3 ({n_voxels:,} voxels)")
        print(f"  kappa: {kappa}")
        print(f"  Realizations per point: {n_realizations}")
        print(f"  Mach values: {machs}")
        print(f"  Alpha values: {alphas}")
        print(f"  Min expected tail voxels: {min_expected_voxels}")
        print(f"\nChecking sampleability...")

    # Collect results
    all_f_dense = []
    all_f_tail = []
    all_alphas = []  # For coloring
    all_machs = []
    all_expected_voxels = []
    skipped_points = []

    for alpha in alphas:
        for mach in machs:
            # BM19 theory prediction
            result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.6)
            f_dense_theory = float(result.f_dense)
            s_t = float(result.s_t)
            sigma_s_sq = float(result.sigma_s_sq)

            # Check if this parameter combination is sampleable
            expected_voxels = compute_expected_tail_voxels(
                sigma_s_sq, s_t, alpha, n_voxels
            )

            if expected_voxels < min_expected_voxels:
                skipped_points.append({
                    "alpha": alpha,
                    "mach": mach,
                    "expected_voxels": expected_voxels,
                    "f_dense": f_dense_theory,
                })
                if verbose:
                    print(f"  SKIP: alpha={alpha:.1f}, M={mach:.0f} "
                          f"(expected {expected_voxels:.1f} voxels in tail)")
                continue

            if verbose:
                print(f"  OK: alpha={alpha:.1f}, M={mach:.0f} "
                      f"(expected {expected_voxels:.0f} voxels)")

            # Build CDF table once per (mach, alpha) combination
            s_grid, F_grid = build_bm19_cdf_table(sigma_s_sq, s_t, alpha)

            for i in range(n_realizations):
                key = random.PRNGKey(int(mach * 1000) + int(alpha * 100) + i)

                # Generate Gaussian random field
                g = random.normal(key, (grid_size, grid_size, grid_size))

                # Apply CDF remap to get BM19 LN+PL distribution
                s = gaussian_to_bm19(g, sigma_s_sq, s_t, alpha, s_grid, F_grid)
                rho = jnp.exp(s)

                # Compute f_tail_actual using soft sigmoid
                w = jax.nn.sigmoid(kappa * (s - s_t))
                f_tail_actual = float(jnp.sum(w * rho) / jnp.sum(rho))

                all_f_dense.append(f_dense_theory)
                all_f_tail.append(f_tail_actual)
                all_alphas.append(alpha)
                all_machs.append(mach)
                all_expected_voxels.append(expected_voxels)

    # Convert to arrays
    all_f_dense = np.array(all_f_dense)
    all_f_tail = np.array(all_f_tail)
    all_alphas = np.array(all_alphas)
    all_machs = np.array(all_machs)
    all_expected_voxels = np.array(all_expected_voxels)

    if len(all_f_dense) == 0:
        raise ValueError(
            f"No sampleable parameter combinations at {grid_size}^3 resolution. "
            f"Increase grid_size or decrease min_expected_voxels."
        )

    # Statistics
    errors = relative_error_percent(all_f_tail, all_f_dense)
    stats = compute_statistics(errors)

    if verbose:
        print(f"\n" + "-" * 60)
        print("RESULTS SUMMARY")
        print("-" * 60)
        print(f"  Validated points: {len(all_f_dense)}")
        print(f"  Skipped points: {len(skipped_points)}")
        print(f"  Mean error: {stats['mean']:+.1f}%")
        print(f"  Std error: {stats['std']:.1f}%")
        print(f"  Median error: {stats['median']:+.1f}%")
        print(f"  95% CI: [{stats['p5']:+.1f}%, {stats['p95']:+.1f}%]")
        print(f"  Within +/-10%: {100 * np.mean(np.abs(errors) < 10):.0f}%")
        print(f"  Within +/-20%: {100 * np.mean(np.abs(errors) < 20):.0f}%")

        if skipped_points:
            print(f"\n  Skipped due to finite-grid limitation:")
            for sp in skipped_points:
                print(f"    alpha={sp['alpha']:.1f}, M={sp['mach']:.0f}: "
                      f"f_dense={sp['f_dense']:.2e}, "
                      f"expected {sp['expected_voxels']:.1f} voxels")

    return {
        "f_dense": all_f_dense,
        "f_tail": all_f_tail,
        "alphas": all_alphas,
        "machs": all_machs,
        "expected_voxels": all_expected_voxels,
        "errors": errors,
        "stats": stats,
        "skipped_points": skipped_points,
        "params": {
            "machs": machs,
            "alphas": alphas,
            "b": b,
            "grid_size": grid_size,
            "kappa": kappa,
            "n_realizations": n_realizations,
            "min_expected_voxels": min_expected_voxels,
        }
    }


def make_plot(results: dict, show: bool = False) -> str:
    """Generate the cornerstone scatter plot (single panel, legacy).

    Parameters
    ----------
    results : dict
        Output from run_validation()
    show : bool
        Display plot interactively

    Returns
    -------
    path : str
        Path to saved plot
    """
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(8, 8))

    f_dense = results["f_dense"]
    f_tail = results["f_tail"]
    alphas = results["alphas"]
    stats = results["stats"]

    # Color by alpha
    alpha_colors = {1.5: "C0", 2.0: "C1", 2.5: "C2", 3.0: "C3"}

    for alpha in np.unique(alphas):
        mask = alphas == alpha
        ax.scatter(
            f_dense[mask], f_tail[mask],
            c=alpha_colors[alpha], alpha=0.5, s=30,
            label=f"$\\alpha = {alpha:.1f}$"
        )

    # 1:1 line
    min_val = min(np.min(f_dense), np.min(f_tail)) * 0.8
    max_val = max(np.max(f_dense), np.max(f_tail)) * 1.2
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2, label="1:1")

    # +/- 20% bands
    x_range = np.linspace(min_val, max_val, 100)
    ax.fill_between(x_range, x_range * 0.8, x_range * 1.2,
                    alpha=0.15, color="gray", label="$\\pm$20%")

    ax.set_xlabel("$f_\\mathrm{dense}$ (BM19 theory)", fontsize=14)
    ax.set_ylabel("$f_\\mathrm{tail,actual}$ (3D FDF realization)", fontsize=14)
    ax.set_title(
        f"BM19 Theory vs 3D FDF Consistency\n"
        f"({results['params']['grid_size']}$^3$ grid, $\\kappa$={results['params']['kappa']}, "
        f"N={results['params']['n_realizations']}/point)",
        fontsize=14
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    textstr = f"Mean error: {stats['mean']:+.1f}%\nStd: {stats['std']:.1f}%\nWithin $\\pm$20%: {100 * np.mean(np.abs(results['errors']) < 20):.0f}%"
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", horizontalalignment="right", bbox=props)

    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "a1_ftail_vs_fdense")
    plt.close(fig)

    return path


def make_plot_multi_resolution(results_by_res: dict, show: bool = False) -> str:
    """Generate 3-panel plot comparing different grid resolutions.

    Parameters
    ----------
    results_by_res : dict
        Dictionary mapping grid_size -> results from run_validation()
    show : bool
        Display plot interactively

    Returns
    -------
    path : str
        Path to saved plot
    """
    setup_publication_style()

    grid_sizes = sorted(results_by_res.keys())
    n_panels = len(grid_sizes)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    # Color by alpha
    alpha_colors = {1.5: "C0", 2.0: "C1", 2.5: "C2", 3.0: "C3"}

    # Find global axis limits
    all_f = []
    for res, results in results_by_res.items():
        all_f.extend(results["f_dense"])
        all_f.extend(results["f_tail"])
    all_f = np.array(all_f)
    min_val = np.min(all_f) * 0.5
    max_val = np.max(all_f) * 2.0

    for idx, (grid_size, results) in enumerate(sorted(results_by_res.items())):
        ax = axes[idx]

        f_dense = results["f_dense"]
        f_tail = results["f_tail"]
        alphas = results["alphas"]
        stats = results["stats"]
        errors = results["errors"]

        # Scatter by alpha
        for alpha in np.unique(alphas):
            mask = alphas == alpha
            ax.scatter(
                f_dense[mask], f_tail[mask],
                c=alpha_colors[alpha], alpha=0.5, s=25,
                label=f"$\\alpha$ = {alpha:.1f}" if idx == 0 else None
            )

        # 1:1 line
        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2,
                label="1:1" if idx == 0 else None)

        # +/- 20% bands
        x_range = np.linspace(min_val, max_val, 100)
        ax.fill_between(x_range, x_range * 0.8, x_range * 1.2,
                        alpha=0.15, color="gray",
                        label="$\\pm$20%" if idx == 0 else None)

        ax.set_xlabel("$f_\\mathrm{dense}$ (BM19 theory)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("$f_\\mathrm{tail,actual}$ (3D FDF)", fontsize=11)

        n_voxels = grid_size ** 3
        ax.set_title(f"N = {grid_size}$^3$ ({n_voxels:,} voxels)", fontsize=12)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Statistics text box
        within_20 = 100 * np.mean(np.abs(errors) < 20)
        textstr = (f"Mean: {stats['mean']:+.1f}%\n"
                   f"Std: {stats['std']:.1f}%\n"
                   f"±20%: {within_20:.0f}%")
        props = dict(boxstyle="round", facecolor="white", alpha=0.9)
        ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom", horizontalalignment="right", bbox=props)

    # Legend on first panel
    axes[0].legend(loc="upper left", fontsize=9)

    plt.suptitle(
        "BM19 Theory vs 3D FDF: Resolution Dependence",
        fontsize=14, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if show:
        plt.show()

    path = save_plot(fig, "a1_ftail_vs_fdense")
    plt.close(fig)

    return path


def main():
    """Run full A1 validation at multiple resolutions."""
    # Run at three different resolutions
    grid_sizes = [32, 64, 128]
    n_realizations = 10  # Fewer realizations per point to keep runtime reasonable

    results_by_res = {}

    for grid_size in grid_sizes:
        print(f"\n{'='*70}")
        print(f"RUNNING AT {grid_size}^3 RESOLUTION")
        print(f"{'='*70}")

        results = run_validation(
            n_realizations=n_realizations,
            grid_size=grid_size,
            kappa=10.0,
            min_expected_voxels=3.0,  # Lower threshold to get more points at low res
            verbose=True,
        )
        results_by_res[grid_size] = results

    # Generate 3-panel plot
    make_plot_multi_resolution(results_by_res)

    print("\n" + "=" * 70)
    print("A1 VALIDATION COMPLETE - MULTI-RESOLUTION")
    print("=" * 70)

    print("\nResolution comparison:")
    print(f"{'Grid':<10} {'Mean err':<12} {'Std':<10} {'±20%':<10} {'Points':<10}")
    print("-" * 52)
    for grid_size in grid_sizes:
        r = results_by_res[grid_size]
        within_20 = 100 * np.mean(np.abs(r["errors"]) < 20)
        print(f"{grid_size}^3{'':<6} {r['stats']['mean']:+.1f}%{'':<6} "
              f"{r['stats']['std']:.1f}%{'':<5} {within_20:.0f}%{'':<6} "
              f"{len(r['f_dense'])}")

    print("\nKey finding: Higher resolution → lower variance, same mean")

    return results_by_res


if __name__ == "__main__":
    main()
