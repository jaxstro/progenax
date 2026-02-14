#!/usr/bin/env python
"""E2: Resolution Convergence — LEAN EXPERIMENT.

Tests 64³, 128³, 256³ on hero environments to justify default grid size.

Output: e2_resolution.png
"""

from __future__ import annotations

import time
import tracemalloc
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np

from progenax.gravoturb import bm19_model as bm19
from progenax.gravoturb import gaussian_to_bm19, build_bm19_cdf_table
from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19

from .helpers import (
    setup_publication_style,
    save_plot,
    COLORS,
)


class ResolutionResult(NamedTuple):
    """Result for one (environment, grid_size) combination."""
    env_name: str
    mach: float
    alpha: float
    grid_size: int
    f_dense: float
    f_tail_values: list[float]
    f_tail_mean: float
    f_tail_std: float
    delta_f_mean: float
    delta_f_std: float
    runtime_sec: float
    memory_mb: float


def compute_f_tail_with_memory(
    key: jax.Array,
    mach: float,
    alpha: float,
    b: float,
    kappa: float,
    grid_size: int,
) -> tuple[float, float, float, float]:
    """Compute f_tail with runtime and memory tracking.

    Returns (f_tail, f_dense, runtime_sec, memory_mb).
    """
    # Start memory tracking
    tracemalloc.start()
    start = time.perf_counter()

    # BM19 theory
    sigma_s_sq = float(bm19.sigma_s_squared(mach, b))
    s_t = float(bm19.transition_density(sigma_s_sq, alpha))
    f_dense = float(bm19.f_dense_bm19_full(sigma_s_sq, s_t, alpha))

    # Generate field via CDF remap
    s_grid, F_grid = build_bm19_cdf_table(sigma_s_sq, s_t, alpha)
    g = random.normal(key, (grid_size, grid_size, grid_size))
    s_field = gaussian_to_bm19(g, sigma_s_sq, s_t, alpha, s_grid, F_grid)
    rho_grid = jnp.exp(s_field)

    # Compute tail PMF
    pmf_result = compute_tail_pmfs_bm19(rho_grid, s_t, kappa)
    f_tail = float(pmf_result.f_tail_actual)

    runtime = time.perf_counter() - start

    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak / (1024 * 1024)

    return f_tail, f_dense, runtime, memory_mb


def run_validation(
    hero_envs: dict[str, tuple[float, float]] | None = None,
    grid_sizes: list[int] = [64, 128, 256],
    kappa: float = 10.0,
    b: float = 0.4,
    n_realizations: int = 10,
    verbose: bool = True,
):
    """Test resolution convergence on hero environments.

    Parameters
    ----------
    hero_envs : dict
        Environment name -> (Mach, alpha) tuples
    grid_sizes : list
        Grid resolutions to test
    kappa : float
        Sigmoid sharpness (from E1)
    b : float
        Driving parameter
    n_realizations : int
        Realizations per combination
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Resolution convergence results
    """
    if hero_envs is None:
        hero_envs = {
            "GMC (M=10, α=2.0)": (10.0, 2.0),
            "Challenging (M=20, α=2.5)": (20.0, 2.5),
        }

    if verbose:
        print("=" * 70)
        print("E2: RESOLUTION CONVERGENCE")
        print("=" * 70)
        print(f"\nHero environments: {list(hero_envs.keys())}")
        print(f"Grid sizes: {grid_sizes}")
        print(f"κ = {kappa}, b = {b}, realizations = {n_realizations}")

    all_results: list[ResolutionResult] = []

    if verbose:
        print(f"\n{'Env':<25} | {'Grid':>8} | {'f_dense':>8} | {'f_tail':>8} | "
              f"{'δf%':>8} | {'σ(δf)%':>8} | {'t(s)':>6} | {'Mem(MB)':>8}")
        print("-" * 100)

    for env_name, (mach, alpha) in hero_envs.items():
        for grid_size in grid_sizes:
            f_tail_values = []
            runtimes = []
            memories = []
            f_dense = None

            for i in range(n_realizations):
                key = random.PRNGKey(int(mach * 1000 + alpha * 100 + grid_size + i))
                f_tail, f_dense_i, runtime, memory = compute_f_tail_with_memory(
                    key, mach, alpha, b, kappa, grid_size
                )
                f_tail_values.append(f_tail)
                runtimes.append(runtime)
                memories.append(memory)
                if f_dense is None:
                    f_dense = f_dense_i

            # Statistics
            f_tail_mean = np.mean(f_tail_values)
            f_tail_std = np.std(f_tail_values)
            delta_f_values = 100 * (np.array(f_tail_values) - f_dense) / f_dense
            delta_f_mean = np.mean(delta_f_values)
            delta_f_std = np.std(delta_f_values)
            runtime_mean = np.mean(runtimes)
            memory_mean = np.mean(memories)

            result = ResolutionResult(
                env_name=env_name,
                mach=mach,
                alpha=alpha,
                grid_size=grid_size,
                f_dense=f_dense,
                f_tail_values=f_tail_values,
                f_tail_mean=f_tail_mean,
                f_tail_std=f_tail_std,
                delta_f_mean=delta_f_mean,
                delta_f_std=delta_f_std,
                runtime_sec=runtime_mean,
                memory_mb=memory_mean,
            )
            all_results.append(result)

            if verbose:
                print(f"{env_name:<25} | {grid_size:>8} | {f_dense:>8.4f} | "
                      f"{f_tail_mean:>8.4f} | {delta_f_mean:>+8.1f} | {delta_f_std:>8.1f} | "
                      f"{runtime_mean:>6.2f} | {memory_mean:>8.1f}")

    return {
        "results": all_results,
        "params": {
            "hero_envs": hero_envs,
            "grid_sizes": grid_sizes,
            "kappa": kappa,
            "b": b,
            "n_realizations": n_realizations,
        }
    }


def make_plot(results: dict, show: bool = False) -> str:
    """Generate resolution convergence visualization.

    2-panel: δf vs grid size (by env), performance table.

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

    all_results = results["results"]
    grid_sizes = results["params"]["grid_sizes"]
    hero_envs = results["params"]["hero_envs"]

    env_colors = {"GMC (M=10, α=2.0)": "C0", "Challenging (M=20, α=2.5)": "C1"}

    # Panel 1: δf vs grid size
    ax1 = axes[0]

    for env_name in hero_envs.keys():
        delta_f_by_grid = []
        delta_f_std_by_grid = []
        for grid_size in grid_sizes:
            env_results = [r for r in all_results if r.env_name == env_name and r.grid_size == grid_size]
            if env_results:
                delta_f_by_grid.append(env_results[0].delta_f_mean)
                delta_f_std_by_grid.append(env_results[0].delta_f_std)

        color = env_colors.get(env_name, "C2")
        ax1.errorbar(
            grid_sizes, delta_f_by_grid, yerr=delta_f_std_by_grid,
            marker="o", color=color, label=env_name,
            capsize=5, linewidth=2, markersize=10
        )

    ax1.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax1.axhspan(-5, 5, alpha=0.1, color="green", label="$\\pm$5% target")
    ax1.set_xlabel("Grid Size (N)", fontsize=12)
    ax1.set_ylabel("Mean $\\delta f$ [%]", fontsize=12)
    ax1.set_title("BM19 Consistency vs Resolution", fontsize=14)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(grid_sizes)
    ax1.set_xticklabels([f"{g}$^3$" for g in grid_sizes])
    ax1.grid(True, alpha=0.3)

    # Panel 2: Performance table
    ax2 = axes[1]
    ax2.axis("off")

    # Create table data
    table_data = []
    headers = ["Environment", "Grid", "δf [%]", "σ(δf) [%]", "Time [s]", "Memory [MB]"]

    for r in all_results:
        row = [
            r.env_name[:20],
            f"{r.grid_size}³",
            f"{r.delta_f_mean:+.1f}",
            f"{r.delta_f_std:.1f}",
            f"{r.runtime_sec:.2f}",
            f"{r.memory_mb:.0f}",
        ]
        table_data.append(row)

    table = ax2.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.1, 0.12, 0.12, 0.12, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    ax2.set_title("Performance Summary", fontsize=14, pad=20)

    plt.suptitle(
        f"E2: Resolution Convergence (κ={results['params']['kappa']}, "
        f"{results['params']['n_realizations']} realizations)",
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "e2_resolution")
    plt.close(fig)

    return path


def print_summary(results: dict):
    """Print summary with rule-of-thumb."""
    all_results = results["results"]
    grid_sizes = results["params"]["grid_sizes"]

    print("\n" + "=" * 70)
    print("RESOLUTION CONVERGENCE SUMMARY")
    print("=" * 70)

    # Summary by grid size
    print(f"\n{'Grid':>10} | {'Mean |δf|%':>12} | {'Mean σ(δf)%':>12} | {'Runtime':>10} | {'Memory':>10}")
    print("-" * 60)

    for grid_size in grid_sizes:
        grid_results = [r for r in all_results if r.grid_size == grid_size]
        mean_abs_delta = np.mean([abs(r.delta_f_mean) for r in grid_results])
        mean_std = np.mean([r.delta_f_std for r in grid_results])
        mean_runtime = np.mean([r.runtime_sec for r in grid_results])
        mean_memory = np.mean([r.memory_mb for r in grid_results])

        print(f"{grid_size:>10}³ | {mean_abs_delta:>12.1f} | {mean_std:>12.1f} | "
              f"{mean_runtime:>10.2f}s | {mean_memory:>10.0f} MB")

    print("\n" + "-" * 60)
    print("RULE OF THUMB:")
    print("  - 64³:  Fast CI tests, |δf| ~10-30%")
    print("  - 128³: Production default, |δf| < 10%")
    print("  - 256³: High precision, |δf| < 5%")


def main():
    """Run full E2 validation."""
    results = run_validation(verbose=True)
    make_plot(results)
    print_summary(results)

    print("\n" + "=" * 70)
    print("E2 VALIDATION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
