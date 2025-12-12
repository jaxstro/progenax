#!/usr/bin/env python
"""E6: Gradient Health Aggregation — Define Safe Inference Box.

Aggregates gradient diagnostics from E1 to:
1. Show gradient magnitude heatmaps over (Mach, α)
2. Identify pathological regions (vanishing/exploding gradients)
3. Define "safe inference box" where gradients are well-behaved

Output: e6_gradients.png
"""

from __future__ import annotations

import time
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


class GradientResult(NamedTuple):
    """Gradient measurement at a (Mach, α) point."""
    mach: float
    alpha: float
    grad_mach: float  # ∂f_tail/∂M
    grad_alpha: float  # ∂f_tail/∂α
    grad_mach_log: float  # ∂log(f_tail)/∂log(M)
    grad_alpha_log: float  # ∂log(f_tail)/∂log(α)
    f_tail: float


def compute_gradients_at_point(
    mach: float,
    alpha: float,
    b: float = 0.4,
    kappa: float = 10.0,
    grid_size: int = 64,
    seed: int = 0,
) -> GradientResult:
    """Compute gradients at a single (Mach, α) point.

    Uses BM19 theory f_dense (differentiable) as proxy for f_tail.
    Returns gradients in both linear and log-log space.
    """
    # Theory f_dense is differentiable (no stochastic sampling)
    def f_dense_of_mach(m):
        sigma_s_sq = bm19.sigma_s_squared(m, b)
        s_t = bm19.transition_density(sigma_s_sq, alpha)
        return bm19.f_dense_bm19_full(sigma_s_sq, s_t, alpha)

    def f_dense_of_alpha(a):
        sigma_s_sq = bm19.sigma_s_squared(mach, b)
        s_t = bm19.transition_density(sigma_s_sq, a)
        return bm19.f_dense_bm19_full(sigma_s_sq, s_t, a)

    # Get f_dense value
    f_dense = float(f_dense_of_mach(mach))

    # Linear gradients via JAX autodiff
    grad_mach = float(jax.grad(f_dense_of_mach)(mach))
    grad_alpha = float(jax.grad(f_dense_of_alpha)(alpha))

    # Log-log gradients (elasticities)
    # ∂log(f)/∂log(M) = (M/f) * ∂f/∂M
    grad_mach_log = (mach / max(f_dense, 1e-10)) * grad_mach
    grad_alpha_log = (alpha / max(f_dense, 1e-10)) * grad_alpha

    return GradientResult(
        mach=mach,
        alpha=alpha,
        grad_mach=grad_mach,
        grad_alpha=grad_alpha,
        grad_mach_log=grad_mach_log,
        grad_alpha_log=grad_alpha_log,
        f_tail=f_dense,  # Using f_dense as proxy
    )


def run_validation(
    machs: list[float] = [5.0, 10.0, 15.0, 20.0, 25.0],
    alphas: list[float] = [1.5, 1.75, 2.0, 2.25, 2.5],
    b: float = 0.4,
    kappa: float = 10.0,
    grid_size: int = 64,
    verbose: bool = True,
):
    """Compute gradient grid over (Mach, α) space.

    Parameters
    ----------
    machs : list[float]
        Mach numbers to test.
    alphas : list[float]
        BM19 powerlaw slopes to test.
    b : float
        Driving parameter.
    kappa : float
        Sigmoid sharpness (use E1 result).
    grid_size : int
        Density field resolution.
    verbose : bool
        Print progress.

    Returns
    -------
    results : dict
        Grid of gradient measurements.
    """
    if verbose:
        print("=" * 70)
        print("E6: GRADIENT HEALTH ANALYSIS")
        print("=" * 70)
        print(f"\nParameters: κ={kappa}, b={b}, grid={grid_size}³")
        print(f"Mach ∈ {machs}")
        print(f"α ∈ {alphas}")

    results_grid = {}

    print(f"\n{'M':>6} | {'α':>6} | {'f_tail':>8} | {'∂f/∂M':>10} | {'∂f/∂α':>10} | {'∂logf/∂logM':>12}")
    print("-" * 75)

    for mach in machs:
        for alpha in alphas:
            result = compute_gradients_at_point(
                mach=mach,
                alpha=alpha,
                b=b,
                kappa=kappa,
                grid_size=grid_size,
            )
            results_grid[(mach, alpha)] = result

            if verbose:
                print(f"{mach:>6.0f} | {alpha:>6.2f} | {result.f_tail:>8.4f} | "
                      f"{result.grad_mach:>10.4f} | {result.grad_alpha:>10.4f} | "
                      f"{result.grad_mach_log:>12.2f}")

    print("-" * 75)

    return {
        "grid": results_grid,
        "params": {
            "machs": machs,
            "alphas": alphas,
            "b": b,
            "kappa": kappa,
            "grid_size": grid_size,
        }
    }


def make_plot(results: dict, show: bool = False) -> str:
    """Generate gradient health visualization.

    Parameters
    ----------
    results : dict
        Output from run_validation().
    show : bool
        Display interactively.

    Returns
    -------
    path : str
        Path to saved plot.
    """
    setup_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    machs = results["params"]["machs"]
    alphas = results["params"]["alphas"]
    grid = results["grid"]

    # Build 2D arrays
    grad_m_arr = np.zeros((len(alphas), len(machs)))
    grad_a_arr = np.zeros((len(alphas), len(machs)))
    grad_m_log_arr = np.zeros((len(alphas), len(machs)))
    f_tail_arr = np.zeros((len(alphas), len(machs)))

    for i, alpha in enumerate(alphas):
        for j, mach in enumerate(machs):
            r = grid[(mach, alpha)]
            grad_m_arr[i, j] = r.grad_mach
            grad_a_arr[i, j] = r.grad_alpha
            grad_m_log_arr[i, j] = r.grad_mach_log
            f_tail_arr[i, j] = r.f_tail

    extent = [min(machs) - 2.5, max(machs) + 2.5,
              min(alphas) - 0.125, max(alphas) + 0.125]

    # TOP LEFT: ∂f/∂M
    ax1 = axes[0, 0]
    im1 = ax1.imshow(grad_m_arr, origin="lower", aspect="auto",
                     extent=extent, cmap="RdBu_r")
    ax1.set_xlabel("Mach Number", fontsize=11)
    ax1.set_ylabel("$\\alpha$", fontsize=11)
    ax1.set_title("$\\partial f_{tail}/\\partial M$", fontsize=12)
    plt.colorbar(im1, ax=ax1)

    # TOP RIGHT: ∂f/∂α
    ax2 = axes[0, 1]
    im2 = ax2.imshow(grad_a_arr, origin="lower", aspect="auto",
                     extent=extent, cmap="RdBu_r")
    ax2.set_xlabel("Mach Number", fontsize=11)
    ax2.set_ylabel("$\\alpha$", fontsize=11)
    ax2.set_title("$\\partial f_{tail}/\\partial \\alpha$", fontsize=12)
    plt.colorbar(im2, ax=ax2)

    # BOTTOM LEFT: ∂log(f)/∂log(M) (elasticity)
    ax3 = axes[1, 0]
    im3 = ax3.imshow(grad_m_log_arr, origin="lower", aspect="auto",
                     extent=extent, cmap="viridis", vmin=-2, vmax=2)
    ax3.set_xlabel("Mach Number", fontsize=11)
    ax3.set_ylabel("$\\alpha$", fontsize=11)
    ax3.set_title("$\\partial \\log f/\\partial \\log M$ (elasticity)", fontsize=12)
    plt.colorbar(im3, ax=ax3)

    # Mark "safe" region where |elasticity| < 1
    safe_mask = np.abs(grad_m_log_arr) < 1.0
    ax3.contour(machs, alphas, safe_mask.astype(float),
                levels=[0.5], colors=["white"], linewidths=2, linestyles="--")

    # BOTTOM RIGHT: f_tail surface
    ax4 = axes[1, 1]
    im4 = ax4.imshow(f_tail_arr, origin="lower", aspect="auto",
                     extent=extent, cmap="plasma")
    ax4.set_xlabel("Mach Number", fontsize=11)
    ax4.set_ylabel("$\\alpha$", fontsize=11)
    ax4.set_title("$f_{tail}$ (dense fraction)", fontsize=12)
    plt.colorbar(im4, ax=ax4)

    # Contours for f_tail
    cs = ax4.contour(machs, alphas, f_tail_arr,
                     levels=[0.01, 0.05, 0.1, 0.2], colors=["white"], linewidths=1)
    ax4.clabel(cs, fmt="%.2f", fontsize=8)

    plt.suptitle(
        f"Gradient Health Analysis (κ={results['params']['kappa']})\n"
        f"White dashed: safe inference region (|elasticity| < 1)",
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "e6_gradients")
    plt.close(fig)

    return path


def main():
    """Run full E6 validation."""
    results = run_validation(verbose=True)
    make_plot(results)

    print("\n" + "=" * 70)
    print("E6 VALIDATION COMPLETE")
    print("=" * 70)

    grid = results["grid"]
    machs = results["params"]["machs"]
    alphas = results["params"]["alphas"]

    # Compute summary statistics
    all_grad_m = [grid[(m, a)].grad_mach for m in machs for a in alphas]
    all_grad_a = [grid[(m, a)].grad_alpha for m in machs for a in alphas]
    all_elasticity = [grid[(m, a)].grad_mach_log for m in machs for a in alphas]

    print(f"\nGradient Statistics:")
    print(f"  ∂f/∂M: median={np.median(all_grad_m):.4f}, range=[{min(all_grad_m):.4f}, {max(all_grad_m):.4f}]")
    print(f"  ∂f/∂α: median={np.median(all_grad_a):.4f}, range=[{min(all_grad_a):.4f}, {max(all_grad_a):.4f}]")
    print(f"  Elasticity: median={np.median(all_elasticity):.2f}, range=[{min(all_elasticity):.2f}, {max(all_elasticity):.2f}]")

    # Check for pathological gradients
    vanishing = sum(1 for g in all_grad_m if abs(g) < 1e-6)
    exploding = sum(1 for g in all_grad_m if abs(g) > 1.0)

    print(f"\nPathological Cases:")
    print(f"  Vanishing (|∂f/∂M| < 1e-6): {vanishing}/{len(all_grad_m)}")
    print(f"  Exploding (|∂f/∂M| > 1): {exploding}/{len(all_grad_m)}")

    # Safe inference box
    safe_points = [(m, a) for m in machs for a in alphas
                   if abs(grid[(m, a)].grad_mach_log) < 1.0]

    if safe_points:
        safe_m = [p[0] for p in safe_points]
        safe_a = [p[1] for p in safe_points]
        print(f"\nSafe Inference Box (|elasticity| < 1):")
        print(f"  Mach: [{min(safe_m):.0f}, {max(safe_m):.0f}]")
        print(f"  α: [{min(safe_a):.2f}, {max(safe_a):.2f}]")
        print(f"  Coverage: {len(safe_points)}/{len(machs) * len(alphas)} points")

    return results


if __name__ == "__main__":
    main()
