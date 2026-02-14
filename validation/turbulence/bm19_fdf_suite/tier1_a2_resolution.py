#!/usr/bin/env python
"""A2: Resolution Convergence — GRID SIZE SANITY.

GMC-like & CMZ-like at 32^3 -> 128^3, plot |f_tail - f_dense| vs resolution.

Uses CDF remap to generate proper BM19 LN+PL fields (same as A1).
Justifies "we adopt 128^3 because..." for methods section.

Output: a2_resolution_convergence.png
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np

from progenax.gravoturb import bm19_model as bm19
from progenax.gravoturb import gaussian_to_bm19, build_bm19_cdf_table
from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19

from .tier1_a1_ftail_fdense import compute_expected_tail_voxels
from .helpers import (
    setup_publication_style,
    save_plot,
    compute_statistics,
    ENVIRONMENT_PRESETS,
    COLORS,
)


def run_validation(
    resolutions: list[int] = [32, 48, 64, 96, 128],
    n_realizations: int = 10,
    kappa: float = 10.0,
    verbose: bool = True,
):
    """Run resolution convergence validation.

    Parameters
    ----------
    resolutions : list
        Grid sizes to test
    n_realizations : int
        Number of realizations per resolution
    kappa : float
        Soft sigmoid sharpness
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Validation results
    """
    if verbose:
        print("=" * 70)
        print("A2: RESOLUTION CONVERGENCE VALIDATION")
        print("=" * 70)

    # Test environments
    envs = {
        "GMC": ENVIRONMENT_PRESETS["gmc_solar"],
        "CMZ": ENVIRONMENT_PRESETS["cmz_like"],
    }

    results_by_env = {}

    for env_name, env in envs.items():
        if verbose:
            print(f"\n{env_name}: Mach={env.Mach}, alpha={env.alpha}")
            print("-" * 50)

        # BM19 theory
        bm19_result = bm19.bm19_pipeline(env.Mach, env.b, env.alpha, eta_survive=0.6)
        f_dense_theory = float(bm19_result.f_dense)
        s_t = float(bm19_result.s_t)
        sigma_s_sq = float(bm19_result.sigma_s_sq)

        # Build CDF table once per environment (resolution-independent)
        s_grid, F_grid = build_bm19_cdf_table(sigma_s_sq, s_t, env.alpha)

        res_data = {"resolution": [], "mean_error": [], "std_error": [], "errors": [],
                    "expected_voxels": [], "skipped": False}

        for res in resolutions:
            n_voxels = res ** 3

            # Check sampleability at this resolution
            expected_voxels = compute_expected_tail_voxels(sigma_s_sq, s_t, env.alpha, n_voxels)
            if expected_voxels < 1.0:
                if verbose:
                    print(f"  {res}^3: SKIP (expected {expected_voxels:.1f} voxels in tail)")
                res_data["resolution"].append(res)
                res_data["mean_error"].append(np.nan)
                res_data["std_error"].append(np.nan)
                res_data["errors"].append([])
                res_data["expected_voxels"].append(expected_voxels)
                continue

            errors = []

            for i in range(n_realizations):
                key = random.PRNGKey(int(env.Mach * 100) + res + i)

                # Generate BM19 LN+PL field via CDF remap
                g = random.normal(key, (res, res, res))
                s = gaussian_to_bm19(g, sigma_s_sq, s_t, env.alpha, s_grid, F_grid)
                rho_grid = jnp.exp(s)

                # Compute f_tail_actual using soft sigmoid
                w = jax.nn.sigmoid(kappa * (s - s_t))
                f_tail = float(jnp.sum(w * rho_grid) / jnp.sum(rho_grid))

                rel_error = 100 * (f_tail - f_dense_theory) / f_dense_theory
                errors.append(rel_error)

            mean_err = np.mean(errors)
            std_err = np.std(errors)

            res_data["resolution"].append(res)
            res_data["mean_error"].append(mean_err)
            res_data["std_error"].append(std_err)
            res_data["errors"].append(errors)
            res_data["expected_voxels"].append(expected_voxels)

            if verbose:
                print(f"  {res}^3: error = {mean_err:+.1f}% +/- {std_err:.1f}% ({expected_voxels:.0f} tail voxels)")

        res_data["f_dense_theory"] = f_dense_theory
        results_by_env[env_name] = res_data

    return {
        "by_env": results_by_env,
        "params": {
            "resolutions": resolutions,
            "n_realizations": n_realizations,
            "kappa": kappa,
        }
    }


def make_plot(results: dict, show: bool = False) -> str:
    """Generate resolution convergence plot.

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"GMC": COLORS["gmc"], "CMZ": COLORS["cmz"]}
    markers = {"GMC": "o", "CMZ": "s"}

    # Left: Mean error vs resolution
    ax1 = axes[0]
    for env_name, data in results["by_env"].items():
        ax1.errorbar(
            data["resolution"],
            np.abs(data["mean_error"]),
            yerr=data["std_error"],
            marker=markers[env_name],
            color=colors[env_name],
            label=env_name,
            capsize=3,
            linewidth=2,
            markersize=8,
        )

    ax1.axhline(y=5, color="green", linestyle="--", alpha=0.7, label="5% target")
    ax1.axhline(y=10, color="orange", linestyle="--", alpha=0.7, label="10% threshold")
    ax1.set_xlabel("Grid Resolution (N)", fontsize=12)
    ax1.set_ylabel("|Mean Error| (%)", fontsize=12)
    ax1.set_title("Mean Error Convergence", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(results["params"]["resolutions"])
    ax1.set_xticklabels([str(r) for r in results["params"]["resolutions"]])
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, None)

    # Right: Scatter variance vs resolution
    ax2 = axes[1]
    for env_name, data in results["by_env"].items():
        ax2.plot(
            data["resolution"],
            data["std_error"],
            marker=markers[env_name],
            color=colors[env_name],
            label=env_name,
            linewidth=2,
            markersize=8,
        )

    ax2.axhline(y=5, color="green", linestyle="--", alpha=0.7, label="5% target")
    ax2.axhline(y=10, color="orange", linestyle="--", alpha=0.7, label="10% threshold")
    ax2.set_xlabel("Grid Resolution (N)", fontsize=12)
    ax2.set_ylabel("Std Error (%)", fontsize=12)
    ax2.set_title("Realization-to-Realization Scatter", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(results["params"]["resolutions"])
    ax2.set_xticklabels([str(r) for r in results["params"]["resolutions"]])
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, None)

    plt.suptitle(
        "Resolution Convergence for f_tail vs f_dense\n"
        f"(N={results['params']['n_realizations']} realizations per point)",
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "a2_resolution_convergence")
    plt.close(fig)

    return path


def main():
    """Run full A2 validation."""
    # Use smaller resolutions for faster runtime
    results = run_validation(
        resolutions=[32, 48, 64, 96, 128],
        n_realizations=10,
        verbose=True
    )
    make_plot(results)

    print("\n" + "=" * 70)
    print("A2 VALIDATION COMPLETE")
    print("=" * 70)

    # Recommend resolution
    for env_name, data in results["by_env"].items():
        for i, res in enumerate(data["resolution"]):
            if data["std_error"][i] < 10:
                print(f"{env_name}: Recommend res >= {res}^3 (std < 10%)")
                break

    return results


if __name__ == "__main__":
    main()
