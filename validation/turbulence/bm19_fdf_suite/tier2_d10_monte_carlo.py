#!/usr/bin/env python
"""D10: Monte Carlo Uncertainty Propagation.

Sample (M, alpha, b, eta) from priors, propagate to f_sub distributions.

Output: d10_monte_carlo.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from progenax.gravoturb import bm19_model as bm19

from .helpers import (
    setup_publication_style,
    save_plot,
    compute_statistics,
    ENVIRONMENT_PRESETS,
    COLORS,
)


def run_validation(
    n_samples: int = 10000,
    verbose: bool = True,
):
    """Monte Carlo uncertainty propagation for f_sub.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo samples
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        MC distributions
    """
    if verbose:
        print("=" * 70)
        print("D10: MONTE CARLO UNCERTAINTY PROPAGATION")
        print("=" * 70)
        print(f"\nN samples: {n_samples}")

    np.random.seed(42)

    # Define priors (truncated normal / uniform)
    priors = {
        "GMC": {
            "mach": {"mean": 10.0, "std": 3.0, "min": 3.0, "max": 25.0},
            "alpha": {"mean": 2.0, "std": 0.3, "min": 1.3, "max": 3.5},
            "b": {"mean": 0.4, "std": 0.1, "min": 0.25, "max": 0.6},
            "eta": {"mean": 0.5, "std": 0.2, "min": 0.1, "max": 0.9},
        },
        "CMZ": {
            "mach": {"mean": 30.0, "std": 10.0, "min": 10.0, "max": 60.0},
            "alpha": {"mean": 1.8, "std": 0.3, "min": 1.3, "max": 3.5},
            "b": {"mean": 0.4, "std": 0.1, "min": 0.25, "max": 0.6},
            "eta": {"mean": 0.6, "std": 0.2, "min": 0.2, "max": 1.0},
        },
        "YMC": {
            "mach": {"mean": 20.0, "std": 5.0, "min": 10.0, "max": 40.0},
            "alpha": {"mean": 2.0, "std": 0.3, "min": 1.3, "max": 3.5},
            "b": {"mean": 0.4, "std": 0.1, "min": 0.25, "max": 0.6},
            "eta": {"mean": 0.7, "std": 0.15, "min": 0.3, "max": 1.0},
        },
    }

    results_by_env = {}

    for env_name, prior in priors.items():
        if verbose:
            print(f"\n{env_name}:")
            print(f"  Priors: M~N({prior['mach']['mean']}, {prior['mach']['std']}), etc.")

        # Sample from truncated normals
        def sample_truncated_normal(p, n):
            """Sample from truncated normal given prior dict."""
            samples = np.random.normal(p["mean"], p["std"], n * 2)  # oversample
            samples = samples[(samples >= p["min"]) & (samples <= p["max"])]
            return samples[:n]

        mach_samples = sample_truncated_normal(prior["mach"], n_samples)
        alpha_samples = sample_truncated_normal(prior["alpha"], n_samples)
        b_samples = sample_truncated_normal(prior["b"], n_samples)
        eta_samples = sample_truncated_normal(prior["eta"], n_samples)

        # Ensure we have enough samples
        n_actual = min(len(mach_samples), len(alpha_samples), len(b_samples), len(eta_samples))
        mach_samples = mach_samples[:n_actual]
        alpha_samples = alpha_samples[:n_actual]
        b_samples = b_samples[:n_actual]
        eta_samples = eta_samples[:n_actual]

        # Propagate through BM19
        f_dense_samples = []
        f_sub_samples = []

        for i in range(n_actual):
            result = bm19.bm19_pipeline(
                mach_samples[i], b_samples[i], alpha_samples[i], eta_samples[i]
            )
            f_dense_samples.append(float(result.f_dense))
            f_sub_samples.append(float(result.f_sub))

        f_dense_samples = np.array(f_dense_samples)
        f_sub_samples = np.array(f_sub_samples)

        stats_f_dense = compute_statistics(f_dense_samples)
        stats_f_sub = compute_statistics(f_sub_samples)

        results_by_env[env_name] = {
            "prior": prior,
            "samples": {
                "mach": mach_samples,
                "alpha": alpha_samples,
                "b": b_samples,
                "eta": eta_samples,
                "f_dense": f_dense_samples,
                "f_sub": f_sub_samples,
            },
            "stats_f_dense": stats_f_dense,
            "stats_f_sub": stats_f_sub,
        }

        if verbose:
            print(f"  f_dense: {stats_f_dense['mean']:.4f} +/- {stats_f_dense['std']:.4f}")
            print(f"    95% CI: [{stats_f_dense['p5']:.4f}, {stats_f_dense['p95']:.4f}]")
            print(f"  f_sub: {stats_f_sub['mean']:.4f} +/- {stats_f_sub['std']:.4f}")
            print(f"    95% CI: [{stats_f_sub['p5']:.4f}, {stats_f_sub['p95']:.4f}]")

    return {
        "by_env": results_by_env,
        "n_samples": n_samples,
    }


def make_plot(results: dict, show: bool = False) -> str:
    """Generate Monte Carlo distribution plots.

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

    env_names = list(results["by_env"].keys())
    n_envs = len(env_names)

    fig, axes = plt.subplots(2, n_envs, figsize=(5 * n_envs, 8))

    env_colors = {"GMC": COLORS["gmc"], "CMZ": COLORS["cmz"], "YMC": COLORS["ymc"]}

    for i, env_name in enumerate(env_names):
        data = results["by_env"][env_name]
        samples = data["samples"]
        stats_f_dense = data["stats_f_dense"]
        stats_f_sub = data["stats_f_sub"]
        color = env_colors.get(env_name, "C0")

        # Top: f_dense histogram
        ax1 = axes[0, i]
        ax1.hist(
            samples["f_dense"], bins=50, density=True,
            alpha=0.7, color=color, edgecolor="black", linewidth=0.5
        )
        ax1.axvline(stats_f_dense["mean"], color="red", linewidth=2, label="Mean")
        ax1.axvline(stats_f_dense["p5"], color="red", linestyle="--", linewidth=1, label="5-95%")
        ax1.axvline(stats_f_dense["p95"], color="red", linestyle="--", linewidth=1)
        ax1.set_xlabel("$f_\\mathrm{dense}$", fontsize=11)
        ax1.set_ylabel("Density", fontsize=11)
        ax1.set_title(f"{env_name}: $f_{{dense}}$\n$\\mu$={stats_f_dense['mean']:.4f}, $\\sigma$={stats_f_dense['std']:.4f}", fontsize=12)
        ax1.legend(fontsize=9)

        # Bottom: f_sub histogram
        ax2 = axes[1, i]
        ax2.hist(
            samples["f_sub"], bins=50, density=True,
            alpha=0.7, color=color, edgecolor="black", linewidth=0.5
        )
        ax2.axvline(stats_f_sub["mean"], color="red", linewidth=2, label="Mean")
        ax2.axvline(stats_f_sub["p5"], color="red", linestyle="--", linewidth=1, label="5-95%")
        ax2.axvline(stats_f_sub["p95"], color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel("$f_\\mathrm{sub} = \\eta \\times f_\\mathrm{dense}$", fontsize=11)
        ax2.set_ylabel("Density", fontsize=11)
        ax2.set_title(f"{env_name}: $f_{{sub}}$\n$\\mu$={stats_f_sub['mean']:.4f}, $\\sigma$={stats_f_sub['std']:.4f}", fontsize=12)
        ax2.legend(fontsize=9)

    plt.suptitle(
        f"Monte Carlo Uncertainty Propagation (N={results['n_samples']:,})",
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "d10_monte_carlo")
    plt.close(fig)

    return path


def make_correlation_plot(results: dict, show: bool = False) -> str:
    """Correlation between input parameters and f_sub."""
    setup_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Use GMC as example
    env_name = "GMC"
    data = results["by_env"][env_name]
    samples = data["samples"]

    param_names = ["mach", "alpha", "b", "eta"]
    param_labels = ["$\\mathcal{M}$", "$\\alpha$", "$b$", "$\\eta_\\mathrm{survive}$"]

    for ax, param, label in zip(axes.flat, param_names, param_labels):
        ax.scatter(
            samples[param], samples["f_sub"],
            alpha=0.3, s=5, c=COLORS["gmc"]
        )

        # Add correlation coefficient
        corr = np.corrcoef(samples[param], samples["f_sub"])[0, 1]
        ax.annotate(f"$r$ = {corr:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=12)

        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("$f_\\mathrm{sub}$", fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Correlation Analysis: {env_name} Environment",
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "d10_monte_carlo_correlation")
    plt.close(fig)

    return path


def main():
    """Run full D10 validation."""
    results = run_validation(n_samples=10000, verbose=True)
    make_plot(results)
    make_correlation_plot(results)

    print("\n" + "=" * 70)
    print("D10 VALIDATION COMPLETE")
    print("=" * 70)
    print("\nKey findings:")

    for env_name, data in results["by_env"].items():
        relative_uncertainty = 100 * data["stats_f_sub"]["std"] / data["stats_f_sub"]["mean"]
        print(f"  {env_name}: f_sub = {data['stats_f_sub']['mean']:.4f} +/- {data['stats_f_sub']['std']:.4f} ({relative_uncertainty:.0f}% uncertainty)")

    return results


if __name__ == "__main__":
    main()
