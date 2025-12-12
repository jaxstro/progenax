#!/usr/bin/env python
"""C7: Column Density Threshold — OBSERVATIONAL ANCHOR.

Convert s_t -> N_H, compare to Lada threshold ~7e21 cm^-2.

"Does our s_t look like the observed star-forming threshold?"

Output: c7_column_density.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from progenax.gravoturb import bm19_model as bm19

from .helpers import (
    setup_publication_style,
    save_plot,
    s_to_column_density,
    column_density_to_s,
    ENVIRONMENT_PRESETS,
    OBSERVATIONAL_ANCHORS,
    COLORS,
)


def run_validation(
    machs: np.ndarray | None = None,
    alphas: list[float] = [1.5, 2.0, 2.5, 3.0],
    Sigma: float = 100.0,
    depth_pc: float = 1.0,
    verbose: bool = True,
):
    """Compare BM19 s_t to Lada column density threshold.

    Parameters
    ----------
    machs : array
        Mach number range
    alphas : list
        BM19 powerlaw slopes
    Sigma : float
        Surface density [Msun/pc^2]
    depth_pc : float
        Cloud depth [pc]
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Column density comparisons
    """
    if verbose:
        print("=" * 70)
        print("C7: COLUMN DENSITY THRESHOLD VALIDATION")
        print("=" * 70)
        print(f"\nObservational reference: Lada+ 2010")
        print(f"  N_H threshold = {OBSERVATIONAL_ANCHORS['lada_threshold_cm2']:.1e} cm^-2")

    if machs is None:
        machs = np.linspace(5, 35, 50)

    b = 0.4

    # Lada threshold in s-units at this Sigma
    s_lada = column_density_to_s(OBSERVATIONAL_ANCHORS["lada_threshold_cm2"], Sigma, depth_pc)

    if verbose:
        print(f"\nCloud parameters: Sigma={Sigma} Msun/pc^2, depth={depth_pc} pc")
        print(f"Lada threshold corresponds to s = {s_lada:.2f}")

    results_by_alpha = {}

    for alpha in alphas:
        s_t_vals = []
        N_H_t_vals = []

        for mach in machs:
            result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.6)
            s_t = float(result.s_t)
            N_H_t = s_to_column_density(s_t, Sigma, depth_pc)

            s_t_vals.append(s_t)
            N_H_t_vals.append(N_H_t)

        s_t_vals = np.array(s_t_vals)
        N_H_t_vals = np.array(N_H_t_vals)

        results_by_alpha[alpha] = {
            "s_t": s_t_vals,
            "N_H_t": N_H_t_vals,
        }

        if verbose:
            print(f"\nalpha={alpha}:")
            print(f"  s_t range: [{s_t_vals.min():.2f}, {s_t_vals.max():.2f}]")
            print(f"  N_H_t range: [{N_H_t_vals.min():.2e}, {N_H_t_vals.max():.2e}] cm^-2")

    return {
        "by_alpha": results_by_alpha,
        "machs": machs,
        "lada_threshold": OBSERVATIONAL_ANCHORS["lada_threshold_cm2"],
        "s_lada": s_lada,
        "params": {
            "alphas": alphas,
            "b": b,
            "Sigma": Sigma,
            "depth_pc": depth_pc,
        }
    }


def make_plot(results: dict, show: bool = False) -> str:
    """Generate column density comparison plot.

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

    machs = results["machs"]
    alphas = results["params"]["alphas"]

    # Color palette
    alpha_colors = {1.5: "C0", 2.0: "C1", 2.5: "C2", 3.0: "C3"}

    # LEFT: s_t vs Mach
    ax1 = axes[0]

    for alpha in alphas:
        ax1.plot(
            machs, results["by_alpha"][alpha]["s_t"],
            color=alpha_colors[alpha], linewidth=2,
            label=f"$\\alpha$ = {alpha}"
        )

    # Lada threshold in s-units
    ax1.axhline(y=results["s_lada"], color="red", linestyle="--", linewidth=2,
                label=f"Lada threshold ($s$ = {results['s_lada']:.2f})")

    ax1.set_xlabel("Mach Number ($\\mathcal{M}$)", fontsize=12)
    ax1.set_ylabel("$s_t = \\ln(\\rho_t / \\bar{\\rho})$", fontsize=12)
    ax1.set_title("BM19 Transition Density vs Mach", fontsize=14)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(machs.min(), machs.max())

    # RIGHT: N_H_t vs Mach
    ax2 = axes[1]

    for alpha in alphas:
        ax2.semilogy(
            machs, results["by_alpha"][alpha]["N_H_t"],
            color=alpha_colors[alpha], linewidth=2,
            label=f"$\\alpha$ = {alpha}"
        )

    # Lada threshold
    ax2.axhline(y=results["lada_threshold"], color="red", linestyle="--", linewidth=2,
                label=f"Lada threshold ({results['lada_threshold']:.0e} cm$^{{-2}}$)")

    # Shade region around Lada threshold
    ax2.axhspan(
        results["lada_threshold"] * 0.5, results["lada_threshold"] * 2,
        alpha=0.1, color="red", label="$\\pm$ 2x Lada"
    )

    ax2.set_xlabel("Mach Number ($\\mathcal{M}$)", fontsize=12)
    ax2.set_ylabel("$N_H(s_t)$ [cm$^{-2}$]", fontsize=12)
    ax2.set_title(
        f"BM19 Threshold as Column Density\n($\\Sigma$ = {results['params']['Sigma']} M$_\\odot$/pc$^2$)",
        fontsize=14
    )
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(machs.min(), machs.max())

    plt.suptitle(
        "Comparison with Lada+ 2010 Star-Formation Threshold",
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "c7_column_density")
    plt.close(fig)

    return path


def make_sigma_dependence_plot(show: bool = False) -> str:
    """Show how N_H(s_t) depends on Sigma.

    Since N_H ~ Sigma * exp(s_t), the actual column density threshold
    varies with cloud surface density.
    """
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    machs = np.linspace(5, 35, 50)
    Sigmas = [50, 100, 200, 500, 1000]
    alpha = 2.0
    b = 0.4

    colors_sigma = plt.cm.viridis(np.linspace(0.1, 0.9, len(Sigmas)))

    for i, Sigma in enumerate(Sigmas):
        N_H_t_vals = []
        for mach in machs:
            result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.6)
            s_t = float(result.s_t)
            N_H_t = s_to_column_density(s_t, Sigma)
            N_H_t_vals.append(N_H_t)

        ax.semilogy(
            machs, N_H_t_vals,
            color=colors_sigma[i], linewidth=2,
            label=f"$\\Sigma$ = {Sigma} M$_\\odot$/pc$^2$"
        )

    # Lada threshold
    lada = OBSERVATIONAL_ANCHORS["lada_threshold_cm2"]
    ax.axhline(y=lada, color="red", linestyle="--", linewidth=2, label=f"Lada ({lada:.0e} cm$^{{-2}}$)")

    ax.set_xlabel("Mach Number ($\\mathcal{M}$)", fontsize=12)
    ax.set_ylabel("$N_H(s_t)$ [cm$^{-2}$]", fontsize=12)
    ax.set_title(
        f"BM19 Threshold Column Density Depends on $\\Sigma$\n($\\alpha$ = {alpha})",
        fontsize=14
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(machs.min(), machs.max())

    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "c7_column_density_sigma_dependence")
    plt.close(fig)

    return path


def main():
    """Run full C7 validation."""
    results = run_validation(verbose=True)
    make_plot(results)
    make_sigma_dependence_plot()

    print("\n" + "=" * 70)
    print("C7 VALIDATION COMPLETE")
    print("=" * 70)

    # Summary
    lada = OBSERVATIONAL_ANCHORS["lada_threshold_cm2"]
    s_lada = results["s_lada"]

    print("\nKey findings:")
    print(f"  1. Lada threshold N_H = {lada:.0e} cm^-2")
    print(f"  2. At Sigma=100 Msun/pc^2, this corresponds to s = {s_lada:.2f}")
    print("  3. BM19 s_t typically ranges from 2-6 depending on (M, alpha)")
    print("  4. Column density interpretation depends strongly on Sigma")

    # Check which conditions match Lada threshold
    print("\nConditions matching Lada threshold (within 2x):")
    for alpha in results["params"]["alphas"]:
        N_H_t = results["by_alpha"][alpha]["N_H_t"]
        match_mask = (N_H_t > lada * 0.5) & (N_H_t < lada * 2)
        if np.any(match_mask):
            match_machs = results["machs"][match_mask]
            print(f"  alpha={alpha}: M in [{match_machs.min():.0f}, {match_machs.max():.0f}]")

    return results


if __name__ == "__main__":
    main()
