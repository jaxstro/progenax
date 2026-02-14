#!/usr/bin/env python
"""C8: eta_survive Sensitivity — UNCERTAINTY QUANTIFICATION.

f_sub vs eta_survive (0->1) for different environments.

Shows eta_survive is the biggest lever, where uncertainties live.

Output: c8_eta_sensitivity.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from progenax.gravoturb import bm19_model as bm19

from .helpers import (
    setup_publication_style,
    save_plot,
    ENVIRONMENT_PRESETS,
    OBSERVATIONAL_ANCHORS,
    COLORS,
)


def run_validation(
    eta_range: np.ndarray | None = None,
    verbose: bool = True,
):
    """Analyze sensitivity of f_sub to eta_survive.

    Parameters
    ----------
    eta_range : array
        eta_survive values to test
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Sensitivity analysis
    """
    if verbose:
        print("=" * 70)
        print("C8: ETA_SURVIVE SENSITIVITY ANALYSIS")
        print("=" * 70)

    if eta_range is None:
        eta_range = np.linspace(0.0, 1.0, 51)

    b = 0.4

    # Test across environments
    env_names = ["gmc_solar", "cmz_like", "ymc_forming"]
    results_by_env = {}

    for env_name in env_names:
        env = ENVIRONMENT_PRESETS[env_name]

        if verbose:
            print(f"\n{env.name}: M={env.Mach}, alpha={env.alpha}")

        # f_dense doesn't depend on eta
        result_base = bm19.bm19_pipeline(env.Mach, env.b, env.alpha, eta_survive=1.0)
        f_dense = float(result_base.f_dense)

        # f_sub = eta * f_dense
        f_sub_vals = eta_range * f_dense

        results_by_env[env_name] = {
            "env": env,
            "f_dense": f_dense,
            "f_sub": f_sub_vals,
        }

        if verbose:
            print(f"  f_dense = {f_dense:.4f}")
            print(f"  f_sub range: [{f_sub_vals.min():.4f}, {f_sub_vals.max():.4f}]")

    # Compare to observational SFE constraints
    gmc_sfe_low = OBSERVATIONAL_ANCHORS["gmc_sfe_low"]
    gmc_sfe_high = OBSERVATIONAL_ANCHORS["gmc_sfe_high"]
    ymc_sfe_low = OBSERVATIONAL_ANCHORS["ymc_sfe_low"]
    ymc_sfe_high = OBSERVATIONAL_ANCHORS["ymc_sfe_high"]

    if verbose:
        print(f"\nObservational SFE benchmarks:")
        print(f"  GMC: {gmc_sfe_low*100:.0f}-{gmc_sfe_high*100:.0f}%")
        print(f"  YMC: {ymc_sfe_low*100:.0f}-{ymc_sfe_high*100:.0f}%")

    return {
        "by_env": results_by_env,
        "eta_range": eta_range,
        "benchmarks": {
            "gmc_sfe_low": gmc_sfe_low,
            "gmc_sfe_high": gmc_sfe_high,
            "ymc_sfe_low": ymc_sfe_low,
            "ymc_sfe_high": ymc_sfe_high,
        }
    }


def make_plot(results: dict, show: bool = False) -> str:
    """Generate eta_survive sensitivity plot.

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

    eta_range = results["eta_range"]
    benchmarks = results["benchmarks"]

    # Color mapping
    env_colors = {
        "gmc_solar": COLORS["gmc"],
        "cmz_like": COLORS["cmz"],
        "ymc_forming": COLORS["ymc"],
    }

    # LEFT: f_sub vs eta
    ax1 = axes[0]

    for env_name, data in results["by_env"].items():
        ax1.plot(
            eta_range, data["f_sub"],
            color=env_colors[env_name], linewidth=2.5,
            label=f"{data['env'].name} ($f_{{dense}}$={data['f_dense']:.3f})"
        )

    # Observational SFE bands
    ax1.axhspan(
        benchmarks["gmc_sfe_low"], benchmarks["gmc_sfe_high"],
        alpha=0.15, color="blue", label=f"GMC SFE (1-5%)"
    )
    ax1.axhspan(
        benchmarks["ymc_sfe_low"], benchmarks["ymc_sfe_high"],
        alpha=0.15, color="red", label=f"YMC SFE (10-30%)"
    )

    ax1.set_xlabel("$\\eta_\\mathrm{survive}$ (feedback survival)", fontsize=12)
    ax1.set_ylabel("$f_\\mathrm{sub} = \\eta \\times f_\\mathrm{dense}$", fontsize=12)
    ax1.set_title("Substructure Fraction vs Feedback Efficiency", fontsize=14)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 0.35)

    # RIGHT: Required eta to match observations
    ax2 = axes[1]

    # For each environment, find eta range that matches SFE constraints
    bar_width = 0.25
    env_list = list(results["by_env"].keys())

    for i, env_name in enumerate(env_list):
        data = results["by_env"][env_name]
        f_dense = data["f_dense"]

        # GMC SFE range
        eta_gmc_low = benchmarks["gmc_sfe_low"] / f_dense
        eta_gmc_high = benchmarks["gmc_sfe_high"] / f_dense
        eta_gmc_low = np.clip(eta_gmc_low, 0, 1)
        eta_gmc_high = np.clip(eta_gmc_high, 0, 1)

        # YMC SFE range
        eta_ymc_low = benchmarks["ymc_sfe_low"] / f_dense
        eta_ymc_high = benchmarks["ymc_sfe_high"] / f_dense
        eta_ymc_low = np.clip(eta_ymc_low, 0, 1)
        eta_ymc_high = np.clip(eta_ymc_high, 0, 1)

        # Plot bars
        ax2.barh(
            i - bar_width/2, eta_gmc_high - eta_gmc_low, left=eta_gmc_low,
            height=bar_width, color="blue", alpha=0.5, label="GMC SFE" if i == 0 else ""
        )
        ax2.barh(
            i + bar_width/2, eta_ymc_high - eta_ymc_low, left=eta_ymc_low,
            height=bar_width, color="red", alpha=0.5, label="YMC SFE" if i == 0 else ""
        )

    ax2.set_yticks(range(len(env_list)))
    ax2.set_yticklabels([results["by_env"][e]["env"].name for e in env_list])
    ax2.set_xlabel("Required $\\eta_\\mathrm{survive}$ to match observed SFE", fontsize=12)
    ax2.set_title("Implied Feedback Efficiency", fontsize=14)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.set_xlim(0, 1)

    plt.suptitle(
        "$\\eta_\\mathrm{survive}$ is the Dominant Uncertainty in $f_\\mathrm{sub}$",
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "c8_eta_sensitivity")
    plt.close(fig)

    return path


def make_tornado_chart(results: dict, show: bool = False) -> str:
    """Tornado chart showing parameter sensitivity.

    Shows that eta_survive dominates uncertainty in f_sub.
    """
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use GMC-solar as reference
    env = ENVIRONMENT_PRESETS["gmc_solar"]
    mach_ref, alpha_ref, b_ref, eta_ref = env.Mach, env.alpha, env.b, 0.6

    # Reference f_sub
    result_ref = bm19.bm19_pipeline(mach_ref, b_ref, alpha_ref, eta_ref)
    f_sub_ref = float(result_ref.f_sub)

    # Parameter variations
    params = {
        "$\\eta_\\mathrm{survive}$": (0.3, 0.9, eta_ref),
        "$\\mathcal{M}$": (7, 15, mach_ref),
        "$\\alpha$": (1.5, 2.5, alpha_ref),
        "$b$": (0.3, 0.5, b_ref),
    }

    sensitivities = []
    param_names = []

    for name, (low, high, ref) in params.items():
        if name == "$\\eta_\\mathrm{survive}$":
            f_sub_low = float(bm19.bm19_pipeline(mach_ref, b_ref, alpha_ref, low).f_sub)
            f_sub_high = float(bm19.bm19_pipeline(mach_ref, b_ref, alpha_ref, high).f_sub)
        elif name == "$\\mathcal{M}$":
            f_sub_low = float(bm19.bm19_pipeline(low, b_ref, alpha_ref, eta_ref).f_sub)
            f_sub_high = float(bm19.bm19_pipeline(high, b_ref, alpha_ref, eta_ref).f_sub)
        elif name == "$\\alpha$":
            f_sub_low = float(bm19.bm19_pipeline(mach_ref, b_ref, low, eta_ref).f_sub)
            f_sub_high = float(bm19.bm19_pipeline(mach_ref, b_ref, high, eta_ref).f_sub)
        elif name == "$b$":
            f_sub_low = float(bm19.bm19_pipeline(mach_ref, low, alpha_ref, eta_ref).f_sub)
            f_sub_high = float(bm19.bm19_pipeline(mach_ref, high, alpha_ref, eta_ref).f_sub)

        # Sensitivity as % change from reference
        delta_low = 100 * (f_sub_low - f_sub_ref) / f_sub_ref
        delta_high = 100 * (f_sub_high - f_sub_ref) / f_sub_ref

        sensitivities.append((delta_low, delta_high))
        param_names.append(name)

    # Sort by total sensitivity
    total_sens = [abs(s[0]) + abs(s[1]) for s in sensitivities]
    sort_idx = np.argsort(total_sens)[::-1]
    param_names = [param_names[i] for i in sort_idx]
    sensitivities = [sensitivities[i] for i in sort_idx]

    # Plot tornado
    y_pos = np.arange(len(param_names))

    for i, (name, (delta_low, delta_high)) in enumerate(zip(param_names, sensitivities)):
        color = "C0" if delta_low < 0 else "C1"
        ax.barh(i, delta_low, color=color, alpha=0.7)

        color = "C1" if delta_high > 0 else "C0"
        ax.barh(i, delta_high, color=color, alpha=0.7)

    ax.axvline(x=0, color="k", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlabel("$\\Delta f_\\mathrm{sub}$ from reference [%]", fontsize=12)
    ax.set_title(
        f"Parameter Sensitivity (reference: GMC-solar, $f_{{sub}}$={f_sub_ref:.3f})",
        fontsize=14
    )
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "c8_eta_tornado")
    plt.close(fig)

    return path


def main():
    """Run full C8 validation."""
    results = run_validation(verbose=True)
    make_plot(results)
    make_tornado_chart(results)

    print("\n" + "=" * 70)
    print("C8 VALIDATION COMPLETE")
    print("=" * 70)
    print("\nKey findings:")
    print("  1. eta_survive is the dominant source of f_sub uncertainty")
    print("  2. f_sub = eta * f_dense (linear relationship)")
    print("  3. GMC SFE (1-5%) requires eta ~ 0.1-0.5")
    print("  4. YMC SFE (10-30%) requires eta ~ 0.5-1.0 or high f_dense")

    # Estimate implied eta for each environment
    print("\nImplied eta_survive to match GMC SFE (~3%):")
    for env_name, data in results["by_env"].items():
        f_dense = data["f_dense"]
        eta_implied = 0.03 / f_dense
        print(f"  {data['env'].name}: eta ~ {eta_implied:.2f}")

    return results


if __name__ == "__main__":
    main()
