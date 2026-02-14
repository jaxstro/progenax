#!/usr/bin/env python
"""Validation plots for paper-calibrated environment-dependent IMF.

Creates publication-quality validation plots:
1. Alpha3 surface vs environment (log_mecl, [Fe/H])
2. GC validation vs Marks+2012 Table 1
3. Gradient flow demonstration
4. IMF shape comparison across environments

Run with:
    python validation/validate_env_imf.py
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

from progenax.imf import (
    BirthEnvironment,
    env_to_imf_params,
    alpha3_jerabkova_mecl,
    alpha3_marks_plane,
    lowmass_slopes_metallicity,
    JERABKOVA_COEFFICIENTS,
    MARKS_COEFFICIENTS,
    IMFParams,
)
from progenax.imf.differentiable import individual_mass_nll, log_prob_masses


# Output directory - dedicated subfolder for environment IMF plots
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_alpha3_surface():
    """Plot 1: Alpha3 as function of log_mecl and [Fe/H].

    Shows Jerabkova+2018 Eq. 9 predictions with calibration domain boundaries.
    """
    print("Creating alpha3 surface plot...")

    # Create grid
    log_mecl_range = np.linspace(3, 8, 100)  # 10^3 to 10^8 M_sun
    FeH_range = np.linspace(-2.5, 0.5, 100)
    log_mecl_grid, FeH_grid = np.meshgrid(log_mecl_range, FeH_range)

    # Convert to JAX and compute alpha3
    log_mecl_6 = jnp.array(log_mecl_grid) - 6.0
    FeH_jax = jnp.array(FeH_grid)

    alpha3 = jax.vmap(jax.vmap(alpha3_jerabkova_mecl))(log_mecl_6, FeH_jax)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Contour plot - z_min=0.7 since alpha3 doesn't go below ~0.7 in this domain
    levels = np.linspace(0.7, 2.3, 17)
    cs = ax.contourf(log_mecl_grid, FeH_grid, np.array(alpha3), levels=levels, cmap="RdYlBu_r")
    cbar = plt.colorbar(cs, ax=ax, label=r"$\alpha_3$ (high-mass slope)")

    # Add contour lines
    cs_lines = ax.contour(log_mecl_grid, FeH_grid, np.array(alpha3), levels=[0.7, 1.0, 1.5, 2.0, 2.3], colors="k", linewidths=0.5)
    ax.clabel(cs_lines, inline=True, fontsize=8, fmt="%.1f")

    # Mark calibration domain boundaries
    ax.axhline(-2.5, color="gray", linestyle="--", linewidth=0.5, label="Calibration boundary")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(3, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(8, color="gray", linestyle="--", linewidth=0.5)

    # Mark specific GCs
    gc_data = [
        ("NGC 104", 6.0, -0.76),   # 47 Tuc
        ("NGC 7078", 6.5, -2.16),  # M15 (most top-heavy)
    ]
    for name, log_m, feh in gc_data:
        ax.scatter(log_m, feh, marker="*", s=200, c="yellow", edgecolors="k", zorder=10)
        ax.annotate(name, (log_m + 0.1, feh + 0.1), fontsize=9, color="yellow",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    ax.set_xlabel(r"$\log_{10}(M_{\rm ecl} / M_\odot)$", fontsize=12)
    ax.set_ylabel(r"[Fe/H]", fontsize=12)
    ax.set_title("Jerabkova+2018 Eq. 9: High-Mass Slope vs Environment", fontsize=14)

    # Add equation annotation - READ FROM MODULE COEFFICIENTS (no hardcoding!)
    c = JERABKOVA_COEFFICIENTS
    eq_text = (
        rf"$x = {c['FeH_coeff']}\,{{\rm [Fe/H]}} + {c['logMecl_coeff']}\,\log_{{10}}(M_{{\rm ecl}}/10^6) + {c['constant']}$"
        "\n"
        rf"$\alpha_3 = {c['alpha3_slope']}\,x + {c['alpha3_intercept']}$ if $x \geq {c['x_threshold']}$"
    )
    ax.text(0.02, 0.02, eq_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    outpath = PLOT_DIR / "env_imf_alpha3_surface.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")


def plot_gc_validation():
    """Plot 2: Validation against Marks+2012 Table 1 GC values.

    Compares computed alpha3 vs published values for all 17 GCs in Table 1.

    IMPORTANT: ω Cen (NGC 5139) is a known edge case - see note below.
    """
    print("Creating GC validation plot...")

    # Marks+2012 Table 1 data: (name, FeH, rho_1e6 [M☉/pc³], expected_alpha3, is_edge_case)
    # COMPLETE table with all 17 globular clusters from the paper
    gc_data = [
        # Most top-heavy (lowest α₃)
        ("NGC 7078", -2.16, 258.13, 0.76, False),   # M15 - most extreme!
        ("NGC 7099", -2.12, 133.38, 0.92, False),   # M30
        ("NGC 6093", -1.75, 186.94, 0.93, False),   # M80
        ("NGC 6681", -1.51, 243.76, 0.97, False),   # M70
        # Intermediate top-heavy
        ("NGC 2298", -1.85, 26.53, 1.05, False),
        ("NGC 5286", -1.67, 95.27, 1.09, False),
        ("NGC 6284", -1.32, 141.39, 1.10, False),
        ("NGC 6341", -2.28, 66.03, 1.11, False),    # M92
        ("NGC 362", -1.30, 96.65, 1.12, False),
        ("NGC 6624", -0.44, 197.97, 1.14, False),
        # Moderate
        ("NGC 6522", -1.44, 38.26, 1.22, False),
        ("NGC 6752", -1.56, 31.78, 1.27, False),
        ("NGC 104", -0.76, 9.54, 1.34, False),      # 47 Tuc
        ("NGC 6388", -0.60, 22.43, 1.35, False),
        ("NGC 6656", -1.64, 9.16, 1.42, False),     # M22
        ("NGC 6441", -0.53, 10.65, 1.57, False),
        # Near canonical (highest α₃) - EDGE CASE
        ("NGC 5139", -1.62, 2.35, 1.91, True),      # ω Cen - see note below
    ]

    # ω Cen (NGC 5139) EDGE CASE DOCUMENTATION:
    # ------------------------------------
    # ω Cen has extremely low density (ρ = 2.35 × 10⁶ M☉/pc³), the lowest in Table 1.
    # log₁₀(ρ/10⁶) = 0.37, which is below the Marks threshold (0.87).
    #
    # The Fundamental Plane formula predicts α₃ ≈ 2.3 (canonical) because it's in
    # the "sub-threshold" regime. However, Table 1 reports α₃ = 1.91 from actual
    # stellar counts, not the formula.
    #
    # This discrepancy is expected: ω Cen's unique formation history (possibly a
    # stripped dwarf galaxy nucleus) means the simple density-metallicity relation
    # doesn't capture its true IMF. We mark it as an edge case and expect larger error.

    computed = []
    expected = []
    names = []
    edge_cases = []

    for name, FeH, rho_1e6, exp_alpha3, is_edge_case in gc_data:
        log_rho_6 = float(jnp.log10(jnp.array(rho_1e6)))
        comp_alpha3 = float(alpha3_marks_plane(jnp.array(log_rho_6), jnp.array(FeH)))
        computed.append(comp_alpha3)
        expected.append(exp_alpha3)
        names.append(name)
        edge_cases.append(is_edge_case)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 1:1 line
    ax.plot([0.5, 2.4], [0.5, 2.4], "k--", label="1:1 line", linewidth=1)

    # Scatter with error region (±0.15 tolerance band)
    ax.fill_between([0.5, 2.4], [0.35, 2.25], [0.65, 2.55], alpha=0.2, color="gray", label="$\\pm 0.15$ tolerance")

    # Plot GCs - separate normal GCs from edge cases
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(gc_data)))
    for i, (name, comp, exp, is_edge) in enumerate(zip(names, computed, expected, edge_cases)):
        if is_edge:
            # ω Cen - special marker and annotation
            ax.scatter(exp, comp, s=200, c="red", marker="s", edgecolors="k", linewidths=2.0, zorder=15,
                       label="ω Cen (edge case)")
            ax.annotate(
                f"{name}\n(ω Cen)",
                (exp + 0.05, comp + 0.05),
                fontsize=8,
                color="red",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="red", alpha=0.9),
            )
        else:
            ax.scatter(exp, comp, s=80, c=[colors[i]], edgecolors="k", linewidths=0.8, zorder=10)
            # Only annotate selected clusters to avoid crowding
            if name in ["NGC 7078", "NGC 104", "NGC 6341"]:  # M15, 47 Tuc, M92
                ax.annotate(name, (exp + 0.03, comp + 0.03), fontsize=7, alpha=0.8)

    ax.set_xlabel(r"$\alpha_3$ from Marks+2012 Table 1", fontsize=12)
    ax.set_ylabel(r"$\alpha_3$ computed (Fundamental Plane)", fontsize=12)
    ax.set_title("Validation: All 17 GCs from Marks+2012 Table 1", fontsize=14)
    ax.set_xlim(0.6, 2.4)
    ax.set_ylim(0.6, 2.4)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Compute statistics EXCLUDING edge case
    computed_arr = np.array(computed)
    expected_arr = np.array(expected)
    edge_mask = np.array(edge_cases)

    rmse_all = np.sqrt(np.mean((computed_arr - expected_arr) ** 2))
    rmse_excl = np.sqrt(np.mean((computed_arr[~edge_mask] - expected_arr[~edge_mask]) ** 2))
    max_err = np.max(np.abs(computed_arr[~edge_mask] - expected_arr[~edge_mask]))

    # Stats box
    stats_text = (
        f"All 17 GCs: RMSE = {rmse_all:.3f}\n"
        f"Excl. ω Cen (16 GCs): RMSE = {rmse_excl:.3f}\n"
        f"Max error (excl.): {max_err:.3f}"
    )
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    # ω Cen explanation box
    omega_cen_note = (
        "ω Cen EDGE CASE:\n"
        "• Lowest density in Table 1 (ρ = 2.35×10⁶)\n"
        "• Below threshold → formula gives canonical\n"
        "• Table 1 value from stellar counts, not formula\n"
        "• Unique formation history (stripped dSph?)"
    )
    ax.text(0.02, 0.98, omega_cen_note,
            transform=ax.transAxes, fontsize=7, ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="orange", alpha=0.9))

    plt.tight_layout()
    outpath = PLOT_DIR / "env_imf_gc_validation.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")

    # Print table
    print("\n  GC Validation Results (17 GCs from Marks+2012 Table 1):")
    print("  " + "-" * 60)
    print(f"  {'GC Name':<12} {'[Fe/H]':>8} {'Expected':>10} {'Computed':>10} {'Error':>8} {'Note':<10}")
    print("  " + "-" * 60)
    for (name, FeH, _, exp, is_edge), comp in zip(gc_data, computed):
        err = comp - exp
        note = "EDGE CASE" if is_edge else ""
        print(f"  {name:<12} {FeH:>8.2f} {exp:>10.2f} {comp:>10.2f} {err:>+8.2f} {note:<10}")
    print("  " + "-" * 60)
    print(f"  RMSE (all 17): {rmse_all:.4f}")
    print(f"  RMSE (excl. ω Cen): {rmse_excl:.4f}")


def plot_gradient_flow():
    """Plot 3: Gradient flow and α₃ vs environment.

    Shows:
    - Left: NLL landscape with gradient arrows
    - Right: α₃ vs cluster mass at BOTH solar and metal-poor metallicities
    """
    print("Creating gradient flow plot...")

    # Fixed observed masses (top-heavy sample)
    observed_masses = jnp.array([5.0, 10.0, 20.0, 30.0, 50.0, 80.0])

    # Compute NLL over log_mecl grid at fixed FeH
    log_mecl_range = jnp.linspace(3.0, 8.0, 100)
    FeH_mp = jnp.array(-1.5)  # Metal-poor

    def nll_fn(log_mecl):
        env = BirthEnvironment(metallicity=FeH_mp, log_mecl=log_mecl)
        params = env_to_imf_params(env, smooth_alpha3=True)
        return individual_mass_nll(observed_masses, params)

    nll_values = jax.vmap(nll_fn)(log_mecl_range)

    # Compute gradients at a few points
    grad_fn = jax.grad(nll_fn)
    gradient_points = jnp.array([4.0, 5.0, 6.0, 7.0])
    gradients = jax.vmap(grad_fn)(gradient_points)
    nll_at_points = jax.vmap(nll_fn)(gradient_points)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: NLL landscape
    ax1.plot(np.array(log_mecl_range), np.array(nll_values), "b-", linewidth=2)
    ax1.set_xlabel(r"$\log_{10}(M_{\rm ecl} / M_\odot)$", fontsize=12)
    ax1.set_ylabel("Negative Log-Likelihood", fontsize=12)
    ax1.set_title(f"NLL Landscape ([Fe/H] = {float(FeH_mp):.1f})", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Mark minimum
    min_idx = np.argmin(np.array(nll_values))
    ax1.axvline(float(log_mecl_range[min_idx]), color="r", linestyle="--", label=f"Minimum at {float(log_mecl_range[min_idx]):.1f}")
    ax1.legend()

    # Add gradient arrows
    scale = 0.5  # Arrow scale
    for pt, grad, nll in zip(gradient_points, gradients, nll_at_points):
        # Arrow points in -gradient direction (optimization direction)
        dx = -float(grad) * scale / (np.abs(float(grad)) + 1e-6)
        ax1.annotate("", xy=(float(pt) + dx, float(nll)), xytext=(float(pt), float(nll)),
                     arrowprops=dict(arrowstyle="->", color="red", lw=2))

    ax1.text(0.05, 0.95, "Arrows show\noptimization direction",
             transform=ax1.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Right: Alpha3 vs log_mecl for BOTH metallicities
    metallicities = [
        (0.0, "Solar [Fe/H]=0", "C0", "-"),
        (-1.5, "Metal-poor [Fe/H]=-1.5", "C1", "--"),
    ]

    for FeH, label, color, ls in metallicities:
        alpha3_values = []
        for lm in log_mecl_range:
            env = BirthEnvironment(metallicity=jnp.array(FeH), log_mecl=lm)
            params = env_to_imf_params(env, smooth_alpha3=True)
            alpha3_values.append(float(params.alpha3))
        ax2.plot(np.array(log_mecl_range), alpha3_values, color=color, ls=ls, linewidth=2, label=label)

    # Canonical Kroupa reference line
    ax2.axhline(2.3, color="gray", linestyle=":", linewidth=2, label=r"Canonical Kroupa ($\alpha_3=2.3$)")

    ax2.set_xlabel(r"$\log_{10}(M_{\rm ecl} / M_\odot)$", fontsize=12)
    ax2.set_ylabel(r"$\alpha_3$ (high-mass slope)", fontsize=12)
    ax2.set_title("IMF Slope vs Cluster Mass: Jeřábková+2018", fontsize=14)
    ax2.set_ylim(0.4, 2.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    # Shade "top-heavy" region (below canonical)
    ax2.axhspan(0.4, 2.3, alpha=0.08, color="red")
    ax2.text(3.5, 0.7, "Top-heavy\n(α₃ < 2.3)", fontsize=10, ha="center",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Key insight annotation - compute threshold from module coefficients (no hardcoding!)
    # At solar metallicity, x = logMecl_coeff * log_mecl_6 + constant < x_threshold
    # log_mecl_6 < (x_threshold - constant) / logMecl_coeff
    c = JERABKOVA_COEFFICIENTS
    log_mecl_6_threshold = (c["x_threshold"] - c["constant"]) / c["logMecl_coeff"]
    log_mecl_threshold = log_mecl_6_threshold + 6.0
    mecl_threshold = 10**log_mecl_threshold
    ax2.text(0.02, 0.02,
             f"At solar Z: canonical Kroupa when\n$M_{{\\rm ecl}} < {mecl_threshold:.0f}$ $M_\\odot$ ($\\log_{{10}} M < {log_mecl_threshold:.1f}$)",
             transform=ax2.transAxes, fontsize=9, va="bottom",
             bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.6))

    plt.tight_layout()
    outpath = PLOT_DIR / "env_imf_gradient_flow.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")


def plot_imf_shapes():
    """Plot 4: IMF PDF comparison across environments.

    Shows how the IMF shape changes with environment.
    Two panels:
    - Left: dN/dm (number density) - ALWAYS decreases with mass
    - Right: ξ(m) = m × dN/dm (mass per log interval) - can increase for top-heavy IMFs
    """
    print("Creating IMF shapes comparison plot...")

    # Different environments with appropriate models
    # (label, env, model)
    environments = [
        ("Solar neighborhood", BirthEnvironment.solar(), "jerabkova_generalized"),
        ("Massive GC ([Fe/H]=-1.5)", BirthEnvironment.massive_gc(FeH=-1.5), "jerabkova_generalized"),
        ("NGC 7078 (Marks+2012)", BirthEnvironment.ngc_7078(), "marks_plane"),  # From Table 1
        ("Universal Kroupa", None, None),  # Reference
    ]

    masses = jnp.logspace(-2, 2, 500)  # 0.01 to 100 M_sun

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["C0", "C1", "C2", "gray"]
    linestyles = ["-", "-", "-", "--"]

    for (label, env, model), color, ls in zip(environments, colors, linestyles):
        if env is None:
            # Universal Kroupa
            params = IMFParams.kroupa()
        else:
            params = env_to_imf_params(env, model=model)

        log_probs = log_prob_masses(masses, params)
        pdf = jnp.exp(log_probs)  # This is dN/dm (number density)

        # Left panel: dN/dm (number density) - ALWAYS decreases
        ax1.loglog(np.array(masses), np.array(pdf), color=color, ls=ls, linewidth=2, label=label)

        # Right panel: ξ(m) = m × dN/dm (mass per log interval)
        xi = pdf * masses
        ax2.loglog(np.array(masses), np.array(xi), color=color, ls=ls, linewidth=2, label=label)

    # Left panel formatting
    ax1.set_xlabel(r"Mass [$M_\odot$]", fontsize=12)
    ax1.set_ylabel(r"$dN/dm$ (number density)", fontsize=12)
    ax1.set_title("Number Density: Always Decreases", fontsize=14)
    ax1.set_xlim(0.01, 100)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.text(0.02, 0.02,
             "More low-mass stars by NUMBER\n(even for top-heavy IMFs)",
             transform=ax1.transAxes, fontsize=9, va="bottom",
             bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    # Right panel formatting
    ax2.set_xlabel(r"Mass [$M_\odot$]", fontsize=12)
    ax2.set_ylabel(r"$\xi(m) = m \cdot dN/dm$ (mass per log interval)", fontsize=12)
    ax2.set_title("Mass Budget: Can Increase for Top-Heavy IMFs", fontsize=14)
    ax2.set_xlim(0.01, 100)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.text(0.02, 0.02,
             "Mass budget dominated by massive stars\nwhen $\\alpha_3 < 1$",
             transform=ax2.transAxes, fontsize=9, va="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Add break points to both panels
    for ax in [ax1, ax2]:
        ax.axvline(0.08, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    outpath = PLOT_DIR / "env_imf_shapes.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")


def print_summary():
    """Print coefficient summary and validation results."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT-DEPENDENT IMF: PAPER COEFFICIENTS")
    print("=" * 60)

    print("\nJerabkova+2018 Eq. 9 (RECOMMENDED):")
    for k, v in JERABKOVA_COEFFICIENTS.items():
        print(f"  {k}: {v}")

    print("\nMarks+2012 Fundamental Plane:")
    for k, v in MARKS_COEFFICIENTS.items():
        print(f"  {k}: {v}")

    print("\nCRITICAL THRESHOLDS:")
    print(f"  Jerabkova: x >= {JERABKOVA_COEFFICIENTS['x_threshold']} (NEGATIVE)")
    print(f"  Marks: x_hat >= {MARKS_COEFFICIENTS['x_hat_threshold']} (POSITIVE)")

    print("\n" + "=" * 60)


def main():
    """Generate all validation plots."""
    print("=" * 60)
    print("Environment-Dependent IMF Validation")
    print("=" * 60)

    plot_alpha3_surface()
    plot_gc_validation()
    plot_gradient_flow()
    plot_imf_shapes()

    print_summary()

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print(f"All plots saved to: {PLOT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
