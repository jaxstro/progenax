#!/usr/bin/env python
"""Validation script comparing BM19 vs PN11/FK12 frameworks.

This script demonstrates the differences between:
- BM19: Modern piecewise lognormal+powerlaw framework
- PN11/FK12: Classical pure lognormal with s_crit formula

Key differences shown:
1. Different threshold formulas (s_t vs s_crit)
2. Different f_dense predictions
3. Parameter count (BM19 has fewer free parameters)

Run with:
    python validate_bm19_vs_pn11.py
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = Path(__file__).parent / "plots"

from progenax.gravoturb import bm19_model as bm19, legacy_pn11


def compare_thresholds():
    """Compare BM19 s_t vs PN11 s_crit formulas."""
    print("\n" + "=" * 70)
    print("1. THRESHOLD FORMULA COMPARISON")
    print("=" * 70)
    print("\nBM19 s_t = (α - 0.5) × σ_s²   [derived from PDF continuity]")
    print("PN11 s_crit = ln((π²φ_x²/5) × α_vir × M²)   [parameterized]")

    # Parameters
    machs = [5.0, 10.0, 15.0, 20.0]
    alpha = 2.0  # BM19 powerlaw slope
    b = 0.4

    # PN11 parameters
    Sigma = 100.0  # M☉/pc²
    phi_x = 0.35
    alpha_0 = 2.0
    Sigma_0 = 85.0

    print(f"\nBM19: α={alpha}, b={b}")
    print(f"PN11: Σ={Sigma} M☉/pc², φ_x={phi_x}, α₀={alpha_0}, Σ₀={Sigma_0}")
    print("-" * 70)
    print(f"{'Mach':>6} | {'σ_s²':>8} | {'s_t (BM19)':>12} | {'s_crit (PN11)':>14} | {'Difference':>12}")
    print("-" * 70)

    for mach in machs:
        sigma_sq = float(bm19.sigma_s_squared(mach, b))
        s_t = float(bm19.transition_density(sigma_sq, alpha))

        alpha_vir = float(legacy_pn11.alpha_vir_from_sigma(Sigma, alpha_0, Sigma_0))
        s_crit = float(legacy_pn11.s_crit_pn11(mach, alpha_vir, phi_x))

        diff = s_t - s_crit
        print(f"{mach:>6.0f} | {sigma_sq:>8.3f} | {s_t:>12.3f} | {s_crit:>14.3f} | {diff:>+12.3f}")

    print("-" * 70)
    print("\nKey insight: BM19 threshold scales with σ_s² (from PDF continuity)")
    print("             PN11 threshold has independent φ_x, α_vir parameters")


def compare_f_dense():
    """Compare BM19 full integral vs PN11 lognormal erfc."""
    print("\n" + "=" * 70)
    print("2. f_dense PREDICTION COMPARISON")
    print("=" * 70)

    machs = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    alpha = 2.0
    b = 0.4

    # PN11 parameters
    Sigma = 100.0
    phi_x = 0.35
    alpha_0 = 2.0
    Sigma_0 = 85.0
    alpha_vir = float(legacy_pn11.alpha_vir_from_sigma(Sigma, alpha_0, Sigma_0))

    print(f"\nBM19: α={alpha}, b={b}")
    print(f"PN11: Σ={Sigma}, φ_x={phi_x}, α_vir={alpha_vir:.2f}")
    print("-" * 70)
    print(f"{'Mach':>6} | {'f_dense (BM19)':>14} | {'f_dense (PN11)':>14} | {'Ratio':>10}")
    print("-" * 70)

    bm19_values = []
    pn11_values = []

    for mach in machs:
        # BM19
        sigma_sq = float(bm19.sigma_s_squared(mach, b))
        s_t = float(bm19.transition_density(sigma_sq, alpha))
        f_bm19 = float(bm19.f_dense_bm19_full(sigma_sq, s_t, alpha))
        bm19_values.append(f_bm19)

        # PN11
        s_crit = float(legacy_pn11.s_crit_pn11(mach, alpha_vir, phi_x))
        f_pn11 = float(legacy_pn11.f_dense_pn11(sigma_sq, s_crit))
        pn11_values.append(f_pn11)

        ratio = f_bm19 / f_pn11
        print(f"{mach:>6.0f} | {f_bm19:>14.5f} | {f_pn11:>14.5f} | {ratio:>10.2f}")

    print("-" * 70)

    # Check trend consistency
    bm19_monotonic = all(bm19_values[i] > bm19_values[i + 1] for i in range(len(bm19_values) - 1))
    pn11_monotonic = all(pn11_values[i] > pn11_values[i + 1] for i in range(len(pn11_values) - 1))

    print(f"\nBM19 monotonic (f_dense ↓ with Mach): {'Yes ✓' if bm19_monotonic else 'No ✗'}")
    print(f"PN11 monotonic (f_dense ↓ with Mach): {'Yes ✓' if pn11_monotonic else 'No ✗'}")

    return machs, bm19_values, pn11_values


def compare_parameter_count():
    """Compare free parameter count between frameworks."""
    print("\n" + "=" * 70)
    print("3. PARAMETER COUNT COMPARISON")
    print("=" * 70)

    print("\nBM19 Pipeline: (M, b, α, η) → f_sub")
    print("  - M: Mach number")
    print("  - b: Turbulence driving parameter (0.4 default)")
    print("  - α: Powerlaw slope (1.5-3.0)")
    print("  - η: Feedback survival efficiency")
    print("  Total: 4 parameters (b often fixed)")

    print("\nPN11/FK12 Pipeline: (M, b, Σ, η, φ_x, α₀, Σ₀) → f_sub")
    print("  - M: Mach number")
    print("  - b: Turbulence driving parameter")
    print("  - Σ: Surface density")
    print("  - η: Feedback survival efficiency")
    print("  - φ_x: Sonic scale factor (magnetic support)")
    print("  - α₀: Reference virial parameter")
    print("  - Σ₀: Reference surface density")
    print("  Total: 7 parameters")

    print("\nKey advantage: BM19 derives threshold from PDF continuity,")
    print("               reducing free parameters and physical assumptions.")


def generate_comparison_plot(machs, bm19_values, pn11_values):
    """Generate comparison plots."""
    print("\n" + "=" * 70)
    print("4. GENERATING COMPARISON PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: f_dense comparison at fixed environment
    ax1 = axes[0]
    ax1.semilogy(machs, bm19_values, "b-o", linewidth=2, markersize=8, label="BM19 (LN+PL)")
    ax1.semilogy(machs, pn11_values, "r--s", linewidth=2, markersize=8, label="PN11 (pure LN)")
    ax1.set_xlabel("Mach Number", fontsize=12)
    ax1.set_ylabel("f_dense", fontsize=12)
    ax1.set_title("BM19 vs PN11: f_dense at Σ=100 M☉/pc²", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Ratio vs Mach
    ax2 = axes[1]
    ratios = [bm19_values[i] / pn11_values[i] for i in range(len(machs))]
    ax2.plot(machs, ratios, "g-o", linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Mach Number", fontsize=12)
    ax2.set_ylabel("f_dense(BM19) / f_dense(PN11)", fontsize=12)
    ax2.set_title("BM19/PN11 Ratio", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / "bm19_vs_pn11_comparison.png"), dpi=150)
    print(f"  Saved: {PLOT_DIR / 'bm19_vs_pn11_comparison.png'}")
    plt.close()


def generate_surface_density_sweep():
    """Show how PN11 varies with Σ while BM19 doesn't depend on it."""
    print("\n" + "=" * 70)
    print("5. SURFACE DENSITY DEPENDENCE")
    print("=" * 70)
    print("\nBM19: f_dense depends ONLY on (M, b, α) - no Σ dependence")
    print("PN11: f_dense depends on Σ via α_vir = α₀(Σ₀/Σ)")

    mach = 10.0
    alpha = 2.0
    b = 0.4
    sigmas = [50.0, 100.0, 200.0, 500.0, 1000.0]

    # BM19 (constant)
    sigma_sq = float(bm19.sigma_s_squared(mach, b))
    s_t = float(bm19.transition_density(sigma_sq, alpha))
    f_bm19 = float(bm19.f_dense_bm19_full(sigma_sq, s_t, alpha))

    print(f"\nMach={mach}, α={alpha}, b={b}")
    print("-" * 60)
    print(f"{'Σ [M☉/pc²]':>12} | {'α_vir':>8} | {'f_dense (PN11)':>14} | {'f_dense (BM19)':>14}")
    print("-" * 60)

    pn11_values = []
    for Sigma in sigmas:
        alpha_vir = float(legacy_pn11.alpha_vir_from_sigma(Sigma))
        s_crit = float(legacy_pn11.s_crit_pn11(mach, alpha_vir, 0.35))
        f_pn11 = float(legacy_pn11.f_dense_pn11(sigma_sq, s_crit))
        pn11_values.append(f_pn11)

        print(f"{Sigma:>12.0f} | {alpha_vir:>8.3f} | {f_pn11:>14.5f} | {f_bm19:>14.5f}")

    print("-" * 60)
    print(f"\nBM19 f_dense is CONSTANT: {f_bm19:.5f} (no Σ dependence)")
    print(f"PN11 f_dense varies from {min(pn11_values):.5f} to {max(pn11_values):.5f}")

    # Generate plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(sigmas, pn11_values, "r-o", linewidth=2, markersize=8, label="PN11")
    ax.axhline(y=f_bm19, color="b", linestyle="-", linewidth=2, label="BM19 (constant)")
    ax.set_xlabel("Surface Density Σ [M☉/pc²]", fontsize=12)
    ax.set_ylabel("f_dense", fontsize=12)
    ax.set_title("Surface Density Dependence: BM19 vs PN11\n(M=10, α=2.0)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / "bm19_vs_pn11_sigma_dependence.png"), dpi=150)
    print(f"  Saved: {PLOT_DIR / 'bm19_vs_pn11_sigma_dependence.png'}")
    plt.close()


def main():
    """Run all comparison tests."""
    print("=" * 70)
    print("BM19 vs PN11/FK12 FRAMEWORK COMPARISON")
    print("=" * 70)
    print("Comparing modern BM19 piecewise framework to classical PN11 approach")

    # Run comparisons
    compare_thresholds()
    machs, bm19_values, pn11_values = compare_f_dense()
    compare_parameter_count()
    generate_comparison_plot(machs, bm19_values, pn11_values)
    generate_surface_density_sweep()

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print("\n1. THRESHOLD FORMULAS:")
    print("   BM19: s_t = (α - 0.5) × σ_s² (derived from PDF continuity)")
    print("   PN11: s_crit = ln((π²φ_x²/5) × α_vir × M²) (parameterized)")

    print("\n2. f_dense PREDICTIONS:")
    print("   BM19: Piecewise LN+PL integral (full tail contribution)")
    print("   PN11: Pure lognormal erfc (ignores powerlaw tail)")
    print("   BM19 generally predicts LOWER f_dense than PN11")

    print("\n3. PARAMETER COUNT:")
    print("   BM19: 4 parameters (M, b, α, η)")
    print("   PN11: 7 parameters (M, b, Σ, η, φ_x, α₀, Σ₀)")
    print("   BM19 is simpler and more physically motivated")

    print("\n4. SURFACE DENSITY DEPENDENCE:")
    print("   BM19: f_dense independent of Σ (only M, α matter)")
    print("   PN11: f_dense varies with Σ via α_vir")
    print("   BM19 is more predictive for diverse environments")

    print("\n5. PLOTS GENERATED:")
    print(f"   - {PLOT_DIR / 'bm19_vs_pn11_comparison.png'}")
    print(f"   - {PLOT_DIR / 'bm19_vs_pn11_sigma_dependence.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
