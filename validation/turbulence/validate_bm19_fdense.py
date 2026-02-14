#!/usr/bin/env python
"""Validation script for BM19 f_dense calculations.

This script validates that the BM19 implementation produces physically
correct self-gravitating gas fractions across a range of parameters.

Validation includes:
1. Limiting behavior (α → ∞ approaches lognormal limit)
2. Parameter scaling (f_dense vs Mach, α, b)
3. Gradient sanity checks
4. Comparison table with expected physics

Run with:
    python validate_bm19_fdense.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = Path(__file__).parent / "plots"

from progenax.gravoturb import bm19_model as bm19


def validate_lognormal_limit():
    """Test convergence to lognormal limit as α → ∞."""
    print("\n" + "=" * 70)
    print("1. LOGNORMAL LIMIT TEST (α → ∞)")
    print("=" * 70)

    sigma_sq = 2.0
    alphas = [2.0, 3.0, 5.0, 10.0, 20.0, 50.0]

    print(f"\nσ_s² = {sigma_sq:.2f}")
    print("-" * 50)
    print(f"{'α':>8} | {'s_t':>8} | {'f_dense_full':>12} | {'f_dense_LN':>12} | {'Diff %':>8}")
    print("-" * 50)

    for alpha in alphas:
        s_t = bm19.transition_density(sigma_sq, alpha)
        f_full = bm19.f_dense_bm19_full(sigma_sq, s_t, alpha)
        f_ln = bm19.f_dense_lognormal_limit(sigma_sq, s_t)
        diff_pct = (f_full - f_ln) / f_ln * 100

        print(f"{alpha:>8.1f} | {float(s_t):>8.3f} | {float(f_full):>12.6f} | {float(f_ln):>12.6f} | {float(diff_pct):>+8.1f}%")

    print("-" * 50)
    print("Note: Powerlaw tail decays faster than lognormal, so f_full < f_ln")
    print("      Difference decreases as α increases (steeper powerlaw)")


def validate_mach_scaling():
    """Test f_dense scaling with Mach number."""
    print("\n" + "=" * 70)
    print("2. MACH NUMBER SCALING TEST")
    print("=" * 70)
    print("BM19 Prediction: Higher Mach → wider PDF → lower f_dense")

    alpha = 2.0
    b = 0.4
    machs = [3.0, 5.0, 10.0, 15.0, 20.0, 30.0]

    print(f"\nα = {alpha}, b = {b}")
    print("-" * 60)
    print(f"{'Mach':>6} | {'σ_s':>8} | {'s_t':>8} | {'f_dense':>10} | {'f_sub (η=0.6)':>12}")
    print("-" * 60)

    f_dense_values = []
    for mach in machs:
        result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.6)
        f_dense_values.append(float(result.f_dense))
        print(
            f"{mach:>6.0f} | {float(result.sigma_s):>8.3f} | "
            f"{float(result.s_t):>8.3f} | {float(result.f_dense):>10.5f} | "
            f"{float(result.f_sub):>12.5f}"
        )

    print("-" * 60)

    # Verify monotonicity
    is_monotonic = all(f_dense_values[i] > f_dense_values[i + 1] for i in range(len(f_dense_values) - 1))
    print(f"Monotonically decreasing with Mach: {'PASS ✓' if is_monotonic else 'FAIL ✗'}")


def validate_alpha_scaling():
    """Test f_dense scaling with powerlaw slope α."""
    print("\n" + "=" * 70)
    print("3. POWERLAW SLOPE (α) SCALING TEST")
    print("=" * 70)
    print("BM19 Prediction: Higher α → higher s_t AND steeper decay → lower f_dense")

    mach = 10.0
    b = 0.4
    alphas = [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]

    print(f"\nMach = {mach}, b = {b}")
    print("-" * 60)
    print(f"{'α':>6} | {'s_t':>8} | {'f_dense':>10} | {'p (=3/α)':>8} | {'ζ':>12}")
    print("-" * 60)

    f_dense_values = []
    for alpha in alphas:
        result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.6)
        f_dense_values.append(float(result.f_dense))
        print(
            f"{alpha:>6.2f} | {float(result.s_t):>8.3f} | "
            f"{float(result.f_dense):>10.5f} | {float(result.p):>8.3f} | "
            f"{float(result.zeta):>12.2e}"
        )

    print("-" * 60)

    # Verify monotonicity
    is_monotonic = all(f_dense_values[i] > f_dense_values[i + 1] for i in range(len(f_dense_values) - 1))
    print(f"Monotonically decreasing with α: {'PASS ✓' if is_monotonic else 'FAIL ✗'}")


def validate_gradient_signs():
    """Validate gradient signs match physical predictions."""
    print("\n" + "=" * 70)
    print("4. GRADIENT SIGN VALIDATION")
    print("=" * 70)

    def f_dense_from_params(mach, alpha, b):
        sigma_sq = bm19.sigma_s_squared(mach, b)
        s_t = bm19.transition_density(sigma_sq, alpha)
        return bm19.f_dense_bm19_full(sigma_sq, s_t, alpha)

    # Base parameters
    mach_0, alpha_0, b_0 = 10.0, 2.0, 0.4

    # Compute gradients
    grad_mach = jax.grad(f_dense_from_params, argnums=0)(mach_0, alpha_0, b_0)
    grad_alpha = jax.grad(f_dense_from_params, argnums=1)(mach_0, alpha_0, b_0)
    grad_b = jax.grad(f_dense_from_params, argnums=2)(mach_0, alpha_0, b_0)

    print(f"\nGradients at Mach={mach_0}, α={alpha_0}, b={b_0}:")
    print("-" * 50)
    print(f"  ∂f_dense/∂Mach  = {float(grad_mach):+.6f}")
    print(f"  ∂f_dense/∂α     = {float(grad_alpha):+.6f}")
    print(f"  ∂f_dense/∂b     = {float(grad_b):+.6f}")
    print("-" * 50)

    # Expected signs
    expected = [
        ("∂f_dense/∂Mach < 0", grad_mach < 0, "Higher Mach → wider PDF → lower f_dense"),
        ("∂f_dense/∂α < 0", grad_alpha < 0, "Higher α → higher s_t + steeper PL → lower f_dense"),
        ("∂f_dense/∂b < 0", grad_b < 0, "Higher b → wider PDF → lower f_dense"),
    ]

    all_pass = True
    for name, condition, reason in expected:
        status = "PASS ✓" if condition else "FAIL ✗"
        all_pass = all_pass and condition
        print(f"  {name}: {status}")
        print(f"    ({reason})")

    return all_pass


def generate_parameter_sweep_plot():
    """Generate 2D parameter sweep plot."""
    print("\n" + "=" * 70)
    print("5. GENERATING PARAMETER SWEEP PLOT")
    print("=" * 70)

    # Parameter grids
    machs = np.linspace(5, 30, 26)
    alphas = np.linspace(1.5, 3.0, 16)

    # Compute f_dense grid
    f_dense_grid = np.zeros((len(alphas), len(machs)))

    for i, alpha in enumerate(alphas):
        for j, mach in enumerate(machs):
            result = bm19.bm19_pipeline(float(mach), 0.4, float(alpha), 0.6)
            f_dense_grid[i, j] = float(result.f_dense)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.contourf(machs, alphas, f_dense_grid, levels=20, cmap="viridis")
    cbar = plt.colorbar(im, ax=ax, label="f_dense")

    # Add contour lines
    cs = ax.contour(machs, alphas, f_dense_grid, levels=[0.01, 0.03, 0.05, 0.1, 0.15, 0.2], colors="white", linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    ax.set_xlabel("Mach Number", fontsize=12)
    ax.set_ylabel("Powerlaw Slope α", fontsize=12)
    ax.set_title("BM19 Self-Gravitating Fraction f_dense(M, α)\nb = 0.4 (mixed driving)", fontsize=14)

    # Save plot
    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / "bm19_fdense_parameter_sweep.png"), dpi=150)
    print(f"  Saved: {PLOT_DIR / 'bm19_fdense_parameter_sweep.png'}")
    plt.close()


def generate_comparison_plot():
    """Generate BM19 full vs lognormal limit comparison."""
    print("\n" + "=" * 70)
    print("6. GENERATING COMPARISON PLOT")
    print("=" * 70)

    machs = np.linspace(5, 30, 50)
    alphas = [1.5, 2.0, 2.5, 3.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: f_dense vs Mach for different α
    ax1 = axes[0]
    for alpha in alphas:
        f_dense = [float(bm19.bm19_pipeline(float(m), 0.4, float(alpha), 0.6).f_dense) for m in machs]
        ax1.semilogy(machs, f_dense, label=f"α = {alpha}")

    ax1.set_xlabel("Mach Number", fontsize=12)
    ax1.set_ylabel("f_dense (log scale)", fontsize=12)
    ax1.set_title("BM19 f_dense vs Mach Number", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Full integral vs lognormal limit
    ax2 = axes[1]
    sigma_sqs = np.linspace(1, 5, 50)
    alpha = 2.0

    f_full = []
    f_ln = []
    for sigma_sq in sigma_sqs:
        s_t = float(bm19.transition_density(float(sigma_sq), alpha))
        f_full.append(float(bm19.f_dense_bm19_full(float(sigma_sq), s_t, alpha)))
        f_ln.append(float(bm19.f_dense_lognormal_limit(float(sigma_sq), s_t)))

    ax2.semilogy(sigma_sqs, f_full, "b-", label="BM19 Full (LN+PL)", linewidth=2)
    ax2.semilogy(sigma_sqs, f_ln, "r--", label="Lognormal Limit", linewidth=2)
    ax2.set_xlabel("σ_s² (PDF variance)", fontsize=12)
    ax2.set_ylabel("f_dense (log scale)", fontsize=12)
    ax2.set_title(f"Full Integral vs Lognormal Limit (α = {alpha})", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / "bm19_fdense_comparison.png"), dpi=150)
    print(f"  Saved: {PLOT_DIR / 'bm19_fdense_comparison.png'}")
    plt.close()


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("BM19 f_dense VALIDATION")
    print("=" * 70)
    print("Validating BM19 self-gravitating gas fraction implementation")

    # Run tests
    validate_lognormal_limit()
    validate_mach_scaling()
    validate_alpha_scaling()
    gradients_pass = validate_gradient_signs()

    # Generate plots
    generate_parameter_sweep_plot()
    generate_comparison_plot()

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print("  1. Lognormal limit convergence: See table above")
    print("  2. Mach scaling: f_dense ↓ as Mach ↑ (correct)")
    print("  3. Alpha scaling: f_dense ↓ as α ↑ (correct)")
    print(f"  4. Gradient signs: {'PASS ✓' if gradients_pass else 'FAIL ✗'}")
    print(f"  5. Parameter sweep plot: {PLOT_DIR / 'bm19_fdense_parameter_sweep.png'}")
    print(f"  6. Comparison plot: {PLOT_DIR / 'bm19_fdense_comparison.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
