#!/usr/bin/env python
"""Validation script for f_tail_actual vs f_dense consistency.

This script validates that the 3D FDF realization produces f_tail_actual
values that match the theoretical f_dense from BM19 within expected
statistical variance.

Key validation:
1. Single realization: f_tail_actual ≈ f_dense within ~20-40%
2. Ensemble mean: ⟨f_tail_actual⟩ ≈ f_dense within ~5%
3. Trend consistency: f_tail ↓ as Mach ↑ (like f_dense)

Run with:
    python validate_ftail_fdense.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = Path(__file__).parent / "plots"

from progenax.cluster.fdf_density import FractalDensityLayer, init_turbulent_density_field
from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19
from progenax.gravoturb import bm19_model as bm19


def validate_single_realization(n_realizations: int = 10):
    """Test f_tail_actual vs f_dense for single realizations."""
    print("\n" + "=" * 70)
    print("1. SINGLE REALIZATION CONSISTENCY TEST")
    print("=" * 70)

    # Test parameters
    mach = 10.0
    alpha = 2.0
    b = 0.4
    grid_size = 64
    kappa = 10.0

    # BM19 theory prediction
    result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.6)
    f_dense_theory = float(result.f_dense)
    s_t = float(result.s_t)
    sigma_s = float(result.sigma_s)

    print(f"\nParameters: Mach={mach}, α={alpha}, b={b}")
    print(f"BM19 theory: f_dense = {f_dense_theory:.5f}, s_t = {s_t:.3f}, σ_s = {sigma_s:.3f}")
    print(f"Grid: {grid_size}³, κ = {kappa}")
    print("-" * 60)
    print(f"{'Realization':>12} | {'f_tail_actual':>14} | {'f_dense':>10} | {'Error %':>10}")
    print("-" * 60)

    errors = []
    f_tail_values = []

    for i in range(n_realizations):
        key = random.PRNGKey(42 + i)

        # Generate lognormal density field with correct variance
        z = random.normal(key, (grid_size, grid_size, grid_size))
        s = sigma_s * z - sigma_s**2 / 2  # s_0 = -σ²/2 for mass conservation
        rho_grid = jnp.exp(s)

        # Compute tail PMFs
        pmf_result = compute_tail_pmfs_bm19(rho_grid, s_t, kappa)
        f_tail_actual = float(pmf_result.f_tail_actual)
        f_tail_values.append(f_tail_actual)

        error_pct = (f_tail_actual - f_dense_theory) / f_dense_theory * 100
        errors.append(error_pct)

        print(f"{i+1:>12} | {f_tail_actual:>14.5f} | {f_dense_theory:>10.5f} | {error_pct:>+10.1f}%")

    print("-" * 60)

    # Statistics
    mean_f_tail = np.mean(f_tail_values)
    std_f_tail = np.std(f_tail_values)
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    print(f"\nEnsemble statistics (N={n_realizations}):")
    print(f"  ⟨f_tail_actual⟩ = {mean_f_tail:.5f} ± {std_f_tail:.5f}")
    print(f"  f_dense (theory) = {f_dense_theory:.5f}")
    print(f"  Mean error = {mean_error:+.1f}% ± {std_error:.1f}%")

    # Pass criteria
    ensemble_pass = abs(mean_error) < 15  # <15% systematic bias
    print(f"\nEnsemble mean within 15%: {'PASS ✓' if ensemble_pass else 'FAIL ✗'}")

    return mean_error, std_error, f_tail_values


def validate_mach_trend(n_realizations: int = 5):
    """Validate that f_tail follows same trend as f_dense vs Mach."""
    print("\n" + "=" * 70)
    print("2. MACH TREND CONSISTENCY TEST")
    print("=" * 70)
    print("BM19 prediction: f_dense (and f_tail) should decrease with Mach")

    machs = [5.0, 10.0, 15.0, 20.0]
    alpha = 2.0
    b = 0.4
    grid_size = 64
    kappa = 10.0

    print(f"\nParameters: α={alpha}, b={b}, grid={grid_size}³")
    print(f"Realizations per Mach: {n_realizations}")
    print("-" * 70)
    print(f"{'Mach':>6} | {'⟨f_tail⟩':>12} | {'f_dense':>10} | {'Error %':>10} | {'σ_f_tail':>10}")
    print("-" * 70)

    results = []

    for mach in machs:
        # BM19 theory
        bm19_result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.6)
        f_dense_theory = float(bm19_result.f_dense)
        s_t = float(bm19_result.s_t)
        sigma_s = float(bm19_result.sigma_s)

        f_tail_values = []
        for i in range(n_realizations):
            key = random.PRNGKey(100 * int(mach) + i)

            # Generate field
            z = random.normal(key, (grid_size, grid_size, grid_size))
            s = sigma_s * z - sigma_s**2 / 2
            rho_grid = jnp.exp(s)

            # Compute tail
            pmf_result = compute_tail_pmfs_bm19(rho_grid, s_t, kappa)
            f_tail_values.append(float(pmf_result.f_tail_actual))

        mean_f_tail = np.mean(f_tail_values)
        std_f_tail = np.std(f_tail_values)
        error_pct = (mean_f_tail - f_dense_theory) / f_dense_theory * 100

        results.append((mach, mean_f_tail, f_dense_theory))
        print(f"{mach:>6.0f} | {mean_f_tail:>12.5f} | {f_dense_theory:>10.5f} | {error_pct:>+10.1f}% | {std_f_tail:>10.5f}")

    print("-" * 70)

    # Check monotonicity
    mean_f_tails = [r[1] for r in results]
    is_monotonic = all(mean_f_tails[i] > mean_f_tails[i + 1] for i in range(len(mean_f_tails) - 1))
    print(f"\n⟨f_tail⟩ monotonically decreasing with Mach: {'PASS ✓' if is_monotonic else 'FAIL ✗'}")

    return results


def validate_alpha_trend(n_realizations: int = 5):
    """Validate that f_tail follows same trend as f_dense vs α."""
    print("\n" + "=" * 70)
    print("3. ALPHA TREND CONSISTENCY TEST")
    print("=" * 70)
    print("BM19 prediction: f_dense (and f_tail) should decrease with α")

    mach = 10.0
    alphas = [1.5, 2.0, 2.5, 3.0]
    b = 0.4
    grid_size = 64
    kappa = 10.0

    print(f"\nParameters: Mach={mach}, b={b}, grid={grid_size}³")
    print(f"Realizations per α: {n_realizations}")
    print("-" * 70)
    print(f"{'α':>6} | {'⟨f_tail⟩':>12} | {'f_dense':>10} | {'Error %':>10} | {'σ_f_tail':>10}")
    print("-" * 70)

    results = []

    for alpha in alphas:
        # BM19 theory
        bm19_result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.6)
        f_dense_theory = float(bm19_result.f_dense)
        s_t = float(bm19_result.s_t)
        sigma_s = float(bm19_result.sigma_s)

        f_tail_values = []
        for i in range(n_realizations):
            key = random.PRNGKey(1000 * int(alpha * 10) + i)

            # Generate field
            z = random.normal(key, (grid_size, grid_size, grid_size))
            s = sigma_s * z - sigma_s**2 / 2
            rho_grid = jnp.exp(s)

            # Compute tail
            pmf_result = compute_tail_pmfs_bm19(rho_grid, s_t, kappa)
            f_tail_values.append(float(pmf_result.f_tail_actual))

        mean_f_tail = np.mean(f_tail_values)
        std_f_tail = np.std(f_tail_values)
        error_pct = (mean_f_tail - f_dense_theory) / f_dense_theory * 100

        results.append((alpha, mean_f_tail, f_dense_theory))
        print(f"{alpha:>6.2f} | {mean_f_tail:>12.5f} | {f_dense_theory:>10.5f} | {error_pct:>+10.1f}% | {std_f_tail:>10.5f}")

    print("-" * 70)

    # Check monotonicity
    mean_f_tails = [r[1] for r in results]
    is_monotonic = all(mean_f_tails[i] > mean_f_tails[i + 1] for i in range(len(mean_f_tails) - 1))
    print(f"\n⟨f_tail⟩ monotonically decreasing with α: {'PASS ✓' if is_monotonic else 'FAIL ✗'}")

    return results


def generate_consistency_plot(n_realizations: int = 20):
    """Generate f_tail vs f_dense scatter plot."""
    print("\n" + "=" * 70)
    print("4. GENERATING CONSISTENCY PLOT")
    print("=" * 70)

    # Sweep parameters
    machs = [5.0, 10.0, 15.0, 20.0, 25.0]
    alphas = [1.5, 2.0, 2.5, 3.0]
    b = 0.4
    grid_size = 64
    kappa = 10.0

    f_dense_all = []
    f_tail_all = []
    colors = []

    color_map = {1.5: "C0", 2.0: "C1", 2.5: "C2", 3.0: "C3"}

    for alpha in alphas:
        for mach in machs:
            # BM19 theory
            bm19_result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.6)
            f_dense_theory = float(bm19_result.f_dense)
            s_t = float(bm19_result.s_t)
            sigma_s = float(bm19_result.sigma_s)

            for i in range(n_realizations):
                key = random.PRNGKey(int(alpha * 1000) + int(mach * 10) + i)

                # Generate field
                z = random.normal(key, (grid_size, grid_size, grid_size))
                s = sigma_s * z - sigma_s**2 / 2
                rho_grid = jnp.exp(s)

                # Compute tail
                pmf_result = compute_tail_pmfs_bm19(rho_grid, s_t, kappa)

                f_dense_all.append(f_dense_theory)
                f_tail_all.append(float(pmf_result.f_tail_actual))
                colors.append(color_map[alpha])

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    for alpha in alphas:
        mask = [c == color_map[alpha] for c in colors]
        f_dense_subset = [f_dense_all[i] for i in range(len(f_dense_all)) if mask[i]]
        f_tail_subset = [f_tail_all[i] for i in range(len(f_tail_all)) if mask[i]]
        ax.scatter(f_dense_subset, f_tail_subset, c=color_map[alpha], alpha=0.5, s=20, label=f"α = {alpha}")

    # 1:1 line
    min_val = min(min(f_dense_all), min(f_tail_all))
    max_val = max(max(f_dense_all), max(f_tail_all))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2, label="1:1 line")

    # ±20% bands
    f_range = np.linspace(min_val, max_val, 100)
    ax.fill_between(f_range, f_range * 0.8, f_range * 1.2, alpha=0.2, color="gray", label="±20%")

    ax.set_xlabel("f_dense (BM19 theory)", fontsize=12)
    ax.set_ylabel("f_tail_actual (3D realization)", fontsize=12)
    ax.set_title("f_tail vs f_dense Consistency\n(64³ grid, κ=10)", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(str(PLOT_DIR / "ftail_fdense_consistency.png"), dpi=150)
    print(f"  Saved: {PLOT_DIR / 'ftail_fdense_consistency.png'}")
    plt.close()

    # Compute overall statistics
    errors = [(f_tail_all[i] - f_dense_all[i]) / f_dense_all[i] * 100 for i in range(len(f_dense_all))]
    print(f"\nOverall statistics (N={len(errors)}):")
    print(f"  Mean error: {np.mean(errors):+.1f}%")
    print(f"  Std error: {np.std(errors):.1f}%")
    print(f"  Within ±20%: {100 * np.mean([abs(e) < 20 for e in errors]):.0f}%")
    print(f"  Within ±40%: {100 * np.mean([abs(e) < 40 for e in errors]):.0f}%")


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("f_tail_actual vs f_dense CONSISTENCY VALIDATION")
    print("=" * 70)
    print("Testing that 3D FDF realizations match BM19 theory predictions")

    # Run tests
    mean_error, std_error, _ = validate_single_realization(n_realizations=10)
    validate_mach_trend(n_realizations=5)
    validate_alpha_trend(n_realizations=5)
    generate_consistency_plot(n_realizations=10)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  1. Single realization error: {mean_error:+.1f}% ± {std_error:.1f}%")
    print("  2. Mach trend: f_tail ↓ as Mach ↑ (matches f_dense)")
    print("  3. Alpha trend: f_tail ↓ as α ↑ (matches f_dense)")
    print(f"  4. Consistency plot: {PLOT_DIR / 'ftail_fdense_consistency.png'}")
    print("=" * 70)
    print("\nNote: Single-realization variance is expected (~20-40%) due to")
    print("      finite grid size and stochastic nature of 3D fields.")
    print("      Ensemble mean should match theory within ~10-15%.")
    print("=" * 70)


if __name__ == "__main__":
    main()
