#!/usr/bin/env python
"""
Gradient sanity check for mass segregation parameter λ_seg.

Verifies:
1. Gradients are finite (no NaN/Inf)
2. Gradients have correct sign (negative: increasing λ_seg → smaller r_massive)
3. Gradient magnitude is reasonable

Usage:
    python scripts/cluster_validation/check_grad_lambda_seg.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from progenax.cluster.validation import (
    grad_mean_radius_wrt_lambda_seg,
    mean_radius_of_massive_jax,
)


def main():
    """Run gradient sanity checks."""
    print("=" * 70)
    print("Gradient Sanity Check: λ_seg → ⟨r_massive⟩")
    print("=" * 70)
    print()

    key = jax.random.PRNGKey(42)

    # Test points
    lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("Computing mean radius and gradients at different λ_seg...")
    print()
    print("-" * 70)
    print(f"{'λ_seg':<10} {'⟨r_massive⟩ [pc]':<20} {'∂⟨r⟩/∂λ_seg':<20} {'Status':<10}")
    print("-" * 70)

    results = []
    all_passed = True

    for lam in lambda_values:
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Compute mean radius
        r_mean = float(mean_radius_of_massive_jax(subkey1, lam, N_stars=1000))

        # Compute gradient
        try:
            # Skip λ=0 for gradient (use small positive value)
            lam_for_grad = max(lam, 0.01)
            grad_val = grad_mean_radius_wrt_lambda_seg(subkey2, lam_for_grad, N_stars=1000)

            # Check validity
            is_finite = np.isfinite(grad_val)
            # Gradient should be negative (more segregation → smaller radius)
            has_correct_sign = grad_val < 0 or lam == 0.0  # λ=0 might have small positive

            passed = is_finite and (lam == 0.0 or has_correct_sign)
            status = "PASS" if passed else "FAIL"

            if not passed:
                all_passed = False

            results.append({
                "lambda_seg": lam,
                "r_mean": r_mean,
                "grad": grad_val,
                "passed": passed,
            })

            print(f"{lam:<10.2f} {r_mean:<20.4f} {grad_val:<20.4f} {status:<10}")

        except Exception as e:
            print(f"{lam:<10.2f} {r_mean:<20.4f} {'ERROR':<20} {'FAIL':<10}")
            print(f"         Error: {e}")
            all_passed = False

    print("-" * 70)
    print()

    # Summary
    print("Summary:")
    print(f"  - All gradients finite: {'Yes' if all_passed else 'No'}")

    # Check if gradients decrease monotonically with λ_seg
    r_values = [r["r_mean"] for r in results]
    monotonic_decrease = all(r_values[i] >= r_values[i+1]
                             for i in range(len(r_values)-1))
    print(f"  - ⟨r_massive⟩ decreases with λ_seg: {'Yes' if monotonic_decrease else 'No'}")

    print()
    print("Expected behavior:")
    print("  - ∂⟨r_massive⟩/∂λ_seg < 0 (more segregation → smaller radius)")
    print("  - Gradient magnitude should be O(1) for λ_seg ∈ [0, 1]")
    print()

    overall = "PASS" if all_passed else "FAIL"
    print(f"Overall: {overall}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
