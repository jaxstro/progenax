#!/usr/bin/env python
"""
Gradient-based parameter recovery for mass segregation.

Demonstrates that the cluster IC generator can be used in AD-based inference
by recovering a known λ_seg value from a summary statistic.

Usage:
    python scripts/cluster_validation/run_recover_lambda_seg.py
"""

import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from progenax.cluster.validation import recover_lambda_seg_via_gradient_descent


# =============================================================================
# Plot Setup
# =============================================================================

# Seaborn style
sns.set_theme(style="whitegrid", font_scale=1.2)
sns.set_palette("colorblind")

# Enable LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
})

PLOT_DIR = Path(__file__).parent.parent.parent / "validation" / "plots" / "cluster_ic" / "differentiability"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Main
# =============================================================================


def main():
    """Run gradient-based parameter recovery and plot results."""
    print("=" * 70)
    print("Gradient-Based Parameter Recovery: λ_seg")
    print("=" * 70)
    print()

    key = jax.random.PRNGKey(42)

    # Run recovery
    lambda_true = 0.7
    n_steps = 15
    step_size = 0.3

    print(f"Target: λ_seg = {lambda_true}")
    print(f"Initial guess: λ_seg = 0.1")
    print(f"Steps: {n_steps}, Step size: {step_size}")
    print()
    print("Running gradient descent...")

    results = recover_lambda_seg_via_gradient_descent(
        key,
        lambda_true=lambda_true,
        n_steps=n_steps,
        step_size=step_size,
        N_stars=1000,  # Smaller for speed
    )

    lambda_history = results["lambda_history"]
    loss_history = results["loss_history"]
    lambda_final = results["lambda_final"]

    print(f"\nFinal recovered λ_seg: {lambda_final:.4f}")
    print(f"True λ_seg: {lambda_true}")
    print(f"Error: |λ_final - λ_true| = {abs(lambda_final - lambda_true):.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): λ_seg trajectory
    ax = axes[0]
    iterations = np.arange(len(lambda_history))
    ax.plot(iterations, lambda_history, "o-", markersize=8, linewidth=2,
            color=sns.color_palette()[0], label=r"$\lambda_{\rm seg}$ trajectory")
    ax.axhline(lambda_true, ls="--", color="red", lw=2,
               label=rf"True $\lambda_{{\rm seg}} = {lambda_true}$")
    ax.axhline(0.1, ls=":", color="gray", lw=1.5,
               label="Initial guess")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\lambda_{\rm seg}$")
    ax.set_title(r"(a) Parameter Recovery Trajectory")
    ax.set_xlim(-0.5, len(lambda_history) - 0.5)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    # Panel (b): Loss trajectory
    ax = axes[1]
    ax.semilogy(np.arange(len(loss_history)), loss_history, "s-", markersize=8, linewidth=2,
                color=sns.color_palette()[2])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Loss $(\langle r \rangle_{\rm model} - \langle r \rangle_{\rm target})^2$")
    ax.set_title(r"(b) Loss Convergence")
    ax.set_xlim(-0.5, len(loss_history) - 0.5)

    fig.suptitle(r"Gradient-Based Recovery of $\lambda_{\rm seg}$", fontsize=14, y=1.02)
    plt.tight_layout()

    outpath = PLOT_DIR / "fig_gradient_recovery.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outpath}")

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  True λ_seg:      {lambda_true}")
    print(f"  Initial guess:   0.1")
    print(f"  Final recovered: {lambda_final:.4f}")
    print(f"  Error:           {abs(lambda_final - lambda_true):.4f}")
    print()

    # Pass/fail
    initial_error = abs(0.1 - lambda_true)
    final_error = abs(lambda_final - lambda_true)
    converged = final_error < initial_error

    print(f"  Converged (final error < initial error): {'PASS' if converged else 'FAIL'}")
    print(f"    Initial error: {initial_error:.4f}")
    print(f"    Final error:   {final_error:.4f}")
    print()

    if converged and final_error < 0.2:
        print("  The IC generator is usable in AD-based inference!")
    else:
        print("  Warning: Convergence may need tuning (step size, iterations)")

    print("=" * 70)

    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
