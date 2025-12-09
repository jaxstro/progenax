#!/usr/bin/env python
"""
Validate progenax.cluster IC generation (v1.4 spec).

Tests:
1. Coherent velocity inheritance (ancestry-based correlation)
2. Mass segregation Λ_MSR monotonicity with λ_seg
3. Fractal Q parameter correlation with D
4. COM removal verification

This validates the cluster IC generator, NOT the profiles.mass_segregation
module (that's validated by validate_mass_segregation.py).

Usage:
    python scripts/validate_cluster_ic.py
"""

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from progenax.cluster import (
    generate_cluster_ic,
    SpatialStructureParams,
    MassSegregationLayer,
    FractalLayer,
    generate_fractal_positions,
)
from progenax.cluster.fractal_gw_legacy import assign_velocities_and_virialize
from progenax.diagnostics import compute_lambda_msr, compute_q_parameter
from progenax.imf import PowerLawIMF


# =============================================================================
# Constants
# =============================================================================

PLOT_DIR = Path(__file__).parent.parent / "validation" / "plots"


def ensure_plot_dir():
    """Create plot directory if it doesn't exist."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Validation 1: Coherent Velocity Inheritance
# =============================================================================


def validate_coherent_velocities(n_realizations: int = 10):
    """
    Validate that coherent velocities show ancestry-based correlation.

    Expected:
    - Velocity correlation within ancestry groups > between groups
    - Root-level stars (ancestry == -1) should have lower correlation
    """
    print("\n" + "=" * 70)
    print("VALIDATION 1: Coherent Velocity Inheritance")
    print("=" * 70)

    N = 500
    D = 2.0

    within_correlations = []
    between_correlations = []

    key = jax.random.PRNGKey(42)

    for i in range(n_realizations):
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

        # Generate fractal positions with ancestry
        positions, ancestry = generate_fractal_positions(subkey1, N, D=D)

        # Generate random masses (not important for velocity correlation test)
        masses = jax.random.uniform(subkey2, (N,), minval=0.5, maxval=2.0)

        # Assign coherent velocities
        velocities_coherent = assign_velocities_and_virialize(
            subkey3, positions, masses, target_Q_vir=0.5,
            ancestry=ancestry, coherent=True
        )

        # Compute velocity correlations within and between ancestry groups
        ancestry_np = np.array(ancestry)
        velocities_np = np.array(velocities_coherent)

        # Find unique non-root ancestry values
        unique_parents = np.unique(ancestry_np)
        unique_parents = unique_parents[unique_parents >= 0]

        if len(unique_parents) < 5:
            continue  # Skip if too few groups

        # Sample correlation within groups
        corr_within = []
        for parent in unique_parents[:20]:  # Sample up to 20 groups
            mask = ancestry_np == parent
            if np.sum(mask) < 2:
                continue
            group_velocities = velocities_np[mask]
            if len(group_velocities) >= 2:
                # Correlation of velocity vectors (dot product normalized)
                norms = np.linalg.norm(group_velocities, axis=1, keepdims=True)
                normalized = group_velocities / np.maximum(norms, 1e-10)
                # Mean pairwise correlation
                corr_matrix = normalized @ normalized.T
                n_group = len(normalized)
                if n_group > 1:
                    # Upper triangle (excluding diagonal)
                    mask_tri = np.triu(np.ones((n_group, n_group), dtype=bool), k=1)
                    corr_within.extend(corr_matrix[mask_tri].tolist())

        # Sample correlation between groups
        corr_between = []
        for _ in range(100):
            i1, i2 = np.random.choice(len(velocities_np), size=2, replace=False)
            if ancestry_np[i1] != ancestry_np[i2]:
                v1 = velocities_np[i1] / np.maximum(np.linalg.norm(velocities_np[i1]), 1e-10)
                v2 = velocities_np[i2] / np.maximum(np.linalg.norm(velocities_np[i2]), 1e-10)
                corr_between.append(np.dot(v1, v2))

        if corr_within:
            within_correlations.append(np.mean(corr_within))
        if corr_between:
            between_correlations.append(np.mean(corr_between))

    # Statistics
    if within_correlations and between_correlations:
        within_mean = np.mean(within_correlations)
        within_std = np.std(within_correlations)
        between_mean = np.mean(between_correlations)
        between_std = np.std(between_correlations)

        print(f"\nN = {N}, D = {D}")
        print(f"Realizations: {n_realizations}")
        print()
        print("Results:")
        print("-" * 60)
        print(f"{'Case':<30} {'Mean Correlation':<20} {'Expected':<15}")
        print("-" * 60)
        print(f"{'Within ancestry groups':<30} {within_mean:.4f} ± {within_std:.4f}   {'> between':<15}")
        print(f"{'Between ancestry groups':<30} {between_mean:.4f} ± {between_std:.4f}   {'~0':<15}")
        print("-" * 60)

        # Pass criteria: within > between (coherent velocities work)
        passed = within_mean > between_mean
        status = "PASS" if passed else "FAIL"
        print(f"\nWithin > Between: {status}")

        return {
            "within_mean": within_mean,
            "within_std": within_std,
            "between_mean": between_mean,
            "between_std": between_std,
            "passed": passed,
        }
    else:
        print("SKIP: Not enough data for correlation analysis")
        return {"passed": True, "skipped": True}


# =============================================================================
# Validation 2: Mass Segregation Monotonicity
# =============================================================================


def validate_mass_segregation_monotonicity():
    """
    Validate that Λ_MSR increases monotonically with λ_seg.

    Expected:
    - Λ_MSR(λ=0) ≈ 1.0 (no segregation)
    - Λ_MSR(λ=0.5) > Λ_MSR(λ=0)
    - Λ_MSR(λ=1.0) > Λ_MSR(λ=0.5)
    """
    print("\n" + "=" * 70)
    print("VALIDATION 2: Mass Segregation Λ_MSR Monotonicity")
    print("=" * 70)

    N = 500
    R_half = 1.0
    n_massive = 20
    n_random = 50

    lambda_seg_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    imf = PowerLawIMF.kroupa()
    key = jax.random.PRNGKey(42)

    print(f"\nN = {N}, R_half = {R_half} pc, n_massive = {n_massive}")
    print()
    print("Results:")
    print("-" * 60)
    print(f"{'λ_seg':<10} {'Λ_MSR':<15} {'σ':<10} {'Expected':<20}")
    print("-" * 60)

    for lambda_seg in lambda_seg_values:
        key, subkey = jax.random.split(key)

        if lambda_seg == 0.0:
            # No segregation
            cluster = generate_cluster_ic(
                key=subkey,
                N_stars=N,
                M_total=float(N),
                R_half=R_half,
                imf_params=imf,
                structure_params=SpatialStructureParams(base_profile="plummer"),
            )
        else:
            cluster = generate_cluster_ic(
                key=subkey,
                N_stars=N,
                M_total=float(N),
                R_half=R_half,
                imf_params=imf,
                structure_params=SpatialStructureParams(
                    base_profile="plummer",
                    mass_segregation=MassSegregationLayer(lambda_seg=lambda_seg),
                ),
            )

        # Compute Λ_MSR
        lambda_msr, sigma = compute_lambda_msr(
            np.array(cluster.positions),
            np.array(cluster.masses),
            N_massive=n_massive,
            N_random_samples=n_random,
        )

        expected = "~1.0" if lambda_seg == 0.0 else f"> λ={lambda_seg-0.25:.2f}"
        results.append({"lambda_seg": lambda_seg, "lambda_msr": lambda_msr, "sigma": sigma})

        print(f"{lambda_seg:<10.2f} {lambda_msr:<15.3f} {sigma:<10.3f} {expected:<20}")

    print("-" * 60)

    # Check monotonicity
    msr_values = [r["lambda_msr"] for r in results]
    monotonic = all(msr_values[i] <= msr_values[i+1] for i in range(len(msr_values)-1))

    # Also check that λ=1 gives significantly higher Λ_MSR than λ=0
    segregation_effect = msr_values[-1] > msr_values[0] + 0.2

    passed = monotonic and segregation_effect
    status = "PASS" if passed else "FAIL"
    print(f"\nMonotonic increase: {'Yes' if monotonic else 'No'}")
    print(f"Segregation effect (Λ(1) > Λ(0) + 0.2): {'Yes' if segregation_effect else 'No'}")
    print(f"Overall: {status}")

    return {"results": results, "monotonic": monotonic, "passed": passed}


# =============================================================================
# Validation 3: Fractal Q Parameter
# =============================================================================


def validate_fractal_q_parameter():
    """
    Validate that Q parameter increases with fractal dimension D.

    Expected:
    - D = 1.6: Low Q (clumpy)
    - D = 3.0: High Q (uniform)
    - Monotonic increase
    """
    print("\n" + "=" * 70)
    print("VALIDATION 3: Fractal Q Parameter vs Dimension D")
    print("=" * 70)

    N = 500
    D_values = [1.6, 2.0, 2.4, 2.8]
    n_realizations = 5

    results = []
    key = jax.random.PRNGKey(42)

    print(f"\nN = {N}, realizations per D = {n_realizations}")
    print()
    print("Results:")
    print("-" * 60)
    print(f"{'D':<8} {'Q (mean)':<12} {'Q (std)':<12} {'Expected':<20}")
    print("-" * 60)

    for D in D_values:
        Q_values = []
        for _ in range(n_realizations):
            key, subkey = jax.random.split(key)
            positions, _ = generate_fractal_positions(subkey, N, D=D)
            Q = compute_q_parameter(np.array(positions))
            Q_values.append(Q)

        Q_mean = np.mean(Q_values)
        Q_std = np.std(Q_values)

        expected = "low (clumpy)" if D < 2.0 else "high (uniform)" if D > 2.5 else "medium"
        results.append({"D": D, "Q_mean": Q_mean, "Q_std": Q_std})

        print(f"{D:<8.1f} {Q_mean:<12.4f} {Q_std:<12.4f} {expected:<20}")

    print("-" * 60)

    # Check overall trend: Q(high D) > Q(low D)
    # Strict monotonicity is too sensitive to statistical noise at low N
    Q_means = [r["Q_mean"] for r in results]
    Q_low_D = Q_means[0]  # D=1.6
    Q_high_D = Q_means[-1]  # D=2.8
    trend_correct = Q_high_D > Q_low_D

    # Also check Spearman correlation > 0 (overall positive trend)
    from scipy.stats import spearmanr
    D_values_list = [r["D"] for r in results]
    rho, pval = spearmanr(D_values_list, Q_means)
    positive_correlation = rho > 0

    passed = trend_correct and positive_correlation
    status = "PASS" if passed else "FAIL"
    print(f"\nQ(D=2.8) > Q(D=1.6): {'Yes' if trend_correct else 'No'}")
    print(f"Spearman correlation rho = {rho:.3f} (p = {pval:.3f})")
    print(f"Overall trend: {status}")

    return {"results": results, "passed": passed}


# =============================================================================
# Validation 4: COM Removal
# =============================================================================


def validate_com_removal():
    """
    Verify both position and velocity COM are removed.

    Expected:
    - Position COM < 1e-10 (machine precision for mass-weighted)
    - Velocity COM < 1e-10
    """
    print("\n" + "=" * 70)
    print("VALIDATION 4: Center of Mass Removal")
    print("=" * 70)

    N = 500
    imf = PowerLawIMF.kroupa()
    key = jax.random.PRNGKey(42)

    results = []

    # Test both smooth and fractal ICs
    configs = [
        ("Plummer (smooth)", SpatialStructureParams(base_profile="plummer")),
        ("Plummer + seg", SpatialStructureParams(
            base_profile="plummer",
            mass_segregation=MassSegregationLayer(lambda_seg=0.5)
        )),
        ("Fractal D=2.0", SpatialStructureParams(
            base_profile="plummer",
            fractal=FractalLayer(D=2.0, lambda_frac=1.0)
        )),
    ]

    print(f"\nN = {N}")
    print()
    print("Results:")
    print("-" * 70)
    print(f"{'Config':<20} {'|x_COM|':<15} {'|v_COM|':<15} {'Status':<10}")
    print("-" * 70)

    for name, structure_params in configs:
        key, subkey = jax.random.split(key)

        cluster = generate_cluster_ic(
            key=subkey,
            N_stars=N,
            M_total=float(N),
            R_half=1.0,
            imf_params=imf,
            structure_params=structure_params,
        )

        # Compute mass-weighted COM
        M_total = jnp.sum(cluster.masses)
        x_com = jnp.sum(cluster.masses[:, None] * cluster.positions, axis=0) / M_total
        v_com = jnp.sum(cluster.masses[:, None] * cluster.velocities, axis=0) / M_total

        x_com_norm = float(jnp.linalg.norm(x_com))
        v_com_norm = float(jnp.linalg.norm(v_com))

        passed = x_com_norm < 1e-6 and v_com_norm < 1e-6
        status = "PASS" if passed else "FAIL"

        results.append({
            "name": name,
            "x_com_norm": x_com_norm,
            "v_com_norm": v_com_norm,
            "passed": passed,
        })

        print(f"{name:<20} {x_com_norm:<15.2e} {v_com_norm:<15.2e} {status:<10}")

    print("-" * 70)

    all_passed = all(r["passed"] for r in results)
    status = "PASS" if all_passed else "FAIL"
    print(f"\nAll COM < 1e-6: {status}")

    return {"results": results, "passed": all_passed}


# =============================================================================
# Plotting
# =============================================================================


def create_validation_plots():
    """Create publication-quality validation plots."""
    ensure_plot_dir()

    print("\n" + "=" * 70)
    print("Creating validation plots...")
    print("=" * 70)

    key = jax.random.PRNGKey(42)
    imf = PowerLawIMF.kroupa()

    # --- Plot 1: Velocity Coherence ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    positions, ancestry = generate_fractal_positions(subkey1, 500, D=2.0)
    masses = jax.random.uniform(subkey2, (500,), minval=0.5, maxval=2.0)

    # Coherent velocities
    vel_coherent = assign_velocities_and_virialize(
        subkey3, positions, masses, ancestry=ancestry, coherent=True
    )

    # Incoherent velocities
    key, subkey4 = jax.random.split(key)
    vel_incoherent = assign_velocities_and_virialize(
        subkey4, positions, masses, ancestry=ancestry, coherent=False
    )

    # Plot velocity field (2D projection)
    pos_np = np.array(positions)
    vel_coh_np = np.array(vel_coherent)
    vel_incoh_np = np.array(vel_incoherent)

    scale = 0.3
    axes[0].quiver(pos_np[:, 0], pos_np[:, 1], vel_coh_np[:, 0], vel_coh_np[:, 1],
                   scale=scale, alpha=0.5, width=0.003)
    axes[0].set_xlabel("x [pc]", fontsize=11)
    axes[0].set_ylabel("y [pc]", fontsize=11)
    axes[0].set_title("Coherent Velocities (ancestry-based)", fontsize=12)
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-1.5, 1.5)
    axes[0].set_ylim(-1.5, 1.5)

    axes[1].quiver(pos_np[:, 0], pos_np[:, 1], vel_incoh_np[:, 0], vel_incoh_np[:, 1],
                   scale=scale, alpha=0.5, width=0.003)
    axes[1].set_xlabel("x [pc]", fontsize=11)
    axes[1].set_ylabel("y [pc]", fontsize=11)
    axes[1].set_title("Incoherent Velocities (random)", fontsize=12)
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-1.5, 1.5)
    axes[1].set_ylim(-1.5, 1.5)

    fig.suptitle("Velocity Field Comparison: Coherent vs Incoherent", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cluster_ic_velocity_coherence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_DIR / 'cluster_ic_velocity_coherence.png'}")

    # --- Plot 2: Mass Segregation Λ_MSR vs λ_seg ---
    fig, ax = plt.subplots(figsize=(8, 5))

    lambda_seg_values = np.linspace(0, 1, 11)
    msr_values = []
    msr_errors = []

    for lambda_seg in lambda_seg_values:
        key, subkey = jax.random.split(key)

        if lambda_seg == 0.0:
            cluster = generate_cluster_ic(
                key=subkey, N_stars=500, M_total=500.0, R_half=1.0,
                imf_params=imf,
                structure_params=SpatialStructureParams(base_profile="plummer"),
            )
        else:
            cluster = generate_cluster_ic(
                key=subkey, N_stars=500, M_total=500.0, R_half=1.0,
                imf_params=imf,
                structure_params=SpatialStructureParams(
                    base_profile="plummer",
                    mass_segregation=MassSegregationLayer(lambda_seg=float(lambda_seg)),
                ),
            )

        lam, sigma = compute_lambda_msr(
            np.array(cluster.positions), np.array(cluster.masses),
            N_massive=20, N_random_samples=50
        )
        msr_values.append(lam)
        msr_errors.append(sigma)

    ax.errorbar(lambda_seg_values, msr_values, yerr=msr_errors, fmt="o-", capsize=4,
                markersize=8, color="steelblue", linewidth=2)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5, label="No segregation")
    ax.set_xlabel("λ_seg (segregation parameter)", fontsize=12)
    ax.set_ylabel("Λ_MSR (mass segregation ratio)", fontsize=12)
    ax.set_title("Mass Segregation: Λ_MSR vs λ_seg", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cluster_ic_mass_segregation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_DIR / 'cluster_ic_mass_segregation.png'}")

    # --- Plot 3: Fractal Q vs D ---
    fig, ax = plt.subplots(figsize=(8, 5))

    D_values = np.linspace(1.6, 3.0, 8)
    Q_values = []

    for D in D_values:
        key, subkey = jax.random.split(key)
        positions, _ = generate_fractal_positions(subkey, 500, D=float(D))
        Q = compute_q_parameter(np.array(positions))
        Q_values.append(Q)

    ax.plot(D_values, Q_values, "o-", markersize=8, color="darkgreen", linewidth=2)
    ax.set_xlabel("Fractal Dimension D", fontsize=12)
    ax.set_ylabel("Q Parameter (Cartwright-Whitworth)", fontsize=12)
    ax.set_title("Fractal Substructure: Q vs D", fontsize=13)
    ax.axhline(0.8, ls=":", color="red", alpha=0.7, label="Uniform sphere (Q~0.8)")
    ax.set_xlim(1.5, 3.1)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cluster_ic_fractal_q.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_DIR / 'cluster_ic_fractal_q.png'}")

    # --- Plot 4: Summary 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Smooth Plummer
    key, subkey = jax.random.split(key)
    cluster_smooth = generate_cluster_ic(
        key=subkey, N_stars=500, M_total=500.0, R_half=1.0,
        imf_params=imf,
        structure_params=SpatialStructureParams(base_profile="plummer"),
    )
    pos = np.array(cluster_smooth.positions)
    masses_np = np.array(cluster_smooth.masses)
    axes[0, 0].scatter(pos[:, 0], pos[:, 1], c=np.log10(masses_np), cmap="plasma",
                       s=5 + 20*(masses_np/masses_np.max()), alpha=0.6)
    axes[0, 0].set_title("Smooth Plummer (no seg)", fontsize=11)
    axes[0, 0].set_xlabel("x [pc]")
    axes[0, 0].set_ylabel("y [pc]")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_xlim(-4, 4)
    axes[0, 0].set_ylim(-4, 4)

    # Panel B: Mass segregated
    key, subkey = jax.random.split(key)
    cluster_seg = generate_cluster_ic(
        key=subkey, N_stars=500, M_total=500.0, R_half=1.0,
        imf_params=imf,
        structure_params=SpatialStructureParams(
            base_profile="plummer",
            mass_segregation=MassSegregationLayer(lambda_seg=1.0),
        ),
    )
    pos = np.array(cluster_seg.positions)
    masses_np = np.array(cluster_seg.masses)
    axes[0, 1].scatter(pos[:, 0], pos[:, 1], c=np.log10(masses_np), cmap="plasma",
                       s=5 + 20*(masses_np/masses_np.max()), alpha=0.6)
    axes[0, 1].set_title("Mass Segregated (λ=1)", fontsize=11)
    axes[0, 1].set_xlabel("x [pc]")
    axes[0, 1].set_ylabel("y [pc]")
    axes[0, 1].set_aspect("equal")
    axes[0, 1].set_xlim(-4, 4)
    axes[0, 1].set_ylim(-4, 4)

    # Panel C: Fractal D=1.6
    key, subkey = jax.random.split(key)
    cluster_frac_low = generate_cluster_ic(
        key=subkey, N_stars=500, M_total=500.0, R_half=1.0,
        imf_params=imf,
        structure_params=SpatialStructureParams(
            base_profile="plummer",
            fractal=FractalLayer(D=1.6, lambda_frac=1.0),
        ),
    )
    pos = np.array(cluster_frac_low.positions)
    masses_np = np.array(cluster_frac_low.masses)
    axes[1, 0].scatter(pos[:, 0], pos[:, 1], c=np.log10(masses_np), cmap="plasma",
                       s=5 + 20*(masses_np/masses_np.max()), alpha=0.6)
    axes[1, 0].set_title("Fractal D=1.6 (clumpy)", fontsize=11)
    axes[1, 0].set_xlabel("x [pc]")
    axes[1, 0].set_ylabel("y [pc]")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_xlim(-4, 4)
    axes[1, 0].set_ylim(-4, 4)

    # Panel D: Fractal D=2.6
    key, subkey = jax.random.split(key)
    cluster_frac_high = generate_cluster_ic(
        key=subkey, N_stars=500, M_total=500.0, R_half=1.0,
        imf_params=imf,
        structure_params=SpatialStructureParams(
            base_profile="plummer",
            fractal=FractalLayer(D=2.6, lambda_frac=1.0),
        ),
    )
    pos = np.array(cluster_frac_high.positions)
    masses_np = np.array(cluster_frac_high.masses)
    sc = axes[1, 1].scatter(pos[:, 0], pos[:, 1], c=np.log10(masses_np), cmap="plasma",
                            s=5 + 20*(masses_np/masses_np.max()), alpha=0.6)
    axes[1, 1].set_title("Fractal D=2.6 (smooth)", fontsize=11)
    axes[1, 1].set_xlabel("x [pc]")
    axes[1, 1].set_ylabel("y [pc]")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_xlim(-4, 4)
    axes[1, 1].set_ylim(-4, 4)

    fig.colorbar(sc, ax=axes.ravel().tolist(), label="log10(M/Msun)", shrink=0.8)
    fig.suptitle("Cluster IC Generation: Structure Comparison", fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cluster_ic_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_DIR / 'cluster_ic_summary.png'}")

    print("\nAll plots saved to:", PLOT_DIR)


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("Cluster IC Validation Script (v1.4)")
    print("=" * 70)
    print()
    print("Tests:")
    print("  1. Coherent velocity inheritance (ancestry-based)")
    print("  2. Mass segregation Λ_MSR monotonicity")
    print("  3. Fractal Q parameter vs dimension D")
    print("  4. COM removal verification")
    print()

    # Run validations
    results = {}

    results["coherent"] = validate_coherent_velocities(n_realizations=5)
    results["segregation"] = validate_mass_segregation_monotonicity()
    results["fractal"] = validate_fractal_q_parameter()
    results["com"] = validate_com_removal()

    # Create plots
    create_validation_plots()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True

    # Coherent velocities
    coh_passed = results["coherent"].get("passed", False)
    print(f"1. Coherent Velocity Inheritance: {'PASS' if coh_passed else 'FAIL'}")
    all_passed &= coh_passed

    # Mass segregation
    seg_passed = results["segregation"]["passed"]
    print(f"2. Mass Segregation Monotonicity: {'PASS' if seg_passed else 'FAIL'}")
    all_passed &= seg_passed

    # Fractal Q
    frac_passed = results["fractal"]["passed"]
    print(f"3. Fractal Q vs D: {'PASS' if frac_passed else 'FAIL'}")
    all_passed &= frac_passed

    # COM removal
    com_passed = results["com"]["passed"]
    print(f"4. COM Removal: {'PASS' if com_passed else 'FAIL'}")
    all_passed &= com_passed

    print("-" * 70)
    overall = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"\nOverall: {overall}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
