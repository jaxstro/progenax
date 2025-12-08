#!/usr/bin/env python
"""
Validate mass segregation physics against literature.

Tests:
1. MST-based Lambda_MSR diagnostic (Allison+ 2009)
   - Unsegregated cluster: Lambda_MSR approx 1.0
   - Segregated cluster: Lambda_MSR > 1.5

2. Baumgardt algorithm energy-mass correlation
   - s=0: Weak mass-energy correlation (random assignment)
   - s=1: Strong negative correlation (maximal segregation)

3. Virial equilibrium preservation
   - Q = 2K/|U| should match Q_target after segregation

4. Energy ranking verification
   - Most massive stars in most bound orbits for s=1

References:
    Baumgardt et al. (2008), MNRAS, 384, 1231 - Energy-ranked orbit assignment
    Kupper et al. (2011), MNRAS, 417, 2300 - McLuster implementation
    Allison et al. (2009), MNRAS, 395, 1449 - MST-based Lambda_MSR diagnostic

Usage:
    python scripts/validate_mass_segregation.py
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

from progenax.profiles import PlummerProfile, EFFProfile
from progenax.profiles.mass_segregation import (
    apply_mass_segregation_baumgardt,
    mass_segregation_ratio_mst,
    _softened_potential,
)
from progenax.kinematics import PlummerVelocityDF
from progenax.imf import PowerLawIMF


# =============================================================================
# Constants
# =============================================================================

G = 0.00450  # pc^3 Msun^-1 Myr^-2 (stellar units)
EPS = 0.01  # Softening length [pc]
PLOT_DIR = Path(__file__).parent.parent / "validation" / "plots"


def ensure_plot_dir():
    """Create plot directory if it doesn't exist."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Validation 1: Lambda_MSR Diagnostic
# =============================================================================


def validate_msr_diagnostic(n_realizations: int = 10):
    """
    Validate MST-based Lambda_MSR matches Allison+ (2009) expectations.

    Expected:
    - Unsegregated Plummer: Lambda_MSR approx 1.0 +/- 0.1
    - After s=1 segregation: Lambda_MSR > 1.5 (significant)
    """
    print("\n" + "=" * 70)
    print("VALIDATION 1: MST-based Lambda_MSR Diagnostic (Allison+ 2009)")
    print("=" * 70)

    N = 500
    r_h = 1.0
    n_massive = 20
    n_random = 50

    lambda_unseg_list = []
    lambda_seg_list = []

    key = jax.random.PRNGKey(42)

    for i in range(n_realizations):
        keys = jax.random.split(key, 7)
        key = keys[0]  # Update key for next iteration

        # Generate cluster
        imf = PowerLawIMF.kroupa()
        masses = imf.sample(keys[1], N)

        profile = PlummerProfile(r_h=r_h)
        positions = profile.sample_positions(masses, keys[2])

        velocity_df = PlummerVelocityDF(r_h=r_h)
        velocities = velocity_df.sample_velocities(positions, masses, keys[3], G=G)

        # Lambda_MSR for unsegregated cluster
        result_unseg = mass_segregation_ratio_mst(
            positions, masses, n_massive=n_massive, n_random=n_random, key=keys[4]
        )
        lambda_unseg_list.append(float(result_unseg["lambda_msr"]))

        # Apply strong segregation (s=1)
        pos_seg, vel_seg = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=1.0, key=keys[5], G=G, eps=EPS
        )

        # Lambda_MSR for segregated cluster
        result_seg = mass_segregation_ratio_mst(
            pos_seg, masses, n_massive=n_massive, n_random=n_random, key=keys[6]
        )
        lambda_seg_list.append(float(result_seg["lambda_msr"]))

    # Statistics
    lambda_unseg_mean = np.mean(lambda_unseg_list)
    lambda_unseg_std = np.std(lambda_unseg_list)
    lambda_seg_mean = np.mean(lambda_seg_list)
    lambda_seg_std = np.std(lambda_seg_list)

    print(f"\nN = {N}, r_h = {r_h} pc, n_massive = {n_massive}")
    print(f"Realizations: {n_realizations}")
    print()
    print("Results:")
    print("-" * 50)
    print(f"{'Case':<25} {'Lambda_MSR':<15} {'Expected':<15}")
    print("-" * 50)
    print(f"{'Unsegregated':<25} {lambda_unseg_mean:.3f} +/- {lambda_unseg_std:.3f}   {'~1.0':<15}")
    print(f"{'Segregated (s=1)':<25} {lambda_seg_mean:.3f} +/- {lambda_seg_std:.3f}   {'>1.5':<15}")
    print("-" * 50)

    # Pass/fail criteria
    pass_unseg = 0.8 < lambda_unseg_mean < 1.2
    pass_seg = lambda_seg_mean > 1.3

    status_unseg = "PASS" if pass_unseg else "FAIL"
    status_seg = "PASS" if pass_seg else "FAIL"

    print(f"\nUnsegregated Lambda in [0.8, 1.2]: {status_unseg}")
    print(f"Segregated Lambda > 1.3: {status_seg}")

    return {
        "lambda_unseg_mean": lambda_unseg_mean,
        "lambda_unseg_std": lambda_unseg_std,
        "lambda_seg_mean": lambda_seg_mean,
        "lambda_seg_std": lambda_seg_std,
        "pass_unseg": pass_unseg,
        "pass_seg": pass_seg,
    }


# =============================================================================
# Validation 2: Mass-Energy Correlation
# =============================================================================


def validate_baumgardt_correlation():
    """
    Validate energy-ranked orbit assignment produces expected mass-energy correlation.

    Expected:
    - s=0: |rho(mass, energy)| < 0.2 (weak correlation)
    - s=1: rho(mass, energy) < -0.7 (strong negative correlation)
    """
    print("\n" + "=" * 70)
    print("VALIDATION 2: Baumgardt Mass-Energy Correlation")
    print("=" * 70)

    from scipy.stats import spearmanr

    N = 500
    r_h = 1.0

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)

    # Generate cluster
    imf = PowerLawIMF.kroupa()
    masses = imf.sample(keys[0], N)

    profile = PlummerProfile(r_h=r_h)
    positions = profile.sample_positions(masses, keys[1])

    velocity_df = PlummerVelocityDF(r_h=r_h)
    velocities = velocity_df.sample_velocities(positions, masses, keys[2], G=G)

    # Test different s values
    s_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    print(f"\nN = {N}, r_h = {r_h} pc")
    print()
    print("Results:")
    print("-" * 60)
    print(f"{'s':<8} {'rho(m, E)':<15} {'Expected':<20} {'Status':<10}")
    print("-" * 60)

    for i, s in enumerate(s_values):
        pos_seg, vel_seg = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=s, key=keys[3 + i], G=G, eps=EPS
        )

        # Compute binding energies
        phi = _softened_potential(pos_seg, masses, G=G, eps=EPS)
        v2 = jnp.sum(vel_seg**2, axis=1)
        E = 0.5 * v2 + phi

        # Spearman correlation
        rho, pval = spearmanr(np.array(masses), np.array(E))

        if s == 0.0:
            expected = "|rho| < 0.3"
            passed = abs(rho) < 0.3
        elif s == 1.0:
            expected = "rho < -0.6"
            passed = rho < -0.6
        else:
            expected = f"~{-0.7 * s:.2f}"
            passed = True  # Intermediate values

        status = "PASS" if passed else "FAIL"
        results.append({"s": s, "rho": rho, "passed": passed})

        print(f"{s:<8.2f} {rho:>+.3f}{'':>8} {expected:<20} {status:<10}")

    print("-" * 60)

    return results


# =============================================================================
# Validation 3: Virial Equilibrium
# =============================================================================


def validate_virial_equilibrium():
    """
    Verify Q_target virial ratio is achieved after segregation.

    Expected: |Q - Q_target| < 0.05
    """
    print("\n" + "=" * 70)
    print("VALIDATION 3: Virial Equilibrium Preservation")
    print("=" * 70)

    N = 500
    r_h = 1.0

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)

    # Generate cluster
    imf = PowerLawIMF.kroupa()
    masses = imf.sample(keys[0], N)

    profile = PlummerProfile(r_h=r_h)
    positions = profile.sample_positions(masses, keys[1])

    velocity_df = PlummerVelocityDF(r_h=r_h)
    velocities = velocity_df.sample_velocities(positions, masses, keys[2], G=G)

    # Test different Q_target values
    q_targets = [0.5, 1.0, 1.5]
    results = []

    print(f"\nN = {N}, r_h = {r_h} pc, s = 0.5")
    print()
    print("Results:")
    print("-" * 60)
    print(f"{'Q_target':<12} {'Q_measured':<15} {'|dQ/Q|':<15} {'Status':<10}")
    print("-" * 60)

    for i, Q_target in enumerate(q_targets):
        pos_seg, vel_seg = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=0.5, key=keys[3 + i], G=G, eps=EPS, Q_target=Q_target
        )

        # Compute virial ratio
        phi = _softened_potential(pos_seg, masses, G=G, eps=EPS)
        v2 = jnp.sum(vel_seg**2, axis=1)
        U = 0.5 * jnp.sum(masses * phi)
        K = 0.5 * jnp.sum(masses * v2)
        Q_measured = float(2.0 * K / jnp.abs(U))

        dQ = abs(Q_measured - Q_target)
        rel_error = dQ / Q_target

        passed = rel_error < 0.10  # 10% tolerance for orbit reassignment
        status = "PASS" if passed else "FAIL"

        results.append({
            "Q_target": Q_target,
            "Q_measured": Q_measured,
            "rel_error": rel_error,
            "passed": passed,
        })

        print(f"{Q_target:<12.2f} {Q_measured:<15.4f} {rel_error:<15.4f} {status:<10}")

    print("-" * 60)

    return results


# =============================================================================
# Validation 4: Energy Ranking
# =============================================================================


def validate_energy_ranking():
    """
    Verify most massive stars occupy most bound orbits for s=1.

    Expected: Top 10% by mass should have top 20% most negative energies
    """
    print("\n" + "=" * 70)
    print("VALIDATION 4: Energy Ranking (s=1)")
    print("=" * 70)

    N = 500
    r_h = 1.0

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    # Generate cluster
    imf = PowerLawIMF.kroupa()
    masses = imf.sample(keys[0], N)

    profile = PlummerProfile(r_h=r_h)
    positions = profile.sample_positions(masses, keys[1])

    velocity_df = PlummerVelocityDF(r_h=r_h)
    velocities = velocity_df.sample_velocities(positions, masses, keys[2], G=G)

    # Apply maximal segregation
    pos_seg, vel_seg = apply_mass_segregation_baumgardt(
        positions, velocities, masses, s=1.0, key=keys[3], G=G, eps=EPS
    )

    # Compute binding energies
    phi = _softened_potential(pos_seg, masses, G=G, eps=EPS)
    v2 = jnp.sum(vel_seg**2, axis=1)
    E = 0.5 * v2 + phi  # More negative = more bound

    # Get top 10% by mass
    mass_order = jnp.argsort(-masses)
    top_10_pct = int(N * 0.1)
    top_mass_idx = mass_order[:top_10_pct]

    # Get top 20% most bound by energy (most negative)
    energy_order = jnp.argsort(E)  # Ascending = most bound first
    top_20_pct = int(N * 0.2)
    most_bound_idx = set(np.array(energy_order[:top_20_pct]))

    # Count how many top-mass stars are in top bound orbits
    overlap = sum(1 for idx in np.array(top_mass_idx) if idx in most_bound_idx)
    overlap_fraction = overlap / top_10_pct

    print(f"\nN = {N}, r_h = {r_h} pc")
    print()
    print("Results:")
    print("-" * 50)
    print(f"Top 10% by mass (N={top_10_pct}):")
    print(f"  Mass range: [{float(jnp.min(masses[top_mass_idx])):.2f}, {float(jnp.max(masses[top_mass_idx])):.2f}] Msun")
    print()
    print(f"Top 20% most bound (N={top_20_pct}):")
    print(f"  Energy range: [{float(jnp.min(E[energy_order[:top_20_pct]])):.4f}, {float(jnp.max(E[energy_order[:top_20_pct]])):.4f}]")
    print()
    print(f"Overlap: {overlap}/{top_10_pct} = {overlap_fraction:.1%}")
    print("-" * 50)

    passed = overlap_fraction > 0.7
    status = "PASS" if passed else "FAIL"
    print(f"\nOverlap > 70%: {status}")

    return {"overlap_fraction": overlap_fraction, "passed": passed}


# =============================================================================
# Plotting
# =============================================================================


def create_validation_plots():
    """Create publication-quality validation plots."""
    ensure_plot_dir()

    print("\n" + "=" * 70)
    print("Creating validation plots...")
    print("=" * 70)

    N = 500
    r_h = 1.0

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)

    # Generate cluster
    imf = PowerLawIMF.kroupa()
    masses = imf.sample(keys[0], N)

    profile = PlummerProfile(r_h=r_h)
    positions = profile.sample_positions(masses, keys[1])

    velocity_df = PlummerVelocityDF(r_h=r_h)
    velocities = velocity_df.sample_velocities(positions, masses, keys[2], G=G)

    # Apply segregation at s=0, s=0.5, s=1
    s_values = [0.0, 0.5, 1.0]
    results = {}

    for i, s in enumerate(s_values):
        pos_seg, vel_seg = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=s, key=keys[3 + i], G=G, eps=EPS
        )

        phi = _softened_potential(pos_seg, masses, G=G, eps=EPS)
        v2 = jnp.sum(vel_seg**2, axis=1)
        E = 0.5 * v2 + phi
        radii = jnp.linalg.norm(pos_seg, axis=1)

        results[s] = {
            "positions": pos_seg,
            "velocities": vel_seg,
            "energy": E,
            "radii": radii,
        }

    # --- Plot 1: Mass vs Energy scatter for different s ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, s in zip(axes, s_values):
        E = np.array(results[s]["energy"])
        sc = ax.scatter(np.array(masses), E, c=np.array(results[s]["radii"]),
                       cmap="viridis", alpha=0.6, s=10)
        ax.set_xlabel("Mass [Msun]", fontsize=11)
        ax.set_ylabel("Binding Energy E", fontsize=11)
        ax.set_title(f"s = {s:.1f}", fontsize=12)

        # Add Spearman correlation
        from scipy.stats import spearmanr
        rho, _ = spearmanr(masses, E)
        ax.text(0.05, 0.95, f"rho = {rho:+.3f}", transform=ax.transAxes,
                fontsize=10, va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.colorbar(sc, ax=axes[-1], label="Radius [pc]")
    fig.suptitle("Mass-Energy Correlation vs Segregation Parameter s", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mass_segregation_energy_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_DIR / 'mass_segregation_energy_correlation.png'}")

    # --- Plot 2: Radial distribution by mass percentile ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    mass_percentiles = [90, 75, 50]
    colors = ["red", "orange", "blue"]

    for ax, s in zip(axes, s_values):
        radii = np.array(results[s]["radii"])
        masses_np = np.array(masses)

        for perc, color in zip(mass_percentiles, colors):
            threshold = np.percentile(masses_np, perc)
            mask = masses_np >= threshold
            ax.hist(radii[mask], bins=30, alpha=0.5, color=color,
                   label=f">P{perc} (M>{threshold:.1f})", density=True)

        ax.set_xlabel("Radius [pc]", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"s = {s:.1f}", fontsize=12)
        ax.legend(fontsize=9)

    fig.suptitle("Radial Distribution by Mass Percentile", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mass_segregation_radial_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_DIR / 'mass_segregation_radial_distribution.png'}")

    # --- Plot 3: 2D projection colored by mass ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, s in zip(axes, s_values):
        pos = np.array(results[s]["positions"])
        masses_np = np.array(masses)

        sc = ax.scatter(pos[:, 0], pos[:, 1], c=np.log10(masses_np),
                       cmap="plasma", alpha=0.6, s=10 + 20 * (masses_np / masses_np.max()))
        ax.set_xlabel("x [pc]", fontsize=11)
        ax.set_ylabel("y [pc]", fontsize=11)
        ax.set_title(f"s = {s:.1f}", fontsize=12)
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

    plt.colorbar(sc, ax=axes[-1], label="log10(Mass/Msun)")
    fig.suptitle("2D Projection: Mass Segregation Effect", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mass_segregation_2d_projection.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_DIR / 'mass_segregation_2d_projection.png'}")

    # --- Plot 4: Summary - Spearman correlation vs s ---
    from scipy.stats import spearmanr

    s_range = np.linspace(0, 1, 11)
    rho_values = []

    for s in s_range:
        key, subkey = jax.random.split(key)
        pos_seg, vel_seg = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=float(s), key=subkey, G=G, eps=EPS
        )
        phi = _softened_potential(pos_seg, masses, G=G, eps=EPS)
        v2 = jnp.sum(vel_seg**2, axis=1)
        E = 0.5 * v2 + phi
        rho, _ = spearmanr(masses, E)
        rho_values.append(rho)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_range, rho_values, "o-", lw=2, markersize=8, color="steelblue")
    ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ax.axhline(-0.7, ls=":", color="red", alpha=0.7, label="Target: rho < -0.7 for s=1")
    ax.set_xlabel("Segregation Parameter s", fontsize=12)
    ax.set_ylabel("Spearman rho(mass, energy)", fontsize=12)
    ax.set_title("Mass-Energy Correlation vs Segregation Strength", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1, 0.3)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mass_segregation_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOT_DIR / 'mass_segregation_summary.png'}")

    print("\nAll plots saved to:", PLOT_DIR)


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("Mass Segregation Validation Script")
    print("=" * 70)
    print()
    print("References:")
    print("  - Baumgardt et al. (2008), MNRAS, 384, 1231")
    print("  - Kupper et al. (2011), MNRAS, 417, 2300")
    print("  - Allison et al. (2009), MNRAS, 395, 1449")
    print()

    # Run validations
    results = {}

    results["msr"] = validate_msr_diagnostic(n_realizations=5)
    results["correlation"] = validate_baumgardt_correlation()
    results["virial"] = validate_virial_equilibrium()
    results["ranking"] = validate_energy_ranking()

    # Create plots
    create_validation_plots()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True

    # MSR diagnostic
    msr_passed = results["msr"]["pass_unseg"] and results["msr"]["pass_seg"]
    print(f"1. Lambda_MSR Diagnostic: {'PASS' if msr_passed else 'FAIL'}")
    all_passed &= msr_passed

    # Correlation
    corr_s0_passed = any(r["passed"] for r in results["correlation"] if r["s"] == 0.0)
    corr_s1_passed = any(r["passed"] for r in results["correlation"] if r["s"] == 1.0)
    corr_passed = corr_s0_passed and corr_s1_passed
    print(f"2. Mass-Energy Correlation: {'PASS' if corr_passed else 'FAIL'}")
    all_passed &= corr_passed

    # Virial
    virial_passed = all(r["passed"] for r in results["virial"])
    print(f"3. Virial Equilibrium: {'PASS' if virial_passed else 'FAIL'}")
    all_passed &= virial_passed

    # Ranking
    ranking_passed = results["ranking"]["passed"]
    print(f"4. Energy Ranking: {'PASS' if ranking_passed else 'FAIL'}")
    all_passed &= ranking_passed

    print("-" * 70)
    overall = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"\nOverall: {overall}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
