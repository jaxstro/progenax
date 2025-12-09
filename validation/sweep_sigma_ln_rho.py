#!/usr/bin/env python
"""Sweep σ_ln_ρ to find where Q deviates from uniform baseline.

This script tests whether the "wildness" in FDF Q values comes from
too-strong density modulation (σ_ln_ρ) or structural issues.

Expected insight:
    If milder σ_ln_ρ brings Q back towards ~0.82 (uniform baseline),
    then the current σ_ln_ρ = 2.0 default is simply too strong for
    CW04-comparable Q values.

Usage:
    python validation/sweep_sigma_ln_rho.py

Output:
    - Table of Q(σ_ln_ρ) for uniform base profile
    - Comparison with bare uniform baseline
    - ASCII plot of Q vs σ_ln_ρ
"""

import jax
import jax.numpy as jnp
import numpy as np

from progenax.cluster.fdf_density import (
    FractalDensityLayer,
    generate_fractal_ic_density,
)
from progenax.diagnostics.substructure import compute_q_parameter
from progenax.profiles.uniform import UniformSphereProfile
from progenax.imf import PowerLawIMF


def measure_q_baseline(N: int = 300, n_samples: int = 10) -> tuple[float, float]:
    """Measure Q for bare uniform sphere (CW04 '3D0' baseline)."""
    profile = UniformSphereProfile(R=1.0)
    Q_values = []

    for seed in range(n_samples):
        key = jax.random.PRNGKey(seed)
        masses = jnp.ones(N)
        positions = profile.sample_positions(masses, key)
        Q = compute_q_parameter(np.asarray(positions))
        Q_values.append(Q)

    return np.mean(Q_values), np.std(Q_values)


def measure_q_fdf(
    sigma_ln_rho: float,
    chi: float = 2.0,
    N: int = 300,
    n_samples: int = 10,
) -> tuple[float, float]:
    """Measure Q for FDF with given σ_ln_ρ on uniform base."""
    Q_values = []

    # Use Kroupa IMF - only positions matter for Q, not mass values
    imf = PowerLawIMF.kroupa()

    for seed in range(n_samples):
        key = jax.random.PRNGKey(seed + 1000)  # Different seeds from baseline

        # Create FDF layer
        layer = FractalDensityLayer(
            chi=chi,
            sigma_ln_rho=sigma_ln_rho,
            lambda_frac=1.0,
            virial_ratio=0.5,
            base_profile="uniform",
            grid_size=64,
        )

        # Generate IC
        state = generate_fractal_ic_density(
            key=key,
            N_stars=N,
            M_total=float(N),  # 1 Msun per star
            R_half=1.0,
            imf_params=imf,
            layer=layer,
        )

        positions = np.asarray(state.positions)
        Q = compute_q_parameter(positions)
        Q_values.append(Q)

    return np.mean(Q_values), np.std(Q_values)


def main():
    print("=" * 70)
    print("σ_ln_ρ Sweep: Effect of Density Contrast on Q Parameter")
    print("=" * 70)
    print()

    # Parameters
    N = 300
    n_samples = 10
    chi = 2.0  # Fixed chi (intermediate)
    sigma_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Get baseline
    print("Measuring bare uniform sphere baseline (CW04 '3D0')...")
    Q_baseline, Q_baseline_std = measure_q_baseline(N, n_samples)
    print(f"  Baseline Q = {Q_baseline:.3f} ± {Q_baseline_std:.3f}")
    print()

    # Sweep σ_ln_ρ
    print("Sweeping σ_ln_ρ with FDF + uniform base profile...")
    print(f"  N = {N}, n_samples = {n_samples}, χ = {chi}")
    print()

    results = []
    for sigma in sigma_values:
        if sigma == 0.0:
            # σ = 0 should give bare uniform (no modulation)
            Q_mean, Q_std = Q_baseline, Q_baseline_std
            label = "(baseline)"
        else:
            Q_mean, Q_std = measure_q_fdf(sigma, chi, N, n_samples)
            label = ""

        results.append((sigma, Q_mean, Q_std))
        delta_Q = Q_mean - Q_baseline
        print(f"  σ_ln_ρ = {sigma:.1f}: Q = {Q_mean:.3f} ± {Q_std:.3f}  (ΔQ = {delta_Q:+.3f}) {label}")

    print()

    # ASCII plot
    print("-" * 70)
    print("Q vs σ_ln_ρ (ASCII plot)")
    print("-" * 70)

    # Find plot bounds
    Q_min = min(r[1] - r[2] for r in results)
    Q_max = max(r[1] + r[2] for r in results)
    Q_range = Q_max - Q_min
    Q_min -= 0.1 * Q_range
    Q_max += 0.1 * Q_range

    width = 50
    for sigma, Q_mean, Q_std in results:
        # Compute bar position
        pos = int((Q_mean - Q_min) / (Q_max - Q_min) * width)
        pos = max(0, min(width - 1, pos))

        # Create bar
        bar = [' '] * width
        bar[pos] = '●'

        # Add baseline marker
        baseline_pos = int((Q_baseline - Q_min) / (Q_max - Q_min) * width)
        if baseline_pos >= 0 and baseline_pos < width:
            bar[baseline_pos] = '|' if bar[baseline_pos] == ' ' else '●'

        bar_str = ''.join(bar)
        print(f"  σ={sigma:.1f} [{bar_str}] Q={Q_mean:.3f}")

    print()
    print(f"  | = baseline Q = {Q_baseline:.3f}")
    print(f"  Q range: [{Q_min:.2f}, {Q_max:.2f}]")
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    # Find where Q deviates significantly
    threshold = 0.05  # 5% deviation
    significant_devs = [(s, q, d) for s, q, _ in results
                        if (d := abs(q - Q_baseline) / Q_baseline) > threshold]

    if significant_devs:
        first_dev = min(significant_devs, key=lambda x: x[0])
        print(f"Q deviates > {threshold*100:.0f}% from baseline at σ_ln_ρ ≥ {first_dev[0]:.1f}")
    else:
        print(f"Q stays within {threshold*100:.0f}% of baseline for all σ_ln_ρ tested")

    print()
    print("Interpretation:")
    print("  - If Q ≈ baseline for all σ_ln_ρ: FDF doesn't significantly affect Q")
    print("  - If Q diverges at high σ_ln_ρ: current default (2.0) may be too strong")
    print("  - For CW04-comparable Q, use σ_ln_ρ where Q ≈ baseline")
    print()

    # Physical context
    print("-" * 70)
    print("Physical Context (Federrath+2010)")
    print("-" * 70)
    print()
    print("σ_ln_ρ = √(ln(1 + b²M²)) where:")
    print("  b ≈ 0.4 (natural driving mix)")
    print("  M = Mach number")
    print()
    print("Typical values from virial equilibrium:")
    print("  Small OC (10³ M☉): M ~ 21, σ_ln_ρ ~ 2.1")
    print("  Large OC (10⁴ M☉): M ~ 57, σ_ln_ρ ~ 2.5")
    print("  YMC (10⁵ M☉): M ~ 155, σ_ln_ρ ~ 2.9")
    print("  GC (10⁶ M☉): M ~ 422, σ_ln_ρ ~ 3.2")
    print()
    print("Note: These high Mach numbers come from the compact r_h in")
    print("Marks+2012. Real clouds may have lower Mach and σ_ln_ρ.")
    print()


if __name__ == "__main__":
    main()
