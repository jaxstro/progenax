#!/usr/bin/env python
"""
Spatial Profile Validation Script.

Produces publication-quality diagnostic figures for PlummerProfile, KingProfile,
and EFFProfile. Validates scientific and numerical correctness of:

1. Density profiles vs analytical formulas
2. Radial sampling distributions (CDF validation)
3. Half-mass radius verification (Plummer)
4. Isotropy tests (angular distributions)
5. Parameter sensitivity (W0, gamma effects)
6. Tidal truncation validation (King, EFF)
7. King ODE solution verification
8. Profile comparison panel

References:
    Plummer (1911) MNRAS 71, 460
    King (1966) AJ 71, 64
    Elson, Fall & Freeman (1987) ApJ 323, 54

Usage:
    python scripts/validate_profiles.py

Output:
    validation/plots/profiles_*.png
"""
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

jax.config.update("jax_enable_x64", True)

from progenax.profiles import PlummerProfile, KingProfile, EFFProfile, solve_king_profile
from progenax.profiles.king import king_K_function


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = "validation/plots"
N_SAMPLES = 50000  # Number of particles for sampling tests
SEED = 42

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


# =============================================================================
# Utility Functions
# =============================================================================

def compute_empirical_cdf(radii):
    """Compute empirical CDF from sampled radii."""
    sorted_r = jnp.sort(radii)
    ecdf = jnp.arange(1, len(radii) + 1) / len(radii)
    return sorted_r, ecdf


def ks_test(radii, cdf_func, r_max):
    """Kolmogorov-Smirnov test for radial distribution."""
    radii_np = np.array(radii)
    # Normalize to [0, 1] using CDF
    r_grid = np.linspace(0, float(r_max), 1000)
    cdf_vals = np.array(cdf_func(jnp.array(r_grid)))

    # Create CDF interpolator for KS test
    from scipy.interpolate import interp1d
    cdf_interp = interp1d(r_grid, cdf_vals, bounds_error=False, fill_value=(0, 1))

    # KS test
    stat, pvalue = stats.kstest(radii_np, cdf_interp)
    return stat, pvalue


# =============================================================================
# Plummer Profile Validation
# =============================================================================

def validate_plummer(output_dir: str):
    """Validate PlummerProfile implementation."""
    print("\n" + "="*60)
    print("PLUMMER PROFILE VALIDATION")
    print("="*60)

    r_h = 1.0  # Half-mass radius
    profile = PlummerProfile(r_h=r_h)

    key = jax.random.PRNGKey(SEED)
    masses = jnp.ones(N_SAMPLES)
    positions = profile.sample_positions(masses, key)
    radii = jnp.linalg.norm(positions, axis=1)
    a = profile.a

    # -------------------------------------------------------------------------
    # Compute metrics FIRST (needed for plot annotations)
    # -------------------------------------------------------------------------
    median_r = float(jnp.median(radii))
    r_h_error = abs(median_r - r_h) / r_h * 100

    def plummer_cdf(r):
        return r**3 / (r**2 + a**2)**1.5

    # CDF deviation metrics (more interpretable than KS test for large N)
    sorted_r, ecdf = compute_empirical_cdf(radii)
    theoretical_cdf = plummer_cdf(sorted_r)
    cdf_deviations = jnp.abs(ecdf - theoretical_cdf)
    max_cdf_deviation = float(jnp.max(cdf_deviations))
    mean_cdf_deviation = float(jnp.mean(cdf_deviations))

    passed = r_h_error < 1.0 and max_cdf_deviation < 0.02

    # -------------------------------------------------------------------------
    # Figure 1: Density Profile
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Density vs radius
    ax = axes[0]
    r_grid = jnp.linspace(0.01, 5 * r_h, 200)

    # Analytical Plummer density (normalized)
    rho_analytical = (1 + (r_grid / a)**2)**(-2.5)
    rho_analytical = rho_analytical / rho_analytical[0]  # Normalize to central density

    # Computed density
    rho_computed = profile.density(r_grid)
    rho_computed = rho_computed / rho_computed[0]

    ax.semilogy(r_grid / r_h, rho_analytical, 'b-', lw=2, label='Analytical: $(1 + r^2/a^2)^{-5/2}$')
    ax.semilogy(r_grid / r_h, rho_computed, 'r--', lw=2, label='density(r) method')

    # Histogram from samples
    bins = np.linspace(0, 5, 50)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(np.array(radii / r_h), bins=bins, density=False)
    # Convert to density: dn/dr / (4*pi*r^2)
    dr = bins[1] - bins[0]
    shell_volume = 4 * np.pi * (bin_centers * r_h)**2 * dr * r_h
    rho_hist = hist / shell_volume
    # Normalize by matching to analytical curve at bin centers (robust median scaling)
    rho_analytical_at_bins = np.interp(bin_centers, np.array(r_grid / r_h), np.array(rho_analytical))
    valid_bins = (rho_hist > 0) & (rho_analytical_at_bins > 1e-10)
    if np.any(valid_bins):
        scale = np.median(rho_analytical_at_bins[valid_bins] / rho_hist[valid_bins])
        rho_hist = rho_hist * scale
    ax.semilogy(bin_centers[valid_bins], rho_hist[valid_bins], 'go', ms=4, alpha=0.6, label=f'Samples (N={N_SAMPLES})')

    ax.set_xlabel('$r / r_h$')
    ax.set_ylabel('$\\rho(r) / \\rho_0$')
    ax.set_title('Plummer Density Profile')
    ax.set_xlim(0, 5)
    ax.set_ylim(1e-4, 2)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color='gray', ls=':', alpha=0.5, label='$r_h$')

    # Panel B: Cumulative Mass / CDF
    ax = axes[1]

    # Analytical Plummer CDF: M(<r)/M = r^3 / (r^2 + a^2)^(3/2)
    cdf_analytical = r_grid**3 / (r_grid**2 + a**2)**1.5

    ax.plot(r_grid / r_h, cdf_analytical, 'b-', lw=2, label='Analytical CDF')
    ax.plot(sorted_r[::100] / r_h, ecdf[::100], 'r.', ms=2, alpha=0.5, label='Empirical CDF')

    # Half-mass radius verification
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
    ax.axvline(1.0, color='gray', ls='--', alpha=0.5)
    ax.plot(1.0, 0.5, 'ko', ms=10, mfc='none', mew=2, label='Half-mass radius')

    # Add metrics text box
    metrics_text = (
        f"Validation Metrics:\n"
        f"$r_h$ error: {r_h_error:.2f}%\n"
        f"Max CDF dev: {max_cdf_deviation:.4f}\n"
        f"Mean CDF dev: {mean_cdf_deviation:.4f}"
    )
    ax.text(0.98, 0.35, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('$r / r_h$')
    ax.set_ylabel('$M(<r) / M_{total}$')
    ax.set_title('Plummer Cumulative Mass Distribution')
    ax.set_xlim(0, 5)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/profiles_plummer_density.png')
    plt.close()
    print("  ✓ Plummer density plot saved")

    # -------------------------------------------------------------------------
    # Print results (metrics already computed above)
    # -------------------------------------------------------------------------
    print(f"\n  Quantitative Results:")
    print(f"  ---------------------")
    print(f"  Half-mass radius (r_h):     {r_h:.4f}")
    print(f"  Median sampled radius:      {median_r:.4f}")
    print(f"  r_h error:                  {r_h_error:.2f}%  (target < 1%)")
    print(f"  Max CDF deviation:          {max_cdf_deviation:.4f}  (target < 0.02)")
    print(f"  Mean CDF deviation:         {mean_cdf_deviation:.4f}")
    print(f"  Overall:                    {'PASS' if passed else 'FAIL'}")

    return {
        'r_h_error': r_h_error,
        'max_cdf_deviation': max_cdf_deviation,
        'mean_cdf_deviation': mean_cdf_deviation,
        'passed': passed
    }


# =============================================================================
# King Profile Validation
# =============================================================================

def validate_king(output_dir: str):
    """Validate KingProfile implementation."""
    print("\n" + "="*60)
    print("KING PROFILE VALIDATION")
    print("="*60)

    # Test with W0=7 (typical globular cluster)
    W0 = 7.0
    r_c = 1.0
    profile = KingProfile.from_W0_rc(W0=W0, r_c=r_c)

    key = jax.random.PRNGKey(SEED)
    masses = jnp.ones(N_SAMPLES)
    positions = profile.sample_positions(masses, key)
    radii = jnp.linalg.norm(positions, axis=1)

    # -------------------------------------------------------------------------
    # Compute metrics FIRST (needed for plot annotations)
    # -------------------------------------------------------------------------
    max_r = float(jnp.max(radii))
    r_t = float(profile.r_t)
    truncation_ok = max_r <= r_t * 1.001  # Allow 0.1% numerical tolerance
    c_measured = np.log10(r_t / r_c)

    # CDF deviation metrics using precomputed CDF
    sorted_r, ecdf = compute_empirical_cdf(radii)
    # Interpolate theoretical CDF at sampled radii
    theoretical_cdf = jnp.interp(sorted_r, profile._r_grid, profile._cdf_grid)
    cdf_deviations = jnp.abs(ecdf - theoretical_cdf)
    max_cdf_deviation = float(jnp.max(cdf_deviations))
    mean_cdf_deviation = float(jnp.mean(cdf_deviations))

    # -------------------------------------------------------------------------
    # Figure 1: King ODE Solution & Density
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: ODE solution (psi vs xi)
    ax = axes[0]
    xi_grid, psi_grid = solve_king_profile(W0, xi_max=50.0, n_points=500)

    ax.plot(xi_grid, psi_grid, 'b-', lw=2)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.axvline(float(profile.r_t / r_c), color='r', ls='--', alpha=0.7, label=f'$\\xi_t = {float(profile.r_t/r_c):.2f}$')

    ax.set_xlabel('$\\xi = r/r_c$')
    ax.set_ylabel('$\\psi(\\xi)$')
    ax.set_title(f'King ODE Solution ($W_0 = {W0}$)')
    ax.set_xlim(0, 50)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Density profile
    ax = axes[1]
    r_grid = jnp.linspace(0.01, float(profile.r_t) * 0.99, 200)

    # Compute density using K-function formula
    xi = r_grid / r_c
    psi_vals = jnp.interp(xi, profile.xi_grid, profile.psi_grid, left=W0, right=0.0)
    K_W0 = king_K_function(W0)
    K_W0_minus_psi = king_K_function(W0 - psi_vals)
    rho_analytical = jnp.where(K_W0 > 1e-10, (K_W0 - K_W0_minus_psi) / K_W0, 0.0)

    # Computed density
    rho_computed = profile.density(r_grid)

    ax.semilogy(r_grid / r_c, rho_analytical, 'b-', lw=2, label='Analytical (K-function)')
    ax.semilogy(r_grid / r_c, rho_computed, 'r--', lw=2, label='density(r) method')

    # Histogram from samples
    r_max = float(profile.r_t)
    bins = np.linspace(0, r_max / r_c, 50)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(np.array(radii / r_c), bins=bins, density=False)
    dr = bins[1] - bins[0]
    shell_volume = 4 * np.pi * (bin_centers * r_c)**2 * dr * r_c
    rho_hist = hist / shell_volume
    # Normalize by matching to analytical curve at bin centers (robust median scaling)
    rho_analytical_at_bins = np.interp(bin_centers, np.array(r_grid / r_c), np.array(rho_analytical))
    valid = (rho_hist > 0) & (rho_analytical_at_bins > 1e-10)
    if np.any(valid):
        scale = np.median(rho_analytical_at_bins[valid] / rho_hist[valid])
        rho_hist = rho_hist * scale
    ax.semilogy(bin_centers[valid], rho_hist[valid], 'go', ms=4, alpha=0.6, label=f'Samples (N={N_SAMPLES})')

    ax.axvline(float(profile.r_t / r_c), color='gray', ls=':', alpha=0.5, label='$r_t$')
    ax.set_xlabel('$r / r_c$')
    ax.set_ylabel('$\\rho(r) / \\rho_0$')
    ax.set_title(f'King Density Profile ($W_0 = {W0}$)')
    ax.set_xlim(0, float(profile.r_t / r_c) * 1.1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add metrics text box
    metrics_text = (
        f"Validation Metrics:\n"
        f"Max CDF dev: {max_cdf_deviation:.4f}\n"
        f"Mean CDF dev: {mean_cdf_deviation:.4f}\n"
        f"$c = \\log_{{10}}(r_t/r_c)$: {c_measured:.3f}\n"
        f"Truncation: {'OK' if truncation_ok else 'FAIL'}"
    )
    ax.text(0.98, 0.60, metrics_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel C: Concentration effect
    ax = axes[2]
    W0_values = [3, 5, 7, 9]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(W0_values)))

    for W0_test, color in zip(W0_values, colors):
        profile_test = KingProfile.from_W0_rc(W0=W0_test, r_c=1.0)
        r_grid_test = jnp.linspace(0.01, float(profile_test.r_t) * 0.99, 200)
        rho_test = profile_test.density(r_grid_test)
        rho_test = rho_test / rho_test[0]  # Normalize
        ax.semilogy(r_grid_test / profile_test.r_c, rho_test, '-', lw=2,
                   color=color, label=f'$W_0 = {W0_test}$')

    ax.set_xlabel('$r / r_c$')
    ax.set_ylabel('$\\rho(r) / \\rho_0$')
    ax.set_title('King Profile: Concentration Effect')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(1e-6, 2)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/profiles_king_density.png')
    plt.close()
    print("  ✓ King density plot saved")

    # -------------------------------------------------------------------------
    # Print results (metrics already computed above)
    # -------------------------------------------------------------------------
    passed = truncation_ok and max_cdf_deviation < 0.02

    print(f"\n  Quantitative Results:")
    print(f"  ---------------------")
    print(f"  W0:                         {W0:.1f}")
    print(f"  Core radius (r_c):          {r_c:.4f}")
    print(f"  Tidal radius (r_t):         {r_t:.4f}")
    print(f"  Concentration c:            {c_measured:.3f}")
    print(f"  Max CDF deviation:          {max_cdf_deviation:.4f}  (target < 0.02)")
    print(f"  Mean CDF deviation:         {mean_cdf_deviation:.4f}")
    print(f"  Max sampled radius:         {max_r:.4f}")
    print(f"  Truncation test:            {'PASS' if truncation_ok else 'FAIL'}")
    print(f"  Overall:                    {'PASS' if passed else 'FAIL'}")

    return {
        'W0': W0,
        'r_t': r_t,
        'concentration': c_measured,
        'max_cdf_deviation': max_cdf_deviation,
        'mean_cdf_deviation': mean_cdf_deviation,
        'truncation_ok': truncation_ok,
        'passed': passed
    }


# =============================================================================
# EFF Profile Validation
# =============================================================================

def validate_eff(output_dir: str):
    """Validate EFFProfile implementation."""
    print("\n" + "="*60)
    print("EFF PROFILE VALIDATION")
    print("="*60)

    # Test with typical young cluster parameters
    a = 0.5  # Scale radius
    gamma = 3.0  # Power-law index
    r_t = 10.0  # Tidal radius
    profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

    key = jax.random.PRNGKey(SEED)
    masses = jnp.ones(N_SAMPLES)
    positions = profile.sample_positions(masses, key)
    radii = jnp.linalg.norm(positions, axis=1)

    # -------------------------------------------------------------------------
    # Compute metrics FIRST (needed for plot annotations)
    # -------------------------------------------------------------------------
    max_r = float(jnp.max(radii))
    truncation_ok = max_r <= r_t * 1.001

    # CDF deviation metrics using precomputed CDF
    sorted_r, ecdf = compute_empirical_cdf(radii)
    # Interpolate theoretical CDF at sampled radii
    theoretical_cdf = jnp.interp(sorted_r, profile._r_grid, profile._cdf_grid)
    cdf_deviations = jnp.abs(ecdf - theoretical_cdf)
    max_cdf_deviation = float(jnp.max(cdf_deviations))
    mean_cdf_deviation = float(jnp.mean(cdf_deviations))

    # Power-law slope verification (at r >> a, rho ~ r^(-gamma))
    r_grid_for_slope = jnp.linspace(0.01, r_t * 0.99, 200)
    r_outer = r_grid_for_slope[r_grid_for_slope > 3 * a]
    rho_outer = profile.density(r_outer)
    log_r = jnp.log10(r_outer)
    log_rho = jnp.log10(rho_outer + 1e-30)
    slope = float(jnp.sum((log_r - jnp.mean(log_r)) * (log_rho - jnp.mean(log_rho))) /
                  jnp.sum((log_r - jnp.mean(log_r))**2))
    measured_gamma = -slope
    gamma_error = abs(measured_gamma - gamma) / gamma * 100
    passed = truncation_ok and max_cdf_deviation < 0.02

    # -------------------------------------------------------------------------
    # Figure 1: EFF Density & Parameter Effects
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: Density profile
    ax = axes[0]
    r_grid = jnp.linspace(0.01, r_t * 0.99, 200)

    # Analytical EFF density
    rho_analytical = (1 + (r_grid / a)**2)**(-gamma / 2)
    rho_analytical = jnp.where(r_grid <= r_t, rho_analytical, 0.0)

    # Computed density
    rho_computed = profile.density(r_grid)

    ax.semilogy(r_grid / a, rho_analytical, 'b-', lw=2, label=f'Analytical: $(1 + r^2/a^2)^{{-\\gamma/2}}$')
    ax.semilogy(r_grid / a, rho_computed, 'r--', lw=2, label='density(r) method')

    # Histogram from samples
    bins = np.linspace(0, r_t / a, 50)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(np.array(radii / a), bins=bins, density=False)
    dr = bins[1] - bins[0]
    shell_volume = 4 * np.pi * (bin_centers * a)**2 * dr * a
    rho_hist = hist / shell_volume
    # Normalize by matching to analytical curve at bin centers (robust median scaling)
    rho_analytical_at_bins = np.interp(bin_centers, np.array(r_grid / a), np.array(rho_analytical))
    valid = (rho_hist > 0) & (rho_analytical_at_bins > 1e-10)
    if np.any(valid):
        scale = np.median(rho_analytical_at_bins[valid] / rho_hist[valid])
        rho_hist = rho_hist * scale
    ax.semilogy(bin_centers[valid], rho_hist[valid], 'go', ms=4, alpha=0.6, label=f'Samples (N={N_SAMPLES})')

    ax.axvline(r_t / a, color='gray', ls=':', alpha=0.5, label='$r_t$')
    ax.set_xlabel('$r / a$')
    ax.set_ylabel('$\\rho(r) / \\rho_0$')
    ax.set_title(f'EFF Density Profile ($\\gamma = {gamma}$)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Gamma effect
    ax = axes[1]
    gamma_values = [2.0, 2.5, 3.0, 3.5, 4.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(gamma_values)))

    for gamma_test, color in zip(gamma_values, colors):
        profile_test = EFFProfile(a=0.5, gamma=gamma_test, r_t=10.0)
        r_grid_test = jnp.linspace(0.01, 9.9, 200)
        rho_test = profile_test.density(r_grid_test)
        rho_test = rho_test / rho_test[0]
        ax.semilogy(r_grid_test / 0.5, rho_test, '-', lw=2,
                   color=color, label=f'$\\gamma = {gamma_test}$')

    ax.set_xlabel('$r / a$')
    ax.set_ylabel('$\\rho(r) / \\rho_0$')
    ax.set_title('EFF Profile: $\\gamma$ Effect')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.set_ylim(1e-4, 2)

    # Panel C: CDF comparison
    ax = axes[2]

    # Empirical CDF
    sorted_r, ecdf = compute_empirical_cdf(radii)

    # Precomputed CDF from profile
    ax.plot(profile._r_grid / a, profile._cdf_grid, 'b-', lw=2, label='Precomputed CDF')
    ax.plot(sorted_r[::100] / a, ecdf[::100], 'r.', ms=2, alpha=0.5, label='Empirical CDF')

    # Add metrics text box
    metrics_text = (
        f"Validation Metrics:\n"
        f"Max CDF dev: {max_cdf_deviation:.4f}\n"
        f"Mean CDF dev: {mean_cdf_deviation:.4f}\n"
        f"$\\gamma$ slope: {measured_gamma:.2f}\n"
        f"Truncation: {'OK' if truncation_ok else 'FAIL'}"
    )
    ax.text(0.98, 0.40, metrics_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('$r / a$')
    ax.set_ylabel('$M(<r) / M_{total}$')
    ax.set_title('EFF Cumulative Mass Distribution')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/profiles_eff_density.png')
    plt.close()
    print("  ✓ EFF density plot saved")

    # -------------------------------------------------------------------------
    # Print results (metrics already computed above)
    # -------------------------------------------------------------------------
    print(f"\n  Quantitative Results:")
    print(f"  ---------------------")
    print(f"  Scale radius (a):           {a:.4f}")
    print(f"  Power-law index (gamma):    {gamma:.1f}")
    print(f"  Tidal radius (r_t):         {r_t:.4f}")
    print(f"  Max CDF deviation:          {max_cdf_deviation:.4f}  (target < 0.02)")
    print(f"  Mean CDF deviation:         {mean_cdf_deviation:.4f}")
    print(f"  Measured outer slope:       {measured_gamma:.2f}")
    print(f"  Gamma error:                {gamma_error:.1f}%")
    print(f"  Max sampled radius:         {max_r:.4f}")
    print(f"  Truncation test:            {'PASS' if truncation_ok else 'FAIL'}")
    print(f"  Overall:                    {'PASS' if passed else 'FAIL'}")

    return {
        'gamma': gamma,
        'measured_gamma': measured_gamma,
        'gamma_error': gamma_error,
        'max_cdf_deviation': max_cdf_deviation,
        'mean_cdf_deviation': mean_cdf_deviation,
        'truncation_ok': truncation_ok,
        'passed': passed
    }


# =============================================================================
# Isotropy Validation
# =============================================================================

def validate_isotropy(output_dir: str):
    """Validate isotropic angular distribution for all profiles."""
    print("\n" + "="*60)
    print("ISOTROPY VALIDATION")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    profiles = [
        ('Plummer', PlummerProfile(r_h=1.0)),
        ('King', KingProfile.from_W0_rc(W0=7.0, r_c=1.0)),
        ('EFF', EFFProfile(a=0.5, gamma=3.0, r_t=10.0)),
    ]

    results = {}

    for col, (name, profile) in enumerate(profiles):
        key = jax.random.PRNGKey(SEED + col)
        masses = jnp.ones(N_SAMPLES)
        positions = profile.sample_positions(masses, key)

        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        r = jnp.linalg.norm(positions, axis=1)

        # cos(theta) should be uniform [-1, 1]
        cos_theta = z / (r + 1e-10)

        # phi should be uniform [0, 2*pi]
        phi = jnp.arctan2(y, x)
        phi = jnp.where(phi < 0, phi + 2 * jnp.pi, phi)

        # Top row: cos(theta) distribution
        ax = axes[0, col]
        ax.hist(np.array(cos_theta), bins=50, density=True, alpha=0.7, color='steelblue',
                edgecolor='black', linewidth=0.5)
        ax.axhline(0.5, color='red', ls='--', lw=2, label='Uniform expectation')
        ax.set_xlabel('$\\cos(\\theta)$')
        ax.set_ylabel('Probability density')
        ax.set_title(f'{name}: $\\cos(\\theta)$ Distribution')
        ax.set_xlim(-1, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom row: phi distribution
        ax = axes[1, col]
        ax.hist(np.array(phi), bins=50, density=True, alpha=0.7, color='coral',
                edgecolor='black', linewidth=0.5)
        ax.axhline(1 / (2 * np.pi), color='red', ls='--', lw=2, label='Uniform expectation')
        ax.set_xlabel('$\\phi$ [rad]')
        ax.set_ylabel('Probability density')
        ax.set_title(f'{name}: $\\phi$ Distribution')
        ax.set_xlim(0, 2 * np.pi)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # KS tests for isotropy
        cos_theta_ks, cos_theta_p = stats.kstest(np.array(cos_theta), 'uniform', args=(-1, 2))
        phi_ks, phi_p = stats.kstest(np.array(phi) / (2 * np.pi), 'uniform')

        results[name] = {
            'cos_theta_ks': cos_theta_ks,
            'cos_theta_p': cos_theta_p,
            'phi_ks': phi_ks,
            'phi_p': phi_p,
            'passed': cos_theta_p > 0.01 and phi_p > 0.01
        }

        print(f"\n  {name} Profile:")
        print(f"    cos(theta) KS stat:       {cos_theta_ks:.4f}, p-value: {cos_theta_p:.4f}")
        print(f"    phi KS stat:              {phi_ks:.4f}, p-value: {phi_p:.4f}")
        print(f"    Isotropy test:            {'PASS' if results[name]['passed'] else 'FAIL'}")

    plt.tight_layout()
    plt.savefig(f'{output_dir}/profiles_isotropy.png')
    plt.close()
    print("\n  ✓ Isotropy plot saved")

    return results


# =============================================================================
# Profile Comparison Panel
# =============================================================================

def plot_comparison_panel(output_dir: str):
    """Create comparison panel of all three profiles."""
    print("\n" + "="*60)
    print("PROFILE COMPARISON PANEL")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Create profiles with similar characteristic scales
    plummer = PlummerProfile(r_h=1.0)
    king = KingProfile.from_W0_rc(W0=7.0, r_c=0.3)  # r_c chosen so r_t ~ few r_h
    eff = EFFProfile(a=0.5, gamma=3.0, r_t=5.0)

    # Panel A: Density profiles
    ax = axes[0]
    r_grid = jnp.linspace(0.01, 5.0, 200)

    # Plummer
    rho_plummer = plummer.density(r_grid)
    rho_plummer = rho_plummer / rho_plummer[0]
    ax.semilogy(r_grid, rho_plummer, 'b-', lw=2, label='Plummer')

    # King
    r_king = jnp.linspace(0.01, float(king.r_t) * 0.99, 200)
    rho_king = king.density(r_king)
    rho_king = rho_king / rho_king[0]
    ax.semilogy(r_king, rho_king, 'r-', lw=2, label=f'King ($W_0={7}$)')

    # EFF
    r_eff = jnp.linspace(0.01, float(eff.r_t) * 0.99, 200)
    rho_eff = eff.density(r_eff)
    rho_eff = rho_eff / rho_eff[0]
    ax.semilogy(r_eff, rho_eff, 'g-', lw=2, label=f'EFF ($\\gamma={3}$)')

    ax.set_xlabel('Radius [length units]')
    ax.set_ylabel('$\\rho(r) / \\rho_0$')
    ax.set_title('Density Profile Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(1e-4, 2)

    # Panel B: Sample XY projections
    ax = axes[1]

    key = jax.random.PRNGKey(SEED)
    N_plot = 2000
    masses = jnp.ones(N_plot)

    key, subkey = jax.random.split(key)
    pos_plummer = plummer.sample_positions(masses, subkey)
    ax.scatter(pos_plummer[:, 0], pos_plummer[:, 1], s=1, alpha=0.3, c='blue', label='Plummer')

    key, subkey = jax.random.split(key)
    pos_king = king.sample_positions(masses, subkey)
    ax.scatter(pos_king[:, 0], pos_king[:, 1], s=1, alpha=0.3, c='red', label='King')

    key, subkey = jax.random.split(key)
    pos_eff = eff.sample_positions(masses, subkey)
    ax.scatter(pos_eff[:, 0], pos_eff[:, 1], s=1, alpha=0.3, c='green', label='EFF')

    ax.set_xlabel('x [length units]')
    ax.set_ylabel('y [length units]')
    ax.set_title('XY Projection of Sampled Positions')
    ax.legend(markerscale=5)
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/profiles_comparison.png')
    plt.close()
    print("  ✓ Comparison panel saved")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all profile validation tests."""
    print("\n" + "="*70)
    print("PROGENAX SPATIAL PROFILE VALIDATION")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Sample size: N = {N_SAMPLES}")
    print(f"Random seed: {SEED}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run validations
    plummer_results = validate_plummer(OUTPUT_DIR)
    king_results = validate_king(OUTPUT_DIR)
    eff_results = validate_eff(OUTPUT_DIR)
    isotropy_results = validate_isotropy(OUTPUT_DIR)
    plot_comparison_panel(OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    QUANTITATIVE RESULTS                         │")
    print("├─────────────────────────────────────────────────────────────────┤")

    print(f"│ Plummer Profile:                                                │")
    print(f"│   r_h error:           {plummer_results['r_h_error']:6.2f}%  (target < 1%)             │")
    print(f"│   Max CDF deviation:   {plummer_results['max_cdf_deviation']:6.4f}   (target < 0.02)           │")
    print(f"│   Status:              {'✓ PASS' if plummer_results['passed'] else '✗ FAIL'}                                   │")

    print("├─────────────────────────────────────────────────────────────────┤")

    print(f"│ King Profile:                                                   │")
    print(f"│   Max CDF deviation:   {king_results['max_cdf_deviation']:6.4f}   (target < 0.02)           │")
    print(f"│   Concentration c:     {king_results['concentration']:6.3f}                                  │")
    print(f"│   Truncation:          {'✓ OK' if king_results['truncation_ok'] else '✗ FAIL'}                                     │")
    print(f"│   Status:              {'✓ PASS' if king_results['passed'] else '✗ FAIL'}                                   │")

    print("├─────────────────────────────────────────────────────────────────┤")

    print(f"│ EFF Profile:                                                    │")
    print(f"│   Max CDF deviation:   {eff_results['max_cdf_deviation']:6.4f}   (target < 0.02)           │")
    print(f"│   gamma slope:         {eff_results['measured_gamma']:6.2f}   (expected: {eff_results['gamma']:.1f})            │")
    print(f"│   Truncation:          {'✓ OK' if eff_results['truncation_ok'] else '✗ FAIL'}                                     │")
    print(f"│   Status:              {'✓ PASS' if eff_results['passed'] else '✗ FAIL'}                                   │")

    print("├─────────────────────────────────────────────────────────────────┤")

    print(f"│ Isotropy Tests:                                                 │")
    for name, res in isotropy_results.items():
        status = '✓ PASS' if res['passed'] else '✗ FAIL'
        print(f"│   {name:8s}:            {status}                                   │")

    print("└─────────────────────────────────────────────────────────────────┘")

    # Overall pass/fail
    all_passed = (
        plummer_results['passed'] and
        king_results['passed'] and
        eff_results['passed'] and
        all(r['passed'] for r in isotropy_results.values())
    )

    print(f"\n{'='*70}")
    if all_passed:
        print("  ✓ ALL VALIDATION TESTS PASSED")
    else:
        print("  ✗ SOME VALIDATION TESTS FAILED")
    print(f"{'='*70}")

    print(f"\nPlots saved to: {OUTPUT_DIR}/")
    print("  - profiles_plummer_density.png")
    print("  - profiles_king_density.png")
    print("  - profiles_eff_density.png")
    print("  - profiles_isotropy.png")
    print("  - profiles_comparison.png")

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
