#!/usr/bin/env python
"""
IMF Extensions validation plots.

Produces diagnostic figures for:
1. Environment-conditioned IMFs (EnvironmentIMF, CustomEnvironmentIMF)
2. IGIMF (Integrated Galactic IMF)
3. Binary IMFs (MoeDiStefano2017 mass-ratio distributions)

Run: python scripts/validate_imf_extensions.py
Output: validation/plots/extensions/
"""
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from progenax.imf import PowerLawIMF
from progenax.imf.environment import (
    EnvironmentIMF,
    CustomEnvironmentIMF,
    GasEnvironment,
    massive_star_fraction,
)
from progenax.imf.igimf import IGIMF
from progenax.imf.binary import (
    BinaryIMF,
    MoeDiStefano2017,
    FlatMassRatio,
    PowerLawMassRatio,
    TwinPeakedMassRatio,
)


def ensure_output_dir(output_dir: str = "validation/plots/extensions"):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# =============================================================================
# Environment IMF Plots
# =============================================================================


def plot_environment_imf_comparison(output_dir: str):
    """Compare IMFs across different environments."""
    print("\n1. Environment IMF Comparison:")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Create IMFs for different environments
    environments = {
        "Solar Neighborhood": EnvironmentIMF.solar_neighborhood(),
        "Dense Clump": EnvironmentIMF.dense_clump(),
        "Low Metallicity (Z=0.1)": EnvironmentIMF.low_metallicity(Z=0.1),
        "Starburst (SFR=100)": EnvironmentIMF.starburst(sfr=100.0),
    }

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(environments)))

    # Left: PDF comparison
    ax = axes[0]
    m_grid = jnp.logspace(-2, 2, 500)

    for (name, imf), color in zip(environments.items(), colors):
        pdf = jnp.exp(imf.logpdf(m_grid))
        alpha = imf.alpha_high
        ax.loglog(m_grid, m_grid * pdf, '-', color=color, lw=2,
                  label=f'{name} (α={alpha:.2f})')

    ax.set_xlabel('Mass [M$_\\odot$]', fontsize=12)
    ax.set_ylabel('m × ξ(m)', fontsize=12)
    ax.set_title('Environment-Conditioned IMFs', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.01, 100)

    # Right: High-mass slope vs environment parameter
    ax = axes[1]

    # Vary density
    log_n_values = np.linspace(3, 8, 50)
    alphas_density = []
    for log_n in log_n_values:
        env = GasEnvironment(n_H=10**log_n, T_gas=10.0, Z=1.0)
        imf = EnvironmentIMF(env)
        alphas_density.append(imf.alpha_high)

    ax.plot(log_n_values, alphas_density, 'b-', lw=2, label='Varying density (Z=1)')

    # Vary metallicity
    Z_values = np.linspace(0.01, 2.0, 50)
    alphas_Z = []
    for Z in Z_values:
        env = GasEnvironment(n_H=1e4, T_gas=10.0, Z=Z)
        imf = EnvironmentIMF(env)
        alphas_Z.append(imf.alpha_high)

    ax2 = ax.twiny()
    ax2.plot(Z_values, alphas_Z, 'r--', lw=2, label='Varying Z (n=10⁴)')

    ax.set_xlabel('log₁₀(n_H) [cm⁻³]', fontsize=12, color='blue')
    ax2.set_xlabel('Metallicity [Z$_\\odot$]', fontsize=12, color='red')
    ax.set_ylabel('High-mass slope α₃', fontsize=12)
    ax.set_title('IMF Slope vs Environment', fontsize=14)
    ax.axhline(2.3, color='gray', ls=':', label='Kroupa α=2.3')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/environment_imf_comparison.png', dpi=150)
    plt.close()
    print(f"  ✓ Environment IMF comparison saved")


def plot_massive_star_fraction(output_dir: str):
    """Plot massive star fraction across environments."""
    print("\n2. Massive Star Fraction:")

    fig, ax = plt.subplots(figsize=(8, 6))

    key = jax.random.PRNGKey(42)

    # Standard Kroupa
    kroupa = PowerLawIMF.kroupa()
    frac_kroupa = massive_star_fraction(kroupa, m_threshold=8.0, key=key)

    # Environment IMFs
    env_configs = [
        ("Solar", GasEnvironment.solar_neighborhood()),
        ("Dense (n=10⁶)", GasEnvironment.dense_clump()),
        ("Low Z (0.1)", GasEnvironment.low_metallicity(0.1)),
        ("Starburst", GasEnvironment.starburst()),
        ("Primordial", GasEnvironment.primordial()),
    ]

    names = ["Kroupa (ref)"] + [name for name, _ in env_configs]
    fractions = [frac_kroupa]
    alphas = [2.3]

    for name, env in env_configs:
        imf = EnvironmentIMF(env)
        frac = massive_star_fraction(imf, m_threshold=8.0, key=key)
        fractions.append(frac)
        alphas.append(imf.alpha_high)

    x = np.arange(len(names))
    colors = ['gray'] + list(plt.cm.viridis(np.linspace(0, 0.8, len(env_configs))))
    bars = ax.bar(x, np.array(fractions) * 100, color=colors, edgecolor='black')

    # Add alpha labels on bars
    for i, (bar, alpha) in enumerate(zip(bars, alphas)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'α={alpha:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Massive Star Fraction (M > 8 M$_\\odot$) [%]', fontsize=12)
    ax.set_title('Massive Star Fraction by Environment', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/massive_star_fraction.png', dpi=150)
    plt.close()
    print(f"  ✓ Massive star fraction plot saved")


# =============================================================================
# IGIMF Plots
# =============================================================================


def plot_igimf_vs_stellar(output_dir: str):
    """Compare IGIMF to stellar IMF at different SFRs."""
    print("\n3. IGIMF vs Stellar IMF:")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    key = jax.random.PRNGKey(42)
    stellar_imf = PowerLawIMF.kroupa()

    # Left: PDF comparison
    ax = axes[0]

    # Sample from stellar IMF
    m_stellar = stellar_imf.sample(key, 50000)

    # Sample from IGIMF at different SFRs
    sfr_values = [0.001, 0.1, 1.0, 100.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sfr_values)))

    # Histogram bins
    log_bins = np.logspace(-2, 2, 60)
    bin_centers = np.sqrt(log_bins[:-1] * log_bins[1:])

    # Stellar IMF histogram
    counts_stellar, _ = np.histogram(np.array(m_stellar), bins=log_bins, density=True)
    ax.loglog(bin_centers, bin_centers * counts_stellar, 'k-', lw=3,
              label='Stellar IMF (Kroupa)', alpha=0.7)

    for sfr, color in zip(sfr_values, colors):
        igimf = IGIMF(stellar_imf=stellar_imf, sfr=sfr)
        key, subkey = jax.random.split(key)
        m_igimf = igimf.sample(subkey, 20000)
        m_igimf = m_igimf[m_igimf > 0]  # Remove zeros

        counts, _ = np.histogram(np.array(m_igimf), bins=log_bins, density=True)
        ax.loglog(bin_centers, bin_centers * counts, '-', color=color, lw=2,
                  label=f'IGIMF (SFR={sfr} M$_\\odot$/yr)')

    ax.set_xlabel('Mass [M$_\\odot$]', fontsize=12)
    ax.set_ylabel('m × ξ(m)', fontsize=12)
    ax.set_title('IGIMF vs Stellar IMF', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.01, 100)

    # Right: Effective slope vs SFR (Weidner & Kroupa 2005)
    ax = axes[1]

    # Use more SFR points for smoother curve (need 50k+ samples for reliable slope)
    sfr_range = np.logspace(-3, 3, 20)
    slopes = []

    for sfr in sfr_range:
        igimf = IGIMF(stellar_imf=stellar_imf, sfr=sfr)
        key, subkey = jax.random.split(key)
        # Need many samples for reliable slope estimation in [10,100] M_sun range
        slope = igimf.effective_slope_high_mass(key=subkey, n_samples=50000)
        slopes.append(float(slope))

    ax.semilogx(sfr_range, slopes, 'b-', lw=2, marker='o', ms=4, label='IGIMF measured')
    ax.axhline(2.3, color='gray', ls='--', label='Stellar IMF α=2.3')
    ax.axhline(3.3, color='red', ls=':', label='IGIMF limit α≈3.3')

    # Mark literature SFR values (Weidner & Kroupa 2005)
    lit_sfrs = [0.001, 0.2, 1.9, 10.0, 100.0]  # Dwarf, LMC, MW, M82, ULIRG
    lit_labels = ['Dwarf', 'LMC', 'MW', 'M82', 'ULIRG']
    for sfr, label in zip(lit_sfrs, lit_labels):
        ax.axvline(sfr, color='green', ls=':', alpha=0.5)
        ax.annotate(label, xy=(sfr, 2.05), fontsize=8, ha='center', color='green')

    ax.set_xlabel('Star Formation Rate [M$_\\odot$/yr]', fontsize=12)
    ax.set_ylabel('Effective High-Mass Slope α', fontsize=12)
    ax.set_title('IGIMF Slope vs SFR (Weidner & Kroupa 2005)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2.0, 3.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/igimf_comparison.png', dpi=150)
    plt.close()
    print(f"  ✓ IGIMF comparison saved")


def plot_igimf_galaxy_types(output_dir: str):
    """Compare IGIMF for different galaxy types using literature SFR values.

    References:
        Weidner & Kroupa (2005) ApJ 625, 754
        Pflamm-Altenburg et al. (2007) ApJ 671, 1550
    """
    print("\n4. IGIMF Galaxy Types:")

    fig, ax = plt.subplots(figsize=(10, 6))

    key = jax.random.PRNGKey(42)
    stellar_imf = PowerLawIMF.kroupa()

    # Literature-based SFR values (Weidner & Kroupa 2005)
    galaxy_types = {
        "Dwarf (Fornax)": IGIMF(stellar_imf=stellar_imf, sfr=0.001),   # ~10^-3 M☉/yr
        "LMC": IGIMF(stellar_imf=stellar_imf, sfr=0.2),                 # ~0.2 M☉/yr
        "Milky Way": IGIMF(stellar_imf=stellar_imf, sfr=1.9),           # ~1.9 M☉/yr
        "Starburst (M82)": IGIMF(stellar_imf=stellar_imf, sfr=10.0),    # ~10 M☉/yr
    }

    colors = ['blue', 'cyan', 'green', 'red']
    log_bins = np.logspace(-2, 2, 50)
    bin_centers = np.sqrt(log_bins[:-1] * log_bins[1:])

    for (name, igimf), color in zip(galaxy_types.items(), colors):
        key, subkey = jax.random.split(key)
        masses = igimf.sample(subkey, 20000)
        masses = masses[masses > 0]

        counts, _ = np.histogram(np.array(masses), bins=log_bins, density=True)
        ax.loglog(bin_centers, bin_centers * counts, '-', color=color, lw=2.5,
                  label=f'{name} (SFR={igimf.sfr})')

    # Reference stellar IMF
    m_grid = jnp.logspace(-2, 2, 200)
    pdf_stellar = jnp.exp(stellar_imf.logpdf(m_grid))
    ax.loglog(m_grid, m_grid * pdf_stellar, 'k--', lw=2, alpha=0.5,
              label='Stellar IMF (reference)')

    ax.set_xlabel('Mass [M$_\\odot$]', fontsize=12)
    ax.set_ylabel('m × ξ(m)', fontsize=12)
    ax.set_title('IGIMF by Galaxy Type (Weidner & Kroupa 2005)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.01, 100)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/igimf_galaxy_types.png', dpi=150)
    plt.close()
    print(f"  ✓ IGIMF galaxy types saved")


# =============================================================================
# Binary IMF Plots
# =============================================================================


def plot_mass_ratio_distributions(output_dir: str):
    """Compare different mass-ratio distributions."""
    print("\n5. Mass-Ratio Distributions:")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    key = jax.random.PRNGKey(42)
    N = 50000

    # Left: PDF comparison
    ax = axes[0]
    q_grid = jnp.linspace(0.1, 1.0, 200)

    distributions = {
        "Flat": FlatMassRatio(q_min=0.1),
        "Power-law (γ=-0.1)": PowerLawMassRatio(gamma=-0.1, q_min=0.1),
        "Power-law (γ=0.3)": PowerLawMassRatio(gamma=0.3, q_min=0.1),
        "Twin-peaked": TwinPeakedMassRatio(gamma=0.0, f_twin=0.1, q_min=0.1),
    }

    colors = plt.cm.tab10(np.arange(len(distributions)))

    for (name, dist), color in zip(distributions.items(), colors):
        pdf = dist.pdf(q_grid)
        ax.plot(q_grid, pdf, '-', color=color, lw=2, label=name)

    ax.set_xlabel('Mass Ratio q = M₂/M₁', fontsize=12)
    ax.set_ylabel('PDF p(q)', fontsize=12)
    ax.set_title('Mass-Ratio Distribution PDFs', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 1.0)

    # Right: MoeDiStefano2017 mass-dependent
    ax = axes[1]

    moe = MoeDiStefano2017(q_min=0.1, sigma_twin=0.03)

    primary_masses = [0.5, 1.0, 2.0, 10.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(primary_masses)))

    for m1, color in zip(primary_masses, colors):
        key, subkey = jax.random.split(key)
        m1_arr = jnp.ones(N) * m1
        q_samples = moe.sample_given_primary(subkey, m1_arr)

        # Histogram
        hist, bin_edges = np.histogram(np.array(q_samples), bins=50,
                                       range=(0.1, 1.0), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.plot(bin_centers, hist, '-', color=color, lw=2,
                label=f'M₁ = {m1} M$_\\odot$')

        # Analytical PDF
        pdf = moe.pdf_given_primary(jnp.array(bin_centers), m1=m1)
        ax.plot(bin_centers, np.array(pdf), '--', color=color, lw=1.5, alpha=0.7)

    ax.set_xlabel('Mass Ratio q = M₂/M₁', fontsize=12)
    ax.set_ylabel('PDF p(q|M₁)', fontsize=12)
    ax.set_title('Moe & Di Stefano (2017): Mass-Dependent q', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 1.0)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/mass_ratio_distributions.png', dpi=150)
    plt.close()
    print(f"  ✓ Mass-ratio distributions saved")


def plot_binary_fraction(output_dir: str):
    """Plot binary fraction vs primary mass."""
    print("\n6. Binary Fraction:")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    key = jax.random.PRNGKey(42)

    # Left: Binary fraction vs mass
    ax = axes[0]

    from progenax.imf.binary import MassDependentBinaryFraction, ConstantBinaryFraction

    m_grid = jnp.logspace(-1, 2, 100)

    # Mass-dependent (Moe+17)
    f_bin_moe = MassDependentBinaryFraction()(m_grid)
    ax.semilogx(m_grid, f_bin_moe, 'b-', lw=2.5, label='Moe+17 (mass-dependent)')

    # Constant
    ax.axhline(0.5, color='gray', ls='--', lw=2, label='Constant (f=0.5)')

    ax.set_xlabel('Primary Mass [M$_\\odot$]', fontsize=12)
    ax.set_ylabel('Binary Fraction', fontsize=12)
    ax.set_title('Binary Fraction vs Primary Mass', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Right: BinaryIMF mass distribution
    ax = axes[1]

    primary_imf = PowerLawIMF.kroupa()
    binary_imf = BinaryIMF.moe2017(primary_imf)

    m1, m2, is_binary = binary_imf.sample_systems(key, 50000)

    # All masses
    all_primaries = m1
    all_secondaries = m2[is_binary]

    log_bins = np.logspace(-2, 2, 50)
    bin_centers = np.sqrt(log_bins[:-1] * log_bins[1:])

    # Histograms
    counts_primary, _ = np.histogram(np.array(all_primaries), bins=log_bins, density=True)
    counts_secondary, _ = np.histogram(np.array(all_secondaries), bins=log_bins, density=True)

    ax.loglog(bin_centers, bin_centers * counts_primary, 'b-', lw=2,
              label='Primary masses')
    ax.loglog(bin_centers, bin_centers * counts_secondary, 'r--', lw=2,
              label='Secondary masses')

    # Combined
    all_masses = jnp.concatenate([all_primaries, all_secondaries])
    counts_all, _ = np.histogram(np.array(all_masses), bins=log_bins, density=True)
    ax.loglog(bin_centers, bin_centers * counts_all, 'k-', lw=2.5, alpha=0.7,
              label='All stellar masses')

    ax.set_xlabel('Mass [M$_\\odot$]', fontsize=12)
    ax.set_ylabel('m × ξ(m)', fontsize=12)
    ax.set_title('BinaryIMF Mass Distribution (Moe+17)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.01, 100)

    # Add stats
    n_binary = int(jnp.sum(is_binary))
    f_binary = n_binary / len(is_binary)
    ax.text(0.95, 0.95, f'Binary fraction: {f_binary:.1%}\nN_binary: {n_binary}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/binary_fraction.png', dpi=150)
    plt.close()
    print(f"  ✓ Binary fraction plot saved")


def plot_twin_sampling_validation(output_dir: str):
    """Validate MoeDiStefano2017 twin sampling matches PDF."""
    print("\n7. Twin Sampling Validation:")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    key = jax.random.PRNGKey(42)
    moe = MoeDiStefano2017(q_min=0.1, sigma_twin=0.03)

    # Left: Solar-type (high twin fraction)
    ax = axes[0]
    m1 = 1.0
    m1_arr = jnp.ones(100000) * m1
    q_samples = moe.sample_given_primary(key, m1_arr)

    # Histogram
    hist, bin_edges = np.histogram(np.array(q_samples), bins=80,
                                   range=(0.1, 1.0), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0],
           alpha=0.7, color='steelblue', label='Samples (N=100,000)')

    # Analytical PDF
    q_grid = jnp.linspace(0.1, 1.0, 200)
    pdf = moe.pdf_given_primary(q_grid, m1=m1)
    ax.plot(q_grid, pdf, 'r-', lw=2.5, label='Analytical PDF')

    ax.set_xlabel('Mass Ratio q', fontsize=12)
    ax.set_ylabel('PDF', fontsize=12)
    ax.set_title(f'Solar-Type Primary (M₁ = {m1} M$_\\odot$, f_twin=10%)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Zoom on twin peak
    axins = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
    axins.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0],
              alpha=0.7, color='steelblue')
    axins.plot(q_grid, pdf, 'r-', lw=2)
    axins.set_xlim(0.9, 1.0)
    axins.set_ylim(0, max(hist[bin_centers > 0.9]) * 1.2)
    axins.set_title('Twin Peak (q > 0.9)', fontsize=9)

    # Right: O-star (low twin fraction)
    ax = axes[1]
    m1 = 20.0
    key, subkey = jax.random.split(key)
    m1_arr = jnp.ones(100000) * m1
    q_samples = moe.sample_given_primary(subkey, m1_arr)

    # Histogram
    hist, bin_edges = np.histogram(np.array(q_samples), bins=80,
                                   range=(0.1, 1.0), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0],
           alpha=0.7, color='coral', label='Samples (N=100,000)')

    # Analytical PDF
    pdf = moe.pdf_given_primary(q_grid, m1=m1)
    ax.plot(q_grid, pdf, 'darkred', lw=2.5, ls='-', label='Analytical PDF')

    ax.set_xlabel('Mass Ratio q', fontsize=12)
    ax.set_ylabel('PDF', fontsize=12)
    ax.set_title(f'O-Star Primary (M₁ = {m1} M$_\\odot$, f_twin=3%)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/twin_sampling_validation.png', dpi=150)
    plt.close()
    print(f"  ✓ Twin sampling validation saved")


# =============================================================================
# Main
# =============================================================================


def main():
    """Generate all validation plots for IMF extensions."""
    print("=" * 60)
    print("IMF Extensions Validation Plots")
    print("=" * 60)

    output_dir = ensure_output_dir()

    # Environment IMF
    print("\n" + "-" * 40)
    print("ENVIRONMENT-CONDITIONED IMFs")
    print("-" * 40)
    plot_environment_imf_comparison(output_dir)
    plot_massive_star_fraction(output_dir)

    # IGIMF
    print("\n" + "-" * 40)
    print("INTEGRATED GALACTIC IMF (IGIMF)")
    print("-" * 40)
    plot_igimf_vs_stellar(output_dir)
    plot_igimf_galaxy_types(output_dir)

    # Binary IMF
    print("\n" + "-" * 40)
    print("BINARY IMFs")
    print("-" * 40)
    plot_mass_ratio_distributions(output_dir)
    plot_binary_fraction(output_dir)
    plot_twin_sampling_validation(output_dir)

    print("\n" + "=" * 60)
    print("All validation plots complete!")
    print(f"Output: {output_dir}/")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
