#!/usr/bin/env python
"""
IMF validation plots.

Produces diagnostic figures for Chabrier, Kroupa, and Salpeter IMFs:
1. PDF × m vs m (mass distribution shape)
2. High-mass tail (power-law slope verification)
3. CDF vs empirical CDF (sampling validation)
"""
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from progenax.imf import ChabrierIMF, PowerLawIMF


def plot_imf_validation(imf, name: str, output_dir: str = "validation/plots"):
    """Generate validation plots for an IMF."""
    os.makedirs(output_dir, exist_ok=True)

    key = jax.random.PRNGKey(42)
    N_samples = 100000
    masses = imf.sample(key, N_samples)

    # 1. PDF × m vs m (mass distribution)
    fig, ax = plt.subplots(figsize=(8, 6))

    m_grid = jnp.logspace(jnp.log10(imf.m_min), jnp.log10(imf.m_max), 500)
    pdf_grid = jnp.exp(imf.logpdf(m_grid))

    ax.loglog(m_grid, m_grid * pdf_grid, 'b-', lw=2, label='Analytical m×ξ(m)')

    # Histogram
    log_bins = np.logspace(np.log10(float(imf.m_min)), np.log10(float(imf.m_max)), 50)
    counts, edges = np.histogram(np.array(masses), bins=log_bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ax.loglog(centers, centers * counts, 'ro', ms=4, alpha=0.7, label=f'Samples (N={N_samples})')

    ax.set_xlabel('Mass [M$_\\odot$]', fontsize=12)
    ax.set_ylabel('m × ξ(m)', fontsize=12)
    ax.set_title(f'{name} IMF: Mass Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name.lower()}_pdf.png', dpi=150)
    plt.close()
    print(f"  ✓ {name} PDF plot saved")

    # 2. High-mass tail (slope verification)
    fig, ax = plt.subplots(figsize=(8, 6))

    m_high = m_grid[m_grid > 2.0]
    pdf_high = jnp.exp(imf.logpdf(m_high))

    ax.loglog(m_high, pdf_high, 'b-', lw=2, label='ξ(m)')

    # Reference slope line
    if hasattr(imf, 'alpha'):
        alpha = imf.alpha
    elif hasattr(imf, 'exponents'):
        alpha = imf.exponents[-1]  # Last segment slope
    else:
        alpha = 2.35
    m_ref = 5.0
    pdf_ref = float(jnp.exp(imf.logpdf(jnp.array(m_ref))))
    m_line = jnp.array([2.0, 50.0])
    pdf_line = pdf_ref * (m_line / m_ref) ** (-alpha)
    ax.loglog(m_line, pdf_line, 'r--', lw=1.5, label=f'Slope α={alpha}')

    ax.set_xlabel('Mass [M$_\\odot$]', fontsize=12)
    ax.set_ylabel('ξ(m)', fontsize=12)
    ax.set_title(f'{name} IMF: High-Mass Tail', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name.lower()}_tail.png', dpi=150)
    plt.close()
    print(f"  ✓ {name} tail plot saved")

    # 3. CDF comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    cdf_analytical = imf.cdf(m_grid)

    # Empirical CDF
    masses_sorted = jnp.sort(masses)
    ecdf = jnp.arange(1, N_samples + 1) / N_samples

    ax.semilogx(m_grid, cdf_analytical, 'b-', lw=2, label='Analytical CDF')
    ax.semilogx(masses_sorted[::100], ecdf[::100], 'r.', ms=2, alpha=0.5, label='Empirical CDF')

    ax.set_xlabel('Mass [M$_\\odot$]', fontsize=12)
    ax.set_ylabel('F(m)', fontsize=12)
    ax.set_title(f'{name} IMF: CDF Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name.lower()}_cdf.png', dpi=150)
    plt.close()
    print(f"  ✓ {name} CDF plot saved")


def main():
    """Generate validation plots for all IMFs."""
    print("=" * 60)
    print("IMF Validation Plots")
    print("=" * 60)

    # Chabrier (2003)
    print("\nChabrier (2003) - log₁₀-based system IMF:")
    chabrier = ChabrierIMF()
    print(f"  Mean mass: {float(chabrier.mean_mass()):.3f} M☉")
    plot_imf_validation(chabrier, "Chabrier")

    # Kroupa (2001)
    print("\nKroupa (2001):")
    kroupa = PowerLawIMF.kroupa()
    print(f"  Mean mass: {float(kroupa.mean_mass()):.3f} M☉")
    plot_imf_validation(kroupa, "Kroupa")

    # Salpeter (1955)
    print("\nSalpeter (1955):")
    salpeter = PowerLawIMF.salpeter()
    print(f"  Mean mass: {float(salpeter.mean_mass()):.3f} M☉")
    plot_imf_validation(salpeter, "Salpeter")

    print("\n" + "=" * 60)
    print("All validation plots complete!")
    print(f"Output: validation/plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
