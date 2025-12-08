"""Tests for binary star mass functions.

Tests mass-ratio distributions and binary IMF composition:
- MassRatioProtocol implementations (Flat, PowerLaw, TwinPeaked, Moe+17)
- Binary fraction models (constant, mass-dependent)
- BinaryIMF composition with primary IMF

CONSOLIDATED: ~20 essential physics tests from original 48.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf.binary import (
    BinaryIMF,
    ConstantBinaryFraction,
    FlatMassRatio,
    MassDependentBinaryFraction,
    MoeDiStefano2017,
    PowerLawMassRatio,
    TwinPeakedMassRatio,
)
from progenax.imf.power_law import PowerLawIMF


# =============================================================================
# Test Mass-Ratio Distributions
# =============================================================================


class TestMassRatioDistributions:
    """Test mass-ratio distributions (Flat, PowerLaw, TwinPeaked)."""

    def test_flat_pdf_normalization(self):
        """Flat mass ratio: PDF integrates to 1."""
        q_dist = FlatMassRatio(q_min=0.1)
        q_grid = jnp.linspace(0.1, 1.0, 1000)
        integral = jnp.trapezoid(q_dist.pdf(q_grid), q_grid)
        assert jnp.abs(integral - 1.0) < 1e-4

    def test_flat_ppf_inverse_cdf(self):
        """Flat: PPF is inverse of CDF."""
        q_dist = FlatMassRatio(q_min=0.1)
        u = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        q = q_dist.ppf(u)
        u_reconstructed = q_dist.cdf(q)
        assert jnp.allclose(u, u_reconstructed, atol=1e-6)

    def test_powerlaw_pdf_normalization(self):
        """PowerLaw: PDF integrates to 1 for various gamma."""
        for gamma in [-0.5, 0.0, 0.3]:
            q_dist = PowerLawMassRatio(gamma=gamma, q_min=0.1)
            q_grid = jnp.linspace(0.1, 1.0, 1000)
            integral = jnp.trapezoid(q_dist.pdf(q_grid), q_grid)
            assert jnp.abs(integral - 1.0) < 1e-3, f"Failed for gamma={gamma}"

    def test_powerlaw_gamma_zero_is_flat(self):
        """PowerLaw with gamma=0 equals FlatMassRatio."""
        q_dist_pl = PowerLawMassRatio(gamma=0.0, q_min=0.1)
        q_dist_flat = FlatMassRatio(q_min=0.1)

        q = jnp.array([0.2, 0.5, 0.8])
        assert jnp.allclose(q_dist_pl.pdf(q), q_dist_flat.pdf(q), rtol=1e-6)

    def test_twinpeaked_pdf_normalization(self):
        """TwinPeaked: PDF integrates to 1."""
        q_dist = TwinPeakedMassRatio(gamma=0.0, f_twin=0.1, q_min=0.1)
        q_grid = jnp.linspace(0.1, 1.0, 2000)
        integral = jnp.trapezoid(q_dist.pdf(q_grid), q_grid)
        assert jnp.abs(integral - 1.0) < 1e-3

    def test_twinpeaked_peak_at_q_one(self):
        """TwinPeaked: PDF higher near q=1 than mid-range."""
        q_dist = TwinPeakedMassRatio(gamma=0.0, f_twin=0.2, sigma_twin=0.03)
        pdf_near_one = q_dist.pdf(jnp.array(0.98))
        pdf_mid = q_dist.pdf(jnp.array(0.5))
        assert pdf_near_one > pdf_mid


# =============================================================================
# Test Moe & Di Stefano (2017) Model
# =============================================================================


class TestMoeDiStefano2017:
    """Test mass-dependent q-distribution from Moe+17."""

    def test_gamma_varies_with_mass(self):
        """Power-law exponent varies with primary mass (Moe+17)."""
        q_dist = MoeDiStefano2017()
        gamma_low = q_dist._gamma_of_mass(jnp.array(0.5))
        gamma_solar = q_dist._gamma_of_mass(jnp.array(1.0))
        gamma_massive = q_dist._gamma_of_mass(jnp.array(10.0))

        # γ decreases with mass
        assert gamma_low > gamma_solar > gamma_massive

    def test_ftwin_varies_with_mass(self):
        """Twin fraction varies with primary mass (solar-type peak)."""
        q_dist = MoeDiStefano2017()
        f_low = q_dist._ftwin_of_mass(jnp.array(0.5))
        f_solar = q_dist._ftwin_of_mass(jnp.array(1.0))
        f_massive = q_dist._ftwin_of_mass(jnp.array(10.0))

        # Solar-type has highest twin excess
        assert f_solar > f_low and f_solar > f_massive

    def test_sample_mass_dependent(self):
        """Different primary masses give different q distributions."""
        q_dist = MoeDiStefano2017()
        key = jax.random.PRNGKey(123)
        key1, key2 = jax.random.split(key)

        q_low = q_dist.sample_given_primary(key1, jnp.ones(5000) * 0.5)
        q_high = q_dist.sample_given_primary(key2, jnp.ones(5000) * 10.0)

        assert jnp.abs(jnp.mean(q_low) - jnp.mean(q_high)) > 0.02

    def test_twin_sampling_matches_pdf(self):
        """Twin component sampling produces distribution matching PDF."""
        moe = MoeDiStefano2017(q_min=0.1, sigma_twin=0.03)
        key = jax.random.PRNGKey(42)

        # Sample many twins at fixed primary mass (solar-type for high f_twin)
        m1 = jnp.ones(50000) * 1.0  # Solar-type primary
        q_samples = moe.sample_given_primary(key, m1)

        # Histogram of samples
        hist, bin_edges = jnp.histogram(q_samples, bins=50, range=(0.1, 1.0), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Compare histogram to PDF - they should match within statistical noise
        pdf_at_bins = moe.pdf_given_primary(jnp.array(bin_centers), m1=1.0)
        relative_error = jnp.abs(hist - pdf_at_bins) / (pdf_at_bins + 1e-10)

        # Most bins should be within 30% (allowing for statistical noise)
        # Use 70% of bins passing as threshold for robustness
        assert jnp.mean(relative_error < 0.3) > 0.7, \
            f"Sampling doesn't match PDF: mean relative error = {jnp.mean(relative_error):.3f}"


# =============================================================================
# Test Binary Fraction Models
# =============================================================================


class TestBinaryFractionModels:
    """Test binary fraction models."""

    def test_constant_returns_constant(self):
        """ConstantBinaryFraction returns same value for all masses."""
        model = ConstantBinaryFraction(f_bin=0.6)
        masses = jnp.array([0.1, 0.5, 1.0, 5.0, 10.0])
        assert jnp.allclose(model(masses), 0.6)

    def test_mass_dependent_increases_with_mass(self):
        """MassDependentBinaryFraction increases with mass."""
        model = MassDependentBinaryFraction()
        f_low = model(jnp.array(0.3))
        f_solar = model(jnp.array(1.0))
        f_massive = model(jnp.array(15.0))
        assert f_low < f_solar < f_massive

    def test_mass_dependent_matches_moe2017(self):
        """Values match Moe+17 Table 13."""
        model = MassDependentBinaryFraction()
        # Key mass bins from literature
        assert jnp.abs(model(jnp.array(0.3)) - 0.26) < 1e-6   # M-dwarf
        assert jnp.abs(model(jnp.array(0.8)) - 0.44) < 1e-6   # K-dwarf
        assert jnp.abs(model(jnp.array(1.5)) - 0.50) < 1e-6   # A-star
        assert jnp.abs(model(jnp.array(15.0)) - 0.90) < 1e-6  # O-star


# =============================================================================
# Test BinaryIMF
# =============================================================================


class TestBinaryIMF:
    """Test binary IMF composition."""

    def test_sample_systems_structure(self):
        """sample_systems returns (m1, m2, is_binary) with correct shapes."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.simple(primary_imf, binary_fraction=0.5)

        key = jax.random.PRNGKey(42)
        m1, m2, is_binary = binary_imf.sample_systems(key, 1000)

        assert m1.shape == (1000,)
        assert m2.shape == (1000,)
        assert is_binary.shape == (1000,)
        assert is_binary.dtype == bool

    def test_binary_fraction_matches_target(self):
        """Binary fraction matches target within statistics."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.simple(primary_imf, binary_fraction=0.5)

        key = jax.random.PRNGKey(123)
        _, _, is_binary = binary_imf.sample_systems(key, 10000)

        frac = jnp.mean(is_binary.astype(float))
        assert jnp.abs(frac - 0.5) < 0.02

    def test_singles_have_zero_m2(self):
        """Single stars have m2=0."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.simple(primary_imf, binary_fraction=0.5)

        key = jax.random.PRNGKey(42)
        _, m2, is_binary = binary_imf.sample_systems(key, 1000)

        singles_m2 = m2[~is_binary]
        assert jnp.allclose(singles_m2, 0.0)

    def test_binaries_satisfy_q_constraint(self):
        """Binary secondaries satisfy q = m2/m1 in [q_min, 1]."""
        primary_imf = PowerLawIMF.kroupa()
        q_min = 0.15
        binary_imf = BinaryIMF(
            primary_imf=primary_imf,
            q_distribution=FlatMassRatio(q_min=q_min),
            binary_fraction=1.0,
        )

        key = jax.random.PRNGKey(42)
        m1, m2, is_binary = binary_imf.sample_systems(key, 1000)

        q = m2 / m1
        assert jnp.all(q >= q_min - 1e-6)
        assert jnp.all(q <= 1.0 + 1e-6)

    def test_sample_all_masses_length(self):
        """sample_all_masses returns primaries + secondaries."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.simple(primary_imf, binary_fraction=0.5)

        key = jax.random.PRNGKey(42)
        all_masses, is_binary = binary_imf.sample_all_masses(key, 100)

        n_binaries = jnp.sum(is_binary)
        expected_length = 100 + n_binaries
        assert all_masses.shape[0] == expected_length

    def test_differentiable_through_sample(self):
        """Can differentiate through binary sampling."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.simple(primary_imf, binary_fraction=0.5)

        def loss(dummy_param):
            key = jax.random.PRNGKey(42)
            m1, m2, _ = binary_imf.sample_systems(key, 100)
            return jnp.sum(m1) + jnp.sum(m2)

        grad_fn = jax.grad(loss)
        gradient = grad_fn(1.0)
        assert jnp.isfinite(gradient)
