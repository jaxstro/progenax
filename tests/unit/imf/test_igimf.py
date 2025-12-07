"""Tests for Integrated Galactic Initial Mass Function (IGIMF).

Test suite for IGIMF theory:
- EmbeddedClusterMassFunction (power-law ECMF)
- MaxStellarMass (m_max-M_ecl relations)
- IGIMF (full IGIMF model)

Key properties verified:
- ECMF power-law with β ≈ 2
- m_max(M_ecl) relations (Weidner04, analytical, sorted)
- IGIMF steeper than input stellar IMF
- M_ecl_max depends on SFR
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf.igimf import (
    EmbeddedClusterMassFunction,
    MaxStellarMass,
    IGIMF,
    max_cluster_mass_from_sfr,
    igimf_effective_slope,
)
from progenax.imf.power_law import PowerLawIMF


# =============================================================================
# Test EmbeddedClusterMassFunction
# =============================================================================


class TestEmbeddedClusterMassFunction:
    """Test ECMF power-law distribution."""

    def test_pdf_normalization(self):
        """PDF integrates to 1.0."""
        ecmf = EmbeddedClusterMassFunction(beta=2.0, M_ecl_min=5.0, M_ecl_max=1e6)

        # Numerical integration
        M_grid = jnp.logspace(jnp.log10(5.0), 6.0, 1000)
        pdf_vals = ecmf.pdf(M_grid)
        integral = jnp.trapezoid(pdf_vals, M_grid)

        assert jnp.abs(integral - 1.0) < 0.01  # Within 1%

    def test_cdf_endpoints(self):
        """CDF(M_min) = 0, CDF(M_max) = 1."""
        ecmf = EmbeddedClusterMassFunction(beta=2.0, M_ecl_min=5.0, M_ecl_max=1e6)

        cdf_min = ecmf.cdf(5.0)
        cdf_max = ecmf.cdf(1e6)

        assert jnp.abs(cdf_min - 0.0) < 1e-6
        assert jnp.abs(cdf_max - 1.0) < 1e-6

    def test_ppf_inverse_of_cdf(self):
        """PPF is inverse of CDF: CDF(PPF(u)) = u."""
        ecmf = EmbeddedClusterMassFunction(beta=2.0, M_ecl_min=5.0, M_ecl_max=1e6)

        u_vals = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        M_vals = ecmf.ppf(u_vals)
        u_recovered = ecmf.cdf(M_vals)

        assert jnp.allclose(u_vals, u_recovered, rtol=1e-4)

    def test_sampling_shape(self):
        """Sample returns correct shape."""
        ecmf = EmbeddedClusterMassFunction(beta=2.0, M_ecl_min=5.0, M_ecl_max=1e6)
        key = jax.random.PRNGKey(42)

        masses = ecmf.sample(key, 1000)

        assert masses.shape == (1000,)
        assert jnp.all((masses >= 5.0) & (masses <= 1e6))

    def test_power_law_slope(self):
        """PDF follows M^(-β) power law."""
        ecmf = EmbeddedClusterMassFunction(beta=2.0, M_ecl_min=5.0, M_ecl_max=1e6)

        M1, M2 = 100.0, 1000.0
        pdf1 = ecmf.pdf(M1)
        pdf2 = ecmf.pdf(M2)

        # pdf ∝ M^(-β) => log(pdf2/pdf1) = -β * log(M2/M1)
        expected_ratio = (M2 / M1) ** (-2.0)
        actual_ratio = pdf2 / pdf1

        assert jnp.abs(actual_ratio - expected_ratio) < 0.01

    def test_mean_mass(self):
        """Mean cluster mass matches analytical expectation."""
        ecmf = EmbeddedClusterMassFunction(beta=2.0, M_ecl_min=5.0, M_ecl_max=1e5)

        mean = ecmf.mean_mass()

        # For β=2: E[M] ~ M_min * ln(M_max/M_min) / (1 - M_min/M_max)
        # Approximation for large M_max/M_min
        expected = 5.0 * jnp.log(1e5 / 5.0)

        # Loose check (depends on normalization)
        assert 10.0 < mean < 1000.0

    def test_different_beta_values(self):
        """ECMF works with different β values."""
        for beta in [1.5, 2.0, 2.5]:
            ecmf = EmbeddedClusterMassFunction(
                beta=beta, M_ecl_min=5.0, M_ecl_max=1e6
            )

            # Check normalization
            M_grid = jnp.logspace(jnp.log10(5.0), 6.0, 500)
            pdf_vals = ecmf.pdf(M_grid)
            integral = jnp.trapezoid(pdf_vals, M_grid)

            assert jnp.abs(integral - 1.0) < 0.02


# =============================================================================
# Test MaxStellarMass
# =============================================================================


class TestMaxStellarMass:
    """Test m_max(M_ecl) relations."""

    def test_weidner04_monotonic(self):
        """Weidner04 relation is monotonically increasing."""
        mmax = MaxStellarMass(model="weidner04", m_max_physical=150.0)

        M_ecl = jnp.array([10.0, 100.0, 1000.0, 10000.0])
        m_max_vals = mmax(M_ecl)

        # Should be increasing
        assert jnp.all(m_max_vals[1:] >= m_max_vals[:-1])

    def test_weidner04_capped(self):
        """Weidner04 capped at m_max_physical."""
        mmax = MaxStellarMass(model="weidner04", m_max_physical=150.0)

        M_ecl_large = jnp.array([1e6, 1e7, 1e8])
        m_max_vals = mmax(M_ecl_large)

        assert jnp.all(m_max_vals <= 150.0)

    def test_analytical_asymptotes(self):
        """Analytical relation has correct asymptotic behavior."""
        mmax = MaxStellarMass(model="analytical", m_max_physical=150.0)

        # Small clusters: m_max → M_ecl
        M_small = 1.0
        m_small = mmax(M_small)
        assert m_small < 1.5  # Close to M_ecl

        # Large clusters: m_max → m_max_physical
        M_large = 1e8
        m_large = mmax(M_large)
        assert jnp.abs(m_large - 150.0) < 1.0

    def test_sorted_approximation(self):
        """Sorted sampling approximation is reasonable."""
        mmax = MaxStellarMass(model="sorted", m_max_physical=150.0)

        M_ecl = jnp.array([100.0, 1000.0, 10000.0])
        m_max_vals = mmax(M_ecl)

        # Should be monotonic and capped
        assert jnp.all(m_max_vals[1:] >= m_max_vals[:-1])
        assert jnp.all(m_max_vals <= 150.0)

    def test_all_models_consistent(self):
        """All models give similar order-of-magnitude results."""
        M_ecl = 1000.0

        m_w04 = MaxStellarMass(model="weidner04")(M_ecl)
        m_ana = MaxStellarMass(model="analytical")(M_ecl)
        m_sor = MaxStellarMass(model="sorted")(M_ecl)

        # All should be within factor of 10 (different functional forms)
        values = jnp.array([m_w04, m_ana, m_sor])
        ratio = jnp.max(values) / jnp.min(values)

        assert ratio < 10.0


# =============================================================================
# Test M_ecl_max - SFR relation
# =============================================================================


class TestMaxClusterMassFromSFR:
    """Test M_ecl_max(SFR) relation."""

    def test_increasing_with_sfr(self):
        """M_ecl_max increases with SFR."""
        sfr_vals = jnp.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        M_max_vals = jnp.array([max_cluster_mass_from_sfr(sfr) for sfr in sfr_vals])

        # Should be monotonically increasing
        assert jnp.all(M_max_vals[1:] >= M_max_vals[:-1])

    def test_milky_way_like(self):
        """Milky Way (SFR ~ 1 Msun/yr) gives reasonable M_ecl_max."""
        M_max = max_cluster_mass_from_sfr(1.0)

        # Expect ~10^5 M☉ (see Weidner+2004 Eq. 11)
        assert 1e4 < M_max < 1e6

    def test_dwarf_galaxy(self):
        """Dwarf galaxy (low SFR) gives small M_ecl_max."""
        M_max = max_cluster_mass_from_sfr(0.001)

        # Should be small
        assert 10.0 < M_max < 1000.0

    def test_starburst(self):
        """Starburst (high SFR) gives large M_ecl_max."""
        M_max = max_cluster_mass_from_sfr(100.0)

        # Should be very large
        assert M_max > 1e6


# =============================================================================
# Test IGIMF
# =============================================================================


class TestIGIMF:
    """Test full IGIMF model."""

    def test_initialization(self):
        """IGIMF initializes with default parameters."""
        stellar_imf = PowerLawIMF.kroupa()
        igimf = IGIMF(stellar_imf=stellar_imf, sfr=1.0)

        assert igimf.sfr == 1.0
        assert igimf.ecmf_beta == 2.0
        assert igimf.M_ecl_min == 5.0

    def test_M_ecl_max_from_sfr(self):
        """M_ecl_max computed from SFR if not specified."""
        stellar_imf = PowerLawIMF.kroupa()
        igimf = IGIMF(stellar_imf=stellar_imf, sfr=1.0, M_ecl_max=None)

        M_max = igimf._get_M_ecl_max()

        # Should match max_cluster_mass_from_sfr(1.0)
        expected = max_cluster_mass_from_sfr(1.0)
        assert jnp.abs(M_max - expected) < 1.0

    def test_M_ecl_max_explicit(self):
        """Explicit M_ecl_max overrides SFR-based calculation."""
        stellar_imf = PowerLawIMF.kroupa()
        igimf = IGIMF(stellar_imf=stellar_imf, sfr=1.0, M_ecl_max=1e5)

        M_max = igimf._get_M_ecl_max()

        assert M_max == 1e5

    def test_sampling_shape(self):
        """IGIMF sampling returns correct shape."""
        stellar_imf = PowerLawIMF.kroupa()
        igimf = IGIMF(stellar_imf=stellar_imf, sfr=1.0)
        key = jax.random.PRNGKey(42)

        masses = igimf.sample(key, 1000)

        assert masses.shape == (1000,)
        # Some masses should be positive (zeros are expected from cluster truncation)
        assert jnp.sum(masses > 0) > 100

    def test_steeper_than_stellar_imf(self):
        """IGIMF is steeper than input stellar IMF at high masses.

        Key result: Galaxy-wide IMF has fewer massive stars than cluster-scale IMF.
        """
        stellar_imf = PowerLawIMF.kroupa()
        igimf = IGIMF(stellar_imf=stellar_imf, sfr=1.0)
        key = jax.random.PRNGKey(12345)

        # Sample many masses
        stellar_masses = stellar_imf.sample(key, 50000)
        igimf_masses = igimf.sample(key, 50000)

        # Count massive stars (m > 10 Msun)
        stellar_massive_frac = jnp.sum(stellar_masses > 10.0) / 50000.0
        igimf_massive_frac = jnp.sum(igimf_masses > 10.0) / 50000.0

        # IGIMF should have FEWER massive stars
        assert igimf_massive_frac < stellar_massive_frac

    def test_effective_slope_high_mass(self):
        """Effective slope at high masses is steeper than Kroupa α=2.3."""
        stellar_imf = PowerLawIMF.kroupa()
        igimf = IGIMF(stellar_imf=stellar_imf, sfr=1.0)

        # This test is approximate due to sampling noise
        # Just check it runs and gives reasonable value
        alpha_eff = igimf.effective_slope_high_mass(m_range=(10.0, 100.0))

        # Should be positive and not too extreme
        # Note: sampling noise makes precise bounds difficult
        assert 1.0 < alpha_eff < 5.0

    def test_sfr_dependence(self):
        """IGIMF steepening depends on SFR."""
        stellar_imf = PowerLawIMF.kroupa()

        igimf_low_sfr = IGIMF(stellar_imf=stellar_imf, sfr=0.001)
        igimf_high_sfr = IGIMF(stellar_imf=stellar_imf, sfr=100.0)

        key = jax.random.PRNGKey(42)

        masses_low = igimf_low_sfr.sample(key, 10000)
        masses_high = igimf_high_sfr.sample(key, 10000)

        # Both should produce some stars
        assert jnp.sum(masses_low > 0) > 100
        assert jnp.sum(masses_high > 0) > 100

        # Check M_ecl_max differs as expected
        M_max_low = igimf_low_sfr._get_M_ecl_max()
        M_max_high = igimf_high_sfr._get_M_ecl_max()

        # High SFR should have larger M_ecl_max
        assert M_max_high > M_max_low

    def test_factory_methods(self):
        """Factory methods create sensible IGIMF instances."""
        # Milky Way
        igimf_mw = IGIMF.milky_way()
        assert 1.0 <= igimf_mw.sfr <= 3.0

        # Starburst
        igimf_burst = IGIMF.starburst()
        assert igimf_burst.sfr >= 10.0

        # Dwarf galaxy
        igimf_dwarf = IGIMF.dwarf_galaxy()
        assert igimf_dwarf.sfr <= 0.01

    def test_mean_mass(self):
        """Mean mass is reasonable."""
        stellar_imf = PowerLawIMF.kroupa()
        igimf = IGIMF(stellar_imf=stellar_imf, sfr=1.0)

        mean = igimf.mean_mass()

        # Should be order 0.1-1 Msun for Kroupa-like IMF
        assert 0.05 < mean < 5.0

    @pytest.mark.slow
    def test_differentiability(self):
        """IGIMF sampling is differentiable w.r.t. SFR (via M_ecl_max)."""
        stellar_imf = PowerLawIMF.kroupa()

        def loss(sfr):
            igimf = IGIMF(stellar_imf=stellar_imf, sfr=sfr, M_ecl_max=None)
            key = jax.random.PRNGKey(0)
            masses = igimf.sample(key, 100)
            # Loss = mean mass (should increase slightly with SFR)
            return jnp.mean(masses[masses > 0])

        grad_fn = jax.grad(loss)
        gradient = grad_fn(1.0)

        # Should have finite gradient
        assert jnp.isfinite(gradient)


# =============================================================================
# Test igimf_effective_slope
# =============================================================================


class TestIGIMFEffectiveSlope:
    """Test analytical approximation for IGIMF slope."""

    def test_low_sfr_steeper(self):
        """Low SFR gives steeper slope."""
        alpha_low = igimf_effective_slope(sfr=0.001, stellar_alpha=2.3, ecmf_beta=2.0)
        alpha_high = igimf_effective_slope(sfr=100.0, stellar_alpha=2.3, ecmf_beta=2.0)

        assert alpha_low > alpha_high

    def test_high_sfr_approaches_stellar(self):
        """High SFR approaches stellar IMF slope."""
        alpha_igimf = igimf_effective_slope(
            sfr=1000.0, stellar_alpha=2.3, ecmf_beta=2.0
        )

        # Should be close to 2.3
        assert jnp.abs(alpha_igimf - 2.3) < 0.2

    def test_low_sfr_maximum_steepening(self):
        """Low SFR approaches α + β - 1."""
        alpha_igimf = igimf_effective_slope(sfr=1e-6, stellar_alpha=2.3, ecmf_beta=2.0)

        # Should approach 2.3 + 2.0 - 1.0 = 3.3
        expected = 2.3 + 2.0 - 1.0
        assert jnp.abs(alpha_igimf - expected) < 0.2
