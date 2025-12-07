"""Tests for binary star mass functions.

Tests mass-ratio distributions and binary IMF composition following TDD:
- MassRatioProtocol implementations
- Binary fraction models
- BinaryIMF composition with primary IMF
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf.base import BaseIMF
from progenax.imf.binary import (
    BinaryIMF,
    ConstantBinaryFraction,
    FlatMassRatio,
    MassDependentBinaryFraction,
    MassRatioProtocol,
    MoeDiStefano2017,
    PowerLawMassRatio,
    TwinPeakedMassRatio,
)
from progenax.imf.power_law import PowerLawIMF


# =============================================================================
# Test FlatMassRatio
# =============================================================================


class TestFlatMassRatio:
    """Test uniform mass-ratio distribution."""

    def test_initialization(self):
        """Test default initialization."""
        q_dist = FlatMassRatio()
        assert q_dist.q_min == 0.1

        q_dist = FlatMassRatio(q_min=0.2)
        assert q_dist.q_min == 0.2

    def test_pdf_normalization(self):
        """PDF integrates to 1."""
        q_dist = FlatMassRatio(q_min=0.1)
        q_grid = jnp.linspace(0.1, 1.0, 1000)
        pdf_grid = q_dist.pdf(q_grid)
        integral = jnp.trapezoid(pdf_grid, q_grid)
        assert jnp.abs(integral - 1.0) < 1e-4

    def test_pdf_uniform(self):
        """PDF is constant in [q_min, 1]."""
        q_dist = FlatMassRatio(q_min=0.2)
        q_grid = jnp.linspace(0.2, 1.0, 100)
        pdf_grid = q_dist.pdf(q_grid)
        assert jnp.allclose(pdf_grid, 1.0 / (1.0 - 0.2), rtol=1e-6)

    def test_cdf_bounds(self):
        """CDF is 0 at q_min, 1 at q=1."""
        q_dist = FlatMassRatio(q_min=0.1)
        assert jnp.abs(q_dist.cdf(0.1)) < 1e-6
        assert jnp.abs(q_dist.cdf(1.0) - 1.0) < 1e-6

    def test_ppf_inverse_cdf(self):
        """PPF is inverse of CDF."""
        q_dist = FlatMassRatio(q_min=0.1)
        u = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        q = q_dist.ppf(u)
        u_reconstructed = q_dist.cdf(q)
        assert jnp.allclose(u, u_reconstructed, atol=1e-6)

    def test_sample_range(self):
        """Samples are in [q_min, 1]."""
        q_dist = FlatMassRatio(q_min=0.15)
        key = jax.random.PRNGKey(42)
        samples = q_dist.sample(key, 1000)
        assert jnp.all(samples >= 0.15)
        assert jnp.all(samples <= 1.0)

    def test_sample_uniform_distribution(self):
        """Large sample follows uniform distribution."""
        q_dist = FlatMassRatio(q_min=0.1)
        key = jax.random.PRNGKey(123)
        samples = q_dist.sample(key, 10000)

        # Check mean (should be ~0.55)
        expected_mean = (0.1 + 1.0) / 2.0
        assert jnp.abs(jnp.mean(samples) - expected_mean) < 0.01

    def test_protocol_compliance(self):
        """Check MassRatioProtocol compliance."""
        q_dist = FlatMassRatio(q_min=0.1)
        assert isinstance(q_dist, MassRatioProtocol)
        assert hasattr(q_dist, 'pdf')
        assert hasattr(q_dist, 'cdf')
        assert hasattr(q_dist, 'ppf')
        assert hasattr(q_dist, 'sample')


# =============================================================================
# Test PowerLawMassRatio
# =============================================================================


class TestPowerLawMassRatio:
    """Test power-law mass-ratio distribution."""

    def test_initialization(self):
        """Test initialization with different gamma."""
        q_dist = PowerLawMassRatio(gamma=0.0, q_min=0.1)
        assert q_dist.gamma == 0.0
        assert q_dist.q_min == 0.1

        q_dist = PowerLawMassRatio(gamma=-0.5, q_min=0.2)
        assert q_dist.gamma == -0.5

    def test_pdf_normalization(self):
        """PDF integrates to 1 for various gamma."""
        for gamma in [-0.5, 0.0, 0.3, 0.5]:
            q_dist = PowerLawMassRatio(gamma=gamma, q_min=0.1)
            q_grid = jnp.linspace(0.1, 1.0, 1000)
            pdf_grid = q_dist.pdf(q_grid)
            integral = jnp.trapezoid(pdf_grid, q_grid)
            assert jnp.abs(integral - 1.0) < 1e-3, f"Failed for gamma={gamma}"

    def test_gamma_zero_is_flat(self):
        """gamma=0 should match FlatMassRatio."""
        q_dist_pl = PowerLawMassRatio(gamma=0.0, q_min=0.1)
        q_dist_flat = FlatMassRatio(q_min=0.1)

        q = jnp.array([0.2, 0.5, 0.8])
        pdf_pl = q_dist_pl.pdf(q)
        pdf_flat = q_dist_flat.pdf(q)
        assert jnp.allclose(pdf_pl, pdf_flat, rtol=1e-6)

    def test_cdf_bounds(self):
        """CDF is 0 at q_min, 1 at q=1."""
        q_dist = PowerLawMassRatio(gamma=0.3, q_min=0.1)
        assert jnp.abs(q_dist.cdf(0.1)) < 1e-6
        assert jnp.abs(q_dist.cdf(1.0) - 1.0) < 1e-6

    def test_ppf_inverse_cdf(self):
        """PPF is inverse of CDF."""
        q_dist = PowerLawMassRatio(gamma=0.3, q_min=0.1)
        u = jnp.linspace(0.0, 1.0, 11)
        q = q_dist.ppf(u)
        u_reconstructed = q_dist.cdf(q)
        assert jnp.allclose(u, u_reconstructed, atol=1e-5)

    def test_sample_range(self):
        """Samples are in [q_min, 1]."""
        q_dist = PowerLawMassRatio(gamma=-0.1, q_min=0.1)
        key = jax.random.PRNGKey(42)
        samples = q_dist.sample(key, 1000)
        assert jnp.all(samples >= 0.1)
        assert jnp.all(samples <= 1.0)

    def test_gamma_minus_one_edge_case(self):
        """Test gamma=-1 edge case (log integral)."""
        q_dist = PowerLawMassRatio(gamma=-1.0, q_min=0.1)

        # Should still normalize
        q_grid = jnp.linspace(0.1, 1.0, 1000)
        pdf_grid = q_dist.pdf(q_grid)
        integral = jnp.trapezoid(pdf_grid, q_grid)
        assert jnp.abs(integral - 1.0) < 1e-3

    def test_protocol_compliance(self):
        """Check MassRatioProtocol compliance."""
        q_dist = PowerLawMassRatio(gamma=0.0)
        assert isinstance(q_dist, MassRatioProtocol)


# =============================================================================
# Test TwinPeakedMassRatio
# =============================================================================


class TestTwinPeakedMassRatio:
    """Test twin-peaked mass-ratio distribution."""

    def test_initialization(self):
        """Test default initialization."""
        q_dist = TwinPeakedMassRatio()
        assert q_dist.gamma == 0.0
        assert q_dist.f_twin == 0.1
        assert q_dist.sigma_twin == 0.03
        assert q_dist.q_min == 0.1

    def test_pdf_normalization(self):
        """PDF integrates to 1."""
        q_dist = TwinPeakedMassRatio(gamma=0.0, f_twin=0.1, q_min=0.1)
        q_grid = jnp.linspace(0.1, 1.0, 2000)
        pdf_grid = q_dist.pdf(q_grid)
        integral = jnp.trapezoid(pdf_grid, q_grid)
        assert jnp.abs(integral - 1.0) < 1e-3

    def test_twin_peak_at_q_one(self):
        """Twin peak is centered at q=1."""
        q_dist = TwinPeakedMassRatio(gamma=0.0, f_twin=0.2, sigma_twin=0.03)

        # PDF should be higher near q=1
        pdf_near_one = q_dist.pdf(jnp.array(0.98))
        pdf_mid = q_dist.pdf(jnp.array(0.5))
        assert pdf_near_one > pdf_mid

    def test_cdf_bounds(self):
        """CDF is 0 at q_min, 1 at q=1."""
        q_dist = TwinPeakedMassRatio(gamma=0.3, f_twin=0.1)
        assert jnp.abs(q_dist.cdf(0.1)) < 1e-5
        assert jnp.abs(q_dist.cdf(1.0) - 1.0) < 1e-5

    def test_ppf_inverse_cdf(self):
        """PPF is inverse of CDF (via Newton)."""
        q_dist = TwinPeakedMassRatio(gamma=0.0, f_twin=0.1)
        u = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        q = q_dist.ppf(u)
        u_reconstructed = q_dist.cdf(q)
        assert jnp.allclose(u, u_reconstructed, atol=1e-4)

    def test_sample_range(self):
        """Samples are in [q_min, 1]."""
        q_dist = TwinPeakedMassRatio(f_twin=0.15)
        key = jax.random.PRNGKey(42)
        samples = q_dist.sample(key, 1000)
        assert jnp.all(samples >= 0.1)
        assert jnp.all(samples <= 1.0)

    def test_protocol_compliance(self):
        """Check MassRatioProtocol compliance."""
        q_dist = TwinPeakedMassRatio()
        assert isinstance(q_dist, MassRatioProtocol)


# =============================================================================
# Test MoeDiStefano2017
# =============================================================================


class TestMoeDiStefano2017:
    """Test mass-dependent Moe & Di Stefano (2017) model."""

    def test_initialization(self):
        """Test default initialization."""
        q_dist = MoeDiStefano2017()
        assert q_dist.q_min == 0.1
        assert q_dist.sigma_twin == 0.03

    def test_gamma_varies_with_mass(self):
        """Power-law exponent varies with primary mass."""
        q_dist = MoeDiStefano2017()

        gamma_low = q_dist._gamma_of_mass(jnp.array(0.5))   # M-dwarf
        gamma_solar = q_dist._gamma_of_mass(jnp.array(1.0)) # Solar
        gamma_massive = q_dist._gamma_of_mass(jnp.array(10.0)) # O-star

        # Should follow Moe+17: γ decreases with mass
        assert gamma_low > gamma_solar  # 0.4 > 0.3
        assert gamma_solar > gamma_massive  # 0.3 > -0.5

    def test_ftwin_varies_with_mass(self):
        """Twin fraction varies with primary mass."""
        q_dist = MoeDiStefano2017()

        f_low = q_dist._ftwin_of_mass(jnp.array(0.5))
        f_solar = q_dist._ftwin_of_mass(jnp.array(1.0))
        f_massive = q_dist._ftwin_of_mass(jnp.array(10.0))

        # Solar-type should have highest twin excess
        assert f_solar > f_low
        assert f_solar > f_massive

    def test_sample_given_primary_range(self):
        """Samples are in [q_min, 1]."""
        q_dist = MoeDiStefano2017(q_min=0.1)
        key = jax.random.PRNGKey(42)
        m1 = jnp.ones(1000) * 1.0  # 1000 solar-mass primaries

        q_samples = q_dist.sample_given_primary(key, m1)
        assert jnp.all(q_samples >= 0.1)
        assert jnp.all(q_samples <= 1.0)

    def test_sample_mass_dependent(self):
        """Different primary masses give different q distributions."""
        q_dist = MoeDiStefano2017()
        key = jax.random.PRNGKey(123)

        # Low-mass primaries
        key1, key2 = jax.random.split(key)
        m1_low = jnp.ones(5000) * 0.5
        q_low = q_dist.sample_given_primary(key1, m1_low)

        # High-mass primaries
        m1_high = jnp.ones(5000) * 10.0
        q_high = q_dist.sample_given_primary(key2, m1_high)

        # Different distributions
        assert jnp.abs(jnp.mean(q_low) - jnp.mean(q_high)) > 0.02

    def test_pdf_given_primary(self):
        """PDF varies with primary mass."""
        q_dist = MoeDiStefano2017()
        q_grid = jnp.linspace(0.1, 1.0, 100)

        pdf_solar = q_dist.pdf_given_primary(q_grid, m1=1.0)
        pdf_massive = q_dist.pdf_given_primary(q_grid, m1=10.0)

        # Should be different
        assert not jnp.allclose(pdf_solar, pdf_massive, rtol=0.1)


# =============================================================================
# Test Binary Fraction Models
# =============================================================================


class TestConstantBinaryFraction:
    """Test constant binary fraction model."""

    def test_initialization(self):
        """Test default and custom initialization."""
        model = ConstantBinaryFraction()
        assert model.f_bin == 0.5

        model = ConstantBinaryFraction(f_bin=0.7)
        assert model.f_bin == 0.7

    def test_call_returns_constant(self):
        """Returns constant for all masses."""
        model = ConstantBinaryFraction(f_bin=0.6)
        masses = jnp.array([0.1, 0.5, 1.0, 5.0, 10.0])
        f_bin = model(masses)
        assert jnp.allclose(f_bin, 0.6)


class TestMassDependentBinaryFraction:
    """Test mass-dependent binary fraction from Moe+17."""

    def test_increases_with_mass(self):
        """Binary fraction increases with mass."""
        model = MassDependentBinaryFraction()

        f_low = model(jnp.array(0.3))
        f_solar = model(jnp.array(1.0))
        f_massive = model(jnp.array(15.0))

        assert f_low < f_solar < f_massive

    def test_values_match_moe2017(self):
        """Check values match Moe+17 Table 13."""
        model = MassDependentBinaryFraction()

        # Test specific mass bins
        assert jnp.abs(model(jnp.array(0.05)) - 0.22) < 1e-6  # VLM
        assert jnp.abs(model(jnp.array(0.3)) - 0.26) < 1e-6   # M-dwarf
        assert jnp.abs(model(jnp.array(0.8)) - 0.44) < 1e-6   # K-dwarf
        assert jnp.abs(model(jnp.array(1.5)) - 0.50) < 1e-6   # A-star
        assert jnp.abs(model(jnp.array(3.0)) - 0.60) < 1e-6   # B-star
        assert jnp.abs(model(jnp.array(7.0)) - 0.80) < 1e-6   # early B
        assert jnp.abs(model(jnp.array(15.0)) - 0.90) < 1e-6  # O-star


# =============================================================================
# Test BinaryIMF
# =============================================================================


class TestBinaryIMF:
    """Test binary IMF composition."""

    def test_initialization_defaults(self):
        """Test default initialization (Moe+17 model)."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF(primary_imf=primary_imf)

        # Defaults should be set
        q_dist = binary_imf._get_q_distribution()
        assert isinstance(q_dist, MoeDiStefano2017)

        f_bin_model = binary_imf._get_binary_fraction_model()
        assert isinstance(f_bin_model, MassDependentBinaryFraction)

    def test_initialization_simple(self):
        """Test simple constant parameters."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF(
            primary_imf=primary_imf,
            q_distribution=FlatMassRatio(q_min=0.1),
            binary_fraction=0.5,
        )

        q_dist = binary_imf._get_q_distribution()
        assert isinstance(q_dist, FlatMassRatio)

        # Float binary fraction
        assert binary_imf.binary_fraction == 0.5

    def test_sample_primaries(self):
        """Sample primary masses from IMF."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF(primary_imf=primary_imf)

        key = jax.random.PRNGKey(42)
        m1 = binary_imf.sample_primaries(key, 100)

        assert m1.shape == (100,)
        assert jnp.all(m1 >= primary_imf.m_min)
        assert jnp.all(m1 <= primary_imf.m_max)

    def test_sample_mass_ratios(self):
        """Sample mass ratios given primaries."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF(
            primary_imf=primary_imf,
            q_distribution=FlatMassRatio(q_min=0.15),
        )

        key = jax.random.PRNGKey(42)
        m1 = jnp.ones(100) * 1.0
        q = binary_imf.sample_mass_ratios(key, m1)

        assert q.shape == (100,)
        assert jnp.all(q >= 0.15)
        assert jnp.all(q <= 1.0)

    def test_sample_systems_returns_tuple(self):
        """Sample systems returns (m1, m2, is_binary)."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF(
            primary_imf=primary_imf,
            q_distribution=FlatMassRatio(),
            binary_fraction=0.5,
        )

        key = jax.random.PRNGKey(42)
        m1, m2, is_binary = binary_imf.sample_systems(key, 1000)

        assert m1.shape == (1000,)
        assert m2.shape == (1000,)
        assert is_binary.shape == (1000,)
        assert is_binary.dtype == bool

    def test_sample_systems_binary_fraction(self):
        """Binary fraction matches target."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF(
            primary_imf=primary_imf,
            binary_fraction=0.5,
        )

        key = jax.random.PRNGKey(123)
        m1, m2, is_binary = binary_imf.sample_systems(key, 10000)

        frac = jnp.mean(is_binary.astype(float))
        assert jnp.abs(frac - 0.5) < 0.02  # Within 2% of target

    def test_sample_systems_singles_have_zero_m2(self):
        """Single stars have m2=0."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF(
            primary_imf=primary_imf,
            binary_fraction=0.5,
        )

        key = jax.random.PRNGKey(42)
        m1, m2, is_binary = binary_imf.sample_systems(key, 1000)

        singles_m2 = m2[~is_binary]
        assert jnp.allclose(singles_m2, 0.0)

    def test_sample_systems_binaries_satisfy_q_constraint(self):
        """Binary secondaries satisfy q = m2/m1 in [q_min, 1]."""
        primary_imf = PowerLawIMF.kroupa()
        q_min = 0.15
        binary_imf = BinaryIMF(
            primary_imf=primary_imf,
            q_distribution=FlatMassRatio(q_min=q_min),
            binary_fraction=1.0,  # All binaries
        )

        key = jax.random.PRNGKey(42)
        m1, m2, is_binary = binary_imf.sample_systems(key, 1000)

        # All should be binaries
        assert jnp.all(is_binary)

        # Check mass ratios
        q = m2 / m1
        assert jnp.all(q >= q_min - 1e-6)
        assert jnp.all(q <= 1.0 + 1e-6)

    def test_sample_all_masses(self):
        """Sample all masses returns flattened array."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF(
            primary_imf=primary_imf,
            binary_fraction=0.5,
        )

        key = jax.random.PRNGKey(42)
        all_masses, is_binary = binary_imf.sample_all_masses(key, 100)

        # Should have 100 primaries + ~50 secondaries
        n_binaries = jnp.sum(is_binary)
        expected_length = 100 + n_binaries
        assert all_masses.shape[0] == expected_length

    def test_factory_moe2017(self):
        """Factory method creates Moe+17 model."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.moe2017(primary_imf)

        assert isinstance(binary_imf._get_q_distribution(), MoeDiStefano2017)
        assert isinstance(
            binary_imf._get_binary_fraction_model(),
            MassDependentBinaryFraction,
        )

    def test_factory_simple(self):
        """Factory method creates simple model."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.simple(
            primary_imf,
            binary_fraction=0.6,
            q_min=0.2,
        )

        assert isinstance(binary_imf._get_q_distribution(), FlatMassRatio)
        assert binary_imf.binary_fraction == 0.6

    def test_factory_massive_stars(self):
        """Factory method creates massive star model."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.massive_stars(primary_imf)

        q_dist = binary_imf._get_q_distribution()
        assert isinstance(q_dist, PowerLawMassRatio)
        assert q_dist.gamma == -0.1
        assert binary_imf.binary_fraction == 0.7

    def test_mean_system_mass(self):
        """Mean system mass is reasonable."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.simple(
            primary_imf,
            binary_fraction=0.5,
            q_min=0.5,
        )

        mean_m1 = primary_imf.mean_mass()
        mean_sys = binary_imf.mean_system_mass()

        # Should be between mean_m1 and 2*mean_m1
        assert mean_sys > mean_m1
        assert mean_sys < 2.0 * mean_m1

    def test_binary_fraction_overall(self):
        """Overall binary fraction matches."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.simple(
            primary_imf,
            binary_fraction=0.6,
        )

        f_overall = binary_imf.binary_fraction_overall()
        assert jnp.abs(f_overall - 0.6) < 1e-6

    def test_differentiable_through_sample(self):
        """Can differentiate through sampling."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.simple(primary_imf, binary_fraction=0.5)

        def loss(dummy_param):
            key = jax.random.PRNGKey(42)
            m1, m2, _ = binary_imf.sample_systems(key, 100)
            return jnp.sum(m1) + jnp.sum(m2)

        # Should be differentiable
        grad_fn = jax.grad(loss)
        gradient = grad_fn(1.0)
        assert jnp.isfinite(gradient)
