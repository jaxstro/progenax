"""Tests for binary star mass functions.

Physics tests only - distribution properties and literature comparisons.
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


class TestBinaryIMF:
    """Test binary IMF composition."""

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
