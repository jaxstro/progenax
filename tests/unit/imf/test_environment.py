"""
Unit tests for environment-conditioned IMF module.

Tests GasEnvironment, Jeans mass scaling, alpha prescriptions,
EnvironmentIMF, and CustomEnvironmentIMF.

CONSOLIDATED: ~25 essential physics tests from original 57.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf.environment import (
    GasEnvironment,
    jeans_mass,
    characteristic_mass_from_jeans,
    alpha_bounded,
    bonnor_ebert_mass,
    alpha_marks2012,
    alpha_jerabkova2018,
    alpha_from_sfr,
    EnvironmentIMF,
    CustomEnvironmentIMF,
    is_top_heavy,
    massive_star_fraction,
)


# ============================================================================
# Jeans Mass Physics Tests
# ============================================================================


class TestJeansMassPhysics:
    """Test Jeans mass physical scaling relations."""

    def test_temperature_scaling(self):
        """Jeans mass scales as T^(3/2)."""
        M_J1 = jeans_mass(n_H=1e4, T=10.0)
        M_J2 = jeans_mass(n_H=1e4, T=40.0)  # 4× temperature
        # M_J ∝ T^(3/2) → 4^(3/2) = 8×
        assert M_J2 / M_J1 == pytest.approx(8.0, rel=0.1)

    def test_density_scaling(self):
        """Jeans mass scales as rho^(-1/2)."""
        M_J1 = jeans_mass(n_H=1e4, T=10.0)
        M_J2 = jeans_mass(n_H=4e4, T=10.0)  # 4× density
        # M_J ∝ ρ^(-1/2) → (1/4)^(1/2) = 0.5×
        assert M_J2 / M_J1 == pytest.approx(0.5, rel=0.1)

    def test_typical_gmc_value(self):
        """Typical GMC conditions give M_J ~ few solar masses."""
        M_J = jeans_mass(n_H=1e4, T=10.0)
        assert 0.1 < M_J < 10.0

    def test_differentiable(self):
        """Jeans mass is differentiable w.r.t. T."""
        def loss(T):
            return jeans_mass(n_H=1e4, T=T)

        grad_fn = jax.grad(loss)
        gradient = grad_fn(10.0)
        assert gradient > 0  # M_J increases with T


# ============================================================================
# Bonnor-Ebert Mass Physics Tests
# ============================================================================


class TestBonnorEbertMassPhysics:
    """Test Bonnor-Ebert mass physical scaling relations."""

    def test_temperature_scaling(self):
        """BE mass scales as T^2."""
        m_BE1 = bonnor_ebert_mass(T=10.0, P_over_kB=1e5, Z=1.0)
        m_BE2 = bonnor_ebert_mass(T=20.0, P_over_kB=1e5, Z=1.0)
        # m_BE ∝ T^2 → (20/10)^2 = 4×
        assert m_BE2 / m_BE1 == pytest.approx(4.0, rel=1e-6)

    def test_pressure_scaling(self):
        """BE mass scales as P^(-1/2)."""
        m_BE1 = bonnor_ebert_mass(T=10.0, P_over_kB=1e5, Z=1.0)
        m_BE2 = bonnor_ebert_mass(T=10.0, P_over_kB=4e5, Z=1.0)
        # m_BE ∝ P^(-1/2) → (1/4)^(1/2) = 0.5×
        assert m_BE2 / m_BE1 == pytest.approx(0.5, rel=1e-6)

    def test_metallicity_effect(self):
        """Low Z → less cooling → larger m_c."""
        m_BE_solar = bonnor_ebert_mass(T=10.0, P_over_kB=1e5, Z=1.0)
        m_BE_low_Z = bonnor_ebert_mass(T=10.0, P_over_kB=1e5, Z=0.1)
        assert m_BE_low_Z > m_BE_solar


# ============================================================================
# Alpha Prescription Physics Tests
# ============================================================================


class TestAlphaPrescriptions:
    """Test IMF slope (alpha) prescriptions."""

    def test_marks2012_density_dependence(self):
        """Alpha decreases with increasing density (Marks+2012)."""
        alpha1 = alpha_marks2012(log_n=6.0)
        alpha2 = alpha_marks2012(log_n=7.0)
        assert alpha2 < alpha1

    def test_jerabkova2018_metallicity_effect(self):
        """Low Z → top-heavy → lower alpha (Jerabkova+2018)."""
        alpha_solar = alpha_jerabkova2018(Z=1.0, log_n=4.0)
        alpha_low_Z = alpha_jerabkova2018(Z=0.2, log_n=4.0)
        assert alpha_low_Z < alpha_solar

    def test_sfr_dependence(self):
        """Higher SFR → top-heavy → lower alpha."""
        alpha_mw = alpha_from_sfr(sfr=1.9)
        alpha_sb = alpha_from_sfr(sfr=100.0)
        assert alpha_sb < alpha_mw

    def test_alpha_bounded_range(self):
        """Alpha bounded stays within [1.5, 2.7]."""
        alpha_low = alpha_bounded(-100.0)
        alpha_high = alpha_bounded(100.0)
        assert alpha_low == pytest.approx(1.5, rel=1e-3)
        assert alpha_high == pytest.approx(2.7, rel=1e-3)

    def test_alpha_bounded_differentiable(self):
        """Alpha bounded is differentiable."""
        grad_fn = jax.grad(alpha_bounded)
        gradient = grad_fn(0.0)
        assert gradient > 0  # Sigmoid derivative positive


# ============================================================================
# EnvironmentIMF Tests
# ============================================================================


class TestEnvironmentIMF:
    """Test EnvironmentIMF core functionality."""

    def test_density_affects_char_mass(self):
        """Higher density → lower characteristic mass."""
        env_solar = GasEnvironment.solar_neighborhood()
        env_dense = GasEnvironment.dense_clump()
        imf_solar = EnvironmentIMF(env_solar)
        imf_dense = EnvironmentIMF(env_dense)
        assert imf_dense.m_char < imf_solar.m_char

    def test_metallicity_affects_alpha(self):
        """Lower metallicity → top-heavy → lower alpha."""
        env_solar = GasEnvironment.solar_neighborhood()
        env_low_Z = GasEnvironment.low_metallicity(Z=0.1)
        imf_solar = EnvironmentIMF(env_solar)
        imf_low_Z = EnvironmentIMF(env_low_Z)
        assert imf_low_Z.alpha_high < imf_solar.alpha_high

    def test_primordial_is_top_heavy(self):
        """Primordial conditions give top-heavy IMF (alpha <= 2.0)."""
        env = GasEnvironment.primordial()
        imf = EnvironmentIMF(env)
        assert imf.alpha_high <= 2.0

    def test_sampling_basic(self):
        """EnvironmentIMF can sample masses within bounds."""
        env = GasEnvironment.solar_neighborhood()
        imf = EnvironmentIMF(env)
        key = jax.random.PRNGKey(42)
        masses = imf.sample(key, 1000)

        assert masses.shape == (1000,)
        assert jnp.all(masses >= imf.m_min)
        assert jnp.all(masses <= imf.m_max)
        assert jnp.std(masses) > 0

    def test_cdf_normalization(self):
        """CDF is normalized: CDF(m_min)≈0, CDF(m_max)≈1."""
        env = GasEnvironment.solar_neighborhood()
        imf = EnvironmentIMF(env)
        m_test = jnp.linspace(imf.m_min, imf.m_max, 100)
        cdf_vals = jax.vmap(imf.cdf)(m_test)

        assert cdf_vals[0] == pytest.approx(0.0, abs=0.01)
        assert cdf_vals[-1] == pytest.approx(1.0, abs=0.01)

    def test_different_alpha_models(self):
        """Different alpha models give different slopes."""
        env = GasEnvironment(n_H=1e7, T_gas=10.0)
        imf_marks = EnvironmentIMF(env, alpha_model='marks2012')
        imf_jerab = EnvironmentIMF(env, alpha_model='jerabkova2018')

        # Both should give top-heavy at high density
        assert imf_marks.alpha_high < 2.3
        assert imf_jerab.alpha_high < 2.3

    def test_differentiable_ppf(self):
        """EnvironmentIMF PPF is differentiable."""
        env = GasEnvironment(n_H=1e4, T_gas=10.0, Z=1.0)
        imf = EnvironmentIMF(env)

        def loss(u_val):
            u = jnp.array([u_val])
            mass = imf.ppf(u)
            return mass[0]

        grad_fn = jax.grad(loss)
        gradient = grad_fn(0.5)
        assert jnp.isfinite(gradient)
        assert gradient > 0.0


# ============================================================================
# CustomEnvironmentIMF Tests
# ============================================================================


class TestCustomEnvironmentIMF:
    """Test CustomEnvironmentIMF with user-defined functions."""

    def test_custom_alpha_function(self):
        """Custom alpha function overrides default."""
        def my_alpha(env):
            return 2.0  # Fixed top-heavy

        env = GasEnvironment.solar_neighborhood()
        imf = CustomEnvironmentIMF(env, alpha_fn=my_alpha)
        assert imf.alpha_high == pytest.approx(2.0, rel=1e-6)

    def test_custom_char_mass_function(self):
        """Custom characteristic mass function overrides default."""
        def my_char_mass(env):
            return 1.0  # Fixed m_c

        env = GasEnvironment.solar_neighborhood()
        imf = CustomEnvironmentIMF(env, char_mass_fn=my_char_mass)
        assert imf.m_char == pytest.approx(1.0, rel=1e-6)

    def test_sampling(self):
        """CustomEnvironmentIMF can sample masses."""
        env = GasEnvironment.solar_neighborhood()
        imf = CustomEnvironmentIMF(env)
        key = jax.random.PRNGKey(42)
        masses = imf.sample(key, 1000)

        assert masses.shape == (1000,)
        assert jnp.all(masses >= imf.m_min)


# ============================================================================
# Utility Tests
# ============================================================================


class TestUtilities:
    """Test utility functions."""

    def test_is_top_heavy_classification(self):
        """is_top_heavy correctly classifies IMF slopes."""
        assert is_top_heavy(1.8)
        assert not is_top_heavy(2.3)
        assert is_top_heavy(1.9, threshold=2.0)
        assert not is_top_heavy(2.1, threshold=2.0)

    def test_massive_star_fraction_trend(self):
        """Top-heavy IMF has more massive stars than solar."""
        imf_solar = EnvironmentIMF.solar_neighborhood()
        imf_pop3 = EnvironmentIMF.primordial()

        frac_solar = massive_star_fraction(imf_solar, m_threshold=8.0)
        frac_pop3 = massive_star_fraction(imf_pop3, m_threshold=8.0)

        assert frac_pop3 > frac_solar
