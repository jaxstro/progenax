"""
Unit tests for environment-conditioned IMF module.

Tests GasEnvironment, Jeans mass scaling, alpha prescriptions,
EnvironmentIMF, and CustomEnvironmentIMF.
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
    G_CGS,
    K_B_CGS,
    M_P_CGS,
    M_SUN_CGS,
    MU_MOL,
)


# ============================================================================
# Physical Constants Tests
# ============================================================================


def test_physical_constants():
    """Test CGS constants have correct values."""
    assert G_CGS == pytest.approx(6.674e-8, rel=1e-6)
    assert K_B_CGS == pytest.approx(1.381e-16, rel=1e-6)
    assert M_P_CGS == pytest.approx(1.673e-24, rel=1e-6)
    assert M_SUN_CGS == pytest.approx(1.989e33, rel=1e-6)
    assert MU_MOL == pytest.approx(2.33, rel=1e-6)


# ============================================================================
# GasEnvironment Tests
# ============================================================================


def test_gas_environment_default():
    """Test GasEnvironment default parameters."""
    env = GasEnvironment()
    assert env.n_H == 1e4
    assert env.T_gas == 10.0
    assert env.Z == 1.0


def test_gas_environment_custom():
    """Test GasEnvironment with custom parameters."""
    env = GasEnvironment(n_H=1e6, T_gas=20.0, Z=0.5)
    assert env.n_H == 1e6
    assert env.T_gas == 20.0
    assert env.Z == 0.5


def test_gas_environment_rho():
    """Test mass density calculation."""
    env = GasEnvironment(n_H=1e4, T_gas=10.0)
    expected_rho = 1e4 * MU_MOL * M_P_CGS
    assert env.rho == pytest.approx(expected_rho, rel=1e-6)


def test_gas_environment_log_n():
    """Test log density calculation."""
    env = GasEnvironment(n_H=1e4)
    assert env.log_n == pytest.approx(4.0, rel=1e-6)

    env = GasEnvironment(n_H=1e6)
    assert env.log_n == pytest.approx(6.0, rel=1e-6)


def test_gas_environment_solar_neighborhood():
    """Test solar neighborhood factory method."""
    env = GasEnvironment.solar_neighborhood()
    assert env.n_H == 1e4
    assert env.T_gas == 10.0
    assert env.Z == 1.0


def test_gas_environment_dense_clump():
    """Test dense clump factory method."""
    env = GasEnvironment.dense_clump()
    assert env.n_H == 1e6
    assert env.T_gas == 20.0
    assert env.Z == 1.0


def test_gas_environment_low_metallicity():
    """Test low metallicity factory method."""
    env = GasEnvironment.low_metallicity(Z=0.1)
    assert env.n_H == 1e4
    assert env.T_gas == 10.0
    assert env.Z == 0.1


def test_gas_environment_starburst():
    """Test starburst factory method."""
    env = GasEnvironment.starburst()
    assert env.n_H == 1e5
    assert env.T_gas == 50.0
    assert env.Z == 1.5


def test_gas_environment_primordial():
    """Test primordial factory method."""
    env = GasEnvironment.primordial()
    assert env.n_H == 1e4
    assert env.T_gas == 200.0
    assert env.Z == 0.0


# ============================================================================
# Jeans Mass Tests
# ============================================================================


def test_jeans_mass_typical():
    """Test Jeans mass for typical GMC conditions."""
    M_J = jeans_mass(n_H=1e4, T=10.0)
    # Typical GMC: M_J ~ few solar masses
    assert 0.1 < M_J < 10.0


def test_jeans_mass_temperature_scaling():
    """Test Jeans mass scales as T^(3/2)."""
    M_J1 = jeans_mass(n_H=1e4, T=10.0)
    M_J2 = jeans_mass(n_H=1e4, T=40.0)  # 4× temperature
    # M_J ∝ T^(3/2) → 4^(3/2) = 8×
    assert M_J2 / M_J1 == pytest.approx(8.0, rel=0.1)


def test_jeans_mass_density_scaling():
    """Test Jeans mass scales as rho^(-1/2)."""
    M_J1 = jeans_mass(n_H=1e4, T=10.0)
    M_J2 = jeans_mass(n_H=4e4, T=10.0)  # 4× density
    # M_J ∝ ρ^(-1/2) → (1/4)^(1/2) = 0.5×
    assert M_J2 / M_J1 == pytest.approx(0.5, rel=0.1)


def test_jeans_mass_positive():
    """Test Jeans mass is always positive."""
    M_J = jeans_mass(n_H=1e6, T=100.0)
    assert M_J > 0


def test_jeans_mass_primordial():
    """Test Jeans mass for Pop III conditions (high T, low density)."""
    M_J = jeans_mass(n_H=1e4, T=200.0)
    # Primordial: M_J ~ 100s of solar masses
    assert M_J > 10.0


# ============================================================================
# Characteristic Mass Tests
# ============================================================================


def test_characteristic_mass_default_efficiency():
    """Test characteristic mass with default efficiency."""
    m_c = characteristic_mass_from_jeans(n_H=1e4, T=10.0)
    M_J = jeans_mass(n_H=1e4, T=10.0)
    assert m_c == pytest.approx(0.1 * M_J, rel=1e-6)


def test_characteristic_mass_custom_efficiency():
    """Test characteristic mass with custom efficiency."""
    m_c = characteristic_mass_from_jeans(n_H=1e4, T=10.0, efficiency=0.2)
    M_J = jeans_mass(n_H=1e4, T=10.0)
    assert m_c == pytest.approx(0.2 * M_J, rel=1e-6)


def test_characteristic_mass_positive():
    """Test characteristic mass is positive."""
    m_c = characteristic_mass_from_jeans(n_H=1e6, T=20.0)
    assert m_c > 0


# ============================================================================
# Bounded Transformation Tests
# ============================================================================


def test_alpha_bounded_centered():
    """Test alpha_bounded at centered value."""
    alpha = alpha_bounded(0.0)
    # At f=0, sigmoid=0.5 → midpoint of range
    expected = 1.5 + 0.5 * (2.7 - 1.5)
    assert alpha == pytest.approx(expected, rel=1e-6)


def test_alpha_bounded_extreme_low():
    """Test alpha_bounded at extreme low input."""
    alpha = alpha_bounded(-100.0)
    # Should approach alpha_min
    assert alpha == pytest.approx(1.5, rel=1e-3)


def test_alpha_bounded_extreme_high():
    """Test alpha_bounded at extreme high input."""
    alpha = alpha_bounded(100.0)
    # Should approach alpha_max
    assert alpha == pytest.approx(2.7, rel=1e-3)


def test_alpha_bounded_custom_range():
    """Test alpha_bounded with custom range."""
    alpha = alpha_bounded(0.0, alpha_min=1.0, alpha_max=3.0)
    expected = 1.0 + 0.5 * (3.0 - 1.0)
    assert alpha == pytest.approx(expected, rel=1e-6)


# ============================================================================
# Bonnor-Ebert Mass Tests
# ============================================================================


def test_bonnor_ebert_mass_reference():
    """Test BE mass at reference conditions."""
    m_BE = bonnor_ebert_mass(T=10.0, P_over_kB=1e5, Z=1.0)
    # Should be close to normalization value
    assert m_BE == pytest.approx(0.3, rel=0.1)


def test_bonnor_ebert_mass_temperature_scaling():
    """Test BE mass scales as T^2."""
    m_BE1 = bonnor_ebert_mass(T=10.0, P_over_kB=1e5, Z=1.0)
    m_BE2 = bonnor_ebert_mass(T=20.0, P_over_kB=1e5, Z=1.0)
    # m_BE ∝ T^2 → (20/10)^2 = 4×
    assert m_BE2 / m_BE1 == pytest.approx(4.0, rel=1e-6)


def test_bonnor_ebert_mass_pressure_scaling():
    """Test BE mass scales as P^(-1/2)."""
    m_BE1 = bonnor_ebert_mass(T=10.0, P_over_kB=1e5, Z=1.0)
    m_BE2 = bonnor_ebert_mass(T=10.0, P_over_kB=4e5, Z=1.0)
    # m_BE ∝ P^(-1/2) → (1/4)^(1/2) = 0.5×
    assert m_BE2 / m_BE1 == pytest.approx(0.5, rel=1e-6)


def test_bonnor_ebert_mass_metallicity_effect():
    """Test BE mass metallicity dependence."""
    m_BE_solar = bonnor_ebert_mass(T=10.0, P_over_kB=1e5, Z=1.0)
    m_BE_low_Z = bonnor_ebert_mass(T=10.0, P_over_kB=1e5, Z=0.1)
    # Low Z → less cooling → larger m_c
    assert m_BE_low_Z > m_BE_solar


def test_bonnor_ebert_mass_primordial():
    """Test BE mass for primordial conditions."""
    m_BE = bonnor_ebert_mass(T=200.0, P_over_kB=1e5, Z=0.0)
    # Very high due to high T and low Z
    assert m_BE > 10.0


# ============================================================================
# Alpha Prescription Tests
# ============================================================================


def test_alpha_marks2012_low_density():
    """Test Marks+2012 alpha for low density."""
    alpha = alpha_marks2012(log_n=4.0)  # log(n) < 6.5
    # Should be standard Kroupa
    assert alpha == pytest.approx(2.3, rel=1e-6)


def test_alpha_marks2012_high_density():
    """Test Marks+2012 alpha for high density."""
    alpha = alpha_marks2012(log_n=7.0)  # log(n) > 6.5
    # Should be lower (top-heavy)
    assert alpha < 2.3


def test_alpha_marks2012_decreases_with_density():
    """Test alpha decreases with increasing density."""
    alpha1 = alpha_marks2012(log_n=6.0)
    alpha2 = alpha_marks2012(log_n=7.0)
    assert alpha2 < alpha1


def test_alpha_jerabkova2018_solar():
    """Test Jerabkova+2018 alpha for solar conditions."""
    alpha = alpha_jerabkova2018(Z=1.0, log_n=4.0)
    # Should be close to solar
    assert alpha == pytest.approx(2.3, rel=0.1)


def test_alpha_jerabkova2018_low_metallicity():
    """Test Jerabkova+2018 alpha for low metallicity."""
    alpha_solar = alpha_jerabkova2018(Z=1.0, log_n=4.0)
    alpha_low_Z = alpha_jerabkova2018(Z=0.2, log_n=4.0)
    # Low Z → top-heavy → lower alpha
    assert alpha_low_Z < alpha_solar


def test_alpha_jerabkova2018_high_density():
    """Test Jerabkova+2018 alpha for high density."""
    alpha_low = alpha_jerabkova2018(Z=1.0, log_n=4.0)
    alpha_high = alpha_jerabkova2018(Z=1.0, log_n=6.5)
    # High density → top-heavy → lower alpha
    assert alpha_high < alpha_low


def test_alpha_from_sfr_milky_way():
    """Test SFR-based alpha for Milky Way."""
    alpha = alpha_from_sfr(sfr=1.9)  # MW SFR
    # Should be close to solar
    assert alpha == pytest.approx(2.3, rel=0.1)


def test_alpha_from_sfr_starburst():
    """Test SFR-based alpha for starburst."""
    alpha_mw = alpha_from_sfr(sfr=1.9)
    alpha_sb = alpha_from_sfr(sfr=100.0)
    # Higher SFR → top-heavy → lower alpha
    assert alpha_sb < alpha_mw


# ============================================================================
# EnvironmentIMF Tests
# ============================================================================


def test_environment_imf_solar_neighborhood():
    """Test EnvironmentIMF for solar neighborhood."""
    env = GasEnvironment.solar_neighborhood()
    imf = EnvironmentIMF(env)

    # Should have standard Kroupa-like alpha
    assert 2.0 < imf.alpha_high < 2.5

    # Characteristic mass should be reasonable
    assert 0.1 < imf.m_char < 2.0


def test_environment_imf_dense_clump():
    """Test EnvironmentIMF for dense clump."""
    env = GasEnvironment.dense_clump()
    imf = EnvironmentIMF(env)

    # Higher density → lower m_c
    env_solar = GasEnvironment.solar_neighborhood()
    imf_solar = EnvironmentIMF(env_solar)

    assert imf.m_char < imf_solar.m_char


def test_environment_imf_low_metallicity():
    """Test EnvironmentIMF for low metallicity."""
    env = GasEnvironment.low_metallicity(Z=0.1)
    imf = EnvironmentIMF(env)

    # Low Z → top-heavy → lower alpha
    env_solar = GasEnvironment.solar_neighborhood()
    imf_solar = EnvironmentIMF(env_solar)

    assert imf.alpha_high < imf_solar.alpha_high


def test_environment_imf_primordial():
    """Test EnvironmentIMF for primordial conditions."""
    env = GasEnvironment.primordial()
    imf = EnvironmentIMF(env)

    # Zero metallicity → top-heavy (alpha <= 2.0)
    # With default jerabkova2018: Z=0, log_n=4 → alpha = 2.3 - 0.3 = 2.0
    assert imf.alpha_high <= 2.0

    # Characteristic mass clipped to [0.1, 2.0] range
    assert 0.1 <= imf.m_char <= 2.0


def test_environment_imf_sampling():
    """Test EnvironmentIMF can sample masses."""
    env = GasEnvironment.solar_neighborhood()
    imf = EnvironmentIMF(env)

    key = jax.random.PRNGKey(42)
    masses = imf.sample(key, 1000)

    # Check shape
    assert masses.shape == (1000,)

    # Check bounds
    assert jnp.all(masses >= imf.m_min)
    assert jnp.all(masses <= imf.m_max)

    # Check not all identical
    assert jnp.std(masses) > 0


def test_environment_imf_pdf_normalization():
    """Test EnvironmentIMF PDF is normalized."""
    env = GasEnvironment.solar_neighborhood()
    imf = EnvironmentIMF(env)

    # Sample and check CDF bounds
    m_test = jnp.linspace(imf.m_min, imf.m_max, 100)
    cdf_vals = jax.vmap(imf.cdf)(m_test)

    assert cdf_vals[0] == pytest.approx(0.0, abs=0.01)
    assert cdf_vals[-1] == pytest.approx(1.0, abs=0.01)


def test_environment_imf_marks2012_model():
    """Test EnvironmentIMF with Marks+2012 alpha model."""
    env = GasEnvironment(n_H=1e7, T_gas=10.0)
    imf = EnvironmentIMF(env, alpha_model='marks2012')

    # High density should give top-heavy IMF
    assert imf.alpha_high < 2.3


def test_environment_imf_sfr_model():
    """Test EnvironmentIMF with SFR alpha model."""
    env = GasEnvironment.starburst()
    imf = EnvironmentIMF(env, alpha_model='sfr', sfr=100.0)

    # High SFR should give top-heavy IMF
    assert imf.alpha_high < 2.3


def test_environment_imf_factory_methods():
    """Test EnvironmentIMF factory methods."""
    imf_solar = EnvironmentIMF.solar_neighborhood()
    imf_sb = EnvironmentIMF.starburst(sfr=100.0)
    imf_clump = EnvironmentIMF.dense_clump()
    imf_low_z = EnvironmentIMF.low_metallicity(Z=0.1)
    imf_pop3 = EnvironmentIMF.primordial()

    # All should be valid IMFs
    for imf in [imf_solar, imf_sb, imf_clump, imf_low_z, imf_pop3]:
        assert hasattr(imf, 'sample')
        assert hasattr(imf, 'logpdf')


def test_environment_imf_mean_mass():
    """Test EnvironmentIMF mean mass calculation."""
    env = GasEnvironment.solar_neighborhood()
    imf = EnvironmentIMF(env)

    mean_m = imf.mean_mass()

    # Mean mass should be reasonable
    assert 0.1 < mean_m < 10.0


# ============================================================================
# CustomEnvironmentIMF Tests
# ============================================================================


def test_custom_environment_imf_default():
    """Test CustomEnvironmentIMF with default functions."""
    env = GasEnvironment.solar_neighborhood()
    imf = CustomEnvironmentIMF(env)

    # Should use Jerabkova+2018 by default
    assert 2.0 < imf.alpha_high < 2.5


def test_custom_environment_imf_custom_alpha():
    """Test CustomEnvironmentIMF with custom alpha function."""
    def my_alpha(env):
        return 2.0  # Fixed top-heavy

    env = GasEnvironment.solar_neighborhood()
    imf = CustomEnvironmentIMF(env, alpha_fn=my_alpha)

    assert imf.alpha_high == pytest.approx(2.0, rel=1e-6)


def test_custom_environment_imf_custom_char_mass():
    """Test CustomEnvironmentIMF with custom characteristic mass."""
    def my_char_mass(env):
        return 1.0  # Fixed m_c

    env = GasEnvironment.solar_neighborhood()
    imf = CustomEnvironmentIMF(env, char_mass_fn=my_char_mass)

    assert imf.m_char == pytest.approx(1.0, rel=1e-6)


def test_custom_environment_imf_sampling():
    """Test CustomEnvironmentIMF can sample masses."""
    env = GasEnvironment.solar_neighborhood()
    imf = CustomEnvironmentIMF(env)

    key = jax.random.PRNGKey(42)
    masses = imf.sample(key, 1000)

    assert masses.shape == (1000,)
    assert jnp.all(masses >= imf.m_min)
    assert jnp.all(masses <= imf.m_max)


# ============================================================================
# Utility Tests
# ============================================================================


def test_is_top_heavy_true():
    """Test is_top_heavy for top-heavy IMF."""
    assert is_top_heavy(1.8)
    assert is_top_heavy(1.5)


def test_is_top_heavy_false():
    """Test is_top_heavy for bottom-heavy IMF."""
    assert not is_top_heavy(2.3)
    assert not is_top_heavy(2.7)


def test_is_top_heavy_threshold():
    """Test is_top_heavy at threshold."""
    assert is_top_heavy(1.9, threshold=2.0)
    assert not is_top_heavy(2.1, threshold=2.0)


def test_massive_star_fraction_solar():
    """Test massive star fraction for solar neighborhood."""
    imf = EnvironmentIMF.solar_neighborhood()
    frac = massive_star_fraction(imf, m_threshold=8.0)

    # Solar IMF: small fraction in massive stars (stochastic, so be lenient)
    assert 0.0 < frac < 0.2


def test_massive_star_fraction_top_heavy():
    """Test massive star fraction for top-heavy IMF."""
    imf_solar = EnvironmentIMF.solar_neighborhood()
    imf_pop3 = EnvironmentIMF.primordial()

    frac_solar = massive_star_fraction(imf_solar, m_threshold=8.0)
    frac_pop3 = massive_star_fraction(imf_pop3, m_threshold=8.0)

    # Pop III should have more massive stars
    assert frac_pop3 > frac_solar


# ============================================================================
# Gradient Tests
# ============================================================================


def test_environment_imf_differentiable_sampling():
    """Test EnvironmentIMF sampling is differentiable w.r.t. PPF inputs."""
    # Create a fixed IMF and test differentiability of sampling
    env = GasEnvironment(n_H=1e4, T_gas=10.0, Z=1.0)
    imf = EnvironmentIMF(env)

    def loss(u_val):
        # Test differentiability of PPF (inverse CDF)
        u = jnp.array([u_val])
        mass = imf.ppf(u)
        return mass[0]

    grad_fn = jax.grad(loss)
    gradient = grad_fn(0.5)

    # Gradient should be finite and positive (inverse PDF)
    assert jnp.isfinite(gradient)
    assert gradient > 0.0


def test_jeans_mass_differentiable():
    """Test jeans_mass is differentiable."""
    def loss(T):
        return jeans_mass(n_H=1e4, T=T)

    grad_fn = jax.grad(loss)
    gradient = grad_fn(10.0)

    # Gradient should be positive (M_J increases with T)
    assert gradient > 0


def test_alpha_bounded_differentiable():
    """Test alpha_bounded is differentiable."""
    grad_fn = jax.grad(alpha_bounded)
    gradient = grad_fn(0.0)

    # Gradient should be positive (sigmoid derivative)
    assert gradient > 0
