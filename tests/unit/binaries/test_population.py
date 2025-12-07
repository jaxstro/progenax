"""Tests for binary population parameter sampling."""

import jax
import jax.numpy as jnp
import pytest

from progenax.binaries.population import (
    LogUniformPeriod,
    LogNormalPeriod,
    ThermalEccentricity,
    UniformEccentricity,
    sample_isotropic_orientations,
)


class TestLogUniformPeriod:
    """Tests for Öpik's law: log-uniform period distribution."""

    def test_sample_shape(self):
        """Samples have correct shape."""
        dist = LogUniformPeriod(log_P_min=0.0, log_P_max=8.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 1000)
        assert samples.shape == (1000,)

    def test_samples_in_range(self):
        """All samples within [P_min, P_max]."""
        dist = LogUniformPeriod(log_P_min=0.0, log_P_max=8.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 10000)
        log_samples = jnp.log10(samples)
        assert jnp.all(log_samples >= 0.0 - 1e-6)
        assert jnp.all(log_samples <= 8.0 + 1e-6)

    def test_log_uniform_distribution(self):
        """log10(P) is uniformly distributed (Öpik's law)."""
        dist = LogUniformPeriod(log_P_min=0.0, log_P_max=8.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 100000)
        log_samples = jnp.log10(samples)
        mean_log_P = jnp.mean(log_samples)
        assert jnp.abs(mean_log_P - 4.0) < 0.1

    def test_pdf_normalization(self):
        """PDF integrates to 1."""
        dist = LogUniformPeriod(log_P_min=0.0, log_P_max=4.0)
        P_vals = jnp.logspace(0, 4, 1000)
        pdf_vals = dist.pdf(P_vals)
        integral = jnp.trapezoid(pdf_vals, P_vals)
        assert jnp.abs(integral - 1.0) < 0.01


class TestLogNormalPeriod:
    """Tests for log-normal period distribution."""

    def test_sample_shape(self):
        """Samples have correct shape."""
        dist = LogNormalPeriod(mu_log_P=4.0, sigma_log_P=2.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 1000)
        assert samples.shape == (1000,)

    def test_mean_log_period(self):
        """Mean of log10(P) matches mu parameter."""
        dist = LogNormalPeriod(mu_log_P=4.0, sigma_log_P=2.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 100000)
        log_samples = jnp.log10(samples)
        mean_log_P = jnp.mean(log_samples)
        assert jnp.abs(mean_log_P - 4.0) < 0.05

    def test_std_log_period(self):
        """Std of log10(P) matches sigma parameter."""
        dist = LogNormalPeriod(mu_log_P=4.0, sigma_log_P=2.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 100000)
        log_samples = jnp.log10(samples)
        std_log_P = jnp.std(log_samples)
        assert jnp.abs(std_log_P - 2.0) < 0.1

    def test_positive_periods(self):
        """All sampled periods are positive."""
        dist = LogNormalPeriod(mu_log_P=2.0, sigma_log_P=1.5)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 10000)
        assert jnp.all(samples > 0)


class TestThermalEccentricity:
    """Tests for thermal eccentricity distribution f(e) = 2e."""

    def test_sample_shape(self):
        """Samples have correct shape."""
        dist = ThermalEccentricity()
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 1000)
        assert samples.shape == (1000,)

    def test_samples_in_range(self):
        """All samples in [0, 1)."""
        dist = ThermalEccentricity()
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 10000)
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples < 1.0)

    def test_thermal_mean(self):
        """Mean eccentricity for thermal is (2/3) * e_max."""
        dist = ThermalEccentricity(e_max=0.99)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 100000)
        mean_e = jnp.mean(samples)
        expected_mean = (2.0 / 3.0) * 0.99
        assert jnp.abs(mean_e - expected_mean) < 0.01

    def test_thermal_cdf(self):
        """CDF is e² / e_max² for thermal distribution."""
        dist = ThermalEccentricity(e_max=0.99)
        e_vals = jnp.array([0.0, 0.5, 0.7, 0.99])
        expected = (e_vals / 0.99) ** 2
        computed = dist.cdf(e_vals)
        assert jnp.allclose(computed, expected, atol=1e-10)

    def test_ppf_inverse_cdf(self):
        """PPF is inverse of CDF."""
        dist = ThermalEccentricity()
        u_vals = jnp.array([0.0, 0.25, 0.49, 0.81, 1.0])
        e_vals = dist.ppf(u_vals)
        u_recovered = dist.cdf(e_vals)
        assert jnp.allclose(u_recovered, u_vals, atol=1e-10)


class TestUniformEccentricity:
    """Tests for uniform eccentricity distribution."""

    def test_sample_in_range(self):
        """Samples in [e_min, e_max]."""
        dist = UniformEccentricity(e_min=0.0, e_max=0.8)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 10000)
        assert jnp.all(samples >= 0.0 - 1e-6)
        assert jnp.all(samples <= 0.8 + 1e-6)

    def test_uniform_mean(self):
        """Mean eccentricity is (e_min + e_max) / 2."""
        dist = UniformEccentricity(e_min=0.1, e_max=0.9)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 100000)
        mean_e = jnp.mean(samples)
        assert jnp.abs(mean_e - 0.5) < 0.01


class TestIsotropicOrientations:
    """Tests for isotropic orbital orientation sampling."""

    def test_output_shape(self):
        """Returns correct shapes for all angles."""
        key = jax.random.PRNGKey(42)
        inc, Omega, omega, M_anom = sample_isotropic_orientations(key, 1000)
        assert inc.shape == (1000,)
        assert Omega.shape == (1000,)
        assert omega.shape == (1000,)
        assert M_anom.shape == (1000,)

    def test_inclination_range(self):
        """Inclination in [0, π]."""
        key = jax.random.PRNGKey(42)
        inc, _, _, _ = sample_isotropic_orientations(key, 10000)
        assert jnp.all(inc >= 0.0)
        assert jnp.all(inc <= jnp.pi)

    def test_inclination_isotropic(self):
        """cos(i) is uniformly distributed for isotropic orientations."""
        key = jax.random.PRNGKey(42)
        inc, _, _, _ = sample_isotropic_orientations(key, 100000)
        cos_inc = jnp.cos(inc)
        # For isotropic: cos(i) ~ U(-1, 1), so mean = 0
        mean_cos_inc = jnp.mean(cos_inc)
        assert jnp.abs(mean_cos_inc) < 0.02

    def test_angles_in_range(self):
        """Ω, ω, M₀ all in [0, 2π)."""
        key = jax.random.PRNGKey(42)
        _, Omega, omega, M_anom = sample_isotropic_orientations(key, 10000)

        for angle in [Omega, omega, M_anom]:
            assert jnp.all(angle >= 0.0)
            assert jnp.all(angle < 2 * jnp.pi)

    def test_angles_uniform(self):
        """Ω, ω, M₀ are uniformly distributed."""
        key = jax.random.PRNGKey(42)
        _, Omega, omega, M_anom = sample_isotropic_orientations(key, 100000)

        expected_mean = jnp.pi  # Mean of U(0, 2π)
        for angle in [Omega, omega, M_anom]:
            mean_angle = jnp.mean(angle)
            assert jnp.abs(mean_angle - expected_mean) < 0.05

    def test_reproducibility(self):
        """Same key gives same results."""
        key = jax.random.PRNGKey(42)
        result1 = sample_isotropic_orientations(key, 100)
        result2 = sample_isotropic_orientations(key, 100)
        for a1, a2 in zip(result1, result2):
            assert jnp.allclose(a1, a2)


class TestRadialBinaryFraction:
    """Tests for radially varying binary fraction."""

    def test_compute_shape(self):
        """Output has same shape as input radii."""
        from progenax.binaries.population import RadialBinaryFraction

        rbf = RadialBinaryFraction(fb0=0.5, A=0.3, alpha=1.0, r_scale=1.0)
        radii = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
        fb_r = rbf.compute(radii)
        assert fb_r.shape == radii.shape

    def test_fb_in_range(self):
        """Binary fraction clipped to [0, 1]."""
        from progenax.binaries.population import RadialBinaryFraction

        # Test with large A that could push fb > 1
        rbf = RadialBinaryFraction(fb0=0.8, A=2.0, alpha=1.0, r_scale=1.0)
        radii = jnp.linspace(0.01, 5.0, 100)
        fb_r = rbf.compute(radii)
        assert jnp.all(fb_r >= 0.0)
        assert jnp.all(fb_r <= 1.0)

    def test_A_positive_core_enhanced(self):
        """A > 0 gives higher fb at small radii (core-enhanced)."""
        from progenax.binaries.population import RadialBinaryFraction

        rbf = RadialBinaryFraction(fb0=0.5, A=0.5, alpha=1.0, r_scale=1.0)
        r_center = jnp.array([0.1])
        r_outer = jnp.array([5.0])

        fb_center = rbf.compute(r_center)
        fb_outer = rbf.compute(r_outer)

        # Core should have MORE binaries
        assert fb_center[0] > fb_outer[0]

    def test_A_negative_core_depleted(self):
        """A < 0 gives lower fb at small radii (core-depleted)."""
        from progenax.binaries.population import RadialBinaryFraction

        rbf = RadialBinaryFraction(fb0=0.5, A=-0.5, alpha=1.0, r_scale=1.0)
        r_center = jnp.array([0.1])
        r_outer = jnp.array([5.0])

        fb_center = rbf.compute(r_center)
        fb_outer = rbf.compute(r_outer)

        # Core should have FEWER binaries
        assert fb_center[0] < fb_outer[0]

    def test_A_zero_constant(self):
        """A = 0 gives constant binary fraction everywhere."""
        from progenax.binaries.population import RadialBinaryFraction

        rbf = RadialBinaryFraction(fb0=0.6, A=0.0, alpha=1.0, r_scale=1.0)
        radii = jnp.linspace(0.01, 10.0, 50)
        fb_r = rbf.compute(radii)

        # All should equal fb0
        assert jnp.allclose(fb_r, 0.6, atol=1e-6)

    def test_sample_membership_shape(self):
        """Sample membership returns boolean array of correct shape."""
        from progenax.binaries.population import RadialBinaryFraction

        rbf = RadialBinaryFraction(fb0=0.5, A=0.3, alpha=1.0, r_scale=1.0)
        radii = jnp.linspace(0.1, 3.0, 100)
        key = jax.random.PRNGKey(42)

        is_binary = rbf.sample_membership(radii, key)
        assert is_binary.shape == radii.shape
        assert is_binary.dtype == jnp.bool_

    def test_sample_membership_statistics(self):
        """Sample membership statistics match theoretical fb(r)."""
        from progenax.binaries.population import RadialBinaryFraction

        rbf = RadialBinaryFraction(fb0=0.5, A=0.0, alpha=1.0, r_scale=1.0)
        radii = jnp.ones(10000) * 1.0  # All at same radius
        key = jax.random.PRNGKey(42)

        is_binary = rbf.sample_membership(radii, key)
        fb_measured = jnp.mean(is_binary)
        fb_expected = rbf.compute(jnp.array([1.0]))[0]

        # Should be close for large N
        assert jnp.abs(fb_measured - fb_expected) < 0.02

    def test_jit_compatible(self):
        """RadialBinaryFraction works under JIT."""
        from progenax.binaries.population import RadialBinaryFraction

        rbf = RadialBinaryFraction(fb0=0.5, A=0.3, alpha=1.0, r_scale=1.0)

        @jax.jit
        def compute_fb(r):
            return rbf.compute(r)

        radii = jnp.array([0.5, 1.0, 2.0])
        fb_r = compute_fb(radii)
        assert fb_r.shape == (3,)


class TestSanaOBPeriod:
    """Tests for Sana+2012 O/B star period distribution."""

    def test_sample_shape(self):
        """Samples have correct shape."""
        from progenax.binaries.population import SanaOBPeriod

        dist = SanaOBPeriod()
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 1000)
        assert samples.shape == (1000,)

    def test_samples_in_range(self):
        """All samples within Sana+2012 range: log P in [0.3, 3.5]."""
        from progenax.binaries.population import SanaOBPeriod

        dist = SanaOBPeriod()
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 10000)
        log_samples = jnp.log10(samples)
        # Allow small numerical tolerance
        assert jnp.all(log_samples >= 0.3 - 1e-3)
        assert jnp.all(log_samples <= 3.5 + 1e-3)

    def test_mean_shorter_than_solar_type(self):
        """O/B stars have shorter periods on average than solar-type stars."""
        from progenax.binaries.population import SanaOBPeriod

        sana_dist = SanaOBPeriod()
        solar_dist = LogNormalPeriod(mu_log_P=4.8, sigma_log_P=2.3)

        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        sana_samples = sana_dist.sample(key1, 50000)
        solar_samples = solar_dist.sample(key2, 50000)

        mean_sana = jnp.median(sana_samples)
        mean_solar = jnp.median(solar_samples)

        # O/B stars should have shorter periods
        assert mean_sana < mean_solar

    def test_jit_compatible(self):
        """SanaOBPeriod works under JIT."""
        from progenax.binaries.population import SanaOBPeriod

        dist = SanaOBPeriod()

        @jax.jit
        def sample_periods(key):
            return dist.sample(key, 100)

        key = jax.random.PRNGKey(42)
        periods = sample_periods(key)
        assert periods.shape == (100,)


class TestMoeEccentricity:
    """Tests for Moe+2017 period-dependent eccentricity distribution."""

    def test_sample_shape(self):
        """Samples have correct shape."""
        from progenax.binaries.population import MoeEccentricity

        dist = MoeEccentricity()
        key = jax.random.PRNGKey(42)
        periods = jnp.array([1.0, 10.0, 100.0, 1000.0])
        samples = dist.sample(periods, key)
        assert samples.shape == periods.shape

    def test_samples_in_range(self):
        """All samples in [0, 1)."""
        from progenax.binaries.population import MoeEccentricity

        dist = MoeEccentricity()
        key = jax.random.PRNGKey(42)
        periods = jnp.logspace(0, 4, 1000)
        samples = dist.sample(periods, key)
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples < 1.0)

    def test_short_periods_more_circular(self):
        """Short periods (P < 10d) are more circular than long periods (P > 1000d)."""
        from progenax.binaries.population import MoeEccentricity

        dist = MoeEccentricity()
        key = jax.random.PRNGKey(42)

        # Short periods: P ~ 1-10 days (tidally circularized)
        periods_short = jnp.full(10000, 5.0)
        e_short = dist.sample(periods_short, key)

        # Long periods: P ~ 1000-10000 days (thermal-like)
        key = jax.random.PRNGKey(43)
        periods_long = jnp.full(10000, 5000.0)
        e_long = dist.sample(periods_long, key)

        mean_e_short = jnp.mean(e_short)
        mean_e_long = jnp.mean(e_long)

        # Short periods should be more circular (lower e)
        assert mean_e_short < mean_e_long

    def test_jit_compatible(self):
        """MoeEccentricity works under JIT."""
        from progenax.binaries.population import MoeEccentricity

        dist = MoeEccentricity()

        @jax.jit
        def sample_ecc(periods, key):
            return dist.sample(periods, key)

        key = jax.random.PRNGKey(42)
        periods = jnp.array([10.0, 100.0, 1000.0])
        eccentricities = sample_ecc(periods, key)
        assert eccentricities.shape == (3,)


class TestMassDependentBinaryConfig:
    """Tests for mass-dependent binary configuration."""

    def test_config_defaults(self):
        """Config can be created with default distributions."""
        from progenax.binaries.population import (
            MassDependentBinaryConfig,
            LogNormalPeriod,
            SanaOBPeriod,
            ThermalEccentricity,
            MoeEccentricity,
        )

        config = MassDependentBinaryConfig(
            m_break=8.0,
            low_mass_period=LogNormalPeriod(),
            high_mass_period=SanaOBPeriod(),
            low_mass_eccentricity=ThermalEccentricity(),
            high_mass_eccentricity=MoeEccentricity(),
        )

        assert config.m_break == 8.0
        assert isinstance(config.low_mass_period, LogNormalPeriod)
        assert isinstance(config.high_mass_period, SanaOBPeriod)
        assert isinstance(config.low_mass_eccentricity, ThermalEccentricity)
        assert isinstance(config.high_mass_eccentricity, MoeEccentricity)

    def test_config_custom_parameters(self):
        """Config can be created with custom distribution parameters."""
        from progenax.binaries.population import (
            MassDependentBinaryConfig,
            LogNormalPeriod,
            SanaOBPeriod,
            ThermalEccentricity,
            MoeEccentricity,
        )

        config = MassDependentBinaryConfig(
            m_break=10.0,
            low_mass_period=LogNormalPeriod(mu_log_P=5.0, sigma_log_P=1.5),
            high_mass_period=SanaOBPeriod(),
            low_mass_eccentricity=ThermalEccentricity(e_max=0.95),
            high_mass_eccentricity=MoeEccentricity(),
        )

        assert config.m_break == 10.0
        assert config.low_mass_period.mu_log_P == 5.0


class TestSampleMassDependentOrbits:
    """Tests for mass-dependent orbital parameter sampling."""

    def test_output_shapes(self):
        """Output periods and eccentricities have correct shapes."""
        from progenax.binaries.population import (
            sample_mass_dependent_orbits,
            MassDependentBinaryConfig,
            LogNormalPeriod,
            SanaOBPeriod,
            ThermalEccentricity,
            MoeEccentricity,
        )

        config = MassDependentBinaryConfig(
            m_break=8.0,
            low_mass_period=LogNormalPeriod(),
            high_mass_period=SanaOBPeriod(),
            low_mass_eccentricity=ThermalEccentricity(),
            high_mass_eccentricity=MoeEccentricity(),
        )

        masses = jnp.array([1.0, 5.0, 10.0, 20.0, 50.0])
        key = jax.random.PRNGKey(42)

        periods, eccentricities = sample_mass_dependent_orbits(masses, config, key)

        assert periods.shape == masses.shape
        assert eccentricities.shape == masses.shape

    def test_routes_by_mass(self):
        """Low-mass and high-mass stars get different distributions."""
        from progenax.binaries.population import (
            sample_mass_dependent_orbits,
            MassDependentBinaryConfig,
            LogNormalPeriod,
            SanaOBPeriod,
            ThermalEccentricity,
            MoeEccentricity,
        )

        config = MassDependentBinaryConfig(
            m_break=8.0,
            low_mass_period=LogNormalPeriod(mu_log_P=4.8, sigma_log_P=2.3),
            high_mass_period=SanaOBPeriod(),
            low_mass_eccentricity=ThermalEccentricity(),
            high_mass_eccentricity=MoeEccentricity(),
        )

        # Create many low-mass and high-mass stars
        masses_low = jnp.full(5000, 1.0)  # 1 Msun (solar-type)
        masses_high = jnp.full(5000, 20.0)  # 20 Msun (O-star)

        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        periods_low, _ = sample_mass_dependent_orbits(masses_low, config, key1)
        periods_high, _ = sample_mass_dependent_orbits(masses_high, config, key2)

        median_period_low = jnp.median(periods_low)
        median_period_high = jnp.median(periods_high)

        # High-mass stars (Sana) should have shorter periods than low-mass (log-normal)
        assert median_period_high < median_period_low

    def test_jit_compatible(self):
        """sample_mass_dependent_orbits works under JIT."""
        from progenax.binaries.population import (
            sample_mass_dependent_orbits,
            MassDependentBinaryConfig,
            LogNormalPeriod,
            SanaOBPeriod,
            ThermalEccentricity,
            MoeEccentricity,
        )

        config = MassDependentBinaryConfig(
            m_break=8.0,
            low_mass_period=LogNormalPeriod(),
            high_mass_period=SanaOBPeriod(),
            low_mass_eccentricity=ThermalEccentricity(),
            high_mass_eccentricity=MoeEccentricity(),
        )

        @jax.jit
        def sample_orbits(masses, key):
            return sample_mass_dependent_orbits(masses, config, key)

        masses = jnp.array([1.0, 5.0, 10.0, 20.0])
        key = jax.random.PRNGKey(42)

        periods, eccentricities = sample_orbits(masses, key)
        assert periods.shape == (4,)
        assert eccentricities.shape == (4,)

    def test_all_positive_periods(self):
        """All sampled periods are positive."""
        from progenax.binaries.population import (
            sample_mass_dependent_orbits,
            MassDependentBinaryConfig,
            LogNormalPeriod,
            SanaOBPeriod,
            ThermalEccentricity,
            MoeEccentricity,
        )

        config = MassDependentBinaryConfig(
            m_break=8.0,
            low_mass_period=LogNormalPeriod(),
            high_mass_period=SanaOBPeriod(),
            low_mass_eccentricity=ThermalEccentricity(),
            high_mass_eccentricity=MoeEccentricity(),
        )

        masses = jnp.logspace(-1, 2, 100)  # 0.1 to 100 Msun
        key = jax.random.PRNGKey(42)

        periods, eccentricities = sample_mass_dependent_orbits(masses, config, key)

        assert jnp.all(periods > 0)
        assert jnp.all(eccentricities >= 0.0)
        assert jnp.all(eccentricities < 1.0)
