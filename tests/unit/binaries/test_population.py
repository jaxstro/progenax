"""Tests for binary population parameter sampling.

Physics tests only - distribution properties and scaling relations.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.binaries import (
    LogUniformPeriod,
    LogNormalPeriod,
    ThermalEccentricity,
    UniformEccentricity,
    sample_isotropic_orientations,
)


class TestLogUniformPeriod:
    """Test Öpik's law: log-uniform period distribution."""

    def test_log_uniform_distribution(self):
        """log10(P) is uniformly distributed (Öpik's law)."""
        dist = LogUniformPeriod(log_P_min=0.0, log_P_max=8.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 100000)
        log_samples = jnp.log10(samples)
        mean_log_P = jnp.mean(log_samples)
        # Mean of U(0, 8) should be 4.0
        assert jnp.abs(mean_log_P - 4.0) < 0.1

    def test_pdf_normalization(self):
        """PDF integrates to 1."""
        dist = LogUniformPeriod(log_P_min=0.0, log_P_max=4.0)
        P_vals = jnp.logspace(0, 4, 1000)
        pdf_vals = dist.pdf(P_vals)
        integral = jnp.trapezoid(pdf_vals, P_vals)
        assert jnp.abs(integral - 1.0) < 0.01


class TestLogNormalPeriod:
    """Test log-normal period distribution."""

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


class TestThermalEccentricity:
    """Test thermal eccentricity distribution f(e) = 2e."""

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

    def test_canonical_moments_emax1(self):
        """Canonical thermal f(e)=2e on [0,1]: <e>=2/3 and <e^2>=1/2.

        The existing mean test folds e_max into the expectation; this pins the
        textbook moments at e_max=1. Checking both moments jointly constrains the
        f(e)=2e shape (a wrong exponent e^k shifts both). At N=5e4, SEM ~1e-3, so
        the +/-0.01 bound is ~7-10 sigma.

        Reference: Jeans (1919); Heggie (1975) MNRAS 173, 729.
        """
        dist = ThermalEccentricity(e_max=1.0)
        samples = dist.sample(jax.random.PRNGKey(0), 50000)
        mean_e = jnp.mean(samples)
        mean_e2 = jnp.mean(samples ** 2)
        assert jnp.abs(mean_e - 2.0 / 3.0) < 0.01, f"<e>={float(mean_e):.4f}, expected 2/3"
        assert jnp.abs(mean_e2 - 0.5) < 0.01, f"<e^2>={float(mean_e2):.4f}, expected 1/2"


class TestUniformEccentricity:
    """Test uniform eccentricity distribution."""

    def test_uniform_mean(self):
        """Mean eccentricity is (e_min + e_max) / 2."""
        dist = UniformEccentricity(e_min=0.1, e_max=0.9)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 100000)
        mean_e = jnp.mean(samples)
        assert jnp.abs(mean_e - 0.5) < 0.01


class TestIsotropicOrientations:
    """Test isotropic orbital orientation sampling."""

    def test_inclination_isotropic(self):
        """cos(i) is uniformly distributed for isotropic orientations."""
        key = jax.random.PRNGKey(42)
        inc, _, _, _ = sample_isotropic_orientations(key, 100000)
        cos_inc = jnp.cos(inc)
        # For isotropic: cos(i) ~ U(-1, 1), so mean = 0
        mean_cos_inc = jnp.mean(cos_inc)
        assert jnp.abs(mean_cos_inc) < 0.02

    def test_angles_uniform(self):
        """Ω, ω, M₀ are uniformly distributed in [0, 2π)."""
        key = jax.random.PRNGKey(42)
        _, Omega, omega, M_anom = sample_isotropic_orientations(key, 100000)

        expected_mean = jnp.pi  # Mean of U(0, 2π)
        for angle in [Omega, omega, M_anom]:
            mean_angle = jnp.mean(angle)
            assert jnp.abs(mean_angle - expected_mean) < 0.05


class TestRadialBinaryFraction:
    """Test radially varying binary fraction."""

    def test_A_positive_core_enhanced(self):
        """A > 0 gives higher fb at small radii (core-enhanced)."""
        from progenax.binaries import RadialBinaryFraction

        rbf = RadialBinaryFraction(fb0=0.5, A=0.5, alpha=1.0, r_scale=1.0)
        r_center = jnp.array([0.1])
        r_outer = jnp.array([5.0])

        fb_center = rbf.compute(r_center)
        fb_outer = rbf.compute(r_outer)

        # Core should have MORE binaries
        assert fb_center[0] > fb_outer[0]

    def test_A_negative_core_depleted(self):
        """A < 0 gives lower fb at small radii (core-depleted)."""
        from progenax.binaries import RadialBinaryFraction

        rbf = RadialBinaryFraction(fb0=0.5, A=-0.5, alpha=1.0, r_scale=1.0)
        r_center = jnp.array([0.1])
        r_outer = jnp.array([5.0])

        fb_center = rbf.compute(r_center)
        fb_outer = rbf.compute(r_outer)

        # Core should have FEWER binaries
        assert fb_center[0] < fb_outer[0]

    def test_A_zero_constant(self):
        """A = 0 gives constant binary fraction everywhere."""
        from progenax.binaries import RadialBinaryFraction

        rbf = RadialBinaryFraction(fb0=0.6, A=0.0, alpha=1.0, r_scale=1.0)
        radii = jnp.linspace(0.01, 10.0, 50)
        fb_r = rbf.compute(radii)

        # All should equal fb0
        assert jnp.allclose(fb_r, 0.6, atol=1e-6)

    def test_sample_membership_statistics(self):
        """Sample membership statistics match theoretical fb(r)."""
        from progenax.binaries import RadialBinaryFraction

        rbf = RadialBinaryFraction(fb0=0.5, A=0.0, alpha=1.0, r_scale=1.0)
        radii = jnp.ones(10000) * 1.0  # All at same radius
        key = jax.random.PRNGKey(42)

        is_binary = rbf.sample_membership(radii, key)
        fb_measured = jnp.mean(is_binary)
        fb_expected = rbf.compute(jnp.array([1.0]))[0]

        # Should be close for large N
        assert jnp.abs(fb_measured - fb_expected) < 0.02


class TestSanaOBPeriod:
    """Test Sana+2012 O/B star period distribution."""

    def test_mean_shorter_than_solar_type(self):
        """O/B stars have shorter periods on average than solar-type stars."""
        from progenax.binaries import SanaOBPeriod

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

    def test_log_slope_recovers_minus_055(self):
        """Recovered power-law index matches Sana+2012: p(log P) ~ (log P)^-0.55.

        The sampler draws x = log10(P) with p(x) ~ x^power on [0.3, 3.5], so a
        log-log histogram of x has slope = power. Measured -0.551 +/- 0.009 over
        seeds (max dev 0.020), so +/-0.08 is ~9 sigma; it also excludes the
        log-uniform case (slope 0) by ~7 sigma, making the test discriminating.

        Reference: Sana et al. (2012) Science 337, 444.
        """
        from progenax.binaries import SanaOBPeriod

        dist = SanaOBPeriod()  # power = -0.55, log10(P) in [0.3, 3.5]
        key = jax.random.PRNGKey(0)
        x = jnp.log10(dist.sample(key, 50000))

        counts, edges = jnp.histogram(x, bins=24, range=(dist.log_P_min, dist.log_P_max))
        centers = 0.5 * (edges[1:] + edges[:-1])
        mask = counts > 0
        slope = jnp.polyfit(jnp.log(centers[mask]), jnp.log(counts[mask]), 1)[0]

        assert jnp.abs(slope - dist.power) < 0.08, (
            f"Recovered log-slope {float(slope):.3f}, expected {dist.power} (Sana+2012)"
        )


class TestSanaOBPeriodDistribution:
    """B4-9: SanaOBPeriod exposes the full pdf/cdf/ppf protocol (truncated power law)."""

    def test_ppf_inverts_cdf(self):
        from progenax.binaries import SanaOBPeriod
        d = SanaOBPeriod()
        u = jnp.array([0.05, 0.25, 0.5, 0.75, 0.95])
        assert jnp.allclose(d.cdf(d.ppf(u)), u, atol=1e-10)

    def test_cdf_endpoints(self):
        from progenax.binaries import SanaOBPeriod
        d = SanaOBPeriod()
        assert jnp.abs(d.cdf(10.0 ** d.log_P_min)) < 1e-12
        assert jnp.abs(d.cdf(10.0 ** d.log_P_max) - 1.0) < 1e-12

    def test_pdf_normalized(self):
        from progenax.binaries import SanaOBPeriod
        d = SanaOBPeriod()
        P = jnp.logspace(d.log_P_min, d.log_P_max, 20000)
        integral = jnp.trapezoid(d.pdf(P), P)
        assert jnp.abs(integral - 1.0) < 1e-3

    def test_ppf_inverts_cdf_log_uniform_branch(self):
        """alpha=-1 (Öpik) branch also satisfies cdf(ppf(u))=u."""
        from progenax.binaries import SanaOBPeriod
        d = SanaOBPeriod(power=-1.0)
        u = jnp.array([0.1, 0.5, 0.9])
        assert jnp.allclose(d.cdf(d.ppf(u)), u, atol=1e-10)

    def test_default_range_is_sana_2012(self):
        """Default log_P range matches Sana 2012 Fig.2: P ~ 1.4 d (logP=0.15) to ~9 yr (3.5)."""
        from progenax.binaries import SanaOBPeriod
        d = SanaOBPeriod()
        assert d.log_P_min == 0.15, "Sana 2012 Fig.2 fits down to P~1.4d (logP~0.15), not 0.3"
        assert d.log_P_max == 3.5

    def test_nonpositive_log_p_min_raises(self):
        """log_P_min <= 0 is out of domain: p(logP) ∝ (logP)^power undefined for logP<=0.

        Fail fast at construction with a clear ValueError, not a cryptic
        ZeroDivisionError deep inside sample/ppf.
        """
        from progenax.binaries import SanaOBPeriod
        with pytest.raises(ValueError, match="log_P_min"):
            SanaOBPeriod(log_P_min=0.0)
        with pytest.raises(ValueError, match="log_P_min"):
            SanaOBPeriod(log_P_min=-0.5)
        # valid construction still works
        assert SanaOBPeriod(log_P_min=0.15).log_P_min == 0.15


class TestLogisticThermalEccentricity:
    """Test the LogisticThermalEccentricity circular->thermal heuristic."""

    def test_short_periods_more_circular(self):
        """Short periods (P < 10d) are more circular than long periods (P > 1000d)."""
        from progenax.binaries import LogisticThermalEccentricity

        dist = LogisticThermalEccentricity()
        key = jax.random.PRNGKey(42)

        # Short periods: P ~ 1-10 days (tidally circularized)
        periods_short = jnp.full(10000, 5.0)
        e_short = dist.sample(key, periods_short)

        # Long periods: P ~ 1000-10000 days (thermal-like)
        key = jax.random.PRNGKey(43)
        periods_long = jnp.full(10000, 5000.0)
        e_long = dist.sample(key, periods_long)

        mean_e_short = jnp.mean(e_short)
        mean_e_long = jnp.mean(e_long)

        # Short periods should be more circular (lower e)
        assert mean_e_short < mean_e_long


class TestFaithfulMoeEccentricity:
    """4c: faithful Moe & Di Stefano (2017) eccentricity p(e) ∝ e^η(logP, M1).

    §9.2 Eq. 17 (late-type 0.8<M1<3 M☉):  η = 0.6 - 0.7/(logP - 0.5).
    §9.2 Eq. 18 (early-type M1>7 M☉):      η = 0.9 - 0.2/(logP - 0.5).
    Interpolate linearly in M1 for 3 <= M1 <= 7. e ∝ e^η on [0, e_max].
    """

    def test_eta_eq17_late_type(self):
        from progenax.binaries import MoeEccentricity
        moe = MoeEccentricity()
        # logP=3, M1=1: 0.6 - 0.7/2.5 = 0.32 ; logP=6: 0.6 - 0.7/5.5 ~ 0.473
        eta3 = moe.eta(jnp.array([1e3]), jnp.array([1.0]))
        eta6 = moe.eta(jnp.array([1e6]), jnp.array([1.0]))
        assert jnp.allclose(eta3, 0.6 - 0.7 / 2.5, atol=1e-6)
        assert jnp.allclose(eta6, 0.6 - 0.7 / 5.5, atol=1e-6)

    def test_eta_eq18_early_type(self):
        from progenax.binaries import MoeEccentricity
        moe = MoeEccentricity()
        # logP=3, M1=20: 0.9 - 0.2/2.5 = 0.82 ; logP=1: 0.9 - 0.2/0.5 = 0.5
        assert jnp.allclose(moe.eta(jnp.array([1e3]), jnp.array([20.0])), 0.9 - 0.2 / 2.5, atol=1e-6)
        assert jnp.allclose(moe.eta(jnp.array([1e1]), jnp.array([20.0])), 0.9 - 0.2 / 0.5, atol=1e-6)

    def test_eta_interpolates_intermediate_mass(self):
        """M1=5 (midpoint of 3-7) is halfway between Eq.17 and Eq.18."""
        from progenax.binaries import MoeEccentricity
        moe = MoeEccentricity()
        P = jnp.array([1e3])
        late = 0.6 - 0.7 / 2.5
        early = 0.9 - 0.2 / 2.5
        assert jnp.allclose(moe.eta(P, jnp.array([5.0])), 0.5 * (late + early), atol=1e-6)

    def test_sampled_mean_matches_eta(self):
        """<e> of e^η on [0, e_max_eff(P)] is e_max_eff (η+1)/(η+2)."""
        from progenax.binaries import MoeEccentricity
        moe = MoeEccentricity(e_max=0.99)
        key = jax.random.PRNGKey(0)
        P = jnp.full(200000, 1e3)
        M = jnp.full(200000, 20.0)  # eta = 0.82
        eta = 0.9 - 0.2 / 2.5
        e = moe.sample(key, P, M)
        # At P=1e3 d the Roche cap (Moe Eq.3) e_max(P)=1-(500)^(-2/3)=0.984 binds
        # below the 0.99 numerical ceiling, so the mean uses e_max_eff, not 0.99.
        emax_eff = 1.0 - (1e3 / 2.0) ** (-2.0 / 3.0)
        expected = emax_eff * (eta + 1.0) / (eta + 2.0)
        assert jnp.abs(jnp.mean(e) - expected) < 0.01, f"<e>={float(jnp.mean(e)):.3f} vs {expected:.3f}"

    def test_emax_follows_roche_eq3(self):
        """e_max(P) = 1 - (P/2 days)^(-2/3) for P>2 d (Moe & Di Stefano 2017 Eq. 3)."""
        from progenax.binaries import MoeEccentricity
        moe = MoeEccentricity(e_max=0.99)
        P = jnp.array([10.0, 100.0, 1000.0])
        expected = 1.0 - (P / 2.0) ** (-2.0 / 3.0)
        got = moe.e_max_of_period(P)
        assert jnp.allclose(got, expected, atol=1e-6), f"{got} vs {expected}"
        # P=10 d -> e_max ~ 0.658, well below the 0.99 numerical ceiling
        assert jnp.abs(float(moe.e_max_of_period(jnp.array([10.0]))[0]) - 0.658) < 1e-3

    def test_emax_circular_at_and_below_2_days(self):
        """P <= 2 d (Roche/contact limit) -> e_max = 0 (forced circular)."""
        from progenax.binaries import MoeEccentricity
        moe = MoeEccentricity()
        got = moe.e_max_of_period(jnp.array([2.0, 1.0, 0.5]))
        assert jnp.all(got == 0.0), f"{got}"

    def test_emax_ceiling_caps_long_period(self):
        """At very long P the Roche relation -> 1; the e_max field is the ceiling."""
        from progenax.binaries import MoeEccentricity
        moe = MoeEccentricity(e_max=0.99)
        assert jnp.abs(float(moe.e_max_of_period(jnp.array([1e8]))[0]) - 0.99) < 1e-6

    def test_no_roche_overflow_short_period(self):
        """No sampled e exceeds e_max(P) at short period (Roche-lobe physical)."""
        from progenax.binaries import MoeEccentricity
        moe = MoeEccentricity(e_max=0.99)
        key = jax.random.PRNGKey(7)
        P = jnp.full(20000, 10.0)
        M = jnp.full(20000, 20.0)  # early-type: eta(logP=1)=0.5 (NOT circular)
        e = moe.sample(key, P, M)
        emax10 = float(moe.e_max_of_period(jnp.array([10.0]))[0])  # ~0.658
        assert float(jnp.max(e)) <= emax10 + 1e-6, (
            f"max e {float(jnp.max(e))} > e_max(10d) {emax10}"
        )

    def test_sample_grad_finite_wrt_period(self):
        """d<e>/d(period scale) is finite (new P-dependence via e_max(P))."""
        from progenax.binaries import MoeEccentricity
        key = jax.random.PRNGKey(8)
        M = jnp.full(4000, 20.0)

        def loss(scale):
            P = jnp.full(4000, 50.0) * scale
            return jnp.mean(MoeEccentricity().sample(key, P, M))

        g = jax.grad(loss)(1.0)
        assert jnp.isfinite(g), f"grad wrt period scale = {g}"

    def test_short_period_circularizes(self):
        """Very short P (eta <= -1) returns near-circular e ~ 0."""
        from progenax.binaries import MoeEccentricity
        moe = MoeEccentricity()
        key = jax.random.PRNGKey(1)
        P = jnp.full(20000, 10.0 ** 0.6)  # late eta=-6.4, early eta=-1.1 -> circular
        e = moe.sample(key, P, jnp.full(20000, 1.0))
        assert jnp.mean(e) < 0.05, f"<e>={float(jnp.mean(e)):.3f} should be ~circular"

    def test_pdf_normalized(self):
        from progenax.binaries import MoeEccentricity
        moe = MoeEccentricity(e_max=0.99)
        P = jnp.array([1e3])
        M = jnp.array([20.0])
        # Integrate over [0, e_max_eff(P)] where the Roche cap (Eq.3) places support.
        emax_eff = 1.0 - (1e3 / 2.0) ** (-2.0 / 3.0)
        e = jnp.linspace(0.0, float(emax_eff), 20000)
        integral = jnp.trapezoid(moe.pdf(e, P, M).ravel(), e)
        assert jnp.abs(integral - 1.0) < 1e-3

    def test_emax_gradient_matches_fd(self):
        """d<e>/d(e_max) (FD vs autodiff): e = e_max u^(1/(η+1))."""
        from progenax.binaries import MoeEccentricity
        key = jax.random.PRNGKey(3)
        # P=1e8 d: the Roche relation e_max(P)->1, so the e_max FIELD is the binding
        # ceiling and d<e>/d(e_max) is non-degenerate (at shorter P the period cap
        # makes the field non-binding and the gradient legitimately 0).
        P = jnp.full(4000, 1e8)
        M = jnp.full(4000, 20.0)  # eta = 0.873

        def loss(em):
            return jnp.mean(MoeEccentricity(e_max=em).sample(key, P, M))

        em0 = 0.99
        g_ad = jax.grad(loss)(em0)
        rel = min(
            float(jnp.abs(g_ad - (loss(em0 + h) - loss(em0 - h)) / (2.0 * h)) / (jnp.abs(g_ad) + 1e-12))
            for h in (1e-3, 1e-4, 1e-5)
        )
        assert jnp.isfinite(g_ad) and rel < 1e-5, f"Moe e_max grad FD rel-err {rel:.2e}"


class TestMassDependentOrbits:
    """Test mass-dependent orbital parameter sampling."""

    def test_routes_by_mass(self):
        """Low-mass and high-mass stars get different distributions."""
        from progenax.binaries import (
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


class TestBinarySamplingDifferentiability:
    """Reparameterization gradients through the period/eccentricity samplers.

    Each sampler is an inverse-CDF or location-scale reparameterization, so
    mean(samples) is a smooth, differentiable function of the distribution
    parameters at fixed key. These tests assert gradient *correctness* (against a
    finite difference or a closed form), not merely finiteness, since the
    differentiable-IC pipeline relies on these gradients (gradient-validation).
    """

    def test_sana_power_gradient_matches_finite_difference(self):
        """d<log10 P>/d(power) for SanaOB matches a central finite difference.

        The Sana sampler inverts a power-law CDF via jnp.power, so the gradient
        w.r.t. the index is non-trivial. With a fixed key the loss is
        deterministic, so autodiff must match a central difference to <1e-5
        (h-sweep to bracket the truncation/round-off sweet spot).
        """
        from progenax.binaries import SanaOBPeriod

        key = jax.random.PRNGKey(0)

        def loss(power):
            return jnp.mean(jnp.log10(SanaOBPeriod(power=power).sample(key, 20000)))

        p0 = -0.55
        g_ad = jax.grad(loss)(p0)
        assert jnp.isfinite(g_ad)

        rel_errs = []
        for h in (1e-3, 1e-4, 1e-5):
            g_fd = (loss(p0 + h) - loss(p0 - h)) / (2.0 * h)
            rel_errs.append(
                float(jnp.abs(g_ad - g_fd) / (jnp.abs(g_ad) + jnp.abs(g_fd) + 1e-12))
            )
        assert min(rel_errs) < 1e-5, f"min FD rel-err {min(rel_errs):.2e}"

    def test_lognormal_location_gradient_is_unity(self):
        """d<log10 P>/d(mu) = 1 exactly (log10 P = mu + sigma*z)."""
        from progenax.binaries import LogNormalPeriod

        key = jax.random.PRNGKey(1)
        g = jax.grad(
            lambda mu: jnp.mean(jnp.log10(LogNormalPeriod(mu_log_P=mu).sample(key, 20000)))
        )(4.8)
        assert jnp.isfinite(g)
        assert jnp.abs(g - 1.0) < 1e-6, f"d<log10P>/dmu={float(g):.6f}, expected 1"

    def test_thermal_scale_gradient_equals_mean_sqrt_u(self):
        """d<e>/d(e_max) = <sqrt(u)> exactly (e = e_max*sqrt(u)), ~2/3 in the limit."""
        from progenax.binaries import ThermalEccentricity

        key = jax.random.PRNGKey(2)
        # Closed form: with the same key, the sampler draws this exact u.
        expected = jnp.mean(jnp.sqrt(jax.random.uniform(key, (20000,))))
        g = jax.grad(
            lambda em: jnp.mean(ThermalEccentricity(e_max=em).sample(key, 20000))
        )(1.0)
        assert jnp.isfinite(g)
        assert jnp.abs(g - expected) < 1e-6, f"d<e>/d(e_max)={float(g):.6f} != <sqrt u>"
        assert jnp.abs(g - 2.0 / 3.0) < 0.02  # converges to <sqrt u> = 2/3

    def test_lognormal_ppf_grad_finite_at_boundary(self):
        """LogNormalPeriod.ppf has finite grad at u=0,1 (protocol: ppf differentiable)."""
        from progenax.binaries import LogNormalPeriod
        ppf = LogNormalPeriod().ppf
        for u in (0.0, 1.0, 0.5):
            g = jax.grad(ppf)(u)
            assert jnp.isfinite(g), f"LogNormal ppf grad at u={u} = {g}"

    def test_thermal_ppf_grad_finite_at_boundary(self):
        """ThermalEccentricity.ppf has finite grad at u=0 (sqrt singularity guarded)."""
        from progenax.binaries import ThermalEccentricity
        ppf = ThermalEccentricity().ppf
        for u in (0.0, 1.0, 0.5):
            g = jax.grad(ppf)(u)
            assert jnp.isfinite(g), f"Thermal ppf grad at u={u} = {g}"


class TestSanaOBPeriodAlphaMinusOne:
    """B4-2 regression: SanaOBPeriod must be finite + grad-safe at the alpha=-1 index.

    The historical bug raised ZeroDivisionError in sample() and returned a NaN
    gradient at power=-1 (the IMF-B5 double-where trap, population.py:283). The
    alpha=-1 member must ALSO be the true alpha->-1 limit of the power-law family
    (log-uniform in log10 P), i.e. continuous across -1 (finding B4-2b).

    Reference: Sana et al. (2012); power=-1 is the Opik/log-uniform special case.
    """

    def test_sample_finite_at_power_minus_one(self):
        from progenax.binaries import SanaOBPeriod
        key = jax.random.PRNGKey(0)
        s = SanaOBPeriod(power=-1.0).sample(key, 200)
        assert jnp.all(jnp.isfinite(s)) and jnp.all(s > 0)

    def test_power_minus_one_matches_general_limit(self):
        """B4-2b: the alpha=-1 special case must equal the alpha->-1 limit of the
        general inverse-CDF (same key => same u draws => same per-sample log P)."""
        from progenax.binaries import SanaOBPeriod
        key = jax.random.PRNGKey(1)
        x_exact = jnp.log10(SanaOBPeriod(power=-1.0).sample(key, 5000))
        x_near = jnp.log10(SanaOBPeriod(power=-1.0 + 1e-6).sample(key, 5000))
        dev = float(jnp.max(jnp.abs(x_exact - x_near)))
        assert dev < 1e-3, f"alpha=-1 branch deviates from the alpha->-1 limit by {dev:.2e}"

    def test_grad_finite_at_power_minus_one(self):
        from progenax.binaries import SanaOBPeriod
        key = jax.random.PRNGKey(2)
        def loss(p):
            return jnp.mean(jnp.log10(SanaOBPeriod(power=p).sample(key, 2000)))
        assert jnp.isfinite(jax.grad(loss)(-1.0))


class TestMassDependentBinaryConfigPyTree:
    """B4-4 regression: MassDependentBinaryConfig must be a proper JAX PyTree.

    A frozen @dataclass registers as ONE opaque leaf, so its submodule array
    leaves are invisible to tree_map/vmap/grad over the config (the docstring's
    'JIT-native' claim does not hold). An eqx.Module exposes the inner leaves.
    """

    def test_config_is_pytree(self):
        import jax.tree_util as jtu
        from progenax.binaries import (
            MassDependentBinaryConfig, LogNormalPeriod, SanaOBPeriod,
            ThermalEccentricity, MoeEccentricity,
        )
        cfg = MassDependentBinaryConfig(
            m_break=8.0,
            low_mass_period=LogNormalPeriod(),
            high_mass_period=SanaOBPeriod(),
            low_mass_eccentricity=ThermalEccentricity(),
            high_mass_eccentricity=MoeEccentricity(),
        )
        leaves = jtu.tree_leaves(cfg)
        assert len(leaves) > 1, f"config is not a PyTree (tree_leaves={len(leaves)})"


class TestMoreSamplerGradients:
    """B4-15: FD-vs-autodiff grad-checks for the samplers that lacked them
    (Moe eccentricity, Uniform eccentricity, LogUniform period). Fixed key =>
    deterministic loss => autodiff must match a central finite difference.
    """

    def test_logistic_thermal_emax_gradient_matches_fd(self):
        from progenax.binaries import LogisticThermalEccentricity
        key = jax.random.PRNGKey(0)
        periods = jnp.logspace(0.5, 3.5, 4000)  # fixed period grid

        def loss(em):
            return jnp.mean(LogisticThermalEccentricity(e_max=em).sample(key, periods))

        em0 = 0.99
        g_ad = jax.grad(loss)(em0)
        assert jnp.isfinite(g_ad)
        rel = min(
            float(jnp.abs(g_ad - (loss(em0 + h) - loss(em0 - h)) / (2.0 * h)) / (jnp.abs(g_ad) + 1e-12))
            for h in (1e-3, 1e-4, 1e-5)
        )
        assert rel < 1e-5, f"Moe e_max grad FD rel-err {rel:.2e}"

    def test_uniform_eccentricity_emax_gradient(self):
        from progenax.binaries import UniformEccentricity
        key = jax.random.PRNGKey(1)
        # e = e_min + u(e_max - e_min)  =>  d<e>/d(e_max) = <u>
        expected = jnp.mean(jax.random.uniform(key, (20000,)))
        g = jax.grad(
            lambda em: jnp.mean(UniformEccentricity(e_min=0.0, e_max=em).sample(key, 20000))
        )(0.9)
        assert jnp.isfinite(g) and jnp.abs(g - expected) < 1e-6
        assert jnp.abs(g - 0.5) < 0.02

    def test_loguniform_period_logpmax_gradient(self):
        from progenax.binaries import LogUniformPeriod
        key = jax.random.PRNGKey(2)
        # log10 P = lo + u (hi - lo)  =>  d<log10 P>/d(hi) = <u>
        expected = jnp.mean(jax.random.uniform(key, (20000,)))
        g = jax.grad(
            lambda hi: jnp.mean(jnp.log10(LogUniformPeriod(log_P_min=0.0, log_P_max=hi).sample(key, 20000)))
        )(8.0)
        assert jnp.isfinite(g) and jnp.abs(g - expected) < 1e-6
        assert jnp.abs(g - 0.5) < 0.02


class TestDistributionProtocols:
    """B4-9: samplers satisfy the runtime_checkable Period/Eccentricity protocols."""

    def test_period_distributions_conform(self):
        from progenax.protocols import PeriodDistribution
        from progenax.binaries import LogUniformPeriod, LogNormalPeriod, SanaOBPeriod
        for d in (LogUniformPeriod(), LogNormalPeriod(), SanaOBPeriod()):
            assert isinstance(d, PeriodDistribution), type(d).__name__

    def test_unconditional_eccentricity_conform(self):
        from progenax.protocols import EccentricityDistribution
        from progenax.binaries import ThermalEccentricity, UniformEccentricity
        for d in (ThermalEccentricity(), UniformEccentricity()):
            assert isinstance(d, EccentricityDistribution), type(d).__name__

    def test_conditional_eccentricity_protocols(self):
        from progenax.protocols import (
            EccentricityDistribution,
            ConditionalEccentricityDistribution,
            MassPeriodEccentricityDistribution,
        )
        from progenax.binaries import LogisticThermalEccentricity, MoeEccentricity
        # Faithful Moe is period+mass conditional; the heuristic is period-conditional.
        assert isinstance(MoeEccentricity(), MassPeriodEccentricityDistribution)
        assert isinstance(LogisticThermalEccentricity(), ConditionalEccentricityDistribution)
        # Neither exposes the full unconditional sample/pdf/cdf/ppf quartet.
        assert not isinstance(MoeEccentricity(), EccentricityDistribution)
        assert not isinstance(LogisticThermalEccentricity(), EccentricityDistribution)
