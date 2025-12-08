"""Unit tests for differentiable IMF functions."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestLogProbMasses:
    """Test log_prob_masses PDF evaluation."""

    def test_returns_finite_values(self):
        """log_prob returns finite values for valid masses."""
        from progenax.imf.params import IMFParams
        from progenax.imf.differentiable import log_prob_masses

        params = IMFParams.kroupa()
        masses = jnp.array([0.1, 0.5, 1.0, 10.0, 50.0])

        log_probs = log_prob_masses(masses, params)

        assert jnp.all(jnp.isfinite(log_probs))
        assert log_probs.shape == (5,)

    def test_power_law_slope_high_mass(self):
        """PDF follows power law with slope alpha_high above m_break2."""
        from progenax.imf.params import IMFParams
        from progenax.imf.differentiable import log_prob_masses

        params = IMFParams.kroupa()  # alpha_high = 2.3

        # For power law ξ(m) ∝ m^(-α): log(ξ(m2)/ξ(m1)) = -α * log(m2/m1)
        m1, m2 = 10.0, 50.0  # Both above m_break2 = 0.5
        log_p1 = log_prob_masses(jnp.array([m1]), params)[0]
        log_p2 = log_prob_masses(jnp.array([m2]), params)[0]

        # Measured slope from PDF ratio
        measured_alpha = -(log_p2 - log_p1) / jnp.log(m2 / m1)

        assert jnp.isclose(measured_alpha, 2.3, atol=0.01)

    def test_power_law_slope_mid_mass(self):
        """PDF follows power law with slope alpha_mid in middle range."""
        from progenax.imf.params import IMFParams
        from progenax.imf.differentiable import log_prob_masses

        params = IMFParams.kroupa()  # alpha_mid = 1.3

        # Both in range [0.08, 0.5)
        m1, m2 = 0.1, 0.3
        log_p1 = log_prob_masses(jnp.array([m1]), params)[0]
        log_p2 = log_prob_masses(jnp.array([m2]), params)[0]

        measured_alpha = -(log_p2 - log_p1) / jnp.log(m2 / m1)

        assert jnp.isclose(measured_alpha, 1.3, atol=0.01)

    def test_normalization_integrates_to_one(self):
        """PDF integrates to ~1 over mass range (numerical check)."""
        from progenax.imf.params import IMFParams
        from progenax.imf.differentiable import log_prob_masses

        params = IMFParams.kroupa()

        # Numerical integration via Monte Carlo
        masses = jnp.logspace(jnp.log10(0.01), jnp.log10(150.0), 10000)
        log_probs = log_prob_masses(masses, params)
        probs = jnp.exp(log_probs)

        # Trapezoidal integration in log-space
        log_masses = jnp.log(masses)
        integral = jnp.trapezoid(probs * masses, log_masses)  # dm = m d(log m)

        assert jnp.isclose(integral, 1.0, atol=0.05)

    def test_gradient_through_alpha_high(self):
        """Can compute gradient of log_prob wrt alpha_high."""
        from progenax.imf.params import IMFParams
        from progenax.imf.differentiable import log_prob_masses

        def loss(alpha_high):
            params = IMFParams(
                alpha_low=jnp.array(0.3),
                alpha_mid=jnp.array(1.3),
                alpha_high=alpha_high,
            )
            masses = jnp.array([1.0, 10.0, 50.0])
            return jnp.sum(log_prob_masses(masses, params))

        grad_fn = jax.grad(loss)
        gradient = grad_fn(jnp.array(2.3))

        assert jnp.isfinite(gradient)
        assert gradient != 0.0

    def test_jit_compatible(self):
        """log_prob_masses works with JIT compilation."""
        from progenax.imf.params import IMFParams
        from progenax.imf.differentiable import log_prob_masses

        @jax.jit
        def compute_log_prob(masses, params):
            return log_prob_masses(masses, params)

        params = IMFParams.kroupa()
        masses = jnp.array([0.5, 1.0, 10.0])

        result = compute_log_prob(masses, params)

        assert jnp.all(jnp.isfinite(result))


class TestSampleMassesFromParams:
    """Test inverse CDF sampling."""

    def test_returns_correct_shape(self):
        """Output shape matches input uniform samples."""
        from progenax.imf.params import IMFParams
        from progenax.imf.differentiable import sample_masses_from_params

        params = IMFParams.kroupa()
        u = jnp.array([0.1, 0.5, 0.9])

        masses = sample_masses_from_params(params, u)

        assert masses.shape == (3,)

    def test_masses_in_valid_range(self):
        """All masses are within [m_min, m_max]."""
        from progenax.imf.params import IMFParams
        from progenax.imf.differentiable import sample_masses_from_params

        params = IMFParams.kroupa()
        key = jax.random.PRNGKey(42)
        u = jax.random.uniform(key, (1000,))

        masses = sample_masses_from_params(params, u)

        assert jnp.all(masses >= params.m_min)
        assert jnp.all(masses <= params.m_max)

    def test_monotonic_in_u(self):
        """Masses increase monotonically with u (inverse CDF property)."""
        from progenax.imf.params import IMFParams
        from progenax.imf.differentiable import sample_masses_from_params

        params = IMFParams.kroupa()
        u = jnp.linspace(0.01, 0.99, 100)

        masses = sample_masses_from_params(params, u)

        assert jnp.all(jnp.diff(masses) > 0)

    def test_distribution_matches_pdf(self):
        """Sampled masses follow the IMF PDF (KS-style test)."""
        from progenax.imf.params import IMFParams
        from progenax.imf.differentiable import sample_masses_from_params, log_prob_masses

        params = IMFParams.kroupa()
        key = jax.random.PRNGKey(42)
        u = jax.random.uniform(key, (10000,))

        masses = sample_masses_from_params(params, u)

        # Check high-mass slope via histogram
        high_mass = masses[masses > 1.0]
        log_m = jnp.log10(high_mass)
        hist, edges = jnp.histogram(log_m, bins=20)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Fit slope to histogram
        valid = hist > 10
        x = centers[valid]
        y = jnp.log10(hist[valid].astype(float))
        slope = jnp.sum((x - jnp.mean(x)) * (y - jnp.mean(y))) / jnp.sum((x - jnp.mean(x))**2)

        # For dn/d(log m) ∝ m^(1-α), histogram slope should be ~(1-α) = 1-2.3 = -1.3
        assert jnp.isclose(slope, -1.3, atol=0.2)

    def test_gradient_through_params(self):
        """Can compute gradient through sampled masses."""
        from progenax.imf.params import IMFParams
        from progenax.imf.differentiable import sample_masses_from_params

        def loss(alpha_high):
            params = IMFParams(
                alpha_low=jnp.array(0.3),
                alpha_mid=jnp.array(1.3),
                alpha_high=alpha_high,
            )
            u = jnp.array([0.5, 0.9, 0.99])  # Fixed uniforms
            masses = sample_masses_from_params(params, u)
            return jnp.mean(masses)

        grad_fn = jax.grad(loss)
        gradient = grad_fn(jnp.array(2.3))

        assert jnp.isfinite(gradient)
        # Steeper slope → fewer high-mass stars → lower mean mass
        # So d(mean)/d(alpha_high) should be negative
        assert gradient < 0

    def test_jit_compatible(self):
        """sample_masses_from_params works with JIT."""
        from progenax.imf.params import IMFParams
        from progenax.imf.differentiable import sample_masses_from_params

        @jax.jit
        def sample(params, u):
            return sample_masses_from_params(params, u)

        params = IMFParams.kroupa()
        u = jnp.array([0.1, 0.5, 0.9])

        masses = sample(params, u)

        assert jnp.all(jnp.isfinite(masses))
