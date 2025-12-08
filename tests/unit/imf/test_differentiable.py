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
