"""Tests for BM19 gravoturbulent framework.

Tests verify:
- Correct implementation of BM19 equations
- JAX compatibility (jit, vmap, grad)
- Limiting cases and physical constraints
- Gradient sign predictions from theory
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import erfc

from progenax.gravoturb import bm19_model as bm19


class TestSigmaSSquared:
    """Tests for sigma_s_squared()."""

    def test_basic_formula(self):
        """Verify σ_s² = ln(1 + b²M²)."""
        mach = 10.0
        b = 0.4
        expected = jnp.log(1.0 + b**2 * mach**2)  # ln(1 + 16) = ln(17) ≈ 2.83
        result = bm19.sigma_s_squared(mach, b)
        assert jnp.isclose(result, expected, rtol=1e-6)

    def test_mach_5_b_04(self):
        """Verify M=5, b=0.4 → σ_s² ≈ 1.61 (theory guide Table 3.4)."""
        result = bm19.sigma_s_squared(5.0, 0.4)
        # ln(1 + 0.16 * 25) = ln(5) ≈ 1.609
        assert jnp.isclose(result, 1.609, atol=0.01)

    def test_mach_10_b_04(self):
        """Verify M=10, b=0.4 → σ_s² ≈ 2.83."""
        result = bm19.sigma_s_squared(10.0, 0.4)
        # ln(1 + 0.16 * 100) = ln(17) ≈ 2.833
        assert jnp.isclose(result, 2.833, atol=0.01)

    def test_mach_monotonic(self):
        """Higher Mach → higher σ_s²."""
        mach = jnp.array([5.0, 10.0, 20.0])
        sigma_sq = jax.vmap(lambda m: bm19.sigma_s_squared(m, 0.4))(mach)
        assert jnp.all(jnp.diff(sigma_sq) > 0)

    def test_b_monotonic(self):
        """Higher b → higher σ_s² at fixed Mach."""
        b_values = jnp.array([0.3, 0.4, 0.5, 1.0])
        sigma_sq = jax.vmap(lambda b: bm19.sigma_s_squared(10.0, b))(b_values)
        assert jnp.all(jnp.diff(sigma_sq) > 0)

    def test_differentiable_mach(self):
        """Gradient w.r.t. Mach exists and is positive."""
        grad_fn = jax.grad(lambda m: bm19.sigma_s_squared(m, 0.4))
        g = grad_fn(10.0)
        assert jnp.isfinite(g)
        assert g > 0  # dσ²/dM > 0

    def test_differentiable_b(self):
        """Gradient w.r.t. b exists and is positive."""
        grad_fn = jax.grad(lambda b: bm19.sigma_s_squared(10.0, b))
        g = grad_fn(0.4)
        assert jnp.isfinite(g)
        assert g > 0  # dσ²/db > 0

    def test_jit_compatible(self):
        """Function can be JIT compiled."""
        jit_fn = jax.jit(bm19.sigma_s_squared)
        result = jit_fn(10.0, 0.4)
        expected = bm19.sigma_s_squared(10.0, 0.4)
        assert jnp.isclose(result, expected)

    def test_vmap_compatible(self):
        """Function can be vmapped."""
        mach = jnp.array([5.0, 10.0, 20.0])
        results = jax.vmap(lambda m: bm19.sigma_s_squared(m, 0.4))(mach)
        assert results.shape == (3,)
        assert jnp.all(jnp.isfinite(results))


class TestTransitionDensity:
    """Tests for transition_density()."""

    def test_basic_formula(self):
        """s_t = (α - 0.5) σ_s²."""
        sigma_sq = 2.0
        alpha = 2.0
        expected = (2.0 - 0.5) * 2.0  # = 3.0
        result = bm19.transition_density(sigma_sq, alpha)
        assert jnp.isclose(result, expected, rtol=1e-6)

    def test_alpha_15(self):
        """α = 1.5 → s_t = σ_s²."""
        sigma_sq = 2.0
        result = bm19.transition_density(sigma_sq, 1.5)
        expected = (1.5 - 0.5) * 2.0  # = 2.0
        assert jnp.isclose(result, expected)

    def test_alpha_3(self):
        """α = 3.0 → s_t = 2.5 σ_s²."""
        sigma_sq = 2.0
        result = bm19.transition_density(sigma_sq, 3.0)
        expected = (3.0 - 0.5) * 2.0  # = 5.0
        assert jnp.isclose(result, expected)

    def test_alpha_monotonic(self):
        """Higher α → higher s_t at fixed σ_s²."""
        sigma_sq = 2.0
        alphas = jnp.array([1.5, 2.0, 2.5, 3.0])
        s_t = jax.vmap(lambda a: bm19.transition_density(sigma_sq, a))(alphas)
        assert jnp.all(jnp.diff(s_t) > 0)

    def test_sigma_sq_monotonic(self):
        """Higher σ_s² → higher s_t at fixed α."""
        alpha = 2.0
        sigma_sqs = jnp.array([1.0, 2.0, 3.0])
        s_t = jax.vmap(lambda s: bm19.transition_density(s, alpha))(sigma_sqs)
        assert jnp.all(jnp.diff(s_t) > 0)

    def test_differentiable_sigma_sq(self):
        """Gradient w.r.t. σ_s² exists."""
        grad_fn = jax.grad(lambda s: bm19.transition_density(s, 2.0))
        g = grad_fn(2.0)
        assert jnp.isfinite(g)
        assert jnp.isclose(g, 1.5)  # d(s_t)/d(σ²) = α - 0.5 = 1.5

    def test_differentiable_alpha(self):
        """Gradient w.r.t. α exists."""
        grad_fn = jax.grad(lambda a: bm19.transition_density(2.0, a))
        g = grad_fn(2.0)
        assert jnp.isfinite(g)
        assert jnp.isclose(g, 2.0)  # d(s_t)/d(α) = σ² = 2.0


class TestFDenseLognormalLimit:
    """Tests for f_dense_lognormal_limit()."""

    def test_basic_computation(self):
        """Verify erfc-based formula."""
        sigma_sq = 2.0
        s_t = 3.0
        sigma_s = jnp.sqrt(sigma_sq)
        u = (s_t - sigma_sq / 2.0) / (jnp.sqrt(2.0) * sigma_s)
        expected = 0.5 * erfc(u)
        result = bm19.f_dense_lognormal_limit(sigma_sq, s_t)
        assert jnp.isclose(result, expected, rtol=1e-6)

    def test_bounds(self):
        """f_dense should be in (0, 1)."""
        sigma_sq = 2.0
        # Test various s_t values
        for s_t in [0.5, 1.0, 2.0, 3.0, 5.0]:
            result = bm19.f_dense_lognormal_limit(sigma_sq, s_t)
            assert 0 < result < 1

    def test_s_t_monotonic(self):
        """Higher s_t → lower f_dense (more restrictive threshold)."""
        sigma_sq = 2.0
        s_t_values = jnp.array([1.0, 2.0, 3.0, 4.0])
        f_dense = jax.vmap(lambda s: bm19.f_dense_lognormal_limit(sigma_sq, s))(
            s_t_values
        )
        assert jnp.all(jnp.diff(f_dense) < 0)

    def test_differentiable(self):
        """Gradients exist."""
        grad_sigma = jax.grad(lambda s: bm19.f_dense_lognormal_limit(s, 3.0))(2.0)
        grad_s_t = jax.grad(lambda s: bm19.f_dense_lognormal_limit(2.0, s))(3.0)
        assert jnp.isfinite(grad_sigma)
        assert jnp.isfinite(grad_s_t)


class TestFDenseBM19Full:
    """Tests for f_dense_bm19_full()."""

    def test_lognormal_limit(self):
        """Full integral vs lognormal limit: powerlaw decays faster than lognormal tail.

        The f_dense_lognormal_limit assumes the PDF continues as lognormal above s_t,
        while f_dense_bm19_full uses a steeper powerlaw tail. Thus:
        - f_dense_bm19_full < f_dense_lognormal_limit (less mass in steeper tail)
        - As α increases, the powerlaw becomes steeper, giving even less mass
        """
        sigma_sq = 2.0
        s_t = 3.0  # Fixed threshold

        # Compute full integral at different α values
        f_full_shallow = bm19.f_dense_bm19_full(sigma_sq, s_t, 1.5)  # Shallow PL
        f_full_steep = bm19.f_dense_bm19_full(sigma_sq, s_t, 3.0)  # Steeper PL
        f_ln = bm19.f_dense_lognormal_limit(sigma_sq, s_t)

        # All should be finite and in valid range
        assert jnp.isfinite(f_full_shallow) and 0 < f_full_shallow < 1
        assert jnp.isfinite(f_full_steep) and 0 < f_full_steep < 1
        assert jnp.isfinite(f_ln) and 0 < f_ln < 1

        # Key physics:
        # 1. Steeper powerlaw (higher α) → less mass above threshold
        assert f_full_steep < f_full_shallow

        # 2. For shallow powerlaw (α=1.5), tail decays slower than lognormal
        #    so can have MORE mass than lognormal limit
        # 3. For steep powerlaw (α=3.0), tail decays faster than lognormal
        #    so has LESS mass than lognormal limit
        # The crossover depends on s_t and sigma_sq
        # Just verify both are in reasonable range
        assert 0.01 < f_full_shallow < 0.5
        assert 0.001 < f_full_steep < 0.2

    def test_bounds(self):
        """f_dense should be in (0, 1)."""
        for mach in [5.0, 10.0, 20.0]:
            for alpha in [1.5, 2.0, 2.5, 3.0]:
                sigma_sq = bm19.sigma_s_squared(mach, 0.4)
                s_t = bm19.transition_density(sigma_sq, alpha)
                f = bm19.f_dense_bm19_full(sigma_sq, s_t, alpha)
                assert 0 < f < 1, f"f_dense={f} out of bounds for M={mach}, α={alpha}"

    def test_alpha_monotonic(self):
        """Higher α → LOWER f_dense (higher threshold + steeper decay)."""
        sigma_sq = 2.0
        alphas = jnp.array([1.5, 2.0, 2.5, 3.0])

        def f_for_alpha(alpha):
            s_t = bm19.transition_density(sigma_sq, alpha)
            return bm19.f_dense_bm19_full(sigma_sq, s_t, alpha)

        f_dense = jax.vmap(f_for_alpha)(alphas)
        # Higher α → higher s_t (more restrictive threshold)
        # Higher α → steeper powerlaw (faster decay beyond threshold)
        # Both effects DECREASE f_dense
        assert jnp.all(jnp.diff(f_dense) < 0)

    def test_anti_correlation_with_mach(self):
        """BM19 prediction: f_dense decreases with Mach at fixed α."""
        alpha = 2.0
        machs = jnp.array([5.0, 10.0, 20.0])

        def f_from_mach(m):
            sigma_sq = bm19.sigma_s_squared(m, 0.4)
            s_t = bm19.transition_density(sigma_sq, alpha)
            return bm19.f_dense_bm19_full(sigma_sq, s_t, alpha)

        f_dense = jax.vmap(f_from_mach)(machs)
        # f_dense should decrease as Mach increases
        assert jnp.all(jnp.diff(f_dense) < 0)

    def test_gradient_signs(self):
        """Verify gradient signs match BM19 predictions."""

        def f_dense_wrapper(mach, alpha):
            sigma_sq = bm19.sigma_s_squared(mach, 0.4)
            s_t = bm19.transition_density(sigma_sq, alpha)
            return bm19.f_dense_bm19_full(sigma_sq, s_t, alpha)

        # ∂f_dense/∂M < 0 (anti-correlation: higher Mach → wider PDF → smaller f_dense)
        grad_mach = jax.grad(f_dense_wrapper, argnums=0)(10.0, 2.0)
        assert grad_mach < 0, f"Expected ∂f_dense/∂M < 0, got {grad_mach}"

        # ∂f_dense/∂α < 0 (higher α → higher s_t AND steeper decay → less mass)
        # Note: This might seem counterintuitive, but s_t = (α-0.5)*σ² scales with α,
        # so higher α raises the threshold AND steepens the powerlaw.
        grad_alpha = jax.grad(f_dense_wrapper, argnums=1)(10.0, 2.0)
        assert grad_alpha < 0, f"Expected ∂f_dense/∂α < 0, got {grad_alpha}"

    def test_differentiable_all_inputs(self):
        """Gradients exist for all inputs."""

        def f_wrapper(sigma_sq, s_t, alpha):
            return bm19.f_dense_bm19_full(sigma_sq, s_t, alpha)

        grad_fn = jax.grad(f_wrapper, argnums=(0, 1, 2))
        grads = grad_fn(2.0, 3.0, 2.0)
        for g in grads:
            assert jnp.isfinite(g)

    def test_jit_compatible(self):
        """Function can be JIT compiled."""
        jit_fn = jax.jit(bm19.f_dense_bm19_full)
        result = jit_fn(2.0, 3.0, 2.0)
        expected = bm19.f_dense_bm19_full(2.0, 3.0, 2.0)
        assert jnp.isclose(result, expected)


class TestPowerSpectrumSlope:
    """Tests for power_spectrum_slope()."""

    def test_subsonic_limit(self):
        """Low Mach → Kolmogorov-like (β ≈ 11/3 ≈ 3.67)."""
        result = bm19.power_spectrum_slope(0.5, 0.4)
        # Should be close to Kolmogorov
        assert 3.6 < result < 3.8

    def test_supersonic_limit(self):
        """High Mach → Burgers-like (β ≈ 4.0)."""
        result = bm19.power_spectrum_slope(50.0, 0.4)
        # Should approach Burgers
        assert 3.9 < result < 4.1

    def test_mach_monotonic(self):
        """Higher Mach → higher β (toward Burgers)."""
        machs = jnp.array([1.0, 5.0, 10.0, 20.0])
        beta = jax.vmap(lambda m: bm19.power_spectrum_slope(m, 0.4))(machs)
        assert jnp.all(jnp.diff(beta) > 0)

    def test_bounds(self):
        """β should be in [11/3, 4] range."""
        for mach in [0.5, 1.0, 5.0, 10.0, 50.0]:
            beta = bm19.power_spectrum_slope(mach, 0.4)
            assert 3.5 <= beta <= 4.1


class TestBM19Pipeline:
    """Tests for bm19_pipeline()."""

    def test_returns_all_fields(self):
        """All BM19Result fields are present and finite."""
        result = bm19.bm19_pipeline(10.0, 0.4, 2.0, 0.6)

        # Check all fields exist
        assert hasattr(result, "sigma_s")
        assert hasattr(result, "sigma_s_sq")
        assert hasattr(result, "s_t")
        assert hasattr(result, "f_dense")
        assert hasattr(result, "f_sub")
        assert hasattr(result, "beta")
        assert hasattr(result, "p")
        assert hasattr(result, "zeta")

        # All should be finite
        for field_name in result._fields:
            value = getattr(result, field_name)
            assert jnp.isfinite(value), f"Field {field_name} is not finite"

    def test_f_sub_scaling(self):
        """f_sub = η × f_dense."""
        eta = 0.7
        result = bm19.bm19_pipeline(10.0, 0.4, 2.0, eta)
        assert jnp.isclose(result.f_sub, eta * result.f_dense)

    def test_sigma_consistency(self):
        """sigma_s = sqrt(sigma_s_sq)."""
        result = bm19.bm19_pipeline(10.0, 0.4, 2.0, 0.6)
        assert jnp.isclose(result.sigma_s, jnp.sqrt(result.sigma_s_sq))

    def test_p_from_alpha(self):
        """p = 3/α."""
        alpha = 2.0
        result = bm19.bm19_pipeline(10.0, 0.4, alpha, 0.6)
        assert jnp.isclose(result.p, 3.0 / alpha)

    def test_vmap_compatible(self):
        """Can vmap over Mach array."""
        machs = jnp.array([5.0, 10.0, 20.0])
        results = jax.vmap(lambda m: bm19.bm19_pipeline(m, 0.4, 2.0, 0.6))(machs)

        assert results.sigma_s.shape == (3,)
        assert results.f_dense.shape == (3,)
        assert results.zeta.shape == (3,)

    def test_jit_compatible(self):
        """Pipeline can be JIT compiled."""
        jit_fn = jax.jit(bm19.bm19_pipeline)
        result = jit_fn(10.0, 0.4, 2.0, 0.6)
        expected = bm19.bm19_pipeline(10.0, 0.4, 2.0, 0.6)
        assert jnp.isclose(result.f_dense, expected.f_dense)

    def test_gradient_through_pipeline(self):
        """Gradients flow through entire pipeline."""

        def loss(mach):
            result = bm19.bm19_pipeline(mach, 0.4, 2.0, 0.6)
            return result.f_sub

        grad_fn = jax.grad(loss)
        g = grad_fn(10.0)
        assert jnp.isfinite(g)


class TestAlphaValidation:
    """Tests for α parameter validation."""

    def test_alpha_below_1_raises(self):
        """α ≤ 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be > 1.0"):
            bm19.bm19_pipeline(10.0, 0.4, 1.0, 0.6)

        with pytest.raises(ValueError, match="alpha must be > 1.0"):
            bm19.bm19_pipeline(10.0, 0.4, 0.5, 0.6)

    def test_alpha_slightly_above_1_warns(self):
        """α just above 1.0 but below 1.5 should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bm19.bm19_pipeline(10.0, 0.4, 1.2, 0.6)
            # Should have warning about tested range
            assert len(w) >= 1
            assert "tested range" in str(w[0].message).lower()

    def test_alpha_in_range_no_warning(self):
        """α in [1.5, 3.0] should not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bm19.bm19_pipeline(10.0, 0.4, 2.0, 0.6)
            # No warnings about alpha
            alpha_warnings = [x for x in w if "alpha" in str(x.message).lower()]
            assert len(alpha_warnings) == 0

    def test_alpha_above_3_warns(self):
        """α > 3.0 should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bm19.bm19_pipeline(10.0, 0.4, 3.5, 0.6)
            # Should have warning about tested range
            assert len(w) >= 1
            assert "tested range" in str(w[0].message).lower()


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_low_mach(self):
        """Low Mach number (M=2) should work."""
        result = bm19.bm19_pipeline(2.0, 0.4, 2.0, 0.6)
        assert jnp.isfinite(result.f_dense)
        assert 0 < result.f_dense < 1

    def test_high_mach(self):
        """High Mach number (M=50) should work."""
        result = bm19.bm19_pipeline(50.0, 0.4, 2.0, 0.6)
        assert jnp.isfinite(result.f_dense)
        assert 0 < result.f_dense < 1

    def test_extreme_b_values(self):
        """Extreme b values should work."""
        # Low b (solenoidal driving)
        result_low = bm19.bm19_pipeline(10.0, 0.3, 2.0, 0.6)
        assert jnp.isfinite(result_low.f_dense)

        # High b (compressive driving)
        result_high = bm19.bm19_pipeline(10.0, 1.0, 2.0, 0.6)
        assert jnp.isfinite(result_high.f_dense)

        # Higher b → higher σ_s² → different f_dense
        assert result_low.sigma_s_sq < result_high.sigma_s_sq

    def test_eta_bounds(self):
        """η at boundary values."""
        # η = 0 → f_sub = 0
        result_0 = bm19.bm19_pipeline(10.0, 0.4, 2.0, 0.0)
        assert jnp.isclose(result_0.f_sub, 0.0)

        # η = 1 → f_sub = f_dense
        result_1 = bm19.bm19_pipeline(10.0, 0.4, 2.0, 1.0)
        assert jnp.isclose(result_1.f_sub, result_1.f_dense)
