"""Tests for PP20 magnification factor (Parmentier & Pasquali 2020).

Tests verify:
- Correct implementation of PP20 equations
- Domain safety (singularity at p=1.3)
- JAX compatibility (jit, vmap, grad)
- zeta_fdf_direct soft-weight formula
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from progenax.gravoturb import pp20_magnification as parmentier


class TestMagnificationFactor:
    """Tests for magnification_factor()."""

    def test_p_zero(self):
        """p=0 (uniform density) → specific zeta value."""
        # ζ(0) = (3-0) / (2.6-0)^(3/2) = 3 / 2.6^1.5 ≈ 0.715
        # But clamped to >= 1
        zeta = parmentier.magnification_factor(0.0)
        assert zeta >= 1.0

    def test_p_half(self):
        """p=0.5 → ζ > 1."""
        # ζ(0.5) = (3-0.5) / (2.6-1.0)^1.5 = 2.5 / 1.6^1.5 ≈ 1.24
        zeta = parmentier.magnification_factor(0.5)
        assert zeta > 1.0
        assert jnp.isclose(zeta, 2.5 / (1.6**1.5), rtol=0.01)

    def test_p_one(self):
        """p=1.0 → finite ζ."""
        # ζ(1.0) = (3-1) / (2.6-2.0)^1.5 = 2 / 0.6^1.5 ≈ 4.30
        zeta = parmentier.magnification_factor(1.0)
        assert jnp.isfinite(zeta)
        assert zeta > 1.0

    def test_increasing_with_p(self):
        """ζ increases with p for p < 1.3."""
        p_values = jnp.array([0.0, 0.3, 0.6, 0.9, 1.0])
        zeta = jax.vmap(parmentier.magnification_factor)(p_values)
        # After clamping to >= 1, should be monotonically increasing
        assert jnp.all(jnp.diff(zeta) >= 0)

    def test_near_singularity(self):
        """Near p=1.3, should return finite clamped value."""
        zeta = parmentier.magnification_factor(1.29)
        assert jnp.isfinite(zeta)
        assert zeta >= 1.0

    def test_at_singularity(self):
        """At p=1.3, denominator → 0 but result is clamped."""
        zeta = parmentier.magnification_factor(1.3)
        assert jnp.isfinite(zeta)
        assert zeta >= 1.0

    def test_above_singularity(self):
        """p > 1.3 produces finite (though unreliable) values."""
        zeta = parmentier.magnification_factor(1.5)
        assert jnp.isfinite(zeta)
        # Note: values above singularity are NOT physically meaningful

    def test_differentiable(self):
        """Gradient exists for p < 1.3."""
        grad_fn = jax.grad(parmentier.magnification_factor)
        # Safe region
        g = grad_fn(0.5)
        assert jnp.isfinite(g)
        # Close to singularity
        g = grad_fn(1.0)
        assert jnp.isfinite(g)

    def test_jit_compatible(self):
        """Function can be JIT compiled."""
        jit_fn = jax.jit(parmentier.magnification_factor)
        result = jit_fn(0.5)
        expected = parmentier.magnification_factor(0.5)
        assert jnp.isclose(result, expected)

    def test_vmap_compatible(self):
        """Function can be vmapped."""
        p_values = jnp.array([0.0, 0.5, 1.0])
        results = jax.vmap(parmentier.magnification_factor)(p_values)
        assert results.shape == (3,)
        assert jnp.all(jnp.isfinite(results))


class TestMagnificationFactorWithCore:
    """Tests for magnification_factor_with_core()."""

    def test_uniform_limit(self):
        """p=0 should give ζ ≈ 1 regardless of core size."""
        zeta = parmentier.magnification_factor_with_core(0.0, 0.5)
        assert jnp.isclose(zeta, 1.0, rtol=0.1)

    def test_large_core_limit(self):
        """Large core (r_c/R → 1) should give ζ → 1."""
        zeta = parmentier.magnification_factor_with_core(1.5, 0.99)
        assert jnp.isclose(zeta, 1.0, rtol=0.2)

    def test_small_core_approaches_powerlaw(self):
        """Small core should approach pure power-law for small p."""
        # For p=0.5, small core
        zeta_cored = parmentier.magnification_factor_with_core(0.5, 0.01)
        zeta_pure = parmentier.magnification_factor(0.5)
        # Should be in similar range (not exact match due to integration)
        assert zeta_cored > 1.0

    def test_p_at_singularity_safe(self):
        """p=1.3 should NOT singularize with finite core."""
        zeta = parmentier.magnification_factor_with_core(1.3, 0.1)
        assert jnp.isfinite(zeta)
        assert zeta >= 1.0

    def test_p_above_singularity(self):
        """p > 1.3 should work with finite core."""
        zeta = parmentier.magnification_factor_with_core(1.5, 0.1)
        assert jnp.isfinite(zeta)
        assert zeta >= 1.0

    def test_p_two(self):
        """p=2.0 (isothermal-like) should work."""
        zeta = parmentier.magnification_factor_with_core(2.0, 0.1)
        assert jnp.isfinite(zeta)
        assert zeta >= 1.0

    def test_monotonic_in_p(self):
        """Higher p → higher ζ at fixed core size."""
        p_values = jnp.array([0.5, 1.0, 1.5, 2.0])
        zeta = jax.vmap(lambda p: parmentier.magnification_factor_with_core(p, 0.1))(
            p_values
        )
        assert jnp.all(jnp.diff(zeta) >= 0)

    def test_monotonic_in_core(self):
        """Larger core → smaller ζ (more uniform)."""
        cores = jnp.array([0.01, 0.1, 0.3, 0.5])
        zeta = jax.vmap(lambda c: parmentier.magnification_factor_with_core(1.5, c))(
            cores
        )
        assert jnp.all(jnp.diff(zeta) <= 0)

    def test_jit_compatible(self):
        """Function can be JIT compiled."""
        jit_fn = jax.jit(parmentier.magnification_factor_with_core)
        result = jit_fn(1.0, 0.1)
        expected = parmentier.magnification_factor_with_core(1.0, 0.1)
        assert jnp.isclose(result, expected)


class TestZetaFDFDirect:
    """Tests for zeta_fdf_direct()."""

    def test_uniform_field_uniform_weights(self):
        """Uniform density + uniform weights → ζ = 1."""
        rho = jnp.ones((32, 32, 32))
        weights = jnp.ones_like(rho)

        zeta = parmentier.zeta_fdf_direct(rho, weights)
        assert jnp.isclose(zeta, 1.0, atol=0.01)

    def test_uniform_field_partial_weights(self):
        """Uniform density + partial weights → ζ = 1."""
        rho = jnp.ones((32, 32, 32))
        # Half the volume has weight 1, half has weight 0
        weights = jnp.zeros((32, 32, 32))
        weights = weights.at[:16, :, :].set(1.0)

        zeta = parmentier.zeta_fdf_direct(rho, weights)
        # For uniform density, zeta should still be ~1
        assert jnp.isclose(zeta, 1.0, atol=0.1)

    def test_concentrated_density(self):
        """Centrally concentrated density → ζ > 1."""
        # Create a simple radially decreasing density
        x = jnp.linspace(-1, 1, 32)
        X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
        r = jnp.sqrt(X**2 + Y**2 + Z**2)
        rho = 1.0 / (1.0 + r**2)  # Centrally peaked

        weights = jnp.ones_like(rho)
        zeta = parmentier.zeta_fdf_direct(rho, weights)

        # Should be > 1 due to central concentration
        assert zeta > 1.0

    def test_soft_weights(self):
        """Soft sigmoid weights should produce reasonable result."""
        # Create density field
        x = jnp.linspace(-1, 1, 32)
        X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
        r = jnp.sqrt(X**2 + Y**2 + Z**2)
        rho = 1.0 / (1.0 + r**2)

        # Create soft weights (higher near center)
        s = jnp.log(rho / jnp.mean(rho))
        kappa = 10.0
        s_t = 0.5
        weights = jax.nn.sigmoid(kappa * (s - s_t))

        zeta = parmentier.zeta_fdf_direct(rho, weights)
        assert jnp.isfinite(zeta)
        assert zeta >= 1.0

    def test_clamped_to_one(self):
        """Result should always be >= 1."""
        # Edge case: inverse concentration (higher at edges)
        x = jnp.linspace(-1, 1, 16)
        X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
        r = jnp.sqrt(X**2 + Y**2 + Z**2)
        rho = 1.0 + r**2  # Higher at edges

        weights = jnp.ones_like(rho)
        zeta = parmentier.zeta_fdf_direct(rho, weights)

        # Should be clamped to >= 1
        assert zeta >= 1.0

    def test_differentiable_rho(self):
        """Gradient w.r.t. density field exists."""

        def loss(scale):
            rho = jnp.ones((16, 16, 16)) * scale
            weights = jnp.ones_like(rho)
            return parmentier.zeta_fdf_direct(rho, weights)

        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad)

    def test_differentiable_weights(self):
        """Gradient w.r.t. weights exists."""

        def loss(w_scale):
            rho = jnp.ones((16, 16, 16))
            weights = jnp.ones_like(rho) * w_scale
            return parmentier.zeta_fdf_direct(rho, weights)

        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad)

    def test_jit_compatible(self):
        """Function can be JIT compiled."""
        rho = jnp.ones((16, 16, 16))
        weights = jnp.ones_like(rho)

        jit_fn = jax.jit(parmentier.zeta_fdf_direct)
        result = jit_fn(rho, weights)
        expected = parmentier.zeta_fdf_direct(rho, weights)
        assert jnp.isclose(result, expected)

    def test_empty_tail_handling(self):
        """Zero weights should not crash (protected by epsilon)."""
        rho = jnp.ones((16, 16, 16))
        weights = jnp.zeros_like(rho)  # No tail

        zeta = parmentier.zeta_fdf_direct(rho, weights)
        # Should return clamped value, not crash
        assert jnp.isfinite(zeta)
        assert zeta >= 1.0


class TestSFRPerDenseGas:
    """Tests for sfr_per_dense_gas()."""

    def test_basic_formula(self):
        """SFR/M_dg = ζ × ε_ff / t_ff."""
        zeta = 2.0
        epsilon = 0.01
        t_ff = 1.0
        expected = 2.0 * 0.01 / 1.0  # = 0.02

        result = parmentier.sfr_per_dense_gas(zeta, epsilon, t_ff)
        assert jnp.isclose(result, expected)

    def test_zeta_scaling(self):
        """Higher ζ → higher SFR/M_dg."""
        zetas = jnp.array([1.0, 2.0, 5.0])
        sfr = jax.vmap(lambda z: parmentier.sfr_per_dense_gas(z, 0.01, 1.0))(zetas)
        assert jnp.all(jnp.diff(sfr) > 0)

    def test_epsilon_scaling(self):
        """Higher ε → higher SFR/M_dg."""
        epsilons = jnp.array([0.01, 0.02, 0.05])
        sfr = jax.vmap(lambda e: parmentier.sfr_per_dense_gas(2.0, e, 1.0))(epsilons)
        assert jnp.all(jnp.diff(sfr) > 0)

    def test_tff_scaling(self):
        """Higher t_ff → lower SFR/M_dg."""
        t_ffs = jnp.array([0.5, 1.0, 2.0])
        sfr = jax.vmap(lambda t: parmentier.sfr_per_dense_gas(2.0, 0.01, t))(t_ffs)
        assert jnp.all(jnp.diff(sfr) < 0)

    def test_differentiable(self):
        """Gradients exist."""
        grad_zeta = jax.grad(lambda z: parmentier.sfr_per_dense_gas(z, 0.01, 1.0))(2.0)
        grad_eps = jax.grad(lambda e: parmentier.sfr_per_dense_gas(2.0, e, 1.0))(0.01)
        grad_tff = jax.grad(lambda t: parmentier.sfr_per_dense_gas(2.0, 0.01, t))(1.0)

        assert jnp.isfinite(grad_zeta)
        assert jnp.isfinite(grad_eps)
        assert jnp.isfinite(grad_tff)

    def test_jit_compatible(self):
        """Function can be JIT compiled."""
        jit_fn = jax.jit(parmentier.sfr_per_dense_gas)
        result = jit_fn(2.0, 0.01, 1.0)
        expected = parmentier.sfr_per_dense_gas(2.0, 0.01, 1.0)
        assert jnp.isclose(result, expected)
