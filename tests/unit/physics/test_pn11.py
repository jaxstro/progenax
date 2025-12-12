"""Tests for PN11 gravoturbulent framework.

This module tests the PN11 (Padoan-Nordlund 2011) model as an alternative
to the BM19 model. Tests also compare PN11 vs BM19 to demonstrate differences.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from progenax.gravoturb import bm19_model as bm19
from progenax.gravoturb import pn11_model as pn11


class TestSCritPN11:
    """Tests for s_crit_pn11()."""

    def test_basic_formula(self):
        """s_crit = ln((π²φ_x²/5) × α_vir × M²)."""
        mach = 10.0
        alpha_vir = 2.0
        phi_x = 0.35
        prefactor = jnp.pi**2 * phi_x**2 / 5.0  # ≈ 0.242
        expected = jnp.log(prefactor * alpha_vir * mach**2)

        result = pn11.s_crit_pn11(mach, alpha_vir, phi_x)
        assert jnp.isclose(result, expected, rtol=1e-6)

    def test_mach_scaling(self):
        """s_crit scales with M²."""
        alpha_vir = 2.0
        phi_x = 0.35

        s_crit_5 = pn11.s_crit_pn11(5.0, alpha_vir, phi_x)
        s_crit_10 = pn11.s_crit_pn11(10.0, alpha_vir, phi_x)
        s_crit_20 = pn11.s_crit_pn11(20.0, alpha_vir, phi_x)

        # Check M² scaling
        assert s_crit_10 > s_crit_5
        assert s_crit_20 > s_crit_10

    def test_alpha_vir_scaling(self):
        """Higher α_vir → higher s_crit."""
        mach = 10.0
        phi_x = 0.35

        alpha_virs = jnp.array([1.0, 2.0, 4.0])
        s_crits = jax.vmap(lambda a: pn11.s_crit_pn11(mach, a, phi_x))(alpha_virs)
        assert jnp.all(jnp.diff(s_crits) > 0)

    def test_phi_x_scaling(self):
        """Higher φ_x → higher s_crit."""
        mach = 10.0
        alpha_vir = 2.0

        phi_xs = jnp.array([0.17, 0.35, 0.5])
        s_crits = jax.vmap(lambda p: pn11.s_crit_pn11(mach, alpha_vir, p))(phi_xs)
        assert jnp.all(jnp.diff(s_crits) > 0)

    def test_default_phi_x(self):
        """Default φ_x = 0.35."""
        mach = 10.0
        alpha_vir = 2.0

        result_default = pn11.s_crit_pn11(mach, alpha_vir)
        result_explicit = pn11.s_crit_pn11(mach, alpha_vir, 0.35)
        assert jnp.isclose(result_default, result_explicit)

    def test_jit_compatible(self):
        """Function can be JIT compiled."""
        jit_fn = jax.jit(pn11.s_crit_pn11)
        result = jit_fn(10.0, 2.0, 0.35)
        expected = pn11.s_crit_pn11(10.0, 2.0, 0.35)
        assert jnp.isclose(result, expected)


class TestFDensePN11:
    """Tests for f_dense_pn11()."""

    def test_bounds(self):
        """f_dense should be in (0, 1)."""
        sigma_s_sq = 2.0
        for s_crit in [1.0, 2.0, 3.0, 4.0]:
            result = pn11.f_dense_pn11(sigma_s_sq, s_crit)
            assert 0 < result < 1

    def test_s_crit_monotonic(self):
        """Higher s_crit → lower f_dense."""
        sigma_s_sq = 2.0
        s_crits = jnp.array([1.0, 2.0, 3.0, 4.0])
        f_dense = jax.vmap(lambda s: pn11.f_dense_pn11(sigma_s_sq, s))(s_crits)
        assert jnp.all(jnp.diff(f_dense) < 0)

    def test_sigma_sq_effect(self):
        """Higher σ_s² at fixed s_crit changes f_dense."""
        s_crit = 3.0
        sigma_sqs = jnp.array([1.0, 2.0, 3.0])
        f_dense = jax.vmap(lambda s: pn11.f_dense_pn11(s, s_crit))(sigma_sqs)
        # All should be valid
        assert jnp.all((f_dense > 0) & (f_dense < 1))

    def test_matches_bm19_lognormal_limit(self):
        """Should match BM19 lognormal limit for same inputs."""
        sigma_s_sq = 2.0
        s_crit = 3.0

        pn11_result = pn11.f_dense_pn11(sigma_s_sq, s_crit)
        bm19_result = bm19.f_dense_lognormal_limit(sigma_s_sq, s_crit)

        # These use the same erfc formula, should be identical
        assert jnp.isclose(pn11_result, bm19_result, rtol=1e-6)

    def test_differentiable(self):
        """Gradients exist."""
        grad_sigma = jax.grad(lambda s: pn11.f_dense_pn11(s, 3.0))(2.0)
        grad_s_crit = jax.grad(lambda s: pn11.f_dense_pn11(2.0, s))(3.0)
        assert jnp.isfinite(grad_sigma)
        assert jnp.isfinite(grad_s_crit)

    def test_jit_compatible(self):
        """Function can be JIT compiled."""
        jit_fn = jax.jit(pn11.f_dense_pn11)
        result = jit_fn(2.0, 3.0)
        expected = pn11.f_dense_pn11(2.0, 3.0)
        assert jnp.isclose(result, expected)


class TestAlphaVirFromSigma:
    """Tests for alpha_vir_from_sigma()."""

    def test_basic_formula(self):
        """α_vir = α₀ × (Σ₀/Σ)."""
        Sigma = 100.0
        alpha_0 = 2.0
        Sigma_0 = 85.0
        expected = alpha_0 * (Sigma_0 / Sigma)

        result = pn11.alpha_vir_from_sigma(Sigma, alpha_0, Sigma_0)
        assert jnp.isclose(result, expected)

    def test_inverse_scaling(self):
        """Higher Σ → lower α_vir."""
        Sigmas = jnp.array([50.0, 100.0, 200.0])
        alpha_virs = jax.vmap(lambda s: pn11.alpha_vir_from_sigma(s, 2.0, 85.0))(
            Sigmas
        )
        assert jnp.all(jnp.diff(alpha_virs) < 0)

    def test_reference_values(self):
        """At Σ = Σ₀, α_vir = α₀."""
        alpha_0 = 2.0
        Sigma_0 = 85.0
        result = pn11.alpha_vir_from_sigma(Sigma_0, alpha_0, Sigma_0)
        assert jnp.isclose(result, alpha_0)

    def test_defaults(self):
        """Default α₀ = 2.0, Σ₀ = 85.0."""
        Sigma = 85.0
        result = pn11.alpha_vir_from_sigma(Sigma)
        assert jnp.isclose(result, 2.0)  # At reference, α_vir = α₀

    def test_jit_compatible(self):
        """Function can be JIT compiled."""
        jit_fn = jax.jit(pn11.alpha_vir_from_sigma)
        result = jit_fn(100.0, 2.0, 85.0)
        expected = pn11.alpha_vir_from_sigma(100.0, 2.0, 85.0)
        assert jnp.isclose(result, expected)


class TestPN11Pipeline:
    """Tests for pn11_pipeline()."""

    def test_returns_pn11result(self):
        """Pipeline returns PN11Result namedtuple."""
        result = pn11.pn11_pipeline(10.0, 100.0, 0.6, 0.4, 0.35, 2.0, 85.0)

        assert hasattr(result, "sigma_s")
        assert hasattr(result, "sigma_s_sq")
        assert hasattr(result, "alpha_vir")
        assert hasattr(result, "s_crit")
        assert hasattr(result, "f_dense")
        assert hasattr(result, "f_sub")

    def test_all_values_finite(self):
        """All returned values are finite."""
        result = pn11.pn11_pipeline(10.0, 100.0, 0.6, 0.4, 0.35, 2.0, 85.0)

        assert jnp.isfinite(result.sigma_s)
        assert jnp.isfinite(result.sigma_s_sq)
        assert jnp.isfinite(result.alpha_vir)
        assert jnp.isfinite(result.s_crit)
        assert jnp.isfinite(result.f_dense)
        assert jnp.isfinite(result.f_sub)

    def test_f_sub_scaling(self):
        """f_sub = η × f_dense."""
        eta = 0.7
        result = pn11.pn11_pipeline(10.0, 100.0, eta, 0.4, 0.35, 2.0, 85.0)
        assert jnp.isclose(result.f_sub, eta * result.f_dense)

    def test_sigma_sq_consistency(self):
        """σ_s² matches direct calculation."""
        mach = 10.0
        b = 0.4
        result = pn11.pn11_pipeline(mach, 100.0, 0.6, b, 0.35, 2.0, 85.0)
        expected = jnp.log(1.0 + b**2 * mach**2)
        assert jnp.isclose(result.sigma_s_sq, expected)

    def test_sigma_s_consistency(self):
        """σ_s = √(σ_s²)."""
        result = pn11.pn11_pipeline(10.0, 100.0, 0.6, 0.4, 0.35, 2.0, 85.0)
        assert jnp.isclose(result.sigma_s, jnp.sqrt(result.sigma_s_sq))

    def test_jit_compatible(self):
        """Pipeline can be JIT compiled."""
        jit_fn = jax.jit(pn11.pn11_pipeline)
        result = jit_fn(10.0, 100.0, 0.6, 0.4, 0.35, 2.0, 85.0)
        expected = pn11.pn11_pipeline(10.0, 100.0, 0.6, 0.4, 0.35, 2.0, 85.0)
        assert jnp.isclose(result.f_dense, expected.f_dense)


class TestPN11VsBM19:
    """Compare PN11 and BM19 predictions.

    These tests demonstrate the key differences between the frameworks.
    """

    def test_different_threshold_formulas(self):
        """PN11 s_crit and BM19 s_t use different formulas."""
        mach = 10.0
        b = 0.4
        sigma_sq = bm19.sigma_s_squared(mach, b)

        # BM19 s_t depends only on σ_s² and α
        alpha = 2.0
        s_t = bm19.transition_density(sigma_sq, alpha)

        # PN11 s_crit depends on α_vir, φ_x, M
        Sigma = 100.0
        alpha_vir = pn11.alpha_vir_from_sigma(Sigma)
        s_crit = pn11.s_crit_pn11(mach, alpha_vir, 0.35)

        # They should be DIFFERENT (different formulas)
        assert not jnp.isclose(s_t, s_crit, rtol=0.1)

    def test_different_f_dense(self):
        """PN11 and BM19 give different f_dense."""
        mach = 10.0
        b = 0.4
        sigma_sq = bm19.sigma_s_squared(mach, b)

        # BM19 full integral
        alpha = 2.0
        s_t = bm19.transition_density(sigma_sq, alpha)
        f_bm19 = bm19.f_dense_bm19_full(sigma_sq, s_t, alpha)

        # PN11 pure lognormal
        Sigma = 100.0
        alpha_vir = pn11.alpha_vir_from_sigma(Sigma)
        s_crit = pn11.s_crit_pn11(mach, alpha_vir, 0.35)
        f_pn11 = pn11.f_dense_pn11(sigma_sq, s_crit)

        # They should differ (this is the whole point!)
        assert not jnp.isclose(f_bm19, f_pn11, rtol=0.01)

    def test_bm19_has_fewer_params(self):
        """BM19 requires fewer free parameters than PN11."""
        # BM19: (M, b, α, η) → f_sub
        bm19_result = bm19.bm19_pipeline(10.0, 0.4, 2.0, 0.6)

        # PN11: (M, Σ, η, b, φ_x, α₀, Σ₀) → f_sub
        pn11_result = pn11.pn11_pipeline(10.0, 100.0, 0.6, 0.4, 0.35, 2.0, 85.0)

        # Both produce valid f_sub
        assert 0 < bm19_result.f_sub < 1
        assert 0 < pn11_result.f_sub < 1

    def test_same_lognormal_formula(self):
        """PN11 and BM19 use same lognormal formula when comparing directly."""
        sigma_sq = 2.0
        threshold = 3.0

        # Same threshold value → same lognormal result
        f_pn11 = pn11.f_dense_pn11(sigma_sq, threshold)
        f_bm19_ln = bm19.f_dense_lognormal_limit(sigma_sq, threshold)

        # Should be identical
        assert jnp.isclose(f_pn11, f_bm19_ln, rtol=1e-6)

    def test_bm19_full_differs_from_lognormal(self):
        """BM19 full integral differs from pure lognormal."""
        mach = 10.0
        alpha = 2.0
        sigma_sq = bm19.sigma_s_squared(mach, 0.4)
        s_t = bm19.transition_density(sigma_sq, alpha)

        f_full = bm19.f_dense_bm19_full(sigma_sq, s_t, alpha)
        f_ln = bm19.f_dense_lognormal_limit(sigma_sq, s_t)

        # Full integral includes powerlaw tail contribution
        # Should differ for finite α
        assert not jnp.isclose(f_full, f_ln, rtol=0.01)


class TestEdgeCases:
    """Edge cases for PN11 functions."""

    def test_high_sigma(self):
        """High surface density gives low α_vir."""
        Sigma = 500.0  # High surface density
        alpha_vir = pn11.alpha_vir_from_sigma(Sigma)
        assert alpha_vir < 1.0  # Gravitationally bound

    def test_low_sigma(self):
        """Low surface density gives high α_vir."""
        Sigma = 20.0  # Low surface density
        alpha_vir = pn11.alpha_vir_from_sigma(Sigma)
        assert alpha_vir > 4.0  # Unbound

    def test_extreme_phi_x(self):
        """Extreme φ_x values."""
        mach = 10.0
        alpha_vir = 2.0

        # Strong magnetic support
        s_crit_low = pn11.s_crit_pn11(mach, alpha_vir, 0.17)
        # Weak magnetic support
        s_crit_high = pn11.s_crit_pn11(mach, alpha_vir, 0.5)

        assert jnp.isfinite(s_crit_low)
        assert jnp.isfinite(s_crit_high)
        assert s_crit_high > s_crit_low
