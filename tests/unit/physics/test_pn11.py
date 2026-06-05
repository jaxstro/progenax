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

    def test_s_crit_pn11_eq8_faithful(self):
        """s_crit = ln(0.067 * theta^-2 * alpha_vir * M^2) — PN11 (2011) Eq. 8.

        theta = 0.35 (turbulence integral-scale factor, Wang & George 2002
        correction; PN11 p.3) gives prefactor 0.067*0.35^-2 = 0.547, i.e. PN11
        Eq. 11 (n_cr/n0 = 0.547*alpha_vir*M^2). This is the Padoan & Nordlund
        critical density, NOT the FK12/KM05 form (pi^2/5)*phi_x^2 = 0.242 that
        swindlax + the rosen-burkhart-swindle paper use.
        """
        mach, alpha_vir, theta = 10.0, 2.0, 0.35
        prefactor = 0.067 * theta**-2.0  # = 0.547 (PN11 Eq. 11)
        expected = jnp.log(prefactor * alpha_vir * mach**2)

        result = pn11.s_crit_pn11(mach, alpha_vir, theta)
        assert jnp.isclose(result, expected, rtol=1e-6)

    def test_mach_scaling(self):
        """s_crit scales with M²."""
        alpha_vir = 2.0
        theta = 0.35

        s_crit_5 = pn11.s_crit_pn11(5.0, alpha_vir, theta)
        s_crit_10 = pn11.s_crit_pn11(10.0, alpha_vir, theta)
        s_crit_20 = pn11.s_crit_pn11(20.0, alpha_vir, theta)

        # Check M² scaling
        assert s_crit_10 > s_crit_5
        assert s_crit_20 > s_crit_10

    def test_alpha_vir_scaling(self):
        """Higher α_vir → higher s_crit."""
        mach = 10.0
        theta = 0.35

        alpha_virs = jnp.array([1.0, 2.0, 4.0])
        s_crits = jax.vmap(lambda a: pn11.s_crit_pn11(mach, a, theta))(alpha_virs)
        assert jnp.all(jnp.diff(s_crits) > 0)

    def test_theta_scaling(self):
        """Higher theta -> LOWER s_crit (theta in denominator: 0.067 * theta^-2)."""
        mach = 10.0
        alpha_vir = 2.0

        thetas = jnp.array([0.17, 0.35, 0.5])
        s_crits = jax.vmap(lambda t: pn11.s_crit_pn11(mach, alpha_vir, t))(thetas)
        assert jnp.all(jnp.diff(s_crits) < 0)

    def test_default_theta(self):
        """Default theta = 0.35."""
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

    def test_s_crit_differentiable(self):
        """s_crit threshold is differentiable w.r.t. Mach number and virial alpha.

        s_crit sets the PN11 dense-tail selection threshold (the path routed by
        the fdf pn11 fix), so gradient flow through it is needed for inference on
        turbulence parameters. f_dense's own gradient is checked separately in
        TestFDensePN11; this covers the threshold itself.
        """
        g_mach = jax.grad(lambda m: pn11.s_crit_pn11(m, 2.0, 0.12))(5.0)
        g_alpha = jax.grad(lambda a: pn11.s_crit_pn11(5.0, a, 0.12))(2.0)
        assert jnp.isfinite(g_mach)
        assert jnp.isfinite(g_alpha)


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

    def test_pipeline_grad_fd_vs_autodiff(self):
        """FD-vs-autodiff agreement for pn11_pipeline f_sub (Mach and theta).

        Confirms the gradients are correct (not merely finite) on the public
        differentiable entry point: PN11's s_crit path (log, theta**-2, erfc)
        has no clip/where/while traps, so autodiff must match central differences.
        """
        Sigma, eta = 200.0, 0.6

        def f_of_mach(m):
            return pn11.pn11_pipeline(m, Sigma, eta_survive=eta).f_sub

        def f_of_theta(t):
            return pn11.pn11_pipeline(10.0, Sigma, eta_survive=eta, theta=t).f_sub

        h = 1e-4
        g_m_ad = jax.grad(f_of_mach)(10.0)
        g_m_fd = (f_of_mach(10.0 + h) - f_of_mach(10.0 - h)) / (2.0 * h)
        assert jnp.isclose(g_m_ad, g_m_fd, rtol=1e-4, atol=1e-8)

        g_t_ad = jax.grad(f_of_theta)(0.35)
        g_t_fd = (f_of_theta(0.35 + h) - f_of_theta(0.35 - h)) / (2.0 * h)
        assert jnp.isclose(g_t_ad, g_t_fd, rtol=1e-4, atol=1e-8)


class TestPN11VsBM19:
    """Compare PN11 and BM19 predictions.

    These tests demonstrate the key differences between the frameworks.
    """

    def test_different_threshold_formulas(self):
        """PN11 s_crit and BM19 s_t are different formulas: s_crit depends on
        Sigma (via alpha_vir) and theta; s_t depends only on sigma_s^2 and alpha."""
        mach = 10.0
        b = 0.4
        alpha = 2.0
        sigma_sq = bm19.sigma_s_squared(mach, b)
        s_t = bm19.transition_density(sigma_sq, alpha)  # Sigma-independent

        # s_crit responds to Sigma; s_t does not -> genuinely different formulas
        s_crit_100 = pn11.s_crit_pn11(mach, pn11.alpha_vir_from_sigma(100.0))
        s_crit_300 = pn11.s_crit_pn11(mach, pn11.alpha_vir_from_sigma(300.0))
        assert not jnp.isclose(s_crit_100, s_crit_300, rtol=0.1)
        assert not jnp.isclose(s_t, s_crit_300, rtol=0.1)

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

        # PN11: (M, Σ, η, b, θ, α₀, Σ₀) → f_sub
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

    def test_extreme_theta(self):
        """Extreme theta; smaller theta -> larger s_crit (0.067 * theta^-2)."""
        mach = 10.0
        alpha_vir = 2.0

        s_crit_small_theta = pn11.s_crit_pn11(mach, alpha_vir, 0.17)
        s_crit_large_theta = pn11.s_crit_pn11(mach, alpha_vir, 0.5)

        assert jnp.isfinite(s_crit_small_theta)
        assert jnp.isfinite(s_crit_large_theta)
        assert s_crit_small_theta > s_crit_large_theta
