"""Tests for PP20 magnification factor (Parmentier & Pasquali 2020).

Tests verify:
- Correct implementation of PP20 Eq. 6 across the full physical domain
- p → 2 divergence and P_MAX numerical-safety clipping
- JAX compatibility (jit, vmap, grad)
- zeta_fdf_direct soft-weight formula

Anchored on:
- Parmentier & Pasquali 2020, ApJ 903, 56 (arXiv:2009.10652) Eq. 6
- The integral derivation ζ = ∫ρ^(3/2) dV / (M·√⟨ρ⟩) for ρ(r) ∝ r^(-p)

Note: ζ(p) = (3-p)^(3/2) / [2.6 · (2-p)] (PP20 Eq. 6) is well-behaved over
the full physical 0 ≤ p < 2 domain. The only singularity is at p = 2
(singular isothermal collapse). Earlier versions of this file enforced
values from a transcription bug — see test_pp20_zeta_canonical.py for the
regression trap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from progenax.gravoturb import pp20_magnification as parmentier


# Helper: analytic form derived from the integral definition. Identical to
# PP20 Eq. 6 modulo the substitution 2.6 → 3^(3/2)/2 = 2.598.
def _zeta_analytic(p: float) -> float:
    return 2.0 * (3.0 - p) ** 1.5 / (3.0**1.5 * (2.0 - p))


class TestMagnificationFactor:
    """Tests for magnification_factor()."""

    def test_p_zero_unity(self):
        """p=0 (uniform density): ζ(0) = 1 exactly (top-hat is its own SFR
        reference, by construction of the integral definition)."""
        zeta = parmentier.magnification_factor(0.0)
        assert jnp.isclose(zeta, 1.0, atol=1e-10)

    def test_p_half_pp20_value(self):
        """p=0.5 → ζ ≈ 1.014 (PP20 Eq. 6 / analytic).

        The buggy form (3-p)/(2.6-2p)^(3/2) would give 2.5/1.6^1.5 ≈ 1.235;
        the correct PP20 Eq. 6 form (3-p)^(3/2)/[2.6(2-p)] gives ≈ 1.014.
        """
        zeta = parmentier.magnification_factor(0.5)
        assert zeta > 1.0
        assert jnp.isclose(zeta, _zeta_analytic(0.5), rtol=1e-10)
        assert jnp.isclose(zeta, 1.014, atol=2e-3)

    def test_p_one_exact_analytic(self):
        """p=1.0 → ζ = 2·2^(3/2)/3^(3/2) ≈ 1.0887 (exact analytic value).

        The buggy form (3-p)/(2.6-2p)^(3/2) would give 2/0.6^1.5 ≈ 4.30; the
        correct value is ~1.09.
        """
        zeta = parmentier.magnification_factor(1.0)
        expected = 2.0 * 2.0**1.5 / 3.0**1.5
        assert jnp.isclose(zeta, expected, atol=1e-10)
        assert jnp.isclose(zeta, 1.0887, atol=1e-3)

    def test_p_three_halves_is_sqrt_two(self):
        """ζ(3/2) = √2 exactly (analytic value at p = 3/2).

        Earlier this test asserted ζ(1.5) was an "unreliable" value above a
        spurious p=1.3 domain limit. PP20 Eq. 6 has no such limit; ζ(1.5)
        is exactly √2.
        """
        zeta = parmentier.magnification_factor(1.5)
        assert jnp.isclose(zeta, 2.0**0.5, atol=1e-10)

    def test_p_kainulainen_median(self):
        """ζ(p=1.67) ≈ 1.79 — Kainulainen+2014 median observational p.

        PP20 cites this on page 5 as the typical magnification factor for
        resolved Galactic cloud samples.
        """
        zeta = parmentier.magnification_factor(1.67)
        assert jnp.isclose(zeta, 1.79, atol=0.02)

    def test_increasing_with_p_full_domain(self):
        """ζ is strictly increasing across the full physical 0 ≤ p < 2
        domain (no singularity at p=1.3 or anywhere else inside the domain).
        """
        p_values = jnp.linspace(0.0, 1.9, 20)
        zeta = jax.vmap(parmentier.magnification_factor)(p_values)
        assert jnp.all(jnp.diff(zeta) > 0)

    def test_no_spurious_singularity_at_p_one_three(self):
        """ζ(p) is smooth and finite across p=1.3 — there is no singularity
        there. Earlier versions of this code had a transcription bug that
        produced an artefactual divergence at p=1.3, which this test traps.
        """
        z_below = parmentier.magnification_factor(1.29)
        z_at = parmentier.magnification_factor(1.30)
        z_above = parmentier.magnification_factor(1.31)
        assert jnp.isfinite(z_below) and z_below > 1.0
        assert jnp.isfinite(z_at) and z_at > 1.0
        assert jnp.isfinite(z_above) and z_above > 1.0
        # Smooth: consecutive values differ by < 1%
        assert abs(z_at - z_below) / z_at < 0.01
        assert abs(z_above - z_at) / z_at < 0.01

    def test_diverges_as_p_to_two(self):
        """ζ(p) → ∞ as p → 2 (singular isothermal collapse). The function
        clips at P_MAX < 2 for HMC/NUTS gradient safety, so ζ(p ≥ P_MAX)
        equals ζ(P_MAX) — large but finite."""
        zeta_close = parmentier.magnification_factor(1.95)
        zeta_at_one = parmentier.magnification_factor(1.0)
        # ζ(1.95) should exceed ζ(1) by an order of magnitude
        assert zeta_close > 5.0 * zeta_at_one
        assert jnp.isfinite(zeta_close)

    def test_clamped_at_P_MAX(self):
        """For p ≥ P_MAX, the function returns ζ(P_MAX) (numerical safety
        for HMC/NUTS gradients near p=2)."""
        zeta_above = parmentier.magnification_factor(2.5)
        zeta_at_pmax = parmentier.magnification_factor(parmentier.P_MAX)
        assert jnp.isclose(zeta_above, zeta_at_pmax, atol=1e-10)

    def test_differentiable(self):
        """Gradient exists across the full 0 ≤ p < 2 domain."""
        grad_fn = jax.grad(parmentier.magnification_factor)
        for p_test in [0.5, 1.0, 1.5, 1.8]:
            g = grad_fn(p_test)
            assert jnp.isfinite(g), f"grad failed at p={p_test}"
            # ζ is monotonically increasing → gradient is positive
            assert g > 0.0

    def test_jit_compatible(self):
        """Function can be JIT compiled."""
        jit_fn = jax.jit(parmentier.magnification_factor)
        result = jit_fn(0.5)
        expected = parmentier.magnification_factor(0.5)
        assert jnp.isclose(result, expected)

    def test_vmap_compatible(self):
        """Function can be vmapped."""
        p_values = jnp.array([0.0, 0.5, 1.0, 1.5, 1.8])
        results = jax.vmap(parmentier.magnification_factor)(p_values)
        assert results.shape == (5,)
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
        """Small core should approach pure power-law for moderate p."""
        zeta_cored = parmentier.magnification_factor_with_core(0.5, 0.01)
        zeta_pure = parmentier.magnification_factor(0.5)
        # Cored value should be > 1 (centrally concentrated)
        assert zeta_cored > 1.0
        # And should be in the same ballpark as the pure power-law value
        # (the cored profile rho(r) = rho_c / [1+(r/r_c)^2]^(p/2) is not
        # identical to rho ~ r^(-p) even at vanishing r_c due to the
        # softening, so we tolerate a moderate offset).
        assert zeta_cored == pytest.approx(zeta_pure, rel=0.5)

    def test_cored_safe_at_p_one_three(self):
        """The cored form is finite at p=1.3 (the buggy power-law claimed
        a singularity here, but the cored form was never affected since it
        uses numerical integration, not the analytic typo)."""
        zeta = parmentier.magnification_factor_with_core(1.3, 0.1)
        assert jnp.isfinite(zeta)
        assert zeta >= 1.0

    def test_cored_above_p_one_three(self):
        """The cored form works at p=1.5 (no spurious singularity)."""
        zeta = parmentier.magnification_factor_with_core(1.5, 0.1)
        assert jnp.isfinite(zeta)
        assert zeta >= 1.0

    def test_p_two(self):
        """p=2.0 (isothermal-like) should work with finite core."""
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
        weights = jnp.zeros((32, 32, 32))
        weights = weights.at[:16, :, :].set(1.0)

        zeta = parmentier.zeta_fdf_direct(rho, weights)
        assert jnp.isclose(zeta, 1.0, atol=0.1)

    def test_concentrated_density(self):
        """Centrally concentrated density → ζ > 1."""
        x = jnp.linspace(-1, 1, 32)
        X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
        r = jnp.sqrt(X**2 + Y**2 + Z**2)
        rho = 1.0 / (1.0 + r**2)

        weights = jnp.ones_like(rho)
        zeta = parmentier.zeta_fdf_direct(rho, weights)

        assert zeta > 1.0

    def test_soft_weights(self):
        """Soft sigmoid weights should produce reasonable result."""
        x = jnp.linspace(-1, 1, 32)
        X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
        r = jnp.sqrt(X**2 + Y**2 + Z**2)
        rho = 1.0 / (1.0 + r**2)

        s = jnp.log(rho / jnp.mean(rho))
        kappa = 10.0
        s_t = 0.5
        weights = jax.nn.sigmoid(kappa * (s - s_t))

        zeta = parmentier.zeta_fdf_direct(rho, weights)
        assert jnp.isfinite(zeta)
        assert zeta >= 1.0

    def test_clamped_to_one(self):
        """Result should always be >= 1."""
        x = jnp.linspace(-1, 1, 16)
        X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
        r = jnp.sqrt(X**2 + Y**2 + Z**2)
        rho = 1.0 + r**2

        weights = jnp.ones_like(rho)
        zeta = parmentier.zeta_fdf_direct(rho, weights)

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
        weights = jnp.zeros_like(rho)

        zeta = parmentier.zeta_fdf_direct(rho, weights)
        assert jnp.isfinite(zeta)
        assert zeta >= 1.0


class TestSFRPerDenseGas:
    """Tests for sfr_per_dense_gas()."""

    def test_basic_formula(self):
        """SFR/M_dg = ζ × ε_ff / t_ff."""
        zeta = 2.0
        epsilon = 0.01
        t_ff = 1.0
        expected = 2.0 * 0.01 / 1.0

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
