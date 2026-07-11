"""Tests for progenax.numerics — the shared trapezoid/inverse-CDF primitives.

These helpers must be BIT-IDENTICAL to the inline patterns they replace
(same op order: pairwise average -> cumsum -> concat zero), because five
Poisson passes and eight speed-CDF kernels migrate onto them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.numerics import (
    cumulative_trapz,
    inverse_cdf_draw,
    power_integral_stable,
    power_ppf_stable,
)


class TestCumulativeTrapezoid:
    def test_matches_inline_pattern_bit_identical(self):
        """Bit-identical to the majority inline pattern: cumsum of pairwise
        averages, then * dx (assert_array_equal, not just close)."""
        y = jnp.sin(jnp.linspace(0.0, 3.0, 257)) + 1.1
        dr = 3.0 / 256
        inline = jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(0.5 * (y[1:] + y[:-1])) * dr]
        )
        ours = cumulative_trapz(y, dx=dr)
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(inline))

    def test_linear_function_exact(self):
        """Trapezoid is exact for linear integrands: int_0^x t dt = x^2/2."""
        x = jnp.linspace(0.0, 2.0, 101)
        out = cumulative_trapz(x, dx=float(x[1] - x[0]))
        np.testing.assert_allclose(
            np.asarray(out), np.asarray(x**2 / 2), rtol=0, atol=1e-14
        )

    def test_axis_minus_one_on_2d_rows(self):
        """Row-wise (the SpeedCDFTable / multicomponent axis=1 pattern)."""
        y = jnp.arange(12.0).reshape(3, 4) + 1.0
        dr = 0.5
        inline = jnp.concatenate(
            [jnp.zeros((3, 1)), jnp.cumsum(0.5 * (y[:, 1:] + y[:, :-1]), axis=1) * dr],
            axis=1,
        )
        ours = cumulative_trapz(y, dx=dr, axis=-1)
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(inline))

    def test_differentiable(self):
        g = jax.grad(lambda a: cumulative_trapz(a * jnp.ones(8), dx=0.1)[-1])(2.0)
        assert jnp.isfinite(g) and g > 0


class TestInverseCdfDraw:
    def test_matches_king_sample_unit_speed_pattern(self):
        """Reproduces the king_df._sample_unit_speed CDF+interp chain exactly."""
        W = 4.0
        n_u = 256
        u_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * W), n_u)
        wgt = jnp.maximum(u_grid**2 * (jnp.exp(W - u_grid**2 / 2.0) - 1.0), 0.0)
        du = u_grid[1] - u_grid[0]
        cdf = jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(0.5 * (wgt[1:] + wgt[:-1])) * du]
        )
        cdf = cdf / (cdf[-1] + 1e-30)
        unif = jnp.asarray(0.37)
        expected = jnp.interp(unif, cdf, u_grid)
        ours = inverse_cdf_draw(wgt, u_grid, unif)
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(expected))

    def test_uniform_weight_is_identity(self):
        """Flat weight => draw is linear in unif (CDF of uniform)."""
        grid = jnp.linspace(0.0, 1.0, 200)
        out = inverse_cdf_draw(jnp.ones(200), grid, jnp.asarray(0.25))
        np.testing.assert_allclose(float(out), 0.25, atol=1e-3)

    def test_zero_weight_draws_grid_end_callers_must_guard(self):
        """All-zero weight => interp against a zero CDF clamps to grid[-1].
        This pins the documented behavior: the +reg guard prevents NaN but
        does NOT make the draw physical — callers keep their W-bound guard."""
        grid = jnp.linspace(0.0, 2.0, 64)
        out = inverse_cdf_draw(jnp.zeros(64), grid, jnp.asarray(0.37))
        np.testing.assert_array_equal(np.asarray(out), np.asarray(grid[-1]))

    def test_differentiable_in_weight_parameter(self):
        def f(scale):
            grid = jnp.linspace(0.0, 1.0, 64)
            return inverse_cdf_draw(jnp.exp(-scale * grid), grid, jnp.asarray(0.5))

        g = jax.grad(f)(1.0)
        assert jnp.isfinite(g)


# --- expm1-stable power-law segment kernels (audit S4) -----------------------
#
# power_integral_stable(lo, hi, e) = (hi**e - lo**e)/e via lo**e * D * phi(e*D),
#   D = log(hi/lo), phi(x) = expm1(x)/x  — one smooth expression in e, so autodiff
#   carries the CORRECT gradient through the removable e=0 (alpha=1) singularity
#   (the old exp_safe double-where selected an alpha-independent log branch there,
#   silently zeroing the gradient).
# power_ppf_stable(lo, t, e) = (lo**e + t*e)**(1/e) via exp(log lo + s*psi(e*s)),
#   s = t*lo**(-e), psi(y) = log1p(y)/y — the sibling stable inverse.

_LO_HI_CASES = [(0.1, 100.0), (0.08, 0.5), (0.01, 0.08), (1.0, 150.0)]


class TestPowerIntegralStable:
    @pytest.mark.parametrize("lo,hi", _LO_HI_CASES)
    @pytest.mark.parametrize("e", [-1.35, -1.0, -0.3, 0.3, 0.7, 1.0])
    def test_matches_naive_form_away_from_singularity(self, lo, hi, e):
        """Regression: bit-close (<1e-12 rel) to (hi**e - lo**e)/e at e != 0."""
        naive = (hi**e - lo**e) / e
        ours = float(power_integral_stable(lo, hi, e))
        np.testing.assert_allclose(ours, naive, rtol=1e-12)

    @pytest.mark.parametrize("lo,hi", _LO_HI_CASES)
    def test_value_at_e_zero_is_log_ratio(self, lo, hi):
        """lim_{e->0} (hi**e - lo**e)/e = log(hi/lo)."""
        ours = float(power_integral_stable(lo, hi, 0.0))
        np.testing.assert_allclose(ours, np.log(hi / lo), rtol=1e-14)

    @pytest.mark.parametrize("lo,hi", _LO_HI_CASES)
    def test_grad_at_e_zero_matches_analytic(self, lo, hi):
        """d/de[(hi**e - lo**e)/e] at e=0 is (log^2 hi - log^2 lo)/2 (Taylor)."""
        g = float(jax.grad(lambda e: power_integral_stable(lo, hi, e))(0.0))
        expected = (np.log(hi) ** 2 - np.log(lo) ** 2) / 2.0
        np.testing.assert_allclose(g, expected, rtol=1e-10)

    @pytest.mark.parametrize("e0", [0.0, 1e-8, -1e-8, -1.35, 0.7])
    def test_grad_ad_equals_fd_through_zero(self, e0):
        """AD == central FD everywhere, including exactly at and straddling e=0."""
        lo, hi = 0.1, 100.0
        f = lambda e: power_integral_stable(lo, hi, e)
        g = float(jax.grad(f)(e0))
        h = 1e-5
        fd = float((f(e0 + h) - f(e0 - h)) / (2 * h))
        np.testing.assert_allclose(g, fd, rtol=1e-6)


class TestPowerPpfStable:
    @pytest.mark.parametrize("lo,hi", _LO_HI_CASES)
    @pytest.mark.parametrize("e", [-1.35, -0.3, 0.0, 0.7])
    def test_inverts_power_integral_stable(self, lo, hi, e):
        """ppf(lo, integral(lo,hi,e), e) == hi — the round-trip identity."""
        t = power_integral_stable(lo, hi, e)
        m = float(power_ppf_stable(lo, t, e))
        # rtol 1e-10 (the S4 regression budget), not 1e-12: the log-space form
        # amplifies rounding by 1/(1+e*s) as t approaches the full segment
        # integral on a steep, wide segment ((hi/lo)**e ~ 1e-4 here) — measured
        # ~2e-12 worst case, intrinsic to the smooth formulation.
        np.testing.assert_allclose(m, hi, rtol=1e-10)

    @pytest.mark.parametrize("u", [0.0, 0.25, 0.75, 1.0])
    @pytest.mark.parametrize("e", [-1.35, -0.3, 0.7])
    def test_matches_naive_form_away_from_singularity(self, u, e):
        """Regression vs (u*(hi**e - lo**e) + lo**e)**(1/e) at e != 0."""
        lo, hi = 0.1, 100.0
        naive = (u * (hi**e - lo**e) + lo**e) ** (1.0 / e)
        t = u * (hi**e - lo**e) / e
        ours = float(power_ppf_stable(lo, t, e))
        # rtol 1e-10: same u -> 1 log-form rounding amplification as the
        # round-trip test above (S4 regression budget).
        np.testing.assert_allclose(ours, naive, rtol=1e-10)

    def test_value_at_e_zero_is_lo_exp_t(self):
        """lim_{e->0} (lo**e + t*e)**(1/e) = lo * exp(t)."""
        lo, t = 0.1, 2.3
        ours = float(power_ppf_stable(lo, t, 0.0))
        np.testing.assert_allclose(ours, lo * np.exp(t), rtol=1e-14)

    @pytest.mark.parametrize("e0", [0.0, 1e-8, -1.35, 0.7])
    def test_grad_ad_equals_fd_in_e(self, e0):
        """AD == central FD in e, including exactly e=0 (the S4 fix)."""
        lo, hi, u = 0.1, 100.0, 0.5
        # t follows e as in the real ppf call chain: t = u * integral(lo, hi, e)
        f = lambda e: power_ppf_stable(lo, u * power_integral_stable(lo, hi, e), e)
        g = float(jax.grad(f)(e0))
        h = 1e-5
        fd = float((f(e0 + h) - f(e0 - h)) / (2 * h))
        np.testing.assert_allclose(g, fd, rtol=1e-6)

    def test_grad_finite_under_jit(self):
        f = jax.jit(
            jax.grad(lambda e: power_ppf_stable(0.1, power_integral_stable(0.1, 100.0, e), e))
        )
        assert jnp.isfinite(f(0.0)) and jnp.isfinite(f(-1.35))
