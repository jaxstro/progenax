"""Tests for progenax.numerics — the shared trapezoid/inverse-CDF primitives.

These helpers must be BIT-IDENTICAL to the inline patterns they replace
(same op order: pairwise average -> cumsum -> concat zero), because five
Poisson passes and eight speed-CDF kernels migrate onto them.
"""

import jax
import jax.numpy as jnp
import numpy as np

from progenax.numerics import cumulative_trapz, inverse_cdf_draw


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
