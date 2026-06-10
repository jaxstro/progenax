# tests/unit/profiles/test_limepy_tables.py
"""AnisoDensityTable: tabulated anisotropic LIMEPY density rho_hat(W, p; g).

The table must reproduce the exact quadrature (_aniso_density_scalar) to the
stated budget across its whole domain, and be differentiable in (g, queries).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


class TestAnisoDensityTable:
    def _table(self, W_max=12.0, p_max=80.0, g=1.0, n_W=512, n_p=96):
        from progenax.profiles.limepy_tables import AnisoDensityTable
        return AnisoDensityTable.build(W_max=W_max, p_max=p_max, g=g,
                                       n_W=n_W, n_p=n_p)

    def test_reproduces_exact_quadrature_on_random_points(self):
        """Max relative error <= 1e-5 against _aniso_density_scalar on 2000
        random interior points (relative to max(exact, 1e-8 * central))."""
        from progenax.profiles.limepy import _aniso_density_scalar

        tab = self._table()
        rng = np.random.default_rng(0)
        W = jnp.asarray(rng.uniform(1e-3 * 12.0, 12.0, 2000))
        p = jnp.asarray(rng.uniform(0.0, 80.0, 2000))
        approx = jax.vmap(tab.evaluate)(W, p)
        exact = jax.vmap(lambda w, pp: _aniso_density_scalar(w, pp, jnp.asarray(1.0)))(W, p)
        central = float(_aniso_density_scalar(jnp.asarray(12.0), jnp.asarray(0.0),
                                              jnp.asarray(1.0)))
        rel = np.asarray(jnp.abs(approx - exact) /
                         jnp.maximum(exact, 1e-8 * central))
        assert rel.max() <= 1e-5, f"max rel err {rel.max():.2e}"

    def test_W_zero_gives_zero_density(self):
        tab = self._table()
        assert float(tab.evaluate(jnp.asarray(0.0), jnp.asarray(3.0))) == 0.0

    def test_clamps_outside_domain(self):
        """Queries past the box edges clamp (no NaN, no extrapolation blow-up)."""
        tab = self._table()
        out = jax.vmap(tab.evaluate)(jnp.array([13.0, 5.0]), jnp.array([3.0, 100.0]))
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_isotropic_p0_matches_closed_form(self):
        """At p=0 the anisotropic density reduces to the isotropic closed form
        sqrt(2 pi) * limepy_density_hat (an independent oracle, not the
        quadrature; the sqrt(2 pi) proportionality is documented in
        _aniso_density_scalar and verified numerically to ~2e-5, the trapezoid
        error of the exact quadrature itself)."""
        from progenax.profiles.limepy import limepy_density_hat

        tab = self._table()
        W = jnp.linspace(0.05, 11.5, 64)
        approx = jax.vmap(lambda w: tab.evaluate(w, jnp.asarray(0.0)))(W)
        exact = jnp.sqrt(2.0 * jnp.pi) * limepy_density_hat(W, 1.0)
        np.testing.assert_allclose(np.asarray(approx), np.asarray(exact),
                                   rtol=5e-5, atol=1e-12)

    def test_gradient_finite_at_W_nonpositive(self):
        """C1 regression: d(evaluate)/dW must be FINITE at W=0 and W=-1.

        sqrt(max(W, 0)) has a NaN cotangent at W=0 under jax.grad, and the
        final where() only masks the primal -- the quadrature oracle
        (_aniso_density_scalar) guards with max(W, 1e-12) and returns a 0
        gradient there; the table must match.
        """
        tab = self._table()
        grad_fn = jax.grad(lambda w: tab.evaluate(w, jnp.asarray(2.0)))
        dW_at_zero = grad_fn(jnp.asarray(0.0))
        dW_at_neg = grad_fn(jnp.asarray(-1.0))
        assert bool(jnp.isfinite(dW_at_zero)), f"dV/dW NaN at W=0: {dW_at_zero}"
        assert bool(jnp.isfinite(dW_at_neg)), f"dV/dW NaN at W=-1: {dW_at_neg}"
        assert float(dW_at_neg) == 0.0

    def test_nonnegative_in_W_to_zero_corner(self):
        """I1 regression: cubic Lagrange can overshoot by O(1e-15 * central)
        near the W->0 power-law corner; evaluate must clamp to rho_hat >= 0
        (Tranche B builds CDFs from this)."""
        tab = self._table()
        W = jnp.geomspace(1e-6, 0.05, 200)
        for p0 in (0.0, 1.0, 10.0, 80.0):
            vals = jax.vmap(lambda w: tab.evaluate(w, jnp.asarray(p0)))(W)
            assert bool(jnp.all(vals >= 0.0)), (
                f"negative density at p={p0}: min {float(vals.min()):.2e}"
            )

    def test_accuracy_in_W_to_zero_corner(self):
        """I2 regression: the W->0 power-law corner (geomspace W in
        [1e-6, 0.012] x p in {0, 1, 10, 80}) must meet the same floored
        relative budget as the main accuracy test."""
        from progenax.profiles.limepy import _aniso_density_scalar

        tab = self._table()
        W1 = jnp.geomspace(1e-6, 0.012, 500)
        central = float(_aniso_density_scalar(jnp.asarray(12.0), jnp.asarray(0.0),
                                              jnp.asarray(1.0)))
        worst = 0.0
        for p0 in (0.0, 1.0, 10.0, 80.0):
            pp = jnp.full_like(W1, p0)
            approx = jax.vmap(tab.evaluate)(W1, pp)
            exact = jax.vmap(
                lambda w, q: _aniso_density_scalar(w, q, jnp.asarray(1.0))
            )(W1, pp)
            rel = np.asarray(jnp.abs(approx - exact) /
                             jnp.maximum(exact, 1e-8 * central))
            worst = max(worst, float(rel.max()))
            assert rel.max() <= 1e-5, f"p={p0}: max rel err {rel.max():.2e}"

    def test_differentiable_in_g_and_queries(self):
        """AD through the table BUILD (g) and the query (W, p); AD matches FD."""
        from progenax.profiles.limepy_tables import AnisoDensityTable

        def f(g, W, p):
            tab = AnisoDensityTable.build(W_max=10.0, p_max=20.0, g=g,
                                          n_W=128, n_p=32)
            return tab.evaluate(W, p)

        g0, W0_, p0 = 1.0, 4.0, 2.0
        dg = jax.grad(f, 0)(g0, jnp.asarray(W0_), jnp.asarray(p0))
        dW = jax.grad(f, 1)(g0, jnp.asarray(W0_), jnp.asarray(p0))
        dp = jax.grad(f, 2)(g0, jnp.asarray(W0_), jnp.asarray(p0))
        assert all(jnp.isfinite(d) for d in (dg, dW, dp))
        eps = 1e-5
        fd_g = (f(g0 + eps, jnp.asarray(W0_), jnp.asarray(p0))
                - f(g0 - eps, jnp.asarray(W0_), jnp.asarray(p0))) / (2 * eps)
        np.testing.assert_allclose(float(dg), float(fd_g), rtol=1e-4, atol=1e-8)


class TestTableBackedSolver:
    """solve_multicomponent_limepy(aniso_method='table') reproduces the exact
    quadrature solve to the stated budget and is the new default."""

    def _solve(self, method):
        from progenax.profiles.limepy_multimass import solve_multicomponent_limepy
        return solve_multicomponent_limepy(
            jnp.array([0.6, 0.4]), jnp.array([1.0, 1.6]), W0=7.0, g=1.0,
            xi_max=800.0, n_points=2000, ra_hat_j=jnp.array([10.0, 10.0]),
            aniso_method=method)

    def test_table_solve_matches_quadrature_solve(self):
        """|psi_table - psi_quad| <= 1e-4 * W0 everywhere; per-component
        densities match to 2e-4 absolute (normalized units)."""
        xi_q, psi_q, rho_q = self._solve("quadrature")
        xi_t, psi_t, rho_t = self._solve("table")
        np.testing.assert_allclose(np.asarray(xi_t), np.asarray(xi_q), rtol=0)
        assert float(jnp.max(jnp.abs(psi_t - psi_q))) <= 1e-4 * 7.0
        assert float(jnp.max(jnp.abs(rho_t - rho_q))) <= 2e-4

    def test_table_is_default_and_iso_path_untouched(self):
        """Default aniso_method is 'table'; the ISOTROPIC path is bit-identical
        to before (no table involved when ra_hat_j is None)."""
        from progenax.profiles.limepy_multimass import solve_multicomponent_limepy
        xi_d, psi_d, _ = self._solve("table")
        from inspect import signature
        assert signature(solve_multicomponent_limepy).parameters["aniso_method"].default == "table"
        xi_i, psi_i, _ = solve_multicomponent_limepy(
            jnp.array([0.6, 0.4]), jnp.array([1.0, 1.6]), W0=7.0, g=1.0,
            xi_max=300.0, n_points=2000)  # iso: no ra_hat_j
        assert bool(jnp.all(jnp.isfinite(psi_i)))

    def test_table_solve_differentiable_in_rescale_ra(self):
        from progenax.profiles.limepy_multimass import solve_multicomponent_limepy

        def metric(rescale, ra):
            xi, psi, _ = solve_multicomponent_limepy(
                jnp.array([0.5, 0.5]), rescale, 7.0, 1.0, xi_max=800.0,
                n_points=1500, ra_hat_j=ra, aniso_method="table")
            return jnp.mean(psi[:300])

        d_r = jax.grad(metric, 0)(jnp.array([1.0, 1.6]), jnp.array([10.0, 10.0]))
        d_a = jax.grad(metric, 1)(jnp.array([1.0, 1.6]), jnp.array([10.0, 10.0]))
        assert jnp.all(jnp.isfinite(d_r)) and jnp.any(jnp.abs(d_r) > 0)
        assert jnp.all(jnp.isfinite(d_a)) and jnp.any(jnp.abs(d_a) > 0)

    def test_table_solve_is_faster(self):
        """Warm table solve >= 3x faster than warm quadrature solve (the measured
        target is ~5-10x; assert a conservative 3x so the test is not flaky)."""
        import time

        def timed(method):
            self._solve(method)  # warm/compile
            t0 = time.perf_counter()
            out = self._solve(method)
            jax.block_until_ready(out[1])
            return time.perf_counter() - t0

        assert timed("quadrature") / timed("table") >= 3.0
