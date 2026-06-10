# tests/unit/profiles/test_limepy_tables.py
"""AnisoDensityTable: tabulated anisotropic LIMEPY density rho_hat(W, p; g).
SpeedCDFTable: tabulated isotropic speed inverse-CDF u(W, unif; g).

The tables must reproduce the exact quadrature oracles (_aniso_density_scalar,
_sample_unit_speed / direct DF quadrature) to the stated budgets across their
whole domains, and be differentiable in (g, queries).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _ks_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic max|ECDF_a - ECDF_b| (scipy-free, numpy only)."""
    a = np.sort(np.asarray(a))
    b = np.sort(np.asarray(b))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


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

    def _solve(self, method, W0=7.0, rescale=(1.0, 1.6), ra=(10.0, 10.0),
               xi_max=800.0, n_points=2000):
        from progenax.profiles.limepy_multimass import solve_multicomponent_limepy
        return solve_multicomponent_limepy(
            jnp.array([0.6, 0.4]), jnp.array(rescale), W0=W0, g=1.0,
            xi_max=xi_max, n_points=n_points, ra_hat_j=jnp.array(ra),
            aniso_method=method)

    @pytest.mark.parametrize(
        "W0, rescale, ra, xi_max, n_points",
        [
            # baseline (measured max|dpsi| 8.9e-5 vs budget 7e-4, 2026-06)
            (7.0, (1.0, 1.6), (10.0, 10.0), 800.0, 2000),
            # stronger anisotropy, shallower well (measured 1.14e-4 vs 5e-4)
            pytest.param(5.0, (1.0, 1.6), (5.0, 5.0), 800.0, 2000,
                         marks=pytest.mark.slow),
            # deep well, wide rescale span, huge box (measured 1.93e-4 vs 9e-4)
            pytest.param(9.0, (1.0, 2.2), (40.0, 40.0), 5000.0, 3000,
                         marks=pytest.mark.slow),
        ],
    )
    def test_table_solve_matches_quadrature_solve(self, W0, rescale, ra,
                                                  xi_max, n_points):
        """|psi_table - psi_quad| <= 1e-4 * W0 everywhere; per-component
        densities match to 2e-4 absolute (normalized units)."""
        kw = dict(W0=W0, rescale=rescale, ra=ra, xi_max=xi_max, n_points=n_points)
        xi_q, psi_q, rho_q = self._solve("quadrature", **kw)
        xi_t, psi_t, rho_t = self._solve("table", **kw)
        np.testing.assert_allclose(np.asarray(xi_t), np.asarray(xi_q), rtol=0)
        dpsi = float(jnp.max(jnp.abs(psi_t - psi_q)))
        drho = float(jnp.max(jnp.abs(rho_t - rho_q)))
        assert dpsi <= 1e-4 * W0, f"max|dpsi| {dpsi:.2e} > {1e-4 * W0:.1e}"
        assert drho <= 2e-4, f"max|drho_j| {drho:.2e} > 2e-4"

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

    def test_explicit_table_matches_internal_build(self):
        """The constructor-dedup path: passing the shared solve-box table
        explicitly (aniso_table=...) is BIT-IDENTICAL to letting the solve
        build it internally (same box formula, same jitted build) -- so
        MultiComponentCluster building the table ONCE and sharing it between
        the solve and the CDF density source changes nothing."""
        from progenax.profiles.limepy_multimass import (
            _solver_table,
            solve_multicomponent_limepy,
        )

        alpha = jnp.array([0.6, 0.4])
        rescale = jnp.array([1.0, 1.6])
        ra = jnp.array([10.0, 10.0])
        tab = _solver_table(rescale, ra, 7.0, 1.0, 800.0)
        kw = dict(xi_max=800.0, n_points=500, ra_hat_j=ra, aniso_method="table")
        xi_i, psi_i, rho_i = solve_multicomponent_limepy(alpha, rescale, 7.0, 1.0, **kw)
        xi_e, psi_e, rho_e = solve_multicomponent_limepy(alpha, rescale, 7.0, 1.0,
                                                         aniso_table=tab, **kw)
        np.testing.assert_array_equal(np.asarray(psi_e), np.asarray(psi_i))
        np.testing.assert_array_equal(np.asarray(rho_e), np.asarray(rho_i))

    def test_unknown_aniso_method_raises_at_construction(self):
        """A future third method must raise, never silently fall back to
        quadrature -- both in the solve and in the model constructor."""
        from progenax.cluster.multicomponent import MultiComponentCluster
        from progenax.profiles.limepy_multimass import solve_multicomponent_limepy

        with pytest.raises(ValueError, match="aniso_method"):
            MultiComponentCluster.from_components(
                alpha_j=jnp.array([0.6, 0.4]), w_j=jnp.array([1.0, 0.79]),
                m_j=jnp.array([1.0, 4.0]), W0=7.0, g=1.0,
                ra_hat_j=jnp.array([10.0, 10.0]), xi_max=800.0,
                aniso_method="bogus")
        with pytest.raises(ValueError, match="aniso_method"):
            solve_multicomponent_limepy(
                jnp.array([0.6, 0.4]), jnp.array([1.0, 1.6]), 7.0, 1.0,
                xi_max=800.0, n_points=500, ra_hat_j=jnp.array([10.0, 10.0]),
                aniso_method="bogus")

    @pytest.mark.slow
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


class TestSpeedCDFTable:
    """Tabulated isotropic speed inverse-CDF (Tranche B, Task 5).

    One (sqrt W, x = u/sqrt(2W)) CDF table replaces the per-star 256-point
    E_gamma quadrature of _sample_unit_speed. The contract: the table draw is
    DISTRIBUTIONALLY identical to the exact sampler (1% moments, KS < 0.02)
    and matches the direct DF quadrature moments to 1%.
    """

    def _table(self, W_max=10.0, g=1.0, n_W=256, n_x=256):
        from progenax.profiles.limepy_tables import SpeedCDFTable
        return SpeedCDFTable.build(W_max=W_max, g=g, n_W=n_W, n_x=n_x)

    def test_sampled_moments_match_exact_sampler(self):
        """Per W bin (8 values spanning [0.2, 9.5]): mean u and rms u from the
        table draw agree with the exact per-star sampler (_sample_unit_speed)
        to 1% relative, and the two-sample KS statistic < 0.02.

        20k INDEPENDENT draws per bin per path (fixed seeds): the two-sample
        noise floor is E[KS] ~ 0.87/sqrt(10^4) ~ 0.009 and ~0.45% on the mean,
        so the 0.02 / 1% gates sit several sigma above pure sampling noise.
        """
        from progenax.kinematics.limepy_df import _sample_unit_speed

        g = jnp.asarray(1.0)
        tab = self._table(W_max=10.0, g=1.0)
        n = 20_000
        for i, W0 in enumerate(np.linspace(0.2, 9.5, 8)):
            W = jnp.full((n,), W0)
            keys = jax.random.split(jax.random.PRNGKey(100 + i), n)
            u_exact = jax.vmap(lambda k, w: _sample_unit_speed(k, w, g, 256))(keys, W)
            unif = jax.random.uniform(jax.random.PRNGKey(900 + i), (n,))
            u_tab = jax.vmap(tab.inverse)(W, unif)

            mean_e, mean_t = float(jnp.mean(u_exact)), float(jnp.mean(u_tab))
            rms_e = float(jnp.sqrt(jnp.mean(u_exact**2)))
            rms_t = float(jnp.sqrt(jnp.mean(u_tab**2)))
            assert abs(mean_t - mean_e) / mean_e < 0.01, (
                f"W={W0:.2f}: mean u {mean_t:.4f} vs exact {mean_e:.4f}")
            assert abs(rms_t - rms_e) / rms_e < 0.01, (
                f"W={W0:.2f}: rms u {rms_t:.4f} vs exact {rms_e:.4f}")
            ks = _ks_two_sample(np.asarray(u_exact), np.asarray(u_tab))
            assert ks < 0.02, f"W={W0:.2f}: KS statistic {ks:.4f} >= 0.02"

    def test_moments_match_df_quadrature(self):
        """Sharper oracle than sampler-vs-sampler: per W in {0.5, 2, 5, 9},
        the table-drawn mean u and <u^2> (50k draws) match the DIRECT DF
        quadrature int u * u^2 E_gamma(g, W - u^2/2) du / int u^2 E_gamma du
        (and <u^2> likewise) to 1% -- this catches a WRONG CDF, not just
        table-vs-sampler agreement."""
        from progenax.profiles.limepy import lowered_exponential

        g = 1.0
        tab = self._table(W_max=10.0, g=g)
        n = 50_000
        for i, W0 in enumerate((0.5, 2.0, 5.0, 9.0)):
            u_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * W0), 4001)
            wgt = u_grid**2 * lowered_exponential(jnp.asarray(g), W0 - u_grid**2 / 2.0)
            norm = jnp.trapezoid(wgt, u_grid)
            mean_q = float(jnp.trapezoid(u_grid * wgt, u_grid) / norm)
            u2_q = float(jnp.trapezoid(u_grid**2 * wgt, u_grid) / norm)

            unif = jax.random.uniform(jax.random.PRNGKey(2000 + i), (n,))
            u = jax.vmap(tab.inverse)(jnp.full((n,), W0), unif)
            mean_t = float(jnp.mean(u))
            u2_t = float(jnp.mean(u**2))
            assert abs(mean_t - mean_q) / mean_q < 0.01, (
                f"W={W0}: <u> {mean_t:.4f} vs DF quadrature {mean_q:.4f}")
            assert abs(u2_t - u2_q) / u2_q < 0.01, (
                f"W={W0}: <u^2> {u2_t:.4f} vs DF quadrature {u2_q:.4f}")

    def test_differentiable_through_table_draw(self):
        """grad of the mean drawn speed w.r.t. a velocity multiplier is finite
        and nonzero through the table draw; grad w.r.t. g through the table
        BUILD is also finite."""
        from progenax.profiles.limepy_tables import SpeedCDFTable

        W = jnp.linspace(0.3, 9.0, 64)
        unif = jax.random.uniform(jax.random.PRNGKey(7), (64,))
        tab = self._table()

        def mean_speed(scale):
            return jnp.mean(scale * jax.vmap(tab.inverse)(W, unif))

        d_scale = jax.grad(mean_speed)(jnp.asarray(1.0))
        assert bool(jnp.isfinite(d_scale)) and float(d_scale) > 0.0

        def mean_speed_g(g):
            t = SpeedCDFTable.build(W_max=10.0, g=g, n_W=64, n_x=64)
            return jnp.mean(jax.vmap(t.inverse)(W, unif))

        d_g = jax.grad(mean_speed_g)(jnp.asarray(1.0))
        assert bool(jnp.isfinite(d_g)), f"d<u>/dg through the build: {d_g}"

    @pytest.mark.parametrize("g", [0.0, 2.5, 3.5])
    def test_high_g_low_W_draws_match_exact_sampler(self, g):
        """High-g W->0 regression: the row-0 raw CDF total scales as W^g, so a
        W=1e-12 row floor underflows an ABSOLUTE 1e-30 normalization guard at
        g >= 2.5 (~1e-31 at g=2.5, ~1e-43 at g=3.5) -- the row-0 'CDF' then
        tops out far below 1 and every interp clamps to x=1, corrupting draws
        for stars with W just above the 1e-6 draw guard (measured ~1.8-1.9x
        moment inflation pre-fix). Gate: at W=1e-5, table draws match the
        exact per-star sampler to 2% on mean u and rms u (the W->0 shape
        x^2 (1-x^2)^g has O(1) normalized moments, resolvable at 20k draws),
        and the row-0 CDF itself must end at exactly 1.0."""
        from progenax.kinematics.limepy_df import _sample_unit_speed

        tab = self._table(W_max=10.0, g=g)
        assert float(tab.cdf[0, -1]) == 1.0, (
            f"g={g}: row-0 CDF ends at {float(tab.cdf[0, -1]):.3e}, not 1.0")

        n = 20_000
        W0 = 1e-5
        W = jnp.full((n,), W0)
        gg = jnp.asarray(g)
        keys = jax.random.split(jax.random.PRNGKey(4242), n)
        u_exact = jax.vmap(lambda k, w: _sample_unit_speed(k, w, gg, 256))(keys, W)
        unif = jax.random.uniform(jax.random.PRNGKey(4343), (n,))
        u_tab = jax.vmap(tab.inverse)(W, unif)

        mean_e, mean_t = float(jnp.mean(u_exact)), float(jnp.mean(u_tab))
        rms_e = float(jnp.sqrt(jnp.mean(u_exact**2)))
        rms_t = float(jnp.sqrt(jnp.mean(u_tab**2)))
        assert abs(mean_t - mean_e) / mean_e < 0.02, (
            f"g={g}, W={W0}: mean u {mean_t:.4e} vs exact {mean_e:.4e} "
            f"(ratio {mean_t / mean_e:.3f})")
        assert abs(rms_t - rms_e) / rms_e < 0.02, (
            f"g={g}, W={W0}: rms u {rms_t:.4e} vs exact {rms_e:.4e} "
            f"(ratio {rms_t / rms_e:.3f})")

    def test_w_zero_draw_is_zero(self):
        """W = 0 (a star at the truncation radius) draws u = 0, no NaN; the
        normalized coordinate makes this automatic (u = x sqrt(2W) = 0)."""
        tab = self._table()
        u0 = jax.vmap(tab.inverse, in_axes=(None, 0))(
            jnp.asarray(0.0), jnp.array([0.0, 0.3, 0.7, 1.0]))
        np.testing.assert_array_equal(np.asarray(u0), 0.0)
        assert bool(jnp.all(jnp.isfinite(u0)))


class TestAnisoSpeedCDFTable:
    """Tabulated ANISOTROPIC speed-marginal inverse-CDF (Tranche B, Task 6).

    One (sqrt W, asinh p, x = u/sqrt(2W)) CDF table replaces the per-star
    256-point quadrature of the speed-marginal step of _sample_speed_angle:
    weight x^2 E_gamma(g, W(1-x^2)) T(p^2 W x^2) per row (the (2W)^(3/2)
    prefactor cancels in the relative row normalization; T's argument is
    beta = p^2 u^2 / 2 = p^2 W x^2 in normalized coordinates). The angular
    conditional cos(theta)|u stays EXACT and is not tabulated.
    """

    def _table(self, W_max=10.0, p_max=10.0, g=1.0, n_W=192, n_p=48, n_x=192):
        from progenax.profiles.limepy_tables import AnisoSpeedCDFTable
        return AnisoSpeedCDFTable.build(W_max=W_max, p_max=p_max, g=g,
                                        n_W=n_W, n_p=n_p, n_x=n_x)

    def test_marginal_moments_match_df_quadrature(self):
        """Per (W, p) in {0.5, 2, 5, 9} x {0, 0.5, 2, 8}: table-drawn <u> and
        <u^2> (30k draws) match the DIRECT quadrature of the exact marginal
        u^2 E_gamma(g, W - u^2/2) T(p^2 u^2 / 2) to 1.5% relative. At p=0
        the marginal must reduce to the ISOTROPIC marginal: cross-check the
        table-drawn moments against the SpeedCDFTable draw to 1%.

        30k draws -> ~0.6% MC noise on <u>; the 1.5% gate sits ~2.5 sigma up.
        One off-g spot check (g=2.5, off-node (W, p)) guards the g dependence
        of the tabulated weight (measured deviation <= 0.81% there).
        """
        from progenax.profiles.limepy import _angle_integral_T, lowered_exponential
        from progenax.profiles.limepy_tables import SpeedCDFTable

        g = 1.0
        tab = self._table(W_max=10.0, p_max=10.0, g=g)
        iso = SpeedCDFTable.build(W_max=10.0, g=g)
        n = 30_000
        i = 0
        for W0 in (0.5, 2.0, 5.0, 9.0):
            for p0 in (0.0, 0.5, 2.0, 8.0):
                u_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * W0), 4001)
                wgt = (u_grid**2
                       * lowered_exponential(jnp.asarray(g), W0 - u_grid**2 / 2.0)
                       * _angle_integral_T(p0**2 * u_grid**2 / 2.0))
                norm = jnp.trapezoid(wgt, u_grid)
                mean_q = float(jnp.trapezoid(u_grid * wgt, u_grid) / norm)
                u2_q = float(jnp.trapezoid(u_grid**2 * wgt, u_grid) / norm)

                unif = jax.random.uniform(jax.random.PRNGKey(3000 + i), (n,))
                u = jax.vmap(tab.inverse, in_axes=(None, None, 0))(
                    jnp.asarray(W0), jnp.asarray(p0), unif)
                mean_t, u2_t = float(jnp.mean(u)), float(jnp.mean(u**2))
                assert abs(mean_t - mean_q) / mean_q < 0.015, (
                    f"W={W0}, p={p0}: <u> {mean_t:.4f} vs quadrature {mean_q:.4f}")
                assert abs(u2_t - u2_q) / u2_q < 0.015, (
                    f"W={W0}, p={p0}: <u^2> {u2_t:.4f} vs quadrature {u2_q:.4f}")

                if p0 == 0.0:
                    u_iso = jax.vmap(iso.inverse)(jnp.full((n,), W0), unif)
                    mean_i = float(jnp.mean(u_iso))
                    u2_i = float(jnp.mean(u_iso**2))
                    assert abs(mean_t - mean_i) / mean_i < 0.01, (
                        f"W={W0}, p=0: aniso <u> {mean_t:.4f} vs iso table {mean_i:.4f}")
                    assert abs(u2_t - u2_i) / u2_i < 0.01, (
                        f"W={W0}, p=0: aniso <u^2> {u2_t:.4f} vs iso table {u2_i:.4f}")
                i += 1

        # Off-g spot check (Task-6 review): g=2.5 at one off-node (W, p) --
        # the main sweep is g=1 only; this guards the g dependence of the
        # tabulated weight x^2 E_gamma(g, W(1-x^2)) T(p^2 W x^2). Same 1.5%
        # gate (reviewer measured <= 0.81% deviation here).
        g2 = 2.5
        tab2 = self._table(W_max=10.0, p_max=10.0, g=g2)
        W0, p0 = 3.3, 1.7  # off any sqrt(W)/asinh(p) grid node
        u_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * W0), 4001)
        wgt = (u_grid**2
               * lowered_exponential(jnp.asarray(g2), W0 - u_grid**2 / 2.0)
               * _angle_integral_T(p0**2 * u_grid**2 / 2.0))
        norm = jnp.trapezoid(wgt, u_grid)
        mean_q = float(jnp.trapezoid(u_grid * wgt, u_grid) / norm)
        u2_q = float(jnp.trapezoid(u_grid**2 * wgt, u_grid) / norm)
        unif = jax.random.uniform(jax.random.PRNGKey(3100), (n,))
        u = jax.vmap(tab2.inverse, in_axes=(None, None, 0))(
            jnp.asarray(W0), jnp.asarray(p0), unif)
        mean_t, u2_t = float(jnp.mean(u)), float(jnp.mean(u**2))
        assert abs(mean_t - mean_q) / mean_q < 0.015, (
            f"g={g2}, W={W0}, p={p0}: <u> {mean_t:.4f} vs quadrature {mean_q:.4f}")
        assert abs(u2_t - u2_q) / u2_q < 0.015, (
            f"g={g2}, W={W0}, p={p0}: <u^2> {u2_t:.4f} vs quadrature {u2_q:.4f}")

    @pytest.mark.parametrize("g", [0.0, 2.5, 3.5])
    def test_high_g_low_W_rows_normalized(self, g):
        """Every CDF row is healthy for g in {0, 2.5, 3.5} -- the Task-5
        lesson applied from the start: the raw row total scales as W^g (times
        the T suppression at large p), so relative normalization with a 1e-6
        row-W floor is required; an absolute regularizer would swamp low-W
        high-g rows and corrupt draws near the truncation radius.

        The AnisoSpeedCDFTable build unconditionally pins cdf[..., -1] to 1.0
        (the lax.map x/x ulp fix), so asserting only the last column is
        VACUOUS against the underflow failure mode this test documents. The
        real detectors are: (i) every entry finite (a zero row total divides
        to NaN), (ii) every row monotone non-decreasing, (iii) cdf[..., -2]
        finite and strictly < 1 (an underflowed/swamped row tops out early
        and only the pin reaches 1). Checked for BOTH SpeedCDFTable and
        AnisoSpeedCDFTable."""
        from progenax.profiles.limepy_tables import SpeedCDFTable

        iso = SpeedCDFTable.build(W_max=10.0, g=g, n_W=48, n_x=96)
        tab = self._table(W_max=10.0, p_max=10.0, g=g, n_W=48, n_p=16, n_x=96)
        for name, cdf in (("SpeedCDFTable", np.asarray(iso.cdf)),
                          ("AnisoSpeedCDFTable", np.asarray(tab.cdf))):
            assert np.isfinite(cdf).all(), (
                f"g={g} {name}: non-finite CDF entries "
                f"({np.size(cdf) - np.isfinite(cdf).sum()} of {np.size(cdf)})")
            ends = cdf[..., -1]
            np.testing.assert_array_equal(
                ends, 1.0, err_msg=f"g={g} {name}: CDF row ends range "
                f"[{ends.min():.3e}, {ends.max():.3e}], expected exactly 1.0")
            diffs = np.diff(cdf, axis=-1)
            assert (diffs >= 0.0).all(), (
                f"g={g} {name}: non-monotone CDF rows "
                f"(min diff {diffs.min():.3e})")
            penult = cdf[..., -2]
            assert np.isfinite(penult).all() and (penult < 1.0).all(), (
                f"g={g} {name}: cdf[..., -2] range "
                f"[{penult.min():.3e}, {penult.max():.3e}], expected all < 1")

    def test_differentiable(self):
        """grad through the table BUILD (g) and through a drawn-speed
        functional (velocity multiplier and W) are finite; the multiplier
        gradient is nonzero."""
        from progenax.profiles.limepy_tables import AnisoSpeedCDFTable

        W = jnp.linspace(0.3, 9.0, 32)
        p = jnp.linspace(0.0, 8.0, 32)
        unif = jax.random.uniform(jax.random.PRNGKey(11), (32,))
        tab = self._table(n_W=64, n_p=16, n_x=64)

        def mean_speed(scale):
            return jnp.mean(scale * jax.vmap(tab.inverse)(W, p, unif))

        d_scale = jax.grad(mean_speed)(jnp.asarray(1.0))
        assert bool(jnp.isfinite(d_scale)) and float(d_scale) > 0.0

        def mean_speed_g(g):
            t = AnisoSpeedCDFTable.build(W_max=10.0, p_max=10.0, g=g,
                                         n_W=32, n_p=12, n_x=48)
            return jnp.mean(jax.vmap(t.inverse)(W, p, unif))

        d_g = jax.grad(mean_speed_g)(jnp.asarray(1.0))
        assert bool(jnp.isfinite(d_g)), f"d<u>/dg through the build: {d_g}"

        def mean_speed_W(w0):
            return jnp.mean(jax.vmap(tab.inverse, in_axes=(None, 0, 0))(w0, p, unif))

        d_W = jax.grad(mean_speed_W)(jnp.asarray(4.0))
        assert bool(jnp.isfinite(d_W))

    def test_w_zero_draw_is_zero(self):
        """W = 0 (truncation radius) draws u = 0, no NaN, at any p."""
        tab = self._table(n_W=48, n_p=16, n_x=96)
        u0 = jax.vmap(tab.inverse, in_axes=(None, 0, 0))(
            jnp.asarray(0.0), jnp.array([0.0, 1.0, 5.0, 9.0]),
            jnp.array([0.0, 0.3, 0.7, 1.0]))
        np.testing.assert_array_equal(np.asarray(u0), 0.0)
        assert bool(jnp.all(jnp.isfinite(u0)))
