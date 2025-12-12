"""Tests for BM19 volume PDF and CDF remap utilities."""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from progenax.gravoturb import bm19_pdf, bm19_model as bm19


class TestBM19VolumePDF:
    """Tests for bm19_volume_pdf()."""

    def test_pdf_positive(self):
        """PDF values should be non-negative."""
        s = jnp.linspace(-5.0, 10.0, 100)
        sigma_s_sq = 2.0
        s_t = 3.0  # (2.0 - 0.5) * 2.0
        alpha = 2.0

        p = bm19_pdf.bm19_volume_pdf(s, sigma_s_sq, s_t, alpha)

        assert jnp.all(p >= 0)

    def test_pdf_continuous_at_st(self):
        """PDF should be continuous at transition point s_t."""
        sigma_s_sq = 2.0
        alpha = 2.0
        s_t = (alpha - 0.5) * sigma_s_sq

        # Sample very close to s_t from both sides
        eps = 1e-6
        s_below = s_t - eps
        s_above = s_t + eps

        p_below = bm19_pdf.bm19_volume_pdf(s_below, sigma_s_sq, s_t, alpha)
        p_above = bm19_pdf.bm19_volume_pdf(s_above, sigma_s_sq, s_t, alpha)

        # Should be nearly equal (within numerical precision)
        assert jnp.isclose(p_below, p_above, rtol=1e-4)

    def test_pdf_lognormal_part(self):
        """For s << s_t, should match pure lognormal PDF."""
        sigma_s_sq = 2.0
        sigma_s = jnp.sqrt(sigma_s_sq)
        s_0 = -sigma_s_sq / 2.0  # Mass conservation
        alpha = 2.0
        s_t = (alpha - 0.5) * sigma_s_sq

        # Test at s = s_0 (peak of lognormal)
        s_test = s_0
        p_bm19 = bm19_pdf.bm19_volume_pdf(s_test, sigma_s_sq, s_t, alpha)

        # Pure lognormal PDF at s_0
        p_ln = jnp.exp(-(s_test - s_0) ** 2 / (2 * sigma_s_sq)) / (
            jnp.sqrt(2 * jnp.pi) * sigma_s
        )

        assert jnp.isclose(p_bm19, p_ln, rtol=1e-6)

    def test_pdf_powerlaw_tail(self):
        """For s >> s_t, should follow powerlaw decay."""
        sigma_s_sq = 2.0
        alpha = 2.0
        s_t = (alpha - 0.5) * sigma_s_sq

        # Two points in the powerlaw tail
        s1 = s_t + 2.0
        s2 = s_t + 4.0

        p1 = bm19_pdf.bm19_volume_pdf(s1, sigma_s_sq, s_t, alpha)
        p2 = bm19_pdf.bm19_volume_pdf(s2, sigma_s_sq, s_t, alpha)

        # Powerlaw: p ~ exp(-alpha * s) => p2/p1 = exp(-alpha * (s2 - s1))
        expected_ratio = jnp.exp(-alpha * (s2 - s1))
        actual_ratio = p2 / p1

        assert jnp.isclose(actual_ratio, expected_ratio, rtol=1e-6)

    def test_differentiable(self):
        """PDF should be differentiable."""

        def loss(sigma_s_sq):
            s = jnp.linspace(-3.0, 5.0, 50)
            s_t = 1.5 * sigma_s_sq
            p = bm19_pdf.bm19_volume_pdf(s, sigma_s_sq, s_t, 2.0)
            return jnp.sum(p)

        grad = jax.grad(loss)(2.0)
        assert jnp.isfinite(grad)


class TestBuildBM19CDFTable:
    """Tests for build_bm19_cdf_table()."""

    def test_cdf_monotonic(self):
        """CDF should be monotonically increasing."""
        sigma_s_sq = 2.0
        s_t = 3.0
        alpha = 2.0

        s_grid, F_grid = bm19_pdf.build_bm19_cdf_table(sigma_s_sq, s_t, alpha)

        # CDF should be increasing
        dF = jnp.diff(F_grid)
        assert jnp.all(dF >= -1e-10)  # Allow tiny numerical noise

    def test_cdf_bounds(self):
        """CDF should go from 0 to 1."""
        sigma_s_sq = 2.0
        s_t = 3.0
        alpha = 2.0

        s_grid, F_grid = bm19_pdf.build_bm19_cdf_table(sigma_s_sq, s_t, alpha)

        assert jnp.isclose(F_grid[0], 0.0, atol=1e-6)
        assert jnp.isclose(F_grid[-1], 1.0, atol=1e-6)

    def test_cdf_grid_size(self):
        """Grid size parameter should work."""
        sigma_s_sq = 2.0
        s_t = 3.0
        alpha = 2.0

        s_grid, F_grid = bm19_pdf.build_bm19_cdf_table(
            sigma_s_sq, s_t, alpha, n_grid=500
        )

        assert len(s_grid) == 500
        assert len(F_grid) == 500

    def test_different_alpha_values(self):
        """Should work for various alpha values in tested range."""
        sigma_s_sq = 2.0
        alphas = [1.5, 2.0, 2.5, 3.0]

        for alpha in alphas:
            s_t = (alpha - 0.5) * sigma_s_sq
            s_grid, F_grid = bm19_pdf.build_bm19_cdf_table(sigma_s_sq, s_t, alpha)

            assert jnp.all(jnp.isfinite(s_grid))
            assert jnp.all(jnp.isfinite(F_grid))
            assert jnp.isclose(F_grid[-1], 1.0, atol=1e-4)


class TestBM19ICDF:
    """Tests for bm19_icdf()."""

    def test_icdf_bounds(self):
        """Inverse CDF should map (0,1) to s range."""
        sigma_s_sq = 2.0
        s_t = 3.0
        alpha = 2.0

        s_grid, F_grid = bm19_pdf.build_bm19_cdf_table(sigma_s_sq, s_t, alpha)

        # Test at u = 0.5 (median)
        u = jnp.array([0.5])
        s = bm19_pdf.bm19_icdf(u, s_grid, F_grid)

        # Result should be within grid range
        assert s[0] >= s_grid[0]
        assert s[0] <= s_grid[-1]

    def test_icdf_monotonic(self):
        """Inverse CDF should be monotonically increasing."""
        sigma_s_sq = 2.0
        s_t = 3.0
        alpha = 2.0

        s_grid, F_grid = bm19_pdf.build_bm19_cdf_table(sigma_s_sq, s_t, alpha)

        u = jnp.linspace(0.01, 0.99, 100)
        s = bm19_pdf.bm19_icdf(u, s_grid, F_grid)

        ds = jnp.diff(s)
        assert jnp.all(ds >= 0)

    def test_icdf_inverse_of_cdf(self):
        """F^{-1}(F(s)) should equal s (up to discretization)."""
        sigma_s_sq = 2.0
        s_t = 3.0
        alpha = 2.0

        s_grid, F_grid = bm19_pdf.build_bm19_cdf_table(
            sigma_s_sq, s_t, alpha, n_grid=5000  # High resolution
        )

        # Pick some F values and invert
        u_test = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
        s_recovered = bm19_pdf.bm19_icdf(u_test, s_grid, F_grid)

        # Now find F at those s values (via linear interpolation)
        for i, u in enumerate(u_test):
            s_val = s_recovered[i]
            # Find where s_val falls in s_grid
            idx = jnp.searchsorted(s_grid, s_val)
            idx = jnp.clip(idx, 1, len(s_grid) - 1)
            # Linear interpolation
            t = (s_val - s_grid[idx - 1]) / (s_grid[idx] - s_grid[idx - 1] + 1e-12)
            F_at_s = F_grid[idx - 1] + t * (F_grid[idx] - F_grid[idx - 1])
            # Should match original u
            assert jnp.isclose(F_at_s, u, atol=0.01)

    def test_differentiable(self):
        """Inverse CDF should be differentiable."""
        sigma_s_sq = 2.0
        s_t = 3.0
        alpha = 2.0
        s_grid, F_grid = bm19_pdf.build_bm19_cdf_table(sigma_s_sq, s_t, alpha)

        def loss(u_val):
            u = jnp.array([u_val])
            s = bm19_pdf.bm19_icdf(u, s_grid, F_grid)
            return s[0]

        grad = jax.grad(loss)(0.5)
        assert jnp.isfinite(grad)
        assert grad > 0  # Should be positive (ICDF is monotonic increasing)


class TestGaussianToBM19:
    """Tests for gaussian_to_bm19()."""

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        key = random.PRNGKey(42)
        g = random.normal(key, (32, 32, 32))

        sigma_s_sq = 2.0
        s_t = 3.0
        alpha = 2.0

        s = bm19_pdf.gaussian_to_bm19(g, sigma_s_sq, s_t, alpha)

        assert s.shape == g.shape

    def test_output_distribution(self):
        """Output should approximately match BM19 distribution."""
        key = random.PRNGKey(42)
        # Large sample for statistics
        g = random.normal(key, (64, 64, 64))

        sigma_s_sq = 2.0
        alpha = 2.0
        s_t = (alpha - 0.5) * sigma_s_sq

        s = bm19_pdf.gaussian_to_bm19(g, sigma_s_sq, s_t, alpha)

        # Check mean is approximately s_0 = -sigma_s_sq/2
        expected_mean = -sigma_s_sq / 2.0
        actual_mean = float(jnp.mean(s))
        assert jnp.isclose(actual_mean, expected_mean, atol=0.3)

    def test_f_tail_matches_f_dense(self):
        """f_tail_actual should match f_dense for CDF-remapped field."""
        key = random.PRNGKey(42)
        g = random.normal(key, (64, 64, 64))

        sigma_s_sq = 2.0
        alpha = 2.0
        s_t = (alpha - 0.5) * sigma_s_sq
        kappa = 10.0

        s = bm19_pdf.gaussian_to_bm19(g, sigma_s_sq, s_t, alpha)
        rho = jnp.exp(s)

        # Compute f_tail_actual with soft sigmoid
        w = jax.nn.sigmoid(kappa * (s - s_t))
        f_tail_actual = float(jnp.sum(w * rho) / jnp.sum(rho))

        # Theoretical f_dense
        f_dense_theory = float(bm19.f_dense_bm19_full(sigma_s_sq, s_t, alpha))

        # Should match within ~10% for single realization
        relative_error = abs(f_tail_actual - f_dense_theory) / f_dense_theory
        assert relative_error < 0.15  # 15% tolerance for single 64^3 realization

    def test_f_tail_high_alpha(self):
        """CDF remap should work for high alpha where lognormal fails."""
        sigma_s_sq = 2.0
        alpha = 3.0  # High alpha - lognormal would fail
        s_t = (alpha - 0.5) * sigma_s_sq

        # Average over multiple realizations to reduce variance
        n_realizations = 10
        f_tails = []
        for i in range(n_realizations):
            key = random.PRNGKey(42 + i * 100)
            g = random.normal(key, (64, 64, 64))
            s = bm19_pdf.gaussian_to_bm19(g, sigma_s_sq, s_t, alpha)
            rho = jnp.exp(s)
            f_tail = float(jnp.sum(rho[s > s_t]) / jnp.sum(rho))
            f_tails.append(f_tail)

        f_tail_mean = jnp.mean(jnp.array(f_tails))
        f_dense_theory = float(bm19.f_dense_bm19_full(sigma_s_sq, s_t, alpha))

        # Should have non-trivial tail (unlike pure lognormal which would be ~0)
        assert f_tail_mean > 0.001  # Should not be essentially zero
        assert f_dense_theory > 0.001

        # Mean should match theory within ~30% (sample variance is high at extreme params)
        relative_error = abs(f_tail_mean - f_dense_theory) / f_dense_theory
        assert relative_error < 0.30

    def test_differentiable(self):
        """CDF remap should be differentiable."""
        key = random.PRNGKey(42)
        g = random.normal(key, (16, 16, 16))  # Small for speed

        def loss(sigma_s_sq):
            s_t = 1.5 * sigma_s_sq
            s = bm19_pdf.gaussian_to_bm19(g, sigma_s_sq, s_t, 2.0)
            return jnp.mean(s)

        grad = jax.grad(loss)(2.0)
        assert jnp.isfinite(grad)


class TestValidateBM19Field:
    """Tests for validate_bm19_field()."""

    def test_returns_all_keys(self):
        """Should return all expected statistics."""
        key = random.PRNGKey(42)
        g = random.normal(key, (32, 32, 32))

        sigma_s_sq = 2.0
        alpha = 2.0
        s_t = (alpha - 0.5) * sigma_s_sq

        s = bm19_pdf.gaussian_to_bm19(g, sigma_s_sq, s_t, alpha)
        stats = bm19_pdf.validate_bm19_field(s, sigma_s_sq, s_t, alpha)

        expected_keys = [
            "mean_s",
            "expected_mean_s",
            "var_s",
            "expected_var_s",
            "f_tail_actual",
            "f_dense_theory",
            "relative_error_percent",
        ]

        for key in expected_keys:
            assert key in stats

    def test_correctly_identifies_good_field(self):
        """Should show low error for correctly generated field."""
        key = random.PRNGKey(42)
        g = random.normal(key, (64, 64, 64))

        sigma_s_sq = 2.0
        alpha = 2.0
        s_t = (alpha - 0.5) * sigma_s_sq

        s = bm19_pdf.gaussian_to_bm19(g, sigma_s_sq, s_t, alpha)
        stats = bm19_pdf.validate_bm19_field(s, sigma_s_sq, s_t, alpha)

        # Error should be moderate for single realization
        assert abs(stats["relative_error_percent"]) < 20.0


class TestIntegration:
    """Integration tests combining multiple functions."""

    @pytest.mark.parametrize("alpha", [1.5, 2.0, 2.5, 3.0])
    def test_full_pipeline_all_alphas(self, alpha):
        """Full pipeline should work for all tested alpha values."""
        sigma_s_sq = 2.0
        s_t = (alpha - 0.5) * sigma_s_sq

        # Average over realizations for stable test
        n_realizations = 5
        f_tails = []
        for i in range(n_realizations):
            key = random.PRNGKey(int(alpha * 1000) + i * 100)
            g = random.normal(key, (64, 64, 64))
            s = bm19_pdf.gaussian_to_bm19(g, sigma_s_sq, s_t, alpha)

            # Should have finite values
            assert jnp.all(jnp.isfinite(s))

            rho = jnp.exp(s)
            f_tail = float(jnp.sum(rho[s > s_t]) / jnp.sum(rho))
            f_tails.append(f_tail)

        f_tail_mean = jnp.mean(jnp.array(f_tails))
        f_dense_theory = float(bm19.f_dense_bm19_full(sigma_s_sq, s_t, alpha))

        # Should have reasonable f_tail (not ~0 like lognormal at high alpha)
        assert f_tail_mean > 1e-4
        # Mean should match theory within 40% (high alpha has more variance)
        relative_error = abs(f_tail_mean - f_dense_theory) / f_dense_theory
        assert relative_error < 0.40

    @pytest.mark.parametrize("mach", [5.0, 10.0, 20.0])
    def test_full_pipeline_all_machs(self, mach):
        """Full pipeline should work for different Mach numbers."""
        # Get BM19 parameters from pipeline
        result = bm19.bm19_pipeline(mach, alpha=2.0, eta_survive=0.6)
        sigma_s_sq = float(result.sigma_s_sq)
        s_t = float(result.s_t)

        # Average over realizations for stable test
        n_realizations = 5
        f_tails = []
        for i in range(n_realizations):
            key = random.PRNGKey(int(mach * 100) + i * 100)
            g = random.normal(key, (64, 64, 64))
            s = bm19_pdf.gaussian_to_bm19(g, sigma_s_sq, s_t, 2.0)

            assert jnp.all(jnp.isfinite(s))

            rho = jnp.exp(s)
            f_tail = float(jnp.sum(rho[s > s_t]) / jnp.sum(rho))
            f_tails.append(f_tail)

        f_tail_mean = jnp.mean(jnp.array(f_tails))
        f_dense_theory = float(bm19.f_dense_bm19_full(sigma_s_sq, s_t, 2.0))

        assert f_tail_mean > 1e-4
        # Mean should match theory within 30%
        relative_error = abs(f_tail_mean - f_dense_theory) / f_dense_theory
        assert relative_error < 0.30
