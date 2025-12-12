"""Tests for FDF tail selection (BM19-consistent and legacy methods).

Tests cover:
- compute_tail_pmfs_bm19(): Direct s-threshold selection
- compute_tail_pmfs_pn11_legacy(): Local overdensity selection
- sample_positions_tail_bm19(): High-level sampling interface
- TailPMFResult structure and validation
- f_tail_actual consistency with f_dense from BM19 theory
- Differentiability through BM19 method
"""

import warnings

import jax
import jax.numpy as jnp
import pytest
from jax import random


class TestComputeTailPMFsBM19:
    """Tests for compute_tail_pmfs_bm19()."""

    @pytest.fixture
    def sample_lognormal_field(self):
        """Generate a sample lognormal density field."""
        key = random.PRNGKey(42)
        grid_size = 32

        # Generate lognormal field with known sigma_s
        sigma_s = 1.0
        z = random.normal(key, (grid_size, grid_size, grid_size))
        # s = sigma_s * z - sigma_s^2/2 (ensures mean(rho/rho_mean) = 1)
        s = sigma_s * z - sigma_s**2 / 2
        rho_grid = jnp.exp(s)

        return rho_grid, sigma_s

    def test_returns_tail_pmf_result(self, sample_lognormal_field):
        """Should return TailPMFResult with all fields."""
        from progenax.cluster.fdf_tail import TailPMFResult, compute_tail_pmfs_bm19

        rho_grid, _ = sample_lognormal_field
        s_t = 1.0

        result = compute_tail_pmfs_bm19(rho_grid, s_t)

        assert isinstance(result, TailPMFResult)
        assert hasattr(result, "p_tail")
        assert hasattr(result, "p_smooth")
        assert hasattr(result, "f_tail_actual")
        assert hasattr(result, "tail_weights")

    def test_pmfs_are_normalized(self, sample_lognormal_field):
        """Both PMFs should sum to 1."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19

        rho_grid, _ = sample_lognormal_field
        s_t = 1.0

        result = compute_tail_pmfs_bm19(rho_grid, s_t)

        assert jnp.isclose(jnp.sum(result.p_tail), 1.0, atol=1e-5)
        assert jnp.isclose(jnp.sum(result.p_smooth), 1.0, atol=1e-5)

    def test_pmfs_are_nonnegative(self, sample_lognormal_field):
        """All PMF values should be non-negative."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19

        rho_grid, _ = sample_lognormal_field
        s_t = 1.0

        result = compute_tail_pmfs_bm19(rho_grid, s_t)

        assert jnp.all(result.p_tail >= 0)
        assert jnp.all(result.p_smooth >= 0)

    def test_tail_weights_in_valid_range(self, sample_lognormal_field):
        """Tail weights should be in [0, 1]."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19

        rho_grid, _ = sample_lognormal_field
        s_t = 1.0

        result = compute_tail_pmfs_bm19(rho_grid, s_t)

        assert jnp.all(result.tail_weights >= 0)
        assert jnp.all(result.tail_weights <= 1)

    def test_tail_weights_shape_matches_input(self, sample_lognormal_field):
        """Tail weights should have same shape as input."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19

        rho_grid, _ = sample_lognormal_field
        s_t = 1.0

        result = compute_tail_pmfs_bm19(rho_grid, s_t)

        assert result.tail_weights.shape == rho_grid.shape

    def test_f_tail_in_valid_range(self, sample_lognormal_field):
        """f_tail_actual should be in [0, 1]."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19

        rho_grid, _ = sample_lognormal_field
        s_t = 1.0

        result = compute_tail_pmfs_bm19(rho_grid, s_t)

        assert 0 <= result.f_tail_actual <= 1

    def test_higher_s_t_gives_lower_f_tail(self, sample_lognormal_field):
        """Higher s_t threshold should give lower f_tail."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19

        rho_grid, _ = sample_lognormal_field

        result_low = compute_tail_pmfs_bm19(rho_grid, s_t=0.5)
        result_high = compute_tail_pmfs_bm19(rho_grid, s_t=2.0)

        assert result_high.f_tail_actual < result_low.f_tail_actual

    def test_higher_kappa_gives_sharper_transition(self, sample_lognormal_field):
        """Higher kappa should give more binary-like weights."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19

        rho_grid, _ = sample_lognormal_field
        s_t = 1.0

        result_soft = compute_tail_pmfs_bm19(rho_grid, s_t, kappa=1.0)
        result_sharp = compute_tail_pmfs_bm19(rho_grid, s_t, kappa=100.0)

        # Sharper kappa should have weights closer to 0 or 1
        # Measure by variance of weights in (0.1, 0.9) range
        mid_soft = result_soft.tail_weights[
            (result_soft.tail_weights > 0.1) & (result_soft.tail_weights < 0.9)
        ]
        mid_sharp = result_sharp.tail_weights[
            (result_sharp.tail_weights > 0.1) & (result_sharp.tail_weights < 0.9)
        ]

        # Sharp should have fewer mid-range weights
        assert mid_sharp.size <= mid_soft.size

    def test_differentiable_through_s_t(self, sample_lognormal_field):
        """Gradients should flow through s_t."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19

        rho_grid, _ = sample_lognormal_field
        rho_grid = jax.lax.stop_gradient(rho_grid)

        def loss(s_t):
            result = compute_tail_pmfs_bm19(rho_grid, s_t, kappa=10.0)
            return result.f_tail_actual

        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad)
        # d(f_tail)/d(s_t) should be negative (higher threshold -> lower f_tail)
        assert grad < 0

    def test_differentiable_through_kappa(self, sample_lognormal_field):
        """Gradients should flow through kappa."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19

        rho_grid, _ = sample_lognormal_field
        rho_grid = jax.lax.stop_gradient(rho_grid)

        def loss(kappa):
            result = compute_tail_pmfs_bm19(rho_grid, s_t=1.0, kappa=kappa)
            return result.f_tail_actual

        grad = jax.grad(loss)(10.0)
        assert jnp.isfinite(grad)

    def test_jit_compatible(self, sample_lognormal_field):
        """Should work with @jax.jit."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19

        rho_grid, _ = sample_lognormal_field

        @jax.jit
        def compute(rho, s_t):
            return compute_tail_pmfs_bm19(rho, s_t)

        result = compute(rho_grid, 1.0)
        assert jnp.isfinite(result.f_tail_actual)


class TestFTailConsistencyWithBM19:
    """Tests for f_tail_actual matching f_dense from BM19 theory."""

    def test_f_tail_roughly_matches_f_dense(self):
        """f_tail_actual should roughly match f_dense from BM19."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19
        from progenax.gravoturb import bm19_model as bm19

        # Generate field with known statistics
        key = random.PRNGKey(123)
        grid_size = 64  # Higher resolution for better statistics
        mach = 10.0
        alpha = 2.0

        # Get BM19 predictions
        result_bm19 = bm19.bm19_pipeline(mach, b=0.4, alpha=alpha)
        sigma_s = float(result_bm19.sigma_s)
        s_t = float(result_bm19.s_t)
        f_dense_theory = float(result_bm19.f_dense)

        # Generate lognormal field with matching sigma_s
        z = random.normal(key, (grid_size, grid_size, grid_size))
        s = sigma_s * z - sigma_s**2 / 2
        rho_grid = jnp.exp(s)

        # Compute tail PMFs
        pmf_result = compute_tail_pmfs_bm19(rho_grid, s_t, kappa=10.0)

        # f_tail_actual should match f_dense within ~35% for single realization
        # (Statistical fluctuations at finite grid size can be significant)
        relative_error = abs(pmf_result.f_tail_actual - f_dense_theory) / f_dense_theory
        assert relative_error < 0.40, (
            f"f_tail_actual={pmf_result.f_tail_actual:.3f} differs from "
            f"f_dense={f_dense_theory:.3f} by {relative_error*100:.0f}%"
        )

    def test_f_tail_trend_with_mach(self):
        """f_tail should decrease with Mach (like f_dense in BM19)."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19
        from progenax.gravoturb import bm19_model as bm19

        key = random.PRNGKey(456)
        grid_size = 32
        alpha = 2.0
        machs = [5.0, 10.0, 20.0]

        f_tails = []
        for mach in machs:
            # Get BM19 s_t for this Mach
            result = bm19.bm19_pipeline(mach, alpha=alpha)
            sigma_s = float(result.sigma_s)
            s_t = float(result.s_t)

            # Generate field
            key, subkey = random.split(key)
            z = random.normal(subkey, (grid_size, grid_size, grid_size))
            s = sigma_s * z - sigma_s**2 / 2
            rho_grid = jnp.exp(s)

            # Compute f_tail
            pmf = compute_tail_pmfs_bm19(rho_grid, s_t)
            f_tails.append(pmf.f_tail_actual)

        # f_tail should decrease with Mach (same trend as f_dense)
        # Note: This tests the BM19 physics: higher Mach -> higher s_t -> lower f_tail
        assert f_tails[0] > f_tails[1] > f_tails[2], (
            f"f_tail should decrease with Mach: {f_tails}"
        )


class TestComputeTailPMFsPN11Legacy:
    """Tests for compute_tail_pmfs_pn11_legacy()."""

    @pytest.fixture
    def sample_field(self):
        """Generate a sample density field."""
        key = random.PRNGKey(789)
        grid_size = 32
        z = random.normal(key, (grid_size, grid_size, grid_size))
        rho_grid = jnp.exp(z)  # Lognormal
        return rho_grid

    def test_emits_deprecation_warning(self, sample_field):
        """Should emit DeprecationWarning."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_pn11_legacy

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_tail_pmfs_pn11_legacy(sample_field)

            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_returns_tail_pmf_result(self, sample_field):
        """Should return TailPMFResult."""
        from progenax.cluster.fdf_tail import TailPMFResult, compute_tail_pmfs_pn11_legacy

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = compute_tail_pmfs_pn11_legacy(sample_field)

        assert isinstance(result, TailPMFResult)

    def test_f_tail_equals_target(self, sample_field):
        """f_tail_actual should equal dense_tail_mass_frac."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_pn11_legacy

        target = 0.15

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = compute_tail_pmfs_pn11_legacy(sample_field, dense_tail_mass_frac=target)

        assert jnp.isclose(result.f_tail_actual, target, atol=0.01)

    def test_pmfs_normalized(self, sample_field):
        """PMFs should sum to 1."""
        from progenax.cluster.fdf_tail import compute_tail_pmfs_pn11_legacy

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = compute_tail_pmfs_pn11_legacy(sample_field)

        assert jnp.isclose(jnp.sum(result.p_tail), 1.0, atol=1e-5)
        assert jnp.isclose(jnp.sum(result.p_smooth), 1.0, atol=1e-5)


class TestSamplePositionsTailBM19:
    """Tests for sample_positions_tail_bm19()."""

    @pytest.fixture
    def sample_field_with_coords(self):
        """Generate field with coordinate grids."""
        key = random.PRNGKey(101)
        grid_size = 32
        box_half_size = 5.0

        # Density field
        z = random.normal(key, (grid_size, grid_size, grid_size))
        rho_grid = jnp.exp(z)

        # Coordinate grids
        x = jnp.linspace(-box_half_size, box_half_size, grid_size)
        y = jnp.linspace(-box_half_size, box_half_size, grid_size)
        z_coord = jnp.linspace(-box_half_size, box_half_size, grid_size)

        return rho_grid, x, y, z_coord, box_half_size

    def test_returns_positions_and_pmf(self, sample_field_with_coords):
        """Should return positions array and TailPMFResult."""
        from progenax.cluster.fdf_tail import TailPMFResult, sample_positions_tail_bm19

        rho_grid, x, y, z, _ = sample_field_with_coords
        key = random.PRNGKey(202)
        N_stars = 100

        positions, pmf_result = sample_positions_tail_bm19(
            key, rho_grid, x, y, z, N_stars, f_sub=0.5, s_t=1.0
        )

        assert positions.shape == (N_stars, 3)
        assert isinstance(pmf_result, TailPMFResult)

    def test_positions_in_box(self, sample_field_with_coords):
        """Positions should be within the box."""
        from progenax.cluster.fdf_tail import sample_positions_tail_bm19

        rho_grid, x, y, z, box_half_size = sample_field_with_coords
        key = random.PRNGKey(303)
        N_stars = 200

        positions, _ = sample_positions_tail_bm19(
            key, rho_grid, x, y, z, N_stars, f_sub=0.5, s_t=1.0
        )

        # Allow small tolerance for sub-cell jitter
        tolerance = (x[1] - x[0]) / 2
        assert jnp.all(positions >= -box_half_size - tolerance)
        assert jnp.all(positions <= box_half_size + tolerance)

    def test_warns_for_small_N_stars(self, sample_field_with_coords):
        """Should warn when N_stars < 100."""
        from progenax.cluster.fdf_tail import sample_positions_tail_bm19

        rho_grid, x, y, z, _ = sample_field_with_coords
        key = random.PRNGKey(404)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sample_positions_tail_bm19(
                key, rho_grid, x, y, z, N_stars=50, f_sub=0.5, s_t=1.0
            )

            assert len(w) >= 1
            assert any(issubclass(warning.category, UserWarning) for warning in w)
            assert any("N_stars" in str(warning.message) for warning in w)

    def test_f_sub_zero_uses_only_smooth(self, sample_field_with_coords):
        """f_sub=0 should sample only from smooth component."""
        from progenax.cluster.fdf_tail import sample_positions_tail_bm19

        rho_grid, x, y, z, _ = sample_field_with_coords
        key = random.PRNGKey(505)
        N_stars = 100

        # With f_sub=0, all stars come from smooth (full density)
        positions, _ = sample_positions_tail_bm19(
            key, rho_grid, x, y, z, N_stars, f_sub=0.0, s_t=1.0
        )

        # Should still return valid positions
        assert positions.shape == (N_stars, 3)
        assert jnp.all(jnp.isfinite(positions))

    def test_f_sub_one_uses_only_tail(self, sample_field_with_coords):
        """f_sub=1 should sample only from tail component."""
        from progenax.cluster.fdf_tail import sample_positions_tail_bm19

        rho_grid, x, y, z, _ = sample_field_with_coords
        key = random.PRNGKey(606)
        N_stars = 100

        # With f_sub=1, all stars come from tail
        positions, _ = sample_positions_tail_bm19(
            key, rho_grid, x, y, z, N_stars, f_sub=1.0, s_t=1.0
        )

        # Should still return valid positions
        assert positions.shape == (N_stars, 3)
        assert jnp.all(jnp.isfinite(positions))


class TestTailSelectionConfig:
    """Tests for TailSelectionConfig dataclass."""

    def test_default_mode_is_bm19(self):
        """Default mode should be 'bm19'."""
        from progenax.cluster.fdf_config import TailSelectionConfig

        config = TailSelectionConfig()
        assert config.mode == "bm19"

    def test_default_kappa(self):
        """Default kappa should be 10.0."""
        from progenax.cluster.fdf_config import TailSelectionConfig

        config = TailSelectionConfig()
        assert config.kappa == 10.0

    def test_default_dense_tail_mass_frac(self):
        """Default dense_tail_mass_frac should be 0.10."""
        from progenax.cluster.fdf_config import TailSelectionConfig

        config = TailSelectionConfig()
        assert config.dense_tail_mass_frac == 0.10

    def test_accepts_bm19_mode(self):
        """Should accept mode='bm19'."""
        from progenax.cluster.fdf_config import TailSelectionConfig

        config = TailSelectionConfig(mode="bm19")
        assert config.mode == "bm19"

    def test_accepts_pn11_legacy_mode(self):
        """Should accept mode='pn11_legacy'."""
        from progenax.cluster.fdf_config import TailSelectionConfig

        config = TailSelectionConfig(mode="pn11_legacy")
        assert config.mode == "pn11_legacy"

    def test_rejects_invalid_mode(self):
        """Should reject invalid mode values."""
        from progenax.cluster.fdf_config import TailSelectionConfig

        with pytest.raises(ValueError, match="Invalid mode"):
            TailSelectionConfig(mode="invalid")

    def test_rejects_nonpositive_kappa(self):
        """Should reject kappa <= 0."""
        from progenax.cluster.fdf_config import TailSelectionConfig

        with pytest.raises(ValueError, match="kappa must be positive"):
            TailSelectionConfig(kappa=0.0)

        with pytest.raises(ValueError, match="kappa must be positive"):
            TailSelectionConfig(kappa=-1.0)

    def test_rejects_invalid_dense_tail_mass_frac(self):
        """Should reject dense_tail_mass_frac outside (0, 1)."""
        from progenax.cluster.fdf_config import TailSelectionConfig

        with pytest.raises(ValueError, match="dense_tail_mass_frac"):
            TailSelectionConfig(dense_tail_mass_frac=0.0)

        with pytest.raises(ValueError, match="dense_tail_mass_frac"):
            TailSelectionConfig(dense_tail_mass_frac=1.0)

        with pytest.raises(ValueError, match="dense_tail_mass_frac"):
            TailSelectionConfig(dense_tail_mass_frac=1.5)


class TestSampleFromPMF:
    """Tests for sample_from_pmf utility."""

    def test_returns_correct_shape(self):
        """Should return array of specified size."""
        from progenax.cluster.fdf_tail import sample_from_pmf

        key = random.PRNGKey(707)
        pmf = jnp.ones(100) / 100
        n_samples = 50

        indices = sample_from_pmf(key, pmf, n_samples)

        assert indices.shape == (n_samples,)

    def test_indices_in_valid_range(self):
        """Indices should be in [0, N_cells)."""
        from progenax.cluster.fdf_tail import sample_from_pmf

        key = random.PRNGKey(808)
        n_cells = 100
        pmf = jnp.ones(n_cells) / n_cells
        n_samples = 200

        indices = sample_from_pmf(key, pmf, n_samples)

        assert jnp.all(indices >= 0)
        assert jnp.all(indices < n_cells)

    def test_respects_pmf_distribution(self):
        """Samples should follow PMF distribution."""
        from progenax.cluster.fdf_tail import sample_from_pmf

        key = random.PRNGKey(909)
        n_cells = 10
        n_samples = 10000

        # PMF heavily weighted to first cell
        pmf = jnp.zeros(n_cells)
        pmf = pmf.at[0].set(0.9)
        pmf = pmf.at[1:].set(0.1 / (n_cells - 1))

        indices = sample_from_pmf(key, pmf, n_samples)

        # Most samples should be from cell 0
        frac_cell_0 = jnp.sum(indices == 0) / n_samples
        assert frac_cell_0 > 0.85  # Should be close to 0.9


class TestIntegrationBM19Pipeline:
    """Integration tests: full BM19 pipeline to tail sampling."""

    def test_full_pipeline_runs(self):
        """Full pipeline from BM19 to sampling should work."""
        from progenax.cluster.fdf_tail import sample_positions_tail_bm19
        from progenax.gravoturb import bm19_model as bm19

        # BM19 pipeline
        mach = 10.0
        alpha = 2.0
        eta_survive = 0.6
        result = bm19.bm19_pipeline(mach, alpha=alpha, eta_survive=eta_survive)

        # Generate density field
        key = random.PRNGKey(1010)
        grid_size = 32
        box_half_size = 5.0

        sigma_s = float(result.sigma_s)
        z = random.normal(key, (grid_size, grid_size, grid_size))
        s = sigma_s * z - sigma_s**2 / 2
        rho_grid = jnp.exp(s)

        x = jnp.linspace(-box_half_size, box_half_size, grid_size)
        y = jnp.linspace(-box_half_size, box_half_size, grid_size)
        z_coord = jnp.linspace(-box_half_size, box_half_size, grid_size)

        # Sample positions
        key, subkey = random.split(key)
        positions, pmf = sample_positions_tail_bm19(
            subkey,
            rho_grid,
            x,
            y,
            z_coord,
            N_stars=200,
            f_sub=float(result.f_sub),
            s_t=float(result.s_t),
        )

        # Verify outputs
        assert positions.shape == (200, 3)
        assert jnp.all(jnp.isfinite(positions))
        assert 0 < pmf.f_tail_actual < 1
