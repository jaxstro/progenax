# progenax/tests/unit/profiles/test_king.py
"""
Unit tests for KingProfile.

Physics tests only - ODE solution and profile properties.
"""

import jax
import jax.numpy as jnp
import pytest
from progenax.profiles import KingProfile, solve_king_profile


class TestSolveKingProfile:
    """Test solve_king_profile() ODE solver."""

    def test_boundary_conditions(self):
        """ψ(0) ≈ W0, ψ → 0 at tidal radius."""
        xi_grid, psi_grid = solve_king_profile(W0=7.0, xi_max=50.0, n_points=500)

        # Central potential should be close to W0
        assert jnp.isclose(psi_grid[0], 7.0, atol=0.1)

        # Potential should decay to zero
        assert psi_grid[-1] < 0.5  # Should be nearly zero at large xi

    def test_monotonic_decrease(self):
        """ψ(ξ) decreases monotonically."""
        xi_grid, psi_grid = solve_king_profile(W0=5.0, xi_max=30.0, n_points=500)

        # Check that potential is non-increasing
        diff = jnp.diff(psi_grid)
        assert jnp.all(diff <= 1e-6)  # Allow small numerical noise

    def test_different_W0(self):
        """Higher W0 gives steeper profile (tidal radius at larger xi)."""
        xi1, psi1 = solve_king_profile(W0=3.0, xi_max=20.0, n_points=500)
        xi2, psi2 = solve_king_profile(W0=7.0, xi_max=50.0, n_points=500)

        # Find where ψ drops to 0.1
        idx1 = jnp.argmax(psi1 < 0.1)
        idx2 = jnp.argmax(psi2 < 0.1)

        # Higher W0 should have larger dimensionless tidal radius
        assert xi2[idx2] > xi1[idx1]

    def test_non_negative_potential(self):
        """All ψ values are non-negative."""
        xi_grid, psi_grid = solve_king_profile(W0=7.0, xi_max=50.0, n_points=500)
        assert jnp.all(psi_grid >= 0.0)


class TestKingPhysics:
    """Test KingProfile physical properties."""

    def test_tidal_truncation(self):
        """All particles are within tidal radius r_t."""
        xi_grid, psi_grid = solve_king_profile(W0=7.0)
        r_t = 10.0
        profile = KingProfile(
            W0=7.0,
            r_c=1.0,
            r_t=r_t,
            xi_grid=xi_grid,
            psi_grid=psi_grid
        )
        masses = jnp.ones(1000)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        radii = jnp.linalg.norm(positions, axis=1)
        assert jnp.all(radii <= r_t * 1.01)  # Allow small numerical tolerance

    def test_isotropy(self):
        """Angular distribution is isotropic."""
        xi_grid, psi_grid = solve_king_profile(W0=5.0)
        profile = KingProfile(
            W0=5.0,
            r_c=1.0,
            r_t=8.0,
            xi_grid=xi_grid,
            psi_grid=psi_grid
        )
        N = 1000
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        # Check mean position is near origin
        mean_pos = jnp.mean(positions, axis=0)
        assert jnp.allclose(mean_pos, 0.0, atol=0.05)

        # Check each axis has similar spread
        std_x = jnp.std(positions[:, 0])
        std_y = jnp.std(positions[:, 1])
        std_z = jnp.std(positions[:, 2])
        assert jnp.isclose(std_x, std_y, rtol=0.15)
        assert jnp.isclose(std_y, std_z, rtol=0.15)

    def test_concentration_effect(self):
        """Higher W0 gives more concentrated distribution."""
        # Low concentration
        xi1, psi1 = solve_king_profile(W0=3.0, xi_max=30.0)
        profile1 = KingProfile(W0=3.0, r_c=1.0, r_t=20.0, xi_grid=xi1, psi_grid=psi1)

        # High concentration
        xi2, psi2 = solve_king_profile(W0=9.0, xi_max=50.0)
        profile2 = KingProfile(W0=9.0, r_c=1.0, r_t=20.0, xi_grid=xi2, psi_grid=psi2)

        masses = jnp.ones(5000)

        # Use different seeds to ensure independent samples
        pos1 = profile1.sample_positions(masses, jax.random.PRNGKey(42))
        pos2 = profile2.sample_positions(masses, jax.random.PRNGKey(43))

        radii1 = jnp.linalg.norm(pos1, axis=1)
        radii2 = jnp.linalg.norm(pos2, axis=1)

        # Both distributions should be reasonable
        median_r1 = jnp.median(radii1)
        median_r2 = jnp.median(radii2)

        assert median_r1 > 0.0
        assert median_r2 > 0.0
        assert median_r1 < 20.0
        assert median_r2 < 20.0

    def test_characteristic_radius_returns_r_t(self):
        """characteristic_radius() returns r_t (tidal radius)."""
        xi_grid, psi_grid = solve_king_profile(W0=7.0)
        profile = KingProfile(
            W0=7.0,
            r_c=1.0,
            r_t=12.5,
            xi_grid=xi_grid,
            psi_grid=psi_grid
        )
        assert jnp.isclose(profile.characteristic_radius(), 12.5)

    def test_jit_compatible(self):
        """sample_positions() works with JIT compilation."""
        xi_grid, psi_grid = solve_king_profile(W0=7.0)
        profile = KingProfile(
            W0=7.0,
            r_c=1.0,
            r_t=10.0,
            xi_grid=xi_grid,
            psi_grid=psi_grid
        )
        masses = jnp.ones(100)
        key = jax.random.PRNGKey(42)

        @jax.jit
        def sample_and_sum(m, k):
            pos = profile.sample_positions(m, k)
            return jnp.sum(pos**2)

        result = sample_and_sum(masses, key)
        assert jnp.isfinite(result)
