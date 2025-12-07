"""Tests for tidal physics utilities."""

import jax
import jax.numpy as jnp
import pytest

from progenax.tidal import (
    jacobi_radius,
    jacobi_radius_isothermal,
    apply_tidal_truncation,
    fill_factor_to_r_h,
)


class TestJacobiRadius:
    """Tests for Jacobi/tidal radius calculation."""

    def test_jacobi_radius_formula(self):
        """Jacobi radius follows r_J = R * (M_cl / 3M_gal)^(1/3)."""
        M_cluster = 1e4  # 10^4 Msun cluster
        M_galaxy = 1e11  # 10^11 Msun galaxy
        R_galactic = 8000.0  # 8 kpc from galactic center

        r_J = jacobi_radius(M_cluster, M_galaxy, R_galactic)

        # Expected: r_J = R * (M_cl / 3M_gal)^(1/3)
        expected = R_galactic * (M_cluster / (3.0 * M_galaxy)) ** (1.0/3.0)
        assert jnp.abs(r_J - expected) / expected < 1e-10

    def test_jacobi_scales_with_cluster_mass(self):
        """More massive clusters have larger tidal radii."""
        M_galaxy = 1e11
        R_galactic = 8000.0

        r_J_small = jacobi_radius(1e3, M_galaxy, R_galactic)
        r_J_large = jacobi_radius(1e5, M_galaxy, R_galactic)

        assert r_J_large > r_J_small
        # Should scale as M^(1/3)
        ratio = r_J_large / r_J_small
        expected_ratio = (1e5 / 1e3) ** (1.0/3.0)
        assert jnp.abs(ratio - expected_ratio) < 0.01

    def test_jacobi_scales_with_distance(self):
        """Clusters further from galactic center have larger tidal radii."""
        M_cluster = 1e4
        M_galaxy = 1e11

        r_J_inner = jacobi_radius(M_cluster, M_galaxy, 4000.0)
        r_J_outer = jacobi_radius(M_cluster, M_galaxy, 8000.0)

        assert r_J_outer > r_J_inner


class TestJacobiRadiusIsothermal:
    """Tests for isothermal halo Jacobi radius."""

    def test_formula(self):
        """Jacobi radius scales correctly for isothermal halo."""
        M_cluster = 1e4
        V_circ = 220.0  # km/s
        R_galactic = 8000.0  # pc
        G = 0.00450  # stellar units

        r_J = jacobi_radius_isothermal(M_cluster, V_circ, R_galactic, G)

        # Should scale as M^(1/3)
        r_J_2 = jacobi_radius_isothermal(8 * M_cluster, V_circ, R_galactic, G)
        assert jnp.allclose(r_J_2 / r_J, 2.0, rtol=0.01)


class TestTidalTruncation:
    """Tests for tidal truncation of particle distributions."""

    def test_removes_particles_beyond_r_t(self):
        """Particles beyond r_t are removed."""
        N = 1000
        key = jax.random.PRNGKey(42)
        positions = jax.random.normal(key, (N, 3)) * 5.0
        velocities = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)

        r_t = 3.0
        pos_out, vel_out, mass_out, mask = apply_tidal_truncation(
            positions, velocities, masses, r_t
        )

        radii_out = jnp.linalg.norm(pos_out, axis=1)
        assert jnp.all(radii_out <= r_t + 1e-10)

    def test_preserves_particles_within_r_t(self):
        """Particles within r_t are preserved."""
        N = 100
        # All particles well within tidal radius
        positions = jax.random.normal(jax.random.PRNGKey(42), (N, 3)) * 0.5
        velocities = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)

        r_t = 3.0
        pos_out, vel_out, mass_out, mask = apply_tidal_truncation(
            positions, velocities, masses, r_t
        )

        assert jnp.sum(mask) == N  # All preserved
        assert jnp.allclose(pos_out, positions)

    def test_returns_mask(self):
        """Returns boolean mask indicating kept particles."""
        positions = jnp.array([
            [1.0, 0.0, 0.0],  # r = 1, keep
            [5.0, 0.0, 0.0],  # r = 5, remove
            [0.0, 2.0, 0.0],  # r = 2, keep
        ])
        velocities = jnp.zeros((3, 3))
        masses = jnp.ones(3)

        r_t = 3.0
        _, _, _, mask = apply_tidal_truncation(positions, velocities, masses, r_t)

        expected_mask = jnp.array([True, False, True])
        assert jnp.all(mask == expected_mask)


class TestFillFactor:
    """Tests for fill factor to half-mass radius conversion."""

    def test_fill_factor_formula(self):
        """r_h = fill_factor * r_J."""
        r_J = 10.0
        fill_factor = 0.2

        r_h = fill_factor_to_r_h(fill_factor, r_J)

        assert jnp.abs(r_h - 2.0) < 1e-10

    def test_fill_factor_bounds(self):
        """Fill factor should be in (0, 1)."""
        r_J = 10.0

        r_h_low = fill_factor_to_r_h(0.1, r_J)
        r_h_high = fill_factor_to_r_h(0.5, r_J)

        assert r_h_low < r_h_high
        assert r_h_low > 0
        assert r_h_high < r_J
