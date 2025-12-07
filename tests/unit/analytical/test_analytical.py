"""Tests for analytical initial conditions."""

import jax
import jax.numpy as jnp
import pytest


class TestAnalyticalIC:
    """Test AnalyticalIC dataclass."""

    def test_analytical_ic_importable(self):
        from progenax.analytical import AnalyticalIC
        assert AnalyticalIC is not None


class TestTwoBodyKepler:
    """Test two-body Keplerian orbits."""

    def test_two_body_kepler_importable(self):
        from progenax.analytical import two_body_kepler
        assert callable(two_body_kepler)

    def test_circular_orbit_com_at_origin(self):
        """COM should be at origin."""
        from progenax.analytical import two_body_kepler
        G = 1.0
        ic = two_body_kepler(M1=1.0, M2=1.0, a=1.0, e=0.0, G=G)
        com = jnp.sum(ic.positions * ic.masses[:, None], axis=0) / jnp.sum(ic.masses)
        assert jnp.allclose(com, 0.0, atol=1e-10)

    def test_circular_orbit_total_momentum_zero(self):
        """Total momentum should be zero in COM frame."""
        from progenax.analytical import two_body_kepler
        G = 1.0
        ic = two_body_kepler(M1=1.0, M2=1.0, a=1.0, e=0.0, G=G)
        p_total = jnp.sum(ic.masses[:, None] * ic.velocities, axis=0)
        assert jnp.allclose(p_total, 0.0, atol=1e-10)

    def test_period_circular_orbit(self):
        """Period should be T = 2π√(a³/GM)."""
        from progenax.analytical import two_body_period
        G = 39.478  # Binary units (AU³/Msun/yr²)
        T = two_body_period(M1=1.0, M2=0.0, a=1.0, G=G)  # Approximately 1 yr
        expected = 2.0 * jnp.pi * jnp.sqrt(1.0**3 / (G * 1.0))
        assert jnp.abs(T - expected) < 0.01


class TestThreeBodyFigureEight:
    """Test figure-8 three-body orbit."""

    def test_figure_eight_importable(self):
        from progenax.analytical import three_body_figure_eight
        assert callable(three_body_figure_eight)

    def test_figure_eight_com_zero(self):
        """Figure-8 should have COM at origin."""
        from progenax.analytical import three_body_figure_eight
        G = 1.0
        ic = three_body_figure_eight(mass=1.0, scale=1.0, G=G)
        com = jnp.sum(ic.positions * ic.masses[:, None], axis=0) / jnp.sum(ic.masses)
        # Relaxed tolerance due to finite precision in published initial conditions
        assert jnp.allclose(com, 0.0, atol=1e-7)

    def test_figure_eight_planar(self):
        """Figure-8 should be in x-y plane (z=0)."""
        from progenax.analytical import three_body_figure_eight
        G = 1.0
        ic = three_body_figure_eight(mass=1.0, G=G)
        assert jnp.allclose(ic.positions[:, 2], 0.0, atol=1e-10)
        assert jnp.allclose(ic.velocities[:, 2], 0.0, atol=1e-10)


class TestSolarSystemData:
    """Test solar system planetary data."""

    def test_get_planet_importable(self):
        from progenax.analytical import get_planet
        assert callable(get_planet)

    def test_get_planet_earth(self):
        """Earth should have a=1 AU."""
        from progenax.analytical import get_planet
        earth = get_planet("Earth")
        assert earth["a"] == pytest.approx(1.0, abs=0.01)

    def test_solar_system_planets_data(self):
        """Should have all 8 planets."""
        from progenax.analytical import SOLAR_SYSTEM_PLANETS
        assert len(SOLAR_SYSTEM_PLANETS) == 8


class TestSolarSystemFunctions:
    """Test solar system IC generators."""

    def test_earth_sun_2body(self):
        from progenax.analytical import earth_sun_2body
        G = 39.478  # Binary units
        ic = earth_sun_2body(G=G)
        assert ic.positions.shape == (2, 3)
        assert ic.masses.shape == (2,)

    def test_solar_system_inner_4(self):
        from progenax.analytical import solar_system_inner_4
        G = 39.478
        ic = solar_system_inner_4(G=G)
        assert ic.positions.shape == (5, 3)  # Sun + 4 planets

    def test_solar_system_full(self):
        from progenax.analytical import solar_system_full
        G = 39.478
        ic = solar_system_full(G=G)
        assert ic.positions.shape == (9, 3)  # Sun + 8 planets
