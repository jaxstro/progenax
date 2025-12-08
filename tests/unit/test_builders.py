"""Tests for progenax builders - physics only."""

import jax
import jax.numpy as jnp
import pytest


class TestStellarRadii:
    """Test stellar radii mass-radius relation."""

    def test_stellar_radii_solar_mass(self):
        """1 Msun star should have ~1 Rsun radius (R ∝ M^0.8)."""
        from progenax.builders import compute_stellar_radii
        masses = jnp.array([1.0])
        radii = compute_stellar_radii(masses)
        assert jnp.abs(radii[0] - 1.0) < 0.1

    def test_stellar_radii_massive_star(self):
        """10 Msun star should have R ~ 10^0.8 ~ 6.3 Rsun."""
        from progenax.builders import compute_stellar_radii
        masses = jnp.array([10.0])
        radii = compute_stellar_radii(masses)
        expected = 10.0**0.8
        assert jnp.abs(radii[0] - expected) < 0.5

    def test_stellar_radii_brown_dwarf(self):
        """Brown dwarf (0.01 Msun) should have R ~ 0.1 Rsun."""
        from progenax.builders import compute_stellar_radii
        masses = jnp.array([0.01])
        radii = compute_stellar_radii(masses)
        assert 0.05 < radii[0] < 0.15


class TestToCOMFrame:
    """Test center-of-mass frame transformation."""

    def test_com_is_zero_after_transform(self):
        """COM position and velocity should be zero after transform."""
        from progenax.builders import to_com_frame

        positions = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        velocities = jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        masses = jnp.array([1.0, 1.0, 2.0])

        pos_com, vel_com = to_com_frame(positions, velocities, masses)

        com_pos = jnp.sum(pos_com * masses[:, None], axis=0) / jnp.sum(masses)
        com_vel = jnp.sum(vel_com * masses[:, None], axis=0) / jnp.sum(masses)

        assert jnp.allclose(com_pos, 0.0, atol=1e-10)
        assert jnp.allclose(com_vel, 0.0, atol=1e-10)


class TestVirialScale:
    """Test virial scaling physics."""

    def test_virial_ratio_is_target(self):
        """After scaling, Q = 2T/|V| should equal target."""
        from progenax.builders import virial_scale, compute_kinetic_energy, compute_potential_energy

        positions = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        velocities = jnp.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
        masses = jnp.array([1.0, 1.0])
        G = 1.0

        vel_scaled = virial_scale(positions, velocities, masses, Q_target=1.0, G=G)

        T = compute_kinetic_energy(vel_scaled, masses)
        V = compute_potential_energy(positions, masses, G=G)
        Q = 2.0 * T / jnp.abs(V)

        assert jnp.abs(Q - 1.0) < 0.01

    def test_q_one_equilibrium(self):
        """Q = 1.0 gives virial equilibrium (2T + V = 0)."""
        from progenax.builders import virial_scale, compute_kinetic_energy, compute_potential_energy

        positions = jnp.array([
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        ])
        velocities = jnp.array([
            [0.0, 0.1, 0.0], [0.0, -0.1, 0.0],
            [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
        ])
        masses = jnp.ones(4)
        G = 1.0

        vel_scaled = virial_scale(positions, velocities, masses, Q_target=1.0, G=G)

        T = compute_kinetic_energy(vel_scaled, masses)
        V = compute_potential_energy(positions, masses, G=G)
        Q = 2.0 * T / jnp.abs(V)

        assert jnp.abs(Q - 1.0) < 0.02

    def test_q_less_than_one_collapsing(self):
        """Q < 1.0 gives sub-virial (cold) system."""
        from progenax.builders import virial_scale, compute_kinetic_energy, compute_potential_energy

        positions = jnp.array([
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        ])
        velocities = jnp.array([
            [0.0, 0.1, 0.0], [0.0, -0.1, 0.0],
            [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
        ])
        masses = jnp.ones(4)
        G = 1.0

        vel_scaled = virial_scale(positions, velocities, masses, Q_target=0.5, G=G)

        T = compute_kinetic_energy(vel_scaled, masses)
        V = compute_potential_energy(positions, masses, G=G)
        Q = 2.0 * T / jnp.abs(V)

        assert Q < 1.0
        assert jnp.abs(Q - 0.5) < 0.05


class TestBuildSpatialIC:
    """Test main IC builder."""

    def test_build_with_plummer(self):
        """Build IC with Plummer profile produces valid output."""
        from progenax.builders import build_spatial_ic, ICResult
        from progenax.profiles import PlummerProfile
        from progenax.kinematics import PlummerVelocityDF

        masses = jnp.ones(100)
        profile = PlummerProfile(r_h=1.0)
        velocity_df = PlummerVelocityDF(r_h=1.0)
        key = jax.random.PRNGKey(42)
        G = 1.0

        result = build_spatial_ic(
            profile=profile,
            masses=masses,
            velocity_df=velocity_df,
            Q=1.0,
            key=key,
            G=G,
        )

        assert isinstance(result, ICResult)
        assert jnp.all(jnp.isfinite(result.positions))
        assert jnp.all(jnp.isfinite(result.velocities))

    def test_ic_in_com_frame(self):
        """IC should be in COM frame (zero mean position/velocity)."""
        from progenax.builders import build_spatial_ic
        from progenax.profiles import PlummerProfile
        from progenax.kinematics import PlummerVelocityDF

        masses = jnp.ones(100)
        profile = PlummerProfile(r_h=1.0)
        velocity_df = PlummerVelocityDF(r_h=1.0)
        key = jax.random.PRNGKey(42)
        G = 1.0

        result = build_spatial_ic(
            profile=profile,
            masses=masses,
            velocity_df=velocity_df,
            Q=1.0,
            key=key,
            G=G,
        )

        com_pos = jnp.sum(result.positions * masses[:, None], axis=0) / jnp.sum(masses)
        assert jnp.allclose(com_pos, 0.0, atol=1e-6)
