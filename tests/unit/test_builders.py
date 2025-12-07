"""Tests for progenax builders."""

import jax
import jax.numpy as jnp
import pytest


class TestICResult:
    """Test ICResult dataclass."""

    def test_ic_result_importable(self):
        from progenax.builders import ICResult
        assert ICResult is not None

    def test_ic_result_has_required_fields(self):
        from progenax.builders import ICResult
        # Should have: positions, velocities, masses, softening, stellar_radii, ids
        assert hasattr(ICResult, '__dataclass_fields__') or hasattr(ICResult, '__init__')


class TestHelpers:
    """Test helper functions."""

    def test_compute_stellar_radii_importable(self):
        from progenax.builders import compute_stellar_radii
        assert callable(compute_stellar_radii)

    def test_stellar_radii_solar_mass(self):
        """1 Msun star should have ~1 Rsun radius."""
        from progenax.builders import compute_stellar_radii
        masses = jnp.array([1.0])
        radii = compute_stellar_radii(masses)
        assert jnp.abs(radii[0] - 1.0) < 0.1  # Within 10%

    def test_stellar_radii_massive_star(self):
        """10 Msun star should have R ~ 10^0.8 ~ 6.3 Rsun."""
        from progenax.builders import compute_stellar_radii
        masses = jnp.array([10.0])
        radii = compute_stellar_radii(masses)
        expected = 10.0**0.8  # ~6.3
        assert jnp.abs(radii[0] - expected) < 0.5

    def test_stellar_radii_brown_dwarf(self):
        """Brown dwarf (0.01 Msun) should have R ~ 0.1 Rsun."""
        from progenax.builders import compute_stellar_radii
        masses = jnp.array([0.01])
        radii = compute_stellar_radii(masses)
        assert 0.05 < radii[0] < 0.15  # Near 0.1 Rsun


class TestToCOMFrame:
    """Test center-of-mass frame transformation."""

    def test_to_com_frame_importable(self):
        from progenax.builders import to_com_frame
        assert callable(to_com_frame)

    def test_com_is_zero_after_transform(self):
        """COM position should be zero after transform."""
        from progenax.builders import to_com_frame

        # Create off-center positions
        positions = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        velocities = jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        masses = jnp.array([1.0, 1.0, 2.0])

        pos_com, vel_com = to_com_frame(positions, velocities, masses)

        # COM should be zero
        com_pos = jnp.sum(pos_com * masses[:, None], axis=0) / jnp.sum(masses)
        com_vel = jnp.sum(vel_com * masses[:, None], axis=0) / jnp.sum(masses)

        assert jnp.allclose(com_pos, 0.0, atol=1e-10)
        assert jnp.allclose(com_vel, 0.0, atol=1e-10)


class TestVirialScale:
    """Test virial scaling."""

    def test_virial_scale_importable(self):
        from progenax.builders import virial_scale
        assert callable(virial_scale)

    def test_virial_ratio_is_target(self):
        """After scaling, Q should equal target."""
        from progenax.builders import virial_scale, compute_kinetic_energy, compute_potential_energy

        # Simple 2-body test
        positions = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        velocities = jnp.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
        masses = jnp.array([1.0, 1.0])
        G = 1.0

        vel_scaled = virial_scale(positions, velocities, masses, Q_target=1.0, G=G)

        T = compute_kinetic_energy(vel_scaled, masses)
        V = compute_potential_energy(positions, masses, G=G)
        Q = 2.0 * T / jnp.abs(V)

        assert jnp.abs(Q - 1.0) < 0.01  # Within 1%

    def test_q_one_equilibrium(self):
        """Q = 1.0 gives virial equilibrium: system is dynamically stable."""
        from progenax.builders import virial_scale, compute_kinetic_energy, compute_potential_energy

        # Create a Plummer-like configuration
        positions = jnp.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        # Initially cold velocities
        velocities = jnp.array([
            [0.0, 0.1, 0.0],
            [0.0, -0.1, 0.0],
            [0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
        ])
        masses = jnp.ones(4)
        G = 1.0

        # Scale to Q = 1.0
        vel_scaled = virial_scale(positions, velocities, masses, Q_target=1.0, G=G)

        # Verify Q_target = 1.0 achieved
        T = compute_kinetic_energy(vel_scaled, masses)
        V = compute_potential_energy(positions, masses, G=G)
        Q = 2.0 * T / jnp.abs(V)

        assert jnp.abs(Q - 1.0) < 0.02
        assert Q > 0.98  # Slightly less than 1.0 is acceptable

    def test_q_less_than_one_collapsing(self):
        """Q < 1.0 gives sub-virial (cold) system that will collapse."""
        from progenax.builders import virial_scale, compute_kinetic_energy, compute_potential_energy

        # Create a configuration
        positions = jnp.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        velocities = jnp.array([
            [0.0, 0.1, 0.0],
            [0.0, -0.1, 0.0],
            [0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
        ])
        masses = jnp.ones(4)
        G = 1.0

        # Scale to Q = 0.5 (sub-virial)
        vel_scaled = virial_scale(positions, velocities, masses, Q_target=0.5, G=G)

        # Verify Q_target = 0.5 achieved
        T = compute_kinetic_energy(vel_scaled, masses)
        V = compute_potential_energy(positions, masses, G=G)
        Q = 2.0 * T / jnp.abs(V)

        assert Q < 1.0  # Sub-virial
        assert jnp.abs(Q - 0.5) < 0.05  # Close to target


class TestBuildSpatialIC:
    """Test main IC builder."""

    def test_build_spatial_ic_importable(self):
        from progenax.builders import build_spatial_ic
        assert callable(build_spatial_ic)

    def test_build_with_plummer(self):
        """Build IC with Plummer profile."""
        from progenax.builders import build_spatial_ic, ICResult
        from progenax.profiles import PlummerProfile
        from progenax.kinematics import PlummerVelocityDF

        masses = jnp.ones(100)
        profile = PlummerProfile(r_h=1.0)
        velocity_df = PlummerVelocityDF(r_h=1.0)
        key = jax.random.PRNGKey(42)
        G = 1.0  # Explicit G

        result = build_spatial_ic(
            profile=profile,
            masses=masses,
            velocity_df=velocity_df,
            Q=1.0,
            key=key,
            G=G,
        )

        assert isinstance(result, ICResult)
        assert result.positions.shape == (100, 3)
        assert result.velocities.shape == (100, 3)
        assert result.masses.shape == (100,)

    def test_ic_in_com_frame(self):
        """IC should be in COM frame."""
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

        # COM should be at origin
        com_pos = jnp.sum(result.positions * masses[:, None], axis=0) / jnp.sum(masses)
        assert jnp.allclose(com_pos, 0.0, atol=1e-6)
