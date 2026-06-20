"""Tests for progenax builders - physics only."""

import jax
import jax.numpy as jnp
import pytest

from progenax.builders import compute_stellar_radii


class TestComputeStellarRadii:
    """Anchored to Demircan & Kahraman (1991) Ap&SS 181, 313 + observed radii.

    Audit R6: the previous exponents were INVERTED vs MS homology
    (M^0.8 above 1 Msun instead of below), giving 10 Msun -> 6.3 Rsun
    (ZAMS ~4) and 0.2 Msun -> 0.40 Rsun (observed ~0.22). The old tests
    asserted the code's own formula back at itself.
    """

    def test_solar(self):
        r = compute_stellar_radii(jnp.array([1.0]))
        assert abs(float(r[0]) - 1.06) < 0.01  # D&K91: 1.06 * 1^0.945

    def test_massive_star_near_zams(self):
        r = float(compute_stellar_radii(jnp.array([10.0]))[0])
        assert abs(r - 1.33 * 10.0**0.555) < 1e-6  # = 4.77 Rsun
        assert 3.0 < r < 5.5  # within ~25% of ZAMS ~4 Rsun

    def test_m_dwarf(self):
        r = float(compute_stellar_radii(jnp.array([0.2]))[0])
        assert abs(r - 1.06 * 0.2**0.945) < 1e-6  # = 0.231 Rsun
        assert 0.17 < r < 0.28  # observed ~0.22 Rsun

    def test_brown_dwarf_plateau(self):
        r = compute_stellar_radii(jnp.array([0.05, 0.02]))
        assert jnp.all(jnp.abs(r - 0.1) < 0.05)

    def test_near_continuous_at_hydrogen_burning_limit(self):
        r = compute_stellar_radii(jnp.array([0.079, 0.081]))
        assert (
            abs(float(r[1]) / float(r[0]) - 1.0) < 0.1
        )  # no factor-2.4 jump (audit F7)


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
    """Test virial scaling physics.

    Convention: Q = T / |V| (equilibrium at Q = 0.5)
    """

    def test_virial_ratio_is_target(self):
        """After scaling, Q = T/|V| should equal target."""
        from progenax.builders import (
            compute_kinetic_energy,
            compute_potential_energy,
            virial_scale,
        )

        positions = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        velocities = jnp.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
        masses = jnp.array([1.0, 1.0])
        G = 1.0

        vel_scaled = virial_scale(positions, velocities, masses, Q_target=0.5, G=G)

        T = compute_kinetic_energy(vel_scaled, masses)
        V = compute_potential_energy(positions, masses, G=G)
        Q = T / jnp.abs(V)  # Q = T/|V|, NOT 2T/|V|

        assert jnp.abs(Q - 0.5) < 0.01

    def test_q_half_equilibrium(self):
        """Q = 0.5 gives virial equilibrium (2T + V = 0)."""
        from progenax.builders import (
            compute_kinetic_energy,
            compute_potential_energy,
            virial_scale,
        )

        positions = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        velocities = jnp.array(
            [
                [0.0, 0.1, 0.0],
                [0.0, -0.1, 0.0],
                [0.1, 0.0, 0.0],
                [-0.1, 0.0, 0.0],
            ]
        )
        masses = jnp.ones(4)
        G = 1.0

        vel_scaled = virial_scale(positions, velocities, masses, Q_target=0.5, G=G)

        T = compute_kinetic_energy(vel_scaled, masses)
        V = compute_potential_energy(positions, masses, G=G)
        Q = T / jnp.abs(V)  # Q = T/|V|

        assert jnp.abs(Q - 0.5) < 0.02

    def test_q_less_than_half_collapsing(self):
        """Q < 0.5 gives sub-virial (cold) system."""
        from progenax.builders import (
            compute_kinetic_energy,
            compute_potential_energy,
            virial_scale,
        )

        positions = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        velocities = jnp.array(
            [
                [0.0, 0.1, 0.0],
                [0.0, -0.1, 0.0],
                [0.1, 0.0, 0.0],
                [-0.1, 0.0, 0.0],
            ]
        )
        masses = jnp.ones(4)
        G = 1.0

        vel_scaled = virial_scale(positions, velocities, masses, Q_target=0.25, G=G)

        T = compute_kinetic_energy(vel_scaled, masses)
        V = compute_potential_energy(positions, masses, G=G)
        Q = T / jnp.abs(V)  # Q = T/|V|

        assert Q < 0.5  # Subvirial
        assert jnp.abs(Q - 0.25) < 0.05

    def test_virial_scale_zero_velocities_raises(self):
        """Audit J5: cold input (T=0) used to return all-NaN velocities silently."""
        from progenax.builders import virial_scale

        pos = jax.random.normal(jax.random.PRNGKey(0), (50, 3))
        vel = jnp.zeros((50, 3))
        m = jnp.ones(50)
        with pytest.raises(ValueError, match="zero kinetic"):
            virial_scale(pos, vel, m, Q_target=0.5, G=0.00449)

    def test_rescale_velocities_to_virial_zero_raises(self):
        """The deduped sibling shares the same eager T=0 guard (audit J5)."""
        from progenax.dynamics.virial import rescale_velocities_to_virial

        pos = jax.random.normal(jax.random.PRNGKey(1), (50, 3))
        vel = jnp.zeros((50, 3))
        m = jnp.ones(50)
        with pytest.raises(ValueError, match="zero kinetic"):
            rescale_velocities_to_virial(pos, vel, m, G=0.00449, target_Q=0.5)

    def test_virial_scale_traced_zero_returns_nan(self):
        """Under tracing the guard can't fire; NaN is the honest sentinel."""
        from progenax.builders import virial_scale

        pos = jax.random.normal(jax.random.PRNGKey(2), (10, 3))
        m = jnp.ones(10)

        def f(vel):
            return virial_scale(pos, vel, m, Q_target=0.5, G=0.00449)

        out = jax.jit(f)(jnp.zeros((10, 3)))
        assert jnp.all(jnp.isnan(out))


class TestBuildSpatialIC:
    """Test main IC builder."""

    def test_build_with_plummer(self):
        """Build IC with Plummer profile produces valid output."""
        from progenax.builders import ICResult, build_spatial_ic
        from progenax.kinematics import PlummerVelocityDF
        from progenax.profiles import PlummerProfile

        masses = jnp.ones(100)
        profile = PlummerProfile(r_h=1.0)
        velocity_df = PlummerVelocityDF(r_h=1.0)
        key = jax.random.PRNGKey(42)
        G = 1.0

        result = build_spatial_ic(
            profile=profile,
            masses=masses,
            velocity_df=velocity_df,
            Q=0.5,  # Q = T/|V|, 0.5 for equilibrium
            key=key,
            G=G,
        )

        assert isinstance(result, ICResult)
        assert jnp.all(jnp.isfinite(result.positions))
        assert jnp.all(jnp.isfinite(result.velocities))

    def test_ic_in_com_frame(self):
        """IC should be in COM frame (zero mean position/velocity)."""
        from progenax.builders import build_spatial_ic
        from progenax.kinematics import PlummerVelocityDF
        from progenax.profiles import PlummerProfile

        masses = jnp.ones(100)
        profile = PlummerProfile(r_h=1.0)
        velocity_df = PlummerVelocityDF(r_h=1.0)
        key = jax.random.PRNGKey(42)
        G = 1.0

        result = build_spatial_ic(
            profile=profile,
            masses=masses,
            velocity_df=velocity_df,
            Q=0.5,  # Q = T/|V|, 0.5 for equilibrium
            key=key,
            G=G,
        )

        com_pos = jnp.sum(result.positions * masses[:, None], axis=0) / jnp.sum(masses)
        assert jnp.allclose(com_pos, 0.0, atol=1e-6)
