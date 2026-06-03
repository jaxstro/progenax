"""Tests for the velocity sampling pipeline (progenax.kinematics.api).

Coverage-gap tests for sample_velocities_pipeline:
    - G=None default (resolves to progenax.DEFAULT_UNITS.G)
    - Osipkov-Merritt anisotropy stage
    - solid-body rotation stage
    - differential rotation stage
    - full integration: final virial Q ~= target, COM velocity ~= 0, shape (N,3)

Real positions are drawn from a Plummer profile so the O(N^2) virial
rescaling produces a physically meaningful equilibrium.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR
from progenax import defaults
from progenax.profiles.plummer import PlummerProfile
from progenax.kinematics import (
    PlummerVelocityDF,
    VelocityModel,
    AnisotropyParams,
    RotationParams,
    sample_velocities_pipeline,
)
from progenax.dynamics.virial import compute_virial_ratio


G = STELLAR.G


@pytest.fixture
def plummer_setup():
    """Equal-mass Plummer positions + masses for pipeline tests."""
    N = 500
    r_h = 1.0
    key = jax.random.PRNGKey(0)
    masses = jnp.ones(N)
    profile = PlummerProfile(r_h=r_h)
    positions = profile.sample_positions(masses, key)
    return positions, masses, r_h


class TestPipelineGDefault:
    """Cover the G=None default branch (line ~218)."""

    def test_g_none_uses_default_units(self, plummer_setup):
        """G=None resolves to DEFAULT_UNITS.G and matches an explicit pass."""
        positions, masses, r_h = plummer_setup
        model = VelocityModel(df=PlummerVelocityDF(r_h=r_h), target_Q=0.5)

        key = jax.random.PRNGKey(7)
        v_default = sample_velocities_pipeline(key, positions, masses, model, G=None)
        v_explicit = sample_velocities_pipeline(
            key, positions, masses, model, G=defaults.DEFAULT_UNITS.G
        )

        # Same key + same resolved G -> identical result
        assert jnp.allclose(v_default, v_explicit, atol=1e-12), (
            "G=None must resolve to DEFAULT_UNITS.G exactly"
        )
        assert v_default.shape == positions.shape
        assert jnp.all(jnp.isfinite(v_default))


class TestPipelineAnisotropy:
    """Cover the Osipkov-Merritt anisotropy branch (line ~228)."""

    def test_anisotropy_makes_velocities_radially_biased(self, plummer_setup):
        """Enabling OM anisotropy increases the radial velocity fraction."""
        positions, masses, r_h = plummer_setup
        df = PlummerVelocityDF(r_h=r_h)

        iso_model = VelocityModel(df=df, target_Q=0.5)
        aniso_model = VelocityModel(
            df=df,
            anisotropy=AnisotropyParams(use_osipkov_merritt=True, r_a=0.5),
            target_Q=0.5,
        )

        key = jax.random.PRNGKey(11)
        v_iso = sample_velocities_pipeline(key, positions, masses, iso_model, G=G)
        v_aniso = sample_velocities_pipeline(key, positions, masses, aniso_model, G=G)

        assert v_aniso.shape == positions.shape
        assert jnp.all(jnp.isfinite(v_aniso))

        # Radial fraction beta-proxy: <v_r^2> / <|v|^2>. OM biases toward radial,
        # especially at r > r_a, so the radial fraction should rise.
        r_hat = positions / jnp.linalg.norm(positions, axis=1, keepdims=True)
        def radial_frac(v):
            v_r = jnp.sum(v * r_hat, axis=1)
            return jnp.sum(v_r**2) / jnp.sum(v**2)

        assert radial_frac(v_aniso) > radial_frac(v_iso), (
            "Osipkov-Merritt should increase the radial velocity fraction"
        )

    def test_anisotropy_disabled_when_flag_false(self, plummer_setup):
        """anisotropy present but use_osipkov_merritt=False -> branch NOT taken."""
        positions, masses, r_h = plummer_setup
        df = PlummerVelocityDF(r_h=r_h)

        model_off = VelocityModel(
            df=df,
            anisotropy=AnisotropyParams(use_osipkov_merritt=False, r_a=0.5),
            target_Q=0.5,
        )
        model_none = VelocityModel(df=df, target_Q=0.5)

        key = jax.random.PRNGKey(3)
        v_off = sample_velocities_pipeline(key, positions, masses, model_off, G=G)
        v_none = sample_velocities_pipeline(key, positions, masses, model_none, G=G)
        # Flag off => identical pipeline to no anisotropy at all
        assert jnp.allclose(v_off, v_none, atol=1e-12)


class TestPipelineRotation:
    """Cover the rotation branches: axis default, solid-body, differential."""

    def test_solid_body_rotation_adds_angular_momentum(self, plummer_setup):
        """Solid-body rotation about z adds net L_z compared to no rotation."""
        positions, masses, r_h = plummer_setup
        df = PlummerVelocityDF(r_h=r_h)

        base = VelocityModel(df=df, target_Q=0.5)
        rot = VelocityModel(
            df=df,
            rotation=RotationParams(solid_body=True, pattern_speed=0.3),
            target_Q=0.5,
        )

        key = jax.random.PRNGKey(5)
        v_base = sample_velocities_pipeline(key, positions, masses, base, G=G)
        v_rot = sample_velocities_pipeline(key, positions, masses, rot, G=G)

        assert v_rot.shape == positions.shape
        assert jnp.all(jnp.isfinite(v_rot))

        # L_z = sum m (x vy - y vx). Positive pattern speed -> positive net L_z.
        def Lz(v):
            return jnp.sum(masses * (positions[:, 0] * v[:, 1] - positions[:, 1] * v[:, 0]))

        Lz_rot = Lz(v_rot)
        assert float(Lz_rot) > float(Lz(v_base)), "solid-body rotation should add +L_z"
        assert float(Lz_rot) > 0.0

    def test_solid_body_zero_pattern_speed_skips(self, plummer_setup):
        """pattern_speed=0 short-circuits the solid-body branch (no-op)."""
        positions, masses, r_h = plummer_setup
        df = PlummerVelocityDF(r_h=r_h)

        rot0 = VelocityModel(
            df=df,
            rotation=RotationParams(solid_body=True, pattern_speed=0.0),
            target_Q=0.5,
        )
        base = VelocityModel(df=df, target_Q=0.5)

        key = jax.random.PRNGKey(8)
        v_rot0 = sample_velocities_pipeline(key, positions, masses, rot0, G=G)
        v_base = sample_velocities_pipeline(key, positions, masses, base, G=G)
        assert jnp.allclose(v_rot0, v_base, atol=1e-12)

    def test_differential_rotation_adds_angular_momentum(self, plummer_setup):
        """Differential rotation adds net L_z and stays finite."""
        positions, masses, r_h = plummer_setup
        df = PlummerVelocityDF(r_h=r_h)

        base = VelocityModel(df=df, target_Q=0.5)
        diff = VelocityModel(
            df=df,
            rotation=RotationParams(differential=True, v_peak=0.5, r_peak=1.0),
            target_Q=0.5,
        )

        key = jax.random.PRNGKey(13)
        v_base = sample_velocities_pipeline(key, positions, masses, base, G=G)
        v_diff = sample_velocities_pipeline(key, positions, masses, diff, G=G)

        assert v_diff.shape == positions.shape
        assert jnp.all(jnp.isfinite(v_diff))

        def Lz(v):
            return jnp.sum(masses * (positions[:, 0] * v[:, 1] - positions[:, 1] * v[:, 0]))

        assert float(Lz(v_diff)) > float(Lz(v_base)), "differential rotation should add +L_z"

    def test_custom_rotation_axis_used(self, plummer_setup):
        """A non-default rotation axis is honored (x-axis -> L_x, not L_z)."""
        positions, masses, r_h = plummer_setup
        df = PlummerVelocityDF(r_h=r_h)

        rot_x = VelocityModel(
            df=df,
            rotation=RotationParams(
                solid_body=True, pattern_speed=0.3, axis=(1.0, 0.0, 0.0)
            ),
            target_Q=0.5,
        )
        key = jax.random.PRNGKey(21)
        v = sample_velocities_pipeline(key, positions, masses, rot_x, G=G)

        # Rotation about x-axis -> net L_x > 0, while L_z stays near the no-rotation level
        Lx = jnp.sum(masses * (positions[:, 1] * v[:, 2] - positions[:, 2] * v[:, 1]))
        assert float(Lx) > 0.0, "x-axis rotation should produce positive L_x"


class TestPipelineFullIntegration:
    """End-to-end: virial equilibrium, COM removal, shape, with all stages on."""

    def test_full_pipeline_virial_and_com(self, plummer_setup):
        """All stages on: final Q ~= target, COM velocity ~= 0, shape (N,3)."""
        positions, masses, r_h = plummer_setup
        target_Q = 0.5
        model = VelocityModel(
            df=PlummerVelocityDF(r_h=r_h),
            anisotropy=AnisotropyParams(use_osipkov_merritt=True, r_a=1.0),
            rotation=RotationParams(solid_body=True, pattern_speed=0.1),
            target_Q=target_Q,
        )

        key = jax.random.PRNGKey(42)
        v = sample_velocities_pipeline(key, positions, masses, model, G=G)

        # Shape
        assert v.shape == positions.shape == (500, 3)
        assert jnp.all(jnp.isfinite(v))

        # COM velocity removed (mass-weighted mean ~ 0)
        M_total = jnp.sum(masses)
        v_com = jnp.sum(masses[:, None] * v, axis=0) / M_total
        assert jnp.allclose(v_com, 0.0, atol=1e-10), f"COM velocity not removed: {v_com}"

        # Virial ratio matches target. Rescaling targets Q BEFORE COM subtraction
        # and rotation injects ordered KE, so allow a modest tolerance.
        Q = compute_virial_ratio(positions, v, masses, G=G)
        assert jnp.isclose(Q, target_Q, atol=0.05), (
            f"final Q={float(Q):.4f} vs target {target_Q}"
        )

    def test_subvirial_target_is_colder(self, plummer_setup):
        """A subvirial target_Q=0.3 yields less kinetic energy than Q=0.5."""
        positions, masses, r_h = plummer_setup
        df = PlummerVelocityDF(r_h=r_h)

        key = jax.random.PRNGKey(99)
        v_cold = sample_velocities_pipeline(
            key, positions, masses, VelocityModel(df=df, target_Q=0.3), G=G
        )
        v_eq = sample_velocities_pipeline(
            key, positions, masses, VelocityModel(df=df, target_Q=0.5), G=G
        )

        T_cold = 0.5 * jnp.sum(masses * jnp.sum(v_cold**2, axis=1))
        T_eq = 0.5 * jnp.sum(masses * jnp.sum(v_eq**2, axis=1))
        assert float(T_cold) < float(T_eq), "subvirial Q=0.3 should be colder than Q=0.5"

        # And the cold system's virial ratio lands near 0.3 (no rotation here)
        Q_cold = compute_virial_ratio(positions, v_cold, masses, G=G)
        assert jnp.isclose(Q_cold, 0.3, atol=0.05)

    def test_pipeline_is_differentiable_in_target_Q(self, plummer_setup):
        """Pipeline supports jax.grad w.r.t. target_Q (KE scales with Q)."""
        positions, masses, r_h = plummer_setup
        df = PlummerVelocityDF(r_h=r_h)
        key = jax.random.PRNGKey(1)

        def total_ke(target_Q):
            model = VelocityModel(df=df, target_Q=target_Q)
            v = sample_velocities_pipeline(key, positions, masses, model, G=G)
            return 0.5 * jnp.sum(masses * jnp.sum(v**2, axis=1))

        g = jax.grad(total_ke)(0.5)
        assert jnp.isfinite(g), "gradient w.r.t. target_Q must be finite"
        # KE ~ Q * |V|, so dKE/dQ > 0
        assert float(g) > 0.0, "kinetic energy should increase with target_Q"
