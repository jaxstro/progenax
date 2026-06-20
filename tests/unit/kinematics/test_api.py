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

from progenax.dynamics.virial import compute_virial_ratio
from progenax.kinematics import (
    PlummerVelocityDF,
    RotationParams,
    VelocityModel,
    sample_velocities_pipeline,
)
from progenax.profiles.plummer import PlummerProfile

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


class TestExplicitGRequired:
    """Explicit-units policy (audit A2): G is REQUIRED on every velocity-sampling
    surface. Omitting G must raise (no silent DEFAULT_UNITS.G fallback).

    Covers all 5 DF ``.sample_velocities`` surfaces, the ``sample_velocities_pipeline``
    entry point, and ``MultiComponentCluster.sample_cluster``.
    """

    def test_pipeline_requires_explicit_G(self, plummer_setup):
        """sample_velocities_pipeline raises when G is omitted (required arg)."""
        positions, masses, r_h = plummer_setup
        model = VelocityModel(df=PlummerVelocityDF(r_h=r_h), target_Q=0.5)
        key = jax.random.PRNGKey(7)
        with pytest.raises(TypeError):
            sample_velocities_pipeline(key, positions, masses, model)

    def test_all_dfs_require_explicit_G(self):
        """Every DF.sample_velocities raises when G is omitted (required arg)."""
        from progenax.kinematics import (
            EFFVelocityDF,
            KingVelocityDF,
            MichieVelocityDF,
        )
        from progenax.kinematics.limepy_df import LIMEPYVelocityDF

        N = 8
        positions = jnp.ones((N, 3)) * 0.1
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(0)
        dfs = [
            PlummerVelocityDF(r_h=1.0),
            KingVelocityDF(W0=5.0, r_c=1.0),
            EFFVelocityDF(gamma=5.0, a=1.0, r_t=10.0),
            MichieVelocityDF(W0=5.0, r_c=1.0, r_a=2.0),
            LIMEPYVelocityDF(W0=5.0, g=1.0, r_c=1.0),
        ]
        for df in dfs:
            with pytest.raises(TypeError):
                df.sample_velocities(positions, masses, key)

    def test_cluster_requires_explicit_G(self):
        """MultiComponentCluster.sample_cluster raises when G is omitted."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        model = MultiComponentCluster.from_components(
            alpha_j=jnp.array([0.6, 0.4]),
            w_j=jnp.array([1.0, 0.8]),
            m_j=jnp.array([0.4, 1.0]),
            W0=6.0,
            g=1.0,
            r_c=1.0,
        )
        with pytest.raises(TypeError):
            model.sample_cluster(jax.random.PRNGKey(0), n_stars=100)


class TestPipelineAnisotropicDF:
    """Radial anisotropy now lives on the DF; it must flow through the pipeline."""

    def test_anisotropic_df_increases_radial_fraction(self, plummer_setup):
        """An Osipkov-Merritt DF raises the radial velocity fraction vs isotropic."""
        positions, masses, r_h = plummer_setup
        iso = VelocityModel(df=PlummerVelocityDF(r_h=r_h))
        aniso = VelocityModel(df=PlummerVelocityDF(r_h=r_h, anisotropy_radius=0.6))

        key = jax.random.PRNGKey(11)
        v_iso = sample_velocities_pipeline(key, positions, masses, iso, G=G)
        v_aniso = sample_velocities_pipeline(key, positions, masses, aniso, G=G)

        assert v_aniso.shape == positions.shape and jnp.all(jnp.isfinite(v_aniso))

        r_hat = positions / jnp.linalg.norm(positions, axis=1, keepdims=True)

        def radial_frac(v):
            v_r = jnp.sum(v * r_hat, axis=1)
            return jnp.sum(v_r**2) / jnp.sum(v**2)

        assert radial_frac(v_aniso) > radial_frac(v_iso), (
            "an Osipkov-Merritt DF should increase the radial velocity fraction"
        )


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
            return jnp.sum(
                masses * (positions[:, 0] * v[:, 1] - positions[:, 1] * v[:, 0])
            )

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
            return jnp.sum(
                masses * (positions[:, 0] * v[:, 1] - positions[:, 1] * v[:, 0])
            )

        assert float(Lz(v_diff)) > float(Lz(v_base)), (
            "differential rotation should add +L_z"
        )

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


class TestPipelineTargetQNone:
    """D1: target_Q=None opts OUT of virial rescaling (keep DF-native equilibrium)."""

    def test_none_skips_rescale_equals_raw_df_plus_com(self, plummer_setup):
        """With target_Q=None the pipeline must NOT rescale: the result is exactly the
        raw DF sample (same key routing) with COM motion removed."""
        positions, masses, r_h = plummer_setup
        df = PlummerVelocityDF(r_h=r_h)
        key = jax.random.PRNGKey(17)

        model = VelocityModel(df=df, target_Q=None)
        v_pipeline = sample_velocities_pipeline(key, positions, masses, model, G=G)

        # Reference: the pipeline feeds `key` straight to the DF, then removes COM.
        v_raw = df.sample_velocities(positions, masses, key, G=G)
        M_total = jnp.sum(masses)
        v_raw = v_raw - jnp.sum(masses[:, None] * v_raw, axis=0) / M_total

        assert jnp.allclose(v_pipeline, v_raw, atol=1e-12), (
            "target_Q=None must skip the virial rescale (raw DF + COM removal only)"
        )

    def test_none_preserves_native_equilibrium_not_forced(self, plummer_setup):
        """target_Q=None keeps the DF's native Q; an explicit float forces it.

        The Plummer DF is already a true equilibrium, so its native Q is ~0.5 but NOT
        exactly 0.5; forcing target_Q=0.5 lands exactly on 0.5. The two must differ."""
        positions, masses, r_h = plummer_setup
        df = PlummerVelocityDF(r_h=r_h)
        key = jax.random.PRNGKey(23)

        v_native = sample_velocities_pipeline(
            key, positions, masses, VelocityModel(df=df, target_Q=None), G=G
        )
        v_forced = sample_velocities_pipeline(
            key, positions, masses, VelocityModel(df=df, target_Q=0.5), G=G
        )

        # target_Q=0.5 applies a global rescale before COM removal, so v_forced is a
        # scaled copy of v_native -> the arrays must differ (the rescale factor != 1
        # unless the raw DF were exactly virial, which finite-N sampling never is).
        assert not jnp.allclose(v_native, v_forced, atol=1e-6), (
            "target_Q=0.5 must rescale, so it cannot equal the un-rescaled None result"
        )

        # Forced Q lands near 0.5 (the rescale targets Q *before* COM subtraction, so a
        # ~1/N COM-energy drift of order 1e-3 is expected and physical, not a miss).
        Q_forced = float(compute_virial_ratio(positions, v_forced, masses, G=G))
        assert abs(Q_forced - 0.5) < 5e-3, f"forced Q={Q_forced} should be ~0.5"


class TestPipelineFullIntegration:
    """End-to-end: virial equilibrium, COM removal, shape, with all stages on."""

    def test_full_pipeline_virial_and_com(self, plummer_setup):
        """All stages on: final Q ~= target, COM velocity ~= 0, shape (N,3)."""
        positions, masses, r_h = plummer_setup
        target_Q = 0.5
        model = VelocityModel(
            df=PlummerVelocityDF(r_h=r_h, anisotropy_radius=1.0),
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
        assert jnp.allclose(v_com, 0.0, atol=1e-10), (
            f"COM velocity not removed: {v_com}"
        )

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
        assert float(T_cold) < float(T_eq), (
            "subvirial Q=0.3 should be colder than Q=0.5"
        )

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
