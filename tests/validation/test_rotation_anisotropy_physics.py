"""
Physics validation for rotation transforms + Osipkov-Merritt anisotropy.

Rotation (Binney & Tremaine 2008, Sec 4.8): adds streaming v_phi to an isotropic IC.
  - solid body:  v_phi(R) = Omega * R  (cylindrical R)
  - differential: v_phi(R) = v_peak (R/R_peak) exp(1 - R/R_peak)  (phenomenological)
Anisotropy (Merritt 1985): the `anisotropy_radius` r_a realizes the Osipkov-Merritt
profile beta(r) = r^2/(r^2 + r_a^2) *exactly* via a velocity-direction stretch -- in
contrast to the self-consistent Michie-King DF, whose beta is suppressed below OM.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.kinematics import EFFVelocityDF, PlummerVelocityDF
from progenax.kinematics.rotation import (
    apply_differential_rotation,
    apply_solid_body_rotation,
)
from progenax.profiles import EFFProfile, PlummerProfile

G = 1.0
ZAXIS = jnp.array([0.0, 0.0, 1.0])


def _cyl(pos, vel):
    """Cylindrical R, v_phi, v_R about the z-axis."""
    x, y = pos[:, 0], pos[:, 1]
    R = jnp.sqrt(x**2 + y**2)
    v_phi = (x * vel[:, 1] - y * vel[:, 0]) / (R + 1e-30)
    v_R = (x * vel[:, 0] + y * vel[:, 1]) / (R + 1e-30)
    return np.asarray(R), np.asarray(v_phi), np.asarray(v_R)


def _isotropic_ic(n=40_000, seed=0):
    prof = PlummerProfile(r_h=1.0)
    df = PlummerVelocityDF(r_h=1.0)
    m = jnp.ones(n)
    kp, kv = jax.random.split(jax.random.PRNGKey(seed))
    pos = prof.sample_positions(m, kp)
    vel = df.sample_velocities(pos, m, kv, G=G)
    return m, pos, vel


class TestSolidBodyRotation:
    # Rotation is an *additive* transform v -> v + v_rot, so validating the streaming
    # field v_rot = v_after - v_before isolates it exactly (no isotropic-scatter noise).
    def test_v_phi_linear_in_R(self):
        """Added streaming v_phi(R) = Omega * R exactly (slope = Omega, residual ~0)."""
        omega = 0.3
        m, pos, vel = _isotropic_ic()
        dvel = apply_solid_body_rotation(vel, pos, omega, ZAXIS) - vel
        R, dphi, _ = _cyl(pos, dvel)
        sel = R < 3.0
        slope, intercept = np.polyfit(R[sel], dphi[sel], 1)
        resid = np.max(np.abs(dphi[sel] - omega * R[sel]))
        assert abs(slope - omega) < 1e-6 and resid < 1e-6, (
            f"slope={slope:.6f} vs Omega={omega}, max resid={resid:.2e}"
        )

    def test_angular_momentum_budget(self):
        """Added L_z = Omega * sum(m R^2) exactly."""
        omega = 0.3
        m, pos, vel = _isotropic_ic()
        R = np.sqrt(np.asarray(pos[:, 0]) ** 2 + np.asarray(pos[:, 1]) ** 2)
        dvel = apply_solid_body_rotation(vel, pos, omega, ZAXIS) - vel
        dLz = float(jnp.sum(m * (pos[:, 0] * dvel[:, 1] - pos[:, 1] * dvel[:, 0])))
        Lz_expected = omega * float(np.sum(np.asarray(m) * R**2))
        assert abs(dLz - Lz_expected) / Lz_expected < 1e-6, (
            f"dLz={dLz:.1f} vs {Lz_expected:.1f}"
        )

    def test_radial_velocity_unchanged(self):
        """Rotation is purely azimuthal: cylindrical v_R is unchanged."""
        m, pos, vel = _isotropic_ic()
        _, _, vR0 = _cyl(pos, vel)
        vrot = apply_solid_body_rotation(vel, pos, 0.3, ZAXIS)
        _, _, vR1 = _cyl(pos, vrot)
        assert np.allclose(vR0, vR1, atol=1e-9), "v_R must be unchanged by rotation"


class TestDifferentialRotation:
    def test_v_phi_curve_matches(self):
        """Added streaming v_phi(R) = v_peak (R/R_peak) exp(1 - R/R_peak) exactly."""
        v_peak, R_peak = 2.0, 1.0
        m, pos, vel = _isotropic_ic()
        dvel = apply_differential_rotation(vel, pos, v_peak, R_peak, ZAXIS) - vel
        R, dphi, _ = _cyl(pos, dvel)
        expected = v_peak * (R / R_peak) * np.exp(1 - R / R_peak)
        assert np.max(np.abs(dphi - expected)) < 1e-6, (
            "differential v_phi must match the curve"
        )

    def test_peak_value_at_R_peak(self):
        """The added v_phi(R_peak) = v_peak (the curve maximum)."""
        v_peak, R_peak = 2.0, 1.0
        m, pos, vel = _isotropic_ic()
        dvel = apply_differential_rotation(vel, pos, v_peak, R_peak, ZAXIS) - vel
        R, dphi, _ = _cyl(pos, dvel)
        sel = (R >= 0.95 * R_peak) & (R < 1.05 * R_peak)
        assert abs(dphi[sel].mean() - v_peak) < 0.02


def _beta_binned(pos, vel, edges):
    radii = np.linalg.norm(np.asarray(pos), axis=1)
    r_hat = np.asarray(pos) / (radii[:, None] + 1e-30)
    v_r = np.sum(np.asarray(vel) * r_hat, axis=1)
    v_t = np.linalg.norm(np.asarray(vel) - v_r[:, None] * r_hat, axis=1)
    mids, beta = [], []
    for lo, hi in edges:
        m = (radii >= lo) & (radii < hi)
        if m.sum() < 200:
            continue
        mids.append(0.5 * (lo + hi))
        beta.append(1.0 - np.mean(v_t[m] ** 2) / (2.0 * np.mean(v_r[m] ** 2)))
    return np.array(mids), np.array(beta)


class TestOsipkovMerrittAnisotropy:
    def test_plummer_beta_matches_exact_om(self):
        """Plummer OM realizes beta(r) = r^2/(r^2+r_a^2) *exactly* (stretch split)."""
        r_a = 1.5
        prof = PlummerProfile(r_h=1.0)
        df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=r_a)
        m = jnp.ones(60_000)
        kp, kv = jax.random.split(jax.random.PRNGKey(1))
        pos = prof.sample_positions(m, kp)
        vel = df.sample_velocities(pos, m, kv, G=G)
        mids, beta = _beta_binned(
            pos, vel, [(0.4, 0.8), (0.9, 1.3), (1.6, 2.2), (2.6, 3.6)]
        )
        target = mids**2 / (mids**2 + r_a**2)
        assert np.all(np.abs(beta - target) < 0.04), f"beta={beta} vs OM={target}"

    def test_eff_beta_matches_om(self):
        """EFF OM realizes the OM beta profile via the same stretch."""
        r_a = 1.5
        prof = EFFProfile(a=1.0, gamma=4.0, r_t=15.0)
        df = EFFVelocityDF(a=1.0, gamma=4.0, r_t=15.0, anisotropy_radius=r_a)
        m = jnp.ones(60_000)
        kp, kv = jax.random.split(jax.random.PRNGKey(2))
        pos = prof.sample_positions(m, kp)
        vel = df.sample_velocities(pos, m, kv, G=G)
        mids, beta = _beta_binned(
            pos, vel, [(0.4, 0.8), (0.9, 1.3), (1.6, 2.2), (2.6, 3.6)]
        )
        target = mids**2 / (mids**2 + r_a**2)
        assert np.all(np.abs(beta - target) < 0.05), f"beta={beta} vs OM={target}"

    def test_none_is_isotropic(self):
        """anisotropy_radius=None gives beta ~ 0 (isotropic)."""
        prof = PlummerProfile(r_h=1.0)
        df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=None)
        m = jnp.ones(40_000)
        kp, kv = jax.random.split(jax.random.PRNGKey(3))
        pos = prof.sample_positions(m, kp)
        vel = df.sample_velocities(pos, m, kv, G=G)
        _, beta = _beta_binned(pos, vel, [(0.5, 1.0), (1.0, 1.8)])
        assert np.all(np.abs(beta) < 0.04), f"isotropic beta should be ~0: {beta}"


# AD-vs-FD for apply_solid_body_rotation(omega) and apply_differential_rotation(v_peak)
# is owned by the grad-audit registry (tests/validation/grad_audit/registry.py ::
# apply_solid_body_rotation, apply_differential_rotation); see
# docs/website/50-validation/differentiability-audit.md. The former
# TestRotationDifferentiability class was removed here (audit T6 consolidation; registry is SoT).
# All physics tests in this file (solid-body / differential rotation curves, OM anisotropy) stay.


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
