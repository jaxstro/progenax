"""Osipkov-Merritt anisotropy via the `anisotropy_radius` (r_a) DF parameter.

The headline test: the *realized* anisotropy profile must match Merritt (1985) Eq. 15/17,

    beta(r) = 1 - sigma_t^2 / (2 sigma_r^2) = r^2 / (r^2 + r_a^2),

profile-independent. This is the exact check the old heuristic apply_osipkov_merritt
transform failed (it over-biased at intermediate beta). r_a=None must reproduce the
isotropic DF (beta(r) == 0).
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR
from progenax.kinematics.plummer_df import PlummerVelocityDF
from progenax.kinematics.eff_df import EFFVelocityDF
from progenax.profiles.plummer import PlummerProfile
from progenax.dynamics.virial import compute_virial_ratio

G = STELLAR.G


def _shell(r, N, seed):
    """N isotropic positions on a thin shell of radius r."""
    dirs = jax.random.normal(jax.random.PRNGKey(seed), (N, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    return r * dirs


def _measure_beta(velocities, positions):
    """beta = 1 - <v_t^2> / (2 <v_r^2>) over a population at one radius."""
    r_hat = positions / jnp.linalg.norm(positions, axis=1, keepdims=True)
    v_r = jnp.sum(velocities * r_hat, axis=1)
    v_t2 = jnp.sum(velocities**2, axis=1) - v_r**2
    return 1.0 - jnp.mean(v_t2) / (2.0 * jnp.mean(v_r**2))


class TestPlummerOMRealizedBeta:
    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 4.0])
    def test_realized_beta_matches_target(self, r):
        r_h, r_a = 1.0, 1.5
        df = PlummerVelocityDF(r_h=r_h, anisotropy_radius=r_a)
        N = 40000
        pos = _shell(r, N, seed=0)
        masses = jnp.ones(N)
        v = df.sample_velocities(pos, masses, jax.random.PRNGKey(1), G=G)

        beta = float(_measure_beta(v, pos))
        target = r**2 / (r**2 + r_a**2)
        assert abs(beta - target) < 0.03, f"r={r}: realized beta={beta:.3f} vs target {target:.3f}"

    def test_r_a_none_is_isotropic(self):
        df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=None)
        N = 40000
        pos = _shell(2.0, N, seed=2)
        v = df.sample_velocities(pos, jnp.ones(N), jax.random.PRNGKey(3), G=G)
        assert abs(float(_measure_beta(v, pos))) < 0.03, "r_a=None must stay isotropic (beta~0)"

    def test_large_r_a_recovers_isotropy(self):
        """The f-table OM path with very large r_a must reduce to isotropy (beta~0)."""
        df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=30.0)
        N = 40000
        pos = _shell(1.0, N, seed=4)
        v = df.sample_velocities(pos, jnp.ones(N), jax.random.PRNGKey(5), G=G)
        assert abs(float(_measure_beta(v, pos))) < 0.04, "large r_a must be ~isotropic"


class TestEFFOMRealizedBeta:
    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 4.0])
    def test_realized_beta_matches_target(self, r):
        a, gamma, r_t, r_a = 1.0, 3.0, 10.0, 1.5
        df = EFFVelocityDF(a=a, gamma=gamma, r_t=r_t, anisotropy_radius=r_a)
        N = 40000
        pos = _shell(r, N, seed=10)
        v = df.sample_velocities(pos, jnp.ones(N), jax.random.PRNGKey(11), G=G)

        beta = float(_measure_beta(v, pos))
        target = r**2 / (r**2 + r_a**2)
        assert abs(beta - target) < 0.03, f"r={r}: realized beta={beta:.3f} vs target {target:.3f}"


class TestOMVirialEquilibrium:
    """OM is a self-consistent DF for the SAME density, so ICs stay ~virial (Q~0.5)."""

    def test_plummer_om_is_virial(self):
        N = 4000
        r_h, r_a = 1.0, 1.0
        masses = jnp.ones(N)
        pos = PlummerProfile(r_h=r_h).sample_positions(masses, jax.random.PRNGKey(0))
        df = PlummerVelocityDF(r_h=r_h, anisotropy_radius=r_a)
        v = df.sample_velocities(pos, masses, jax.random.PRNGKey(1), G=G)
        Q = float(compute_virial_ratio(pos, v, masses, G=G))
        assert abs(Q - 0.5) < 0.06, f"OM Plummer Q={Q:.3f} should be ~0.5 (still equilibrium)"


class TestOMNonNegativity:
    """Refuse an unphysical (negative) OM DF rather than silently clamping it."""

    def test_plummer_below_bound_raises(self):
        # a = sqrt(2^(2/3)-1) ~ 0.766; bound 0.75 a ~ 0.575.
        with pytest.raises(ValueError, match="0.75 a"):
            PlummerVelocityDF(r_h=1.0, anisotropy_radius=0.4)

    def test_plummer_just_above_bound_ok(self):
        df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=0.6)  # > 0.575
        assert df.anisotropy_radius is not None

    def test_eff_too_anisotropic_raises(self):
        with pytest.raises(ValueError, match="negative"):
            EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0, anisotropy_radius=0.02)


class TestOMGradients:
    # AD-vs-FD for PlummerVelocityDF+OM.sample_velocities(r_a) is owned by the grad-audit
    # registry (tests/validation/grad_audit/registry.py :: PlummerVelocityDF+OM.sample_velocities,
    # with an edge probing the Merritt 0.75a bound); see
    # docs/website/50-validation/differentiability-audit.md. The former
    # test_plummer_grad_wrt_r_a_matches_fd was removed here (audit T6 consolidation; registry is SoT).
    def test_plummer_om_jit(self):
        df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=1.5)
        N = 128
        pos = _shell(1.0, N, seed=32)
        v = jax.jit(lambda p, m, k: df.sample_velocities(p, m, k, G=G))(
            pos, jnp.ones(N), jax.random.PRNGKey(33)
        )
        assert v.shape == (128, 3) and jnp.all(jnp.isfinite(v))

    def test_eff_om_jit(self):
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0, anisotropy_radius=1.5)
        N = 128
        pos = _shell(1.0, N, seed=34)
        v = jax.jit(lambda p, m, k: df.sample_velocities(p, m, k, G=G))(
            pos, jnp.ones(N), jax.random.PRNGKey(35)
        )
        assert v.shape == (128, 3) and jnp.all(jnp.isfinite(v))
