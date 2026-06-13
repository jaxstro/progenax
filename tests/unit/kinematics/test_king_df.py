"""
Tests for King (1966) velocity distribution function.

Physics tests (isotropy, dispersion profile) plus the table-routing contract:
speed_method="table" (default) must agree with the exact per-star quadrature
oracle (speed_method="quadrature").
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.units import STELLAR
from progenax.kinematics.king_df import KingVelocityDF

G = STELLAR.G


class TestKingVelocityDFApi:
    """Audit A3: r_t was a stored-but-never-used field (sigma^2 = GM/(9 r_c mu),
    truncation comes from psi(xi)). Removed — no backwards-compat shim."""

    def test_constructs_without_r_t(self):
        df = KingVelocityDF(W0=7.0, r_c=1.0)
        assert not hasattr(df, "r_t")

    def test_passing_r_t_is_rejected(self):
        with pytest.raises(TypeError):
            KingVelocityDF(W0=7.0, r_c=1.0, r_t=10.0)


class TestKingVelocityDFPhysics:
    """Test King velocity DF physical properties."""

    def test_isotropic_distribution(self):
        """Velocities are isotropically distributed."""
        df = KingVelocityDF(W0=5.0, r_c=1.0)
        N = 1000
        r = 2.0
        positions = jnp.array([[r, 0.0, 0.0]] * N)
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        vx2_mean = jnp.mean(velocities[:, 0]**2)
        vy2_mean = jnp.mean(velocities[:, 1]**2)
        vz2_mean = jnp.mean(velocities[:, 2]**2)

        assert jnp.abs(vx2_mean - vy2_mean) / vx2_mean < 0.20
        assert jnp.abs(vy2_mean - vz2_mean) / vy2_mean < 0.20

    def test_mean_velocity_zero(self):
        """Mean velocity is zero (no bulk motion)."""
        df = KingVelocityDF(W0=5.0, r_c=1.0)
        N = 1000
        r = 2.0
        positions = jnp.array([[r, 0.0, 0.0]] * N)
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        mean_vel = jnp.mean(velocities, axis=0)
        v_std = jnp.std(velocities, axis=0)

        for i in range(3):
            assert jnp.abs(mean_vel[i]) < 3 * v_std[i] / jnp.sqrt(N)

    def test_dispersion_decreases_outward(self):
        """Velocity dispersion decreases with radius."""
        df = KingVelocityDF(W0=7.0, r_c=1.0)
        N = 500

        r_inner = 1.0
        r_outer = 5.0
        positions_inner = jnp.array([[r_inner, 0.0, 0.0]] * N)
        positions_outer = jnp.array([[r_outer, 0.0, 0.0]] * N)
        masses = jnp.ones(N)

        key_inner, key_outer = jax.random.split(jax.random.PRNGKey(42))

        velocities_inner = df.sample_velocities(positions_inner, masses, key_inner, G=G)
        velocities_outer = df.sample_velocities(positions_outer, masses, key_outer, G=G)

        sigma_inner = jnp.std(velocities_inner[:, 0])
        sigma_outer = jnp.std(velocities_outer[:, 0])

        assert sigma_inner > sigma_outer


class TestKingTableRouting:
    """speed_method='table' (default) must agree with the exact quadrature
    oracle (speed_method='quadrature') distributionally and in moments —
    the same contract the LIMEPY routing passed (Task 5,
    tests/unit/kinematics/test_limepy_df.py::TestLimepyTableRouting).

    N is capped at 2e4 for every draw through the quadrature oracle (the
    review-mandated oracle-N convention): the two-sample KS 95% critical D
    at n=2e4 is ~0.0136 < the 0.02 threshold, and the table/quadrature
    draws are variate-paired (same per-star key splits)."""

    def _two_dfs(self):
        kw = dict(W0=5.0, r_c=1.0)
        return (KingVelocityDF(**kw),                       # default: table
                KingVelocityDF(**kw, speed_method="quadrature"))

    def _speeds(self, df, n=20000, seed=0):
        from progenax.profiles.king import KingProfile

        prof = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)
        masses = jnp.ones(n)
        pos = prof.sample_positions(masses, jax.random.PRNGKey(seed))
        vel = df.sample_velocities(pos, masses, jax.random.PRNGKey(seed + 1),
                                   G=1.0)
        return np.asarray(jnp.linalg.norm(vel, axis=1))

    def test_speed_moments_match_quadrature_oracle(self):
        df_t, df_q = self._two_dfs()
        s_t, s_q = self._speeds(df_t), self._speeds(df_q)
        assert abs(s_t.mean() / s_q.mean() - 1.0) < 0.02
        assert abs((s_t**2).mean() / (s_q**2).mean() - 1.0) < 0.03

    def test_speed_distribution_ks(self):
        from scipy.stats import ks_2samp

        df_t, df_q = self._two_dfs()
        D = ks_2samp(self._speeds(df_t), self._speeds(df_q)).statistic
        assert D < 0.02

    def test_table_default_and_quadrature_static(self):
        df = KingVelocityDF(W0=5.0, r_c=1.0)
        assert df.speed_method == "table"
        with pytest.raises(ValueError, match="speed_method"):
            KingVelocityDF(W0=5.0, r_c=1.0, speed_method="exact")

    def test_g1_lowered_exponential_is_king_weight(self):
        """The exact identity the routing relies on: the LIMEPY lowered
        exponential at g=1 IS the King lowering, E_gamma(1, x) = e^x - 1
        (Gieles & Zocchi 2015 Eq. 2), so SpeedCDFTable.build(W0, g=1)
        tabulates exactly the King speed weight u^2 (exp(W - u^2/2) - 1)."""
        from progenax.profiles.limepy import lowered_exponential

        x = jnp.linspace(0.0, 10.0, 201)
        np.testing.assert_allclose(
            np.asarray(lowered_exponential(jnp.asarray(1.0), x)),
            np.asarray(jnp.exp(x) - 1.0),
            rtol=1e-12,
        )


class TestKingSamplerOptimization:
    """2026-06 sampler-fusion micro-batch: table cached at construction
    (depends only on W0; g=1 fixed) + jitted sampling core."""

    def test_table_cached_at_construction(self):
        from progenax.profiles.limepy_tables import SpeedCDFTable

        df = KingVelocityDF(W0=5.0, r_c=1.0)
        assert isinstance(df.speed_table, SpeedCDFTable)

    def test_quadrature_method_has_no_table(self):
        df = KingVelocityDF(W0=5.0, r_c=1.0,
                            speed_method="quadrature")
        assert df.speed_table is None

    def test_cached_table_bit_identical_to_fresh_build(self):
        from progenax.profiles.limepy_tables import SpeedCDFTable

        df = KingVelocityDF(W0=5.0, r_c=1.0)
        fresh = SpeedCDFTable.build(df.W0, jnp.asarray(1.0))
        np.testing.assert_array_equal(np.asarray(df.speed_table.cdf),
                                      np.asarray(fresh.cdf))

    def test_same_key_same_velocities(self):
        from progenax.profiles.king import KingProfile

        prof = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)
        df = KingVelocityDF(W0=5.0, r_c=1.0)
        m = jnp.ones(500)
        pos = prof.sample_positions(m, jax.random.PRNGKey(0))
        v1 = df.sample_velocities(pos, m, jax.random.PRNGKey(1), G=1.0)
        v2 = df.sample_velocities(pos, m, jax.random.PRNGKey(1), G=1.0)
        np.testing.assert_array_equal(np.asarray(v1), np.asarray(v2))
