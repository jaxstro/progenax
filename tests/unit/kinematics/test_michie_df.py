"""MichieVelocityDF: the 2-D (v_r, v_t) sampler for the Michie-King anisotropic model.

Headline checks: beta(r) ~ 0 at the centre and increases outward (radial anisotropy);
virial equilibrium Q ~ 0.5 unscaled; large r_a -> isotropic; JIT.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.units import STELLAR
from progenax.profiles.michie import MichieProfile

G = STELLAR.G


def _shell(r, N, seed):
    dirs = jax.random.normal(jax.random.PRNGKey(seed), (N, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    return r * dirs


def _beta(v, pos):
    r_hat = pos / jnp.linalg.norm(pos, axis=1, keepdims=True)
    v_r = jnp.sum(v * r_hat, axis=1)
    v_t2 = jnp.sum(v**2, axis=1) - v_r**2
    return 1.0 - jnp.mean(v_t2) / (2.0 * jnp.mean(v_r**2))


class TestMichieVelocityDF:
    def test_beta_isotropic_center_radial_outward(self):
        from progenax.kinematics.michie_df import MichieVelocityDF

        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0)
        N = 40000
        betas = []
        for r, seed in [(1.0, 0), (8.0, 1), (25.0, 2)]:
            pos = _shell(r, N, seed)
            v = df.sample_velocities(pos, jnp.ones(N), jax.random.PRNGKey(seed + 10), G=G)
            betas.append(float(_beta(v, pos)))
        assert abs(betas[0]) < 0.06, f"beta(center)={betas[0]:.3f} should be ~0"
        assert betas[0] < betas[1] < betas[2], f"beta must increase outward: {betas}"
        assert betas[2] > 0.3, f"outer beta={betas[2]:.3f} should be clearly radial"

    def test_large_r_a_isotropic(self):
        from progenax.kinematics.michie_df import MichieVelocityDF

        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=1e4)
        N = 40000
        pos = _shell(10.0, N, seed=3)
        v = df.sample_velocities(pos, jnp.ones(N), jax.random.PRNGKey(13), G=G)
        assert abs(float(_beta(v, pos))) < 0.05, "large r_a must be ~isotropic"

    def test_virial_equilibrium(self):
        from progenax.kinematics.michie_df import MichieVelocityDF
        from progenax.dynamics.virial import compute_virial_ratio

        N = 4000
        masses = jnp.ones(N)
        prof = MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0)
        pos = prof.sample_positions(masses, jax.random.PRNGKey(0))
        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0)
        v = df.sample_velocities(pos, masses, jax.random.PRNGKey(1), G=G)
        Q = float(compute_virial_ratio(pos, v, masses, G=G))
        assert abs(Q - 0.5) < 0.08, f"Michie-King Q={Q:.3f} should be ~0.5 unscaled"

    def test_jit_compatible(self):
        from progenax.kinematics.michie_df import MichieVelocityDF

        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0)
        N = 128
        pos = _shell(5.0, N, seed=4)
        v = jax.jit(lambda p, m, k: df.sample_velocities(p, m, k, G=G))(
            pos, jnp.ones(N), jax.random.PRNGKey(5)
        )
        assert v.shape == (128, 3) and jnp.all(jnp.isfinite(v))

    def test_grad_wrt_W0_matches_fd(self):
        """jax.grad flows through the anisotropic ODE solve + 2-D sampler (W0)."""
        from progenax.kinematics.michie_df import MichieVelocityDF

        pos = jnp.array([[2.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]])
        masses = jnp.ones(4)
        key = jax.random.PRNGKey(0)

        def loss(W0):
            df = MichieVelocityDF(W0=W0, r_c=1.0, r_a=8.0)
            v = df.sample_velocities(pos, masses, key, G=G)
            return jnp.mean(jnp.sum(v**2, axis=1))

        g = jax.grad(loss)(7.0)
        g_fd = (loss(7.0 + 1e-3) - loss(7.0 - 1e-3)) / 2e-3
        assert jnp.isfinite(g), "grad through the Michie ODE+sampler must be finite"
        assert jnp.abs(g - g_fd) <= 5e-2 * jnp.abs(g_fd) + 1e-9, (
            f"grad d<|v|^2>/dW0={float(g)} vs FD {float(g_fd)}"
        )


class TestMichieTableRouting:
    """speed_method='table' (default) must agree with the exact 2-D
    (u_r, u_t) quadrature oracle (speed_method='quadrature')
    distributionally and in moments — the same contract the LIMEPY aniso
    routing passed (tests/unit/kinematics/test_limepy_df.py::
    TestLimepyTableRouting); the underlying joint is identical (Michie's
    exp(-s^2 u_t^2 / 2) IS exp(-(s^2 u^2 / 2)(1 - cos^2 theta)) under
    u_t = u sin(theta)).

    N is capped at 2e4 for every draw through the quadrature oracle (the
    review-mandated oracle-N convention): the two-sample KS 95% critical D
    at n=2e4 is ~0.0136 < the 0.02 threshold."""

    def _two_dfs(self):
        from progenax.kinematics.michie_df import MichieVelocityDF

        kw = dict(W0=7.0, r_c=1.0, r_a=8.0)
        return (MichieVelocityDF(**kw),                       # default: table
                MichieVelocityDF(**kw, speed_method="quadrature"))

    def _pos_vel(self, df, n=20000, seed=0):
        masses = jnp.ones(n)
        prof = MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0)
        pos = prof.sample_positions(masses, jax.random.PRNGKey(seed))
        vel = df.sample_velocities(pos, masses, jax.random.PRNGKey(seed + 1),
                                   G=G)
        return pos, vel

    def _speeds(self, df, n=20000, seed=0):
        _, vel = self._pos_vel(df, n=n, seed=seed)
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

    def test_beta_profile_preserved(self):
        """The table path must keep the validated beta(r): the angular
        conditional stays EXACT, so only the speed marginal changed."""
        df_t, df_q = self._two_dfs()
        N = 20000
        betas_t, betas_q = [], []
        for r, seed in [(1.0, 0), (8.0, 1), (25.0, 2)]:
            pos = _shell(r, N, seed)
            key = jax.random.PRNGKey(seed + 10)
            v_t = df_t.sample_velocities(pos, jnp.ones(N), key, G=G)
            v_q = df_q.sample_velocities(pos, jnp.ones(N), key, G=G)
            betas_t.append(float(_beta(v_t, pos)))
            betas_q.append(float(_beta(v_q, pos)))
        np.testing.assert_allclose(betas_t, betas_q, atol=0.06)

    def test_table_default_and_quadrature_static(self):
        from progenax.kinematics.michie_df import MichieVelocityDF

        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0)
        assert df.speed_method == "table"
        with pytest.raises(ValueError, match="speed_method"):
            MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0, speed_method="exact")


class TestMichieSamplerOptimization:
    """2026-06 sampler-fusion micro-batch: aniso table cached at construction
    (depends only on (W0, r_t/r_a); g=1 fixed) + jitted sampling core."""

    def test_table_cached_at_construction(self):
        from progenax.kinematics.michie_df import MichieVelocityDF
        from progenax.profiles.limepy_tables import AnisoSpeedCDFTable

        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0)
        assert isinstance(df.speed_table, AnisoSpeedCDFTable)

    def test_quadrature_method_has_no_table(self):
        from progenax.kinematics.michie_df import MichieVelocityDF

        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0,
                              speed_method="quadrature")
        assert df.speed_table is None

    def test_cached_table_bit_identical_to_fresh_build(self):
        from progenax.kinematics.michie_df import MichieVelocityDF
        from progenax.profiles.king import _find_tidal_radius
        from progenax.profiles.limepy_tables import AnisoSpeedCDFTable

        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0)
        p_box = jnp.maximum(
            df.r_c * _find_tidal_radius(df.xi_grid, df.psi_grid) / df.r_a, 1e-3)
        fresh = AnisoSpeedCDFTable.build(df.W0, p_box, jnp.asarray(1.0))
        np.testing.assert_array_equal(np.asarray(df.speed_table.cdf),
                                      np.asarray(fresh.cdf))

    def test_same_key_same_velocities(self):
        from progenax.kinematics.michie_df import MichieVelocityDF

        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0)
        pos = _shell(5.0, 500, seed=0)
        m = jnp.ones(500)
        v1 = df.sample_velocities(pos, m, jax.random.PRNGKey(1), G=G)
        v2 = df.sample_velocities(pos, m, jax.random.PRNGKey(1), G=G)
        np.testing.assert_array_equal(np.asarray(v1), np.asarray(v2))
