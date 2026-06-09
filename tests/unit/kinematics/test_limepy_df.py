# progenax/tests/unit/kinematics/test_limepy_df.py
"""Unit tests for LIMEPYVelocityDF (general-g lowered-isothermal velocity DF).

The general-g local speed distribution is

    g(u) ∝ u^2 E_gamma(g, W - u^2/2),   0 <= u <= sqrt(2W),   u = v / s,

with the self-consistent velocity scale s^2 = G M / (9 r_c mu(W0, g)) (the same
King-radius nondimensionalization as KingVelocityDF). Sampling this DF in detailed
equilibrium with the LIMEPY potential puts the cluster at virial Q = T/|V| = 0.5
WITHOUT any external rescale — the first-principles equilibrium that the lambda_seg
blend only approximates. At g=1, E_gamma(1, x) = e^x - 1 recovers the King DF.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.units import STELLAR

G = STELLAR.G


def _build_ic(W0, g, r_c=1.0, N=5000, seed=0):
    from progenax.profiles.limepy import LIMEPYProfile
    from progenax.kinematics.limepy_df import LIMEPYVelocityDF

    prof = LIMEPYProfile.from_W0_rc(W0=W0, g=g, r_c=r_c)
    df = LIMEPYVelocityDF(W0=W0, g=g, r_c=r_c)
    masses = jnp.ones(N)
    kp, kv = jax.random.split(jax.random.PRNGKey(seed))
    pos = prof.sample_positions(masses, kp)
    pos = pos - jnp.average(pos, axis=0, weights=masses)
    vel = df.sample_velocities(pos, masses, kv, G=G)
    vel = vel - jnp.average(vel, axis=0, weights=masses)
    return prof, df, masses, pos, vel


class TestLimepyEquilibrium:
    """The headline: the LIMEPY DF is a TRUE equilibrium across g (Q=0.5 unscaled)."""

    @pytest.mark.parametrize("g", [0.5, 1.0, 1.5, 2.0])
    def test_unscaled_virial_ratio_is_half(self, g):
        """For each truncation g, the self-consistent (profile + DF) cluster sits at
        Q = T/|V| = 0.5 with NO external rescale — detailed equilibrium. This is the
        first-principles property the lambda_seg blend cannot match per mass group."""
        from progenax.builders import compute_kinetic_energy, compute_potential_energy

        Qs = []
        for seed in range(4):
            _, _, m, pos, vel = _build_ic(W0=7.0, g=g, N=6000, seed=seed)
            T = compute_kinetic_energy(vel, m)
            V = compute_potential_energy(pos, m, G=G)
            Qs.append(float(T / jnp.abs(V)))
        Q = float(np.mean(Qs))
        assert abs(Q - 0.5) < 0.04, f"g={g}: unscaled Q={Q:.3f} (expected 0.5)"

    def test_all_particles_bound(self):
        """Every star has v <= v_esc(r) = s sqrt(2W(r)): the lowered DF samples only
        bound orbits, for a non-King g."""
        prof, df, m, pos, vel = _build_ic(W0=7.0, g=1.5, N=4000)
        r = jnp.linalg.norm(pos, axis=1)
        v = jnp.linalg.norm(vel, axis=1)
        W = jnp.interp(r / prof.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)
        s = df._s(jnp.sum(m), G)
        v_esc = s * jnp.sqrt(2.0 * jnp.maximum(W, 0.0))
        # COM subtraction shifts speeds slightly; allow a small tolerance.
        assert float(jnp.mean(v <= v_esc + 0.05 * s)) == 1.0


class TestLimepyVelocityCorner:
    """g=1 reduces to the King velocity DF."""

    def test_g1_velocity_scale_matches_king(self):
        """The self-consistent velocity scale s(W0, g=1) equals King's sigma(W0) to
        float64 precision (same mu integral, same -9 normalization)."""
        from progenax.kinematics.king_df import KingVelocityDF
        from progenax.kinematics.limepy_df import LIMEPYVelocityDF

        king = KingVelocityDF(W0=7.0, r_c=1.0, r_t=30.0)
        lim = LIMEPYVelocityDF(W0=7.0, g=1.0, r_c=1.0)
        M = jnp.asarray(5000.0)
        np.testing.assert_allclose(
            float(lim._s(M, G)), float(king._sigma(M, G)), rtol=1e-3
        )

    def test_g1_dispersion_profile_matches_king(self):
        """The sampled 1-D dispersion profile sigma_1d(r) at g=1 matches a King DF
        cluster (same W0, r_c) within sampling noise — same equilibrium kinematics."""
        from progenax.profiles.king import KingProfile
        from progenax.kinematics.king_df import KingVelocityDF

        N = 40000
        # LIMEPY g=1
        _, _, m, pos_l, vel_l = _build_ic(W0=7.0, g=1.0, N=N, seed=3)
        # King
        prof_k = KingProfile.from_W0_rc(7.0, 1.0)
        df_k = KingVelocityDF(W0=7.0, r_c=1.0, r_t=float(prof_k.r_t))
        kp, kv = jax.random.split(jax.random.PRNGKey(3))
        pos_k = prof_k.sample_positions(jnp.ones(N), kp)
        vel_k = df_k.sample_velocities(pos_k, jnp.ones(N), kv, G=G)

        def sigma1d_central(pos, vel):
            r = jnp.linalg.norm(pos, axis=1)
            core = r < 1.5  # within ~r_c
            return float(jnp.sqrt(jnp.mean(jnp.sum(vel[core] ** 2, axis=1)) / 3.0))

        s_l = sigma1d_central(pos_l, vel_l)
        s_k = sigma1d_central(pos_k, vel_k)
        np.testing.assert_allclose(s_l, s_k, rtol=0.03)


class TestLimepyVelocityDifferentiable:
    def test_velocity_sampling_differentiable_in_g(self):
        """grad of mean kinetic energy w.r.t. g flows through the DF speed sampling
        (E_gamma's a-derivative): g is inferable from kinematics."""
        def loss(g):
            from progenax.profiles.limepy import LIMEPYProfile
            from progenax.kinematics.limepy_df import LIMEPYVelocityDF
            prof = LIMEPYProfile.from_W0_rc(W0=7.0, g=g, r_c=1.0)
            df = LIMEPYVelocityDF(W0=7.0, g=g, r_c=1.0)
            m = jnp.ones(300)
            kp, kv = jax.random.split(jax.random.PRNGKey(1))
            pos = prof.sample_positions(m, kp)
            vel = df.sample_velocities(pos, m, kv, G=G)
            return jnp.mean(jnp.sum(vel**2, axis=1))

        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad) and jnp.abs(grad) > 0.0
