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


def _build_ic(W0, g, r_c=1.0, N=5000, seed=0, r_a=None, xi_max=300.0, n_ode=2000):
    from progenax.profiles.limepy import LIMEPYProfile
    from progenax.kinematics.limepy_df import LIMEPYVelocityDF

    prof = LIMEPYProfile.from_W0_rc(W0=W0, g=g, r_c=r_c, r_a=r_a,
                                    xi_max=xi_max, n_ode_points=n_ode)
    df = LIMEPYVelocityDF(W0=W0, g=g, r_c=r_c, r_a=r_a, xi_max=xi_max, n_ode_points=n_ode)
    masses = jnp.ones(N)
    kp, kv = jax.random.split(jax.random.PRNGKey(seed))
    pos = prof.sample_positions(masses, kp)
    pos = pos - jnp.average(pos, axis=0, weights=masses)
    vel = df.sample_velocities(pos, masses, kv, G=G)
    vel = vel - jnp.average(vel, axis=0, weights=masses)
    return prof, df, masses, pos, vel


def _beta_profile(pos, vel, r_edges):
    """Measured anisotropy beta(r) = 1 - sigma_t^2 / (2 sigma_r^2) in radial shells."""
    r = jnp.linalg.norm(pos, axis=1)
    r_hat = pos / (r[:, None] + 1e-30)
    v_r = jnp.sum(vel * r_hat, axis=1)
    v_t2 = jnp.sum(vel**2, axis=1) - v_r**2  # |v_t|^2 (two tangential components)
    betas, centers = [], []
    for lo, hi in zip(r_edges[:-1], r_edges[1:]):
        m = (r >= lo) & (r < hi)
        if int(jnp.sum(m)) < 50:
            continue
        sr2 = jnp.mean(v_r[m] ** 2)
        st2 = jnp.mean(v_t2[m])  # = sigma_theta^2 + sigma_phi^2
        betas.append(float(1.0 - st2 / (2.0 * sr2)))
        centers.append(0.5 * (lo + hi))
    return np.array(centers), np.array(betas)


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


class TestLimepyAnisotropicVelocity:
    """The anisotropic (Michie/OM) velocity DF: radially biased, true equilibrium,
    matching MichieVelocityDF at g=1.
    """

    def test_sampler_core_isotropic_outskirts_radial(self):
        """Direct unit test of the speed-angle sampler: at small s = r/r_a the
        directions are isotropic (beta ~ 0), and at large s they are radially biased
        (beta > 0). This is the kinematic content of the Michie/OM term, isolated from
        the profile/shell binning."""
        from progenax.kinematics.limepy_df import _sample_speed_angle

        keys = jax.random.split(jax.random.PRNGKey(0), 20000)
        for s_val, lo, hi, label in [(0.05, -0.05, 0.05, "core->isotropic"),
                                     (1.0, 0.25, 0.55, "outskirts->radial")]:
            ur, ut = jax.vmap(lambda k: _sample_speed_angle(
                k, jnp.asarray(7.0), jnp.asarray(s_val), jnp.asarray(1.0), 256, 128))(keys)
            beta = 1.0 - float(jnp.mean(ut**2)) / (2.0 * float(jnp.mean(ur**2)))
            assert lo < beta < hi, f"{label}: s={s_val} beta={beta:.3f} not in ({lo},{hi})"

    def test_aniso_unscaled_virial_is_half(self):
        """A finite r_a anisotropic LIMEPY cluster is STILL a true equilibrium:
        Q = T/|V| = 0.5 unscaled (the anisotropic density sets mu, hence the velocity
        scale, self-consistently). Uses r_a/r_c=8 at W0=7 (a finite-r_t model; smaller
        r_a hits the radial-orbit / infinite-mass regime the solver now refuses)."""
        from progenax.builders import compute_kinetic_energy, compute_potential_energy

        Qs = []
        for seed in range(4):
            _, _, m, pos, vel = _build_ic(W0=7.0, g=1.0, r_a=8.0, N=6000, seed=seed,
                                          xi_max=800.0, n_ode=3000)
            T = compute_kinetic_energy(vel, m)
            V = compute_potential_energy(pos, m, G=G)
            Qs.append(float(T / jnp.abs(V)))
        Q = float(np.mean(Qs))
        assert abs(Q - 0.5) < 0.04, f"aniso unscaled Q={Q:.3f} (expected 0.5)"

    def test_radial_anisotropy_increases_outward(self):
        """The defining Michie/OM kinematic signature in a full IC: beta(r) is small in
        the core (r << r_a) and rises (radially biased) outward. r_a/r_c=8, r_t~56."""
        _, df, m, pos, vel = _build_ic(W0=7.0, g=1.0, r_a=8.0, N=60000, seed=1,
                                       xi_max=800.0, n_ode=3000)
        r_t = float(df.r_t)
        edges = np.linspace(0.0, 0.8 * r_t, 8)
        centers, betas = _beta_profile(pos, vel, edges)
        assert betas[0] < 0.2, f"core not near-isotropic: beta={betas[0]:.2f}"
        assert betas[-1] > betas[0] + 0.15, f"beta not rising outward: {betas}"
        assert np.all(betas > -0.15), f"unexpected tangential bias: {betas}"

    def test_g1_aniso_matches_michie_velocity_df(self):
        """g=1 anisotropic LIMEPY velocity sampling reproduces MichieVelocityDF: same
        self-consistent velocity scale and same beta(r) anisotropy profile."""
        from progenax.profiles.michie import MichieProfile
        from progenax.kinematics.michie_df import MichieVelocityDF
        from progenax.kinematics.limepy_df import LIMEPYVelocityDF

        W0, r_a = 7.0, 8.0
        lim = LIMEPYVelocityDF(W0=W0, g=1.0, r_c=1.0, r_a=r_a, xi_max=800.0, n_ode_points=3000)
        mic = MichieVelocityDF(W0=W0, r_c=1.0, r_a=r_a, xi_max=800.0, n_ode_points=3000)
        M = jnp.asarray(6000.0)
        # velocity scale s (LIMEPY) == sigma (Michie)
        s_lim = float(lim._s(M, G))
        s_mic = float(jnp.sqrt(G * M / (9.0 * mic.r_c * mic.mu)))
        np.testing.assert_allclose(s_lim, s_mic, rtol=1e-3)

        # beta(r) profiles agree within sampling noise
        N = 60000
        prof = MichieProfile.from_W0_rc(W0, 1.0, r_a)
        kp, kvl, kvm = jax.random.split(jax.random.PRNGKey(7), 3)
        pos = prof.sample_positions(jnp.ones(N), kp)
        pos = pos - jnp.mean(pos, axis=0)
        vel_l = lim.sample_velocities(pos, jnp.ones(N), kvl, G=G)
        vel_m = mic.sample_velocities(pos, jnp.ones(N), kvm, G=G)
        edges = np.linspace(0.0, 0.8 * float(prof.r_t), 6)
        _, bl = _beta_profile(pos, vel_l, edges)
        _, bm = _beta_profile(pos, vel_m, edges)
        np.testing.assert_allclose(bl, bm, atol=0.06)

    def test_aniso_all_bound(self):
        _, df, m, pos, vel = _build_ic(W0=7.0, g=1.5, r_a=8.0, N=4000, seed=2,
                                       xi_max=800.0, n_ode=3000)
        r = jnp.linalg.norm(pos, axis=1)
        v = jnp.linalg.norm(vel, axis=1)
        W = jnp.interp(r / df.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)
        s = df._s(jnp.sum(m), G)
        v_esc = s * jnp.sqrt(2.0 * jnp.maximum(W, 0.0))
        assert float(jnp.mean(v <= v_esc + 0.05 * s)) == 1.0


class TestLimepyTableRouting:
    """speed_method='table' (default) must agree with the exact quadrature
    oracle (speed_method='quadrature') distributionally and in moments —
    the same contract AnisoSpeedCDFTable passed against the DF quadrature
    (tests/unit/profiles/test_limepy_tables.py: moments to 1.5%)."""

    def _two_dfs(self, r_a):
        from progenax.kinematics.limepy_df import LIMEPYVelocityDF

        kw = dict(W0=5.0, g=1.0, r_c=1.0, r_a=r_a)
        return (LIMEPYVelocityDF(**kw),                       # default: table
                LIMEPYVelocityDF(**kw, speed_method="quadrature"))

    def _pos_vel(self, df, n=30000, seed=0):
        from progenax.profiles.limepy import LIMEPYProfile

        prof = LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0)
        masses = jnp.ones(n)
        pos = prof.sample_positions(masses, jax.random.PRNGKey(seed))
        vel = df.sample_velocities(pos, masses, jax.random.PRNGKey(seed + 1),
                                   G=1.0)
        return pos, vel

    def _speeds(self, df, n=30000, seed=0):
        _, vel = self._pos_vel(df, n=n, seed=seed)
        return np.asarray(jnp.linalg.norm(vel, axis=1))

    @pytest.mark.parametrize("r_a", [None, 4.0])
    def test_speed_moments_match_quadrature_oracle(self, r_a):
        df_t, df_q = self._two_dfs(r_a)
        s_t, s_q = self._speeds(df_t), self._speeds(df_q)
        assert abs(s_t.mean() / s_q.mean() - 1.0) < 0.02
        assert abs((s_t**2).mean() / (s_q**2).mean() - 1.0) < 0.03

    @pytest.mark.parametrize("r_a", [None, 4.0])
    def test_speed_distribution_ks(self, r_a):
        from scipy.stats import ks_2samp

        df_t, df_q = self._two_dfs(r_a)
        D = ks_2samp(self._speeds(df_t), self._speeds(df_q)).statistic
        assert D < 0.02

    def test_aniso_beta_profile_preserved(self):
        """The table path must keep the validated beta(r): the angular
        conditional stays EXACT, so only the speed marginal changed."""
        df_t, df_q = self._two_dfs(4.0)
        pos, vel_t = self._pos_vel(df_t, n=60000, seed=2)
        _, vel_q = self._pos_vel(df_q, n=60000, seed=2)
        edges = np.linspace(0.0, 0.8 * float(df_t.r_t), 6)
        _, beta_t = _beta_profile(pos, vel_t, edges)
        _, beta_q = _beta_profile(pos, vel_q, edges)
        np.testing.assert_allclose(beta_t, beta_q, atol=0.06)

    def test_table_default_and_quadrature_static(self):
        from progenax.kinematics.limepy_df import LIMEPYVelocityDF

        df = LIMEPYVelocityDF(W0=5.0, g=1.0, r_c=1.0)
        assert df.speed_method == "table"
        with pytest.raises(ValueError, match="speed_method"):
            LIMEPYVelocityDF(W0=5.0, g=1.0, r_c=1.0, speed_method="exact")

    def test_differentiable_in_g_through_table(self):
        from progenax.kinematics.limepy_df import LIMEPYVelocityDF

        def mean_ke(g):
            df = LIMEPYVelocityDF(W0=5.0, g=g, r_c=1.0, r_a=4.0)
            prof_pos = jnp.array([[0.5, 0.0, 0.0]] * 64)
            v = df.sample_velocities(prof_pos, jnp.ones(64),
                                     jax.random.PRNGKey(0), G=1.0)
            return jnp.mean(jnp.sum(v**2, axis=1))

        g = jax.grad(mean_ke)(1.0)
        assert jnp.isfinite(g) and g != 0.0


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
