"""
Physics validation tests for the Michie-King anisotropic model.

The self-consistent Michie (1963) radial-anisotropy term on King's (1966) lowered
cutoff. Tests anchor the quantitative physics with explicit tolerances:
- Michie (1963), MNRAS 126, 499 (anisotropy term)
- King (1966), AJ 71, 64 (lowered-isothermal cutoff)
- Binney & Tremaine (2008), Galactic Dynamics (Osipkov-Merritt beta(r))

These validate the *existing* implementation (already unit-tested); each asserts a
measured-vs-expected match the validation figures (scripts/validate_michie.py) also
reproduce.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.builders import compute_kinetic_energy, compute_potential_energy
from progenax.kinematics import MichieVelocityDF
from progenax.profiles import KingProfile, MichieProfile

G = 1.0
W0, RC = 7.0, 1.0
RA_ISO = 1.0e4  # r_a >> r_t  -> isotropic King limit
RA_ANISO = 8.0  # clear radial anisotropy that still truncates at W0=7 (r_t/r_c~56)


def _radial_tangential(positions, velocities):
    """Decompose velocities into radial (signed) and tangential (magnitude)."""
    radii = jnp.linalg.norm(positions, axis=1)
    r_hat = positions / (radii[:, None] + 1e-30)
    v_r = jnp.sum(velocities * r_hat, axis=1)
    v_t = jnp.linalg.norm(velocities - v_r[:, None] * r_hat, axis=1)
    return radii, v_r, v_t


def _michie_beta_oracle(W, s, n=400):
    """Analytic anisotropy beta(W,s) from the 2nd moments of the *exact* sampled
    density p(u_r,u_t) proportional to u_t exp(-s^2 u_t^2/2)[exp(W-(u_r^2+u_t^2)/2)-1]
    on u_r^2+u_t^2 < 2W. This is the model's own beta -- the lowering term breaks the
    pure f(Q) form, so it sits *below* the Osipkov-Merritt ceiling r^2/(r^2+r_a^2).
    """
    if W <= 0:
        return 0.0
    umax = np.sqrt(2.0 * W)
    ur = np.linspace(-umax, umax, n)
    ut = np.linspace(0.0, umax, n)
    UR, UT = np.meshgrid(ur, ut)
    bound = UR**2 + UT**2 < 2.0 * W
    w = UT * np.exp(-(s**2) * UT**2 / 2.0) * (np.exp(W - (UR**2 + UT**2) / 2.0) - 1.0)
    w = np.where(bound, np.maximum(w, 0.0), 0.0)
    norm = w.sum()
    ur2 = (w * UR**2).sum() / norm
    ut2 = (w * UT**2).sum() / norm
    return 1.0 - ut2 / (2.0 * ur2)


def _beta_binned(positions, velocities, edges):
    """Anisotropy beta(r) = 1 - <v_t^2>/(2 <v_r^2>) in radial bins."""
    radii, v_r, v_t = _radial_tangential(positions, velocities)
    mids, beta = [], []
    for lo, hi in edges:
        m = (radii >= lo) & (radii < hi)
        if int(jnp.sum(m)) < 200:
            continue
        sr2 = float(jnp.mean(v_r[m] ** 2))
        st2 = float(jnp.mean(v_t[m] ** 2))
        mids.append(0.5 * (lo + hi))
        beta.append(1.0 - st2 / (2.0 * sr2))
    return np.array(mids), np.array(beta)


class TestMichieIsotropicLimit:
    def test_density_matches_king_at_large_ra(self):
        """r_a -> infinity recovers the King density profile."""
        king = KingProfile.from_W0_rc(W0, RC)
        mich = MichieProfile.from_W0_rc(W0, RC, RA_ISO)
        r = jnp.linspace(0.05, 0.9 * float(king.r_t), 40)
        rk = np.asarray(king.density(r))
        rm = np.asarray(mich.density(r))
        rel = np.max(np.abs(rm / rm[0] - rk / rk[0]))
        assert rel < 1e-2, f"Michie(r_a=1e4) vs King max rel = {rel:.2e}"

    def test_beta_near_zero_at_large_ra(self):
        """The isotropic limit has beta(r) ~ 0 everywhere."""
        prof = MichieProfile.from_W0_rc(W0, RC, RA_ISO)
        df = MichieVelocityDF(W0=W0, r_c=RC, r_a=RA_ISO)
        m = jnp.ones(40_000)
        kp, kv = jax.random.split(jax.random.PRNGKey(0))
        pos = prof.sample_positions(m, kp)
        vel = df.sample_velocities(pos, m, kv, G=G)
        r_t = float(prof.r_t)
        edges = [(0.2 * r_t, 0.4 * r_t), (0.4 * r_t, 0.6 * r_t), (0.6 * r_t, 0.8 * r_t)]
        _, beta = _beta_binned(pos, vel, edges)
        assert np.all(np.abs(beta) < 0.05), f"isotropic-limit beta = {beta}"


class TestMichieAnisotropyProfile:
    def test_beta_matches_df_oracle(self):
        """Sampled beta(r) matches the Michie-King DF's own analytic beta (2nd-moment
        oracle). This validates the sampler faithfully reproduces the model anisotropy.
        """
        prof = MichieProfile.from_W0_rc(W0, RC, RA_ANISO)
        df = MichieVelocityDF(W0=W0, r_c=RC, r_a=RA_ANISO)
        m = jnp.ones(60_000)
        kp, kv = jax.random.split(jax.random.PRNGKey(1))
        pos = prof.sample_positions(m, kp)
        vel = df.sample_velocities(pos, m, kv, G=G)
        edges = [(1.0, 3.0), (4.0, 7.0), (9.0, 15.0), (18.0, 30.0)]
        mids, beta = _beta_binned(pos, vel, edges)
        # oracle at each bin midpoint: interpolate W(r) from the ODE, s = r/r_a
        W_mid = np.asarray(
            jnp.interp(
                jnp.asarray(mids) / RC, df.xi_grid, df.psi_grid, left=df.W0, right=0.0
            )
        )
        beta_oracle = np.array(
            [
                _michie_beta_oracle(float(w), float(r) / RA_ANISO)
                for w, r in zip(W_mid, mids)
            ]
        )
        assert np.all(np.abs(beta - beta_oracle) < 0.05), (
            f"sampled beta={beta} vs DF oracle={beta_oracle}"
        )

    def test_beta_below_osipkov_merritt_ceiling(self):
        """The King energy cutoff suppresses beta below the pure Osipkov-Merritt
        ceiling r^2/(r^2+r_a^2) -- a defining property of the *lowered* Michie model."""
        prof = MichieProfile.from_W0_rc(W0, RC, RA_ANISO)
        df = MichieVelocityDF(W0=W0, r_c=RC, r_a=RA_ANISO)
        m = jnp.ones(60_000)
        kp, kv = jax.random.split(jax.random.PRNGKey(1))
        pos = prof.sample_positions(m, kp)
        vel = df.sample_velocities(pos, m, kv, G=G)
        mids, beta = _beta_binned(pos, vel, [(4.0, 7.0), (9.0, 15.0), (18.0, 30.0)])
        beta_om = mids**2 / (mids**2 + RA_ANISO**2)
        assert np.all(beta < beta_om), (
            f"beta={beta} should be below OM ceiling {beta_om}"
        )

    def test_beta_increases_outward(self):
        """Radial anisotropy grows with radius (beta_inner < beta_outer)."""
        prof = MichieProfile.from_W0_rc(W0, RC, RA_ANISO)
        df = MichieVelocityDF(W0=W0, r_c=RC, r_a=RA_ANISO)
        m = jnp.ones(60_000)
        kp, kv = jax.random.split(jax.random.PRNGKey(2))
        pos = prof.sample_positions(m, kp)
        vel = df.sample_velocities(pos, m, kv, G=G)
        _, beta = _beta_binned(pos, vel, [(1.0, 3.0), (18.0, 30.0)])
        assert beta[0] < beta[-1], f"beta should increase outward: {beta}"


class TestMichieEquilibrium:
    def test_virial_ratio_half_unscaled(self):
        """Unscaled virial Q = T/|V| ~ 0.5 (no external rescale)."""
        prof = MichieProfile.from_W0_rc(W0, RC, RA_ANISO)
        df = MichieVelocityDF(W0=W0, r_c=RC, r_a=RA_ANISO)
        m = jnp.ones(5_000)
        kp, kv = jax.random.split(jax.random.PRNGKey(3))
        pos = prof.sample_positions(m, kp)
        vel = df.sample_velocities(pos, m, kv, G=G)
        Q = float(
            compute_kinetic_energy(vel, m)
            / jnp.abs(compute_potential_energy(pos, m, G=G))
        )
        assert abs(Q - 0.5) < 0.05, f"unscaled Q = {Q:.3f}"

    def test_all_particles_bound(self):
        """100% of velocities satisfy v <= v_esc(r) = sigma sqrt(2 W(r))."""
        prof = MichieProfile.from_W0_rc(W0, RC, RA_ANISO)
        df = MichieVelocityDF(W0=W0, r_c=RC, r_a=RA_ANISO)
        m = jnp.ones(5_000)
        kp, kv = jax.random.split(jax.random.PRNGKey(4))
        pos = prof.sample_positions(m, kp)
        vel = df.sample_velocities(pos, m, kv, G=G)
        radii = jnp.linalg.norm(pos, axis=1)
        W = jnp.maximum(
            jnp.interp(radii / df.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0),
            0.0,
        )
        sigma = jnp.sqrt(G * jnp.sum(m) / (9.0 * df.r_c * df.mu))
        v_esc = sigma * jnp.sqrt(2.0 * W)
        v = jnp.linalg.norm(vel, axis=1)
        assert float(jnp.mean(v <= v_esc + 1e-9)) == 1.0


class TestMichieAnisotropyStructure:
    def test_more_anisotropic_more_extended(self):
        """Stronger anisotropy (smaller r_a) yields a more extended model
        (larger tidal radius r_t at fixed W0, r_c)."""
        rt_aniso = float(MichieProfile.from_W0_rc(W0, RC, RA_ANISO).r_t)
        rt_iso = float(MichieProfile.from_W0_rc(W0, RC, RA_ISO).r_t)
        assert rt_aniso > rt_iso, f"r_t(aniso)={rt_aniso} !> r_t(iso)={rt_iso}"

    def test_too_anisotropic_raises(self):
        """Excessive anisotropy (no finite tidal radius) raises ValueError."""
        with pytest.raises(ValueError):
            MichieProfile.from_W0_rc(W0, RC, 0.1)


class TestMichieDifferentiability:
    # AD-vs-FD for the Michie density observable log rho(r=1.5) is owned by the grad-audit
    # registry (tests/validation/grad_audit/registry.py :: MichieProfile.density[log rho(r)],
    # which has BOTH the r_c AND the W0 channel of the verbatim log rho(1.5) observable at
    # r_a=8). NOTE: the density() formula is a DIFFERENT code path than the inverse-CDF
    # sampler MichieProfile.sample_positions[W0], so the W0 density channel has its OWN
    # registry case (added in the 4.2b review-fix) rather than relying on the sampler case.
    # See docs/website/50-validation/differentiability-audit.md. The former
    # test_grad_profile_observable was removed here (audit T6 consolidation; registry is SoT).
    # test_grad_wrt_mass_velocity_scale (closed-form sigma(M)) and all physics tests stay.
    def test_grad_wrt_mass_velocity_scale(self):
        """Velocity scale sigma ~ sqrt(M) is differentiable in total mass."""

        def sigma(M):
            df = MichieVelocityDF(W0=W0, r_c=RC, r_a=RA_ANISO)
            return jnp.sqrt(G * M / (9.0 * df.r_c * df.mu))

        ad = float(jax.grad(sigma)(1000.0))
        fd = float((sigma(1000.0 + 1.0) - sigma(1000.0 - 1.0)) / 2.0)
        rel = abs(ad - fd) / (abs(ad) + abs(fd) + 1e-30)
        assert np.isfinite(ad) and rel < 1e-5, f"AD={ad}, FD={fd}, rel={rel:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
