"""Unit tests for per-mass-group virial diagnostics (Phase 0 of the LIMEPY plan).

The per-group scalar virial theorem: a subsystem (mass group j) in steady state in the
TOTAL gravitational field satisfies 2 T_j + W_j = 0, with
W_j = sum_{i in j} m_i r_i . a_i  (a_i = acceleration from ALL stars).
So Q_j == T_j / |W_j| == 0.5 at equilibrium, matching the global convention.

Exact anchor: the Clausius identity sum_i m_i r_i . a_i = V (the 1/r potential energy),
so n_groups=1 must reproduce the existing compute_virial_ratio.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.units import STELLAR

G = STELLAR.G


def _equal_mass_plummer(key, N=300, Q=0.5):
    """An equal-mass Plummer cluster rescaled to global virial ratio Q (COM frame)."""
    from progenax import PlummerProfile, PlummerVelocityDF, build_spatial_ic
    from progenax.dynamics.virial import rescale_velocities_to_virial
    masses = jnp.ones(N)
    kpos, kvel = jax.random.split(key)
    profile = PlummerProfile(r_h=1.0)
    df = PlummerVelocityDF(r_h=1.0)
    ic = build_spatial_ic(profile, masses, df, key=key, G=G)
    pos, vel = ic.positions, ic.velocities
    # center
    pos = pos - jnp.average(pos, axis=0, weights=masses)
    vel = vel - jnp.average(vel, axis=0, weights=masses)
    vel = rescale_velocities_to_virial(pos, vel, masses, G=G, target_Q=Q)
    return pos, vel, masses


# --------------------------------------------------------------------------
# mass_group_masks
# --------------------------------------------------------------------------
class TestMassGroupMasks:
    def test_shape_and_partition(self):
        from progenax.dynamics.virial import mass_group_masks

        masses = jnp.array([0.5, 1.0, 3.0, 0.2, 8.0, 2.0])
        masks = mass_group_masks(masses, n_groups=3)
        assert masks.shape == (3, 6)
        # disjoint + cover all
        assert jnp.all(jnp.sum(masks, axis=0) == 1)
        assert int(jnp.sum(masks)) == 6

    def test_ordered_light_to_heavy(self):
        from progenax.dynamics.virial import mass_group_masks

        masses = jnp.linspace(0.1, 10.0, 60)
        masks = mass_group_masks(masses, n_groups=3)
        means = [float(jnp.sum(jnp.where(masks[g], masses, 0.0)) / jnp.sum(masks[g]))
                 for g in range(3)]
        assert means[0] < means[1] < means[2]


# --------------------------------------------------------------------------
# per_group_virial_ratio
# --------------------------------------------------------------------------
class TestPerGroupVirialRatio:
    def test_single_group_reproduces_global_virial(self):
        """n_groups=1: Clausius W = V, so Q == compute_virial_ratio (exact anchor)."""
        from progenax.dynamics.virial import per_group_virial_ratio, compute_virial_ratio

        pos, vel, m = _equal_mass_plummer(jax.random.PRNGKey(1), Q=0.37)
        mask = jnp.ones((1, m.shape[0]), dtype=bool)
        Q_group = per_group_virial_ratio(pos, vel, m, G=G, group_masks=mask)[0]
        Q_global = compute_virial_ratio(pos, vel, m, G=G)
        np.testing.assert_allclose(float(Q_group), float(Q_global), rtol=1e-6)

    def test_equal_mass_equilibrium_all_groups_near_half(self):
        """Equal-mass virial cluster: random mass-rank groups are statistically
        identical sub-samples, so each Q_j ~ 0.5."""
        from progenax.dynamics.virial import per_group_virial_ratio, mass_group_masks

        Qs = []
        for s in range(6):
            pos, vel, m = _equal_mass_plummer(jax.random.PRNGKey(s), Q=0.5)
            masks = mass_group_masks(m, n_groups=3)
            Qs.append(np.asarray(per_group_virial_ratio(pos, vel, m, G=G, group_masks=masks)))
        Qmean = np.mean(Qs, axis=0)
        np.testing.assert_allclose(Qmean, 0.5, atol=0.12)

    def test_differentiable(self):
        from progenax.dynamics.virial import per_group_virial_ratio, mass_group_masks

        pos, vel, m = _equal_mass_plummer(jax.random.PRNGKey(2))
        masks = mass_group_masks(m, n_groups=2)

        def loss(scale):
            return jnp.sum(per_group_virial_ratio(pos, vel * scale, m, G=G, group_masks=masks))

        g = jax.grad(loss)(1.0)
        assert jnp.isfinite(g)

    def test_jit(self):
        from progenax.dynamics.virial import per_group_virial_ratio, mass_group_masks

        pos, vel, m = _equal_mass_plummer(jax.random.PRNGKey(3))
        masks = mass_group_masks(m, n_groups=3)
        f = jax.jit(lambda p, v: per_group_virial_ratio(p, v, m, G=G, group_masks=masks))
        assert jnp.all(jnp.isfinite(f(pos, vel)))


# --------------------------------------------------------------------------
# _accelerations (blocked row-scan)
# --------------------------------------------------------------------------
class TestBlockedAccelerations:
    def test_matches_dense_oracle_with_padding(self):
        """N=37, block=16 vs an explicit numpy double loop."""
        from progenax.dynamics.virial import _accelerations

        key = jax.random.PRNGKey(11)
        pos = jax.random.normal(key, (37, 3))
        m = jnp.abs(jax.random.normal(jax.random.PRNGKey(12), (37,))) + 0.1
        a = np.asarray(_accelerations(pos, m, G=1.7, block_size=16))
        p = np.asarray(pos)
        mm = np.asarray(m)
        a_ref = np.zeros_like(p)
        for i in range(37):
            for k in range(37):
                if i == k:
                    continue
                d = p[i] - p[k]
                a_ref[i] -= 1.7 * mm[k] * d / np.sum(d**2) ** 1.5
        np.testing.assert_allclose(a, a_ref, rtol=1e-12, atol=1e-13)

    def test_clausius_identity_survives_blocking(self):
        """sum_i m_i r_i . a_i == V — ties Task 2 and Task 3 together and is the
        physics contract per_group_virial_ratio depends on (existing
        test_single_group_reproduces_global_virial re-checks at N=300)."""
        from progenax.dynamics.virial import _accelerations, compute_potential_energy

        key = jax.random.PRNGKey(13)
        pos = jax.random.normal(key, (123, 3))
        m = jnp.ones(123)
        a = _accelerations(pos, m, G=1.0, block_size=32)
        W = float(jnp.sum(m * jnp.sum(pos * a, axis=1)))
        V = float(compute_potential_energy(pos, m, G=1.0, block_size=32))
        np.testing.assert_allclose(W, V, rtol=1e-10)

    def test_grad_finite_at_zero_softening(self):
        from progenax.dynamics.virial import _accelerations

        pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.5, 0.0]])
        m = jnp.ones(3)
        g = jax.grad(lambda p: jnp.sum(_accelerations(p, m, G=1.0,
                                                      block_size=2) ** 2))(pos)
        assert bool(jnp.all(jnp.isfinite(g)))
