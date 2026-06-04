"""Dynamic binary diagnostics (Batch 4g): bound-pair detection from current state.

A primordial pairing goes stale under evolution (ionization/formation/exchange),
so the *current* binary population must be measured from (r, v, m), not read off
the IC labels. relative_energy is the differentiable two-body primitive;
find_bound_pairs is the mutual-nearest-neighbour + bound diagnostic;
primordial_survival compares the current pairing to the t=0 provenance.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR

from progenax.binaries import KeplerElements

G = STELLAR.G


def _circular_binary(a=1e-3, m1=2.0, m2=1.0):
    """Two components of a circular binary at COM origin (separation ~ a)."""
    bs = KeplerElements(a=a, e=0.0, i=0.3, Omega=0.1, omega=0.2, M0=0.7).to_binary_state(
        m1=m1, m2=m2, G=G
    )
    return bs.r1, bs.v1, bs.r2, bs.v2, m1, m2


class TestRelativeEnergy:
    def test_circular_binary_energy(self):
        """E_rel of a bound two-body orbit = -G m1 m2 / (2a)."""
        from progenax.binaries import relative_energy
        r1, v1, r2, v2, m1, m2 = _circular_binary(a=1e-3)
        E = relative_energy(r1, r2, v1, v2, m1, m2, G=G)
        expected = -G * m1 * m2 / (2.0 * 1e-3)
        assert jnp.abs(E - expected) / jnp.abs(expected) < 1e-6, f"{E} vs {expected}"
        assert E < 0  # bound

    def test_unbound_pair_positive(self):
        from progenax.binaries import relative_energy
        # two distant, fast-separating particles
        r1 = jnp.array([0.0, 0.0, 0.0]); r2 = jnp.array([5.0, 0.0, 0.0])
        v1 = jnp.array([-1.0, 0.0, 0.0]); v2 = jnp.array([1.0, 0.0, 0.0])
        E = relative_energy(r1, r2, v1, v2, 1.0, 1.0, G=G)
        assert E > 0  # unbound

    def test_differentiable(self):
        from progenax.binaries import relative_energy
        r1, v1, r2, v2, m1, m2 = _circular_binary(a=1e-3)
        g = jax.grad(lambda a_scale: relative_energy(r1, r2 * a_scale, v1, v2, m1, m2, G=G))(1.0)
        assert jnp.isfinite(g)


def _hard_binary_cluster(n_systems=40, P_days=1.0, seed=2):
    """All-binary cluster of very hard (short-period) binaries for round-trip tests."""
    import equinox as eqx
    from jaxtyping import Array, Float
    from progenax import PlummerProfile, PlummerVelocityDF, ThermalEccentricity
    from progenax.builders import build_binary_cluster
    from progenax.imf import PowerLawIMF
    from progenax.imf.binary import BinaryIMF, ConstantBinaryFraction, FlatMassRatio

    class FixedPeriod(eqx.Module):
        P_days: float

        def sample(self, key, n: int) -> Float[Array, "n"]:
            return jnp.full((n,), self.P_days)

    imf = BinaryIMF(
        primary_imf=PowerLawIMF.kroupa(),
        q_distribution=FlatMassRatio(q_min=0.3),
        binary_fraction=ConstantBinaryFraction(1.0),
    )
    return build_binary_cluster(
        profile=PlummerProfile(r_h=1.0),
        velocity_df=PlummerVelocityDF(r_h=1.0),
        binary_imf=imf,
        period_dist=FixedPeriod(P_days=P_days),
        ecc_dist=ThermalEccentricity(),
        n_systems=n_systems,
        key=jax.random.PRNGKey(seed),
        units=STELLAR,
    )


class TestFindBoundPairs:
    def test_recovers_primordial_hard_binaries(self):
        from progenax.binaries import find_bound_pairs
        ic = _hard_binary_cluster(n_systems=40, P_days=1.0)
        pairs, E = find_bound_pairs(ic.positions, ic.velocities, ic.masses, G=G)
        assert pairs.shape[0] == 40, f"found {pairs.shape[0]} pairs, expected 40"
        assert jnp.all(E < 0)
        # every recovered pair shares a primordial_system_id
        psid = ic.primordial_system_id
        for k in range(pairs.shape[0]):
            i, j = int(pairs[k, 0]), int(pairs[k, 1])
            assert int(psid[i]) == int(psid[j])

    def test_pairs_are_unique_and_ordered(self):
        from progenax.binaries import find_bound_pairs
        ic = _hard_binary_cluster(n_systems=20, P_days=1.0)
        pairs, _ = find_bound_pairs(ic.positions, ic.velocities, ic.masses, G=G)
        assert jnp.all(pairs[:, 0] < pairs[:, 1])  # i < j
        flat = jnp.concatenate([pairs[:, 0], pairs[:, 1]])
        assert len(jnp.unique(flat)) == flat.shape[0]  # no particle in two pairs


class TestPrimordialSurvival:
    def test_all_survive_at_t0(self):
        from progenax.binaries import find_bound_pairs, primordial_survival
        ic = _hard_binary_cluster(n_systems=30, P_days=1.0)
        pairs, _ = find_bound_pairs(ic.positions, ic.velocities, ic.masses, G=G)
        surv = primordial_survival(pairs, ic.primordial_system_id)
        assert surv["survived"] == 30
        assert surv["disrupted"] == 0
        assert surv["newly_formed"] == 0

    def test_disruption_counted(self):
        """Fling one binary's secondary far away -> that primordial binary is disrupted."""
        from progenax.binaries import find_bound_pairs, primordial_survival
        ic = _hard_binary_cluster(n_systems=30, P_days=1.0)
        # find a secondary of system 0 and move it far with high speed
        sec_idx = int(jnp.where((ic.primordial_system_id == 0) & ic.is_primordial_secondary)[0][0])
        pos = ic.positions.at[sec_idx].set(jnp.array([50.0, 50.0, 50.0]))
        vel = ic.velocities.at[sec_idx].add(jnp.array([100.0, 0.0, 0.0]))
        pairs, _ = find_bound_pairs(pos, vel, ic.masses, G=G)
        surv = primordial_survival(pairs, ic.primordial_system_id)
        assert surv["disrupted"] >= 1
        assert surv["survived"] <= 29


def _isolated_binaries(K=3, a=1e-3, sep=100.0):
    """K hard binaries placed far apart with distinct COM velocities (COMs unbound)."""
    pos, vel, mass = [], [], []
    for k in range(K):
        bs = KeplerElements(a=a, e=0.0, i=0.2, Omega=0.1, omega=0.3, M0=0.5).to_binary_state(
            m1=1.0, m2=1.0, G=G
        )
        center = jnp.array([k * sep, 0.0, 0.0])
        com_vel = jnp.array([0.0, float(k) * 10.0, 0.0])  # distinct -> COM pairs unbound
        pos += [bs.r1 + center, bs.r2 + center]
        vel += [bs.v1 + com_vel, bs.v2 + com_vel]
        mass += [1.0, 1.0]
    return jnp.stack(pos), jnp.stack(vel), jnp.array(mass)


class TestFindBoundMultiples:
    def test_isolated_binaries_multiplicity_2(self):
        """Well-separated hard binaries (COM-unbound) -> every particle multiplicity 2."""
        from progenax.binaries import find_bound_multiples
        positions, velocities, masses = _isolated_binaries(K=4)
        system_id, multiplicity = find_bound_multiples(positions, velocities, masses, G=G)
        assert jnp.all(multiplicity == 2), f"multiplicities {jnp.unique(multiplicity)}"
        # each detected system has exactly 2 members
        for sid in jnp.unique(system_id):
            assert int(jnp.sum(system_id == sid)) == 2

    def test_cluster_groups_primordial_binaries(self):
        """In a cluster, each primordial binary's two members land in the same detected
        system (possibly inside a larger hierarchy) -> multiplicity >= 2."""
        from progenax.binaries import find_bound_multiples
        ic = _hard_binary_cluster(n_systems=30, P_days=1.0)
        system_id, multiplicity = find_bound_multiples(
            ic.positions, ic.velocities, ic.masses, G=G
        )
        psid = ic.primordial_system_id
        for sid in jnp.unique(psid):
            members = jnp.where(psid == sid)[0]
            assert int(system_id[members[0]]) == int(system_id[members[1]])
            assert int(multiplicity[members[0]]) >= 2

    def test_hierarchical_triple(self):
        """A tight inner binary + a third star bound to the binary COM -> multiplicity 3."""
        from progenax.binaries import find_bound_multiples
        inner = KeplerElements(a=1e-3, e=0.0, i=0.0, Omega=0.0, omega=0.0, M0=0.0)
        bs = inner.to_binary_state(m1=1.0, m2=1.0, G=G)  # COM at origin
        third_pos = jnp.array([0.1, 0.0, 0.0])
        third_vel = jnp.array([0.0, 0.2, 0.0])  # bound to the m=2 binary COM at origin
        positions = jnp.stack([bs.r1, bs.r2, third_pos])
        velocities = jnp.stack([bs.v1, bs.v2, third_vel])
        masses = jnp.array([1.0, 1.0, 1.0])
        system_id, multiplicity = find_bound_multiples(positions, velocities, masses, G=G)
        assert jnp.all(multiplicity == 3), f"multiplicities {multiplicity}"
        assert len(jnp.unique(system_id)) == 1  # all one hierarchical system

    def test_singles_multiplicity_1(self):
        """Fast-moving well-separated stars (unbound) -> all singles (multiplicity 1).

        (Note: STATIONARY separated stars are bound, E_rel=-G m1 m2/r < 0, so they
        are NOT singles — unboundness requires enough relative kinetic energy.)
        """
        from progenax.binaries import find_bound_multiples
        positions = jnp.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        velocities = jnp.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
        masses = jnp.ones(3)
        _, multiplicity = find_bound_multiples(positions, velocities, masses, G=G)
        assert jnp.all(multiplicity == 1)

    def test_jit_safe(self):
        """find_bound_multiples is fixed-shape -> jittable."""
        from progenax.binaries import find_bound_multiples
        ic = _hard_binary_cluster(n_systems=12, P_days=1.0)
        jitted = jax.jit(lambda p, v, m: find_bound_multiples(p, v, m, G=G))
        sid, mult = jitted(ic.positions, ic.velocities, ic.masses)
        assert mult.shape == (24,)
