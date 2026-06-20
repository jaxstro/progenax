"""Internal binary energy budget diagnostic (Batch 4j).

`binary_energy_budget` separates the two energy scales a primordial-binary cluster
carries: the cluster's COM virial (what `build_binary_cluster` scales to `Q`) and the
internal binary binding energy (a separate reservoir, untouched by `Q` — the McLuster
scale-separation convention). Keyed on `system_id` so it works on the compacted
`ICResult` (`primordial_system_id`) OR the masked `ResolvedBinaries` (ghosts m=0 -> 0).

RED tests pin: vis-viva E_internal = -G m1 m2/2a, singles contribute 0, Q_com recovers
the virial target while Q_resolved (naive, mixes scales) does not, the dE/da gradient,
and masked==compacted.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from jaxtyping import Array, Float

G = STELLAR.G


class _FixedPeriod(eqx.Module):
    P_days: float

    def sample(self, key, n: int) -> Float[Array, "n"]:
        return jnp.full((n,), self.P_days)


def _one_binary(m1, m2, a, e=0.3, shift=0.0):
    from progenax.binaries import KeplerElements

    elem = KeplerElements(
        a=jnp.asarray(a),
        e=jnp.asarray(e),
        i=jnp.asarray(0.5),
        Omega=jnp.asarray(1.0),
        omega=jnp.asarray(2.0),
        M0=jnp.asarray(1.5),
    )
    bs = elem.to_binary_state(m1=jnp.asarray(m1), m2=jnp.asarray(m2), G=G)
    d = jnp.array([shift, 0.0, 0.0])
    pos = jnp.stack([bs.r1 + d, bs.r2 + d])
    vel = jnp.stack([bs.v1, bs.v2])
    mass = jnp.array([float(m1), float(m2)])
    return pos, vel, mass


def _cluster(fbin=0.5, P_days=2.0, n_systems=200, seed=0, compact=True):
    from progenax import PlummerProfile, PlummerVelocityDF, ThermalEccentricity
    from progenax.binaries import IndependentCompanions
    from progenax.builders import Systems, build_binary_cluster
    from progenax.imf import PowerLawIMF
    from progenax.imf.binary import ConstantBinaryFraction, FlatMassRatio

    return build_binary_cluster(
        profile=PlummerProfile(r_h=1.0),
        velocity_df=PlummerVelocityDF(r_h=1.0),
        primary_imf=PowerLawIMF.kroupa(),
        companion_model=IndependentCompanions(
            binary_fraction=ConstantBinaryFraction(fbin),
            q_distribution=FlatMassRatio(q_min=0.3),
            period_distribution=_FixedPeriod(P_days=P_days),
            eccentricity_distribution=ThermalEccentricity(),
        ),
        target=Systems(n_systems),
        key=jax.random.PRNGKey(seed),
        units=STELLAR,
        compact=compact,
    )


class TestEInternal:
    def test_vis_viva_single_binary(self):
        from progenax.binaries import binary_energy_budget

        m1, m2, a = 2.0, 1.0, 0.5
        pos, vel, mass = _one_binary(m1, m2, a)
        sid = jnp.array([0, 0])
        b = binary_energy_budget(pos, vel, mass, sid, G=G)
        expected = -G * m1 * m2 / (2 * a)  # vis-viva: E_rel = -G m1 m2 / 2a
        assert jnp.abs(b.E_internal - expected) / jnp.abs(expected) < 1e-9
        assert b.n_binaries == 1

    def test_singles_contribute_zero(self):
        from progenax.binaries import binary_energy_budget

        pos = jnp.array([[0.0, 0, 0], [10.0, 0, 0]])
        vel = jnp.zeros((2, 3))
        mass = jnp.array([1.0, 2.0])
        sid = jnp.array([0, 1])  # two separate single systems
        b = binary_energy_budget(pos, vel, mass, sid, G=G)
        assert jnp.abs(b.E_internal) < 1e-12
        assert b.n_binaries == 0

    def test_sums_over_binaries(self):
        from progenax.binaries import binary_energy_budget

        p1, v1, m1 = _one_binary(2.0, 1.0, 0.5)
        p2, v2, m2 = _one_binary(3.0, 2.0, 1.0, shift=100.0)  # far apart
        pos = jnp.concatenate([p1, p2])
        vel = jnp.concatenate([v1, v2])
        mass = jnp.concatenate([m1, m2])
        sid = jnp.array([0, 0, 1, 1])
        b = binary_energy_budget(pos, vel, mass, sid, G=G)
        exp = -G * 2 * 1 / (2 * 0.5) + -G * 3 * 2 / (2 * 1.0)
        assert jnp.abs(b.E_internal - exp) / jnp.abs(exp) < 1e-6
        assert b.n_binaries == 2

    def test_grad_dE_internal_da(self):
        """E_internal = -G m1 m2 / 2a (exact, vis-viva) -> dE/da = +G m1 m2 / 2a^2."""
        from progenax.binaries import KeplerElements, binary_energy_budget

        m1, m2 = 2.0, 1.0
        sid = jnp.array([0, 0])

        def E_int(a):
            elem = KeplerElements(
                a=a,
                e=jnp.asarray(0.3),
                i=jnp.asarray(0.5),
                Omega=jnp.asarray(1.0),
                omega=jnp.asarray(2.0),
                M0=jnp.asarray(1.5),
            )
            bs = elem.to_binary_state(m1=jnp.asarray(m1), m2=jnp.asarray(m2), G=G)
            pos = jnp.stack([bs.r1, bs.r2])
            vel = jnp.stack([bs.v1, bs.v2])
            mass = jnp.array([m1, m2])
            return binary_energy_budget(pos, vel, mass, sid, G=G).E_internal

        ad = jax.grad(E_int)(0.5)
        analytic = G * m1 * m2 / (2 * 0.5**2)
        assert jnp.abs(ad - analytic) / jnp.abs(analytic) < 1e-5


class TestQConvention:
    def test_Q_com_recovers_virial_target(self):
        """The cluster is virialized to Q=0.5 on the system COMs; binary_energy_budget
        reconstructs those COMs from the resolved stars and recovers Q_com ~ 0.5."""
        from progenax.binaries import binary_energy_budget

        ic = _cluster(fbin=0.5, P_days=2.0, seed=1)
        b = binary_energy_budget(
            ic.positions, ic.velocities, ic.masses, ic.primordial_system_id, G=G
        )
        assert jnp.abs(b.Q_com - 0.5) < 5e-3, f"Q_com={b.Q_com}"

    def test_internal_reservoir_dwarfs_cluster_potential(self):
        """The internal binary binding energy is a *separate, much larger* reservoir than
        the cluster's own potential — which is exactly why virializing on the COMs (not the
        resolved stars) is correct. (A naive Q_resolved is contaminated by, and for hard
        binaries pulled toward 0.5 by, the internally-virialized binaries — so it is NOT the
        cluster's virial ratio: Q_resolved != Q_com.)"""
        from progenax.binaries import binary_energy_budget

        ic = _cluster(fbin=1.0, P_days=1.0, seed=2)  # all hard (short-period) binaries
        b = binary_energy_budget(
            ic.positions, ic.velocities, ic.masses, ic.primordial_system_id, G=G
        )
        assert b.E_internal < 0.0  # bound binaries
        assert jnp.abs(b.Q_com - 0.5) < 5e-3  # cluster virial intact on the COMs
        assert jnp.abs(b.E_internal) > 100.0 * jnp.abs(
            b.W_com
        )  # reservoir >> cluster potential
        assert (
            jnp.abs(b.Q_resolved - b.Q_com) > 1e-3
        )  # resolved ratio is not the cluster's

    def test_masked_matches_compacted(self):
        """system_id keying gives the same E_internal on the masked ResolvedBinaries
        (ghost secondaries m=0 contribute 0) as on the compacted ICResult."""
        from progenax.binaries import binary_energy_budget

        ic = _cluster(fbin=0.5, seed=3, compact=True)
        rb = _cluster(fbin=0.5, seed=3, compact=False)
        b_c = binary_energy_budget(
            ic.positions, ic.velocities, ic.masses, ic.primordial_system_id, G=G
        )
        b_m = binary_energy_budget(
            rb.positions, rb.velocities, rb.masses, rb.primordial_system_id, G=G
        )
        assert b_m.n_binaries == b_c.n_binaries
        assert jnp.abs(b_m.E_internal - b_c.E_internal) <= 1e-6 * jnp.abs(
            b_c.E_internal
        )
