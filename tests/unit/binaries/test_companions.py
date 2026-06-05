"""Companion/orbit layer for build_binary_cluster (Batch 4k-2).

A `CompanionModel` owns the binary statistics in one place: multiplicity
(f_b → is_binary) AND companion properties (q → m2, P → a, e, orientation), all
keyed on the primary masses. Two implementations:

- `IndependentCompanions(binary_fraction, q_distribution, period_distribution,
  eccentricity_distribution)` — versatile; reproduces today's independent-marginal
  draws (component-level bit-identical given the same key).
- `MoeCompanions(q_min)` — faithful Moe+2017: f_b = MassDependentBinaryFraction,
  (q,P,e) jointly from MoeJointOrbit; the SAME q sets m2 (self-consistent), so the
  P–q interrelation shows up in the secondary masses.

RED tests pin: shapes/singles, m2=q·m1, the entropy-layout equivalence contract,
Moe self-consistency + mass-dependent f_b, protocol conformance, jit, and grads.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR

DAY = 86400.0 / STELLAR.time_scale_cgs
G = STELLAR.G


def _independent(fbin=0.5, qmin=0.2, e_max=0.99):
    from progenax.binaries import IndependentCompanions
    from progenax.imf.binary import ConstantBinaryFraction, FlatMassRatio
    from progenax import LogUniformPeriod, ThermalEccentricity

    return IndependentCompanions(
        binary_fraction=ConstantBinaryFraction(fbin),
        q_distribution=FlatMassRatio(q_min=qmin),
        period_distribution=LogUniformPeriod(log_P_min=2.0, log_P_max=4.0),
        eccentricity_distribution=ThermalEccentricity(e_max=e_max),
    )


class TestCompanionElements:
    def test_namedtuple_fields(self):
        from progenax.binaries import CompanionElements

        el = CompanionElements(
            m2=jnp.zeros(2), a=jnp.ones(2), e=jnp.zeros(2),
            inc=jnp.zeros(2), Omega=jnp.zeros(2), omega=jnp.zeros(2), M_anom=jnp.zeros(2),
        )
        assert el.m2.shape == (2,) and el.a.shape == (2,) and el.M_anom.shape == (2,)


class TestIndependentCompanions:
    def test_shapes_and_singles(self):
        m1 = jnp.array([1.0, 2.0, 0.5, 8.0, 1.0])
        is_binary, el = _independent().sample(jax.random.PRNGKey(0), m1, G=G, day_in_time_units=DAY)
        assert is_binary.shape == (5,) and el.m2.shape == (5,)
        # singles (not binary) -> m2 == 0 exactly
        assert jnp.all(el.m2[~is_binary] == 0.0)
        # binaries -> 0 < m2 <= m1 (q in (0,1])
        bm = is_binary
        assert jnp.all(el.m2[bm] > 0.0) and jnp.all(el.m2[bm] <= m1[bm] + 1e-9)

    def test_m2_is_q_times_m1(self):
        m1 = jnp.full(5000, 3.0)
        is_binary, el = _independent(fbin=1.0, qmin=0.2).sample(
            jax.random.PRNGKey(1), m1, G=G, day_in_time_units=DAY
        )
        q = el.m2 / m1
        assert jnp.all((q >= 0.2 - 1e-6) & (q <= 1.0 + 1e-6))

    def test_component_equivalence_pins_entropy_layout(self):
        """The default companion path reproduces the standalone sub-distribution
        draws given the same key — the equivalence contract for the rewire.
        Entropy layout: split(key, 5) -> [is_binary, q, P, e, orientation]."""
        from progenax.imf.binary import ConstantBinaryFraction
        from progenax import LogUniformPeriod, ThermalEccentricity
        from progenax.binaries import period_to_semimajor_axis

        m1 = jnp.linspace(0.5, 10.0, 64)
        key = jax.random.PRNGKey(7)
        model = _independent(fbin=0.5, qmin=0.2)
        is_binary, el = model.sample(key, m1, G=G, day_in_time_units=DAY)

        kb, kq, kP, ke, ko = jax.random.split(key, 5)
        # multiplicity
        f_bin = ConstantBinaryFraction(0.5)(m1)
        assert jnp.array_equal(is_binary, jax.random.uniform(kb, (64,)) < f_bin)
        # eccentricity
        assert jnp.allclose(el.e, ThermalEccentricity(e_max=0.99).sample(ke, 64))
        # period -> a (using the SAME m2 the model produced)
        P_exp = LogUniformPeriod(log_P_min=2.0, log_P_max=4.0).sample(kP, 64)
        a_exp = period_to_semimajor_axis(P_exp * DAY, m1 + el.m2, G)
        assert jnp.allclose(el.a, a_exp)

    def test_mass_dependent_q_dispatch(self):
        """A mass-dependent q (MoeDiStefano2017) is dispatched via sample_given_primary."""
        from progenax.binaries import IndependentCompanions
        from progenax.imf.binary import ConstantBinaryFraction, MoeDiStefano2017
        from progenax import LogUniformPeriod, ThermalEccentricity

        model = IndependentCompanions(
            binary_fraction=ConstantBinaryFraction(1.0),
            q_distribution=MoeDiStefano2017(q_min=0.1),
            period_distribution=LogUniformPeriod(2.0, 4.0),
            eccentricity_distribution=ThermalEccentricity(),
        )
        # M-dwarfs favour equal q (gamma=+0.4); OB favour unequal (gamma=-0.5)
        m_low = jnp.full(40000, 0.5)
        m_high = jnp.full(40000, 10.0)
        _, el_lo = model.sample(jax.random.PRNGKey(2), m_low, G=G, day_in_time_units=DAY)
        _, el_hi = model.sample(jax.random.PRNGKey(2), m_high, G=G, day_in_time_units=DAY)
        assert jnp.mean(el_lo.m2 / m_low) > jnp.mean(el_hi.m2 / m_high)

    def test_grad_fd_accurate_eccentricity(self):
        m1 = jnp.full(4000, 2.0)
        key = jax.random.PRNGKey(3)

        def loss(emax):
            _, el = _independent(e_max=emax).sample(key, m1, G=G, day_in_time_units=DAY)
            return jnp.mean(el.e)

        ad = jax.grad(loss)(0.9)
        h = 1e-4
        fd = (loss(0.9 + h) - loss(0.9 - h)) / (2 * h)
        assert jnp.abs(ad - fd) / (jnp.abs(ad) + 1e-12) < 1e-3

    def test_jit(self):
        m1 = jnp.full(100, 1.5)
        f = jax.jit(lambda k, m: _independent().sample(k, m, G=G, day_in_time_units=DAY))
        is_binary, el = f(jax.random.PRNGKey(4), m1)
        assert jnp.all(jnp.isfinite(el.a)) and jnp.all(jnp.isfinite(el.e))


class TestMoeCompanions:
    def test_shapes_singles_ranges(self):
        from progenax.binaries import MoeCompanions

        m1 = jnp.full(20000, 5.0)
        is_binary, el = MoeCompanions().sample(jax.random.PRNGKey(0), m1, G=G, day_in_time_units=DAY)
        assert is_binary.shape == (20000,) and el.m2.shape == (20000,)
        assert jnp.all(el.m2[~is_binary] == 0.0)
        assert jnp.all((el.e >= 0.0) & (el.e < 1.0))
        q = el.m2[is_binary] / m1[is_binary]
        assert jnp.all((q >= 0.1 - 1e-3) & (q <= 1.0 + 1e-3))

    def test_mass_dependent_binary_fraction(self):
        """Moe's own f_b: massive primaries are far more often binary than M-dwarfs."""
        from progenax.binaries import MoeCompanions

        mc = MoeCompanions()
        ib_lo, _ = mc.sample(jax.random.PRNGKey(1), jnp.full(50000, 0.3), G=G, day_in_time_units=DAY)
        ib_hi, _ = mc.sample(jax.random.PRNGKey(1), jnp.full(50000, 20.0), G=G, day_in_time_units=DAY)
        assert jnp.mean(ib_hi) > jnp.mean(ib_lo)

    def test_p_q_interrelation_in_secondary_masses(self):
        """Self-consistency: the joint q sets m2, so short-period (small-a) binaries
        of massive primaries carry larger q = m2/m1 than long-period (large-a) ones."""
        from progenax.binaries import MoeCompanions

        m1 = jnp.full(400000, 12.0)  # early-B: gamma_largeq steepens strongly with logP
        is_binary, el = MoeCompanions().sample(jax.random.PRNGKey(3), m1, G=G, day_in_time_units=DAY)
        bm = is_binary
        a = el.a[bm]
        q = el.m2[bm] / m1[bm]
        a_med = jnp.median(a)
        q_short = jnp.mean(q[a < a_med])
        q_long = jnp.mean(q[a > a_med])
        assert q_short > q_long, f"q(short P)={q_short:.3f} should exceed q(long P)={q_long:.3f}"

    def test_jit(self):
        from progenax.binaries import MoeCompanions

        m1 = jnp.full(500, 3.0)
        f = jax.jit(lambda k, m: MoeCompanions().sample(k, m, G=G, day_in_time_units=DAY))
        is_binary, el = f(jax.random.PRNGKey(5), m1)
        assert jnp.all(jnp.isfinite(el.a)) and jnp.all(jnp.isfinite(el.m2))


class TestProtocolConformance:
    def test_both_are_companion_models(self):
        from progenax.binaries import MoeCompanions
        from progenax.protocols import CompanionModel

        assert isinstance(_independent(), CompanionModel)
        assert isinstance(MoeCompanions(), CompanionModel)
