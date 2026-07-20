r"""The ⟨e^s⟩ convention, and the fact that it cancels downstream.

``s = ln(rho/rho_0)``. Two docstrings in this subsystem used to assert **opposite**
invariants for the realized field, and the false one cost a debugging session:

    pipeline.py   "log-density ln(rho/rho_0), <e^s>=1"          <- WRONG, now corrected
    copula.py     "<e^s> = mean_density (>=1) ... not a forced 1"  <- correct

The correct statement is the second. Burkhart & Mocz (2019), ApJ 879, 129 §2 note that
their ``rho_0`` is a **pre-collapse reference density**, and that the powerlaw tail raises
the mean above it: "the PDF has been normalized to a pre-collapse reference density rho_0,
and predicts that the average density in a collapsing region grows with time as alpha
flattens." They give the mass-conserving alternative as a density shift ``s_new = s - s_s``
(their Eq. 3) for "numerical box simulations [that] use the condition of mass conservation".

gravoturb deliberately keeps the BM19 pre-collapse convention and does NOT apply the shift,
because it is immaterial here: ``gas.normalized_cloud_density`` rescales to ``\int rho dV = M_cl``
exactly and is explicitly invariant to any additive constant in ``s``. A multiplicative
factor on ``e^s`` IS an additive constant in ``s``, so the offset divides straight out.

These tests pin BOTH halves, so the two facts can never again drift apart:

  1. the offset is real, equals ``mean_density``, and matches BM19's analytic value
  2. it has exactly zero effect on the normalized cloud density

Measured offsets (deterministic across seeds -- the quantile construction fixes the
multiset of values, so every realization has the identical mean):

    mach     0.5     1.0     2.0     4.0     8.0    16.0
    <e^s>  1.7245  1.4436  1.2035  1.0780  1.0305  1.0131
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.experimental

SHAPE = (48, 48, 48)
B = 0.5
ALPHA = 1.8

# Finite-N quantile discretisation of a heavy-tailed PDF; not a modelling error.
GRID_RTOL = 0.03


def _field(mach, seed=0, shape=SHAPE, alpha=ALPHA):
    from gravoturb.realization.pipeline import build_turbulent_field

    return build_turbulent_field(
        mach=mach, b=B, alpha=alpha, beta=3.5, shape=shape, key=jax.random.PRNGKey(seed)
    )


class TestTheOffsetIsRealAndAnalytic:
    """<e^s> is mean_density, not 1 -- and the theory predicts the exact value."""

    @pytest.mark.parametrize(
        "mach, expected", [(0.5, 1.7245), (1.0, 1.4436), (2.0, 1.2035), (8.0, 1.0305)]
    )
    def test_realized_mean_matches_theory(self, mach, expected):
        from gravoturb.theory.density_cdf import mean_density

        got = float(jnp.mean(jnp.exp(_field(mach).s)))
        assert got == pytest.approx(expected, rel=GRID_RTOL), f"mach={mach}: {got:.4f}"
        assert got == pytest.approx(
            float(mean_density(mach, B, ALPHA)), rel=GRID_RTOL
        ), "realized mean must track the analytic mean_density"

    def test_offset_decreases_toward_unity_with_mach(self):
        """The tail recedes as s_t = (alpha-1/2) sigma_s^2 grows, so the offset shrinks."""
        from gravoturb.theory.density_cdf import mean_density

        vals = [float(mean_density(m, B, ALPHA)) for m in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]
        assert vals == sorted(vals, reverse=True), vals
        assert vals[0] > 1.7 and vals[-1] < 1.02

    def test_offset_is_deterministic_across_seeds(self):
        """The quantile construction fixes the value multiset, so the mean is seed-free."""
        vals = [float(jnp.mean(jnp.exp(_field(8.0, seed=s).s))) for s in range(4)]
        assert max(vals) - min(vals) < 1e-12, vals

    def test_offset_vanishes_when_the_tail_is_negligible(self):
        """Steep alpha pushes s_t far out; with no tail the lognormal is mass-conserving."""
        from gravoturb.theory.density_cdf import mean_density

        assert float(mean_density(8.0, B, 6.0)) == pytest.approx(1.0, abs=0.02)


class TestTheOffsetCancelsDownstream:
    """Why the convention is immaterial: normalize_cloud divides it out exactly."""

    def test_normalized_cloud_integrates_to_M_cl(self):
        from gravoturb.realization.gas import normalized_cloud_density

        f = _field(8.0)
        M_cl = 1234.5
        rho, dV = normalized_cloud_density(f.s, box_size=4.0, M_cl=M_cl)
        assert float(jnp.sum(rho) * dV) == pytest.approx(M_cl, rel=1e-10)

    @pytest.mark.parametrize("shift", [-2.0, 0.0, 3.7])
    def test_normalized_cloud_is_invariant_to_an_additive_shift_in_s(self, shift):
        """THE point: applying (or not applying) BM19's s_s cannot change rho_cl.

        A multiplicative factor on e^s is an additive constant in s, so this directly
        establishes that the convention choice has no downstream consequence.
        """
        from gravoturb.realization.gas import normalized_cloud_density

        f = _field(8.0)
        base, _ = normalized_cloud_density(f.s, box_size=4.0, M_cl=1000.0)
        shifted, _ = normalized_cloud_density(f.s + shift, box_size=4.0, M_cl=1000.0)
        assert jnp.allclose(base, shifted, rtol=1e-12)

    def test_applying_the_bm19_shift_leaves_rho_cl_unchanged(self):
        """Concretely: subtract ln(mean_density) -- rho_cl must not move."""
        from gravoturb.realization.gas import normalized_cloud_density
        from gravoturb.theory.density_cdf import mean_density

        f = _field(8.0)
        s_s = jnp.log(mean_density(8.0, B, ALPHA))
        base, _ = normalized_cloud_density(f.s, box_size=4.0, M_cl=1000.0)
        conserving, _ = normalized_cloud_density(f.s - s_s, box_size=4.0, M_cl=1000.0)
        assert jnp.allclose(base, conserving, rtol=1e-12)
