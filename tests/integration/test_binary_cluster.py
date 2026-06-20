"""End-to-end binary-cluster IC assembly (Batch 4k SoTA composition).

build_binary_cluster composes primary_imf x companion_model x profile x velocity_df
x target -> system COMs (build_spatial_ic) + resolve_binary_components, returning a
compacted ICResult (real particles + primordial provenance) or the masked
ResolvedBinaries. The companion model owns f_b + (q, P, e); the target chooses the
population-size budget (Systems / Stars / TotalMass).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxstro.units import STELLAR
from jaxtyping import Array, Float

from progenax import (
    LogUniformPeriod,
    PlummerProfile,
    PlummerVelocityDF,
    ThermalEccentricity,
)
from progenax.binaries import IndependentCompanions, MoeCompanions
from progenax.builders import Stars, Systems, TotalMass, build_binary_cluster
from progenax.imf import PowerLawIMF
from progenax.imf.binary import ConstantBinaryFraction, FlatMassRatio

DAY_IN_TU = 86400.0 / STELLAR.time_scale_cgs  # 1 day in Myr (STELLAR)


class FixedPeriod(eqx.Module):
    """Degenerate period distribution returning a constant P [days] (test oracle)."""

    P_days: float

    def sample(self, key, n: int) -> Float[Array, "n"]:
        return jnp.full((n,), self.P_days)


def _independent(fbin=0.5, period=None, qmin=0.2):
    return IndependentCompanions(
        binary_fraction=ConstantBinaryFraction(fbin),
        q_distribution=FlatMassRatio(q_min=qmin),
        period_distribution=period
        if period is not None
        else LogUniformPeriod(log_P_min=2.0, log_P_max=4.0),
        eccentricity_distribution=ThermalEccentricity(),
    )


def _cluster(fbin=0.5, period=None, n_systems=300, seed=0, target=None, **kw):
    return build_binary_cluster(
        profile=PlummerProfile(r_h=1.0),
        velocity_df=PlummerVelocityDF(r_h=1.0),
        primary_imf=PowerLawIMF.kroupa(),
        companion_model=_independent(fbin=fbin, period=period),
        target=target if target is not None else Systems(n_systems),
        key=jax.random.PRNGKey(seed),
        units=STELLAR,
        **kw,
    )


class TestBuildBinaryCluster:
    def test_count_and_provenance(self):
        ic = _cluster()
        n_sec = int(jnp.sum(ic.is_primordial_secondary))
        n_part = ic.masses.shape[0]
        n_systems = int(jnp.max(ic.primordial_system_id)) + 1
        # particles = singles + 2*binaries = n_systems + n_secondaries
        assert n_part == n_systems + n_sec
        assert ic.primordial_system_id is not None
        assert ic.positions.shape == (n_part, 3)
        assert jnp.all(ic.masses > 0.0)  # no ghosts in the compacted result

    def test_no_softening_field(self):
        ic = _cluster()
        assert not hasattr(
            ic, "softening"
        )  # softening is integration-time, not IC state

    def test_com_preserved(self):
        """Resolving binaries preserves COMs, so the mass-weighted cluster COM stays ~0."""
        ic = _cluster()
        com = jnp.sum(ic.positions * ic.masses[:, None], axis=0) / jnp.sum(ic.masses)
        vcom = jnp.sum(ic.velocities * ic.masses[:, None], axis=0) / jnp.sum(ic.masses)
        assert jnp.allclose(com, 0.0, atol=1e-10), f"cluster COM {com}"
        assert jnp.allclose(vcom, 0.0, atol=1e-10), f"cluster Vcom {vcom}"

    def test_units_kepler_third_law_roundtrip(self):
        """A fixed-period all-binary cluster: recover a from the resolved pair, and
        compute_period(a) must return the input period — verifies the day->time-unit
        conversion is correct (the Batch-4f units gotcha)."""
        from progenax.binaries import KeplerElements, compute_period

        ic = _cluster(fbin=1.0, period=FixedPeriod(P_days=100.0), n_systems=60, seed=3)
        # system 0 is a binary -> its two real particles are the first two rows
        i0 = jnp.where(ic.primordial_system_id == 0)[0]
        assert i0.shape[0] == 2
        a_idx, b_idx = int(i0[0]), int(i0[1])
        r_rel = ic.positions[b_idx] - ic.positions[a_idx]
        v_rel = ic.velocities[b_idx] - ic.velocities[a_idx]
        M_total = float(ic.masses[a_idx] + ic.masses[b_idx])
        elem = KeplerElements.from_state(r_rel, v_rel, M_total, G=STELLAR.G)
        P_tu = compute_period(elem.a, M_total, G=STELLAR.G)
        P_days = float(P_tu) / DAY_IN_TU
        assert jnp.abs(P_days - 100.0) < 1e-3, f"recovered P={P_days} d, expected 100 d"

    def test_compact_false_returns_masked(self):
        from progenax.binaries import ResolvedBinaries

        rb = _cluster(n_systems=120, compact=False)
        assert isinstance(rb, ResolvedBinaries)
        assert rb.positions.shape == (240, 3)  # 2N masked slots
        assert rb.is_real.shape == (240,)

    def test_grad_through_r_h(self):
        """The compacted COM positions are differentiable wrt the spatial scale r_h."""

        def spread(r_h):
            rb = build_binary_cluster(
                profile=PlummerProfile(r_h=r_h),
                velocity_df=PlummerVelocityDF(r_h=r_h),
                primary_imf=PowerLawIMF.kroupa(),
                companion_model=_independent(fbin=0.5),
                target=Systems(100),
                key=jax.random.PRNGKey(5),
                units=STELLAR,
                compact=False,  # static shape -> grad-safe
            )
            return jnp.mean(jnp.linalg.norm(rb.positions, axis=1))

        g = jax.grad(spread)(1.0)
        assert jnp.isfinite(g) and g > 0.0, f"d<|r|>/d(r_h) = {g}"


class TestBudgetTargets:
    def test_stars_counts_companions(self):
        """Stars(n) yields n or n+1 resolved stars (companions counted; whole systems)."""
        n = 400
        ic = _cluster(fbin=0.6, target=Stars(n), seed=1)
        assert n <= ic.masses.shape[0] <= n + 1

    def test_systems_does_not_count_companions(self):
        """Systems(n) yields n systems -> n + n_binary stars (companions on top)."""
        ic = _cluster(fbin=0.5, target=Systems(300), seed=1)
        n_sec = int(jnp.sum(ic.is_primordial_secondary))
        assert ic.masses.shape[0] == 300 + n_sec

    def test_totalmass_reaches_budget(self):
        """TotalMass(M) yields total stellar mass >= M, minimal whole-system overshoot."""
        M = 800.0
        ic = _cluster(fbin=0.5, target=TotalMass(M), seed=2)
        total = float(jnp.sum(ic.masses))
        assert total >= M
        assert total <= M + float(jnp.max(ic.masses)) * 2  # <= one (binary) system over

    def test_stars_compact_false_raises(self):
        """Stars/TotalMass have dynamic counts -> the masked differentiable path is rejected."""
        with pytest.raises(ValueError, match="Systems"):
            _cluster(target=Stars(100), compact=False)


class TestMoeCompanionsCluster:
    def test_invariants_with_moe_joint(self):
        """A cluster built from the faithful MoeCompanions satisfies the same IC invariants."""
        ic = build_binary_cluster(
            profile=PlummerProfile(r_h=1.0),
            velocity_df=PlummerVelocityDF(r_h=1.0),
            primary_imf=PowerLawIMF.kroupa(),
            companion_model=MoeCompanions(),
            target=Systems(400),
            key=jax.random.PRNGKey(7),
            units=STELLAR,
        )
        assert jnp.all(ic.masses > 0.0)  # no ghosts
        com = jnp.sum(ic.positions * ic.masses[:, None], axis=0) / jnp.sum(ic.masses)
        assert jnp.allclose(com, 0.0, atol=1e-10), f"cluster COM {com}"
        # secondaries are real: m2 <= m1 within each primordial binary -> q in (0,1]
        n_sec = int(jnp.sum(ic.is_primordial_secondary))
        assert n_sec > 0
