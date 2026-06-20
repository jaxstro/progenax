"""Population-size budget targets for build_binary_cluster (Batch 4k-1).

`Systems(n)` / `Stars(n)` / `TotalMass(M)` choose what the cluster builder holds
fixed. `Systems` is the paper/observational convention (companions NOT counted);
`Stars`/`TotalMass` count companions (real gravitating bodies in a dynamical IC).
RED tests pin the prefix-cut mask semantics + the draw-sufficiency predicate.

The cut keeps WHOLE systems in draw order (never splits a binary), so a star/mass
budget overshoots by at most one system — physically honest (McLuster-style mass
filling), not an exact-N truncation that would mutilate the boundary binary.
"""

import jax.numpy as jnp


def _imports():
    from progenax.builders import (
        Stars,
        Systems,
        TotalMass,
        _target_satisfied,
        _target_system_mask,
    )

    return Systems, Stars, TotalMass, _target_system_mask, _target_satisfied


class TestTargetTypes:
    def test_construct_and_fields(self):
        Systems, Stars, TotalMass, _, _ = _imports()
        assert Systems(10).n == 10
        assert Stars(500).n == 500
        assert TotalMass(1.0e4).m == 1.0e4

    def test_hashable_static(self):
        # Targets are static config (drive shapes) -> must be hashable for jit static args.
        Systems, Stars, TotalMass, _, _ = _imports()
        assert hash(Systems(5)) == hash(Systems(5))
        assert {Systems(5), Stars(5), TotalMass(5.0)}  # no TypeError


class TestSystemsMask:
    def test_all_kept_when_len_equals_n(self):
        Systems, _, _, mask, _ = _imports()
        is_binary = jnp.array([True, False, True, False, True])
        m = jnp.array([2.0, 1.0, 2.0, 1.0, 2.0])
        keep = mask(Systems(5), is_binary, m)
        assert bool(jnp.all(keep))

    def test_prefix_when_overdrawn(self):
        Systems, _, _, mask, _ = _imports()
        is_binary = jnp.array([True, False, True, False, True])
        m = jnp.ones(5)
        keep = mask(Systems(3), is_binary, m)
        assert keep.tolist() == [True, True, True, False, False]


class TestStarsMask:
    def test_all_singles_exact(self):
        _, Stars, _, mask, _ = _imports()
        is_binary = jnp.array([False] * 6)
        m = jnp.ones(6)
        keep = mask(Stars(4), is_binary, m)
        # 1 star/system -> keep exactly 4 systems = 4 stars
        n_stars = int(jnp.sum((1 + is_binary.astype(jnp.int32))[keep]))
        assert n_stars == 4

    def test_all_binaries_overshoot_by_one(self):
        _, Stars, _, mask, _ = _imports()
        is_binary = jnp.array([True] * 6)
        m = 2.0 * jnp.ones(6)
        keep = mask(Stars(5), is_binary, m)
        n_stars = int(jnp.sum((1 + is_binary.astype(jnp.int32))[keep]))
        # 2 stars/system: crossing system takes 4 -> 6, so n_stars == 6 (n+1)
        assert 5 <= n_stars <= 6
        assert n_stars == 6

    def test_mixed_within_one(self):
        _, Stars, _, mask, _ = _imports()
        is_binary = jnp.array([True, False, True, True, False, True, False])
        m = jnp.ones(7)
        n = 5
        keep = mask(Stars(n), is_binary, m)
        n_stars = int(jnp.sum((1 + is_binary.astype(jnp.int32))[keep]))
        assert n <= n_stars <= n + 1


class TestTotalMassMask:
    def test_cut_reaches_budget_minimal_overshoot(self):
        _, _, TotalMass, mask, _ = _imports()
        sys_mass = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
        M = 5.0
        keep = mask(TotalMass(M), sys_mass, sys_mass)  # is_binary unused for TotalMass
        kept_mass = float(jnp.sum(sys_mass[keep]))
        assert kept_mass >= M
        assert kept_mass <= M + float(jnp.max(sys_mass))  # overshoot <= one system

    def test_variable_masses(self):
        _, _, TotalMass, mask, _ = _imports()
        sys_mass = jnp.array([0.5, 3.0, 1.0, 10.0, 0.2])
        M = 4.0
        keep = mask(TotalMass(M), jnp.zeros(5, bool), sys_mass)
        # cumulative-before = [0,0.5,3.5,4.5,14.5]; keep where <4 -> [T,T,T,F,F]
        # (system 2 crosses 4.0 at 4.5; the wide 10.0 system is correctly excluded)
        assert keep.tolist() == [True, True, True, False, False]
        assert float(jnp.sum(sys_mass[keep])) >= M


class TestSatisfiedPredicate:
    def test_systems(self):
        Systems, _, _, _, satisfied = _imports()
        ib = jnp.zeros(5, bool)
        assert satisfied(Systems(5), ib, jnp.ones(5))
        assert not satisfied(Systems(6), ib, jnp.ones(5))

    def test_stars_counts_companions(self):
        _, Stars, _, _, satisfied = _imports()
        ib = jnp.array([True, True, False])  # 2+2+1 = 5 stars from 3 systems
        m = jnp.array([2.0, 2.0, 1.0])
        assert satisfied(Stars(5), ib, m)
        assert not satisfied(Stars(6), ib, m)

    def test_totalmass(self):
        _, _, TotalMass, _, satisfied = _imports()
        m = jnp.array([2.0, 2.0, 2.0])
        assert satisfied(TotalMass(6.0), jnp.zeros(3, bool), m)
        assert not satisfied(TotalMass(6.001), jnp.zeros(3, bool), m)
