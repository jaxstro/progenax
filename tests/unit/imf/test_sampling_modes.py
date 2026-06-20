"""Tests for the BaseIMF mass-target sampling modes (audit R5: previously zero tests).

Semantics (from progenax/imf/base.py):
  - sample_m_total(key, m_total, n_max=None) -> (masses_padded, n_live):
        prefix-cut; the realized total OVERSHOOTS m_total by at most one star
        (the boundary star pushes the cumsum past the target).
  - sample_m_total_packed(key, m_total, n_max) -> (masses_packed, n_live_float):
        fractional boundary star -> hits m_total (near-)exactly.
  - sample_fixed_n(key, n, m_total) -> (n,):
        stratified quantile stretch (q <= 1). The achievable total is capped at
        sum(ppf(u_base)); a target above that ceiling used to undershoot SILENTLY.
"""

import jax
import pytest

from progenax.imf import Maschberger


class TestSampleFixedN:
    def test_reachable_target_is_hit(self):
        imf = Maschberger()
        key = jax.random.PRNGKey(0)
        n = 1000
        # natural total ~ n * E[m]; pick a comfortably reachable target
        target = 250.0
        m = imf.sample_fixed_n(key, n, target)
        assert m.shape == (n,)
        assert abs(float(m.sum()) / target - 1.0) < 0.05  # ~target, not "exactly"

    def test_unreachable_target_raises(self):
        """Pre-fix behavior: silently returned 349 Msun for a 500 Msun target."""
        imf = Maschberger()
        key = jax.random.PRNGKey(0)
        with pytest.raises(ValueError, match="unreachable"):
            imf.sample_fixed_n(key, 1000, 500.0)

    def test_masses_within_imf_bounds(self):
        imf = Maschberger()
        m = imf.sample_fixed_n(jax.random.PRNGKey(1), 500, 120.0)
        assert float(m.min()) >= imf.m_min
        assert float(m.max()) <= imf.m_max


class TestSampleMTotalModes:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_sample_m_total_hits_target(self, seed):
        imf = Maschberger()
        masses, n_live = imf.sample_m_total(jax.random.PRNGKey(seed), m_total=500.0)
        total = float(masses.sum())
        # prefix-cut overshoots by AT MOST one (max-mass) star, never undershoots
        # (unless n_max is exhausted; the auto n_max is sized 2x so it isn't).
        assert 500.0 <= total <= 500.0 + imf.m_max
        assert int(n_live) == int((masses > 0).sum())

    def test_sample_m_total_packed_consistent_with_unpacked(self):
        imf = Maschberger()
        key = jax.random.PRNGKey(0)
        n_max = 4000  # generous: ~1500-2500 stars make 500 Msun
        packed, n_float = imf.sample_m_total_packed(key, 500.0, n_max)
        # the fractional boundary star makes the packed total hit the target
        # (near-)exactly -- well within one stellar mass.
        assert abs(float(packed.sum()) - 500.0) < imf.m_max
        # same draw, unpacked: the live count brackets the packed fractional count
        # (n_live = floor(boundary)+1; n_float = floor(boundary)+fraction).
        unpacked, n_live = imf.sample_m_total(key, 500.0, n_max=n_max)
        assert abs(float(n_float) - float(n_live)) <= 1.5
