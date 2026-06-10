# progenax/tests/unit/cluster/test_mass_segregation.py
"""Unit tests for the energy-ordered mass-segregation assignment.

Every test is a *contract* or *physics* check that a wrong implementation would fail —
no shape-only smoke tests, no re-derivation of the function's own internals.

Covers `energy_sorted_segregation` (Baumgardt+2008 S=1 orbit assignment):
    - no orbit reuse for ANY mass spectrum (regression for the cumulative-mass
      bin-collapse bug that produced coincident stars for steep IMFs)
    - the S=1 physics: assigned specific energy is monotonic in mass
    - the assignment selects pool orbits (never interpolates) and is deterministic
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import spearmanr

from progenax.cluster.mass_segregation import energy_sorted_segregation


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


def harmonic_potential(positions):
    """Toy potential Phi = 0.5 r^2 (omega=1): orbits are ranked by E = 0.5 v^2 + 0.5 r^2."""
    return 0.5 * jnp.sum(positions**2, axis=1)


def negative_harmonic_potential(positions):
    """Gravitational-like Phi = -0.5 r^2 (bound orbits have negative total energy)."""
    return -0.5 * jnp.sum(positions**2, axis=1)


def _mass_spectrum(kind, key, N):
    """Mass spectra spanning the regimes that stress the orbit-binning."""
    if kind == "uniform":
        return jnp.ones(N)
    if kind == "bimodal":
        return jnp.where(jax.random.uniform(key, (N,)) < 0.1, 10.0, 1.0)
    if kind == "kroupa-like":  # steep: most stars low-mass
        return 0.08 * (100.0 / 0.08) ** jax.random.uniform(key, (N,)) ** 2.3
    if kind == "extreme-steep":  # 0.01-50 Msun, very bottom-heavy
        return 0.01 * (50.0 / 0.01) ** jax.random.uniform(key, (N,)) ** 3.0
    raise ValueError(kind)


class TestEnergySortedSegregation:
    """Contract + physics for the S=1 energy-ordered orbit assignment."""

    @pytest.mark.parametrize("spectrum",
                             ["uniform", "bimodal", "kroupa-like", "extreme-steep"])
    def test_no_orbit_reuse_for_any_mass_spectrum(self, key, spectrum):
        """The core contract, at the scale and IMF steepness that broke the old code:
        every star lands on a DISTINCT orbit, so no two stars are coincident.

        Regression: the previous per-bin sampler binned the energy-sorted pool by
        cumulative mass; for a steep IMF a low-mass star's bin is sub-orbit, so the
        clamp forced many ranks onto the SAME orbit -> coincident stars -> V = -inf.
        """
        N, N_pool = 600, 4 * 600
        k1, k2, k3, k4 = jax.random.split(key, 4)
        masses = _mass_spectrum(spectrum, k1, N)
        pos_pool = jax.random.normal(k2, (N_pool, 3))  # all distinct
        vel_pool = jax.random.normal(k3, (N_pool, 3))

        _, pos_out, _ = energy_sorted_segregation(
            k4, masses, pos_pool, vel_pool, harmonic_potential
        )
        n_unique = len(jnp.unique(jnp.round(pos_out, 8), axis=0))
        assert n_unique == N, f"{spectrum}: {N - n_unique} of {N} stars coincide (orbit reuse)"

    def test_assigned_energy_is_monotonic_in_mass(self, key):
        """S=1 physics, as one discriminating number: the assigned specific energy is a
        monotonically decreasing function of stellar mass (more massive -> more bound).
        Spearman rho ~ -1 for the correct ordering; an inverted or partial assignment
        fails. Uses a continuous spectrum so there are no mass ties."""
        N, N_pool = 400, 4 * 400
        k1, k2, k3, k4 = jax.random.split(key, 4)
        masses = jax.random.uniform(k1, (N,), minval=0.1, maxval=50.0)
        pos_pool = jax.random.normal(k2, (N_pool, 3)) * 2.0
        vel_pool = jax.random.normal(k3, (N_pool, 3))

        _, pos_out, vel_out = energy_sorted_segregation(
            k4, masses, pos_pool, vel_pool, negative_harmonic_potential
        )
        E = 0.5 * jnp.sum(vel_out**2, axis=1) + negative_harmonic_potential(pos_out)
        rho = spearmanr(np.asarray(masses), np.asarray(E)).correlation
        assert rho < -0.95, f"mass->energy ordering not monotonic (Spearman={rho:.3f})"

    def test_most_massive_star_is_the_most_bound(self, key):
        """The single most massive star is assigned the most bound orbit of the realised
        cluster (the exact bound-end of the S=1 ordering)."""
        N, N_pool = 300, 4 * 300
        k1, k2, k3, k4 = jax.random.split(key, 4)
        masses = jax.random.uniform(k1, (N,), minval=0.5, maxval=20.0)
        pos_pool = jax.random.normal(k2, (N_pool, 3))
        vel_pool = jax.random.normal(k3, (N_pool, 3))

        _, pos_out, vel_out = energy_sorted_segregation(
            k4, masses, pos_pool, vel_pool, negative_harmonic_potential
        )
        E = 0.5 * jnp.sum(vel_out**2, axis=1) + negative_harmonic_potential(pos_out)
        most_massive = int(jnp.argmax(masses))
        n_more_bound = int(jnp.sum(E < E[most_massive]))
        assert n_more_bound == 0, f"{n_more_bound} stars more bound than the most massive"

    def test_assignment_selects_pool_orbits_and_preserves_masses(self, key):
        """The function SELECTS orbits from the pool (never interpolates): every output
        (position, velocity) pair is an exact pool member, and masses are returned
        unchanged in the caller's ordering."""
        N, N_pool = 80, 320
        k1, k2, k3, k4 = jax.random.split(key, 4)
        masses = jax.random.uniform(k1, (N,), minval=0.5, maxval=10.0)
        pos_pool = jax.random.normal(k2, (N_pool, 3))
        vel_pool = jax.random.normal(k3, (N_pool, 3))

        m_out, pos_out, vel_out = energy_sorted_segregation(
            k4, masses, pos_pool, vel_pool, harmonic_potential
        )
        assert jnp.array_equal(m_out, masses)
        pool = set(map(tuple, np.asarray(jnp.concatenate([pos_pool, vel_pool], axis=1)).round(8)))
        out = np.asarray(jnp.concatenate([pos_out, vel_out], axis=1)).round(8)
        assert all(tuple(row) in pool for row in out), "an output orbit is not a pool member"

    def test_deterministic_given_the_pool(self, key):
        """The assignment is deterministic given (masses, pool): repeated calls with
        DIFFERENT keys return identical catalogs. Realisation variety comes from the
        random orbit pool, not from the assignment (which is a fixed monotonic map)."""
        N, N_pool = 100, 400
        k1, k2, k3 = jax.random.split(key, 3)
        masses = jax.random.uniform(k1, (N,), minval=0.5, maxval=10.0)
        pos_pool = jax.random.normal(k2, (N_pool, 3))
        vel_pool = jax.random.normal(k3, (N_pool, 3))

        _, p1, v1 = energy_sorted_segregation(
            jax.random.PRNGKey(1), masses, pos_pool, vel_pool, harmonic_potential)
        _, p2, v2 = energy_sorted_segregation(
            jax.random.PRNGKey(2), masses, pos_pool, vel_pool, harmonic_potential)
        assert jnp.array_equal(p1, p2) and jnp.array_equal(v1, v2)

    def test_different_pools_give_different_assignments(self, key):
        """Variety enters through the orbit pool: two independent pools (same masses)
        yield genuinely different catalogs."""
        N, N_pool = 100, 400
        k1, ka, kb = jax.random.split(key, 3)
        masses = jax.random.uniform(k1, (N,), minval=0.5, maxval=10.0)

        def draw(kp):
            kk1, kk2 = jax.random.split(kp)
            return jax.random.normal(kk1, (N_pool, 3)), jax.random.normal(kk2, (N_pool, 3))

        pa, va = draw(ka)
        pb, vb = draw(kb)
        _, p1, _ = energy_sorted_segregation(key, masses, pa, va, harmonic_potential)
        _, p2, _ = energy_sorted_segregation(key, masses, pb, vb, harmonic_potential)
        assert not jnp.allclose(p1, p2)

    def test_jit_matches_eager(self, key):
        """The assignment runs under jit (it is called inside the jitted IC generator)
        and produces the identical result to eager execution."""
        N, N_pool = 100, 400
        k1, k2, k3, k4 = jax.random.split(key, 4)
        masses = jax.random.uniform(k1, (N,), minval=0.5, maxval=10.0)
        pos_pool = jax.random.normal(k2, (N_pool, 3))
        vel_pool = jax.random.normal(k3, (N_pool, 3))

        eager = energy_sorted_segregation(k4, masses, pos_pool, vel_pool, harmonic_potential)[1]
        jit_fn = jax.jit(lambda k, m, p, v: energy_sorted_segregation(k, m, p, v, harmonic_potential)[1])
        assert jnp.allclose(eager, jit_fn(k4, masses, pos_pool, vel_pool))
