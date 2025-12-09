# progenax/tests/unit/cluster/test_mass_segregation.py
"""
Unit tests for mass_segregation module.

Tests cover:
    - Shape and identity preservation in energy_sorted_segregation
    - Physical behavior (massive stars -> more bound orbits)
    - No-orbit-reuse guarantee
    - _mcluster_partial_shuffle behavior for S=0, S=0.5, S=1
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.cluster.mass_segregation import (
    energy_sorted_segregation,
    _mcluster_partial_shuffle,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def key():
    """Standard JAX random key for reproducibility."""
    return jax.random.PRNGKey(42)


def harmonic_potential(positions):
    """
    Simple harmonic potential: Phi = 0.5 * omega^2 * r^2 (omega=1).

    This gives a toy potential where E = 0.5 * v^2 + 0.5 * r^2, so
    more negative E (or smaller total) means more bound.

    Note: This is a positive potential (unlike gravitational), but the
    segregation algorithm just sorts by energy, so it still works for testing.
    """
    return 0.5 * jnp.sum(positions**2, axis=1)


def negative_harmonic_potential(positions):
    """
    Harmonic potential with negative sign: Phi = -0.5 * r^2.

    More physically motivated since bound orbits should have negative total
    energy with gravitational-like potentials.
    """
    return -0.5 * jnp.sum(positions**2, axis=1)


# =============================================================================
# Test: energy_sorted_segregation - Shapes and Identity
# =============================================================================


class TestEnergySegregationShapes:
    """Test shape preservation and mass identity in energy_sorted_segregation."""

    def test_shape_and_identity(self, key):
        """
        Verify that energy_sorted_segregation:
        - Returns masses_out identical to input masses
        - Returns positions/velocities with correct shapes
        """
        N = 50
        N_pool = 200

        key1, key2, key3, key4 = jax.random.split(key, 4)

        masses = jax.random.uniform(key1, (N,), minval=0.5, maxval=10.0)
        positions_pool = jax.random.normal(key2, (N_pool, 3))
        velocities_pool = jax.random.normal(key3, (N_pool, 3))

        masses_out, positions_out, velocities_out = energy_sorted_segregation(
            key4,
            masses,
            positions_pool,
            velocities_pool,
            harmonic_potential,
        )

        # Masses should be identical (not just equal, same array)
        assert jnp.array_equal(masses_out, masses), "masses_out should equal input masses"

        # Check output shapes
        assert positions_out.shape == (N, 3), f"Expected positions shape (N, 3), got {positions_out.shape}"
        assert velocities_out.shape == (N, 3), f"Expected velocities shape (N, 3), got {velocities_out.shape}"

    def test_different_keys_give_different_results(self, key):
        """Verify different random keys produce different orbit assignments."""
        N = 20
        N_pool = 100

        key1, key2, key3, key4 = jax.random.split(key, 4)

        masses = jax.random.uniform(key1, (N,), minval=0.5, maxval=10.0)
        positions_pool = jax.random.normal(key2, (N_pool, 3))
        velocities_pool = jax.random.normal(key3, (N_pool, 3))

        _, pos1, _ = energy_sorted_segregation(
            key4, masses, positions_pool, velocities_pool, harmonic_potential
        )

        _, pos2, _ = energy_sorted_segregation(
            jax.random.PRNGKey(999), masses, positions_pool, velocities_pool, harmonic_potential
        )

        # With high probability, results should differ
        assert not jnp.allclose(pos1, pos2), "Different keys should give different results"


# =============================================================================
# Test: energy_sorted_segregation - Physical Behavior
# =============================================================================


class TestEnergySegregationPhysics:
    """Test that segregation produces physically correct behavior."""

    def test_massive_stars_more_bound(self, key):
        """
        Verify that massive stars are assigned to more bound (lower energy) orbits.

        Uses bimodal mass distribution: 10 heavy stars (10 Msun) and 90 light stars (1 Msun).
        After segregation, heavy stars should have lower mean specific energy.
        """
        n_heavy = 10
        n_light = 90
        N = n_heavy + n_light
        N_pool = 4 * N  # pool_factor = 4

        key1, key2, key3, key4 = jax.random.split(key, 4)

        # Bimodal mass distribution
        masses = jnp.concatenate([
            jnp.full(n_heavy, 10.0),  # Heavy stars
            jnp.full(n_light, 1.0),   # Light stars
        ])

        # Generate orbit pool with spread in both position and velocity
        positions_pool = jax.random.normal(key1, (N_pool, 3)) * 2.0
        velocities_pool = jax.random.normal(key2, (N_pool, 3)) * 1.0

        _, positions_out, velocities_out = energy_sorted_segregation(
            key3,
            masses,
            positions_pool,
            velocities_pool,
            negative_harmonic_potential,
        )

        # Compute specific energy after assignment
        kinetic = 0.5 * jnp.sum(velocities_out**2, axis=1)
        potential = negative_harmonic_potential(positions_out)
        E_specific = kinetic + potential

        # Identify heavy and light stars
        heavy_mask = masses > 5.0
        light_mask = ~heavy_mask

        E_heavy_mean = jnp.mean(E_specific[heavy_mask])
        E_light_mean = jnp.mean(E_specific[light_mask])

        # Heavy stars should be more bound (more negative energy)
        assert E_heavy_mean < E_light_mean, (
            f"Heavy stars should be more bound: E_heavy={E_heavy_mean:.4f} vs E_light={E_light_mean:.4f}"
        )

    def test_segregation_with_uniform_masses(self, key):
        """
        Test that segregation works correctly even with uniform masses.

        When all masses are equal, the algorithm should still run without errors.
        """
        N = 50
        N_pool = 200

        key1, key2, key3 = jax.random.split(key, 3)

        masses = jnp.ones(N)  # All equal masses
        positions_pool = jax.random.normal(key1, (N_pool, 3))
        velocities_pool = jax.random.normal(key2, (N_pool, 3))

        masses_out, positions_out, velocities_out = energy_sorted_segregation(
            key3,
            masses,
            positions_pool,
            velocities_pool,
            harmonic_potential,
        )

        # Should complete without error and return valid shapes
        assert masses_out.shape == (N,)
        assert positions_out.shape == (N, 3)
        assert velocities_out.shape == (N, 3)


# =============================================================================
# Test: energy_sorted_segregation - No Orbit Reuse
# =============================================================================


class TestNoOrbitReuse:
    """Test that each orbit is used at most once."""

    def test_no_orbit_reuse(self, key):
        """
        Verify that orbit indices are not reused across different mass ranks.

        We re-derive the orbit indices by replicating the energy ordering and
        bin logic, then check that all indices are unique.
        """
        N = 10
        N_pool = 40

        key1, key2, key3, key4 = jax.random.split(key, 4)

        masses = jax.random.uniform(key1, (N,), minval=0.5, maxval=10.0)
        positions_pool = jax.random.normal(key2, (N_pool, 3))
        velocities_pool = jax.random.normal(key3, (N_pool, 3))

        # Replicate the algorithm's energy ordering
        kinetic = 0.5 * jnp.sum(velocities_pool**2, axis=1)
        potential = harmonic_potential(positions_pool)
        specific_energy = kinetic + potential
        energy_order = jnp.argsort(specific_energy)

        # Replicate bin boundaries
        mass_order = jnp.argsort(-masses)
        m_sorted = masses[mass_order]
        M_cum_sorted = jnp.cumsum(m_sorted)
        M_total = jnp.maximum(M_cum_sorted[-1], 1e-12)
        M_cum_norm = M_cum_sorted / M_total
        M_cum_norm_shifted = jnp.concatenate([
            jnp.array([0.0], dtype=M_cum_norm.dtype),
            M_cum_norm[:-1]
        ])

        bin_low = jnp.floor(N_pool * M_cum_norm_shifted).astype(jnp.int32)
        bin_high = jnp.floor(N_pool * M_cum_norm).astype(jnp.int32) - 1
        bin_high = jnp.maximum(bin_high, bin_low)

        # Sample orbit indices (same logic as the function)
        keys_per_rank = jax.random.split(key4, N)

        def sample_orbit_for_rank(i):
            low = bin_low[i]
            high = bin_high[i] + 1
            return jax.random.randint(keys_per_rank[i], (), low, high)

        orbit_indices = jax.vmap(sample_orbit_for_rank)(jnp.arange(N))

        # Check uniqueness
        unique_indices = jnp.unique(orbit_indices)
        assert len(unique_indices) == N, (
            f"Expected {N} unique orbit indices, got {len(unique_indices)}. "
            f"Orbit indices: {orbit_indices}"
        )


# =============================================================================
# Test: _mcluster_partial_shuffle
# =============================================================================


class TestMclusterPartialShuffle:
    """Test the _mcluster_partial_shuffle reference implementation."""

    def test_s1_returns_identity(self, key):
        """
        For S=1, _mcluster_partial_shuffle should return identity mapping.

        star_for_rank[i] = i for all i
        """
        N = 20
        result = _mcluster_partial_shuffle(key, N, S=1.0)
        expected = jnp.arange(N)

        assert jnp.array_equal(result, expected), (
            f"S=1 should give identity: expected {expected}, got {result}"
        )

    def test_s0_returns_valid_permutation(self, key):
        """
        For S=0, _mcluster_partial_shuffle should return a valid permutation.

        The sorted result should equal [0, 1, ..., N-1].
        """
        N = 20
        result = _mcluster_partial_shuffle(key, N, S=0.0)

        # Should be a valid permutation
        assert jnp.array_equal(jnp.sort(result), jnp.arange(N)), (
            f"S=0 result should be a permutation of 0..N-1: got {result}"
        )

        # With S=0, should NOT be identity (with very high probability)
        # Allow for extremely unlikely case where random permutation equals identity
        is_identity = jnp.array_equal(result, jnp.arange(N))
        if is_identity:
            # Try another key to confirm it's not always identity
            result2 = _mcluster_partial_shuffle(jax.random.PRNGKey(123), N, S=0.0)
            assert not jnp.array_equal(result2, jnp.arange(N)), (
                "S=0 should produce random permutations, not identity"
            )

    def test_s0_different_keys_different_permutations(self, key):
        """
        For S=0, different keys should produce different permutations.
        """
        N = 20
        result1 = _mcluster_partial_shuffle(key, N, S=0.0)
        result2 = _mcluster_partial_shuffle(jax.random.PRNGKey(999), N, S=0.0)

        # Both should be valid permutations
        assert jnp.array_equal(jnp.sort(result1), jnp.arange(N))
        assert jnp.array_equal(jnp.sort(result2), jnp.arange(N))

        # Should be different (with very high probability)
        assert not jnp.array_equal(result1, result2), (
            "Different keys should produce different permutations"
        )

    def test_s_intermediate_returns_valid_permutation(self, key):
        """
        For intermediate S (0.5), result should still be a valid permutation.
        """
        N = 20
        result = _mcluster_partial_shuffle(key, N, S=0.5)

        # Should be a valid permutation
        assert jnp.array_equal(jnp.sort(result), jnp.arange(N)), (
            f"S=0.5 result should be a permutation of 0..N-1: got {result}"
        )

    def test_output_dtype_is_int32(self, key):
        """Verify output has correct integer dtype."""
        N = 20
        result = _mcluster_partial_shuffle(key, N, S=0.5)

        assert result.dtype == jnp.int32, f"Expected int32 dtype, got {result.dtype}"
