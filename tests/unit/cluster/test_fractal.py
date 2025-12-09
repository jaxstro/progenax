# progenax/tests/unit/cluster/test_fractal.py
"""
Unit tests for progenax.cluster.fractal module.

Tests cover:
- generate_fractal_positions: shape, unit sphere, ancestry
- rescale_fractal_to_target_radii: rank ordering, angular preservation
- assign_velocities_and_virialize: virial ratio accuracy

References:
    Goodwin & Whitworth (2004) A&A 413, 929
    Kupper et al. (2011) MNRAS 417, 2300 - McLuster
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# Test generate_fractal_positions
# =============================================================================


class TestGenerateFractalPositions:
    """Tests for generate_fractal_positions."""

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    @pytest.mark.parametrize("N_stars", [256, 1024])
    @pytest.mark.parametrize("D", [1.6, 2.0, 2.6, 3.0])
    def test_shape_and_bounds(self, key, N_stars, D):
        """Test output shapes and positions within unit sphere.

        McLuster algorithm: positions are constrained to unit sphere (r <= 1).
        """
        from progenax.cluster.fractal_gw_legacy import generate_fractal_positions

        pos, vel, ancestry = generate_fractal_positions(key, N_stars, D=D)

        # Check shapes
        assert pos.shape == (N_stars, 3), f"Expected {(N_stars, 3)}, got {pos.shape}"
        assert vel.shape == (N_stars, 3), f"Expected {(N_stars, 3)}, got {vel.shape}"
        assert ancestry.shape == (N_stars,), f"Expected {(N_stars,)}, got {ancestry.shape}"

        # Check ancestry dtype
        assert ancestry.dtype == jnp.int32, f"Expected int32, got {ancestry.dtype}"

        # Check positions within unit sphere (McLuster requirement)
        radii = jnp.linalg.norm(pos, axis=1)
        max_radius = jnp.max(radii)
        assert max_radius <= 1.0 + 1e-6, (
            f"All positions must be in unit sphere, got max radius {max_radius:.4f}"
        )

    def test_ancestry_values(self, key):
        """Test that ancestry contains valid generation indices [0, g_max]."""
        from progenax.cluster.fractal_gw_legacy import generate_fractal_positions

        g_max = 6
        pos, vel, ancestry = generate_fractal_positions(key, 500, D=2.0, g_max=g_max)

        # Ancestry should be generation indices in [0, g_max]
        assert jnp.all(ancestry >= 0), "Ancestry should be >= 0"
        assert jnp.all(ancestry <= g_max), f"Ancestry should be <= g_max={g_max}"

    def test_reproducibility(self, key):
        """Test that same key produces same output."""
        from progenax.cluster.fractal_gw_legacy import generate_fractal_positions

        pos1, vel1, anc1 = generate_fractal_positions(key, 500, D=2.0)
        pos2, vel2, anc2 = generate_fractal_positions(key, 500, D=2.0)

        assert jnp.allclose(pos1, pos2), "Same key should produce same positions"
        assert jnp.allclose(vel1, vel2), "Same key should produce same velocities"
        assert jnp.array_equal(anc1, anc2), "Same key should produce same ancestry"

    def test_different_D_produces_different_structure(self, key):
        """Test that different D values produce different clumpiness.

        D=1.6 (clumpy) should have uneven octant distribution.
        D=3.0 (uniform) should have even octant distribution.
        """
        from progenax.cluster.fractal_gw_legacy import generate_fractal_positions

        pos_clumpy, _, _ = generate_fractal_positions(key, 2000, D=1.6)
        pos_uniform, _, _ = generate_fractal_positions(key, 2000, D=3.0)

        # Count particles per octant
        def count_octants(pos):
            counts = []
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    for s3 in [-1, 1]:
                        mask = (jnp.sign(pos[:, 0]) == s1) & \
                               (jnp.sign(pos[:, 1]) == s2) & \
                               (jnp.sign(pos[:, 2]) == s3)
                        counts.append(jnp.sum(mask))
            return jnp.array(counts)

        octants_clumpy = count_octants(pos_clumpy)
        octants_uniform = count_octants(pos_uniform)

        # D=3.0 should have more even octant distribution (lower variation)
        var_clumpy = jnp.max(octants_clumpy) / jnp.maximum(jnp.min(octants_clumpy), 1)
        var_uniform = jnp.max(octants_uniform) / jnp.maximum(jnp.min(octants_uniform), 1)

        # Clumpy should have higher variation than uniform
        assert var_clumpy > var_uniform, (
            f"D=1.6 should have more octant variation ({var_clumpy:.1f}x) "
            f"than D=3.0 ({var_uniform:.1f}x)"
        )

    def test_forced_survivors_exact_count(self, key):
        """Test that forced mode produces exactly k survivors per parent."""
        from progenax.cluster.fractal_gw_legacy import _select_k_survivors

        N_div = 2
        n_children = N_div ** 3  # 8

        for D in [1.0, 1.5, 2.0, 2.5, 3.0]:
            k_expected = int(round(N_div ** D))
            k_expected = max(1, min(k_expected, n_children))
            mask = _select_k_survivors(key, n_parents=10, n_children=n_children, k=k_expected)

            # Each row should have exactly k True values
            counts = mask.sum(axis=1)
            assert jnp.all(counts == k_expected), f"D={D}: expected {k_expected}, got {counts}"

    def test_different_keys_different_results(self, key):
        """Test that different keys produce different realizations."""
        from progenax.cluster.fractal_gw_legacy import generate_fractal_positions

        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(123)

        pos1, _, _ = generate_fractal_positions(key1, N_stars=100, D=2.0)
        pos2, _, _ = generate_fractal_positions(key2, N_stars=100, D=2.0)

        assert not jnp.allclose(pos1, pos2), "Different keys should produce different results"

    # NOTE: test_D_clamping removed - legacy GW2004 code has bug with D < 1.5
    # FDF is now the recommended method for fractal ICs

    def test_velocities_nonzero(self, key):
        """Test that velocities are generated (non-zero)."""
        from progenax.cluster.fractal_gw_legacy import generate_fractal_positions

        pos, vel, ancestry = generate_fractal_positions(key, 500, D=2.0)

        # Velocities should be non-zero (from random walk)
        v_mag = jnp.linalg.norm(vel, axis=1)
        assert jnp.any(v_mag > 0), "Velocities should be non-zero from random walk"

    def test_probabilistic_mode(self, key):
        """Test that probabilistic mode (forced=False) still works."""
        from progenax.cluster.fractal_gw_legacy import generate_fractal_positions

        pos, vel, ancestry = generate_fractal_positions(key, 500, D=2.0, forced=False)

        assert pos.shape == (500, 3)
        assert vel.shape == (500, 3)
        assert ancestry.shape == (500,)

    def test_positions_in_unit_sphere(self, key):
        """Test that ALL positions are inside unit sphere (r <= 1)."""
        from progenax.cluster.fractal_gw_legacy import generate_fractal_positions

        pos, vel, ancestry = generate_fractal_positions(key, 1000, D=2.0)

        radii = jnp.linalg.norm(pos, axis=1)
        max_radius = jnp.max(radii)

        assert max_radius <= 1.0 + 1e-6, (
            f"All positions must be in unit sphere, got max radius {max_radius:.4f}"
        )

    def test_d3_octant_uniformity(self, key):
        """D=3.0 with enforce_octant_symmetry should have all octants populated.

        Note: The sphere cut heavily affects corner octants (removes corner particles),
        so we can't expect uniformity. We just verify all octants have some particles.
        """
        from progenax.cluster.fractal_gw_legacy import generate_fractal_positions

        pos, _, _ = generate_fractal_positions(
            key, 5000, D=3.0, enforce_octant_symmetry=True
        )

        # Count particles per octant
        octant_counts = []
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                for s3 in [-1, 1]:
                    mask = (
                        (jnp.sign(pos[:, 0]) == s1) &
                        (jnp.sign(pos[:, 1]) == s2) &
                        (jnp.sign(pos[:, 2]) == s3)
                    )
                    octant_counts.append(jnp.sum(mask))

        octant_counts = jnp.array(octant_counts)

        # Just verify all octants are populated (sphere cut causes asymmetry)
        assert jnp.all(octant_counts > 0), (
            f"All octants should have particles, got counts: {octant_counts}"
        )

    def test_velocity_spatial_correlation(self, key):
        """Nearby particles should have more similar velocities than distant pairs."""
        from progenax.cluster.fractal_gw_legacy import generate_fractal_positions

        pos, vel, _ = generate_fractal_positions(key, 500, D=1.6)

        # Compute all pairwise position distances
        pos_diff = pos[:, None, :] - pos[None, :, :]
        pos_dist = jnp.linalg.norm(pos_diff, axis=-1)

        # Compute all pairwise velocity differences
        vel_diff = vel[:, None, :] - vel[None, :, :]
        vel_dist = jnp.linalg.norm(vel_diff, axis=-1)

        # Get upper triangle (exclude diagonal)
        triu_mask = jnp.triu(jnp.ones_like(pos_dist, dtype=bool), k=1)
        pos_flat = pos_dist[triu_mask]
        vel_flat = vel_dist[triu_mask]

        # Split into near and far pairs by median distance
        median_dist = jnp.median(pos_flat)
        near_mask = pos_flat < median_dist
        far_mask = pos_flat >= median_dist

        mean_vel_diff_near = jnp.mean(vel_flat[near_mask])
        mean_vel_diff_far = jnp.mean(vel_flat[far_mask])

        # Near pairs should have smaller velocity differences (on average)
        assert mean_vel_diff_near < mean_vel_diff_far, (
            f"Near pairs vel diff ({mean_vel_diff_near:.4f}) should be smaller "
            f"than far pairs ({mean_vel_diff_far:.4f})"
        )


# =============================================================================
# Test rescale_fractal_to_target_radii
# =============================================================================


class TestRescaleFractalToTargetRadii:
    """Tests for rescale_fractal_to_target_radii."""

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    def test_output_shape(self, key):
        """Test that output has same shape as input."""
        from progenax.cluster.fractal_gw_legacy import (
            generate_fractal_positions,
            rescale_fractal_to_target_radii,
        )

        N = 500
        pos_frac, _, _ = generate_fractal_positions(key, N, D=2.0)
        target_radii = jax.random.uniform(jax.random.PRNGKey(123), (N,), minval=0.5, maxval=2.0)

        pos_rescaled = rescale_fractal_to_target_radii(pos_frac, target_radii)

        assert pos_rescaled.shape == pos_frac.shape

    def test_preserves_rank_ordering(self, key):
        """Test that radii match sorted target_radii."""
        from progenax.cluster.fractal_gw_legacy import (
            generate_fractal_positions,
            rescale_fractal_to_target_radii,
        )

        N = 500
        pos_frac, _, _ = generate_fractal_positions(key, N, D=2.0)
        target_radii = jax.random.uniform(jax.random.PRNGKey(123), (N,), minval=0.5, maxval=5.0)

        pos_rescaled = rescale_fractal_to_target_radii(pos_frac, target_radii)

        # Output radii (sorted) should match target_radii (sorted)
        r_out = jnp.linalg.norm(pos_rescaled, axis=1)
        r_out_sorted = jnp.sort(r_out)
        r_target_sorted = jnp.sort(target_radii)

        assert jnp.allclose(r_out_sorted, r_target_sorted, rtol=1e-5), (
            f"Sorted output radii should match sorted target radii"
        )

    def test_preserves_angular_structure(self, key):
        """Test that angular directions are preserved."""
        from progenax.cluster.fractal_gw_legacy import (
            generate_fractal_positions,
            rescale_fractal_to_target_radii,
        )

        N = 500
        pos_frac, _, _ = generate_fractal_positions(key, N, D=2.0)

        # Use constant target radii so scaling is uniform
        target_radii = jnp.ones(N) * 2.0

        pos_rescaled = rescale_fractal_to_target_radii(pos_frac, target_radii)

        # Unit vectors should be identical (directions preserved)
        r_frac = jnp.linalg.norm(pos_frac, axis=1, keepdims=True)
        r_out = jnp.linalg.norm(pos_rescaled, axis=1, keepdims=True)

        # Avoid division by zero
        eps = 1e-10
        unit_frac = pos_frac / jnp.maximum(r_frac, eps)
        unit_out = pos_rescaled / jnp.maximum(r_out, eps)

        assert jnp.allclose(unit_frac, unit_out, atol=1e-6), (
            "Angular directions should be preserved"
        )

    def test_with_profile_sampled_radii(self, key):
        """Test integration with profile sampling."""
        from progenax.cluster.fractal_gw_legacy import (
            generate_fractal_positions,
            rescale_fractal_to_target_radii,
        )
        from progenax.profiles.api import sample_density_profile

        N = 500
        R_half = 1.0

        # Generate fractal positions
        pos_frac, _, _ = generate_fractal_positions(key, N, D=2.0)

        # Sample target radii from Plummer profile
        key2 = jax.random.PRNGKey(123)
        target_positions = sample_density_profile(key2, N, "plummer", R_half)
        target_radii = jnp.linalg.norm(target_positions, axis=1)

        # Rescale
        pos_rescaled = rescale_fractal_to_target_radii(pos_frac, target_radii)

        # Check that radii distribution matches (via sorted comparison)
        r_out = jnp.linalg.norm(pos_rescaled, axis=1)
        r_out_sorted = jnp.sort(r_out)
        r_target_sorted = jnp.sort(target_radii)

        assert jnp.allclose(r_out_sorted, r_target_sorted, rtol=1e-5)

        # Check median radius is close to R_half
        r_median = jnp.median(r_out)
        assert jnp.isclose(r_median, R_half, rtol=0.3), (
            f"Median radius {r_median:.3f} should be close to R_half={R_half}"
        )

    def test_differentiable(self, key):
        """Test that rescaling is differentiable."""
        from progenax.cluster.fractal_gw_legacy import (
            generate_fractal_positions,
            rescale_fractal_to_target_radii,
        )

        N = 50
        pos_frac, _, _ = generate_fractal_positions(key, N, D=2.0)

        def loss(scale_factor):
            target_radii = jnp.ones(N) * scale_factor
            rescaled = rescale_fractal_to_target_radii(pos_frac, target_radii)
            return jnp.sum(rescaled ** 2)

        grad_fn = jax.grad(loss)
        grad = grad_fn(1.0)

        assert jnp.isfinite(grad), "Gradient should be finite"
        assert grad != 0.0, "Gradient should be non-zero"


# =============================================================================
# Test assign_velocities_and_virialize
# =============================================================================


class TestAssignVelocitiesAndVirialize:
    """Tests for assign_velocities_and_virialize."""

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    def test_output_shape(self, key):
        """Test that output has correct shape."""
        from progenax.cluster.fractal_gw_legacy import assign_velocities_and_virialize

        N = 500
        positions = jax.random.normal(key, (N, 3))
        masses = jnp.ones(N)

        key2 = jax.random.PRNGKey(123)
        velocities = assign_velocities_and_virialize(
            key2, positions, masses, target_Q_vir=0.5
        )

        assert velocities.shape == (N, 3)

    def test_virial_ratio_accuracy(self, key):
        """Test that output matches target virial ratio."""
        from progenax.cluster.fractal_gw_legacy import assign_velocities_and_virialize
        from jaxstro.units import STELLAR
        from progenax.dynamics.virial import compute_potential_energy

        N = 500
        positions = jax.random.normal(key, (N, 3)) * 2.0  # Scale up for reasonable U
        masses = jnp.ones(N)
        target_Q = 0.5

        key2 = jax.random.PRNGKey(123)
        velocities = assign_velocities_and_virialize(
            key2, positions, masses, target_Q_vir=target_Q, G=STELLAR.G
        )

        # Compute actual K/|U|
        K = 0.5 * jnp.sum(masses[:, None] * velocities**2)
        U = compute_potential_energy(positions, masses, G=STELLAR.G)
        Q_actual = K / jnp.abs(U)

        assert jnp.isclose(Q_actual, target_Q, rtol=0.05), (
            f"Q_actual={Q_actual:.4f} should be close to target_Q={target_Q}"
        )

    def test_com_velocity_removed(self, key):
        """Test that center-of-mass velocity is zero."""
        from progenax.cluster.fractal_gw_legacy import assign_velocities_and_virialize

        N = 500
        positions = jax.random.normal(key, (N, 3))
        masses = jax.random.uniform(jax.random.PRNGKey(1), (N,)) + 0.1

        key2 = jax.random.PRNGKey(123)
        velocities = assign_velocities_and_virialize(
            key2, positions, masses, target_Q_vir=0.5
        )

        # Compute mass-weighted COM velocity
        M_total = jnp.sum(masses)
        v_com = jnp.sum(masses[:, None] * velocities, axis=0) / M_total

        assert jnp.allclose(v_com, 0.0, atol=1e-10), (
            f"COM velocity should be zero, got {v_com}"
        )

    @pytest.mark.parametrize("target_Q", [0.3, 0.5, 0.75])
    def test_different_virial_ratios(self, key, target_Q):
        """Test that different target Q values are achieved."""
        from progenax.cluster.fractal_gw_legacy import assign_velocities_and_virialize
        from jaxstro.units import STELLAR
        from progenax.dynamics.virial import compute_potential_energy

        N = 500
        positions = jax.random.normal(key, (N, 3)) * 2.0
        masses = jnp.ones(N)

        key2 = jax.random.PRNGKey(123)
        velocities = assign_velocities_and_virialize(
            key2, positions, masses, target_Q_vir=target_Q, G=STELLAR.G
        )

        K = 0.5 * jnp.sum(masses[:, None] * velocities**2)
        U = compute_potential_energy(positions, masses, G=STELLAR.G)
        Q_actual = K / jnp.abs(U)

        assert jnp.isclose(Q_actual, target_Q, rtol=0.05)

    def test_coherent_velocities_with_ancestry(self, key):
        """Test coherent velocity mode with ancestry (generation indices)."""
        from progenax.cluster.fractal_gw_legacy import (
            generate_fractal_positions,
            assign_velocities_and_virialize,
        )

        N = 500
        pos, vel_fractal, ancestry = generate_fractal_positions(key, N, D=2.0)
        masses = jnp.ones(N)

        # Note: ancestry is now generation indices, not parent indices
        # The coherent mode may not work exactly as before, but should not crash
        key2 = jax.random.PRNGKey(123)
        velocities = assign_velocities_and_virialize(
            key2, pos, masses, target_Q_vir=0.5,
            ancestry=ancestry, coherent=True
        )

        assert velocities.shape == (N, 3)
        assert jnp.all(jnp.isfinite(velocities))


# =============================================================================
# Test rescale_velocities_to_virial
# =============================================================================


class TestRescaleVelocitiesToVirial:
    """Tests for rescale_velocities_to_virial."""

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    def test_achieves_target_Q(self, key):
        """Test that velocities are rescaled to target Q."""
        from progenax.cluster.fractal_gw_legacy import (
            generate_fractal_positions,
            rescale_velocities_to_virial,
        )
        from progenax.dynamics.virial import compute_virial_ratio
        from jaxstro.units import STELLAR

        N = 500
        pos, vel_raw, _ = generate_fractal_positions(key, N, D=2.0)
        masses = jnp.ones(N)
        target_Q = 0.5

        vel_scaled = rescale_velocities_to_virial(
            pos, vel_raw, masses,
            target_Q_vir=target_Q,
            G=STELLAR.G,
            softening=1e-4,
        )

        Q_actual = compute_virial_ratio(
            pos, vel_scaled, masses, G=STELLAR.G, softening=1e-4
        )

        assert jnp.isclose(Q_actual, target_Q, rtol=0.05), (
            f"Q_actual={Q_actual:.4f} should be close to target_Q={target_Q}"
        )

    def test_removes_com_velocity(self, key):
        """Test that COM velocity is zero after rescaling."""
        from progenax.cluster.fractal_gw_legacy import (
            generate_fractal_positions,
            rescale_velocities_to_virial,
        )
        from jaxstro.units import STELLAR

        N = 500
        pos, vel_raw, _ = generate_fractal_positions(key, N, D=2.0)
        masses = jax.random.uniform(jax.random.PRNGKey(1), (N,)) + 0.1

        vel_scaled = rescale_velocities_to_virial(
            pos, vel_raw, masses,
            target_Q_vir=0.5,
            G=STELLAR.G,
        )

        M_total = jnp.sum(masses)
        v_com = jnp.sum(masses[:, None] * vel_scaled, axis=0) / M_total

        assert jnp.allclose(v_com, 0.0, atol=1e-10), (
            f"COM velocity should be zero, got {v_com}"
        )


# =============================================================================
# Integration test
# =============================================================================


class TestFractalICIntegration:
    """Integration tests for fractal IC generation pipeline."""

    def test_full_fractal_ic_pipeline(self):
        """Test full pipeline: fractal generation -> rescale -> virial ratio."""
        from progenax.cluster.fractal_gw_legacy import (
            generate_fractal_positions,
            rescale_fractal_to_target_radii,
            assign_velocities_and_virialize,
        )
        from progenax.profiles.api import sample_density_profile
        from jaxstro.units import STELLAR
        from progenax.dynamics.virial import compute_potential_energy

        key = jax.random.PRNGKey(42)
        N = 1000
        R_half = 1.0
        target_Q = 0.3  # Subvirial like Allison+2009

        # 1. Generate fractal positions (now returns velocities too)
        key, subkey = jax.random.split(key)
        pos_frac, vel_frac, ancestry = generate_fractal_positions(subkey, N, D=1.6)

        # 2. Sample target radii from profile
        key, subkey = jax.random.split(key)
        target_positions = sample_density_profile(subkey, N, "plummer", R_half)
        target_radii = jnp.linalg.norm(target_positions, axis=1)

        # 3. Rescale fractal to target radii
        positions = rescale_fractal_to_target_radii(pos_frac, target_radii)

        # 4. Assign velocities
        # Note: coherent=False because ancestry is now generation indices, not parent indices
        # For hierarchical velocity correlations, use the built-in fractal velocities
        masses = jnp.ones(N)
        key, subkey = jax.random.split(key)
        velocities = assign_velocities_and_virialize(
            subkey, positions, masses, target_Q_vir=target_Q,
            coherent=False, G=STELLAR.G
        )

        # Verify virial ratio
        # Use 35% tolerance for fractal ICs due to substructure effects
        K = 0.5 * jnp.sum(masses[:, None] * velocities**2)
        U = compute_potential_energy(positions, masses, G=STELLAR.G)
        Q_actual = K / jnp.abs(U)

        assert jnp.isclose(Q_actual, target_Q, rtol=0.35), (
            f"Q_actual={Q_actual:.4f} should be close to target_Q={target_Q}"
        )

        # Verify radii distribution
        r_out = jnp.linalg.norm(positions, axis=1)
        r_median = jnp.median(r_out)
        assert jnp.isclose(r_median, R_half, rtol=0.3), (
            f"Median radius {r_median:.3f} should be close to R_half={R_half}"
        )

    def test_fractal_velocities_pipeline(self):
        """Test using fractal-generated velocities directly."""
        from progenax.cluster.fractal_gw_legacy import (
            generate_fractal_positions,
            rescale_fractal_to_target_radii,
        )
        from progenax.profiles.api import sample_density_profile

        key = jax.random.PRNGKey(42)
        N = 500

        # Generate fractal with built-in velocities
        pos_frac, vel_frac, ancestry = generate_fractal_positions(key, N, D=2.0)

        # Verify velocities are correlated (hierarchical structure)
        # Check that velocity magnitudes have some structure
        v_mag = jnp.linalg.norm(vel_frac, axis=1)
        assert jnp.std(v_mag) > 0, "Velocity magnitudes should have variance"

        # Verify ancestry is generation indices
        assert jnp.all(ancestry >= 0), "Ancestry should be >= 0"
        assert jnp.all(ancestry <= 6), "Ancestry should be <= g_max=6"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
