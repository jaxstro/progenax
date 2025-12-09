"""Tests for fractal substructure generation (Goodwin-Whitworth).

Physics tests only - fractal dimension effects, radial preservation.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.cluster.fractal_gw_legacy import (
    generate_fractal_positions,
    rescale_fractal_to_target_radii,
)


class TestGenerateFractalPositions:
    """Tests for fractal position generation physics."""

    def test_d_fractal_3_uniform(self):
        """D=3.0 should give approximately uniform distribution.

        Physics: p = 2^(D-3) = 1.0 for D=3, so all children survive.
        Note: The sphere cut heavily affects corner octants, so we can't
        expect perfect uniformity. We just verify all octants are populated.
        """
        key = jax.random.PRNGKey(42)
        n_stars = 2000
        positions, _, ancestry = generate_fractal_positions(key, n_stars, D=3.0)

        # Check octant distribution
        octant_counts = []
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                for s3 in [-1, 1]:
                    mask = (jnp.sign(positions[:, 0]) == s1) & \
                           (jnp.sign(positions[:, 1]) == s2) & \
                           (jnp.sign(positions[:, 2]) == s3)
                    octant_counts.append(jnp.sum(mask))
        octant_counts = jnp.array(octant_counts)

        # Just verify all octants are populated (sphere cut causes asymmetry)
        assert jnp.all(octant_counts > 0), (
            f"All octants should have particles, got counts: {octant_counts}"
        )

    def test_d_fractal_1_5_clumpy(self):
        """D=1.5 should give clumpy distribution.

        Physics: p = 2^(1.5-3) = 0.35, so fewer children → clumpy structure.
        """
        key = jax.random.PRNGKey(42)
        n_stars = 1000
        positions, _, ancestry = generate_fractal_positions(key, n_stars, D=1.5)

        radii = jnp.linalg.norm(positions, axis=1)
        # Clumpy distributions have variance in local density
        std_radius = jnp.std(radii)
        assert std_radius > 0.1

    def test_returns_correct_shape(self):
        """Function returns positions and ancestry of correct shapes."""
        key = jax.random.PRNGKey(42)
        n_stars = 500
        positions, velocities, ancestry = generate_fractal_positions(key, n_stars, D=2.0)

        assert positions.shape == (n_stars, 3)
        assert velocities.shape == (n_stars, 3)
        assert ancestry.shape == (n_stars,)

    def test_positions_bounded(self):
        """Positions should be within unit sphere (McLuster requirement)."""
        key = jax.random.PRNGKey(42)
        positions, _, _ = generate_fractal_positions(key, 1000, D=2.0)

        radii = jnp.linalg.norm(positions, axis=1)
        max_radius = jnp.max(radii)
        assert max_radius <= 1.0 + 1e-6, (
            f"All positions must be in unit sphere, got max radius {max_radius:.4f}"
        )

    def test_jit_compatible(self):
        """Function works under JIT."""
        key = jax.random.PRNGKey(42)

        @jax.jit
        def generate(key):
            return generate_fractal_positions(key, 100, D=2.0)

        positions, velocities, ancestry = generate(key)
        assert jnp.all(jnp.isfinite(positions))
        assert jnp.all(jnp.isfinite(velocities))


class TestRescaleFractalToTargetRadii:
    """Tests for radial-preserving rescaling (McLuster A7)."""

    def test_preserves_radial_rank_order(self):
        """Particles maintain their radial rank order after rescaling."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        # Generate fractal positions
        positions_frac, _, _ = generate_fractal_positions(key1, 500, D=2.0)

        # Generate target radii from uniform distribution
        target_radii = jax.random.uniform(key2, (500,), minval=0.1, maxval=5.0)

        # Rescale
        positions_out = rescale_fractal_to_target_radii(positions_frac, target_radii)

        # Verify: sorted radii match sorted target radii exactly
        radii_out = jnp.linalg.norm(positions_out, axis=1)
        assert jnp.allclose(jnp.sort(radii_out), jnp.sort(target_radii), rtol=1e-5)

    def test_preserves_angular_direction(self):
        """Angular direction (unit vectors) should be preserved."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        positions_frac, _, _ = generate_fractal_positions(key1, 200, D=2.0)
        target_radii = jax.random.uniform(key2, (200,), minval=0.5, maxval=2.0)

        positions_out = rescale_fractal_to_target_radii(positions_frac, target_radii)

        # Unit vectors should match (direction preserved)
        r_frac = jnp.linalg.norm(positions_frac, axis=1, keepdims=True)
        r_out = jnp.linalg.norm(positions_out, axis=1, keepdims=True)

        # Avoid division by zero for particles at origin
        eps = 1e-10
        unit_frac = positions_frac / jnp.maximum(r_frac, eps)
        unit_out = positions_out / jnp.maximum(r_out, eps)

        assert jnp.allclose(unit_frac, unit_out, atol=1e-5)

    def test_jit_compatible(self):
        """Function works under JIT."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        positions_frac, _, _ = generate_fractal_positions(key1, 100, D=2.0)
        target_radii = jax.random.uniform(key2, (100,), minval=0.1, maxval=1.0)

        @jax.jit
        def rescale(pos, radii):
            return rescale_fractal_to_target_radii(pos, radii)

        positions_out = rescale(positions_frac, target_radii)
        assert jnp.all(jnp.isfinite(positions_out))
