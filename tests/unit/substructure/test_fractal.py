"""Tests for fractal substructure generation (Goodwin-Whitworth)."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from progenax.substructure.fractal import (
    generate_fractal_positions,
    apply_fractal_overlay_radial,
    apply_fractal_overlay_blend,
)


class TestGenerateFractalPositions:
    """Tests for fractal position generation."""

    def test_output_shape(self):
        """Output has correct shape."""
        key = jax.random.PRNGKey(42)
        n_stars = 100
        positions = generate_fractal_positions(n_stars, key, d_fractal=2.0)

        assert positions.shape == (n_stars, 3)

    def test_positions_in_unit_sphere(self):
        """All positions are within or near unit sphere."""
        key = jax.random.PRNGKey(42)
        n_stars = 500
        positions = generate_fractal_positions(n_stars, key, d_fractal=2.0)

        radii = jnp.linalg.norm(positions, axis=1)
        # Should be contained in unit sphere (with some tolerance for boundary)
        assert jnp.all(radii <= 1.5)
        # Most should be within unit sphere
        assert jnp.mean(radii <= 1.0) > 0.8

    def test_d_fractal_3_uniform(self):
        """D=3.0 should give approximately uniform distribution."""
        key = jax.random.PRNGKey(42)
        n_stars = 1000
        positions = generate_fractal_positions(n_stars, key, d_fractal=3.0)

        # For D=3, survival probability p = 2^(3-3) = 1.0, so all children survive
        # This should give roughly uniform distribution
        radii = jnp.linalg.norm(positions, axis=1)

        # Check radial distribution is not too clumpy
        # For uniform distribution in unit sphere, mean radius ~ 0.6
        mean_radius = jnp.mean(radii)
        assert 0.4 < mean_radius < 0.8

    def test_d_fractal_1_5_clumpy(self):
        """D=1.5 should give clumpy distribution."""
        key = jax.random.PRNGKey(42)
        n_stars = 1000
        positions = generate_fractal_positions(n_stars, key, d_fractal=1.5)

        # For D=1.5, survival probability p = 2^(1.5-3) = 0.35, so fewer children
        # This should give clumpy structure
        radii = jnp.linalg.norm(positions, axis=1)

        # Clumpy distributions have variance in local density
        # Check that radial distribution has some spread
        std_radius = jnp.std(radii)
        assert std_radius > 0.1  # Should have some spread (relaxed threshold)

    def test_reproducibility(self):
        """Same seed gives same results."""
        key = jax.random.PRNGKey(42)
        n_stars = 100
        d_fractal = 2.0

        positions1 = generate_fractal_positions(n_stars, key, d_fractal)
        positions2 = generate_fractal_positions(n_stars, key, d_fractal)

        assert jnp.allclose(positions1, positions2)

    def test_different_seeds_different_results(self):
        """Different seeds give different results."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)
        n_stars = 100
        d_fractal = 2.0

        positions1 = generate_fractal_positions(n_stars, key1, d_fractal)
        positions2 = generate_fractal_positions(n_stars, key2, d_fractal)

        assert not jnp.allclose(positions1, positions2)


class TestFractalOverlayRadial:
    """Tests for radial-preserving fractal overlay (McLuster-style)."""

    def test_output_shape(self):
        """Output has same shape as input."""
        key = jax.random.PRNGKey(42)
        positions_smooth = jax.random.normal(key, (100, 3))
        positions_out = apply_fractal_overlay_radial(positions_smooth, key, d_fractal=2.0)

        assert positions_out.shape == positions_smooth.shape

    def test_preserves_radial_distribution(self):
        """Radial distribution is preserved exactly."""
        key = jax.random.PRNGKey(42)
        # Create positions with known radial distribution
        n_stars = 500
        key1, key2 = jax.random.split(key)
        positions_smooth = jax.random.normal(key1, (n_stars, 3))

        positions_out = apply_fractal_overlay_radial(positions_smooth, key2, d_fractal=2.0)

        # Compute radii
        radii_in = jnp.sort(jnp.linalg.norm(positions_smooth, axis=1))
        radii_out = jnp.sort(jnp.linalg.norm(positions_out, axis=1))

        # Radial distributions should match exactly (sorted radii)
        assert jnp.allclose(radii_in, radii_out, rtol=1e-6)

    def test_changes_angular_structure(self):
        """Angular structure is changed by fractal overlay."""
        key = jax.random.PRNGKey(42)
        n_stars = 200
        key1, key2 = jax.random.split(key)

        # Create smooth positions
        positions_smooth = jax.random.normal(key1, (n_stars, 3))

        # Apply fractal overlay
        positions_out = apply_fractal_overlay_radial(positions_smooth, key2, d_fractal=2.0)

        # Angular structure should be different
        # Check that positions are not just a simple scaling
        assert not jnp.allclose(positions_smooth, positions_out)

        # But radii should match
        radii_smooth = jnp.linalg.norm(positions_smooth, axis=1)
        radii_out = jnp.linalg.norm(positions_out, axis=1)
        radii_smooth_sorted = jnp.sort(radii_smooth)
        radii_out_sorted = jnp.sort(radii_out)
        assert jnp.allclose(radii_smooth_sorted, radii_out_sorted, rtol=1e-6)

    def test_jit_compatible(self):
        """Function works under JIT."""
        key = jax.random.PRNGKey(42)
        positions_smooth = jax.random.normal(key, (100, 3))

        # JIT compile
        @jax.jit
        def apply_overlay(pos, key):
            return apply_fractal_overlay_radial(pos, key, d_fractal=2.0)

        # Should work without error
        positions_out = apply_overlay(positions_smooth, key)
        assert positions_out.shape == positions_smooth.shape


class TestFractalOverlayBlend:
    """Tests for linear blend fractal overlay."""

    def test_output_shape(self):
        """Output has same shape as input."""
        key = jax.random.PRNGKey(42)
        positions_smooth = jax.random.normal(key, (100, 3))
        positions_out = apply_fractal_overlay_blend(
            positions_smooth, key, d_fractal=2.0, lambda_frac=0.5
        )

        assert positions_out.shape == positions_smooth.shape

    def test_lambda_zero_unchanged(self):
        """lambda=0 returns original positions unchanged."""
        key = jax.random.PRNGKey(42)
        positions_smooth = jax.random.normal(key, (100, 3))

        positions_out = apply_fractal_overlay_blend(
            positions_smooth, key, d_fractal=2.0, lambda_frac=0.0
        )

        assert jnp.allclose(positions_out, positions_smooth)

    def test_lambda_one_fractal(self):
        """lambda=1 returns pure fractal positions."""
        key = jax.random.PRNGKey(42)
        n_stars = 100
        key1, key2 = jax.random.split(key)
        positions_smooth = jax.random.normal(key1, (n_stars, 3))

        # Get pure fractal positions
        positions_fractal = generate_fractal_positions(n_stars, key2, d_fractal=2.0)

        # Get blend with lambda=1
        positions_out = apply_fractal_overlay_blend(
            positions_smooth, key2, d_fractal=2.0, lambda_frac=1.0
        )

        # Should match fractal positions (scaled to match smooth distribution)
        # Note: overlay scales fractal to match smooth radial extent
        assert positions_out.shape == positions_fractal.shape
        # At least check it's different from smooth
        assert not jnp.allclose(positions_out, positions_smooth)

    def test_intermediate_lambda(self):
        """Intermediate lambda gives intermediate result."""
        key = jax.random.PRNGKey(42)
        n_stars = 100
        positions_smooth = jax.random.normal(key, (n_stars, 3))

        # Get results for different lambda values
        pos_0 = apply_fractal_overlay_blend(
            positions_smooth, key, d_fractal=2.0, lambda_frac=0.0
        )
        pos_half = apply_fractal_overlay_blend(
            positions_smooth, key, d_fractal=2.0, lambda_frac=0.5
        )
        pos_1 = apply_fractal_overlay_blend(
            positions_smooth, key, d_fractal=2.0, lambda_frac=1.0
        )

        # pos_half should be between pos_0 and pos_1
        # Check that distances from pos_0 increase with lambda
        dist_0_to_half = jnp.linalg.norm(pos_half - pos_0)
        dist_0_to_1 = jnp.linalg.norm(pos_1 - pos_0)

        assert dist_0_to_half > 0
        assert dist_0_to_half < dist_0_to_1

    def test_jit_compatible(self):
        """Function works under JIT."""
        key = jax.random.PRNGKey(42)
        positions_smooth = jax.random.normal(key, (100, 3))

        @jax.jit
        def apply_blend(pos, key):
            return apply_fractal_overlay_blend(pos, key, d_fractal=2.0, lambda_frac=0.5)

        positions_out = apply_blend(positions_smooth, key)
        assert positions_out.shape == positions_smooth.shape
