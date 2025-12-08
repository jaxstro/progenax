"""Tests for fractal substructure generation (Goodwin-Whitworth).

Physics tests only - fractal dimension effects, radial preservation.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.substructure.fractal import (
    generate_fractal_positions,
    apply_fractal_overlay_radial,
    apply_fractal_overlay_blend,
)


class TestGenerateFractalPositions:
    """Tests for fractal position generation physics."""

    def test_d_fractal_3_uniform(self):
        """D=3.0 should give approximately uniform distribution.

        Physics: p = 2^(D-3) = 1.0 for D=3, so all children survive.
        """
        key = jax.random.PRNGKey(42)
        n_stars = 1000
        positions = generate_fractal_positions(n_stars, key, d_fractal=3.0)

        radii = jnp.linalg.norm(positions, axis=1)
        # For uniform distribution in unit sphere, mean radius ~ 0.6
        mean_radius = jnp.mean(radii)
        assert 0.4 < mean_radius < 0.8

    def test_d_fractal_1_5_clumpy(self):
        """D=1.5 should give clumpy distribution.

        Physics: p = 2^(1.5-3) = 0.35, so fewer children → clumpy structure.
        """
        key = jax.random.PRNGKey(42)
        n_stars = 1000
        positions = generate_fractal_positions(n_stars, key, d_fractal=1.5)

        radii = jnp.linalg.norm(positions, axis=1)
        # Clumpy distributions have variance in local density
        std_radius = jnp.std(radii)
        assert std_radius > 0.1


class TestFractalOverlayRadial:
    """Tests for radial-preserving fractal overlay (McLuster-style)."""

    def test_preserves_radial_distribution(self):
        """Radial distribution is preserved exactly.

        KEY PHYSICS: McLuster approach preserves r(m) while changing θ,φ.
        """
        key = jax.random.PRNGKey(42)
        n_stars = 500
        key1, key2 = jax.random.split(key)
        positions_smooth = jax.random.normal(key1, (n_stars, 3))

        positions_out = apply_fractal_overlay_radial(positions_smooth, key2, d_fractal=2.0)

        # Sorted radii should match exactly
        radii_in = jnp.sort(jnp.linalg.norm(positions_smooth, axis=1))
        radii_out = jnp.sort(jnp.linalg.norm(positions_out, axis=1))
        assert jnp.allclose(radii_in, radii_out, rtol=1e-6)

    def test_changes_angular_structure(self):
        """Angular structure is changed by fractal overlay."""
        key = jax.random.PRNGKey(42)
        n_stars = 200
        key1, key2 = jax.random.split(key)
        positions_smooth = jax.random.normal(key1, (n_stars, 3))

        positions_out = apply_fractal_overlay_radial(positions_smooth, key2, d_fractal=2.0)

        # Positions should be different (angular redistribution)
        assert not jnp.allclose(positions_smooth, positions_out)

        # But sorted radii should still match
        radii_smooth = jnp.sort(jnp.linalg.norm(positions_smooth, axis=1))
        radii_out = jnp.sort(jnp.linalg.norm(positions_out, axis=1))
        assert jnp.allclose(radii_smooth, radii_out, rtol=1e-6)


class TestFractalOverlayBlend:
    """Tests for linear blend fractal overlay."""

    def test_lambda_zero_unchanged(self):
        """λ=0 returns original positions unchanged."""
        key = jax.random.PRNGKey(42)
        positions_smooth = jax.random.normal(key, (100, 3))

        positions_out = apply_fractal_overlay_blend(
            positions_smooth, key, d_fractal=2.0, lambda_frac=0.0
        )
        assert jnp.allclose(positions_out, positions_smooth)

    def test_lambda_one_fractal(self):
        """λ=1 returns pure fractal positions (different from smooth)."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        positions_smooth = jax.random.normal(key1, (100, 3))

        positions_out = apply_fractal_overlay_blend(
            positions_smooth, key2, d_fractal=2.0, lambda_frac=1.0
        )
        assert not jnp.allclose(positions_out, positions_smooth)

    def test_intermediate_lambda(self):
        """Intermediate λ gives linear interpolation between smooth and fractal."""
        key = jax.random.PRNGKey(42)
        positions_smooth = jax.random.normal(key, (100, 3))

        pos_0 = apply_fractal_overlay_blend(
            positions_smooth, key, d_fractal=2.0, lambda_frac=0.0
        )
        pos_half = apply_fractal_overlay_blend(
            positions_smooth, key, d_fractal=2.0, lambda_frac=0.5
        )
        pos_1 = apply_fractal_overlay_blend(
            positions_smooth, key, d_fractal=2.0, lambda_frac=1.0
        )

        # Distance from λ=0 increases with λ
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
        assert jnp.all(jnp.isfinite(positions_out))
