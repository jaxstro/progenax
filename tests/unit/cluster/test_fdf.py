"""Tests for Fractal Displacement Field (FDF) implementation."""

import pytest
import jax
import jax.numpy as jnp
from jax import random


class TestFractalField:
    """Tests for FractalField dataclass."""

    @pytest.fixture
    def key(self):
        return random.PRNGKey(42)

    def test_fractal_field_construction(self):
        """FractalField can be constructed with correct shapes."""
        from progenax.cluster.fdf import FractalField

        M = 64  # number of modes
        k_vecs = jnp.ones((M, 3))
        phases = jnp.zeros((M,))
        base_vecs = jnp.ones((M, 3))

        field = FractalField(k_vecs=k_vecs, phases=phases, base_vecs=base_vecs)

        assert field.k_vecs.shape == (M, 3)
        assert field.phases.shape == (M,)
        assert field.base_vecs.shape == (M, 3)

    def test_fractal_field_is_pytree(self):
        """FractalField is a valid JAX pytree."""
        from progenax.cluster.fdf import FractalField

        M = 32
        field = FractalField(
            k_vecs=jnp.ones((M, 3)),
            phases=jnp.zeros((M,)),
            base_vecs=jnp.ones((M, 3)),
        )

        # Should be flattenable
        leaves, treedef = jax.tree_util.tree_flatten(field)
        assert len(leaves) == 3

        # Should be reconstructable
        field2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(field.k_vecs, field2.k_vecs)


class TestFractalDisplacementLayer:
    """Tests for FractalDisplacementLayer parameter bundle."""

    def test_default_construction(self):
        """FractalDisplacementLayer has sensible defaults."""
        from progenax.cluster.fdf import FractalDisplacementLayer

        layer = FractalDisplacementLayer()

        assert layer.chi == 2.0
        assert layer.lambda_frac == 1.0
        assert layer.sigma_u == 0.3
        assert layer.n_modes == 64
        assert layer.k_min_factor == 0.5
        assert layer.k_max_factor == 20.0
        assert layer.radial_mode == "remap"
        assert layer.virial_ratio == 0.5
        assert layer.coherent_velocities is True
        assert layer.lambda_vel == 0.3

    def test_custom_construction(self):
        """FractalDisplacementLayer accepts custom parameters."""
        from progenax.cluster.fdf import FractalDisplacementLayer

        layer = FractalDisplacementLayer(
            chi=1.6,
            lambda_frac=0.5,
            sigma_u=0.4,
            radial_mode="full",
            virial_ratio=0.3,
        )

        assert layer.chi == 1.6
        assert layer.lambda_frac == 0.5
        assert layer.sigma_u == 0.4
        assert layer.radial_mode == "full"
        assert layer.virial_ratio == 0.3

    def test_layer_is_pytree(self):
        """FractalDisplacementLayer is a valid JAX pytree."""
        from progenax.cluster.fdf import FractalDisplacementLayer

        layer = FractalDisplacementLayer(chi=2.0, lambda_frac=0.8)

        leaves, treedef = jax.tree_util.tree_flatten(layer)
        layer2 = jax.tree_util.tree_unflatten(treedef, leaves)

        assert layer.chi == layer2.chi
        assert layer.lambda_frac == layer2.lambda_frac
