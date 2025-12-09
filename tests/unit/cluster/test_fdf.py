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


class TestInitFractalField:
    """Tests for init_fractal_field function."""

    @pytest.fixture
    def key(self):
        return random.PRNGKey(42)

    def test_output_shapes(self, key):
        """init_fractal_field produces correct shapes."""
        from progenax.cluster.fdf import init_fractal_field

        n_modes = 64
        R_half = 1.0
        field = init_fractal_field(key, n_modes, R_half)

        assert field.k_vecs.shape == (n_modes, 3)
        assert field.phases.shape == (n_modes,)
        assert field.base_vecs.shape == (n_modes, 3)

    def test_k_directions_are_unit_vectors(self, key):
        """Wavevector directions are normalized."""
        from progenax.cluster.fdf import init_fractal_field

        field = init_fractal_field(key, n_modes=64, R_half=1.0)

        k_mags = jnp.linalg.norm(field.k_vecs, axis=1)
        # Extract directions by dividing by magnitudes
        k_dirs = field.k_vecs / k_mags[:, None]
        dir_norms = jnp.linalg.norm(k_dirs, axis=1)

        assert jnp.allclose(dir_norms, 1.0, atol=1e-6)

    def test_base_vecs_are_unit_vectors(self, key):
        """Polarization vectors are normalized."""
        from progenax.cluster.fdf import init_fractal_field

        field = init_fractal_field(key, n_modes=64, R_half=1.0)

        base_norms = jnp.linalg.norm(field.base_vecs, axis=1)
        assert jnp.allclose(base_norms, 1.0, atol=1e-6)

    def test_k_magnitudes_are_log_spaced(self, key):
        """Wavenumber magnitudes are log-spaced in [k_min, k_max]."""
        from progenax.cluster.fdf import init_fractal_field

        R_half = 2.0
        k_min_factor = 0.5
        k_max_factor = 20.0

        field = init_fractal_field(
            key, n_modes=64, R_half=R_half,
            k_min_factor=k_min_factor, k_max_factor=k_max_factor
        )

        k_mags = jnp.linalg.norm(field.k_vecs, axis=1)

        # Check bounds
        k_min_expected = k_min_factor / R_half
        k_max_expected = k_max_factor / R_half
        assert k_mags[0] >= k_min_expected * 0.99
        assert k_mags[-1] <= k_max_expected * 1.01

        # Check monotonicity (log-spaced means strictly increasing)
        assert jnp.all(jnp.diff(k_mags) > 0)

    def test_phases_in_valid_range(self, key):
        """Phases are in [0, 2*pi]."""
        from progenax.cluster.fdf import init_fractal_field

        field = init_fractal_field(key, n_modes=64, R_half=1.0)

        assert jnp.all(field.phases >= 0)
        assert jnp.all(field.phases <= 2 * jnp.pi)

    def test_different_keys_produce_different_fields(self):
        """Different random keys produce different fields."""
        from progenax.cluster.fdf import init_fractal_field

        key1 = random.PRNGKey(42)
        key2 = random.PRNGKey(123)

        field1 = init_fractal_field(key1, n_modes=32, R_half=1.0)
        field2 = init_fractal_field(key2, n_modes=32, R_half=1.0)

        # Phases should differ
        assert not jnp.allclose(field1.phases, field2.phases)

    def test_jit_compatible(self, key):
        """init_fractal_field can be JIT compiled."""
        from progenax.cluster.fdf import init_fractal_field

        @jax.jit
        def make_field(key):
            return init_fractal_field(key, n_modes=32, R_half=1.0)

        field = make_field(key)
        assert field.k_vecs.shape == (32, 3)
