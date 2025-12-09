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


class TestComputeAmplitudes:
    """Tests for compute_amplitudes function."""

    @pytest.fixture
    def key(self):
        return random.PRNGKey(42)

    @pytest.fixture
    def field(self, key):
        from progenax.cluster.fdf import init_fractal_field
        return init_fractal_field(key, n_modes=64, R_half=1.0)

    def test_output_shape(self, field):
        """compute_amplitudes returns correct shape."""
        from progenax.cluster.fdf import compute_amplitudes

        a_vecs = compute_amplitudes(field, chi=2.0, sigma_u=0.3)
        assert a_vecs.shape == (64, 3)

    def test_amplitude_normalization(self, field):
        """Sum of squared amplitudes equals sigma_u^2."""
        from progenax.cluster.fdf import compute_amplitudes

        sigma_u = 0.4
        a_vecs = compute_amplitudes(field, chi=2.0, sigma_u=sigma_u)

        # ||a_n||^2 summed should equal sigma_u^2
        # Note: a_vecs = amps[:, None] * base_vecs, where base_vecs are unit
        # So ||a_n||^2 = amps[n]^2
        amps_squared = jnp.sum(a_vecs ** 2, axis=1)  # ||a_n||^2 per mode
        total_amp_sq = jnp.sum(amps_squared)

        assert jnp.isclose(total_amp_sq, sigma_u ** 2, rtol=1e-5)

    def test_lower_chi_gives_more_small_scale_power(self, field):
        """Lower chi (more clumpy) gives relatively more power to small scales."""
        from progenax.cluster.fdf import compute_amplitudes

        # chi=1.5 (clumpy) vs chi=3.0 (smooth)
        a_vecs_clumpy = compute_amplitudes(field, chi=1.5, sigma_u=0.3)
        a_vecs_smooth = compute_amplitudes(field, chi=3.0, sigma_u=0.3)

        # Get amplitude magnitudes
        amps_clumpy = jnp.linalg.norm(a_vecs_clumpy, axis=1)
        amps_smooth = jnp.linalg.norm(a_vecs_smooth, axis=1)

        # Ratio of small-scale to large-scale power
        # Last 10 modes (small scale) vs first 10 modes (large scale)
        ratio_clumpy = jnp.sum(amps_clumpy[-10:]) / jnp.sum(amps_clumpy[:10])
        ratio_smooth = jnp.sum(amps_smooth[-10:]) / jnp.sum(amps_smooth[:10])

        # Clumpy should have higher small/large ratio
        assert ratio_clumpy > ratio_smooth

    def test_differentiable_in_chi(self, field):
        """Gradients flow through chi."""
        from progenax.cluster.fdf import compute_amplitudes

        def loss(chi):
            a_vecs = compute_amplitudes(field, chi=chi, sigma_u=0.3)
            return jnp.sum(a_vecs ** 2)

        grad_chi = jax.grad(loss)(2.0)
        assert jnp.isfinite(grad_chi)
        assert grad_chi != 0.0

    def test_differentiable_in_sigma_u(self, field):
        """Gradients flow through sigma_u."""
        from progenax.cluster.fdf import compute_amplitudes

        def loss(sigma_u):
            a_vecs = compute_amplitudes(field, chi=2.0, sigma_u=sigma_u)
            return jnp.sum(a_vecs ** 2)

        grad_sigma = jax.grad(loss)(0.3)
        assert jnp.isfinite(grad_sigma)
        # d/d(sigma_u) of sigma_u^2 = 2*sigma_u = 0.6
        assert jnp.isclose(grad_sigma, 0.6, rtol=1e-4)


class TestEvaluateDisplacement:
    """Tests for evaluate_displacement function."""

    @pytest.fixture
    def key(self):
        return random.PRNGKey(42)

    @pytest.fixture
    def field(self, key):
        from progenax.cluster.fdf import init_fractal_field
        return init_fractal_field(key, n_modes=32, R_half=1.0)

    @pytest.fixture
    def a_vecs(self, field):
        from progenax.cluster.fdf import compute_amplitudes
        return compute_amplitudes(field, chi=2.0, sigma_u=0.3)

    def test_output_shape(self, field, a_vecs):
        """evaluate_displacement returns correct shape."""
        from progenax.cluster.fdf import evaluate_displacement

        N = 100
        positions = jnp.ones((N, 3))
        displacements = evaluate_displacement(positions, field, a_vecs)

        assert displacements.shape == (N, 3)

    def test_zero_amplitudes_give_zero_displacement(self, field):
        """If amplitudes are zero, displacement is zero."""
        from progenax.cluster.fdf import evaluate_displacement

        positions = jnp.ones((50, 3))
        a_vecs_zero = jnp.zeros((32, 3))

        displacements = evaluate_displacement(positions, field, a_vecs_zero)

        assert jnp.allclose(displacements, 0.0)

    def test_different_positions_give_different_displacements(self, field, a_vecs):
        """Different positions produce different displacements."""
        from progenax.cluster.fdf import evaluate_displacement

        pos1 = jnp.array([[0.0, 0.0, 0.0]])
        pos2 = jnp.array([[1.0, 1.0, 1.0]])

        disp1 = evaluate_displacement(pos1, field, a_vecs)
        disp2 = evaluate_displacement(pos2, field, a_vecs)

        assert not jnp.allclose(disp1, disp2)

    def test_jit_compatible(self, field, a_vecs):
        """evaluate_displacement can be JIT compiled."""
        from progenax.cluster.fdf import evaluate_displacement

        @jax.jit
        def compute_disp(positions):
            return evaluate_displacement(positions, field, a_vecs)

        positions = jnp.ones((20, 3))
        displacements = compute_disp(positions)

        assert displacements.shape == (20, 3)

    def test_differentiable_in_positions(self, field, a_vecs):
        """Gradients flow through positions."""
        from progenax.cluster.fdf import evaluate_displacement

        def loss(positions):
            disp = evaluate_displacement(positions, field, a_vecs)
            return jnp.sum(disp ** 2)

        positions = jnp.ones((10, 3))
        grad_pos = jax.grad(loss)(positions)

        assert grad_pos.shape == (10, 3)
        assert jnp.all(jnp.isfinite(grad_pos))

    def test_differentiable_in_a_vecs(self, field):
        """Gradients flow through amplitude vectors."""
        from progenax.cluster.fdf import evaluate_displacement

        positions = jnp.ones((10, 3))
        a_vecs = jnp.ones((32, 3)) * 0.1

        def loss(a_vecs):
            disp = evaluate_displacement(positions, field, a_vecs)
            return jnp.sum(disp ** 2)

        grad_a = jax.grad(loss)(a_vecs)

        assert grad_a.shape == (32, 3)
        assert jnp.all(jnp.isfinite(grad_a))


class TestApplyDisplacement:
    """Tests for apply_displacement function."""

    @pytest.fixture
    def key(self):
        return random.PRNGKey(42)

    def test_full_mode_basic(self):
        """'full' mode adds displacement directly."""
        from progenax.cluster.fdf import apply_displacement

        positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        displacements = jnp.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        target_radii = jnp.array([1.0, 1.0])

        result = apply_displacement(
            positions, displacements, lambda_frac=1.0,
            target_radii=target_radii, mode="full"
        )

        expected = positions + displacements
        assert jnp.allclose(result, expected)

    def test_full_mode_with_lambda_frac(self):
        """'full' mode respects lambda_frac scaling."""
        from progenax.cluster.fdf import apply_displacement

        positions = jnp.array([[1.0, 0.0, 0.0]])
        displacements = jnp.array([[1.0, 0.0, 0.0]])
        target_radii = jnp.array([1.0])

        result = apply_displacement(
            positions, displacements, lambda_frac=0.5,
            target_radii=target_radii, mode="full"
        )

        expected = jnp.array([[1.5, 0.0, 0.0]])
        assert jnp.allclose(result, expected)

    def test_remap_mode_preserves_radial_cdf(self, key):
        """'remap' mode exactly preserves sorted radii."""
        from progenax.cluster.fdf import apply_displacement

        N = 100
        # Base positions with known radii
        key, subkey = random.split(key)
        target_radii = random.uniform(subkey, (N,), minval=0.1, maxval=2.0)

        # Random positions
        key, subkey = random.split(key)
        positions = random.normal(subkey, (N, 3))

        # Large displacements that would change radii
        key, subkey = random.split(key)
        displacements = random.normal(subkey, (N, 3)) * 0.5

        result = apply_displacement(
            positions, displacements, lambda_frac=1.0,
            target_radii=target_radii, mode="remap"
        )

        result_radii = jnp.linalg.norm(result, axis=1)

        # Sorted radii must match exactly
        assert jnp.allclose(
            jnp.sort(result_radii),
            jnp.sort(target_radii),
            rtol=1e-5
        )

    def test_tangential_mode_preserves_radius_per_star(self):
        """'tangential' mode preserves each star's original radius."""
        from progenax.cluster.fdf import apply_displacement

        # Positions at various radii
        positions = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])
        original_radii = jnp.linalg.norm(positions, axis=1)

        # Large displacements
        displacements = jnp.array([
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ])
        target_radii = original_radii

        result = apply_displacement(
            positions, displacements, lambda_frac=1.0,
            target_radii=target_radii, mode="tangential"
        )

        result_radii = jnp.linalg.norm(result, axis=1)

        # Each star's radius should be preserved
        assert jnp.allclose(result_radii, original_radii, rtol=1e-5)

    def test_lambda_frac_zero_returns_original(self):
        """lambda_frac=0 returns original positions."""
        from progenax.cluster.fdf import apply_displacement

        positions = jnp.array([[1.0, 2.0, 3.0]])
        displacements = jnp.array([[10.0, 10.0, 10.0]])
        target_radii = jnp.linalg.norm(positions, axis=1)

        for mode in ["full", "remap", "tangential"]:
            result = apply_displacement(
                positions, displacements, lambda_frac=0.0,
                target_radii=target_radii, mode=mode
            )
            assert jnp.allclose(result, positions, rtol=1e-5), f"Failed for mode={mode}"

    def test_differentiable_in_lambda_frac(self, key):
        """Gradients flow through lambda_frac."""
        from progenax.cluster.fdf import apply_displacement

        positions = random.normal(key, (20, 3))
        displacements = random.normal(random.split(key)[0], (20, 3)) * 0.1
        target_radii = jnp.linalg.norm(positions, axis=1)

        def loss(lambda_frac):
            result = apply_displacement(
                positions, displacements, lambda_frac=lambda_frac,
                target_radii=target_radii, mode="remap"
            )
            return jnp.sum(result ** 2)

        grad_lambda = jax.grad(loss)(0.5)
        assert jnp.isfinite(grad_lambda)
        assert grad_lambda != 0.0

    def test_jit_compatible(self, key):
        """apply_displacement can be JIT compiled."""
        from progenax.cluster.fdf import apply_displacement

        @jax.jit
        def apply_disp(positions, displacements, lambda_frac):
            target_radii = jnp.linalg.norm(positions, axis=1)
            return apply_displacement(
                positions, displacements, lambda_frac,
                target_radii, mode="remap"
            )

        positions = random.normal(key, (30, 3))
        displacements = random.normal(random.split(key)[0], (30, 3))

        result = apply_disp(positions, displacements, 0.5)
        assert result.shape == (30, 3)
