"""Tests for JAX-native Q parameter approximation."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def generate_uniform_sphere_jax(N: int, key) -> jnp.ndarray:
    """Generate uniform sphere using JAX."""
    key_r, key_theta, key_phi = jax.random.split(key, 3)
    u = jax.random.uniform(key_r, (N,))
    r = u ** (1 / 3)
    cos_theta = jax.random.uniform(key_theta, (N,), minval=-1.0, maxval=1.0)
    sin_theta = jnp.sqrt(1 - cos_theta ** 2)
    phi = jax.random.uniform(key_phi, (N,), minval=0.0, maxval=2 * jnp.pi)
    x = r * sin_theta * jnp.cos(phi)
    y = r * sin_theta * jnp.sin(phi)
    z = r * cos_theta
    return jnp.stack([x, y, z], axis=1)


class TestImports:
    def test_imports(self):
        """Module should be importable."""
        from progenax.diagnostics.q_approx import (
            q_approx_naive,
            q_approx_fast,
            q_approx,
            DEFAULT_CALIBRATION,
        )
        assert callable(q_approx_naive)
        assert callable(q_approx_fast)
        assert callable(q_approx)


class TestQApproxNaive:
    """Tests for q_approx_naive implementation."""

    def test_returns_scalar(self):
        """q_approx_naive should return a scalar."""
        from progenax.diagnostics.q_approx import q_approx_naive
        key = jax.random.PRNGKey(42)
        positions = generate_uniform_sphere_jax(100, key)
        Q = q_approx_naive(positions)
        assert Q.shape == (), f"Expected scalar, got shape {Q.shape}"

    def test_uniform_sphere_reasonable_range(self):
        """Q for uniform sphere should be in [0.5, 1.2]."""
        from progenax.diagnostics.q_approx import q_approx_naive
        Q_values = []
        for seed in range(10):
            key = jax.random.PRNGKey(seed)
            positions = generate_uniform_sphere_jax(300, key)
            Q = q_approx_naive(positions)
            Q_values.append(float(Q))
        Q_mean = np.mean(Q_values)
        assert 0.5 < Q_mean < 1.2, f"Q_mean = {Q_mean:.3f} outside expected range"

    def test_degenerate_small_n(self):
        """Small N should return default value 0.79."""
        from progenax.diagnostics.q_approx import q_approx_naive
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        Q = q_approx_naive(positions)
        assert jnp.isclose(Q, 0.79), f"Expected 0.79 for N=2, got {Q}"

    def test_jit_compatible(self):
        """Function should work with @jax.jit."""
        from progenax.diagnostics.q_approx import q_approx_naive
        key = jax.random.PRNGKey(42)
        positions = generate_uniform_sphere_jax(100, key)
        q_jit = jax.jit(q_approx_naive)
        Q = q_jit(positions)
        assert jnp.isfinite(Q), f"JIT result should be finite, got {Q}"

    def test_vmap_compatible(self):
        """Function should work with jax.vmap."""
        from progenax.diagnostics.q_approx import q_approx_naive
        keys = jax.random.split(jax.random.PRNGKey(42), 5)
        batch_positions = jax.vmap(lambda k: generate_uniform_sphere_jax(100, k))(keys)
        Q_batch = jax.vmap(q_approx_naive)(batch_positions)
        assert Q_batch.shape == (5,), f"Expected (5,), got {Q_batch.shape}"

    def test_grad_compatible(self):
        """Function should support gradients."""
        from progenax.diagnostics.q_approx import q_approx_naive
        key = jax.random.PRNGKey(42)
        positions = generate_uniform_sphere_jax(50, key)
        grad = jax.grad(lambda p: q_approx_naive(p))(positions)
        assert grad.shape == positions.shape
        assert jnp.all(jnp.isfinite(grad)), "Gradients should be finite"


class TestQApproxFast:
    """Tests for q_approx_fast implementation."""

    def test_returns_scalar(self):
        """q_approx_fast should return a scalar."""
        from progenax.diagnostics.q_approx import q_approx_fast
        key = jax.random.PRNGKey(42)
        positions = generate_uniform_sphere_jax(500, key)
        Q = q_approx_fast(positions)
        assert Q.shape == (), f"Expected scalar, got shape {Q.shape}"

    def test_consistent_with_naive(self):
        """Fast version should give similar results to naive."""
        from progenax.diagnostics.q_approx import q_approx_naive, q_approx_fast
        key = jax.random.PRNGKey(42)
        positions = generate_uniform_sphere_jax(300, key)
        Q_naive = q_approx_naive(positions)
        Q_fast = q_approx_fast(positions)
        rel_diff = abs(float(Q_naive) - float(Q_fast)) / float(Q_naive)
        assert rel_diff < 0.30, f"Fast vs naive differ by {rel_diff*100:.1f}%"

    def test_jit_compatible(self):
        """Function should work with @jax.jit."""
        from progenax.diagnostics.q_approx import q_approx_fast
        key = jax.random.PRNGKey(42)
        positions = generate_uniform_sphere_jax(500, key)
        q_jit = jax.jit(q_approx_fast)
        Q = q_jit(positions)
        assert jnp.isfinite(Q), f"JIT result should be finite"

    def test_scales_to_large_n(self):
        """Should handle large N efficiently."""
        from progenax.diagnostics.q_approx import q_approx_fast
        key = jax.random.PRNGKey(42)
        positions = generate_uniform_sphere_jax(5000, key)
        Q = q_approx_fast(positions, nbins_per_dim=32)
        assert jnp.isfinite(Q), f"Large N should work"


class TestQApproxUnified:
    """Tests for unified q_approx interface."""

    def test_auto_dispatches_correctly(self):
        """Auto should dispatch based on N."""
        from progenax.diagnostics.q_approx import q_approx
        key = jax.random.PRNGKey(42)
        positions = generate_uniform_sphere_jax(500, key)
        Q = q_approx(positions, method="auto")
        assert jnp.isfinite(Q)

    def test_explicit_method_selection(self):
        """Explicit method selection should work."""
        from progenax.diagnostics.q_approx import q_approx
        key = jax.random.PRNGKey(42)
        positions = generate_uniform_sphere_jax(300, key)
        Q_naive = q_approx(positions, method="naive")
        Q_fast = q_approx(positions, method="fast")
        assert jnp.isfinite(Q_naive) and jnp.isfinite(Q_fast)

    def test_auto_dispatch_small_n_equals_naive(self):
        """Audit J6: auto picks the naive path for small N (Python-if dispatch)."""
        from progenax.diagnostics.q_approx import q_approx
        pos = jax.random.normal(jax.random.PRNGKey(0), (200, 3))
        assert jnp.allclose(q_approx(pos, method="auto"), q_approx(pos, method="naive"))

    def test_auto_dispatch_large_n_equals_fast(self):
        """Audit J6: auto picks the fast path for large N (Python-if dispatch)."""
        from progenax.diagnostics.q_approx import q_approx
        pos = jax.random.normal(jax.random.PRNGKey(0), (1500, 3))
        assert jnp.allclose(q_approx(pos, method="auto"), q_approx(pos, method="fast"))


class TestExports:
    """Test module exports from progenax.diagnostics."""

    def test_import_from_diagnostics(self):
        """Should be importable from progenax.diagnostics."""
        from progenax.diagnostics import q_approx, q_approx_naive, q_approx_fast
        assert callable(q_approx)


class TestCalibration:
    """Tests for calibration against scipy baseline."""

    @pytest.mark.slow
    def test_calibration_produces_valid_factors(self):
        """Calibration should produce reasonable factors."""
        from progenax.diagnostics.q_approx import calibrate_q_approx
        results = calibrate_q_approx(n_samples=50, N_stars=300, seed=42)
        assert 0.5 < results["calibration_naive"] < 2.0
        assert results["correlation_naive"] > 0.5

    def test_monotonicity_preserved(self):
        """Q_approx should preserve ordering vs Q_exact."""
        from progenax.diagnostics import compute_q_parameter
        from progenax.diagnostics.q_approx import q_approx_naive

        # Create two distributions with different substructure levels
        # More substructure = lower Q
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(100)

        # Smooth distribution
        smooth = generate_uniform_sphere_jax(300, key1)

        # Clumpy distribution (add two offset spheres)
        clump1 = generate_uniform_sphere_jax(150, key2) * 0.3 + jnp.array([0.5, 0.0, 0.0])
        clump2 = generate_uniform_sphere_jax(150, jax.random.PRNGKey(101)) * 0.3 + jnp.array([-0.5, 0.0, 0.0])
        clumpy = jnp.concatenate([clump1, clump2], axis=0)

        Q_smooth_exact = compute_q_parameter(np.asarray(smooth))
        Q_clumpy_exact = compute_q_parameter(np.asarray(clumpy))
        Q_smooth_approx = float(q_approx_naive(smooth))
        Q_clumpy_approx = float(q_approx_naive(clumpy))

        # Clumpy should have lower Q than smooth (both methods)
        # Only check if exact difference is significant (>10%)
        if abs(Q_smooth_exact - Q_clumpy_exact) / Q_smooth_exact > 0.1:
            exact_ordering = Q_clumpy_exact < Q_smooth_exact
            approx_ordering = Q_clumpy_approx < Q_smooth_approx
            assert exact_ordering == approx_ordering, "Monotonicity violated!"
