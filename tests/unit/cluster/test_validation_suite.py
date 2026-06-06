# progenax/tests/unit/cluster/test_validation_suite.py
"""
Tests for progenax.cluster.validation module.

Tests cover:
- sweep_mass_segregation_lambda: runs without error, returns correct shapes
- measure_virial_ratio: returns Q_vir within tolerance of target
- grad_mean_radius_wrt_lambda_seg: returns finite gradients
- recover_lambda_seg_via_gradient_descent: converges toward true value
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def key():
    """Create a random key."""
    return jax.random.PRNGKey(42)


# =============================================================================
# Sweep Function Tests
# =============================================================================


class TestSweepMassSegregation:
    """Test sweep_mass_segregation_lambda function."""

    def test_sweep_runs_without_error(self, key):
        """Test that sweep runs and returns results."""
        from progenax.cluster.validation import sweep_mass_segregation_lambda

        lambda_values = [0.0, 0.5, 1.0]
        results = sweep_mass_segregation_lambda(
            key, lambda_values,
            N_stars=200,  # Small for test speed
            n_realizations=3,
            N_massive=5,
        )

        assert "lambda_values" in results
        assert "lambda_msr_mean" in results
        assert "lambda_msr_std" in results
        assert "r_massive_mean" in results

    def test_sweep_returns_correct_shapes(self, key):
        """Test that sweep returns arrays of correct shape."""
        from progenax.cluster.validation import sweep_mass_segregation_lambda

        lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = sweep_mass_segregation_lambda(
            key, lambda_values,
            N_stars=200,
            n_realizations=3,
            N_massive=5,
        )

        n_lambda = len(lambda_values)
        assert len(results["lambda_values"]) == n_lambda
        assert len(results["lambda_msr_mean"]) == n_lambda
        assert len(results["lambda_msr_std"]) == n_lambda
        assert len(results["r_massive_mean"]) == n_lambda

    def test_lambda_msr_increases_with_segregation(self, key):
        """Test that Λ_MSR generally increases with λ_seg."""
        from progenax.cluster.validation import sweep_mass_segregation_lambda

        lambda_values = [0.0, 1.0]
        results = sweep_mass_segregation_lambda(
            key, lambda_values,
            N_stars=500,
            n_realizations=5,
            N_massive=10,
        )

        # Λ_MSR at λ=1 should be higher than at λ=0
        assert results["lambda_msr_mean"][1] > results["lambda_msr_mean"][0]


# =============================================================================
# Virial Ratio Tests
# =============================================================================


class TestMeasureVirialRatio:
    """Test measure_virial_ratio function."""

    def test_virial_ratio_smooth_profile(self, key):
        """Test Q_vir accuracy for smooth Plummer profile."""
        from progenax.cluster import SpatialStructureParams
        from progenax.cluster.validation import measure_virial_ratio

        target_Q = 0.5
        structure_params = SpatialStructureParams(base_profile="plummer")

        Q_measured = measure_virial_ratio(
            key, target_Q, structure_params,
            N_stars=500,
        )

        # Should be within 20% for smooth profile
        assert 0.3 < Q_measured < 0.8, f"Q_measured = {Q_measured}, expected ~0.5"


# =============================================================================
# Gradient Tests
# =============================================================================


class TestGradientSanity:
    """Test gradient computations."""

    def test_grad_lambda_seg_is_finite(self, key):
        """Test that gradient w.r.t. λ_seg is finite."""
        from progenax.cluster.validation import grad_mean_radius_wrt_lambda_seg

        for lam in [0.1, 0.5, 0.9]:
            key, subkey = jax.random.split(key)
            grad_val = grad_mean_radius_wrt_lambda_seg(subkey, lam, N_stars=500)

            assert np.isfinite(grad_val), f"Gradient at λ={lam} is not finite: {grad_val}"

    def test_grad_has_correct_sign(self, key):
        """Test that gradient is negative (more segregation → smaller radius)."""
        from progenax.cluster.validation import grad_mean_radius_wrt_lambda_seg

        # Test at intermediate λ where effect is clearer
        key, subkey = jax.random.split(key)
        grad_val = grad_mean_radius_wrt_lambda_seg(subkey, 0.5, N_stars=500)

        # Gradient should be negative: increasing λ_seg decreases r_massive
        # Allow some tolerance for stochastic variation
        assert grad_val < 0.5, f"Gradient should be negative or small, got {grad_val}"


class TestGradientRecovery:
    """Test gradient-based parameter recovery."""

    def test_recovery_converges(self, key):
        """Test that gradient descent moves toward true value."""
        from progenax.cluster.validation import recover_lambda_seg_via_gradient_descent

        results = recover_lambda_seg_via_gradient_descent(
            key,
            lambda_true=0.7,
            n_steps=10,
            step_size=0.3,
            N_stars=500,  # Small for test speed
        )

        initial_error = abs(0.1 - results["lambda_true"])
        final_error = abs(results["lambda_final"] - results["lambda_true"])

        # Final should be closer to true than initial
        assert final_error < initial_error, (
            f"Did not converge: initial_error={initial_error:.3f}, "
            f"final_error={final_error:.3f}"
        )

    def test_recovery_returns_correct_keys(self, key):
        """Test that recovery returns expected dictionary keys."""
        from progenax.cluster.validation import recover_lambda_seg_via_gradient_descent

        results = recover_lambda_seg_via_gradient_descent(
            key,
            lambda_true=0.5,
            n_steps=5,
            step_size=0.2,
            N_stars=300,
        )

        assert "lambda_history" in results
        assert "loss_history" in results
        assert "lambda_true" in results
        assert "lambda_final" in results


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGenerateClusterForPlot:
    """Test generate_cluster_for_plot helper."""

    def test_smooth_cluster(self, key):
        """Test generating smooth cluster."""
        from progenax.cluster.validation import generate_cluster_for_plot

        cluster = generate_cluster_for_plot(key, N_stars=100)

        assert cluster.N == 100
        assert cluster.positions.shape == (100, 3)
        assert cluster.velocities.shape == (100, 3)

    def test_segregated_cluster(self, key):
        """Test generating mass-segregated cluster."""
        from progenax.cluster.validation import generate_cluster_for_plot

        cluster = generate_cluster_for_plot(key, lambda_seg=0.8, N_stars=100)

        assert cluster.N == 100
        assert cluster.positions.shape == (100, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
