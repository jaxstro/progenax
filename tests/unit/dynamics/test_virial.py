"""Tests for virial ratio utilities."""
import jax.numpy as jnp
import pytest
from progenax.dynamics.virial import (
    compute_virial_ratio,
    rescale_velocities_to_virial,
)


class TestComputeVirialRatio:
    """Test Q = T / |V| computation."""

    def test_formula_is_T_over_V(self):
        """Q = T / |V|, NOT 2T / |V|."""
        T = 100.0  # Kinetic energy
        V = -200.0  # Potential energy (negative)
        # Q = T / |V| = 100 / 200 = 0.5
        Q = T / jnp.abs(V)
        assert jnp.isclose(Q, 0.5, atol=1e-10)

    def test_virial_equilibrium_is_half(self):
        """For 2T + V = 0, Q = T/|V| = 0.5."""
        # Virial theorem: 2T + V = 0  =>  T = |V|/2  =>  Q = T/|V| = 0.5
        V = -100.0
        T = jnp.abs(V) / 2.0  # T = 50 for virial equilibrium
        Q = T / jnp.abs(V)
        assert jnp.isclose(Q, 0.5, atol=1e-10)


class TestRescaleVelocitiesToVirial:
    """Test velocity rescaling to target Q."""

    @pytest.fixture
    def simple_system(self):
        """Two-body system for testing."""
        positions = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        velocities = jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        masses = jnp.array([1.0, 1.0])
        G = 1.0
        return positions, velocities, masses, G

    def test_default_target_is_half(self, simple_system):
        """Default target_Q should be 0.5 (virial equilibrium)."""
        pos, vel, masses, G = simple_system
        vel_scaled = rescale_velocities_to_virial(pos, vel, masses, G=G)
        # Verify Q ≈ 0.5 after scaling
        Q = compute_virial_ratio(pos, vel_scaled, masses, G=G)
        assert jnp.isclose(Q, 0.5, atol=0.02)

    def test_subvirial_q_less_than_half(self, simple_system):
        """Q < 0.5 is subvirial (cold, collapsing)."""
        pos, vel, masses, G = simple_system
        vel_scaled = rescale_velocities_to_virial(pos, vel, masses, G=G, target_Q=0.3)
        Q = compute_virial_ratio(pos, vel_scaled, masses, G=G)
        assert jnp.isclose(Q, 0.3, atol=0.02)
        assert Q < 0.5  # Subvirial

    def test_supervirial_q_greater_than_half(self, simple_system):
        """Q > 0.5 is supervirial (hot, expanding)."""
        pos, vel, masses, G = simple_system
        vel_scaled = rescale_velocities_to_virial(pos, vel, masses, G=G, target_Q=0.7)
        Q = compute_virial_ratio(pos, vel_scaled, masses, G=G)
        assert jnp.isclose(Q, 0.7, atol=0.02)
        assert Q > 0.5  # Supervirial
