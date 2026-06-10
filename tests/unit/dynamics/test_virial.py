"""Tests for virial ratio + energy utilities (progenax.dynamics.virial).

Mutation-sensitive by construction: every test calls the module functions and
compares against an INDEPENDENTLY derived analytic value (not an inline
re-implementation of the same arithmetic).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.dynamics.virial import (
    compute_kinetic_energy,
    compute_potential_energy,
    compute_virial_ratio,
    rescale_velocities_to_virial,
)


def _central_fd(f, t, h=1e-5):
    return (f(t + h) - f(t - h)) / (2 * h)


class TestComputeKineticEnergy:
    """T = 0.5 * sum(m_i * |v_i|^2)."""

    def test_matches_half_sum_m_v2(self):
        v = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        m = jnp.array([2.0, 3.0])
        # T = 0.5 * (2*1 + 3*4) = 7
        assert jnp.isclose(compute_kinetic_energy(v, m), 7.0, atol=1e-12)

    def test_zero_velocity_is_zero(self):
        assert jnp.isclose(compute_kinetic_energy(jnp.zeros((5, 3)), jnp.ones(5)), 0.0, atol=1e-12)

    def test_grad_fd_match(self):
        v0 = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        m = jnp.ones(2)
        f = lambda s: compute_kinetic_energy(s * v0, m)
        g = jax.grad(f)(1.0)
        assert jnp.isfinite(g) and jnp.isclose(g, _central_fd(f, 1.0), rtol=1e-6)


class TestComputePotentialEnergy:
    """V = -G * sum_{i<j} m_i m_j / r_ij (G factor present, grad-safe at soft=0)."""

    def test_two_body_analytic(self):
        # two unit masses separated by r=2 -> V = -G * m1 m2 / r = -G/2
        pos = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        G = 3.7
        assert jnp.isclose(compute_potential_energy(pos, jnp.ones(2), G, 0.0), -G / 2.0, atol=1e-12)

    def test_linear_in_G(self):
        pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        m = jnp.ones(2)
        v1 = compute_potential_energy(pos, m, 1.0, 0.0)
        v2 = compute_potential_energy(pos, m, 2.0, 0.0)
        assert jnp.isclose(v2, 2.0 * v1, atol=1e-12)

    def test_grad_fd_match_softened(self):
        pos0 = jax.random.normal(jax.random.PRNGKey(5), (10, 3))
        m, G = jnp.ones(10), 0.00450
        f = lambda s: compute_potential_energy(s * pos0, m, G, 0.05)
        g = jax.grad(f)(1.0)
        assert jnp.isfinite(g) and jnp.isclose(g, _central_fd(f, 1.0), rtol=1e-5)


class TestComputeVirialRatio:
    """Q = T/|V| computed BY THE MODULE (regression: F5 tautology removed)."""

    def _system(self):
        pos = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        vel = jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        return pos, vel, jnp.ones(2), 1.0

    def test_matches_T_over_absV(self):
        pos, vel, m, G = self._system()
        Q = compute_virial_ratio(pos, vel, m, G=G)
        T = compute_kinetic_energy(vel, m)
        V = compute_potential_energy(pos, m, G=G)
        assert jnp.isclose(Q, T / jnp.abs(V), atol=1e-12)

    def test_equilibrium_satisfies_2T_plus_V(self):
        pos, vel, m, G = self._system()
        vel_eq = rescale_velocities_to_virial(pos, vel, m, G=G, target_Q=0.5)
        Q = compute_virial_ratio(pos, vel_eq, m, G=G)
        assert jnp.isclose(Q, 0.5, atol=1e-6)
        T = compute_kinetic_energy(vel_eq, m)
        V = compute_potential_energy(pos, m, G=G)
        assert jnp.isclose(2 * T + V, 0.0, atol=1e-6 * jnp.abs(V))  # virial theorem


class TestRescaleVelocitiesToVirial:
    """Velocity rescaling to a target Q (calls the real function)."""

    @pytest.fixture
    def simple_system(self):
        positions = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        velocities = jnp.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        masses = jnp.array([1.0, 1.0])
        G = 1.0
        return positions, velocities, masses, G

    def test_default_target_is_half(self, simple_system):
        pos, vel, masses, G = simple_system
        vel_scaled = rescale_velocities_to_virial(pos, vel, masses, G=G)
        Q = compute_virial_ratio(pos, vel_scaled, masses, G=G)
        assert jnp.isclose(Q, 0.5, atol=0.02)

    def test_subvirial_q_less_than_half(self, simple_system):
        pos, vel, masses, G = simple_system
        vel_scaled = rescale_velocities_to_virial(pos, vel, masses, G=G, target_Q=0.3)
        Q = compute_virial_ratio(pos, vel_scaled, masses, G=G)
        assert jnp.isclose(Q, 0.3, atol=0.02)
        assert Q < 0.5

    def test_supervirial_q_greater_than_half(self, simple_system):
        pos, vel, masses, G = simple_system
        vel_scaled = rescale_velocities_to_virial(pos, vel, masses, G=G, target_Q=0.7)
        Q = compute_virial_ratio(pos, vel_scaled, masses, G=G)
        assert jnp.isclose(Q, 0.7, atol=0.02)
        assert Q > 0.5


class TestBlockedPotentialEnergy:
    """The blocked row-scan kernel must reproduce the dense pair sum exactly
    (same pair set, same per-pair arithmetic) at every N/block alignment."""

    def _dense_reference(self, positions, masses, G, softening=0.0):
        """Independent O(N^2) oracle: explicit double loop over i<j in numpy."""
        pos = np.asarray(positions)
        m = np.asarray(masses)
        V = 0.0
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                r = np.sqrt(np.sum((pos[i] - pos[j]) ** 2) + softening**2)
                V -= m[i] * m[j] / r
        return float(G * V)  # -G * sum_{i<j} m_i m_j / r_ij form

    def test_matches_dense_oracle_block_not_dividing_n(self):
        """N=37 with block_size=16 (padding path) vs the explicit pair sum."""
        key = jax.random.PRNGKey(3)
        pos = jax.random.normal(key, (37, 3))
        m = jnp.abs(jax.random.normal(jax.random.PRNGKey(4), (37,))) + 0.1
        V = compute_potential_energy(pos, m, G=2.5, block_size=16)
        V_ref = self._dense_reference(pos, m, G=2.5)
        np.testing.assert_allclose(float(V), V_ref, rtol=1e-13)

    def test_block_size_invariance(self):
        """Result independent of block size to float64 reassociation level."""
        key = jax.random.PRNGKey(5)
        pos = jax.random.normal(key, (100, 3))
        m = jnp.ones(100)
        vals = [float(compute_potential_energy(pos, m, G=1.0, block_size=b))
                for b in (7, 32, 100, 1024)]
        np.testing.assert_allclose(vals, vals[0], rtol=1e-13)

    def test_softened_matches_dense_oracle(self):
        key = jax.random.PRNGKey(6)
        pos = jax.random.normal(key, (25, 3))
        m = jnp.ones(25)
        V = compute_potential_energy(pos, m, G=1.0, softening=0.05, block_size=8)
        V_ref = self._dense_reference(pos, m, G=1.0, softening=0.05)
        np.testing.assert_allclose(float(V), V_ref, rtol=1e-13)

    def test_grad_finite_at_zero_softening_with_padding(self):
        """The where-before-sqrt guard must survive blocking + padding
        (a padded zero-row coincident with a real star at the origin is the
        gradient trap)."""
        pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        m = jnp.ones(3)
        g = jax.grad(lambda p: compute_potential_energy(p, m, G=1.0,
                                                        block_size=2))(pos)
        assert bool(jnp.all(jnp.isfinite(g)))

    def test_memory_bounded_smoke_n20000(self):
        """N=20k ran at 32.8 GB peak dense (measured 2026-06-10); blocked must
        run comfortably — correctness assert only, the RSS evidence lives in
        scripts/profile_cluster_memory.py."""
        key = jax.random.PRNGKey(7)
        pos = jax.random.normal(key, (20000, 3)) * 5.0
        m = jnp.ones(20000)
        V = compute_potential_energy(pos, m, G=1.0)
        assert bool(jnp.isfinite(V)) and float(V) < 0
