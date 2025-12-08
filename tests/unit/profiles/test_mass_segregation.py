# progenax/tests/unit/profiles/test_mass_segregation.py
"""
Unit tests for mass segregation module.

Physical tests only - no trivial shape/type tests.
"""

import jax
import jax.numpy as jnp
import pytest
from scipy.stats import spearmanr

from progenax.profiles.mass_segregation import (
    _mst_length,
    _softened_potential,
    apply_mass_segregation_baumgardt,
    generate_mass_segregated_ic_subr,
    mass_segregation_ratio_mst,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def key():
    """Standard reproducible random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def G():
    """Gravitational constant in stellar units (Msun, pc, Myr)."""
    return 0.00450  # pc^3 Msun^-1 Myr^-2


@pytest.fixture
def eps():
    """Standard softening length."""
    return 0.01  # pc


# =============================================================================
# TestSoftenedPotential - physics verification
# =============================================================================


class TestSoftenedPotential:
    """Test O(N^2) gravitational potential formula."""

    def test_two_body_potential(self, G, eps):
        """Two-body potential matches Phi = -Gm/r_softened."""
        m1, m2 = 1.0, 2.0
        r12 = 1.0
        positions = jnp.array([[0.0, 0.0, 0.0], [r12, 0.0, 0.0]])
        masses = jnp.array([m1, m2])

        phi = _softened_potential(positions, masses, G=G, eps=eps)

        r_soft = jnp.sqrt(r12**2 + eps**2)
        expected_phi_0 = -G * m2 / r_soft
        expected_phi_1 = -G * m1 / r_soft

        assert jnp.isclose(phi[0], expected_phi_0, rtol=1e-6), (
            f"Phi_0={float(phi[0]):.6e}, expected={float(expected_phi_0):.6e}"
        )
        assert jnp.isclose(phi[1], expected_phi_1, rtol=1e-6), (
            f"Phi_1={float(phi[1]):.6e}, expected={float(expected_phi_1):.6e}"
        )


# =============================================================================
# TestMSTLength - geometric correctness
# =============================================================================


class TestMSTLength:
    """Test Prim's MST algorithm."""

    def test_line_of_points(self):
        """MST of collinear points = sum of consecutive gaps."""
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ])
        length = _mst_length(positions)
        expected = 1.0 + 2.0 + 3.0
        assert jnp.isclose(length, expected, rtol=1e-6), (
            f"MST length={float(length):.4f}, expected={expected:.4f}"
        )

    def test_equilateral_triangle(self):
        """MST of equilateral triangle = 2 edges."""
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, jnp.sqrt(3) / 2, 0.0],
        ])
        length = _mst_length(positions)
        expected = 2.0
        assert jnp.isclose(length, expected, rtol=1e-4), (
            f"MST length={float(length):.4f}, expected={expected:.4f}"
        )


# =============================================================================
# TestMassSegregationRatioMST - Allison+ (2009)
# =============================================================================


class TestMassSegregationRatioMST:
    """Test Lambda_MSR diagnostic (Allison+ 2009)."""

    def test_uniform_masses_lambda_near_one(self, key):
        """Unsegregated cluster (equal masses) has Lambda_MSR ~ 1.0."""
        from progenax.profiles import PlummerProfile

        N = 300
        profile = PlummerProfile(r_h=1.0)
        masses = jnp.ones(N)

        positions = profile.sample_positions(masses, key)

        key, subkey = jax.random.split(key)
        result = mass_segregation_ratio_mst(
            positions, masses, n_massive=20, n_random=50, key=subkey
        )

        lambda_msr = float(result["lambda_msr"])
        assert 0.7 < lambda_msr < 1.3, (
            f"Lambda_MSR={lambda_msr:.3f}, expected ~1.0 for equal masses"
        )

    def test_segregated_cluster_lambda_greater_than_one(self, key, G, eps):
        """Mass-segregated cluster has Lambda_MSR > 1."""
        from progenax.profiles import PlummerProfile

        N = 300
        profile = PlummerProfile(r_h=1.0)
        masses = jax.random.uniform(key, (N,), minval=0.1, maxval=10.0)

        key, subkey = jax.random.split(key)
        positions = profile.sample_positions(masses, subkey)
        key, subkey = jax.random.split(key)
        velocities = jax.random.normal(subkey, (N, 3)) * 0.1

        key, subkey = jax.random.split(key)
        pos_seg, _ = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=1.0, key=subkey, G=G, eps=eps
        )

        key, subkey = jax.random.split(key)
        result = mass_segregation_ratio_mst(
            pos_seg, masses, n_massive=20, n_random=50, key=subkey
        )

        lambda_msr = float(result["lambda_msr"])
        assert lambda_msr > 1.3, (
            f"Lambda_MSR={lambda_msr:.3f} for s=1 segregation, expected > 1.3"
        )


# =============================================================================
# TestBaumgardtSegregation - energy-ranked orbit assignment
# =============================================================================


class TestBaumgardtSegregation:
    """Test Baumgardt/McLuster energy-ranked orbit assignment."""

    def test_s_zero_weak_correlation(self, key, G, eps):
        """s=0 gives weak mass-energy correlation."""
        from progenax.profiles import PlummerProfile

        N = 200
        profile = PlummerProfile(r_h=1.0)
        masses = jax.random.uniform(key, (N,), minval=0.1, maxval=10.0)

        key, subkey = jax.random.split(key)
        positions = profile.sample_positions(masses, subkey)
        key, subkey = jax.random.split(key)
        velocities = jax.random.normal(subkey, (N, 3)) * 0.1

        key, subkey = jax.random.split(key)
        pos_out, vel_out = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=0.0, key=subkey, G=G, eps=eps
        )

        phi = _softened_potential(pos_out, masses, G=G, eps=eps)
        E = 0.5 * jnp.sum(vel_out**2, axis=1) + phi
        rho, _ = spearmanr(masses, E)

        assert abs(rho) < 0.3, f"s=0 should give |rho| < 0.3, got rho={rho:.3f}"

    def test_s_one_strong_negative_correlation(self, key, G, eps):
        """s=1 gives strong negative mass-energy correlation."""
        from progenax.profiles import PlummerProfile

        N = 200
        profile = PlummerProfile(r_h=1.0)
        masses = jax.random.uniform(key, (N,), minval=0.1, maxval=10.0)

        key, subkey = jax.random.split(key)
        positions = profile.sample_positions(masses, subkey)
        key, subkey = jax.random.split(key)
        velocities = jax.random.normal(subkey, (N, 3)) * 0.1

        key, subkey = jax.random.split(key)
        pos_out, vel_out = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=1.0, key=subkey, G=G, eps=eps
        )

        phi = _softened_potential(pos_out, masses, G=G, eps=eps)
        E = 0.5 * jnp.sum(vel_out**2, axis=1) + phi
        rho, _ = spearmanr(masses, E)

        assert rho < -0.6, f"s=1 should give rho < -0.6, got rho={rho:.3f}"

    def test_virial_ratio_rescaling(self, key, G, eps):
        """Output has virial ratio Q ~ Q_target."""
        from progenax.profiles import PlummerProfile

        N = 200
        profile = PlummerProfile(r_h=1.0)
        masses = jnp.ones(N)

        key, subkey = jax.random.split(key)
        positions = profile.sample_positions(masses, subkey)
        key, subkey = jax.random.split(key)
        velocities = jax.random.normal(subkey, (N, 3)) * 0.5

        Q_target = 1.0
        key, subkey = jax.random.split(key)
        pos_out, vel_out = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=0.5, key=subkey,
            G=G, eps=eps, Q_target=Q_target
        )

        phi = _softened_potential(pos_out, masses, G=G, eps=eps)
        U = 0.5 * jnp.sum(masses * phi)
        K = 0.5 * jnp.sum(masses * jnp.sum(vel_out**2, axis=1))
        Q = 2.0 * K / jnp.abs(U)

        assert jnp.isclose(Q, Q_target, rtol=0.1), (
            f"Q={float(Q):.3f}, expected Q_target={Q_target}"
        )


# =============================================================================
# TestSubrPlaceholder
# =============================================================================


class TestSubrPlaceholder:
    """Test Subr-Kroupa-Baumgardt placeholder."""

    def test_raises_not_implemented(self):
        """Raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Subr"):
            generate_mass_segregated_ic_subr()
