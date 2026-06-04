"""Tests for binary star orbital mechanics."""

import jax
import jax.numpy as jnp
import pytest


class TestKeplerElements:
    """Test KeplerElements class."""

    def test_kepler_elements_importable(self):
        from progenax.binaries import KeplerElements
        assert KeplerElements is not None

    def test_circular_orbit_creation(self):
        """Create circular orbit and verify structure."""
        from progenax.binaries import KeplerElements
        elements = KeplerElements(a=1.0, e=0.0, i=0.0, Omega=0.0, omega=0.0, M0=0.0)
        assert elements.a == pytest.approx(1.0)
        assert elements.e == pytest.approx(0.0)

    def test_to_state(self):
        """Convert elements to Cartesian state."""
        from progenax.binaries import KeplerElements
        elements = KeplerElements(a=1.0, e=0.0, i=0.0, Omega=0.0, omega=0.0, M0=0.0)
        G = 1.0
        state = elements.to_state(M_total=1.0, G=G)
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)

    def test_to_binary_state(self):
        """Convert elements to resolved binary state."""
        from progenax.binaries import KeplerElements
        elements = KeplerElements(a=1.0, e=0.0, i=0.0, Omega=0.0, omega=0.0, M0=0.0)
        G = 1.0
        r1, v1, r2, v2 = elements.to_binary_state(m1=1.0, m2=1.0, G=G)

        # COM should be at origin
        com = (1.0 * r1 + 1.0 * r2) / 2.0
        assert jnp.allclose(com, 0.0, atol=1e-10)

    def test_round_trip_conversion(self):
        """to_state then from_state should recover elements."""
        from progenax.binaries import KeplerElements
        G = 1.0
        original = KeplerElements(a=1.0, e=0.3, i=0.5, Omega=0.2, omega=0.1, M0=0.0)
        state = original.to_state(M_total=1.0, G=G)
        recovered = KeplerElements.from_state(
            state.position, state.velocity, M_total=1.0, G=G
        )

        assert jnp.abs(recovered.a - original.a) < 0.01
        assert jnp.abs(recovered.e - original.e) < 0.01


class TestKeplerEquation:
    """Test Kepler equation solver."""

    def test_solve_kepler_circular(self):
        """For e=0, E should equal M."""
        from progenax.binaries import KeplerElements
        elements = KeplerElements(a=1.0, e=0.0, i=0.0, Omega=0.0, omega=0.0, M0=1.0)
        E = elements._solve_kepler_equation(1.0, 0.0)
        assert jnp.abs(E - 1.0) < 1e-10

    def test_solve_kepler_eccentric(self):
        """Test eccentric orbit solution."""
        from progenax.binaries import KeplerElements
        elements = KeplerElements(a=1.0, e=0.5, i=0.0, Omega=0.0, omega=0.0, M0=1.0)
        E = elements._solve_kepler_equation(1.0, 0.5)
        # Verify: M = E - e*sin(E)
        M_check = E - 0.5 * jnp.sin(E)
        assert jnp.abs(M_check - 1.0) < 1e-10


class TestPeriodFunctions:
    """Test period/semi-major axis conversions."""

    def test_compute_period(self):
        from progenax.binaries import compute_period
        G = 39.478  # Binary units (AU³/Msun/yr²)
        T = compute_period(a=1.0, M_total=1.0, G=G)
        # Earth orbit: ~1 year
        expected = 2.0 * jnp.pi * jnp.sqrt(1.0 / G)
        assert jnp.abs(T - expected) < 0.01

    def test_period_to_semimajor_axis(self):
        from progenax.binaries import period_to_semimajor_axis, compute_period
        G = 39.478
        T = 1.0  # 1 year
        a = period_to_semimajor_axis(T, M_total=1.0, G=G)
        # Round-trip check
        T_back = compute_period(a, M_total=1.0, G=G)
        assert jnp.abs(T_back - T) < 0.001


class TestBinaryOrbitalState:
    """Test BinaryOrbitalState IC container."""

    def test_binary_orbital_state_importable(self):
        from progenax.binaries import BinaryOrbitalState
        assert BinaryOrbitalState is not None

    def test_from_log_period(self):
        from progenax.binaries import BinaryOrbitalState
        G = 39.478
        state = BinaryOrbitalState.from_log_period(
            m1=1.0, m2=0.5, logP_days=2.0, e=0.3,
            G=G, day_in_time_units=1.0/365.25
        )
        assert state.m1 == pytest.approx(1.0)
        assert state.m2 == pytest.approx(0.5)

    def test_from_semi_major_axis(self):
        from progenax.binaries import BinaryOrbitalState
        G = 39.478
        state = BinaryOrbitalState.from_semi_major_axis(
            m1=1.0, m2=1.0, a=1.0, e=0.0, G=G
        )
        assert state.elements.a == pytest.approx(1.0)

    def test_to_resolved_positions(self):
        from progenax.binaries import BinaryOrbitalState
        G = 1.0
        state = BinaryOrbitalState.from_semi_major_axis(
            m1=1.0, m2=1.0, a=1.0, e=0.0, G=G
        )
        r1, v1, r2, v2 = state.to_resolved_positions(G=G)

        # COM check
        com = 0.5 * (r1 + r2)
        assert jnp.allclose(com, 0.0, atol=1e-10)


class TestBatchOperations:
    """Test vectorized batch operations."""

    def test_batch_elements_to_resolved(self):
        from progenax.binaries import batch_elements_to_resolved
        import jax.numpy as jnp

        G = 1.0
        N = 10
        m1 = jnp.ones(N)
        m2 = jnp.ones(N) * 0.5
        logP_days = jnp.ones(N) * 2.0
        e = jnp.ones(N) * 0.3
        inc = jnp.zeros(N)
        Omega = jnp.zeros(N)
        omega = jnp.zeros(N)
        M_anom = jnp.zeros(N)

        r1, v1, r2, v2 = batch_elements_to_resolved(
            m1, m2, logP_days, e, inc, Omega, omega, M_anom,
            G=G, day_in_time_units=1.0
        )

        assert r1.shape == (N, 3)
        assert v1.shape == (N, 3)
