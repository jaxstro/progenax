"""
Physics validation tests for binary star orbital mechanics.

Tests verify that implementations match theoretical predictions from:
- Kepler's laws of planetary motion
- Murray & Dermott (1999), "Solar System Dynamics"
- Binney & Tremaine (2008), "Galactic Dynamics"

Each test has quantitative error bounds based on theoretical expectations.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import PLANETARY
from progenax.binaries import (
    KeplerElements,
    BinaryOrbitalState,
    compute_period,
    period_to_semimajor_axis,
)

# Use planetary/binary units for orbital mechanics tests
G = PLANETARY.G  # ≈ 39.478 [AU³ Msun⁻¹ yr⁻²] = 4π²


class TestKeplerThirdLaw:
    """Verify Kepler's Third Law: T^2 = (4π^2/GM) a^3."""

    def test_period_formula_circular(self):
        """Period formula correct for circular orbit."""
        a = 1.0
        M_total = 1.0

        period = compute_period(a, M_total, G)
        expected = 2.0 * jnp.pi * jnp.sqrt(a**3 / (G * M_total))

        assert abs(float(period) - float(expected)) < 1e-10, \
            f"Period = {float(period):.6f}, expected {float(expected):.6f}"

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_period_scales_as_a_cubed(self, a):
        """Period scales as T ∝ a^(3/2)."""
        M_total = 1.0

        period_1 = compute_period(1.0, M_total, G)
        period_a = compute_period(a, M_total, G)

        ratio = float(period_a / period_1)
        expected_ratio = a ** 1.5

        assert abs(ratio - expected_ratio) < 1e-10, \
            f"T(a={a})/T(1) = {ratio:.4f}, expected {expected_ratio:.4f}"

    def test_period_semimajor_axis_roundtrip(self):
        """period_to_semimajor_axis is inverse of compute_period."""
        a_original = 2.5
        M_total = 1.0

        period = compute_period(a_original, M_total, G)
        a_recovered = period_to_semimajor_axis(period, M_total, G)

        assert abs(float(a_recovered) - a_original) < 1e-10, \
            f"a recovered = {float(a_recovered):.6f}, expected {a_original}"


class TestKeplerEquation:
    """Verify Kepler's equation solver: E - e*sin(E) = M."""

    def test_circular_orbit_mean_equals_eccentric(self):
        """For e=0: E = M (mean anomaly equals eccentric anomaly)."""
        elements = KeplerElements(a=1.0, e=0.0, M0=jnp.pi/4)
        M_total = 1.0

        state = elements.to_state(M_total, G)

        # For circular orbit, position angle should match mean anomaly
        r = state.position
        angle = jnp.arctan2(r[1], r[0])

        # Should be close to M0 (allowing for 2π periodicity)
        angle_diff = abs(float(angle) - jnp.pi/4)
        assert angle_diff < 0.01, f"Position angle {float(angle):.4f} != M0 = {jnp.pi/4:.4f}"

    def test_eccentric_orbit_periapsis(self):
        """At M=0 (periapsis), r = a(1-e)."""
        e = 0.5
        a = 2.0
        elements = KeplerElements(a=a, e=e, M0=0.0)
        M_total = 1.0

        state = elements.to_state(M_total, G)
        r = jnp.linalg.norm(state.position)

        expected_r = a * (1 - e)
        assert abs(float(r) - expected_r) < 1e-6, \
            f"r at periapsis = {float(r):.6f}, expected {expected_r}"

    def test_eccentric_orbit_apoapsis(self):
        """At M=π (apoapsis), r = a(1+e)."""
        e = 0.5
        a = 2.0
        elements = KeplerElements(a=a, e=e, M0=jnp.pi)
        M_total = 1.0

        state = elements.to_state(M_total, G)
        r = jnp.linalg.norm(state.position)

        expected_r = a * (1 + e)
        assert abs(float(r) - expected_r) < 1e-6, \
            f"r at apoapsis = {float(r):.6f}, expected {expected_r}"


class TestBinaryMomentumConservation:
    """Verify center-of-mass and momentum conservation."""

    def test_com_at_origin(self):
        """Binary center of mass is at origin."""
        m1, m2 = 1.0, 0.5
        M_total = m1 + m2

        # Create binary orbital state and get resolved positions
        binary_state = BinaryOrbitalState.from_semi_major_axis(
            m1=m1, m2=m2, a=1.0, e=0.3, M_anom=1.0, G=G
        )
        pos1, vel1, pos2, vel2 = binary_state.to_resolved_positions(G=G)

        # COM = (m1*r1 + m2*r2) / (m1 + m2)
        com = (m1 * pos1 + m2 * pos2) / M_total

        assert jnp.allclose(com, 0.0, atol=1e-10), \
            f"COM = {com}, expected (0, 0, 0)"

    def test_momentum_zero(self):
        """Total momentum is zero (no bulk motion)."""
        m1, m2 = 2.0, 1.0

        binary_state = BinaryOrbitalState.from_semi_major_axis(
            m1=m1, m2=m2, a=1.0, e=0.3, M_anom=1.0, G=G
        )
        pos1, vel1, pos2, vel2 = binary_state.to_resolved_positions(G=G)

        # Total momentum = m1*v1 + m2*v2
        momentum = m1 * vel1 + m2 * vel2

        assert jnp.allclose(momentum, 0.0, atol=1e-10), \
            f"Total momentum = {momentum}, expected (0, 0, 0)"


class TestOrbitalEnergy:
    """Verify orbital energy formula: E = -GM1M2/(2a)."""

    def test_orbital_energy_formula(self):
        """Orbital energy matches analytical formula."""
        m1, m2 = 1.0, 1.0
        a = 2.0

        binary_state = BinaryOrbitalState.from_semi_major_axis(
            m1=m1, m2=m2, a=a, e=0.0, M_anom=0.0, G=G
        )
        pos1, vel1, pos2, vel2 = binary_state.to_resolved_positions(G=G)

        # Kinetic energy
        T = 0.5 * m1 * jnp.sum(vel1**2) + 0.5 * m2 * jnp.sum(vel2**2)

        # Potential energy
        r12 = jnp.linalg.norm(pos2 - pos1)
        V = -G * m1 * m2 / r12

        # Total energy
        E_computed = T + V

        # Analytical: E = -GM1M2/(2a)
        E_analytical = -G * m1 * m2 / (2 * a)

        rel_error = abs(float(E_computed) - float(E_analytical)) / abs(float(E_analytical))
        assert rel_error < 0.01, \
            f"E_computed = {float(E_computed):.6f}, E_analytical = {float(E_analytical):.6f}"


class TestOrbitalVelocity:
    """Verify orbital velocity formulas."""

    def test_circular_orbit_velocity(self):
        """Circular orbit velocity: v = √(GM/a)."""
        m1, m2 = 1.0, 1.0
        M_total = m1 + m2
        a = 1.0

        binary_state = BinaryOrbitalState.from_semi_major_axis(
            m1=m1, m2=m2, a=a, e=0.0, M_anom=0.0, G=G
        )
        pos1, vel1, pos2, vel2 = binary_state.to_resolved_positions(G=G)

        # Relative velocity
        v_rel = jnp.linalg.norm(vel2 - vel1)

        # Expected: v_rel = √(G * M_total / a)
        expected = jnp.sqrt(G * M_total / a)

        assert abs(float(v_rel) - float(expected)) < 1e-6, \
            f"v_rel = {float(v_rel):.6f}, expected {float(expected):.6f}"

    def test_periapsis_velocity(self):
        """Velocity at periapsis is maximum for eccentric orbit."""
        m1, m2 = 1.0, 1.0
        M_total = m1 + m2
        a = 1.0
        e = 0.5

        # At periapsis (M0=0)
        elements_peri = KeplerElements(a=a, e=e, M0=0.0)
        state_peri = elements_peri.to_state(M_total, G)
        v_peri = jnp.linalg.norm(state_peri.velocity)

        # At apoapsis (M0=π)
        elements_apo = KeplerElements(a=a, e=e, M0=jnp.pi)
        state_apo = elements_apo.to_state(M_total, G)
        v_apo = jnp.linalg.norm(state_apo.velocity)

        assert float(v_peri) > float(v_apo), \
            f"v_periapsis = {float(v_peri):.4f} should > v_apoapsis = {float(v_apo):.4f}"


class TestInclinationAndOrientation:
    """Verify orbital plane orientation."""

    def test_equatorial_orbit_in_xy_plane(self):
        """i=0 orbit lies in xy-plane."""
        elements = KeplerElements(a=1.0, e=0.3, i=0.0, M0=1.0)
        M_total = 1.0

        state = elements.to_state(M_total, G)
        r = state.position
        v = state.velocity

        # z-components should be zero
        assert abs(float(r[2])) < 1e-10, f"z position = {float(r[2])}"
        assert abs(float(v[2])) < 1e-10, f"z velocity = {float(v[2])}"

    def test_polar_orbit_passes_poles(self):
        """i=π/2 orbit passes through poles."""
        elements = KeplerElements(a=1.0, e=0.0, i=jnp.pi/2, Omega=0.0, omega=0.0, M0=0.0)
        M_total = 1.0

        state = elements.to_state(M_total, G)
        r = state.position

        # For i=90°, M0=0, orbit starts at periapsis on ascending node
        # Position should be in xz-plane
        assert abs(float(r[1])) < 1e-10, f"y position = {float(r[1])}"


class TestDifferentiability:
    """Verify orbital mechanics are differentiable."""

    def test_grad_through_kepler_solve(self):
        """Gradient flows through Kepler equation solver."""
        def energy_from_a(a):
            elements = KeplerElements(a=a, e=0.3, M0=1.0)
            state = elements.to_state(M_total=1.0, G=G)
            return jnp.sum(state.velocity**2)

        grad_fn = jax.grad(energy_from_a)
        grad_val = grad_fn(1.0)

        assert jnp.isfinite(grad_val), f"Gradient = {grad_val}, expected finite"
        assert float(grad_val) < 0, \
            "dE/da should be negative (larger orbit → slower velocity)"


class TestSmallSemiMajorAxisSTELLAR:
    """B4-1 regression: to_state mean motion must be exact for ALL physical a.

    The historical bug clamped a**3 at an absolute 1e-12 floor (kepler.py:122),
    so in STELLAR units (pc; the package default) realistic stellar binaries
    (a ~ 1e-6 pc) got velocities/energies wrong by ~100%. For a circular orbit
    |v_rel| must equal sqrt(G M / a) at every scale, with no unit-dependent floor.
    """

    def test_circular_velocity_exact_across_scales_stellar(self):
        from jaxstro.units import STELLAR
        G_st = STELLAR.G
        M = 2.0
        # 1.86e-7 pc ~ 0.04 AU, a typical short-period stellar binary separation.
        for a in [1.0, 1e-2, 1e-4, 1.86e-7]:
            el = KeplerElements(a=a, e=0.0, M0=0.0)
            v = jnp.linalg.norm(el.to_state(M_total=M, G=G_st).velocity)
            v_true = jnp.sqrt(G_st * M / a)
            rel = abs(float(v) - float(v_true)) / float(v_true)
            assert rel < 1e-10, (
                f"a={a:.2e} pc: |v|={float(v):.3e} vs sqrt(GM/a)={float(v_true):.3e}, rel={rel:.2e}"
            )


class TestKeplerEccentricityGradientBoundary:
    """B4-3 regression: gradient of to_state wrt e stays finite at the e->1 boundary."""

    def test_grad_finite_through_e_to_one(self):
        def loss(e):
            el = KeplerElements(a=1.0, e=e, M0=1.0)
            return jnp.sum(el.to_state(M_total=1.0, G=G).velocity ** 2)

        assert jnp.isfinite(jax.grad(loss)(0.9999))
        g_one = jax.grad(loss)(1.0)
        assert jnp.isfinite(g_one), f"grad at e=1.0 = {float(g_one)} (NaN before the B4-3 fix)"


# B4-15: AD-vs-FD grad-checks for the KeplerElements transforms (to_state a/e/M0,
# from_state velocity-scale) are owned by the grad-audit registry
# (tests/validation/grad_audit/registry.py :: KeplerElements.to_state [e, a, M0] +
# KeplerElements.from_state [v_scale]); see
# docs/website/50-validation/differentiability-audit.md. The former
# TestKeplerTransformGradients class was removed here (audit T6 consolidation; registry is SoT).
# test_grad_finite_through_e_to_one (the unique e->1 boundary, B4-3) and all physics stay.


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
