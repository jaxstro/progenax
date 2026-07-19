r"""AC-BE1 / AC-BE2 — Lane-Emden solver core against EXACT closed-form oracles.

The Lane-Emden family admits three exact solutions, which give us an oracle with no
fitting whatsoever (cf. the "exact over fitted" rule):

    n = 0:  theta = 1 - xi^2/6            xi_1 = sqrt(6)
    n = 1:  theta = sin(xi) / xi          xi_1 = pi
    n = 5:  theta = (1 + xi^2/3)^(-1/2)   xi_1 = infinity (no zero)

Each is verified against the ODE itself rather than quoted. For n=0:
``theta'' + (2/xi) theta' = -1/3 - 2/3 = -1 = -theta^0``. Correct.

The isothermal (Bonnor-Ebert) case has no closed form, so AC-BE2 checks it against
its origin series, derived by substituting ``psi = a2 xi^2 + a4 xi^4`` into
``psi'' + (2/xi) psi' = e^{-psi}``:

    LHS = (2 a2 + 12 a4 xi^2) + (4 a2 + 8 a4 xi^2) = 6 a2 + 20 a4 xi^2
    RHS = 1 - psi + O(xi^4)                        = 1 - a2 xi^2 + O(xi^4)

    xi^0:  6 a2 = 1      ->  a2 = 1/6
    xi^2:  20 a4 = -a2   ->  a4 = -1/120

    psi(xi) = xi^2/6 - xi^4/120 + O(xi^6)

Dimensionless enclosed mass is ``m = -xi^2 theta'`` (polytrope) and ``m = xi^2 psi'``
(isothermal); both reduce to ``m -> xi^3/3`` near the origin, which is checked as a
cross-consistency condition between the two branches.
"""

import jax.numpy as jnp
import pytest

from gravoturb.profiles.lane_emden import (
    polytrope_xi1,
    solve_isothermal,
    solve_polytrope,
)

# Tsit5 with rtol=1e-8/atol=1e-10; leave headroom for the origin-series start.
RTOL_EXACT = 1e-7


def _theta_exact_n0(xi):
    return 1.0 - xi**2 / 6.0


def _theta_exact_n1(xi):
    # sin(xi)/xi, with the removable singularity at the origin handled.
    return jnp.where(xi > 0, jnp.sin(jnp.where(xi > 0, xi, 1.0)) / jnp.where(xi > 0, xi, 1.0), 1.0)


def _theta_exact_n5(xi):
    return (1.0 + xi**2 / 3.0) ** (-0.5)


class TestPolytropeExactSolutions:
    """AC-BE1 — the three closed-form Lane-Emden solutions."""

    def test_n0_matches_exact_theta(self):
        # Integrate only to xi_1 = sqrt(6); past the first zero theta^n is not
        # the physical solution any more.
        sol = solve_polytrope(n=0.0, xi_max=float(jnp.sqrt(6.0)), n_points=400)
        assert jnp.allclose(sol.y, _theta_exact_n0(sol.xi), rtol=RTOL_EXACT, atol=1e-9)

    def test_n1_matches_exact_theta(self):
        sol = solve_polytrope(n=1.0, xi_max=float(jnp.pi), n_points=400)
        assert jnp.allclose(sol.y, _theta_exact_n1(sol.xi), rtol=RTOL_EXACT, atol=1e-9)

    def test_n5_matches_exact_theta(self):
        # n=5 has infinite extent, so we can integrate well past any xi_1.
        sol = solve_polytrope(n=5.0, xi_max=30.0, n_points=600)
        assert jnp.allclose(sol.y, _theta_exact_n5(sol.xi), rtol=RTOL_EXACT, atol=1e-9)

    def test_n5_never_reaches_zero(self):
        """The n=5 solution is strictly positive -- it has no finite radius."""
        sol = solve_polytrope(n=5.0, xi_max=50.0, n_points=800)
        assert jnp.all(sol.y > 0.0)


class TestPolytropeFirstZero:
    """AC-BE1 — xi_1 located by event detection, against exact values."""

    def test_xi1_n0_is_sqrt6(self):
        assert jnp.allclose(polytrope_xi1(n=0.0), jnp.sqrt(6.0), rtol=1e-8)

    def test_xi1_n1_is_pi(self):
        assert jnp.allclose(polytrope_xi1(n=1.0), jnp.pi, rtol=1e-8)


class TestIsothermalOriginSeries:
    """AC-BE2 — Bonnor-Ebert psi against its derived origin series."""

    def test_psi_matches_series_near_origin(self):
        sol = solve_isothermal(xi_max=0.2, n_points=200)
        series = sol.xi**2 / 6.0 - sol.xi**4 / 120.0
        # The neglected term is O(xi^6); at xi <= 0.2 that is <~ 1e-6 relative.
        assert jnp.allclose(sol.y, series, rtol=1e-5, atol=1e-12)

    def test_psi_is_monotonically_increasing(self):
        """Density rho = rho_c e^{-psi} must fall outward, so psi must rise."""
        sol = solve_isothermal(xi_max=10.0, n_points=500)
        assert jnp.all(jnp.diff(sol.y) > 0.0)

    def test_isothermal_has_no_finite_edge(self):
        """psi stays finite and density stays positive -- BE needs external pressure."""
        sol = solve_isothermal(xi_max=50.0, n_points=800)
        assert jnp.all(jnp.isfinite(sol.y))
        assert jnp.all(jnp.exp(-sol.y) > 0.0)


class TestEnclosedMass:
    """Cross-consistency: both branches give m -> xi^3/3 at small xi."""

    def test_isothermal_mass_small_xi(self):
        """m = xi^2 psi' = xi^3/3 - xi^5/30, from psi' = xi/3 - xi^3/30.

        The tolerance is set from the KNOWN truncation error, not tuned. Carrying the
        series one order further (42 a6 = 1/120 + 1/72 = 1/45, so a6 = 1/1890) gives a
        neglected term ``+xi^7/315``, i.e. a relative error of ``xi^4/105``. At
        xi = 0.05 that is 6e-8, so rtol=1e-6 leaves an order of magnitude of margin.
        """
        sol = solve_isothermal(xi_max=0.05, n_points=200)
        series = sol.xi**3 / 3.0 - sol.xi**5 / 30.0
        assert jnp.allclose(sol.m, series, rtol=1e-6, atol=1e-14)

    def test_polytrope_mass_small_xi(self):
        """m = -xi^2 theta' = xi^3/3 - n xi^5/30, from theta = 1 - xi^2/6 + n xi^4/120.

        Same truncation argument as the isothermal case; at n=1.5 the neglected xi^7
        term contributes ~8e-8 relative at xi = 0.05.
        """
        n = 1.5
        sol = solve_polytrope(n=n, xi_max=0.05, n_points=200)
        series = sol.xi**3 / 3.0 - n * sol.xi**5 / 30.0
        assert jnp.allclose(sol.m, series, rtol=1e-6, atol=1e-14)

    def test_isothermal_mass_series_converges_at_expected_order(self):
        """The xi^5-truncated series must converge as xi^4 relative -- a rate check.

        This is the real guard: a solver bug would break the ORDER, whereas a fixed
        tolerance at a fixed xi can be satisfied by accident.
        """
        errors = []
        for xi_max in (0.4, 0.2, 0.1):
            sol = solve_isothermal(xi_max=xi_max, n_points=200)
            series = sol.xi**3 / 3.0 - sol.xi**5 / 30.0
            errors.append(float(jnp.abs(sol.m[-1] - series[-1]) / series[-1]))
        # Halving xi must cut the relative error by 2^4 = 16.
        assert errors[0] / errors[1] == pytest.approx(16.0, rel=0.05)
        assert errors[1] / errors[2] == pytest.approx(16.0, rel=0.05)

    def test_polytrope_n0_mass_is_exact(self):
        """For n=0, m = -xi^2 theta' = xi^3/3 exactly, everywhere."""
        sol = solve_polytrope(n=0.0, xi_max=float(jnp.sqrt(6.0)), n_points=400)
        assert jnp.allclose(sol.m, sol.xi**3 / 3.0, rtol=RTOL_EXACT, atol=1e-12)

    def test_isothermal_mass_strictly_increasing(self):
        """m(xi) must be strictly increasing for monotone_inverse_interp to accept it."""
        sol = solve_isothermal(xi_max=20.0, n_points=600)
        assert jnp.all(jnp.diff(sol.m) > 0.0)


class TestSolutionStructure:
    def test_shapes_are_n_points(self):
        sol = solve_isothermal(xi_max=6.0, n_points=257)
        assert sol.xi.shape == (257,)
        assert sol.y.shape == (257,)
        assert sol.dy.shape == (257,)
        assert sol.m.shape == (257,)

    def test_grid_spans_requested_domain(self):
        sol = solve_isothermal(xi_max=6.0, n_points=100)
        assert float(sol.xi[-1]) == pytest.approx(6.0, rel=1e-12)
        assert float(sol.xi[0]) > 0.0  # started off-origin to dodge the singularity
