# progenax/tests/unit/profiles/test_limepy_multimass.py
"""Unit tests for the multi-mass LIMEPY coupled equilibrium (Phase 2, Layer A).

Layer A — solve_multimass_limepy(alpha_j, m_j, W0, g, delta): one coupled Poisson
solve given central density fractions alpha_j,

    (1/xi^2) d/dxi(xi^2 dpsi/dxi) = -9 sum_j alpha_j rho_hat_j(xi),
    rho_hat_j(xi) = limepy_density_hat(mu_j^(2 delta) psi, g)
                    / limepy_density_hat(mu_j^(2 delta) W0, g),
    mu_j = m_j / bar_m,  bar_m = sum_j m_j alpha_j   (Gieles & Zocchi 2015, Eqs 24-29).

The structural oracle: at delta=0 every mu_j^(2 delta)=1, so each rho_hat_j collapses
to the single-mass density and the source is (sum alpha_j) rho_hat = rho_hat -- the
solve must reproduce solve_limepy_profile exactly, for ANY alpha_j / m_j.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


class TestMultiMassCoreDelta0:
    """delta=0 is the single-mass model, structurally (the cleanest oracle)."""

    @pytest.mark.parametrize("g", [0.0, 1.0, 2.0])
    def test_delta0_recovers_single_mass_potential(self, g):
        """solve_multimass_limepy(delta=0) potential psi(xi) is identical to
        solve_limepy_profile(W0, g), independent of the (alpha_j, m_j) supplied."""
        from progenax.profiles.limepy import solve_limepy_profile
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        W0 = 7.0
        alpha_j = jnp.array([0.6, 0.3, 0.1])
        m_j = jnp.array([0.3, 1.0, 8.0])
        xi_s, psi_s = solve_limepy_profile(W0, g=g, xi_max=300.0, n_points=2000)
        xi_m, psi_m, rho_j = solve_multimass_limepy(
            alpha_j, m_j, W0=W0, g=g, delta=0.0, xi_max=300.0, n_points=2000
        )
        np.testing.assert_allclose(np.asarray(xi_m), np.asarray(xi_s), rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(psi_m), np.asarray(psi_s), rtol=1e-9, atol=1e-9)

    def test_delta0_components_share_single_mass_density(self):
        """At delta=0 every component density equals the single-mass normalized
        density (they ride the identical potential with identical rescaling = none)."""
        from progenax.profiles.limepy import solve_limepy_profile, limepy_density_hat
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        W0 = 6.0
        alpha_j = jnp.array([0.5, 0.5])
        m_j = jnp.array([0.5, 5.0])
        xi, psi, rho_j = solve_multimass_limepy(
            alpha_j, m_j, W0=W0, g=1.0, delta=0.0, xi_max=300.0, n_points=2000
        )
        single = limepy_density_hat(psi, 1.0) / limepy_density_hat(jnp.asarray(W0), 1.0)
        for j in range(2):
            np.testing.assert_allclose(np.asarray(rho_j[j]), np.asarray(single),
                                       rtol=1e-6, atol=1e-8)


def _half_mass_xi(xi, rho):
    """Dimensionless half-mass radius of a component from its density rho_hat(xi)."""
    integrand = rho * xi**2
    dxi = xi[1] - xi[0]
    M = jnp.concatenate([jnp.zeros(1),
                         jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1])) * dxi])
    return float(jnp.interp(0.5 * M[-1], M, xi))


class TestMultiMassSegregation:
    """delta>0 produces mass segregation as an equilibrium (heavy more concentrated)."""

    def test_heavy_component_is_more_centrally_concentrated(self):
        """At delta=0.5 the heavy component has a SMALLER half-mass radius than the
        light one -- the equilibrium signature of mass segregation (deeper effective
        well for larger mu_j)."""
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        m_j = jnp.array([0.3, 8.0])
        alpha_j = jnp.array([0.5, 0.5])
        xi, psi, rho_j = solve_multimass_limepy(
            alpha_j, m_j, W0=7.0, g=1.0, delta=0.5, xi_max=300.0, n_points=3000
        )
        xh_light = _half_mass_xi(xi, rho_j[0])
        xh_heavy = _half_mass_xi(xi, rho_j[1])
        assert xh_heavy < xh_light, f"heavy r_h={xh_heavy:.2f} not < light r_h={xh_light:.2f}"

    def test_segregation_strength_increases_with_delta(self):
        """The light/heavy half-mass-radius ratio (a segregation strength) grows
        monotonically with delta, and is ~1 (no segregation) at delta=0."""
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        m_j = jnp.array([0.3, 8.0])
        alpha_j = jnp.array([0.5, 0.5])
        ratios = []
        for delta in (0.0, 0.2, 0.4, 0.6):
            xi, psi, rho_j = solve_multimass_limepy(
                alpha_j, m_j, W0=7.0, g=1.0, delta=delta, xi_max=300.0, n_points=3000
            )
            ratios.append(_half_mass_xi(xi, rho_j[0]) / _half_mass_xi(xi, rho_j[1]))
        assert abs(ratios[0] - 1.0) < 1e-3, f"delta=0 should give no segregation: {ratios[0]:.3f}"
        assert np.all(np.diff(ratios) > 0), f"segregation not monotonic in delta: {ratios}"

    def test_differentiable_in_W0_g_delta_alpha(self):
        """Gradients flow through the coupled solve in all of (W0, g, delta, alpha_j)
        -- the structural and equipartition parameters are jointly inferable."""
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        m_j = jnp.array([0.3, 1.0, 8.0])

        def shape_metric(W0, g, delta, alpha_j):
            xi, psi, rho_j = solve_multimass_limepy(
                alpha_j, m_j, W0=W0, g=g, delta=delta, xi_max=300.0, n_points=2000
            )
            return jnp.mean(psi[:300]) + jnp.sum(rho_j[:, :300])

        alpha0 = jnp.array([0.6, 0.3, 0.1])
        dW0 = jax.grad(shape_metric, 0)(7.0, 1.0, 0.4, alpha0)
        dg = jax.grad(shape_metric, 1)(7.0, 1.0, 0.4, alpha0)
        dd = jax.grad(shape_metric, 2)(7.0, 1.0, 0.4, alpha0)
        da = jax.grad(shape_metric, 3)(7.0, 1.0, 0.4, alpha0)
        assert jnp.isfinite(dW0) and jnp.abs(dW0) > 0
        assert jnp.isfinite(dg) and jnp.abs(dg) > 0
        assert jnp.isfinite(dd) and jnp.abs(dd) > 0  # delta genuinely moves the solution
        assert jnp.all(jnp.isfinite(da))
