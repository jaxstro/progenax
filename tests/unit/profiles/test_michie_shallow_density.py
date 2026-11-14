"""Shallow Michie density/normalization; three shared compiled configurations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import gammainc

from progenax.kinematics.michie_df import MichieVelocityDF
from progenax.profiles.michie import (
    MichieProfile,
    _michie_poisson_rhs,
    _michie_radial_integral,
    michie_density,
)

_radial_values = jax.jit(
    jax.vmap(
        lambda a: (
            _michie_radial_integral(a),
            jax.jacfwd(_michie_radial_integral)(a),
            jax.jacrev(_michie_radial_integral)(a),
        )
    )
)


def _density_point(log_w):
    w = jnp.exp(log_w)
    rho = michie_density(w, 0.0)
    curvature = _michie_poisson_rhs(0.0, jnp.array([w, 0.0]), (w, jnp.inf))[1]
    return jnp.array([jnp.log(rho), curvature])


_density_values = jax.jit(
    jax.vmap(
        lambda log_w: (
            _density_point(log_w),
            jax.jacfwd(_density_point)(log_w),
            jax.jacrev(_density_point)(log_w),
        )
    )
)
_N_ODE = 2048
_N_CDF = 1024
_OUTER = 0.03


def _normalization_chain(log_w):
    w = jnp.exp(log_w)
    profile = MichieProfile.from_W0_rc(
        w, 1.0, jnp.inf, xi_max=_OUTER, n_ode_points=_N_ODE, n_grid=_N_CDF
    )
    df = MichieVelocityDF(
        w,
        1.0,
        jnp.inf,
        xi_max=_OUTER,
        n_ode_points=_N_ODE,
        speed_method="quadrature",
    )
    profile_mu = jnp.trapezoid(
        profile._r_grid**2 * profile.density(profile._r_grid), profile._r_grid
    )
    aux = jnp.array(
        [
            profile.r_t,
            profile_mu,
            df.mu,
            profile.density(jnp.array(0.0)),
            profile._cdf_grid[-1],
        ]
    )
    return jnp.log(df.mu), aux


_chain_values = jax.jit(jax.value_and_grad(_normalization_chain, has_aux=True))


def _king_reference(w):
    """Independent gamma-density/DOP853 continuum isotropic reference."""
    rho0 = np.exp(w) * gammainc(2.5, w)
    start = 1e-7 * np.sqrt(w)

    def rhs(x, y):
        density = np.exp(y[0]) * gammainc(2.5, y[0]) if y[0] > 0 else 0
        return [y[1], -2 * y[1] / x - 9 * density / rho0]

    def boundary(x, y):
        return y[0]

    boundary.terminal = True
    boundary.direction = -1
    sol = solve_ivp(
        rhs,
        (start, _OUTER),
        [w - 1.5 * start**2, -3 * start],
        rtol=3e-12,
        atol=3e-14 * w,
        method="DOP853",
        events=boundary,
        dense_output=True,
    )
    assert sol.success and len(sol.t_events[0]) == 1
    tidal = sol.t_events[0][0]
    mu = -(tidal**2) * sol.sol(tidal)[1] / 9
    return tidal, mu


def test_radial_integral_retains_small_velocity_value_and_derivative():
    """Catch cancellation in the radial lowered-Maxwellian integral and its series join."""
    a = np.array([0.0, 1e-8, 1e-4, 0.5 - 1e-10, 0.5, 0.5 + 1e-10, 1.0, 3.0])
    value, forward, reverse = map(np.asarray, _radial_values(jnp.asarray(a)))
    nodes, weights = np.polynomial.legendre.leggauss(128)
    # Substitute u=a*z in integral_-a^a expm1((a²-u²)/2)du.
    reference = a * np.sum(
        weights * np.expm1(a[:, None] ** 2 * (1 - nodes**2) / 2), axis=1
    )
    slope = a * reference + 2 * a**2
    np.testing.assert_allclose(value, reference, rtol=3e-14, atol=0)
    np.testing.assert_allclose(forward, slope, rtol=3e-14, atol=0)
    np.testing.assert_allclose(reverse, slope, rtol=3e-14, atol=0)


def test_positive_shallow_density_keeps_king_limit_and_normalized_center():
    """Catch positive-W floors, absolute rho0 cutoffs and lost shallow-density gradients."""
    w = np.array([1e-15, 1e-10, 1e-4, 0.125, 0.126, 5.0])
    value, forward, reverse = map(np.asarray, _density_values(jnp.log(jnp.asarray(w))))
    rho = np.sqrt(2 * np.pi) * np.exp(w) * gammainc(2.5, w)
    log_slope = w * gammainc(1.5, w) / gammainc(2.5, w)
    # The unchanged 256-point tangential trapezoid has measured continuum
    # errors up to 1.37e-5 in density and 1.57e-5 in its slope at W=5.
    np.testing.assert_allclose(np.exp(value[:, 0]), rho, rtol=4.2e-5, atol=0)
    np.testing.assert_allclose(forward[:, 0], log_slope, rtol=1e-5, atol=0)
    np.testing.assert_allclose(reverse[:, 0], log_slope, rtol=1e-5, atol=0)
    np.testing.assert_allclose(value[:, 1], -3.0, atol=2e-15, rtol=0)
    # Normalize the derivative to logW; exact central rho/rho0=1 is constant.
    np.testing.assert_allclose(
        forward[:, 1], 0.0, atol=1024 * np.finfo(float).eps, rtol=0
    )
    np.testing.assert_allclose(
        reverse[:, 1], 0.0, atol=1024 * np.finfo(float).eps, rtol=0
    )


@pytest.mark.slow
def test_resolved_shallow_michie_profile_and_df_scale_recover_isotropic_limit():
    """Catch a disabled shallow source or zero normalization via a reverse-ODE chain.

    Slow: cold compilation differentiates the adaptive ODE with its 256-node
    density quadrature; the three-test file took 50.54 seconds on this host.
    """
    w = 1e-4
    (_, actual), derivative = _chain_values(jnp.log(w))
    tidal, profile_mu, df_mu, central_density, final_cdf = np.asarray(actual)
    expected_tidal, expected_mu = _king_reference(w)
    step = 1e-3
    logs = [np.log(_king_reference(w * np.exp(k * step))[1]) for k in [-2, -1, 1, 2]]
    expected_derivative = (logs[0] - 8 * logs[1] + 8 * logs[2] - logs[3]) / (12 * step)
    print(
        "Shallow Michie errors:",
        {
            "tidal_relative": tidal / expected_tidal - 1,
            "profile_mu_relative": profile_mu / expected_mu - 1,
            "df_mu_relative": df_mu / expected_mu - 1,
            "log_mu_log_W_derivative_error": float(derivative) - expected_derivative,
        },
    )
    # Resolved W0^.5 spatial scale; these bounds include the fixed tangential
    # quadrature, interpolation, stored radial grid and ODE error. The measured
    # errors are 1.70e-9,3.03e-7,3.44e-9 and 2.25e-7, respectively; the
    # following bounds allow about 4–4.5 times those errors.
    np.testing.assert_allclose(tidal, expected_tidal, rtol=7e-9, atol=0)
    np.testing.assert_allclose(profile_mu, expected_mu, rtol=1.2e-6, atol=0)
    np.testing.assert_allclose(df_mu, expected_mu, rtol=1.5e-8, atol=0)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=0, atol=1e-6)
    np.testing.assert_allclose(central_density, 1.0, rtol=0, atol=2e-15)
    np.testing.assert_allclose(final_cdf, 1.0, rtol=0, atol=2e-15)
