"""Resolved King profile/potential/DF normalization; one compiled case batch."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammainc

from progenax.kinematics.king_df import KingVelocityDF
from progenax.profiles.api import compute_profile_potential, make_profile

_N_ODE = 4096
_N_CDF = 2048
_FRACTIONS = np.array([0.0, 0.25, 0.5])


def _chain(w0, outer):
    profile = make_profile(
        "king", 1.0, W0=w0, xi_max=outer, n_ode_points=_N_ODE, n_grid=_N_CDF
    )
    radii = jnp.asarray(_FRACTIONS) * profile.r_t
    positions = jnp.stack((radii, jnp.zeros(3), jnp.zeros(3)), axis=1)
    potential = compute_profile_potential(
        positions,
        "king",
        M_total=1.0,
        R_half=1.0,
        G=1.0,
        W0=w0,
        xi_max=outer,
        n_ode_points=_N_ODE,
        n_grid=_N_CDF,
    )
    df = KingVelocityDF(
        W0=w0,
        r_c=1.0,
        xi_max=outer,
        n_ode_points=_N_ODE,
        speed_method="quadrature",
    )
    sigma_squared = df._sigma(1.0, 1.0) ** 2
    mu_profile = jnp.trapezoid(
        profile._r_grid**2 * profile.density(profile._r_grid), profile._r_grid
    )
    return profile.r_t, mu_profile, sigma_squared, potential, profile.xi_grid[-1]


_normalization_batch = jax.jit(jax.vmap(_chain))


def _independent_reference(w0, outer):
    # SciPy's incomplete gamma identity and DOP853 are independent of the
    # JAX density series, Tsit5 solve, and stored-grid trapezoidal normalization.
    rho0 = np.exp(w0) * gammainc(2.5, w0)
    start = 1e-7 * np.sqrt(w0)

    def rhs(xi, state):
        w, slope = state
        rho = np.exp(w) * gammainc(2.5, w) if w > 0 else 0.0
        return [slope, -9 * rho / rho0 - 2 * slope / xi]

    def boundary(xi, state):
        return state[0]

    boundary.terminal = True
    boundary.direction = -1
    solution = solve_ivp(
        rhs,
        (start, outer),
        [w0 - 1.5 * start**2, -3 * start],
        method="DOP853",
        rtol=3e-12,
        atol=3e-14 * w0,
        events=boundary,
        dense_output=True,
    )
    assert solution.success and len(solution.t_events[0]) == 1
    tidal = solution.t_events[0][0]
    mu = -(tidal**2) * solution.sol(tidal)[1] / 9
    sigma_squared = 1 / (9 * mu)
    w = solution.sol(_FRACTIONS * tidal)[0]
    w[0] = w0  # The central value is exact, outside the finite-start grid.
    return tidal, mu, sigma_squared, -sigma_squared * w


def test_resolved_shallow_king_normalization_reaches_potential_and_df():
    """Catch absolute-density cutoffs and ignored ODE options yielding infinite King scales."""
    w0 = np.array([1e-4, 5.0])
    outer = np.array([0.03, 50.0])
    tidal, mu, sigma_squared, potential, actual_outer = map(
        np.asarray, _normalization_batch(jnp.asarray(w0), jnp.asarray(outer))
    )
    np.testing.assert_allclose(actual_outer, outer, rtol=0, atol=0)
    reference = [_independent_reference(w, x) for w, x in zip(w0, outer)]
    expected_tidal = np.array([r[0] for r in reference])
    expected_mu = np.array([r[1] for r in reference])
    expected_sigma = np.array([r[2] for r in reference])
    expected_phi = np.stack([r[3] for r in reference])
    print(
        "King chain relative errors:",
        {
            "tidal": (tidal / expected_tidal - 1).tolist(),
            "mass": (mu / expected_mu - 1).tolist(),
            "sigma_squared": (sigma_squared / expected_sigma - 1).tolist(),
            "potential": (potential / expected_phi - 1).tolist(),
        },
    )
    # The shallow grid resolves its W0^(1/2) radial scale. The measured maxima
    # are 6.7e-9 (tidal), 3.3e-6 (interpolated profile mass), 4e-9 (DF scale),
    # and 3.6e-7 (interpolated potential). Separate budgets allow 3.6–5 times
    # those numerical errors; they do not qualify unresolved default grids.
    np.testing.assert_allclose(tidal, expected_tidal, rtol=3e-8, atol=0)
    np.testing.assert_allclose(mu, expected_mu, rtol=1.2e-5, atol=0)
    np.testing.assert_allclose(sigma_squared, expected_sigma, rtol=2e-8, atol=0)
    np.testing.assert_allclose(potential, expected_phi, rtol=1.5e-6, atol=0)
