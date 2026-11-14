"""Vacuum continuation of finite spherical profiles; three shared JIT programs."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.profiles import compute_profile_potential, make_profile

_KING = dict(W0=5.0, xi_max=50.0, n_ode_points=4096, n_grid=2048)
_SCALE = 1.3


@jax.jit
def _king_tidal_radius():
    profile = make_profile("king", _SCALE, **_KING)
    last_positive = jnp.max(
        jnp.where(profile.psi_grid > 0, profile.xi_grid * profile.r_c, 0.0)
    )
    return profile.r_t, last_positive


def _potential(theta, profile):
    radius, mass, scale, gravity, truncation, gamma = theta
    kwargs = _KING if profile == "king" else dict(r_t=truncation, gamma=gamma)
    return compute_profile_potential(
        jnp.array([[radius, 0.0, 0.0]]), profile, mass, scale, gravity, **kwargs
    )[0]


@partial(jax.jit, static_argnames=("profile",))
def _batch(theta, *, profile):
    def evaluate(row):
        return (
            _potential(row, profile),
            jax.jacfwd(_potential, argnums=0)(row, profile),
            jax.jacrev(_potential, argnums=0)(row, profile),
        )

    def center(mass):
        _, _, scale, gravity, truncation, gamma = theta[0]
        kwargs = _KING if profile == "king" else dict(r_t=truncation, gamma=gamma)
        return compute_profile_potential(
            jnp.zeros((1, 3)), profile, mass, scale, gravity, **kwargs
        )[0]

    mass = theta[0, 1]
    center_outputs = center(mass), jax.jacfwd(center)(mass), jax.grad(center)(mass)
    return jax.vmap(evaluate)(theta), center_outputs


@pytest.fixture(scope="module")
def exterior_results():
    records = {}
    king_tidal, king_last = map(float, _king_tidal_radius())
    eff_tidal = 13.0
    eff_last = eff_tidal * (998 / 999) ** 2
    for profile, tidal, last in (
        ("king", king_tidal, king_last),
        ("eff", eff_tidal, eff_last),
    ):
        # First row pins the actual boundary value; the next three are exterior.
        # The last two probe the final interior interpolation interval.
        theta = np.array(
            [
                [tidal, 1.0, _SCALE, 1.1, 13.0, 3.0],
                [1.25 * tidal, 0.5, _SCALE, 0.7, 13.0, 2.7],
                [2.0 * tidal, 1.0, _SCALE, 1.1, 13.0, 3.0],
                [5.0 * tidal, 2.0, _SCALE, 1.7, 13.0, 3.3],
                [tidal - 0.5 * (tidal - last), 1.0, _SCALE, 1.1, 13.0, 3.0],
                [tidal - 0.25 * (tidal - last), 1.0, _SCALE, 1.1, 13.0, 3.0],
            ]
        )
        outputs, center = _batch(jnp.asarray(theta), profile=profile)
        values, forward, reverse = map(np.asarray, outputs)
        records[profile] = (tidal, theta, values, forward, reverse, np.asarray(center))
    return records


def test_finite_profiles_obey_shell_theorem_in_their_existing_gauges(exterior_results):
    """Catch clamped exterior values/zero forces and a King boundary-potential seam."""
    for profile, (tidal, theta, values, _, _, _) in exterior_results.items():
        theta, values = theta[:4], values[:4]
        radius, mass, _, gravity = theta[:, :4].T
        expected = -gravity * mass / radius
        if profile == "king":
            expected += gravity * mass / tidal
        # Analytic exterior arithmetic scales with GM/rt, including exact-zero
        # King boundary cancellation; this is not an ODE-convergence tolerance.
        scale = gravity * mass / tidal
        error = np.abs(values - expected) / scale
        print(profile, "potential error / (GM/rt):", error.tolist())
        np.testing.assert_array_less(error, 2 * np.finfo(float).eps)


def test_exterior_forces_parameter_ad_and_mass_homology(exterior_results):
    """Catch exterior AD through inactive singular terms or lost mass/scale dependence."""
    for profile, (
        tidal,
        theta,
        values,
        forward,
        reverse,
        _,
    ) in exterior_results.items():
        theta, values = theta[1:4], values[1:4]
        radius, mass, scale, gravity = theta[:, :4].T
        expected = np.zeros_like(theta)
        expected[:, 0] = gravity * mass / radius**2
        expected[:, 1] = values / mass
        expected[:, 3] = values / gravity
        if profile == "king":
            # With fixed W0 and ODE controls, rt = scale * xi_t exactly.
            expected[:, 2] = -gravity * mass / (scale * tidal)
        # EFF exterior is independent of a, gamma, and rt at fixed total mass;
        # King ignores the EFF-only truncation/gamma arguments.
        derivative_scale = gravity * mass / tidal
        for mode, actual in (("forward", forward[1:4]), ("reverse", reverse[1:4])):
            scaled_error = np.abs(actual - expected) / derivative_scale[:, None]
            print(profile, mode, "AD error / (GM/rt):", scaled_error.max())
            np.testing.assert_array_less(scaled_error, 4 * np.finfo(float).eps)
        # Euler's exact degree-one mass identity tests homology at each probe.
        np.testing.assert_allclose(
            mass * reverse[1:4, 1], values, rtol=4 * np.finfo(float).eps, atol=0
        )


def test_boundary_cell_continuity_and_inactive_center_denominator(exterior_results):
    """Catch an unanchored King boundary cell and inactive exterior 1/0 poisoning mass AD."""
    for profile, (tidal, theta, values, _, _, center) in exterior_results.items():
        scale = theta[0, 3] * theta[0, 1] / tidal
        # Both interior interpolants are linear in their last interval. Values
        # at half and quarter of its remaining width extrapolate to the boundary.
        endpoint = 2 * values[5] - values[4]
        seam = abs(endpoint - values[0]) / scale
        print(profile, "boundary extrapolation error / (GM/rt):", seam)
        assert seam < 4 * np.finfo(float).eps
        # Radius is fixed at zero here: only mass AD is asserted. Cartesian
        # radial-norm differentiation at the origin is outside this test's claim.
        np.testing.assert_allclose(
            center[1:], center[0] / theta[0, 1], rtol=4 * np.finfo(float).eps, atol=0
        )
