"""Orbital state identities at coordinate boundaries; three compiled programs."""

import jax
import jax.numpy as jnp
import numpy as np

from progenax.binaries import KeplerElements


def _roundtrip(case):
    eccentricity, inclination, scale = case
    gravity = scale**3
    original = KeplerElements(
        a=scale, e=eccentricity, i=inclination, Omega=0.3, omega=0.7, M0=0.8
    )
    state = original.to_state(1.0, gravity)
    recovered = KeplerElements.from_state(state.position, state.velocity, 1.0, gravity)
    replay = recovered.to_state(1.0, gravity)
    return (replay.position - state.position) / scale, (
        replay.velocity - state.velocity
    ) / scale


_roundtrip_batch = jax.jit(jax.vmap(_roundtrip))


def _pericenter(eccentricity):
    state = KeplerElements(a=1.0, e=eccentricity, M0=0.0).to_state(1.0, 1.0)
    return state.position, state.velocity


_pericenter_batch = jax.jit(jax.vmap(_pericenter))


def _well_conditioned_replay(state):
    elements = KeplerElements.from_state(state[:3], state[3:], 1.0, 1.0)
    replay = elements.to_state(1.0, 1.0)
    return jnp.concatenate((replay.position, replay.velocity))


_replay_derivatives = jax.jit(
    lambda state: (
        jax.jacfwd(_well_conditioned_replay)(state),
        jax.jacrev(_well_conditioned_replay)(state),
    )
)


def test_equatorial_retrograde_and_unit_covariant_state_roundtrip():
    """Catch wrong retrograde gauge orientation and dimensional ascending-node cutoffs."""
    cases = jnp.array(
        [
            [0.4, jnp.pi, 1.0],
            [0.0, jnp.pi, 1.0],
            [0.4, jnp.pi - 0.01, 1.0],
            [0.4, 0.7, 1.0],
            [0.4, 0.7, 1e-6],
            [0.4, 0.7, 1e-12],
        ]
    )
    position_error, velocity_error = map(np.asarray, _roundtrip_batch(cases))
    # A change of native length unit multiplies r,v by scale and G by
    # scale^3 at fixed time/mass units. The physical state cannot change.
    # Measured coordinate error is 6.25e-15; 3e-14 bounds it by about 5 times.
    np.testing.assert_allclose(position_error, 0.0, atol=3e-14, rtol=0)
    np.testing.assert_allclose(velocity_error, 0.0, atol=3e-14, rtol=0)


def test_high_eccentricity_pericenter_matches_kepler_invariants():
    """Catch absolute floors corrupting legitimate near-parabolic periapsis speeds."""
    eccentricity = np.array([0.0, 0.99, 1 - 1e-8, 1 - 1e-14])
    position, velocity = map(np.asarray, _pericenter_batch(jnp.asarray(eccentricity)))
    expected_radius = 1 - eccentricity
    expected_speed = np.sqrt((1 + eccentricity) / (1 - eccentricity))
    np.testing.assert_allclose(position[:, 0], expected_radius, rtol=3e-15, atol=0)
    np.testing.assert_allclose(position[:, 1:], 0.0, atol=0, rtol=0)
    np.testing.assert_allclose(velocity[:, 1], expected_speed, rtol=3e-15, atol=0)
    np.testing.assert_allclose(velocity[:, [0, 2]], 0.0, atol=0, rtol=0)
    angular_momentum = np.cross(position, velocity)[:, 2]
    expected_h = np.sqrt((1 - eccentricity) * (1 + eccentricity))
    np.testing.assert_allclose(angular_momentum, expected_h, rtol=3e-15, atol=0)
    kinetic = np.sum(velocity**2, axis=1) / 2
    potential = -1 / np.linalg.norm(position, axis=1)
    # Near e=1, T and |V| greatly exceed |E|. Bound cancellation by the
    # terms being subtracted, rather than requiring uniform relative E error.
    error = np.abs(kinetic + potential + 0.5) / (kinetic + np.abs(potential))
    assert np.max(error) < 8 * np.finfo(float).eps


def test_state_roundtrip_jacobians_are_identity_away_from_degenerate_angles():
    """Catch broken forward/reverse derivatives in a regular eccentric inclined orbit."""
    # Independently supplied Cartesian state is bound and nondegenerate.
    state = jnp.array([0.8, 0.3, 0.2, -0.2, 0.9, 0.4])
    forward, reverse = _replay_derivatives(state)
    # Measured Jacobian error is 2.67e-15; 1e-14 is fewer than 4 times that.
    np.testing.assert_allclose(forward, np.eye(6), atol=1e-14, rtol=0)
    np.testing.assert_allclose(reverse, np.eye(6), atol=1e-14, rtol=0)
