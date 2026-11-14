"""Concrete OM admissibility and exact-center direction contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.kinematics import PlummerVelocityDF
from progenax.kinematics.eddington import assign_om_directions


@pytest.mark.parametrize("convert", [float, np.float64, jnp.asarray])
def test_concrete_scalar_admissibility(convert):
    with pytest.raises(ValueError, match="0.75 a"):
        PlummerVelocityDF(r_h=1.0, anisotropy_radius=convert(0.4))
    assert (
        PlummerVelocityDF(r_h=1.0, anisotropy_radius=convert(0.6)).anisotropy_radius
        is not None
    )


@pytest.mark.parametrize("array_radius", [False, True])
def test_center_preserves_speed_and_parameter_derivative(array_radius):
    assert jax.config.jax_enable_x64
    positions = jnp.zeros((16, 3), dtype=jnp.float64)
    speeds = jnp.linspace(0.5, 2.0, 16)
    key = jax.random.PRNGKey(42)

    def draw(scale, ra):
        radii = jnp.full((16,), ra) if array_radius else ra
        return assign_om_directions(key, positions, scale * speeds, radii)

    for run in (draw, jax.jit(draw)):
        value, tangent = jax.jvp(
            lambda s: run(s, 1.5), (jnp.array(1.0),), (jnp.array(1.0),)
        )
        np.testing.assert_allclose(
            jnp.linalg.norm(value, axis=1), speeds, rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(tangent, value, rtol=1e-6, atol=1e-8)
        _, dra = jax.jvp(lambda ra: run(1.0, ra), (jnp.array(1.5),), (jnp.array(1.0),))
        np.testing.assert_allclose(dra, 0, rtol=1e-6, atol=1e-8)


def test_cartesian_om_map_and_position_derivatives():
    """Independent projector formula away from zero; analytic zero Jacobian at zero."""
    assert jax.config.jax_enable_x64
    key = jax.random.PRNGKey(42)
    n = jax.random.normal(key, (1, 3), dtype=jnp.float64)
    n = n / jnp.linalg.norm(n, axis=1, keepdims=True)

    def draw(x):
        return assign_om_directions(key, x[None, :], jnp.ones(1), 1.5)[0]

    for run in (draw, jax.jit(draw)):
        np.testing.assert_allclose(run(jnp.zeros(3)), n[0], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(jax.jacfwd(run)(jnp.zeros(3)), 0, atol=1e-8)
        np.testing.assert_allclose(jax.jacrev(run)(jnp.zeros(3)), 0, atol=1e-8)
        for radius in (1e-4, 0.5, 2.5, 7.0):
            x = jnp.array([radius, radius / 2, -radius / 3])

            def projector(p):
                r = jnp.linalg.norm(p)
                axis = p / r
                b = 1 / jnp.sqrt(1 + (r / 1.5) ** 2)
                return b * n[0] + (1 - b) * axis * jnp.dot(axis, n[0])

            np.testing.assert_allclose(run(x), projector(x), rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(
                jax.jacfwd(run)(x), jax.jacfwd(projector)(x), rtol=1e-6, atol=1e-8
            )
    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(
        assign_om_directions(key, pos, jnp.ones(2), jnp.inf),
        assign_om_directions(key, pos, jnp.ones(2), None),
    )


def test_symbolic_cartesian_stretch():
    """The regularized coefficient preserves radial and tangential eigenvalues."""
    import sympy as sp

    radius, ra = sp.symbols("r ra", positive=True)
    h = sp.sqrt(1 + radius**2 / ra**2)
    coefficient = 1 / (ra**2 * h * (h + 1))
    assert sp.simplify(coefficient - (1 - 1 / h) / radius**2) == 0
    assert sp.simplify(1 / h + coefficient * radius**2) == 1
    assert sp.simplify(1 - 1 / h**2 - radius**2 / (radius**2 + ra**2)) == 0
    assert sp.limit(coefficient, radius, 0) == 1 / (2 * ra**2)
