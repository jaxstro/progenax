"""Independent angular integral and Cartesian OM implementation agreement."""

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jaxstro.registry import load_source
from jaxstro.registry.symbolic import build_equation

from progenax.kinematics.eddington import assign_om_directions
from progenax.registry import registry_root


def test_angular_factor_integral_limits_and_kernel(monkeypatch):
    record = build_equation(
        load_source(registry_root(), "progenaxderivations"), "EQ-OM-SPEED-MOMENT-FACTOR"
    )
    r, ra = record.symbols["r"], record.symbols["r_a"]
    mu = sp.Symbol("mu", real=True)
    integrand = mu**2 + (1 - mu**2) / (1 + (r / ra) ** 2)
    assert (
        sp.simplify(sp.integrate(integrand, (mu, -1, 1)) / 2 - record.expression) == 0
    )
    assert sp.limit(record.expression, ra, sp.oo) == 1
    assert sp.limit(record.expression, r, sp.oo) == sp.Rational(1, 3)
    # Three orthogonal unit directions integrate this quadratic exactly.
    monkeypatch.setattr(jax.random, "normal", lambda key, shape: jnp.eye(3))

    def kernel(radius):
        positions = jnp.tile(jnp.array([radius, 0.0, 0.0]), (3, 1))
        velocity = assign_om_directions(
            jax.random.PRNGKey(0), positions, jnp.ones(3), 3.0
        )
        return jnp.mean(jnp.sum(velocity**2, axis=1))

    oracle = sp.lambdify(r, record.expression.subs(ra, 3), "numpy")
    derivative = sp.lambdify(r, sp.diff(record.expression, r).subs(ra, 3), "numpy")
    for radius in (0.0, 0.3, 3.0, 30.0):
        np.testing.assert_allclose(
            kernel(radius), oracle(radius), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            jax.grad(kernel)(radius), derivative(radius), rtol=1e-12, atol=1e-12
        )


def test_spherical_registry_satisfies_shell_theorem():
    source = load_source(registry_root(), "progenaxderivations")
    mass = build_equation(source, "EQ-SPHERICAL-ENCLOSED-MASS")
    potential = build_equation(source, "EQ-SPHERICAL-POTENTIAL")
    r, G, rt = (potential.symbols[x] for x in ("r", "G", "r_t"))
    assert (
        sp.simplify(sp.diff(potential.expression, r) - G * mass.expression / r**2) == 0
    )
    edge = potential.expression.subs(r, rt)
    assert sp.simplify(edge + G * mass.expression.subs(r, rt) / rt) == 0


def test_eddington_registry_inverts_density_and_boundary_term():
    record = build_equation(
        load_source(registry_root(), "progenaxderivations"), "EQ-EDDINGTON-INVERSION"
    )
    E, psi, boundary = (record.symbols[x] for x in ("E", "Psi", "rho_prime_zero"))
    for power in (1, 5):
        inverted = record.expression.subs(
            {sp.Function("rho")(psi): psi**power, boundary: 1 if power == 1 else 0}
        ).doit()
        reconstructed = (
            4
            * sp.sqrt(2)
            * sp.pi
            * sp.integrate(inverted * sp.sqrt(psi - E), (E, 0, psi))
        )
        assert sp.simplify(reconstructed - psi**power) == 0
