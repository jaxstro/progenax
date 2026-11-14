"""Independent registered expressions versus hand-written Plummer kernels."""

import jax
import jax.numpy as jnp
import pytest
import sympy as sp
from jaxstro.registry import load_source
from jaxstro.registry.symbolic import build_equation, lambdify_jax, partial_derivative

from progenax.profiles import PlummerProfile
from progenax.registry import registry_root


@pytest.fixture(scope="module")
def equations():
    source = load_source(registry_root(), "plummer1911distribution")
    derived = load_source(registry_root(), "progenaxderivations")
    return (
        build_equation(source, "EQ-PLUMMER-SHAPE"),
        build_equation(derived, "EQ-PLUMMER-DENSITY"),
        build_equation(derived, "EQ-PLUMMER-CDF"),
    )


def test_normalization_and_cdf_identity(equations):
    """Mass continuity plus endpoints establishes normalization without quadrature."""
    _, density, cdf = equations
    r, a, mass = (density.symbols[n] for n in ("r", "a", "M"))
    assert (
        sp.simplify(
            sp.diff(cdf.expression, r) - 4 * sp.pi * r**2 * density.expression / mass
        )
        == 0
    )
    assert cdf.expression.subs(r, 0) == 0
    assert sp.limit(cdf.expression, r, sp.oo) == 1
    assert (
        sp.simplify(
            cdf.expression.subs(r, a / sp.sqrt(2 ** sp.Rational(2, 3) - 1))
            - sp.Rational(1, 2)
        )
        == 0
    )


@pytest.mark.parametrize("r_h", [0.5, 1.0, 2.5, 7.0])
def test_registered_shape_cdf_and_radial_derivative(equations, r_h):
    """Use existing scale-test rtol=1e-6; this is not a new accuracy claim."""
    shape, _, cdf = equations
    profile = PlummerProfile(r_h=r_h)
    radii = profile.a * jnp.array([0.0, 0.5, 1.0, 2.0, 3.0])
    assert jnp.allclose(
        profile.density(radii), lambdify_jax(shape)(x=radii / profile.a), rtol=1e-6
    )
    assert jnp.allclose(
        profile.enclosed_mass_fraction(radii),
        lambdify_jax(cdf)(r=radii, a=profile.a),
        rtol=1e-6,
    )
    ad = jax.vmap(jax.grad(profile.enclosed_mass_fraction))(radii)
    symbolic = lambdify_jax(cdf, partial_derivative(cdf, "r"))(r=radii, a=profile.a)
    assert jnp.allclose(ad, symbolic, rtol=1e-6)


@pytest.fixture(scope="module")
def sampler_equations():
    derived = load_source(registry_root(), "progenaxderivations")
    return tuple(
        build_equation(derived, name)
        for name in ("EQ-PLUMMER-SCALE", "EQ-PLUMMER-INVERSE-CDF", "EQ-PLUMMER-CDF")
    )


def test_inverse_and_scale_identities(sampler_equations):
    scale, inverse, cdf = sampler_equations
    u, a = inverse.symbols["u"], inverse.symbols["a"]
    rh, r = scale.symbols["r_h"], cdf.symbols["r"]
    # u=(t²/(1+t²))^(3/2) spans (0,1) for positive t. This encodes the
    # bounded domain explicitly without assuming u is unbounded positive.
    t = sp.Symbol("t", positive=True)
    interior_u = (t**2 / (1 + t**2)) ** sp.Rational(3, 2)
    radius = sp.simplify(inverse.expression.subs(u, interior_u))
    assert sp.simplify(radius - a * t) == 0
    assert sp.simplify(cdf.expression.subs(r, radius) - interior_u) == 0
    assert (
        sp.simplify(
            cdf.expression.subs({r: rh, a: scale.expression}) - sp.Rational(1, 2)
        )
        == 0
    )
    scaled = inverse.expression.subs(a, scale.expression)
    assert sp.simplify(sp.diff(scaled, rh) - scaled / rh) == 0


@pytest.mark.parametrize("rh", [0.5, 1.0, 2.5, 7.0])
def test_sampler_scale_and_clipping(sampler_equations, rh):
    scale, inverse, _ = sampler_equations
    assert jax.config.jax_enable_x64
    u = jnp.array([0.01, 0.25, 0.5, 0.75, 0.99], dtype=jnp.float64)
    profile = PlummerProfile(r_h=rh)
    assert jnp.allclose(profile.a, lambdify_jax(scale)(r_h=rh), rtol=1e-6, atol=1e-8)
    oracle = lambdify_jax(inverse)(u=u, a=profile.a)
    for sampler in (
        lambda h, x: PlummerProfile(r_h=h)._sample_radii(x),
        jax.jit(lambda h, x: PlummerProfile(r_h=h)._sample_radii(x)),
    ):
        radii, tangent = jax.jvp(
            lambda h: sampler(h, u), (jnp.asarray(rh),), (jnp.array(1.0),)
        )
        assert jnp.allclose(radii, oracle, rtol=1e-6, atol=1e-8)
        assert jnp.allclose(
            profile.enclosed_mass_fraction(radii), u, rtol=1e-6, atol=1e-8
        )
        assert jnp.allclose(tangent, radii / rh, rtol=1e-6, atol=1e-8)
        eps = 1e-7
        edge = jnp.array([0.0, eps / 2, 2 * eps, 1 - 2 * eps, 1 - eps / 2, 1.0])
        values, dh = jax.jvp(
            lambda h: sampler(h, edge), (jnp.asarray(rh),), (jnp.array(1.0),)
        )
        expected = lambdify_jax(inverse)(u=jnp.clip(edge, eps, 1 - eps), a=profile.a)
        assert jnp.allclose(values, expected, rtol=1e-6, atol=1e-8)
        assert jnp.allclose(dh, values / rh, rtol=1e-6, atol=1e-8)
        du = jax.jvp(lambda x: sampler(rh, x), (edge,), (jnp.ones_like(edge),))[1]
        assert jnp.all(du[jnp.array([0, 1, 4, 5])] == 0)
    masses = jnp.ones(16, dtype=jnp.float64)
    key = jax.random.PRNGKey(42)
    for draw in (
        lambda h: PlummerProfile(r_h=h).sample_positions(masses, key),
        jax.jit(lambda h: PlummerProfile(r_h=h).sample_positions(masses, key)),
    ):
        positions, derivative = jax.jvp(draw, (jnp.asarray(rh),), (jnp.array(1.0),))
        assert jnp.allclose(derivative, positions / rh, rtol=1e-6, atol=1e-8)
