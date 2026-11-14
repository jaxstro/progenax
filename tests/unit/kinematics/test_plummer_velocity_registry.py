"""Independent symbolic and fixed-randomness isotropic velocity checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from jaxstro.registry import load_source
from jaxstro.registry.symbolic import build_equation, lambdify_jax

from progenax.kinematics import PlummerVelocityDF
from progenax.registry import registry_root


@pytest.fixture(scope="module")
def records():
    source = load_source(registry_root(), "progenaxderivations")
    return tuple(
        build_equation(source, name)
        for name in (
            "EQ-PLUMMER-SCALE",
            "EQ-PLUMMER-ESCAPE-SPEED",
            "EQ-PLUMMER-SPEED-PDF",
        )
    )


def test_symbolic_speed_normalization_moments_and_scales(records):
    scale, escape, pdf = records
    q = pdf.symbols["q"]
    z = sp.Symbol("z", positive=True)
    # q=sqrt(z), dq=dz/(2sqrt(z)): derive the Beta density from the record.
    transformed = sp.simplify(pdf.expression.subs(q, sp.sqrt(z)) / (2 * sp.sqrt(z)))
    norm = sp.gamma(6) / (sp.gamma(sp.Rational(3, 2)) * sp.gamma(sp.Rational(9, 2)))
    assert (
        sp.simplify(
            transformed - norm * z ** sp.Rational(1, 2) * (1 - z) ** sp.Rational(7, 2)
        )
        == 0
    )
    moments = [
        sp.simplify(
            norm
            * sp.gamma(sp.Rational(3, 2) + k)
            * sp.gamma(sp.Rational(9, 2))
            / sp.gamma(6 + k)
        )
        for k in (0, 1, 2)
    ]
    assert moments == [1, sp.Rational(1, 4), sp.Rational(5, 56)]
    assert sp.simplify(moments[2] - moments[1] ** 2) == sp.Rational(3, 112)
    a, r, G, M = (escape.symbols[n] for n in ("a", "r", "G", "M"))
    h = scale.symbols["r_h"]
    speed = escape.expression.subs(a, scale.expression)
    assert (
        sp.simplify(
            sp.diff(speed, h) / speed
            + scale.expression**2 / (2 * h * (r**2 + scale.expression**2))
        )
        == 0
    )
    for parameter in (G, M):
        assert sp.simplify(sp.diff(speed, parameter) - speed / (2 * parameter)) == 0
    lam = sp.Symbol("lam", positive=True)
    assert (
        sp.simplify(
            speed.subs({r: lam * r, h: lam * h}, simultaneous=True)
            - speed / sp.sqrt(lam)
        )
        == 0
    )


@pytest.mark.parametrize("rh", [0.5, 1.0, 2.5, 7.0])
def test_fixed_draw_velocity_scales(records, rh):
    assert jax.config.jax_enable_x64
    scale, escape, _ = records
    key = jax.random.PRNGKey(42)
    # Fixed nonzero positions isolate scale derivatives from the norm cusp at x=0.
    positions = jnp.arange(1, 49, dtype=jnp.float64).reshape(16, 3) / 16
    masses = jnp.ones(16, dtype=jnp.float64)
    radii = jnp.linalg.norm(positions, axis=1)
    h = jnp.asarray(rh, dtype=jnp.float64)
    one = jnp.asarray(1.0, dtype=jnp.float64)
    a = lambdify_jax(scale)(r_h=h)
    bound = lambdify_jax(escape)(r=radii, a=a, G=one, M=jnp.sum(masses))
    key_mag = jax.random.split(key, 3)[0]
    q = jnp.sqrt(jax.random.beta(key_mag, 1.5, 4.5, shape=(16,)))

    def sample(h, g, mass_scale, length_scale):
        return PlummerVelocityDF(r_h=h).sample_velocities(
            positions * length_scale, masses * mass_scale, key, G=g
        )

    eager = sample(h, one, one, one)
    for sampler in (sample, jax.jit(sample)):
        value, dh = jax.jvp(lambda x: sampler(x, one, one, one), (h,), (one,))
        np.testing.assert_allclose(value, eager, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            jnp.linalg.norm(value, axis=1), q * bound, rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            dh,
            -a * a / (2 * h * (radii * radii + a * a))[:, None] * value,
            rtol=1e-6,
            atol=1e-8,
        )
        for draw in (
            lambda t: sampler(h, t, one, one),
            lambda t: sampler(h, one, t, one),
        ):
            _, derivative = jax.jvp(draw, (one,), (one,))
            np.testing.assert_allclose(derivative, value / 2, rtol=1e-6, atol=1e-8)
        _, derivative = jax.jvp(lambda t: sampler(h * t, one, one, t), (one,), (one,))
        np.testing.assert_allclose(derivative, -value / 2, rtol=1e-6, atol=1e-8)
        # Reuse approved scale values for finite mass, G and homology checks.
        np.testing.assert_allclose(
            sampler(h, h, one, one), jnp.sqrt(h) * value, rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            sampler(h, one, h, one), jnp.sqrt(h) * value, rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            sampler(h * h, one, one, h), value / jnp.sqrt(h), rtol=1e-6, atol=1e-8
        )
