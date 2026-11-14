"""Analytic conditional-CDF oracle and Cartesian full-sampler derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from jaxstro.registry import load_source
from jaxstro.registry.symbolic import build_equation
from scipy.special import beta, betainc

from progenax.kinematics.plummer_df import PlummerVelocityDF, _om_speed_quantile
from progenax.registry import registry_root


def test_registered_om_cdf_derivative_and_normalization():
    source = load_source(registry_root(), "progenaxderivations")
    pdf = build_equation(source, "EQ-PLUMMER-OM-SPEED-PDF")
    cdf = build_equation(source, "EQ-PLUMMER-OM-SPEED-CDF")
    q = cdf.symbols["q_om"]
    assert sp.simplify(sp.diff(cdf.expression, q) - pdf.expression) == 0
    # Independently integrate the two powers using their beta normalizations.
    for b, exact in [
        (sp.Rational(9, 2), 7 * sp.pi / 256),
        (sp.Rational(5, 2), sp.pi / 16),
    ]:
        assert (
            sp.simplify(
                sp.gamma(sp.Rational(3, 2))
                * sp.gamma(b)
                / sp.gamma(b + sp.Rational(3, 2))
                - exact
            )
            == 0
        )
    assert cdf.expression.subs(q, 0).doit() == 0


@pytest.mark.parametrize("ra_over_a", [0.75, 1.0, 2.0, 10.0])
def test_om_quantiles_against_independent_cdf(ra_over_a):
    assert jax.config.jax_enable_x64
    # Stratified deterministic uniforms include both tails, not Monte Carlo draws.
    uniforms = jnp.asarray(
        np.unique(
            np.r_[
                np.linspace(1e-6, 1 - 1e-6, 1025),
                np.geomspace(1e-9, 1e-3, 32),
                1 - np.geomspace(1e-9, 1e-3, 32),
            ]
        )
    )
    rho = 1 / ra_over_a**2
    for radius in (0.0, 1.0, 10.0, 100.0, 1000.0):
        psi = 6 / np.sqrt(1 + radius**2)
        speeds = jax.vmap(lambda u: _om_speed_quantile(u, psi, rho))(uniforms)
        z = np.asarray(speeds) ** 2 / (2 * psi)
        A = (1 - rho) * psi**2 * beta(1.5, 4.5)
        B = 15.75 * rho * beta(1.5, 2.5)
        actual = (A * betainc(1.5, 4.5, z) + B * betainc(1.5, 2.5, z)) / (A + B)
        assert np.max(abs(actual - np.asarray(uniforms))) < 1e-4


@pytest.mark.parametrize("ra", [None, 1.5])
def test_full_sampler_cartesian_origin_and_nearby_derivatives(ra):
    df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=ra)
    key = jax.random.PRNGKey(42)

    def draw(x):
        return df.sample_velocities(x[None, :], jnp.ones(1), key, G=1.0)[0]

    for run in (draw, jax.jit(draw)):
        zero = jnp.zeros(3)
        assert np.isfinite(run(zero)).all()
        np.testing.assert_allclose(jax.jacfwd(run)(zero), 0, atol=1e-8)
        np.testing.assert_allclose(jax.jacrev(run)(zero), 0, atol=1e-8)
        for radius in (1e-4, 0.5, 2.5):
            x = jnp.array([radius, radius / 2, -radius / 3])
            h = 1e-6
            fd = jnp.stack(
                [(run(x + h * e) - run(x - h * e)) / (2 * h) for e in jnp.eye(3)],
                axis=1,
            )
            np.testing.assert_allclose(jax.jacrev(run)(x), fd, rtol=1e-5, atol=1e-8)
