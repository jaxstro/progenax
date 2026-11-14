"""Escape-boundary speed scaling, checked against the analytic beta limit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import betaincinv

from progenax.kinematics.limepy_df import _sample_speed_angle, _sample_unit_speed
from progenax.profiles.limepy_tables import AnisoSpeedCDFTable, SpeedCDFTable


@pytest.mark.parametrize("g", [0.0, 0.5, 1.0, 2.5, 3.5])
def test_positive_potential_retains_beta_limit_and_scale_derivative(g):
    # As W -> 0 at fixed p, x^2 follows Beta(3/2, g+1), x=u/sqrt(2W).
    probabilities = jnp.array([0.05, 0.3, 0.8, 0.95])
    expected = np.sqrt(betaincinv(1.5, g + 1, np.asarray(probabilities)))
    iso = SpeedCDFTable.build(10.0, g)
    aniso = AnisoSpeedCDFTable.build(10.0, 10.0, g)
    key = jax.random.PRNGKey(31)
    direct_q = float(jax.random.uniform(key))
    angle_q = float(jax.random.uniform(jax.random.split(key)[0]))
    for W in [1e-10, 1e-8, 1e-6]:
        for draw in [
            lambda w, q: iso.inverse(w, q),
            lambda w, q: aniso.inverse(w, 0.0, q),
        ]:
            value = jax.vmap(lambda q: draw(W, q))(probabilities)
            derivative = jax.vmap(lambda q: jax.grad(lambda w: draw(w, q))(W))(
                probabilities
            )
            np.testing.assert_allclose(value / np.sqrt(2 * W), expected, rtol=0.01)
            np.testing.assert_allclose(derivative * np.sqrt(2 * W), expected, rtol=0.01)
            assert float(draw(0.0, 0.3)) == 0.0
            assert np.isfinite(float(jax.grad(lambda w: draw(w, 0.3))(0.0)))
        u = _sample_unit_speed(key, W, g, 256)
        vr, vt = _sample_speed_angle(key, W, 0.0, g, 256, 128)
        np.testing.assert_allclose(
            u / np.sqrt(2 * W), np.sqrt(betaincinv(1.5, g + 1, direct_q)), rtol=0.01
        )
        np.testing.assert_allclose(
            jnp.hypot(vr, vt) / np.sqrt(2 * W),
            np.sqrt(betaincinv(1.5, g + 1, angle_q)),
            rtol=0.01,
        )


@pytest.mark.parametrize("g", [0.0, 0.5, 1.0, 2.5, 3.5])
def test_anisotropic_moments_against_independent_integration(g):
    from scipy.integrate import quad
    from scipy.special import dawsn, gammainc

    from progenax.profiles.limepy_multimass import _aniso_v2hat_scalar

    table = AnisoSpeedCDFTable.build(11.0, 200.0, g)
    nodes, weights = np.polynomial.legendre.leggauss(512)
    probabilities = jnp.asarray((nodes + 1) / 2)
    weights = jnp.asarray(weights / 2)
    for W, p in [(1e-10, 200.0), (1e-3, 200.0), (10.0, 200.0)]:
        b = p * np.sqrt(W)

        def integral(k):
            def weight(t):
                z = W * (1 - t * t)
                a = b * t
                T = 2 * dawsn(a) / a if a else 2.0
                return (
                    t ** (2 + k)
                    * np.exp(z)
                    * (gammainc(g, z) / W**g if g > 0 else 1.0)
                    * T
                )

            return quad(
                weight,
                0,
                1,
                points=[a / b for a in [1, 4, 16, 64] if b > a],
                epsabs=1e-14,
                epsrel=1e-11,
            )[0]

        expected = np.array(
            [
                np.sqrt(2 * W) * integral(1) / integral(0),
                2 * W * integral(2) / integral(0),
            ]
        )
        u = jax.jit(jax.vmap(lambda q: table.inverse(W, p, q)))(probabilities)
        measured = jnp.array([jnp.sum(weights * u), jnp.sum(weights * u * u)])
        # Retain the established speed-moment accuracy budget (1.5%).
        np.testing.assert_allclose(measured, expected, rtol=0.015, atol=0)
        np.testing.assert_allclose(
            _aniso_v2hat_scalar(W, p, g), expected[1], rtol=0.015, atol=0
        )
