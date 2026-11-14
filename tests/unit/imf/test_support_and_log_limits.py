"""Finite-support axioms, the Chabrier log limit, and the exact taper law."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

from progenax.imf import ChabrierIMF, Maschberger, TaperedPowerLaw
from progenax.imf.base import BaseIMF


def _support(alpha, mass):
    model = Maschberger(alpha=alpha)
    return model.cdf(mass), model.logpdf(mass)


_support_check = jax.jit(
    lambda a, m: (
        _support(a, m),
        jax.jacfwd(lambda x: _support(a, x)[0])(m),
        jax.jacrev(lambda x: _support(x, m)[0])(a),
    )
)


def test_base_imf_has_finite_support_in_values_and_derivatives():
    """Catch nonzero probability/density outside the stated truncated mass interval."""
    mass = jnp.array([-1.0, 0.001, 0.01, 0.1, 300.0, 1000.0])
    (cdf, logpdf), dmass, dalpha = _support_check(2.3, mass)
    outside = np.array([0, 1, 5])
    np.testing.assert_array_equal(np.asarray(cdf)[outside], [0, 0, 1])
    assert np.all(np.isneginf(np.asarray(logpdf)[outside]))
    np.testing.assert_array_equal(np.asarray(dmass)[outside], np.zeros((3, 6)))
    np.testing.assert_array_equal(np.asarray(dalpha)[outside], np.zeros(3))


def test_support_masks_preserve_nan_input_and_model_failures():
    """Catch support masks hiding invalid NaN inputs or parameters as zero probability."""
    (cdf, logpdf), _, _ = _support_check(jnp.nan, jnp.array([0.001, 0.1, 1000.0]))
    assert np.all(np.isnan(cdf)) and np.all(np.isnan(logpdf))
    (cdf, logpdf), _, _ = _support_check(2.3, jnp.array([jnp.nan]))
    assert np.all(np.isnan(cdf)) and np.all(np.isnan(logpdf))


def _chabrier_cdf(params):
    return ChabrierIMF(m_c=params[0], sigma=params[1], alpha=params[2]).cdf(
        jnp.array(0.5)
    )


_chabrier_check = jax.jit(
    lambda p: (
        _chabrier_cdf(p),
        jax.jacfwd(_chabrier_cdf)(p),
        jax.jacrev(_chabrier_cdf)(p),
    )
)


def test_chabrier_constructor_and_alpha_one_use_analytic_cdf_derivatives():
    """Catch traced-bool validation and lost/NaN derivatives at the alpha=1 log limit."""
    mc, sigma, alpha = 0.08, 0.69, 1.0
    value, forward, reverse = _chabrier_check(jnp.array([mc, sigma, alpha]))
    coefficient = 0.158

    def integral_and_slopes(upper):
        low = (np.log10(0.08) - np.log10(mc)) / (np.sqrt(2) * sigma)
        high = (np.log10(upper) - np.log10(mc)) / (np.sqrt(2) * sigma)
        integral = coefficient * sigma * np.sqrt(np.pi / 2) * (erf(high) - erf(low))
        dmc = (
            -coefficient / (mc * np.log(10)) * (np.exp(-(high**2)) - np.exp(-(low**2)))
        )
        dsigma = integral / sigma + coefficient * np.sqrt(2) * (
            low * np.exp(-(low**2)) - high * np.exp(-(high**2))
        )
        return integral, np.array([dmc, dsigma])

    part, dpart = integral_and_slopes(0.5)
    whole, dwhole = integral_and_slopes(1.0)
    logmc = -np.log10(mc)
    tail = coefficient / np.log(10) * np.exp(-(logmc**2) / (2 * sigma**2))
    dtail = tail * np.array([logmc / (mc * np.log(10) * sigma**2), logmc**2 / sigma**3])
    length = np.log(100)
    norm = whole + tail * length
    expected = part / norm
    derivatives = np.concatenate(
        (
            (dpart * norm - part * (dwhole + dtail * length)) / norm**2,
            [expected * tail * length**2 / (2 * norm)],
        )
    )
    np.testing.assert_allclose(value, expected, rtol=2e-14)
    np.testing.assert_allclose(forward, derivatives, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(reverse, derivatives, rtol=2e-12, atol=2e-14)


def _chabrier_ppf(alpha):
    return ChabrierIMF(alpha=alpha).ppf(jnp.array([0.2, 0.8]))


_quantile_check = jax.jit(
    lambda alpha: (
        _chabrier_ppf(alpha),
        jax.jacfwd(_chabrier_ppf)(alpha),
        jax.jacrev(_chabrier_ppf)(alpha),
    )
)


def test_chabrier_log_limit_quantiles_and_implicit_derivatives():
    """Catch a singular power-law initial guess or incorrect AD through the PPF solve."""
    value, forward, reverse = map(np.asarray, _quantile_check(1.0))
    mc, width, amplitude = 0.08, 0.69, 0.158
    scale = amplitude * width * np.sqrt(np.pi / 2)
    lower = (np.log10(0.08) - np.log10(mc)) / (width * np.sqrt(2))

    def low_integral(mass):
        upper = (np.log10(mass) - np.log10(mc)) / (width * np.sqrt(2))
        return scale * (erf(upper) - erf(lower))

    tail = amplitude / np.log(10) * np.exp(-(np.log10(mc) ** 2) / (2 * width**2))
    length = np.log(100)
    low_total = low_integral(1.0)
    norm = low_total + tail * length

    def independent_cdf(mass):
        return (
            low_integral(mass) if mass < 1 else low_total + tail * np.log(mass)
        ) / norm

    reference = np.array(
        [
            brentq(lambda m: independent_cdf(m) - u, 0.08, 100, xtol=5e-14)
            for u in [0.2, 0.8]
        ]
    )
    pdf = (
        np.where(
            reference < 1,
            amplitude
            / (reference * np.log(10))
            * np.exp(-(np.log10(reference / mc) ** 2) / (2 * width**2)),
            tail / reference,
        )
        / norm
    )
    partial = (
        tail
        / (2 * norm)
        * (
            np.array([0.2, 0.8]) * length**2
            - np.where(reference >= 1, np.log(reference) ** 2, 0)
        )
    )
    expected_derivative = -partial / pdf
    np.testing.assert_allclose(value, reference, rtol=3e-12, atol=1e-13)
    np.testing.assert_allclose(forward, expected_derivative, rtol=3e-11, atol=1e-13)
    np.testing.assert_allclose(reverse, expected_derivative, rtol=3e-11, atol=1e-13)


_taper_check = jax.jit(
    jax.vmap(
        lambda mass: (
            TaperedPowerLaw()._logpdf_unnorm(mass),
            jax.grad(TaperedPowerLaw()._logpdf_unnorm)(mass),
        )
    )
)


def test_taper_law_and_mass_derivative_are_continuous_at_old_switch():
    """Catch the t<.01 approximation's 0.5% density jump and altered derivative."""
    masses = np.array([0.0101, 0.0299999997, 0.03, 0.0300000003, 0.3, 3.0])
    value, derivative = _taper_check(jnp.asarray(masses))
    t = (masses / 0.3) ** 2
    expected = -2.3 * np.log(masses) + np.log(-np.expm1(-t))
    slope = -2.3 / masses + 2 * t / (masses * np.expm1(t))
    np.testing.assert_allclose(value, expected, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(derivative, slope, rtol=2e-13, atol=2e-13)


class _UniformMass(BaseIMF):
    m_min: float = 0.1
    m_max: float = 100.0

    def _logpdf_unnorm(self, m):
        return jnp.zeros_like(m)

    def _cdf_unnorm(self, m):
        return jnp.asarray(m)


_uniform_mean = jax.jit(lambda: _UniformMass().mean_mass())


def test_mean_quadrature_keeps_exact_mass_support_endpoints():
    """Catch exp(log(bound)) roundoff moving the last node outside a masked PDF."""
    # The trapezoid rule integrates the uniform law's linear first moment exactly.
    np.testing.assert_allclose(_uniform_mean(), (0.1 + 100) / 2, rtol=8e-15, atol=0)
