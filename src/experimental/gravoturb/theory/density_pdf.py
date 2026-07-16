"""BM19 1D gravoturbulent density-PDF theory.

Burkhart & Mocz 2019, ApJ 879, 129 ("The Self-gravitating Gas Fraction and the
Critical Density for Star Formation"). Equation numbers refer to that paper,
verified against the held PDF (docs/core-papers/Burkhart_2019_ApJ_879_129.pdf).

The density PDF is piecewise: a mass-conserving lognormal body (width sigma_s set
by the sonic Mach number) plus a powerlaw tail for s >= s_t, where
s = ln(rho/rho_0) and alpha is the PDF powerlaw slope.

JAX-native and differentiable in (mach, b, alpha).
"""

import jax.numpy as jnp
from jax.scipy.special import erf, erfc
from jaxtyping import Array, Float


def sigma_s_squared(
    mach: Float[Array, ""], b: Float[Array, ""]
) -> Float[Array, ""]:
    r"""Lognormal width of the density PDF (BM19 Eq. 1).

    .. math:: \sigma_s^2 = \ln(1 + b^2 \mathcal{M}^2)

    Parameters
    ----------
    mach : sonic Mach number M_s = sigma_v / c_s (dimensionless).
    b : turbulence driving parameter, b in [1/3, 1] (solenoidal -> compressive).

    Returns
    -------
    sigma_s^2, the variance of s = ln(rho/rho_0) (dimensionless).
    """
    return jnp.log(1.0 + (b * mach) ** 2)


def transition_density(
    alpha: Float[Array, ""], sigma_s_sq: Float[Array, ""]
) -> Float[Array, ""]:
    r"""Lognormal->powerlaw transition density (BM19 Eq. 2).

    .. math:: s_t = (\alpha - 1/2)\,\sigma_s^2

    This is *derived*, not a free parameter: it is the density where the Jeans
    length equals the sonic length (a mathematically motivated critical density).
    For alpha = 3/2 it reduces to s_t = sigma_s^2 (BM19 Eq. 16).

    Parameters
    ----------
    alpha : PDF powerlaw-tail slope (p_PL(s) ~ e^{-alpha s}); radial slope kappa=3/alpha.
    sigma_s_sq : lognormal variance from :func:`sigma_s_squared`.
    """
    return (alpha - 0.5) * sigma_s_sq


def pdf_slope_to_radial(alpha: Float[Array, ""]) -> Float[Array, ""]:
    r"""Radial density slope kappa from the PDF powerlaw slope alpha (BM19 §2).

    A spherical profile rho ~ r^{-kappa} produces a density-PDF powerlaw
    p(s) ~ e^{-alpha s} with kappa = 3/alpha (Kainulainen & Federrath 2017).
    """
    return 3.0 / alpha


def _lognormal_dense_z(
    sigma_s_sq: Float[Array, ""], s_t: Float[Array, ""]
) -> Float[Array, ""]:
    r"""erf/erfc argument z = (s_t - sigma_s^2/2)/(sqrt2 sigma_s) (BM19 Eq. 20)."""
    sigma_s = jnp.sqrt(sigma_s_sq)
    return (s_t - 0.5 * sigma_s_sq) / (jnp.sqrt(2.0) * sigma_s)


def dense_mass_fraction(
    mach: Float[Array, ""], b: Float[Array, ""], alpha: Float[Array, ""]
) -> Float[Array, ""]:
    r"""Self-gravitating (dense) gas mass fraction (BM19 Eq. 17-20).

    The PDF is a mass-conserving lognormal body (mean s0 = -sigma_s^2/2) joined
    continuously at s_t to a powerlaw tail p_PL(s) = C e^{-alpha s}, with
    C = p_LN(s_t) e^{alpha s_t}. The dense fraction is the powerlaw-tail mass over
    the total:

    .. math:: f_\mathrm{dense} = \frac{M_\mathrm{PL}}{M_\mathrm{LN} + M_\mathrm{PL}}

    with (mass-weighted integrals of Eq. 18)

    .. math::
        M_\mathrm{PL} = \frac{C\,e^{(1-\alpha)s_t}}{\alpha - 1}, \qquad
        M_\mathrm{LN} = \tfrac12\!\left[1 + \mathrm{erf}\!\left(
            \frac{s_t - \sigma_s^2/2}{\sqrt2\,\sigma_s}\right)\right].

    Valid for alpha > 1 (M_PL diverges as alpha -> 1, the limit where the entire
    cloud is self-gravitating). Differentiable in (mach, b, alpha).
    """
    sigma_s_sq = sigma_s_squared(mach, b)
    s0 = -0.5 * sigma_s_sq
    s_t = transition_density(alpha, sigma_s_sq)

    # Continuity at s_t: powerlaw amplitude C = p_LN(s_t) e^{alpha s_t}.
    p_ln_st = jnp.exp(-((s_t - s0) ** 2) / (2.0 * sigma_s_sq)) / jnp.sqrt(
        2.0 * jnp.pi * sigma_s_sq
    )
    C = p_ln_st * jnp.exp(alpha * s_t)

    M_PL = C * jnp.exp((1.0 - alpha) * s_t) / (alpha - 1.0)
    M_LN = 0.5 * (1.0 + erf(_lognormal_dense_z(sigma_s_sq, s_t)))
    return M_PL / (M_LN + M_PL)


def dense_mass_fraction_lognormal(
    mach: Float[Array, ""], b: Float[Array, ""], alpha: Float[Array, ""]
) -> Float[Array, ""]:
    r"""Dense-mass fraction of a *pure* lognormal above s_t (BM19 comparison form).

    .. math:: f_\mathrm{dense}^{LN} = \tfrac12\,\mathrm{erfc}\!\left(
        \frac{s_t - \sigma_s^2/2}{\sqrt2\,\sigma_s}\right)

    This is the dense fraction if there were no powerlaw tail. The full BM19
    f_dense exceeds it because the (shallower) powerlaw adds high-density mass.
    """
    sigma_s_sq = sigma_s_squared(mach, b)
    s_t = transition_density(alpha, sigma_s_sq)
    return 0.5 * erfc(_lognormal_dense_z(sigma_s_sq, s_t))
