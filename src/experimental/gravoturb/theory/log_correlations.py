r"""Gaussianization: the analytic log-density 2-point xi_s from the BM19 copula map.

Differentiate the PREDICTED statistic, not the stochastic simulator. The copula map

    s = T(g) = log_density_icdf_analytic(Phi(g)) - log<e^s>

carries the BM19 marginal onto a unit Gaussian field ``g``; its probabilists'-Hermite
expansion gives the analytic log-density 2-point (added in later tasks)

    xi_s(r) = sum_{n>=1} (c_n^2 / n!) rho_g(r)^n ,   c_n = <T(g) He_n(g)> .

Grounding:
  - mean-1 shift <e^s> = 1 .................... Coles & Jones 1991, Eq (21)
  - 2-pt series (Mehler bivariate-Hermite); reduces to C&J Eq (30) 1+xi=exp[Xi]
    in the exp/lognormal case
  - c_n = <T(g) He_n(g)>, probabilists' Hermite

JAX-native; differentiable in (mach, b, alpha). The Gaussian field ``g`` is held fixed
(its arrangement carries beta; only the marginal values depend on mach,b,alpha).
"""

import jax.numpy as jnp
from jax.scipy.special import erf, gammaln
from jaxstro.numerics.quadrature import (
    hermite_coefficients as _quadrature_hermite_coefficients,
)
from jaxtyping import Array, Float

from gravoturb.theory.density_cdf import log_density_icdf_analytic, mean_density


def _standard_normal_cdf(g: Float[Array, " ..."]) -> Float[Array, " ..."]:
    r"""Standard-normal CDF Phi(g) = 0.5 (1 + erf(g / sqrt(2)))."""
    return 0.5 * (1.0 + erf(g / jnp.sqrt(2.0)))


def s_of_g(
    g: Float[Array, " ..."],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
) -> Float[Array, " ..."]:
    r"""Copula map T(g): unit Gaussian -> BM19 log-density ``s`` with <e^s> = 1.

    ``s = log_density_icdf_analytic(Phi(g); M,b,alpha) - log(mean_density)``. The
    subtractive shift = ``log<e^s>`` enforces the rho0 convention (population
    <e^s> = 1; Coles & Jones 1991 Eq 21). The additive shift does NOT affect the
    Hermite coefficients c_n for n >= 1 (they are orthogonal to constants), so the
    2-point ``xi_s`` is shift-invariant; the shift only fixes the marginal's mean.

    Differentiable in (mach, b, alpha); ``g`` is held fixed.
    """
    # Clip u away from {0,1}: the outer Gauss-Hermite quadrature nodes reach |g|~30,
    # where Phi(g) saturates to exactly 0 or 1 in float64 and log_density_icdf_analytic hits
    # its tail singularity (log(0) -> +-inf). Those nodes carry ~e^{-450} weight, so the
    # clip's bias is negligible; delta=1e-10 keeps the tail-icdf cancellation accurate to
    # ~6 digits (so param-gradients stay bounded). Finite-sample realizations (|g|<~5)
    # never trigger the clip.
    u = jnp.clip(_standard_normal_cdf(g), 1e-10, 1.0 - 1e-10)
    s_raw = log_density_icdf_analytic(u, mach, b, alpha)
    shift = jnp.log(mean_density(mach, b, alpha))
    return s_raw - shift


# The probabilists' Gauss-Hermite rule, Hermite-e basis, and the generic
# ``hermite_coefficients`` quadrature now live in jaxstro.numerics.quadrature
# (imported above); they are byte-identical to the former local definitions.
# Only the BM19-specific wrappers below remain local (domain-specific maps).


def log_density_hermite_coefficients(
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    n_max: int,
    n_quad: int = 256,
) -> Float[Array, " n"]:
    r"""Hermite coefficients of the BM19 copula map ``s_of_g(.; mach,b,alpha)``."""
    return _quadrature_hermite_coefficients(
        lambda g: s_of_g(g, mach, b, alpha), n_max, n_quad
    )


def density_hermite_coefficients(
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    n_max: int,
    n_quad: int = 256,
) -> Float[Array, " n"]:
    r"""Hermite coefficients of the BM19 DENSITY map ``rho = e^s = exp(s_of_g(.))``.

    ``d_n = <exp(T(g)) He_n(g)>``, ``T(g) = s_of_g(g; mach,b,alpha)`` (probabilists'
    Hermite), the density analog of :func:`log_density_hermite_coefficients` (which expands the
    log-density ``s`` itself). Feeding ``d`` to :func:`gaussianized_xi` gives the EXACT
    BM19 density 2-point (Mehler bivariate-Hermite),
    ``xi_rho(r) = sum_{n>=1} d_n^2/n! rho_g(r)^n`` -- the copula-faithful density 2-pt,
    not the lognormal-limit ``expm1(xi_s)``.

    Two derived invariants follow from the rho0 convention (``<e^s> = 1``):
      - ``d_0 = <e^s> = 1`` (mean density), and
      - ``xi_rho(0) = sum_{n>=1} d_n^2/n! = Var(rho)`` (the density variance).

    Differentiable in (mach, b, alpha) via the Gauss-Hermite quadrature in
    ``jaxstro.numerics.quadrature.hermite_coefficients``; ``g`` is held fixed.
    """
    return _quadrature_hermite_coefficients(
        lambda g: jnp.exp(s_of_g(g, mach, b, alpha)), n_max, n_quad
    )


def gaussianized_xi(
    rho_g: Float[Array, " ..."], c: Float[Array, " n"]
) -> Float[Array, " ..."]:
    r"""Gaussianization 2-point series ``xi_s(rho_g) = sum_{n>=1} (c_n^2/n!) rho_g^n``.

    The Mehler bivariate-Hermite expansion (Szapudi & Pan 2004); it reduces to
    Coles & Jones 1991 Eq (30) ``1+xi = exp[Xi]`` in the exp/lognormal case. ``c`` is
    the :func:`log_density_hermite_coefficients` output (n=0..n_max); the n=0 (mean) term is
    dropped. ``rho_g`` is the normalized Gaussian correlation (``rho_g(0)=1``), scalar
    or array. Differentiable in both ``c`` (hence theta) and ``rho_g`` (hence beta).
    """
    n = jnp.arange(c.shape[0])
    weights = c**2 * jnp.exp(-gammaln(n + 1.0))  # c_n^2 / n!
    weights = weights.at[0].set(0.0)  # drop the n=0 mean term
    powers = jnp.asarray(rho_g)[..., None] ** n  # (..., n_max+1)
    return (powers * weights).sum(axis=-1)
