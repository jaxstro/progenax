r"""Gaussianization: the analytic log-density 2-point xi_s from the BM19 copula map.

Differentiate the PREDICTED statistic, not the stochastic simulator. The copula map

    s = T(g) = bm19_icdf_analytic(Phi(g)) - log<e^s>

carries the BM19 marginal onto a unit Gaussian field ``g``; its probabilists'-Hermite
expansion gives the analytic log-density 2-point (added in later tasks)

    xi_s(r) = sum_{n>=1} (c_n^2 / n!) rho_g(r)^n ,   c_n = <T(g) He_n(g)> .

Grounding (docs/plans/2026-06-05-gaussianization-formula-verification.md):
  - mean-1 shift <e^s> = 1 .................... Coles & Jones 1991, Eq (21)
  - 2-pt series (Mehler bivariate-Hermite); reduces to C&J Eq (30) 1+xi=exp[Xi]
    in the exp/lognormal case
  - c_n = <T(g) He_n(g)>, probabilists' Hermite

JAX-native; differentiable in (mach, b, alpha). The Gaussian field ``g`` is held fixed
(its arrangement carries beta; only the marginal values depend on mach,b,alpha).
"""

import jax.numpy as jnp
from jax.scipy.special import erf
from jaxtyping import Array, Float

from gravoturb_fdf.theory.pdf import bm19_icdf_analytic, bm19_mean_density


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

    ``s = bm19_icdf_analytic(Phi(g); M,b,alpha) - log(bm19_mean_density)``. The
    subtractive shift = ``log<e^s>`` enforces the rho0 convention (population
    <e^s> = 1; Coles & Jones 1991 Eq 21). The additive shift does NOT affect the
    Hermite coefficients c_n for n >= 1 (they are orthogonal to constants), so the
    2-point ``xi_s`` is shift-invariant; the shift only fixes the marginal's mean.

    Differentiable in (mach, b, alpha); ``g`` is held fixed.
    """
    u = _standard_normal_cdf(g)
    s_raw = bm19_icdf_analytic(u, mach, b, alpha)
    shift = jnp.log(bm19_mean_density(mach, b, alpha))
    return s_raw - shift
