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
import numpy as np  # constants only: Gauss-Hermite nodes/weights at import time
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
    # Clip u away from {0,1}: the outer Gauss-Hermite quadrature nodes reach |g|~30,
    # where Phi(g) saturates to exactly 0 or 1 in float64 and bm19_icdf_analytic hits
    # its tail singularity (log(0) -> +-inf). Those nodes carry ~e^{-450} weight, so the
    # clip's bias is negligible; delta=1e-10 keeps the tail-icdf cancellation accurate to
    # ~6 digits (so param-gradients stay bounded). Finite-sample realizations (|g|<~5)
    # never trigger the clip.
    u = jnp.clip(_standard_normal_cdf(g), 1e-10, 1.0 - 1e-10)
    s_raw = bm19_icdf_analytic(u, mach, b, alpha)
    shift = jnp.log(bm19_mean_density(mach, b, alpha))
    return s_raw - shift


def _gauss_hermite(n_quad: int):
    r"""Probabilists' Gauss-Hermite rule for <f>_phi = int f(g) phi(g) dg.

    Returns ``(g_nodes, weights)`` as jnp constants so that
    ``<f>_phi ~ sum_i weights_i f(g_nodes_i)`` via the substitution ``g = sqrt(2) x``
    on the physicists' rule (weight ``e^{-x^2}``): ``weights_i = w_i / sqrt(pi)``.

    Nodes/weights are CONSTANTS (numpy at import time, frozen to jnp); the quadrature
    sum itself is pure JAX, so coefficients stay differentiable in the map's params.
    """
    x, w = np.polynomial.hermite.hermgauss(n_quad)
    return jnp.asarray(np.sqrt(2.0) * x), jnp.asarray(w / np.sqrt(np.pi))


def _hermite_e_basis(g: Float[Array, " q"], n_max: int) -> Float[Array, " n q"]:
    r"""Probabilists' Hermite He_0..He_{n_max} at points ``g``; shape ``(n_max+1, q)``.

    Stable recurrence ``He_{n+1} = g He_n - n He_{n-1}``, ``He_0 = 1``, ``He_1 = g``.
    """
    rows = [jnp.ones_like(g)]
    if n_max >= 1:
        rows.append(g)
    for n in range(1, n_max):
        rows.append(g * rows[n] - n * rows[n - 1])
    return jnp.stack(rows, axis=0)


def hermite_coefficients(map_fn, n_max: int, n_quad: int = 256) -> Float[Array, " n"]:
    r"""Hermite coefficients ``c_n = <map_fn(g) He_n(g)>`` for n=0..n_max (probabilists').

    Computed by Gauss-Hermite quadrature; ``c_n`` is differentiable in any parameters
    that ``map_fn`` closes over. Returns shape ``(n_max+1,)``; the 2-point series
    ``xi_s`` uses n >= 1 (n=0 is the mean ``<map_fn>``).
    """
    g_nodes, weights = _gauss_hermite(n_quad)
    values = map_fn(g_nodes)
    he = _hermite_e_basis(g_nodes, n_max)
    return (he * (values * weights)[None, :]).sum(axis=1)


def bm19_hermite_coefficients(
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    n_max: int,
    n_quad: int = 256,
) -> Float[Array, " n"]:
    r"""Hermite coefficients of the BM19 copula map ``s_of_g(.; mach,b,alpha)``."""
    return hermite_coefficients(lambda g: s_of_g(g, mach, b, alpha), n_max, n_quad)
