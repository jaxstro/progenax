"""PP20 magnification factor zeta(p) for the dense-gas star-formation rate.

Parmentier & Pasquali 2020, ApJ 903, 56. The magnification factor quantifies how
much a centrally-concentrated (power-law) density profile boosts the dense-gas SFR
relative to a uniform (top-hat) clump, via the mass-weighted freefall rate:

    zeta = <rho^{1/2}>_mass / <rho>^{1/2}
         = int rho^{3/2} dV * (int dV)^{1/2} / (int rho dV)^{3/2}.

For a pure power-law sphere rho ~ r^{-p} this integrates to (PP20 Eq. 6, embedded
in Eq. 9):

    zeta(p) = (3 - p)^{3/2} / [ (3^{3/2}/2) (2 - p) ].

PP20 prints the constant as the rounded "2.6"; the exact value 3^{3/2}/2 = 2.598...
is fixed by the physical top-hat lower limit zeta(0) = 1 (and yields zeta(1.5)=sqrt2).
Valid on 0 <= p < 2; the magnification diverges only as p -> 2.

JAX-native and differentiable in p; the direct estimator is for sampled fields.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

# Exact PP20 constant: 3^{3/2}/2 = 2.5980762... (PP20 Eq. 6 prints it rounded as 2.6).
_PP20_CONST = 3.0**1.5 / 2.0


def magnification_factor(p: Float[Array, ""]) -> Float[Array, ""]:
    r"""Analytic magnification factor for a pure power-law profile (PP20 Eq. 6).

    .. math:: \zeta(p) = \frac{(3-p)^{3/2}}{(3^{3/2}/2)\,(2-p)}

    Valid 0 <= p < 2 (diverges as p -> 2). zeta(0)=1 (top-hat lower limit),
    zeta(1.5)=sqrt(2), zeta(1.67)~1.79. No pole at p=1.3.
    """
    return (3.0 - p) ** 1.5 / (_PP20_CONST * (2.0 - p))


def magnification_factor_with_core(
    p: Float[Array, ""],
    r_c_over_R: Float[Array, ""],
    n_nodes: int = 2048,
) -> Float[Array, ""]:
    r"""Numerical magnification factor for a cored profile (regularizes p -> 2).

    Cored density rho(r) = rho_c [1 + (r/r_c)^2]^{-p/2}, integrated over the sphere
    r/R in [0, 1] by the trapezoid rule (fixed node count -> grad-safe, no Python
    convergence loop). As r_c/R -> 0 the profile approaches a pure power law and
    zeta -> :func:`magnification_factor`; as r_c/R -> inf it approaches a top-hat
    and zeta -> 1.

    Parameters
    ----------
    p : density-profile slope.
    r_c_over_R : core radius in units of the clump radius R.
    n_nodes : trapezoid nodes on x = r/R in (0, 1] (static).
    """
    x = jnp.linspace(1.0 / n_nodes, 1.0, n_nodes)  # r/R in (0,1]
    rho = (1.0 + (x / r_c_over_R) ** 2) ** (-p / 2.0)
    w = x**2  # dV ~ r^2 dr; constant 4*pi*R^3 factor cancels in the ratio
    return zeta_from_field(rho, w)


def zeta_from_field(
    rho: Float[Array, " n"], weights: Float[Array, " n"]
) -> Float[Array, ""]:
    r"""Direct (field) magnification estimator (PP20-consistent).

    .. math:: \zeta = \frac{\sum \rho^{3/2} w \, (\sum w)^{1/2}}{(\sum \rho w)^{3/2}}

    Equivalent to <rho^{1/2}>_mass / <rho>^{1/2} for cells of volume ``weights``.
    Preferred for realistic (cored, non-power-law) geometries and for measuring
    zeta directly from a sampled 3D density field. ``rho`` need not be normalized.
    """
    num = jnp.sum(rho**1.5 * weights) * jnp.sqrt(jnp.sum(weights))
    den = jnp.sum(rho * weights) ** 1.5
    return num / den
