"""BM19 volume density PDF, CDF, and inverse-CDF (the rank-copula engine).

The volume-weighted PDF of s = ln(rho/rho_0) is a mass-conserving lognormal body
(mean s0 = -sigma_s^2/2) for s < s_t, joined continuously at s_t to a powerlaw
tail p_PL(s) = C e^{-alpha s} (continuity: C = p_LN(s_t) e^{alpha s_t}), then
renormalized so int p(s) ds = 1 (volume).

The inverse CDF s = F^{-1}(u) is built from a tabulated CDF + monotone interpolation;
it is smooth in (mach, b, alpha) so the P2 rank-copula field stays differentiable in
the cloud parameters (the ranks are frozen integers; only the CDF table carries grads).

JAX-native.
"""

import jax.numpy as jnp
from jax.scipy.special import erf
from jaxtyping import Array, Float

from gravoturb_fdf.theory.bm19 import sigma_s_squared, transition_density


def _pdf_pieces(mach, b, alpha):
    """Return (s0, sigma, s_t, p_ln_st, C, Z): lognormal mean/width, transition,
    lognormal pdf at s_t, powerlaw amplitude, and the volume-normalization Z."""
    s2 = sigma_s_squared(mach, b)
    sigma = jnp.sqrt(s2)
    s0 = -0.5 * s2
    s_t = transition_density(alpha, s2)

    def p_ln(s):
        return jnp.exp(-((s - s0) ** 2) / (2.0 * s2)) / jnp.sqrt(2.0 * jnp.pi * s2)

    p_ln_st = p_ln(s_t)
    C = p_ln_st * jnp.exp(alpha * s_t)  # continuity at s_t
    # Volume under unnormalized piecewise pdf:
    #   int_{-inf}^{s_t} p_LN ds  +  int_{s_t}^{inf} C e^{-alpha s} ds
    phi_ln_st = 0.5 * (1.0 + erf((s_t - s0) / (jnp.sqrt(2.0) * sigma)))
    vol_pl = C * jnp.exp(-alpha * s_t) / alpha  # = p_ln_st / alpha
    Z = phi_ln_st + vol_pl
    return s0, sigma, s_t, p_ln_st, C, Z


def bm19_volume_pdf(
    s: Float[Array, " n"],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
) -> Float[Array, " n"]:
    r"""Normalized volume PDF p(s) (lognormal body + powerlaw tail), int p ds = 1."""
    s0, sigma, s_t, _p_ln_st, C, Z = _pdf_pieces(mach, b, alpha)
    s2 = sigma**2
    p_ln = jnp.exp(-((s - s0) ** 2) / (2.0 * s2)) / jnp.sqrt(2.0 * jnp.pi * s2)
    p_pl = C * jnp.exp(-alpha * s)
    return jnp.where(s < s_t, p_ln, p_pl) / Z


def build_bm19_cdf_table(
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    n_nodes: int = 4096,
    n_sigma_low: float = 12.0,
    tail_extent: float = 80.0,
):
    r"""Tabulated CDF F(s) on a parameter-dependent s-grid.

    Returns ``(s_grid, cdf)`` with ``cdf`` analytic and monotone:

    - s <= s_t:  F(s) = Phi_LN(s) / Z
    - s  > s_t:  F(s) = [Phi_LN(s_t) + C (e^{-alpha s_t} - e^{-alpha s})/alpha] / Z

    Grid spans s0 - n_sigma_low*sigma to s_t + tail_extent (powerlaw tail). Smooth
    in (mach, b, alpha), so the inverse interpolation is differentiable.
    """
    s0, sigma, s_t, _p_ln_st, C, Z = _pdf_pieces(mach, b, alpha)
    s_lo = s0 - n_sigma_low * sigma
    s_hi = s_t + tail_extent
    s_grid = jnp.linspace(s_lo, s_hi, n_nodes)

    phi_ln = 0.5 * (1.0 + erf((s_grid - s0) / (jnp.sqrt(2.0) * sigma)))
    phi_ln_st = 0.5 * (1.0 + erf((s_t - s0) / (jnp.sqrt(2.0) * sigma)))
    cdf_pl = phi_ln_st + C * (jnp.exp(-alpha * s_t) - jnp.exp(-alpha * s_grid)) / alpha
    cdf = jnp.where(s_grid < s_t, phi_ln, cdf_pl) / Z
    # clamp tiny numerical excursions outside [0, 1]
    cdf = jnp.clip(cdf, 0.0, 1.0)
    return s_grid, cdf


def bm19_icdf(
    u: Float[Array, " m"],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    n_nodes: int = 4096,
) -> Float[Array, " m"]:
    r"""Inverse CDF s = F^{-1}(u) via monotone interpolation of the CDF table.

    ``jnp.interp(u, cdf, s_grid)`` maps uniform u in (0,1) -> s. Differentiable in
    (mach, b, alpha) through the smooth table; the caller's ranks/u are frozen.
    """
    s_grid, cdf = build_bm19_cdf_table(mach, b, alpha, n_nodes=n_nodes)
    return jnp.interp(u, cdf, s_grid)
