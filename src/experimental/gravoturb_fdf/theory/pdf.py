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
from jax.scipy.special import erf, erfinv
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


def bm19_icdf_analytic(
    u: Float[Array, " m"],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
) -> Float[Array, " m"]:
    r"""Closed-form volume inverse-CDF s = F^{-1}(u) (exact in the deep tail).

    Body (u ≤ u_t): ``s = s0 + σ√2 · erfinv(2 u Z − 1)``.
    Tail (u > u_t): invert ``u Z = Φ_LN(s_t) + C(e^{-α s_t} − e^{-α s})/α`` →
    ``s = −ln[e^{-α s_t} − (u Z − Φ_LN(s_t)) α / C] / α``.

    Unlike the tabulated ``bm19_icdf``, this resolves u arbitrarily close to 1 (the
    heavy power-law tail), which the mass-conserving copula needs at ~10⁶ cells. The
    double-``where`` keeps both branches finite so grads in (mach,b,alpha) are NaN-free.
    """
    s0, sigma, s_t, _p_ln_st, C, Z = _pdf_pieces(mach, b, alpha)
    phi_ln_st = 0.5 * (1.0 + erf((s_t - s0) / (jnp.sqrt(2.0) * sigma)))
    u_t = phi_ln_st / Z

    # Evaluate each branch on a SAFE surrogate input outside its region (avoid NaN grads).
    u_body = jnp.where(u <= u_t, u, 0.5 * u_t)
    u_tail = jnp.where(u > u_t, u, 0.5 * (u_t + 1.0))

    body = s0 + sigma * jnp.sqrt(2.0) * erfinv(2.0 * u_body * Z - 1.0)
    inner = jnp.exp(-alpha * s_t) - (u_tail * Z - phi_ln_st) * alpha / C
    tail = -jnp.log(inner) / alpha
    return jnp.where(u <= u_t, body, tail)


def _mass_pieces(mach, b, alpha):
    """Mass-CDF building blocks: (s_t, s1, sigma, C, M_un_st, M_un_inf, Z).

    The mass-weighted lognormal body e^s p_LN(s) is a Gaussian of mean s1 = +σ²/2
    (since the volume body has mean s0 = −σ²/2), so its mass CDF is Φ((s−s1)/σ). The
    power-law tail contributes C e^{(1-α)s}; the unnormalized total mass M_un_inf and
    volume norm Z give the volume-mean density ⟨e^s⟩ = M_un_inf / Z.
    """
    s0, sigma, s_t, _p_ln_st, C, Z = _pdf_pieces(mach, b, alpha)
    s1 = s0 + sigma**2  # = +σ²/2  (mass-weighted lognormal mean)
    M_un_st = 0.5 * (1.0 + erf((s_t - s1) / (jnp.sqrt(2.0) * sigma)))  # Φ((s_t−s1)/σ)
    tail_mass = C * jnp.exp((1.0 - alpha) * s_t) / (alpha - 1.0)
    M_un_inf = M_un_st + tail_mass
    return s_t, s1, sigma, C, M_un_st, M_un_inf, Z


def bm19_mass_cdf(
    s: Float[Array, " n"],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
) -> Float[Array, " n"]:
    r"""Normalized mass CDF M(s) = ∫_{-∞}^s e^{s'} p(s') ds' / ⟨e^s⟩, with M(∞)=1.

    ``1 − M(s_t)`` equals BM19 f_dense. Smooth in (mach, b, alpha) → grad-safe.
    """
    s_t, s1, sigma, C, M_un_st, M_un_inf, _Z = _mass_pieces(mach, b, alpha)
    body = 0.5 * (1.0 + erf((s - s1) / (jnp.sqrt(2.0) * sigma)))  # Φ((s−s1)/σ)
    tail = M_un_st + C * (jnp.exp((1.0 - alpha) * s) - jnp.exp((1.0 - alpha) * s_t)) / (
        1.0 - alpha
    )
    M_un = jnp.where(s < s_t, body, tail)
    return M_un / M_un_inf


def bm19_mean_density(
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Volume-mean density ⟨ρ/ρ_0⟩ = ⟨e^s⟩ = M_un_inf / Z (≥ 1; tail adds mass)."""
    _s_t, _s1, _sigma, _C, _M_un_st, M_un_inf, Z = _mass_pieces(mach, b, alpha)
    return M_un_inf / Z


def bm19_volume_tail_fraction(
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Volume fraction above the transition: ``∫_{s_t}^∞ p(s) ds = vol_pl / Z``.

    Closed form from the normalized piecewise PDF: the powerlaw tail integrates to
    ``vol_pl = p_LN(s_t)/alpha``, divided by the volume normalization ``Z``. Used by
    the field resolution guard to compute the expected number of cells above s_t.
    """
    _s0, _sigma, _s_t, p_ln_st, _C, Z = _pdf_pieces(mach, b, alpha)
    vol_pl = p_ln_st / alpha
    return vol_pl / Z


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
