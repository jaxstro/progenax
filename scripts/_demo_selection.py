"""Shared selection / photometry physics for the demos (magnitude limits, photon-noise errors,
IMF-detectable counts). Reusable by the OED Stage-2 demo, B4 (binary mass function), B5 (IMF), and
any future magnitude-limited demo. Bolometric magnitudes (documented simplification: no band/BC/
extinction); band-specific photometry (BCs, extinction, crowding) is a planned follow-up via the
`fluxax` package once it is finalised. All functions are jnp / differentiable."""

import jax.numpy as jnp

from progenax.stellar import inverse_zams_luminosity, zams_luminosity

M_BOL_SUN = 4.74


def abs_bol_mag(mass, Z=0.02):
    """Absolute bolometric magnitude from the Tout+1996 ZAMS L(M)."""
    L = zams_luminosity(mass, Z)  # [L_sun]
    return M_BOL_SUN - 2.5 * jnp.log10(L)


def distance_modulus(d_pc):
    return 5.0 * jnp.log10(d_pc / 10.0)


def apparent_mag(mass, d_pc, Z=0.02):
    return abs_bol_mag(mass, Z) + distance_modulus(d_pc)


def m_min(m_lim, d_pc, Z=0.02):
    """Minimum detectable mass at limiting (apparent) magnitude m_lim and distance d_pc.
    Differentiable: m_lim -> faintest absolute mag -> L_min -> inverse ZAMS -> mass.

    Note: for limits brighter than the IMF can populate (very shallow m_lim at large d_pc),
    the inverse-ZAMS Newton solve saturates at its mass ceiling (~150 M_sun) and the result
    becomes a flat plateau with zero gradient. Callers should keep m_lim in the regime where
    m_min < imf.m_max; ``detectable_fraction`` already clips to the IMF support so it stays valid."""
    M_abs_max = m_lim - distance_modulus(d_pc)
    L_min = 10.0 ** (-0.4 * (M_abs_max - M_BOL_SUN))  # [L_sun]
    return inverse_zams_luminosity(L_min, Z)


def photon_noise_error(m_app, eps0, m_ref):
    """Per-star measurement error scaling: eps = eps0 * 10^{0.2 (m_app - m_ref)} (flux^-0.5-like)."""
    return eps0 * 10.0 ** (0.2 * (m_app - m_ref))


def detectable_fraction(m_lim, d_pc, imf, Z=0.02):
    """IMF fraction with mass >= m_min(m_lim): 1 - cdf(m_min), clamped to the IMF support."""
    mm = jnp.clip(m_min(m_lim, d_pc, Z), imf.m_min, imf.m_max)
    return 1.0 - imf.cdf(mm)
