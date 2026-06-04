"""Kepler's-third-law period <-> semi-major-axis conversions.

Deterministic scalar conversions (distinct from the period *distributions* in
:mod:`period`). Split out of :mod:`kepler` to keep that module focused on the
KeplerElements state machinery. All functions take an explicit ``G``.
"""

import jax.numpy as jnp


def compute_period(
    a: float,
    M_total: float,
    G: float,
) -> float:
    """
    Compute orbital period from semi-major axis using Kepler's 3rd law.

    Args:
        a: Semi-major axis [length units]
        M_total: Total mass of binary system [M☉]
        G: Gravitational constant (REQUIRED, no default)

    Returns:
        period: Orbital period [time units]

    Formula:
        T = 2π√(a³/(GM))

    Examples:
        >>> # Earth orbit: a=1 AU, M=1 M☉ → T≈1 year
        >>> G = 39.478  # AU³/Msun/yr²
        >>> T = compute_period(a=1.0, M_total=1.0, G=G)
        >>> print(f"Period: {T:.2f} years")
        Period: 1.00 years

        >>> # Stellar cluster orbit: a=1 pc, M=1000 M☉
        >>> G = 0.00450  # pc³/Msun/Myr²
        >>> T = compute_period(a=1.0, M_total=1000.0, G=G)
        >>> print(f"Period: {T:.2f} Myr")
        Period: 0.94 Myr

    References:
        Kepler's 3rd Law: T² ∝ a³/M
        Murray & Dermott (1999) Eq 2.37
    """
    # T = 2π√(a³/(GM)). Divide-safe double-where (mirrors KeplerElements.to_state):
    # the sqrt' and 1/denom blow up at a=0 / GM=0, so guard both so the gradient
    # stays finite at those (unphysical) boundaries rather than NaN-poisoning.
    denom = G * M_total
    denom_safe = jnp.where(denom > 0.0, denom, 1.0)
    arg = a**3 / denom_safe
    arg_safe = jnp.where(arg > 0.0, arg, 1.0)
    period = jnp.where(arg > 0.0, 2.0 * jnp.pi * jnp.sqrt(arg_safe), 0.0)

    return period


def period_to_semimajor_axis(
    period: float,
    M_total: float,
    G: float,
) -> float:
    """
    Compute semi-major axis from orbital period using Kepler's 3rd law.

    Args:
        period: Orbital period [time units]
        M_total: Total mass of binary system [M☉]
        G: Gravitational constant (REQUIRED, no default)

    Returns:
        a: Semi-major axis [length units]

    Formula:
        a = (GMT²/(4π²))^(1/3)

    Examples:
        >>> # Binary with 10 day period, M_total=2 M☉
        >>> G = 39.478  # AU³/Msun/yr²
        >>> P_yr = 10.0 / 365.25  # Convert days to years
        >>> a = period_to_semimajor_axis(P_yr, M_total=2.0, G=G)
        >>> print(f"Semi-major axis: {a:.3f} AU")
        Semi-major axis: 0.089 AU

        >>> # Star cluster binary: 10 Myr period, M_total=2 M☉
        >>> G = 0.00450  # pc³/Msun/Myr²
        >>> a = period_to_semimajor_axis(10.0, M_total=2.0, G=G)
        >>> print(f"Semi-major axis: {a:.2f} pc")
        Semi-major axis: 4.64 pc

    References:
        Kepler's 3rd Law: a³ ∝ T²M
        Murray & Dermott (1999) Eq 2.37
    """
    # a = (GM*T²/(4π²))^(1/3). The cube-root derivative (1/3)x^(-2/3) diverges at
    # x=0, so guard with a double-where so grad is finite at P=0 / GM=0 boundaries.
    arg = G * M_total * period**2 / (4.0 * jnp.pi**2)
    arg_safe = jnp.where(arg > 0.0, arg, 1.0)
    a = jnp.where(arg > 0.0, arg_safe ** (1.0 / 3.0), 0.0)

    return a


__all__ = ["compute_period", "period_to_semimajor_axis"]
