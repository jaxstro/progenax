"""Analytical IC container + planet data (split from analytical/core.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass(frozen=True)
class AnalyticalIC:
    """
    Result from analytical IC generation.

    Attributes:
        positions: Particle positions (N, 3)
        velocities: Particle velocities (N, 3)
        masses: Particle masses (N,)
        name: System name (e.g., "two_body_kepler", "figure_eight")
        period: Orbital period (if applicable)
        energy: Analytical total energy (if applicable)
    """

    positions: Float[Array, "N 3"]
    velocities: Float[Array, "N 3"]
    masses: Float[Array, "N"]
    name: str = ""
    period: Optional[float] = None
    energy: Optional[float] = None


# ============================================================================
# Solar System Data (Single Source of Truth)
# ============================================================================

# Orbital elements (a, e, inc, Omega, omega) are J2000.0 osculating elements from
#   JPL Horizons / Standish & Williams (2012). The `nu` (true anomaly) values are an
#   arbitrary phase spread (0, 45, 90, ... deg) so the planets are not all aligned —
#   a test-IC convenience, not ephemeris positions.
# Masses `M` are planet/Sun mass ratios from the IAU (2009) best estimates
#   (Earth: M_sun/M_E = 332946.05 -> 3.0035e-6; Jupiter: 1/1047.35 -> 9.5479e-4). The
#   solar-mass UNIT itself is the IAU 2015 Resolution B3 nominal mass parameter
#   (Prša et al. 2016, AJ 152, 41 — (GM_sun)^N = 1.3271244e20 m^3 s^-2).
# Used by solar_system_full(), solar_system_inner_4(), and validation scripts.
SOLAR_SYSTEM_PLANETS = [
    {
        "name": "Mercury",
        "M": 1.6601e-7,  # Msun
        "a": 0.38710,  # AU
        "e": 0.20563,
        "inc": 7.00,  # deg
        "Omega": 48.33,  # deg
        "omega": 29.12,  # deg
        "nu": 0.0,  # deg
    },
    {
        "name": "Venus",
        "M": 2.4478e-6,
        "a": 0.72333,
        "e": 0.00677,
        "inc": 3.39,
        "Omega": 76.68,
        "omega": 54.88,
        "nu": 45.0,
    },
    {
        "name": "Earth",
        "M": 3.0035e-6,
        "a": 1.00000,
        "e": 0.01671,
        "inc": 0.00,
        "Omega": 0.00,
        "omega": 102.94,
        "nu": 90.0,
    },
    {
        "name": "Mars",
        "M": 3.2271e-7,
        "a": 1.52368,
        "e": 0.09340,
        "inc": 1.85,
        "Omega": 49.56,
        "omega": 286.50,
        "nu": 135.0,
    },
    {
        "name": "Jupiter",
        "M": 9.5479e-4,
        "a": 5.20440,
        "e": 0.04839,
        "inc": 1.31,
        "Omega": 100.46,
        "omega": 273.87,
        "nu": 180.0,
    },
    {
        "name": "Saturn",
        "M": 2.8588e-4,
        "a": 9.58260,
        "e": 0.05565,
        "inc": 2.49,
        "Omega": 113.66,
        "omega": 339.39,
        "nu": 225.0,
    },
    {
        "name": "Uranus",
        "M": 4.3662e-5,
        "a": 19.2018,
        "e": 0.04638,
        "inc": 0.77,
        "Omega": 74.01,
        "omega": 96.54,
        "nu": 270.0,
    },
    {
        "name": "Neptune",
        "M": 5.1514e-5,
        "a": 30.0470,
        "e": 0.00945,
        "inc": 1.77,
        "Omega": 131.78,
        "omega": 273.25,
        "nu": 315.0,
    },
]


def get_planet(name: str) -> dict:
    """
    Get orbital elements for a Solar System planet.

    Args:
        name: Planet name (case-insensitive)
              Valid: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune

    Returns:
        dict with keys:
            - name: Planet name (str)
            - M: Mass in solar masses (float)
            - a: Semi-major axis in AU (float)
            - e: Eccentricity (float)
            - inc: Inclination in degrees (float)
            - Omega: Longitude of ascending node in degrees (float)
            - omega: Argument of perihelion in degrees (float)
            - nu: True anomaly in degrees (float)

    Raises:
        ValueError: If planet name not recognized

    Example:
        >>> jupiter = get_planet("Jupiter")
        >>> print(f"Mass: {jupiter['M']:.4e} Msun, a: {jupiter['a']:.2f} AU")
        Mass: 9.5479e-04 Msun, a: 5.20 AU
    """
    for planet in SOLAR_SYSTEM_PLANETS:
        if planet["name"].lower() == name.lower():
            return planet.copy()  # Return copy to prevent accidental mutation

    available = [p["name"] for p in SOLAR_SYSTEM_PLANETS]
    raise ValueError(f"Unknown planet: '{name}'. Available planets: {', '.join(available)}")


