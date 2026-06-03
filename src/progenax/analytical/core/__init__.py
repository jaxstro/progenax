"""Analytical N-body test cases with exact / known solutions.

Split out of the former monolithic ``core.py`` (500-LOC file limit). The public
API is unchanged — every symbol below remains importable from
``progenax.analytical`` and ``progenax.analytical.core``.
"""

from .base import AnalyticalIC, SOLAR_SYSTEM_PLANETS, get_planet
from .two_body import two_body_kepler, two_body_period, two_body_energy
from .few_body import (
    three_body_figure_eight,
    figure_eight_period,
    harmonic_oscillator,
    harmonic_solution,
)
from .solar_system import (
    earth_sun_2body,
    earth_sun_eccentric,
    sun_earth_jupiter_3body,
    solar_system_inner_4,
    solar_system_full,
)

__all__ = [
    "AnalyticalIC",
    "SOLAR_SYSTEM_PLANETS",
    "get_planet",
    "two_body_kepler",
    "two_body_period",
    "two_body_energy",
    "three_body_figure_eight",
    "figure_eight_period",
    "harmonic_oscillator",
    "harmonic_solution",
    "earth_sun_2body",
    "earth_sun_eccentric",
    "sun_earth_jupiter_3body",
    "solar_system_inner_4",
    "solar_system_full",
]
