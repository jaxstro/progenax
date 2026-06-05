"""Analytical initial conditions with exact / known solutions for testing.

Flat module (the former ``core/`` subpackage was inlined). Every symbol is
importable from ``progenax.analytical``.
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
