"""Analytical initial conditions with exact solutions for testing."""

from progenax.analytical.core import (
    AnalyticalIC,
    SOLAR_SYSTEM_PLANETS,
    get_planet,
    two_body_kepler,
    two_body_period,
    two_body_energy,
    three_body_figure_eight,
    figure_eight_period,
    harmonic_oscillator,
    harmonic_solution,
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
