"""Solar-system multi-body analytical cases (split from analytical/core.py)."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import AnalyticalIC, SOLAR_SYSTEM_PLANETS
from .two_body import two_body_kepler


# ============================================================================
# Solar System Test Cases
# ============================================================================


def _assemble_planetary_system(planet_dicts: list, G: float, name: str) -> AnalyticalIC:
    """Build a Sun + planets ``AnalyticalIC`` from ``SOLAR_SYSTEM_PLANETS``-style dicts.

    Each planet is placed on its Sun-planet two-body orbit (angles in **degrees** in the
    table, converted here), then the Sun is placed at the barycentre (total momentum = 0).
    Single source of truth for both ``solar_system_inner_4`` and ``solar_system_full``.
    """
    M_sun = 1.0
    deg = jnp.pi / 180.0
    planet_systems = [
        two_body_kepler(
            M1=M_sun, M2=p["M"], a=p["a"], e=p["e"],
            inclination=p["inc"] * deg, Omega=p["Omega"] * deg,
            omega=p["omega"] * deg, true_anomaly=p["nu"] * deg, G=G,
        )
        for p in planet_dicts
    ]
    planet_positions = jnp.array([ic.positions[1] for ic in planet_systems])
    planet_velocities = jnp.array([ic.velocities[1] for ic in planet_systems])
    planet_masses = jnp.array([p["M"] for p in planet_dicts])

    # Barycentric Sun: M_sun·q_sun + Σ M_i·q_i = 0.
    q_sun = -jnp.sum(planet_masses[:, None] * planet_positions, axis=0) / M_sun
    v_sun = -jnp.sum(planet_masses[:, None] * planet_velocities, axis=0) / M_sun

    return AnalyticalIC(
        positions=jnp.vstack([q_sun[None, :], planet_positions]),
        velocities=jnp.vstack([v_sun[None, :], planet_velocities]),
        masses=jnp.concatenate([jnp.array([M_sun]), planet_masses]),
        name=name,
        period=None,  # Multiple periods
        energy=None,  # Not trivial for N>2
    )


def earth_sun_2body(G: float) -> AnalyticalIC:
    """
    Earth-Sun system for validation (circular orbit approximation).

    Physical parameters:
        - M_sun = 1.0 Msun
        - M_earth = 3.0035e-6 Msun (1 Earth mass)
        - a = 1.0 AU (semi-major axis)
        - e = 0.0167 (actual eccentricity, approximated as circular)

    Args:
        G: Gravitational constant [appropriate units, e.g., AU³/Msun/yr²]

    Returns:
        AnalyticalIC with Sun + Earth in circular orbit

    Validation targets:
        - Period: T = 1.0 yr (by construction with binary units)
        - Energy conservation: |ΔE/E| < 10^-12
        - Angular momentum conservation: |ΔL/L| < 10^-15

    Example:
        >>> G = 39.478  # Binary units
        >>> ic = earth_sun_2body(G=G)
        >>> # Integrate for 10 orbits, check energy conservation

    Notes:
        - Uses circular orbit (e=0) for simplicity
        - True Earth orbit has e=0.0167 (use earth_sun_eccentric() for that)
        - Mass ratio: M_sun/M_earth ≈ 333,000

    References:
        - JPL Horizons ephemeris data
        - Murray & Dermott (1999), "Solar System Dynamics", Appendix B
    """
    M_sun = 1.0  # Msun
    M_earth = 3.0035e-6  # Msun (1 Earth mass = 5.972e24 kg)
    a = 1.0  # AU
    e = 0.0  # Circular approximation (true: 0.0167)

    return two_body_kepler(M1=M_sun, M2=M_earth, a=a, e=e, G=G)


def earth_sun_eccentric(G: float) -> AnalyticalIC:
    """
    Earth-Sun system with true eccentricity.

    Uses actual Earth orbital parameters for more realistic validation.

    Physical parameters:
        - M_sun = 1.0 Msun
        - M_earth = 3.0035e-6 Msun
        - a = 1.0 AU
        - e = 0.0167 (true eccentricity)

    Args:
        G: Gravitational constant [appropriate units]

    Returns:
        AnalyticalIC with Sun + Earth in eccentric orbit

    Validation targets:
        - Perihelion distance: r_peri = a(1-e) = 0.9833 AU
        - Aphelion distance: r_aph = a(1+e) = 1.0167 AU
        - Period: T = 1.0 yr
        - Energy conservation: |ΔE/E| < 10^-12

    Example:
        >>> G = 39.478  # Binary units
        >>> ic = earth_sun_eccentric(G=G)
        >>> # Tests adaptive timestep handling (dt varies by ~3%)
    """
    M_sun = 1.0
    M_earth = 3.0035e-6
    a = 1.0
    e = 0.0167  # True Earth eccentricity

    return two_body_kepler(M1=M_sun, M2=M_earth, a=a, e=e, G=G)


def sun_earth_jupiter_3body(G: float) -> AnalyticalIC:
    """
    Sun-Earth-Jupiter system for 3-body validation.

    This is a realistic hierarchical 3-body problem:
    - Inner binary: Sun-Earth (a=1 AU, T=1 yr)
    - Outer planet: Jupiter (a=5.2 AU, T=11.86 yr)

    Physical parameters:
        - M_sun = 1.0 Msun
        - M_earth = 3.0035e-6 Msun
        - M_jupiter = 9.548e-4 Msun (1 Jupiter mass)
        - a_earth = 1.0 AU, e_earth = 0.0167
        - a_jupiter = 5.2044 AU, e_jupiter = 0.0489

    Args:
        G: Gravitational constant [appropriate units]

    Returns:
        AnalyticalIC with Sun + Earth + Jupiter

    Validation targets:
        - Hierarchical timesteps: Jupiter needs ~10× longer dt than Earth
        - Energy conservation: |ΔE/E| < 10^-10 (3-body is harder than 2-body)
        - Perturbations: Earth's orbit perturbed by Jupiter (~0.1% level)

    Example:
        >>> G = 39.478  # Binary units
        >>> ic = sun_earth_jupiter_3body(G=G)
        >>> # Integrate for 100 yr, check energy conservation

    Notes:
        - Jupiter mass ≈ 318 Earth masses
        - Mass ratio: M_jup/M_earth ≈ 318
        - Hierarchical: a_jup/a_earth ≈ 5.2 (well-separated)

    References:
        - JPL Horizons ephemeris (epoch J2000.0)
        - Murray & Dermott (1999), Appendix B
    """
    M_sun = 1.0
    M_earth = 3.0035e-6
    M_jupiter = 9.548e-4

    # Earth orbit
    a_earth = 1.0
    e_earth = 0.0167

    # Jupiter orbit
    a_jupiter = 5.2044
    e_jupiter = 0.0489

    # Create 2-body systems for each planet
    # Earth at perihelion (nu=0), Jupiter at aphelion (nu=π) for good separation
    planet_data = [
        (M_earth, a_earth, e_earth, 0.0),
        (M_jupiter, a_jupiter, e_jupiter, jnp.pi),
    ]

    # Extract planet positions and velocities from 2-body systems
    planet_positions = []
    planet_velocities = []
    for M_planet, a, e, nu in planet_data:
        ic = two_body_kepler(M1=M_sun, M2=M_planet, a=a, e=e, true_anomaly=nu, G=G)
        # Planet is particle 1 in 2-body system
        planet_positions.append(ic.positions[1])
        planet_velocities.append(ic.velocities[1])

    # Stack into arrays
    planet_positions = jnp.array(planet_positions)  # (2, 3)
    planet_velocities = jnp.array(planet_velocities)  # (2, 3)
    planet_masses = jnp.array([M_earth, M_jupiter])

    # Compute Sun position and velocity for barycentric frame
    # Center of mass condition: M_sun*q_sun + sum(M_i*q_i) = 0
    q_sun = -jnp.sum(planet_masses[:, None] * planet_positions, axis=0) / M_sun
    v_sun = -jnp.sum(planet_masses[:, None] * planet_velocities, axis=0) / M_sun

    # Combine into full system (Sun + planets)
    positions = jnp.vstack([q_sun[None, :], planet_positions])
    velocities = jnp.vstack([v_sun[None, :], planet_velocities])
    masses = jnp.array([M_sun, M_earth, M_jupiter])

    return AnalyticalIC(
        positions=positions,
        velocities=velocities,
        masses=masses,
        name="sun_earth_jupiter_3body",
        period=None,  # Multiple periods (Earth: 1 yr, Jupiter: 11.86 yr)
        energy=None,  # Not trivial to compute for 3-body
    )


def solar_system_inner_4(G: float) -> AnalyticalIC:
    """
    Inner solar system: Sun + Mercury + Venus + Earth + Mars.

    Realistic N=5 system for testing block timesteps and hierarchical integration.

    Physical parameters (epoch J2000.0):
        - Sun: M = 1.0 Msun
        - Mercury: M = 1.66e-7 Msun, a = 0.387 AU, e = 0.206
        - Venus: M = 2.45e-6 Msun, a = 0.723 AU, e = 0.007
        - Earth: M = 3.00e-6 Msun, a = 1.000 AU, e = 0.017
        - Mars: M = 3.23e-7 Msun, a = 1.524 AU, e = 0.093

    Args:
        G: Gravitational constant [appropriate units]

    Returns:
        AnalyticalIC with Sun + 4 inner planets

    Validation targets:
        - Timestep hierarchy: Mercury (88 d) → Venus (225 d) → Earth (365 d) → Mars (687 d)
        - Energy conservation: |ΔE/E| < 10^-9 (more challenging than 2-body)
        - Secular perturbations: orbital precession over centuries

    Example:
        >>> G = 39.478  # Binary units
        >>> ic = solar_system_inner_4(G=G)
        >>> # Integrate for 1000 yr, measure energy drift

    Notes:
        - Mercury has highest eccentricity (e=0.206) → needs shortest timestep
        - Venus nearly circular (e=0.007) → can use longer timestep
        - Mars eccentric (e=0.093) → moderate timestep variation

    References:
        - JPL Horizons ephemeris (J2000.0)
        - https://ssd.jpl.nasa.gov/planets/phys_par.html
    """
    # Single source of truth: the first four entries of SOLAR_SYSTEM_PLANETS.
    return _assemble_planetary_system(SOLAR_SYSTEM_PLANETS[:4], G, "solar_system_inner_4")


def solar_system_full(G: float) -> AnalyticalIC:
    """
    Full solar system: Sun + 8 planets with true orbital elements.

    The ultimate N-body validation test for gravitational integrators.
    Includes all 8 planets with realistic masses, semi-major axes, and eccentricities.

    Physical parameters (epoch J2000.0):
        - Sun: M = 1.0 Msun
        - Mercury: a = 0.387 AU, e = 0.206, T = 0.241 yr
        - Venus: a = 0.723 AU, e = 0.007, T = 0.615 yr
        - Earth: a = 1.000 AU, e = 0.017, T = 1.000 yr
        - Mars: a = 1.524 AU, e = 0.093, T = 1.881 yr
        - Jupiter: a = 5.204 AU, e = 0.049, T = 11.86 yr
        - Saturn: a = 9.582 AU, e = 0.057, T = 29.46 yr
        - Uranus: a = 19.20 AU, e = 0.046, T = 84.01 yr
        - Neptune: a = 30.05 AU, e = 0.011, T = 164.8 yr

    Args:
        G: Gravitational constant [appropriate units]

    Returns:
        AnalyticalIC with Sun + 8 planets (N=9)

    Validation targets:
        - Energy conservation: |ΔE/E| < 10^-8 over 1000 yr
        - Secular dynamics: planetary precession over millennia
        - Timestep hierarchy: Mercury (88 d) to Neptune (60,190 d) → 680× range
        - Conservation over long timescales (Myr)

    Example:
        >>> G = 39.478  # Binary units
        >>> ic = solar_system_full(G=G)
        >>> # Integrate for 10,000 yr with block timesteps

    Notes:
        - **Block timestepping essential**: 680× timestep range (Mercury to Neptune)
        - Excludes Pluto (dwarf planet, highly eccentric/inclined)
        - Excludes moons (would need hierarchical treatment)
        - Uses J2000.0 epoch osculating elements
        - True 3D orbits (includes inclinations)

    Warning:
        - Long-term integration (>10 Myr) requires symplectic integrators
        - Chaotic on 5-10 Myr timescales (Laskar 1989)
        - Use for validation, not production solar system ephemeris

    References:
        - JPL Horizons: https://ssd.jpl.nasa.gov/horizons/
        - Standish & Williams (2012), "Orbital Ephemerides"
        - Laskar (1989), Nature, 338, 237 - Solar system chaos
    """
    # Single source of truth: all eight entries of SOLAR_SYSTEM_PLANETS.
    return _assemble_planetary_system(SOLAR_SYSTEM_PLANETS, G, "solar_system_full")
