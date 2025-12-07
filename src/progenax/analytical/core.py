"""
Analytical initial conditions with exact solutions for testing.

This module provides idealized N-body systems with known analytical solutions,
essential for validating integrators and debugging physics bugs.

**CRITICAL: Functions take explicit G parameter (no global unit system)**

Why analytical test cases?
    - **Exact solutions exist** → can measure numerical error precisely
    - **Well-conditioned** → stable under perturbations (unlike chaotic systems)
    - **Simple geometry** → easy to visualize and understand failures
    - **Reference benchmarks** → compare against published results

Available systems:
    - two_body_kepler(): Circular and eccentric Keplerian orbits
    - three_body_figure_eight(): Periodic 3-body orbit (Chenciner & Montgomery 2000)
    - harmonic_oscillator(): 1D/2D simple harmonic motion
    - **Solar System**: earth_sun_2body(), solar_system_full(), etc.

Validation strategy:
    1. Create initial conditions with analytical functions
    2. Integrate for N orbits/periods
    3. Compare final state to analytical solution
    4. Measure |ΔE/E|, position error, etc.

References:
    - Chenciner & Montgomery (2000), Ann. Math., 152, 881 - Figure-8 orbit
    - Murray & Dermott (1999), "Solar System Dynamics" - Kepler orbits
    - Hairer et al. (2006), "Geometric Numerical Integration" - Test problems
    - JPL Horizons: https://ssd.jpl.nasa.gov/horizons/ - Solar System ephemeris
"""

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
# NASA JPL Solar System Data (Single Source of Truth)
# ============================================================================

# Planetary orbital elements at J2000.0 epoch
# Source: https://ssd.jpl.nasa.gov/horizons/
# Used by solar_system_full(), solar_system_inner_4(), and validation scripts
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


# ============================================================================
# Two-Body Keplerian Orbits
# ============================================================================


def two_body_kepler(
    M1: float,
    M2: float,
    a: float,
    G: float,
    e: float = 0.0,
    inclination: float = 0.0,
    Omega: float = 0.0,
    omega: float = 0.0,
    true_anomaly: float = 0.0,
) -> AnalyticalIC:
    """
    Create 2-body Keplerian orbit with analytical solution.

    Orbital Elements:
        - a: Semi-major axis (length units)
        - e: Eccentricity (0 = circular, 0 < e < 1 = ellipse)
        - inclination: Orbital tilt [radians] (0 = x-y plane)
        - Omega: Longitude of ascending node [radians]
        - omega: Argument of periapsis [radians]
        - true_anomaly: Initial position angle [radians]

    Args:
        M1: Primary mass [Msun]
        M2: Secondary mass [Msun]
        a: Semi-major axis [length units]
        G: Gravitational constant [appropriate units]
        e: Eccentricity (default: 0.0, circular)
        inclination: Orbital inclination [radians] (default: 0, planar)
        Omega: Longitude of ascending node [radians] (default: 0)
        omega: Argument of periapsis [radians] (default: 0)
        true_anomaly: Initial true anomaly [radians] (default: 0)

    Returns:
        AnalyticalIC with 2 particles in Keplerian orbit

    Analytical solution:
        - Period: T = 2π√(a³/(G(M1+M2)))
        - Total energy: E = -G M1 M2 / (2a)
        - Total angular momentum: L = μ√(G M_tot a (1-e²))
        - Conservation: ΔE/E and ΔL/L should be < machine precision

    Example:
        >>> G = 39.478  # Binary units (AU³/Msun/yr²)
        >>> ic = two_body_kepler(M1=1.0, M2=0.001, a=1.0, e=0.0, G=G)
        >>> # Period = 1.0 yr (by construction)

    Notes:
        - e=0 (circular) is best for initial validation
        - Reduced mass μ = M1*M2/(M1+M2)
        - Center of mass at origin (p_total = 0)

    References:
        - Murray & Dermott (1999), "Solar System Dynamics", Ch. 2
        - Hairer et al. (2006), "Geometric Numerical Integration", §I.2.4
    """
    # Total mass and reduced mass
    M_total = M1 + M2
    mu = M1 * M2 / M_total

    # Semi-latus rectum
    p = a * (1.0 - e**2)

    # Current radius and velocity in orbital plane
    # At true anomaly ν:
    #   r = p / (1 + e cos ν)
    #   v = sqrt(G M_total / p) * sqrt(1 + e² + 2e cos ν)
    r_orb = p / (1.0 + e * jnp.cos(true_anomaly))
    v_orb = jnp.sqrt(G * M_total / p) * jnp.sqrt(1.0 + e**2 + 2.0 * e * jnp.cos(true_anomaly))

    # Position and velocity in orbital plane (before rotation)
    # Position: along radial direction at true anomaly
    x_orb = r_orb * jnp.cos(true_anomaly)
    y_orb = r_orb * jnp.sin(true_anomaly)
    z_orb = 0.0

    # Velocity: perpendicular to radius (conservation of angular momentum)
    # v_r = (e sin ν) sqrt(G M_total / p)
    # v_perp = (1 + e cos ν) sqrt(G M_total / p)
    v_r = e * jnp.sin(true_anomaly) * jnp.sqrt(G * M_total / p)
    v_perp = (1.0 + e * jnp.cos(true_anomaly)) * jnp.sqrt(G * M_total / p)

    vx_orb = v_r * jnp.cos(true_anomaly) - v_perp * jnp.sin(true_anomaly)
    vy_orb = v_r * jnp.sin(true_anomaly) + v_perp * jnp.cos(true_anomaly)
    vz_orb = 0.0

    # Rotation matrices for 3D orientation (Omega, inclination, omega)
    # R_z(Omega) * R_x(i) * R_z(omega)
    cos_Omega = jnp.cos(Omega)
    sin_Omega = jnp.sin(Omega)
    cos_i = jnp.cos(inclination)
    sin_i = jnp.sin(inclination)
    cos_omega = jnp.cos(omega)
    sin_omega = jnp.sin(omega)

    # Combined rotation matrix (Murray & Dermott Eq. 2.122)
    R11 = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i
    R12 = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i
    R13 = sin_Omega * sin_i
    R21 = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i
    R22 = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i
    R23 = -cos_Omega * sin_i
    R31 = sin_omega * sin_i
    R32 = cos_omega * sin_i
    R33 = cos_i

    # Apply rotation to position and velocity
    x = R11 * x_orb + R12 * y_orb + R13 * z_orb
    y = R21 * x_orb + R22 * y_orb + R23 * z_orb
    z = R31 * x_orb + R32 * y_orb + R33 * z_orb

    vx = R11 * vx_orb + R12 * vy_orb + R13 * vz_orb
    vy = R21 * vx_orb + R22 * vy_orb + R23 * vz_orb
    vz = R31 * vx_orb + R32 * vy_orb + R33 * vz_orb

    # Relative position and velocity (M2 - M1)
    r_rel = jnp.array([x, y, z])
    v_rel = jnp.array([vx, vy, vz])

    # Convert to center-of-mass frame (p_total = 0)
    # r1 = -M2/M_total * r_rel
    # r2 = +M1/M_total * r_rel
    r1 = -(M2 / M_total) * r_rel
    r2 = (M1 / M_total) * r_rel

    v1 = -(M2 / M_total) * v_rel
    v2 = (M1 / M_total) * v_rel

    # Create arrays
    positions = jnp.array([r1, r2])  # (2, 3)
    velocities = jnp.array([v1, v2])  # (2, 3)
    masses = jnp.array([M1, M2])

    # Compute period and energy
    period = two_body_period(M1, M2, a, G)
    energy = two_body_energy(M1, M2, a, G)

    return AnalyticalIC(
        positions=positions,
        velocities=velocities,
        masses=masses,
        name="two_body_kepler",
        period=float(period),
        energy=float(energy),
    )


def two_body_period(M1: float, M2: float, a: float, G: float) -> float:
    """
    Compute orbital period for 2-body system (Kepler's 3rd law).

    Args:
        M1: Primary mass [Msun]
        M2: Secondary mass [Msun]
        a: Semi-major axis [length units]
        G: Gravitational constant [appropriate units]

    Returns:
        Orbital period [time units]

    Example:
        >>> G = 39.478  # Binary units
        >>> T = two_body_period(M1=1.0, M2=0.001, a=1.0, G=G)
        >>> # T = 1.0 yr (Earth orbit)
    """
    M_total = M1 + M2
    return 2.0 * jnp.pi * jnp.sqrt(a**3 / (G * M_total))


def two_body_energy(M1: float, M2: float, a: float, G: float) -> float:
    """
    Compute total energy for 2-body Keplerian orbit.

    Args:
        M1: Primary mass [Msun]
        M2: Secondary mass [Msun]
        a: Semi-major axis [length units]
        G: Gravitational constant [appropriate units]

    Returns:
        Total energy (negative for bound orbit)

    Notes:
        E = -G M1 M2 / (2a)
        Independent of eccentricity!
    """
    return -G * M1 * M2 / (2.0 * a)


# ============================================================================
# Three-Body Figure-8 Orbit
# ============================================================================


def three_body_figure_eight(mass: float = 1.0, scale: float = 1.0, G: float = 1.0) -> AnalyticalIC:
    """
    Create three-body figure-8 periodic orbit.

    This is a famous periodic solution to the 3-body problem discovered by
    Chenciner & Montgomery (2000). Three equal masses chase each other in
    a figure-8 pattern with period T ≈ 6.3259 [dimensionless units].

    The orbit is:
    - Planar (z=0)
    - Periodic with known period
    - Choreographic (particles equally spaced along curve)
    - Stable under small perturbations

    Args:
        mass: Mass of each particle [Msun] (default: 1.0)
        scale: Spatial scale factor (default: 1.0)
        G: Gravitational constant [appropriate units] (default: 1.0)

    Returns:
        AnalyticalIC with 3 equal masses in figure-8 orbit

    Analytical solution:
        - Period: T = 6.3259... [dimensionless, G=1 units]
        - Total energy: E = constant (conserved to machine precision)
        - Angular momentum: L = 0 (zero total angular momentum)
        - Symmetry: 3-fold rotational symmetry

    Example:
        >>> G = 1.0
        >>> ic = three_body_figure_eight(mass=1.0, scale=1.0, G=G)
        >>> # Integrate for 1 period: t_end = 6.3259

    Notes:
        - Uses dimensionless initial conditions from Chenciner & Montgomery
        - Scaled to current unit system via 'scale' parameter
        - Period is T ≈ 6.3259 in dimensionless units (G=1, m=1, scale=1)
        - Works best with zero softening (pure Newtonian)

    References:
        - Chenciner & Montgomery (2000), Ann. Math., 152, 881
        - Simó (2001), private communication (numerical coefficients)
        - Montgomery (2001), Notices AMS, 48, 471 - Popular review
    """
    # Initial conditions from Chenciner & Montgomery (2000)
    # Dimensionless units: G=1, m=1, length scale=1
    # Positions at t=0 (3-fold symmetry)
    x1 = 0.97000436 * scale
    y1 = -0.24308753 * scale

    # Velocities at t=0
    vx1 = 0.4662036850 * scale
    vy1 = 0.4323657300 * scale

    # Use 3-fold rotational symmetry to get other particles
    # Rotation by 120° and 240°
    theta2 = 2.0 * jnp.pi / 3.0
    theta3 = 4.0 * jnp.pi / 3.0

    cos2 = jnp.cos(theta2)
    sin2 = jnp.sin(theta2)
    cos3 = jnp.cos(theta3)
    sin3 = jnp.sin(theta3)

    # Particle 2 (rotated 120°)
    x2 = cos2 * x1 - sin2 * y1
    y2 = sin2 * x1 + cos2 * y1
    vx2 = cos2 * vx1 - sin2 * vy1
    vy2 = sin2 * vx1 + cos2 * vy1

    # Particle 3 (rotated 240°)
    x3 = cos3 * x1 - sin3 * y1
    y3 = sin3 * x1 + cos3 * y1
    vx3 = cos3 * vx1 - sin3 * vy1
    vy3 = sin3 * vx1 + cos3 * vy1

    # Create arrays (planar motion, z=0)
    positions = jnp.array(
        [
            [x1, y1, 0.0],
            [x2, y2, 0.0],
            [x3, y3, 0.0],
        ]
    )

    velocities = jnp.array(
        [
            [vx1, vy1, 0.0],
            [vx2, vy2, 0.0],
            [vx3, vy3, 0.0],
        ]
    )

    masses = jnp.array([mass, mass, mass])

    # Compute period
    period = figure_eight_period(scale, G)

    return AnalyticalIC(
        positions=positions,
        velocities=velocities,
        masses=masses,
        name="three_body_figure_eight",
        period=float(period),
        energy=None,  # Not trivial to compute analytically
    )


def figure_eight_period(scale: float = 1.0, G: float = 1.0) -> float:
    """
    Return period of figure-8 orbit.

    Args:
        scale: Spatial scale factor used in three_body_figure_eight()
        G: Gravitational constant [appropriate units]

    Returns:
        Period in time units

    Notes:
        - Period is T = 6.3259 in dimensionless units (G=1, m=1, scale=1)
        - Scales as T ∝ scale^(3/2) / √(G·m)
    """
    # Dimensionless period (G=1, m=1, scale=1)
    T_dimensionless = 6.32591398

    # Scale to current units: T = T_0 * sqrt(scale³ / (G·m))
    # For equal masses m=1, this simplifies to:
    return T_dimensionless * jnp.sqrt(scale**3 / G)


# ============================================================================
# Harmonic Oscillator
# ============================================================================


def harmonic_oscillator(
    amplitude: float = 1.0,
    omega: float = 1.0,
    phase: float = 0.0,
    mass: float = 1.0,
    dimension: Literal["1D", "2D"] = "1D",
) -> AnalyticalIC:
    """
    Create 1D or 2D harmonic oscillator for integrator testing.

    Simple harmonic motion: F = -k x, with k = m ω²
    Analytical solution: x(t) = A cos(ωt + φ)

    This is the SIMPLEST possible test for an integrator:
    - Linear dynamics (no chaos)
    - Known exact solution
    - Energy conserved exactly
    - Period independent of amplitude

    Args:
        amplitude: Oscillation amplitude (default: 1.0)
        omega: Angular frequency (default: 1.0) → Period = 2π/ω
        phase: Initial phase [radians] (default: 0)
        mass: Particle mass (default: 1.0)
        dimension: "1D" or "2D" (default: "1D")

    Returns:
        AnalyticalIC with 1 particle in harmonic potential

    Analytical solution:
        x(t) = A cos(ωt + φ)
        v(t) = -A ω sin(ωt + φ)
        E = (1/2) k A² = (1/2) m ω² A² (constant)

    Example:
        >>> ic = harmonic_oscillator(amplitude=1.0, omega=2*jnp.pi)
        >>> # Period T = 1.0, x(0)=1.0, v(0)=0

    Notes:
        - This tests integrator mechanics WITHOUT gravitational forces
        - Requires external force implementation: F_ext = -k·x
        - 1D: oscillation along x-axis
        - 2D: circular motion (2D isotropic oscillator)
        - Use for sanity checks before testing gravity

    Warning:
        Most N-body codes don't support external harmonic forces.
        This is a placeholder for future external potential support.
        For now, use two_body_kepler() for gravity tests.
    """
    # Initial position and velocity
    x0 = amplitude * jnp.cos(phase)
    v0 = -amplitude * omega * jnp.sin(phase)

    if dimension == "1D":
        # 1D oscillation along x-axis
        positions = jnp.array([[x0, 0.0, 0.0]])
        velocities = jnp.array([[v0, 0.0, 0.0]])
    elif dimension == "2D":
        # 2D circular motion (isotropic oscillator)
        # x(t) = A cos(ωt + φ)
        # y(t) = A sin(ωt + φ)
        y0 = amplitude * jnp.sin(phase)
        vy0 = amplitude * omega * jnp.cos(phase)
        positions = jnp.array([[x0, y0, 0.0]])
        velocities = jnp.array([[v0, vy0, 0.0]])
    else:
        raise ValueError(f"dimension must be '1D' or '2D', got '{dimension}'")

    masses = jnp.array([mass])

    # Period
    period = 2.0 * jnp.pi / omega

    return AnalyticalIC(
        positions=positions,
        velocities=velocities,
        masses=masses,
        name=f"harmonic_oscillator_{dimension}",
        period=float(period),
        energy=0.5 * mass * omega**2 * amplitude**2,
    )


def harmonic_solution(
    t: float,
    amplitude: float,
    omega: float,
    phase: float = 0.0,
    dimension: Literal["1D", "2D"] = "1D",
) -> tuple[Float[Array, "1 3"], Float[Array, "1 3"]]:
    """
    Compute analytical solution for harmonic oscillator at time t.

    Args:
        t: Time
        amplitude: Oscillation amplitude
        omega: Angular frequency
        phase: Initial phase [radians]
        dimension: "1D" or "2D"

    Returns:
        (positions, velocities) at time t

    Example:
        >>> pos, vel = harmonic_solution(t=1.0, amplitude=1.0, omega=2*jnp.pi)
        >>> # Compare with integrated result
    """
    x = amplitude * jnp.cos(omega * t + phase)
    v = -amplitude * omega * jnp.sin(omega * t + phase)

    if dimension == "1D":
        positions = jnp.array([[x, 0.0, 0.0]])
        velocities = jnp.array([[v, 0.0, 0.0]])
    else:  # 2D
        y = amplitude * jnp.sin(omega * t + phase)
        vy = amplitude * omega * jnp.cos(omega * t + phase)
        positions = jnp.array([[x, y, 0.0]])
        velocities = jnp.array([[v, vy, 0.0]])

    return positions, velocities


# ============================================================================
# Solar System Test Cases
# ============================================================================


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
    M_sun = 1.0

    # Planetary masses [Msun]
    M_mercury = 1.6601e-7
    M_venus = 2.4478e-6
    M_earth = 3.0035e-6
    M_mars = 3.2271e-7

    # Orbital elements (J2000.0)
    planets = [
        {
            "M": M_mercury,
            "a": 0.38710,
            "e": 0.20563,
            "inc": 7.00 * jnp.pi / 180,
            "Omega": 48.33 * jnp.pi / 180,
            "omega": 29.12 * jnp.pi / 180,
            "nu": 0.0,
        },
        {
            "M": M_venus,
            "a": 0.72333,
            "e": 0.00677,
            "inc": 3.39 * jnp.pi / 180,
            "Omega": 76.68 * jnp.pi / 180,
            "omega": 54.88 * jnp.pi / 180,
            "nu": jnp.pi / 2,
        },
        {
            "M": M_earth,
            "a": 1.00000,
            "e": 0.01671,
            "inc": 0.00 * jnp.pi / 180,
            "Omega": 0.00 * jnp.pi / 180,
            "omega": 102.94 * jnp.pi / 180,
            "nu": jnp.pi,
        },
        {
            "M": M_mars,
            "a": 1.52368,
            "e": 0.09340,
            "inc": 1.85 * jnp.pi / 180,
            "Omega": 49.56 * jnp.pi / 180,
            "omega": 286.50 * jnp.pi / 180,
            "nu": 3 * jnp.pi / 2,
        },
    ]

    # Create each planet in its orbit
    planet_systems = []
    for p in planets:
        ic = two_body_kepler(
            M1=M_sun,
            M2=p["M"],
            a=p["a"],
            e=p["e"],
            inclination=p["inc"],
            Omega=p["Omega"],
            omega=p["omega"],
            true_anomaly=p["nu"],
            G=G,
        )
        planet_systems.append(ic)

    # Extract planet positions and velocities (particle 1 in each 2-body system)
    planet_positions = jnp.array([ic.positions[1] for ic in planet_systems])
    planet_masses = jnp.array([p["M"] for p in planets])
    planet_velocities = jnp.array([ic.velocities[1] for ic in planet_systems])

    # Compute Sun position and velocity (barycentric frame)
    # Center of mass condition: M_sun*q_sun + sum(M_i*q_i) = 0
    q_sun = -jnp.sum(planet_masses[:, None] * planet_positions, axis=0) / M_sun
    v_sun = -jnp.sum(planet_masses[:, None] * planet_velocities, axis=0) / M_sun

    # Combine into single system
    positions = jnp.vstack([q_sun[None, :], planet_positions])
    velocities = jnp.vstack([v_sun[None, :], planet_velocities])
    masses = jnp.concatenate([jnp.array([M_sun]), planet_masses])

    return AnalyticalIC(
        positions=positions,
        velocities=velocities,
        masses=masses,
        name="solar_system_inner_4",
        period=None,  # Multiple periods
        energy=None,  # Not trivial for N>2
    )


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
    M_sun = 1.0

    # Use module-level planet data (single source of truth)
    planets = [p.copy() for p in SOLAR_SYSTEM_PLANETS]

    # Convert angles to radians
    for p in planets:
        p["inc_rad"] = p["inc"] * jnp.pi / 180.0
        p["Omega_rad"] = p["Omega"] * jnp.pi / 180.0
        p["omega_rad"] = p["omega"] * jnp.pi / 180.0
        p["nu_rad"] = p["nu"] * jnp.pi / 180.0

    # Create each planet in its orbit
    planet_systems = []
    for p in planets:
        ic = two_body_kepler(
            M1=M_sun,
            M2=p["M"],
            a=p["a"],
            e=p["e"],
            inclination=p["inc_rad"],
            Omega=p["Omega_rad"],
            omega=p["omega_rad"],
            true_anomaly=p["nu_rad"],
            G=G,
        )
        planet_systems.append(ic)

    # Extract planet positions and velocities (particle 1 in each 2-body system)
    planet_positions = jnp.array([ic.positions[1] for ic in planet_systems])
    planet_masses = jnp.array([p["M"] for p in planets])
    planet_velocities = jnp.array([ic.velocities[1] for ic in planet_systems])

    # Compute Sun position and velocity (barycentric frame)
    # Center of mass condition: M_sun*q_sun + sum(M_i*q_i) = 0
    q_sun = -jnp.sum(planet_masses[:, None] * planet_positions, axis=0) / M_sun
    v_sun = -jnp.sum(planet_masses[:, None] * planet_velocities, axis=0) / M_sun

    # Combine into single system
    positions = jnp.vstack([q_sun[None, :], planet_positions])
    velocities = jnp.vstack([v_sun[None, :], planet_velocities])
    masses = jnp.concatenate([jnp.array([M_sun]), planet_masses])

    return AnalyticalIC(
        positions=positions,
        velocities=velocities,
        masses=masses,
        name="solar_system_full",
        period=None,  # Multiple periods
        energy=None,  # Not trivial for N>2
    )
