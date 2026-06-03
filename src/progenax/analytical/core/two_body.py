"""Two-body Kepler analytical cases (split from analytical/core.py)."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import AnalyticalIC


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


