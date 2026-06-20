"""Three-body figure-eight + harmonic-oscillator cases (split from analytical/core.py)."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from .base import AnalyticalIC

# ============================================================================
# Three-Body Figure-8 Orbit
# ============================================================================


def three_body_figure_eight(
    mass: float = 1.0, scale: float = 1.0, G: float = 1.0
) -> AnalyticalIC:
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
        G: Gravitational constant (default: 1.0, dimensionless)

    Returns:
        AnalyticalIC with 3 equal masses in figure-8 orbit

    Analytical solution:
        - Period: T = 6.3259... [dimensionless, G=1 units]
        - Total energy: E = constant (conserved to machine precision)
        - Angular momentum: L = 0 (zero total angular momentum)
        - Symmetry: 3-fold rotational symmetry

    Example:
        >>> # Dimensionless units (default, for mathematical testing)
        >>> ic = three_body_figure_eight(mass=1.0, scale=1.0, G=1.0)
        >>> # Integrate for 1 period: t_end = 6.3259

        >>> # Physical units (use jaxstro.units for consistent G)
        >>> from jaxstro.units import STELLAR
        >>> ic = three_body_figure_eight(mass=1.0, scale=1.0, G=STELLAR.G)

    Notes:
        - **Uses dimensionless units by default** (G=1, m=1, scale=1)
        - This is a mathematical test problem with known exact solution
        - For physical units, pass G from jaxstro.units (STELLAR or PLANETARY)
        - Period scales as T ∝ sqrt(scale³ / G) from dimensionless T₀ = 6.3259
        - Works best with zero softening (pure Newtonian)

    References:
        - Chenciner & Montgomery (2000), Ann. Math., 152, 881
        - Simó (2001), private communication (numerical coefficients)
        - Montgomery (2001), Notices AMS, 48, 471 - Popular review
    """
    # Canonical Chenciner-Montgomery-Simó initial conditions (dimensionless: G=1, m=1,
    # length=1). Bodies 1 and 2 sit at ±(x1, y1) with the SAME velocity; body 3 sits at
    # the origin with velocity -2·(vx1, vy1). This collinear configuration makes BOTH the
    # total momentum and the total angular momentum vanish (the figure-eight has L=0) and
    # is the one that closes after one period. (A 3-fold spatial rotation does NOT — it
    # gives an equilateral choreography with L≠0; verified by one-period integration.)
    x1 = 0.97000436
    y1 = -0.24308753
    vx1 = 0.4662036850
    vy1 = 0.4323657300

    pos0 = jnp.array(
        [
            [x1, y1, 0.0],
            [-x1, -y1, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    vel0 = jnp.array(
        [
            [vx1, vy1, 0.0],
            [vx1, vy1, 0.0],
            [-2.0 * vx1, -2.0 * vy1, 0.0],
        ]
    )

    # Rescale the dimensionless orbit to (mass, scale, G): lengths × scale, velocities ×
    # √(G·m/scale), so the orbit stays a closed figure-eight with period
    # T₀·√(scale³/(G·m)). (At the defaults G=m=scale=1 this is the identity.)
    positions = pos0 * scale
    velocities = vel0 * jnp.sqrt(G * mass / scale)
    masses = jnp.array([mass, mass, mass])

    period = figure_eight_period(scale, G, mass)

    return AnalyticalIC(
        positions=positions,
        velocities=velocities,
        masses=masses,
        name="three_body_figure_eight",
        period=float(period),
        energy=None,  # Constant, but not a simple closed form; measured in tests.
    )


def figure_eight_period(
    scale: ArrayLike = 1.0, G: ArrayLike = 1.0, mass: ArrayLike = 1.0
) -> Float[Array, ""]:
    """
    Return period of figure-8 orbit.

    Args:
        scale: Spatial scale factor used in three_body_figure_eight()
        G: Gravitational constant [appropriate units]
        mass: Mass of each particle [same units as three_body_figure_eight] (default 1.0)

    Returns:
        Period in time units

    Notes:
        - Period is T₀ = 6.32591398 in dimensionless units (G=1, m=1, scale=1)
        - Scales as T = T₀ · √(scale³ / (G·m))
    """
    T_dimensionless = 6.32591398
    return T_dimensionless * jnp.sqrt(scale**3 / (G * mass))


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
