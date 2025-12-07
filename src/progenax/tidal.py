"""Tidal physics utilities for star cluster ICs.

Computes tidal/Jacobi radii and applies tidal truncation.

References:
    King (1962) AJ 67, 471 - Tidal radius definition
    Binney & Tremaine (2008) "Galactic Dynamics" Section 8.3.1
    Baumgardt & Makino (2003) MNRAS 340, 227 - Tidal stripping
"""

from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float


def jacobi_radius(
    M_cluster: float,
    M_galaxy: float,
    R_galactic: float,
) -> float:
    """Compute Jacobi (tidal) radius for a cluster in a galaxy.

    The Jacobi radius is where the cluster's gravitational pull
    equals the tidal force from the galaxy (point mass approximation):

        r_J = R * (M_cluster / (3 * M_galaxy))^(1/3)

    Beyond r_J, stars become unbound from the cluster.

    Args:
        M_cluster: Cluster mass [Msun]
        M_galaxy: Galaxy mass (enclosed within R) [Msun]
        R_galactic: Distance from galactic center [pc]

    Returns:
        Jacobi radius [pc]

    Note:
        This assumes a point-mass galaxy and circular orbit.
        For more realistic models, use galactic potential functions.

    Reference:
        King (1962) AJ 67, 471
        Binney & Tremaine (2008) Eq. 8.91
    """
    r_J = R_galactic * (M_cluster / (3.0 * M_galaxy)) ** (1.0/3.0)
    return r_J


def jacobi_radius_isothermal(
    M_cluster: float,
    V_circ: float,
    R_galactic: float,
    G: float,
) -> float:
    """Compute Jacobi radius for isothermal halo.

    For a singular isothermal sphere with circular velocity V_circ:

        r_J = (G * M_cluster / (2 * Omega^2))^(1/3)

    where Omega = V_circ / R is the angular velocity.

    Args:
        M_cluster: Cluster mass [Msun]
        V_circ: Circular velocity of galaxy [km/s or appropriate units]
        R_galactic: Distance from galactic center [pc]
        G: Gravitational constant

    Returns:
        Jacobi radius [pc]

    Reference:
        Binney & Tremaine (2008) Section 8.3.1
    """
    Omega = V_circ / R_galactic
    r_J = (G * M_cluster / (2.0 * Omega**2)) ** (1.0/3.0)
    return r_J


def apply_tidal_truncation(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    r_t: float,
) -> Tuple[Float[Array, "M 3"], Float[Array, "M 3"], Float[Array, "M"], Float[Array, "N"]]:
    """Remove particles beyond tidal radius.

    Sharp truncation: particles with r > r_t are removed.

    Args:
        positions: Particle positions (N, 3)
        velocities: Particle velocities (N, 3)
        masses: Particle masses (N,)
        r_t: Tidal truncation radius [length units]

    Returns:
        Tuple of (positions_kept, velocities_kept, masses_kept, keep_mask)
        - positions_kept: Positions of retained particles (M, 3)
        - velocities_kept: Velocities of retained particles (M, 3)
        - masses_kept: Masses of retained particles (M,)
        - keep_mask: Boolean mask indicating which particles were kept (N,)

    Note:
        This is a sharp cutoff. For smoother truncation consistent with
        King models, use the King profile directly.
    """
    radii = jnp.linalg.norm(positions, axis=1)
    keep_mask = radii <= r_t

    positions_kept = positions[keep_mask]
    velocities_kept = velocities[keep_mask]
    masses_kept = masses[keep_mask]

    return positions_kept, velocities_kept, masses_kept, keep_mask


def fill_factor_to_r_h(
    fill_factor: float,
    r_J: float,
) -> float:
    """Convert fill factor to half-mass radius.

    Fill factor = r_h / r_J is the ratio of half-mass radius to Jacobi radius.

    Typical values:
        - fill_factor ~ 0.05-0.15: Compact, tidally underfilling
        - fill_factor ~ 0.15-0.30: Typical globular cluster
        - fill_factor ~ 0.30-0.50: Tidally filling

    Args:
        fill_factor: r_h / r_J ratio (typically 0.05 to 0.5)
        r_J: Jacobi radius [length units]

    Returns:
        Half-mass radius r_h [length units]

    Reference:
        Baumgardt & Makino (2003) MNRAS 340, 227
    """
    return fill_factor * r_J


__all__ = [
    "jacobi_radius",
    "jacobi_radius_isothermal",
    "apply_tidal_truncation",
    "fill_factor_to_r_h",
]
