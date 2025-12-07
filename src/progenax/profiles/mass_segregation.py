"""Mass segregation transforms for star cluster ICs.

Implements primordial mass segregation where massive stars
are preferentially located near the cluster center.

References:
    Baumgardt et al. (2008) MNRAS 384, 1231 - Primordial mass segregation
    Subr et al. (2008) A&A 487, 671 - Mass segregation in young clusters
    de Grijs et al. (2002) MNRAS 331, 245 - NGC 330 mass segregation
"""

import jax.numpy as jnp
from jaxtyping import Array, Float


def apply_mass_segregation(
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    eta: float,
    m_ref: float,
) -> Float[Array, "N 3"]:
    """Apply primordial mass segregation to positions.

    Scales particle radii based on their mass:

        r_new = r_old * (m / m_ref)^(-eta)

    This gives:
        - m > m_ref: r_new < r_old (massive stars move inward)
        - m < m_ref: r_new > r_old (low-mass stars move outward)
        - m = m_ref: r_new = r_old (reference mass unchanged)

    The parameter eta controls segregation strength:
        - eta = 0: No segregation
        - eta = 0.5: Moderate segregation (typical observed value)
        - eta = 1: Strong segregation

    Args:
        positions: Input positions (N, 3)
        masses: Particle masses (N,)
        eta: Segregation strength parameter (0 = none, 0.5 = moderate, 1 = strong)
        m_ref: Reference mass [Msun] (typically mean or median mass)

    Returns:
        Segregated positions (N, 3)

    Note:
        This is a simple radial scaling model. More sophisticated models
        use energy-based segregation (Spitzer 1969) or dynamical friction
        timescale arguments.

    Reference:
        Subr et al. (2008) A&A 487, 671
        Baumgardt et al. (2008) MNRAS 384, 1231
    """
    # Compute scale factor for each particle
    # r_new / r_old = (m / m_ref)^(-eta)
    scale_factor = (masses / m_ref) ** (-eta)  # (N,)

    # Apply radial scaling (preserve direction)
    positions_scaled = positions * scale_factor[:, None]  # (N, 3)

    return positions_scaled


def compute_mass_segregation_ratio(
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    mass_threshold: float,
) -> Float[Array, ""]:
    """Compute mass segregation ratio (MSR) diagnostic.

    MSR compares the mean separation of massive stars to that of
    random reference stars. MSR > 1 indicates mass segregation.

        MSR = <r_ref> / <r_massive>

    Args:
        positions: Particle positions (N, 3)
        masses: Particle masses (N,)
        mass_threshold: Mass threshold for "massive" stars [Msun]

    Returns:
        Mass segregation ratio (MSR > 1 indicates segregation)

    Reference:
        Allison et al. (2009) MNRAS 395, 1449 - MSR definition
    """
    radii = jnp.linalg.norm(positions, axis=1)

    massive_mask = masses > mass_threshold
    r_massive = radii[massive_mask]
    r_all = radii

    mean_r_massive = jnp.mean(r_massive)
    mean_r_all = jnp.mean(r_all)

    # Simple MSR: ratio of mean radii
    msr = mean_r_all / mean_r_massive

    return msr


__all__ = ["apply_mass_segregation", "compute_mass_segregation_ratio"]
