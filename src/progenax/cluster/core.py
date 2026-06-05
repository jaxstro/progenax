# progenax/src/progenax/cluster/core.py
"""
Core cluster initial condition generator implementing v1.4 spec.

Provides the main API for generating star cluster initial conditions with:
- Arbitrary density profiles (Plummer, King, EFF)
- Optional mass segregation (Baumgardt+2008 energy ordering)
- Flexible virial ratio control

(Turbulent/fractal substructure ICs live in the experimental ``gravoturb_fdf``
package, not in released progenax.)

Data Classes:
    ClusterState: Immutable container for cluster phase-space coordinates
    MassSegregationLayer: Parameters for mass segregation
    SpatialStructureParams: Combined structure configuration

Functions:
    generate_cluster_ic: Main entry point for IC generation
    sample_velocities_for_profile: Profile-aware velocity sampling

Example:
    >>> from progenax.cluster import generate_cluster_ic, SpatialStructureParams
    >>> from progenax.imf import PowerLawIMF
    >>> import jax
    >>>
    >>> key = jax.random.PRNGKey(42)
    >>> imf = PowerLawIMF.kroupa()
    >>> cluster = generate_cluster_ic(
    ...     key=key,
    ...     N_stars=1000,
    ...     M_total=1000.0,  # Msun
    ...     R_half=1.0,       # pc
    ...     imf_params=imf,
    ...     structure_params=SpatialStructureParams(base_profile="plummer"),
    ... )

References:
    Baumgardt, De Marchi & Kroupa (2008), ApJ 685, 247
    Goodwin & Whitworth (2004), A&A 413, 929
    Küpper et al. (2011), MNRAS 417, 2300 - McLuster
    Allison et al. (2009), ApJ 700, L99
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jax import Array, random
from jaxtyping import Float, PRNGKeyArray

from progenax import defaults
from progenax.profiles.api import (
    sample_density_profile,
    compute_profile_potential,
)
from progenax.cluster.mass_segregation import energy_sorted_segregation
from progenax.dynamics.virial import (
    compute_potential_energy,
    rescale_velocities_to_virial,
)


# =============================================================================
# Data Classes (from v1.4 spec)
# =============================================================================


@dataclass(frozen=True)
class ClusterState:
    """
    Pure data container for cluster phase-space coordinates.

    All fields are jax.Array. This is immutable and can be passed
    through JIT boundaries.

    Attributes:
        masses: Stellar masses (N,) in M_sun
        positions: Positions (N, 3) in pc (stellar unit system)
        velocities: Velocities (N, 3) in pc/Myr (stellar unit system)
                    Note: 1 pc/Myr ≈ 0.978 km/s
    """
    masses: Float[Array, "N"]
    positions: Float[Array, "N 3"]
    velocities: Float[Array, "N 3"]

    @property
    def N(self) -> int:
        """Number of stars in the cluster."""
        return self.masses.shape[0]

    @property
    def M_total(self) -> float:
        """Total stellar mass in M_sun."""
        return float(jnp.sum(self.masses))


@dataclass(frozen=True)
class MassSegregationLayer:
    """
    Parameters for mass segregation layer (Baumgardt+2008).

    Attributes:
        lambda_seg: Smooth blending parameter in [0, 1]. Controls interpolation
                    between unsegregated baseline (lambda_seg=0) and fully
                    Baumgardt-segregated state (lambda_seg=1). This is the primary
                    knob for controlling mass segregation strength in v1.
        pool_factor: Multiplier for orbit pool size: N_pool = pool_factor * N_stars.
                     Must satisfy pool_factor >= 1 so that N_pool >= N_stars.
                     Higher values give better orbit sampling at the cost of memory.
                     Must be a static integer (not traced) for JIT compatibility.
                     Default 4; values < 4 may cause binning issues.

    Notes:
        In v1, mass segregation uses S=1 (full energy ordering) internally.
        The `lambda_seg` parameter provides continuous control by blending
        between an unsegregated baseline and the fully segregated state.

        **v1.5 extension**: Currently, mass-segregated ICs always use Q_vir=0.5
        (virial equilibrium). To support subvirial or supervirial Baumgardt ICs,
        add `virial_ratio: float = 0.5` to this class and update `generate_cluster_ic`
        to use `seg.virial_ratio`.
    """
    lambda_seg: float = 1.0
    pool_factor: int = 4


@dataclass(frozen=True)
class SpatialStructureParams:
    """
    Combined spatial structure parameters.

    Attributes:
        base_profile: Base density profile: "plummer", "king", or "eff"
        mass_segregation: If provided, apply mass segregation after base profile

    Notes:
        Turbulent/fractal substructure ICs are provided by the experimental
        ``gravoturb_fdf`` package (follow-up paper), not by released progenax.
    """
    base_profile: str = "plummer"
    mass_segregation: Optional[MassSegregationLayer] = None


# =============================================================================
# Helper Functions
# =============================================================================


def sample_velocities_for_profile(
    key: PRNGKeyArray,
    positions: Float[Array, "N 3"],
    profile: str,
    R_half: float,
    M_total: float,
    G: float,
    target_Q: float = 0.5,
    **kwargs,
) -> Float[Array, "N 3"]:
    """
    Sample velocities from equilibrium DF for the given profile.

    This is a wrapper around progenax.kinematics that creates the appropriate
    VelocityDF and samples velocities. Used for smooth profile ICs.

    Args:
        key: JAX random key
        positions: Particle positions (N, 3) [pc]
        profile: Profile name - "plummer", "king", or "eff"
        R_half: Half-mass/characteristic radius [pc]
        M_total: Total cluster mass [Msun]
        G: Gravitational constant in stellar units
        target_Q: Target virial ratio (default 0.5 for equilibrium)
        **kwargs: Profile-specific parameters (W0 for King, gamma for EFF)

    Returns:
        velocities: Particle velocities (N, 3) [pc/Myr]
    """
    from progenax.kinematics import (
        PlummerVelocityDF,
        KingVelocityDF,
        EFFVelocityDF,
        VelocityModel,
        sample_velocities_pipeline,
    )

    profile_lower = profile.lower()
    N = positions.shape[0]

    # Create appropriate VelocityDF
    if profile_lower == "plummer":
        df = PlummerVelocityDF(r_h=R_half)

    elif profile_lower == "king":
        W0 = kwargs.get("W0", 5.0)
        r_c = R_half
        # Approximate tidal radius
        r_t = r_c * (3.0 + W0)
        df = KingVelocityDF(W0=W0, r_c=r_c, r_t=r_t)

    elif profile_lower == "eff":
        gamma = kwargs.get("gamma", 3.0)
        r_t = kwargs.get("r_t", 10.0 * R_half)
        df = EFFVelocityDF(a=R_half, gamma=gamma, r_t=r_t)

    else:
        raise ValueError(f"Unknown profile: {profile}")

    # Create velocity model with target virial ratio
    model = VelocityModel(df=df, target_Q=target_Q)

    # Generate dummy masses (needed for virial rescaling)
    masses = jnp.ones(N) * (M_total / N)

    # Sample velocities
    return sample_velocities_pipeline(key, positions, masses, model, G=G)


# =============================================================================
# Structure-layer branches (extracted from generate_cluster_ic)
# =============================================================================


def _apply_segregation_branch(
    key, positions_base, velocities_base, masses, seg,
    profile, R_half, M_total, N_stars, G, **kwargs
):
    """Baumgardt mass-segregation branch. Returns (positions, velocities).

    Blends an unsegregated baseline with an energy-ordered (S=1) state by
    lambda_seg. The key is split three times internally, exactly as the inline
    branch did (RNG-preserving).
    """
    # Unsegregated baseline: equilibrium DF, masses assigned at random
    positions_unseg = positions_base
    velocities_unseg = velocities_base

    # Segregated state: Baumgardt with S=1 from an orbit pool (N_pool > N)
    N_pool = seg.pool_factor * N_stars

    key, subkey = random.split(key)
    pos_pool = sample_density_profile(subkey, N_pool, profile, R_half, **kwargs)

    key, subkey = random.split(key)
    vel_pool = sample_velocities_for_profile(
        subkey, pos_pool, profile, R_half, M_total, G, target_Q=0.5, **kwargs
    )

    def potential_fn(positions):
        return compute_profile_potential(positions, profile, M_total, R_half, G, **kwargs)

    key, subkey = random.split(key)
    _, positions_seg, velocities_seg = energy_sorted_segregation(
        subkey, masses, pos_pool, vel_pool, potential_fn
    )

    lambda_seg = seg.lambda_seg
    positions = (1.0 - lambda_seg) * positions_unseg + lambda_seg * positions_seg
    velocities = (1.0 - lambda_seg) * velocities_unseg + lambda_seg * velocities_seg
    return positions, velocities


# =============================================================================
# Main IC Generator
# =============================================================================


def generate_cluster_ic(
    key: PRNGKeyArray,
    N_stars: int,
    M_total: float,
    R_half: float,
    imf_params,
    structure_params: SpatialStructureParams,
    G: Optional[float] = None,
    **kwargs,
) -> ClusterState:
    """
    Generate complete cluster initial conditions.

    This is the main entry point for creating star cluster ICs following the
    v1.4 spec. Supports smooth profiles and mass segregation.

    Args:
        key: JAX random key for reproducibility
        N_stars: Number of stars in the cluster
        M_total: Total stellar mass in M_sun
        R_half: Half-mass radius in pc (Plummer) or characteristic radius (King/EFF)
        imf_params: IMF instance (e.g., PowerLawIMF.kroupa()) providing a
                    `.sample(key, n)` method for mass sampling
        structure_params: SpatialStructureParams with profile and layers
        G: Gravitational constant. If None, uses progenax.DEFAULT_UNITS.G
        **kwargs: Profile-specific parameters (W0 for King, gamma for EFF)

    Returns:
        ClusterState with masses, positions, velocities

    Notes:
        Follows the layering order from v1.4 spec:
            1. IMF → masses
            2. Base profile → positions_base, velocities_base
            3. Optional mass segregation layer
            4. Velocity finalization → rescale to Q_vir, remove COM

        The `lambda_seg` parameter blends between:
            - Unsegregated baseline: equilibrium DF with random mass assignment
            - Fully segregated state: Baumgardt energy ordering (S=1)

    Example:
        >>> from progenax.cluster import generate_cluster_ic, SpatialStructureParams
        >>> from progenax.cluster import MassSegregationLayer
        >>> from progenax.imf import PowerLawIMF
        >>> import jax
        >>>
        >>> key = jax.random.PRNGKey(42)
        >>> imf = PowerLawIMF.kroupa()
        >>>
        >>> # Mass-segregated Plummer cluster
        >>> cluster = generate_cluster_ic(
        ...     key=key,
        ...     N_stars=1000,
        ...     M_total=1000.0,
        ...     R_half=1.0,
        ...     imf_params=imf,
        ...     structure_params=SpatialStructureParams(
        ...         base_profile="plummer",
        ...         mass_segregation=MassSegregationLayer(lambda_seg=0.8),
        ...     ),
        ... )
    """
    # Handle G parameter
    if G is None:
        G = defaults.DEFAULT_UNITS.G

    profile = structure_params.base_profile
    seg = structure_params.mass_segregation

    # Smooth + mass-segregation ICs target virial equilibrium.
    target_Q_vir = 0.5

    # ─────────────────────────────────────────────────────────────
    # Step 1: Draw masses from IMF
    # ─────────────────────────────────────────────────────────────
    key, subkey = random.split(key)
    masses = imf_params.sample(subkey, N_stars)
    masses = masses * (M_total / jnp.sum(masses))  # Normalize to M_total

    # ─────────────────────────────────────────────────────────────
    # Step 2: Generate base profile positions and velocities
    # ─────────────────────────────────────────────────────────────
    key, subkey = random.split(key)
    positions_base = sample_density_profile(subkey, N_stars, profile, R_half, **kwargs)

    key, subkey = random.split(key)
    velocities_base = sample_velocities_for_profile(
        subkey, positions_base, profile, R_half, M_total, G, target_Q=0.5, **kwargs
    )

    # ─────────────────────────────────────────────────────────────
    # Apply the requested structure layer (mass segregation / none).
    # ─────────────────────────────────────────────────────────────
    if seg is not None:
        positions, velocities = _apply_segregation_branch(
            key, positions_base, velocities_base, masses, seg,
            profile, R_half, M_total, N_stars, G, **kwargs
        )
    else:
        positions = positions_base
        velocities = velocities_base

    # ─────────────────────────────────────────────────────────────
    # Step 3: Final adjustments (COM removal and virial scaling)
    # ─────────────────────────────────────────────────────────────
    # Remove bulk COM position and velocity
    M_total_actual = jnp.sum(masses)
    x_com = jnp.sum(masses[:, None] * positions, axis=0) / M_total_actual
    positions = positions - x_com

    v_com = jnp.sum(masses[:, None] * velocities, axis=0) / M_total_actual
    velocities = velocities - v_com

    # Rescale velocities to exact target Q_vir
    velocities = rescale_velocities_to_virial(
        positions, velocities, masses, G=G, target_Q=target_Q_vir
    )

    return ClusterState(
        masses=masses,
        positions=positions,
        velocities=velocities,
    )


__all__ = [
    "ClusterState",
    "MassSegregationLayer",
    "SpatialStructureParams",
    "generate_cluster_ic",
    "sample_velocities_for_profile",
]
