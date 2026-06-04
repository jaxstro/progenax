"""
IC builders for progenax.

Provides:
- ICResult: Dataclass for initial condition outputs
- build_spatial_ic: Generic builder using profile + velocity DF
- to_com_frame: Center-of-mass frame transformation
- virial_scale: Scale velocities to target virial ratio
- compute_stellar_radii: Estimate stellar radii from mass
- compute_kinetic_energy, compute_potential_energy: Energy helpers
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from .protocols import SpatialProfile, VelocityDF
# Single canonical energy implementation lives in dynamics.virial; re-export it
# here so the public API (progenax.compute_*_energy) and virial_scale share one
# gradient-safe source of truth (Batch 0, F1+F2).
from .dynamics.virial import compute_kinetic_energy, compute_potential_energy
from .binaries import (
    resolve_binary_components,
    sample_isotropic_orientations,
    period_to_semimajor_axis,
)

# Seconds in one (SI) day — exact; used to convert sampled periods (days) into the
# code time unit via units.time_scale_cgs.
_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class ICResult:
    """
    Result from initial conditions generation — pure physical state.

    Immutable dataclass containing all particle data. No dependency on gravax;
    can be converted to any state format.

    Softening is intentionally **NOT** stored here: it is a force-model /
    integration choice (selected on the integrator, e.g. ε=0 for collisional
    Hermite/IAS15), not a property of the initial conditions.

    Attributes:
        positions: Particle positions (N, 3) [length units]
        velocities: Particle velocities (N, 3) [velocity units]
        masses: Particle masses (N,) [M_sun]
        stellar_radii: Stellar radii (N,) [R_sun]
        ids: Particle IDs (N,) or None
        primordial_system_id: (N,) int — which primordial system each particle
            belongs to (paired particles share an id); None for single-only ICs.
            **PROVENANCE at t=0 only** — goes stale under dynamical evolution
            (ionization / formation / exchange). Measure the *current* binary
            population with `binaries.diagnostics.find_bound_pairs`, not this.
        is_primordial_secondary: (N,) bool — True for the secondary of a
            primordial binary; None for single-only ICs.
    """

    positions: Float[Array, "N 3"]
    velocities: Float[Array, "N 3"]
    masses: Float[Array, "N"]
    stellar_radii: Float[Array, "N"]
    ids: Optional[Float[Array, "N"]] = None
    primordial_system_id: Optional[Int[Array, "N"]] = None
    is_primordial_secondary: Optional[Bool[Array, "N"]] = None


def compute_stellar_radii(masses: Float[Array, "N"]) -> Float[Array, "N"]:
    """
    Estimate stellar radii from mass (main-sequence + brown dwarfs).

    Returns radii in SOLAR RADII (R☉).

    R/R☉ relations (3 regimes):
    - M > 1 M☉: R ∝ M^0.8 (massive stars)
    - 0.08 ≤ M ≤ 1 M☉: R ∝ M^0.57 (low-mass main sequence)
    - M < 0.08 M☉: R ∝ M^0.08 (brown dwarfs, ~0.1 R☉)

    Args:
        masses: Particle masses (N,) [M☉]

    Returns:
        Radii in R☉
    """

    def radius_high_mass(m):
        return jnp.power(m, 0.8)

    def radius_low_mass(m):
        return jnp.power(m, 0.57)

    def radius_brown_dwarf(m):
        return 0.1 * jnp.power(m / 0.08, 0.08)

    radii = jax.vmap(
        lambda m: jax.lax.cond(
            m > 1.0,
            radius_high_mass,
            lambda mv: jax.lax.cond(
                mv >= 0.08, radius_low_mass, radius_brown_dwarf, mv
            ),
            m,
        )
    )(masses)

    return radii


def to_com_frame(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
) -> tuple[Float[Array, "N 3"], Float[Array, "N 3"]]:
    """
    Transform to center-of-mass frame.

    Args:
        positions: Particle positions (N, 3)
        velocities: Particle velocities (N, 3)
        masses: Particle masses (N,)

    Returns:
        (positions_com, velocities_com): Transformed coordinates
    """
    M_total = jnp.sum(masses)

    # COM position and velocity
    r_com = jnp.sum(positions * masses[:, None], axis=0) / M_total
    v_com = jnp.sum(velocities * masses[:, None], axis=0) / M_total

    return positions - r_com, velocities - v_com


def virial_scale(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    Q_target: float,
    G: float,
    softening: float = 0.0,
) -> Float[Array, "N 3"]:
    """
    Scale velocities to achieve target virial ratio Q = T/|V|.

    Physical interpretation:
        - Q = 0.5: Virial equilibrium (2T + V = 0)
        - Q < 0.5: Sub-virial (cold), system will collapse
        - Q > 0.5: Super-virial (hot), system will expand/unbind

    Args:
        positions: Particle positions (N, 3)
        velocities: Particle velocities (N, 3)
        masses: Particle masses (N,)
        Q_target: Target virial ratio (0.5 for equilibrium)
        G: Gravitational constant
        softening: Softening length (default: 0)

    Returns:
        Scaled velocities

    References:
        Goodwin & Whitworth (2004) A&A 413, 929 - Sub-virial clusters
        Baumgardt & Kroupa (2007) MNRAS 380, 1589 - Cluster dissolution
    """
    T = compute_kinetic_energy(velocities, masses)
    V = compute_potential_energy(positions, masses, G=G, softening=softening)

    # Q = T / |V| (NOT 2T / |V|)
    Q_current = T / jnp.abs(V)
    scale = jnp.sqrt(Q_target / Q_current)

    return velocities * scale


def build_spatial_ic(
    profile: SpatialProfile,
    masses: Float[Array, "N"],
    velocity_df: VelocityDF,
    key: PRNGKeyArray,
    G: float,
    Q: Optional[float] = 0.5,
    softening: float = 0.0,
    id_offset: int = 0,
) -> ICResult:
    """
    Build initial conditions from spatial profile and velocity DF.

    Args:
        profile: Spatial density profile (must implement SpatialProfile protocol)
        masses: Particle masses (N,) [M_sun]
        velocity_df: Velocity distribution function (must implement VelocityDF protocol)
        key: JAX random key
        G: Gravitational constant (REQUIRED - no default)
        Q: Virial ratio target Q = T/|V| (0.5 for equilibrium, None to disable)
        softening: Softening length used ONLY to virial-scale the IC under the same
            force law it will be integrated with (default: 0.0 = exact Newtonian,
            matching the analytic equilibrium DFs and collisional integration).
            Pass ε>0 for a collisionless science case so the IC is virialized
            consistently. This is a force-model knob (a future shared `ForceModel`
            will supply it); it is **not** stored on the returned `ICResult`.
        id_offset: Offset for particle IDs (default: 0)

    Returns:
        ICResult (pure physical state — no softening field)
    """
    N = masses.shape[0]

    # Split keys
    key_pos, key_vel = jax.random.split(key)

    # 1. Sample positions
    positions = profile.sample_positions(masses, key_pos)

    # 2. Sample velocities (G must be threaded through — see audit C1; without it
    #    the DF silently falls back to DEFAULT_UNITS.G and ignores the caller's units)
    velocities = velocity_df.sample_velocities(positions, masses, key_vel, G=G)

    # 3. Compute stellar radii
    stellar_radii = compute_stellar_radii(masses)

    # 4. Transform to COM frame
    positions, velocities = to_com_frame(positions, velocities, masses)

    # 5. Apply virial scaling under the given force law (ε=0 by default — exact
    #    Newtonian, consistent with the analytic equilibrium DFs).
    if Q is not None:
        velocities = virial_scale(positions, velocities, masses, Q, G, softening)

    # 6. Generate IDs
    ids = jnp.arange(id_offset, id_offset + N) if id_offset != 0 else None

    return ICResult(
        positions=positions,
        velocities=velocities,
        masses=masses,
        stellar_radii=stellar_radii,
        ids=ids,
    )


def build_binary_cluster(
    profile: SpatialProfile,
    velocity_df: VelocityDF,
    binary_imf,
    period_dist,
    ecc_dist,
    n_systems: int,
    key: PRNGKeyArray,
    *,
    units,
    Q: Optional[float] = 0.5,
    softening: float = 0.0,
    compact: bool = True,
):
    """Assemble a star cluster with a primordial binary population.

    Wires the three domains: (1) `binary_imf.sample_systems` -> primary/secondary
    masses + binary flags; (2) `build_spatial_ic` on the **system** masses (m1+m2)
    -> COM positions/velocities (virialized at the system level under the given
    softening, ε=0 by default); (3) sample period/eccentricity/orientation and
    convert period -> semi-major axis; (4) `resolve_binary_components` places the
    two components of each binary around its COM (COM preserved exactly).

    Binaries are **collisional**: integrate the result with a collisional integrator
    (Hermite/IAS15, softening=0). The COM virialization treats binaries as point
    masses; internal binary energy is separate and untouched by `Q`.

    Args:
        profile, velocity_df: spatial profile + velocity DF for the system COMs.
        binary_imf: object with `sample_systems(key, n) -> (m1, m2, is_binary)`
            (e.g. `progenax.imf.binary.BinaryIMF`).
        period_dist: period distribution with `sample(key, n) -> periods [days]`.
        ecc_dist: **unconditional** eccentricity distribution with
            `sample(key, n) -> e` (period/mass-conditional Moe is wired in 4i).
        n_systems: number of stellar *systems* (singles + binaries).
        key: JAX random key.
        units: `UnitSystem` (carries G and the time scale for the day->time-unit
            conversion).
        Q: system-level virial ratio target (0.5 = equilibrium; None to disable).
        softening: virial-scaling softening for the COM cluster (default 0 = exact;
            NOT stored — see `build_spatial_ic`).
        compact: True (default) -> eagerly compacted `ICResult` of real particles;
            False -> the masked fixed-shape `ResolvedBinaries` (jit/grad-safe).

    Returns:
        `ICResult` (compact=True) or `ResolvedBinaries` (compact=False).
    """
    G = units.G
    day_in_time_units = _SECONDS_PER_DAY / units.time_scale_cgs

    key_sys, key_spatial, key_P, key_e, key_orient = jax.random.split(key, 5)

    # 1. System masses + binary flags.
    m1, m2, is_binary = binary_imf.sample_systems(key_sys, n_systems)
    system_masses = m1 + m2

    # 2. System COMs (virialized treating binaries as point masses at COM).
    ic_sys = build_spatial_ic(
        profile, system_masses, velocity_df, key_spatial, G, Q=Q, softening=softening
    )
    com_pos, com_vel = ic_sys.positions, ic_sys.velocities

    # 3. Orbital elements for every system (singles are sanitized in the primitive).
    periods_days = period_dist.sample(key_P, n_systems)
    e = ecc_dist.sample(key_e, n_systems)
    inc, Omega, omega, M_anom = sample_isotropic_orientations(key_orient, n_systems)
    a = period_to_semimajor_axis(periods_days * day_in_time_units, system_masses, G)

    # 4. Resolve binaries into the masked 2N representation.
    resolved = resolve_binary_components(
        com_pos, com_vel, m1, m2, is_binary, a, e, inc, Omega, omega, M_anom, G=G
    )

    if not compact:
        return resolved

    # 5. Eager compaction -> real-particle ICResult (one-shot; dynamic shape).
    mask = resolved.is_real
    masses = resolved.masses[mask]
    return ICResult(
        positions=resolved.positions[mask],
        velocities=resolved.velocities[mask],
        masses=masses,
        stellar_radii=compute_stellar_radii(masses),
        ids=jnp.arange(masses.shape[0]),
        primordial_system_id=resolved.primordial_system_id[mask],
        is_primordial_secondary=resolved.is_primordial_secondary[mask],
    )


__all__ = [
    "ICResult",
    "compute_stellar_radii",
    "compute_kinetic_energy",
    "compute_potential_energy",
    "to_com_frame",
    "virial_scale",
    "build_spatial_ic",
    "build_binary_cluster",
]
