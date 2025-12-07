"""Mass segregation transforms for star cluster ICs.

Implements primordial mass segregation where massive stars
are preferentially located near the cluster center.

References:
    Baumgardt et al. (2008) MNRAS 384, 1231 - Primordial mass segregation
    Subr et al. (2008) A&A 487, 671 - Mass segregation in young clusters
    de Grijs et al. (2002) MNRAS 331, 245 - NGC 330 mass segregation
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple


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


def apply_mass_segregation_baumgardt(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    s: float,
    key: PRNGKeyArray,
    G: float,
) -> Tuple[Float[Array, "N 3"], Float[Array, "N 3"]]:
    """Apply Baumgardt-style mass segregation via energy-ranked orbit assignment.

    This implements the McLuster/Baumgardt et al. (2008) algorithm where:
    1. Orbits are ranked by binding energy (most bound = most negative first)
    2. Masses are sorted descending (most massive first)
    3. Each mass is assigned to an orbit using slot selection with parameter s

    The parameter s controls segregation strength:
        - s = 0: Random assignment (no segregation)
        - s = 1: Maximal segregation (most massive → most bound)
        - 0 < s < 1: Intermediate segregation

    Algorithm uses jax.lax.scan for JAX-native, differentiable orbit assignment.

    Args:
        positions: Initial positions (N, 3)
        velocities: Initial velocities (N, 3)
        masses: Particle masses (N,)
        s: Segregation parameter [0, 1] (0=random, 1=maximal)
        key: JAX random key for stochastic slot selection
        G: Gravitational constant (for potential energy calculation)

    Returns:
        Segregated positions (N, 3) and velocities (N, 3)

    References:
        Baumgardt et al. (2008) MNRAS 384, 1231 - Primordial mass segregation
        Kupper et al. (2011) MNRAS 417, 2300 - McLuster implementation

    Note:
        This function is fully differentiable (uses jax.lax.scan, not while_loop)
        and JIT-compatible. Total mass and orbital structure are preserved,
        only the assignment of masses to orbits changes.
    """
    N = positions.shape[0]

    # Step 1: Compute binding energies E = 0.5*v^2 + Phi(r)
    # Use simple -G*M/r potential (approximate)
    r = jnp.linalg.norm(positions, axis=1)
    v2 = jnp.sum(velocities**2, axis=1)

    # Approximate potential: Phi(r) = -G*M_total/r
    # (In reality, should be sum over all pairs, but this is faster approximation)
    M_total = jnp.sum(masses)
    phi = -G * M_total / jnp.maximum(r, 1e-10)  # Avoid division by zero

    binding_energy = 0.5 * v2 + phi  # More negative = more bound

    # Step 2: Sort orbits by energy (most bound = most negative first)
    energy_sort_idx = jnp.argsort(binding_energy)  # Ascending (most negative first)

    # Step 3: Sort masses descending (most massive first)
    mass_sort_idx = jnp.argsort(-masses)  # Descending

    # Step 4: Assign masses to orbits using scan-based slot selection
    def assign_step(carry, i):
        """Assign mass i to an available orbit slot."""
        available_mask, assignments, subkey = carry
        subkey, new_subkey = jax.random.split(subkey)

        # Number of available slots
        n_available = jnp.sum(available_mask)

        # Sample slot index using power-law distribution
        # j = floor((n_available - 1) * (1 - U^(1-s)))
        U = jax.random.uniform(subkey)
        slot_frac = 1.0 - jnp.power(U, 1.0 - s + 1e-10)  # Add small epsilon for numerical stability
        slot_index = jnp.floor((n_available - 1) * slot_frac).astype(jnp.int32)
        slot_index = jnp.clip(slot_index, 0, N - 1)  # Safety clipping

        # Find the slot_index-th available orbit using cumsum trick
        cumsum = jnp.cumsum(available_mask)
        # Find first position where cumsum > slot_index
        orbit_idx = jnp.argmax(cumsum > slot_index)

        # If slot_index is 0, we want the first available orbit
        # Handle edge case: if all cumsum <= slot_index, take last available
        orbit_idx = jnp.where(
            slot_index == 0,
            jnp.argmax(available_mask),  # First available
            orbit_idx
        )

        # Mark this orbit as used
        new_mask = available_mask.at[orbit_idx].set(False)

        # Record assignment: mass i goes to orbit orbit_idx
        new_assignments = assignments.at[i].set(orbit_idx)

        return (new_mask, new_assignments, new_subkey), None

    # Initialize carry: all orbits available, no assignments yet
    initial_mask = jnp.ones(N, dtype=bool)
    initial_assignments = jnp.zeros(N, dtype=jnp.int32)

    # Scan over all masses (in descending order by mass)
    (final_mask, final_assignments, _), _ = jax.lax.scan(
        assign_step,
        (initial_mask, initial_assignments, key),
        jnp.arange(N)
    )

    # Step 5: Apply assignments
    # final_assignments[i] tells us which orbit (in energy-sorted order) mass i gets
    # mass_sort_idx[i] is the original index of the i-th most massive star
    # energy_sort_idx[j] is the original index of the j-th most bound orbit

    # Build the permutation: for each mass (in descending order),
    # assign it to the selected orbit
    positions_out = jnp.zeros_like(positions)
    velocities_out = jnp.zeros_like(velocities)

    # For each slot i (most massive to least massive)
    for i in range(N):
        mass_idx = mass_sort_idx[i]  # Original index of i-th most massive star
        orbit_slot = final_assignments[i]  # Which orbit slot (in energy order)
        orbit_idx = energy_sort_idx[orbit_slot]  # Original index of that orbit

        # Assign this mass to this orbit's position/velocity
        positions_out = positions_out.at[mass_idx].set(positions[orbit_idx])
        velocities_out = velocities_out.at[mass_idx].set(velocities[orbit_idx])

    # JAX-friendly version using vmap instead of Python loop
    def assign_particle(i):
        mass_idx = mass_sort_idx[i]
        orbit_slot = final_assignments[i]
        orbit_idx = energy_sort_idx[orbit_slot]
        return mass_idx, orbit_idx

    mass_indices, orbit_indices = jax.vmap(assign_particle)(jnp.arange(N))

    # Use advanced indexing to reassign
    positions_out = positions[orbit_indices]
    velocities_out = velocities[orbit_indices]

    # Need to reorder back to original mass order
    # Create inverse permutation
    positions_final = jnp.zeros_like(positions)
    velocities_final = jnp.zeros_like(velocities)

    positions_final = positions_final.at[mass_indices].set(positions_out)
    velocities_final = velocities_final.at[mass_indices].set(velocities_out)

    return positions_final, velocities_final


__all__ = [
    "apply_mass_segregation",
    "compute_mass_segregation_ratio",
    "apply_mass_segregation_baumgardt",
]
