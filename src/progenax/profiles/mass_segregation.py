# progenax/src/progenax/profiles/mass_segregation.py
"""
Primordial mass segregation for star cluster initial conditions.

Implements scientifically correct mass segregation algorithms:
- Baumgardt/McLuster energy-ranked orbit assignment (Baumgardt+ 2008, Küpper+ 2011)
- Allison+ (2009) MST-based mass segregation ratio Λ_MSR diagnostic
- Subr+ (2008) placeholder for future implementation

References:
    Baumgardt et al. (2008), MNRAS, 384, 1231 - Energy-ranked orbit assignment
    Küpper et al. (2011), MNRAS, 417, 2300 - McLuster slot selection algorithm
    Allison et al. (2009), MNRAS, 395, 1449 - MST-based Λ_MSR diagnostic
    Subr, Kroupa & Baumgardt (2008), A&A, 487, 671 - Target potential sequence
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


# =============================================================================
# Internal utility functions
# =============================================================================


def _pairwise_distances(
    positions: Float[Array, "N 3"],
    eps: float = 0.01,
) -> Float[Array, "N N"]:
    """
    Compute O(N²) pairwise softened distance matrix.

    Uses Plummer softening: d_ij = sqrt(|r_i - r_j|² + eps²)

    Args:
        positions: Particle positions (N, 3)
        eps: Softening length [same units as positions]

    Returns:
        Symmetric distance matrix (N, N) with diagonal = eps
    """
    # Compute squared distances: (N, N)
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    r2 = jnp.sum(diff**2, axis=-1)  # (N, N)

    # Softened distance
    return jnp.sqrt(r2 + eps**2)


def _softened_potential(
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    eps: float = 0.01,
) -> Float[Array, "N"]:
    """
    Compute O(N²) gravitational potential at each particle.

    Φ_i = -G × Σ_{j≠i} m_j / d_ij

    Uses Plummer softening to avoid singularities.

    Args:
        positions: Particle positions (N, 3)
        masses: Particle masses (N,)
        G: Gravitational constant
        eps: Softening length [same units as positions]

    Returns:
        Potential at each particle (N,) - all values should be negative
    """
    N = positions.shape[0]

    # Get softened distances
    dist = _pairwise_distances(positions, eps)  # (N, N)

    # Potential contribution: -G * m_j / d_ij
    # Exclude self-interaction by setting diagonal to inf before division
    dist_safe = jnp.where(jnp.eye(N, dtype=bool), jnp.inf, dist)

    # Φ_i = -G × Σ_j m_j / d_ij (excluding i=j)
    phi = -G * jnp.sum(masses[None, :] / dist_safe, axis=1)

    return phi


def _mst_length(positions: Float[Array, "N 3"]) -> Float[Array, ""]:
    """
    Compute Minimum Spanning Tree length using Prim's algorithm.

    JAX-native implementation using jax.lax.fori_loop for JIT compatibility.

    Args:
        positions: Particle positions (N, 3)

    Returns:
        Total MST edge length (scalar)
    """
    N = positions.shape[0]

    # Handle edge cases
    if N <= 1:
        return jnp.array(0.0)

    # Compute pairwise distances (no softening for MST)
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))  # (N, N)

    # Prim's algorithm: start from vertex 0
    # in_tree[i] = True if vertex i is in the MST
    # min_edge[i] = minimum edge weight from tree to vertex i

    in_tree = jnp.zeros(N, dtype=bool).at[0].set(True)
    min_edge = dist[0, :]  # Initial distances from vertex 0
    min_edge = min_edge.at[0].set(jnp.inf)  # Vertex 0 is already in tree

    def prim_step(carry, _):
        """Add one vertex to MST."""
        in_tree_curr, min_edge_curr, total_length = carry

        # Find minimum edge to a vertex not in tree
        masked_edges = jnp.where(in_tree_curr, jnp.inf, min_edge_curr)
        new_vertex = jnp.argmin(masked_edges)
        edge_length = masked_edges[new_vertex]

        # Add vertex to tree
        in_tree_new = in_tree_curr.at[new_vertex].set(True)

        # Update minimum edges
        new_distances = dist[new_vertex, :]
        min_edge_new = jnp.minimum(min_edge_curr, new_distances)

        # Add edge length to total
        total_new = total_length + edge_length

        return (in_tree_new, min_edge_new, total_new), None

    # Run N-1 iterations to add all vertices
    (_, _, total_length), _ = jax.lax.scan(
        prim_step,
        (in_tree, min_edge, jnp.array(0.0)),
        jnp.arange(N - 1),
    )

    return total_length


# =============================================================================
# Public API: Mass Segregation Ratio (Allison+ 2009)
# =============================================================================


def mass_segregation_ratio_mst(
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    n_massive: int,
    n_random: int,
    key: PRNGKeyArray,
) -> Dict[str, Float[Array, ""]]:
    """
    Compute mass segregation ratio Λ_MSR using MST method (Allison+ 2009).

    Λ_MSR = <L_random> / L_massive

    where L_massive is the MST length of the n_massive most massive stars,
    and <L_random> is the average MST length of n_random random subsets
    of the same size.

    Λ_MSR > 1 indicates mass segregation (massive stars more clustered).
    Λ_MSR ≈ 1 indicates no segregation.
    Λ_MSR < 1 would indicate inverse segregation (rare).

    Args:
        positions: Particle positions (N, 3)
        masses: Particle masses (N,)
        n_massive: Number of most massive stars to consider
        n_random: Number of random subsets for comparison
        key: JAX random key

    Returns:
        Dictionary with:
            - lambda_msr: Mass segregation ratio
            - lambda_err: Uncertainty from random sampling
            - l_massive: MST length of massive stars
            - l_random_mean: Mean MST length of random samples

    Raises:
        ValueError: If n_massive < 2 or n_massive >= N

    References:
        Allison et al. (2009), MNRAS, 395, 1449, Eq. 1
    """
    N = positions.shape[0]

    # Validate inputs
    if n_massive < 2:
        raise ValueError(f"n_massive must be >= 2, got {n_massive}")
    if n_massive >= N:
        raise ValueError(f"n_massive ({n_massive}) must be < N ({N})")

    # Find n_massive most massive stars
    mass_order = jnp.argsort(-masses)  # Descending
    massive_idx = mass_order[:n_massive]
    pos_massive = positions[massive_idx]

    # MST length of massive stars
    l_massive = _mst_length(pos_massive)

    # Generate random samples and compute their MST lengths
    def sample_random_mst(subkey):
        """Sample n_massive random stars and compute MST length."""
        idx = jax.random.choice(subkey, N, shape=(n_massive,), replace=False)
        pos_sample = positions[idx]
        return _mst_length(pos_sample)

    keys = jax.random.split(key, n_random)
    l_random_samples = jax.vmap(sample_random_mst)(keys)

    # Statistics
    l_random_mean = jnp.mean(l_random_samples)
    l_random_std = jnp.std(l_random_samples)

    # Mass segregation ratio
    lambda_msr = l_random_mean / (l_massive + 1e-10)

    # Uncertainty propagation: σ_Λ = σ_random / L_massive
    lambda_err = l_random_std / (l_massive + 1e-10)

    return {
        "lambda_msr": lambda_msr,
        "lambda_err": lambda_err,
        "l_massive": l_massive,
        "l_random_mean": l_random_mean,
    }


# =============================================================================
# Public API: Baumgardt/McLuster Energy-Ranked Orbit Assignment
# =============================================================================


def apply_mass_segregation_baumgardt(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    s: float,
    key: PRNGKeyArray,
    G: float | None = None,
    eps: float = 0.01,
    Q_target: float = 0.5,
) -> Tuple[Float[Array, "N 3"], Float[Array, "N 3"]]:
    """
    Apply Baumgardt/McLuster-style primordial mass segregation.

    Implements energy-ranked orbit assignment (Baumgardt+ 2008, Küpper+ 2011):
    1. Compute binding energy E_i = 0.5*v_i² + Φ_i for each particle
    2. Rank orbits by energy (most bound = most negative first)
    3. Assign masses to orbits using stochastic slot selection with parameter s
    4. Rescale velocities to achieve target virial ratio Q_target
    5. Re-center on center of mass

    The parameter s controls segregation strength:
        - s = 0: Random orbit assignment (no segregation)
        - s = 1: Maximal segregation (most massive → most bound orbit)
        - 0 < s < 1: Intermediate segregation

    Args:
        positions: Input positions (N, 3) [length units]
        velocities: Input velocities (N, 3) [velocity units]
        masses: Particle masses (N,) [mass units]
        s: Segregation parameter in [0, 1]
        key: JAX random key for stochastic slot selection
        G: Gravitational constant. If None, uses jaxstro.units.DEFAULT.G
        eps: Softening length for potential calculation [length units]
        Q_target: Target virial ratio Q = K/|U| (default: 0.5 for equilibrium)

    Returns:
        Tuple of (positions_out, velocities_out) with:
            - Masses reassigned to energy-ranked orbits
            - Velocities rescaled to Q_target
            - System centered on COM

    References:
        Baumgardt et al. (2008), MNRAS, 384, 1231
        Küpper et al. (2011), MNRAS, 417, 2300
    """
    # Handle G parameter following progenax pattern
    if G is None:
        from jaxstro.units import DEFAULT

        G = DEFAULT.G

    N = positions.shape[0]

    # Step 1: Compute binding energies using O(N²) potential
    phi = _softened_potential(positions, masses, G, eps)
    v2 = jnp.sum(velocities**2, axis=1)
    E = 0.5 * v2 + phi  # More negative = more bound

    # Step 2: Rank orbits by energy (ascending = most bound first)
    energy_order = jnp.argsort(E)

    # Step 3: Rank masses (descending = most massive first)
    mass_order = jnp.argsort(-masses)

    # Step 4: Stochastic slot selection using scan
    # Each mass i selects from remaining orbits with bias toward low-energy slots

    def assign_step(carry, i):
        """Assign mass i to an available orbit slot."""
        available_mask, assignments, subkey = carry
        subkey, new_subkey = jax.random.split(subkey)

        # Count available slots
        n_available = jnp.sum(available_mask)

        # Sample slot using power-law distribution
        # slot_index ~ floor((n_available - 1) * (1 - U^(1-s)))
        # This gives bias toward slot 0 (most bound) as s → 1
        U = jax.random.uniform(subkey)
        slot_frac = 1.0 - jnp.power(U, 1.0 - s + 1e-10)
        slot_index = jnp.floor((n_available - 1) * slot_frac).astype(int)
        slot_index = jnp.clip(slot_index, 0, N - 1)

        # Find the slot_index-th available orbit
        cumsum = jnp.cumsum(available_mask)
        orbit_idx = jnp.argmax(cumsum > slot_index)

        # Handle edge case: slot_index == 0
        orbit_idx = jnp.where(
            slot_index == 0,
            jnp.argmax(available_mask),
            orbit_idx,
        )

        # Mark orbit as used
        new_mask = available_mask.at[orbit_idx].set(False)
        new_assignments = assignments.at[i].set(orbit_idx)

        return (new_mask, new_assignments, new_subkey), None

    # Initialize: all orbits available
    initial_mask = jnp.ones(N, dtype=bool)
    initial_assignments = jnp.zeros(N, dtype=int)

    (_, final_assignments, _), _ = jax.lax.scan(
        assign_step,
        (initial_mask, initial_assignments, key),
        jnp.arange(N),
    )

    # Step 5: Build output arrays
    # final_assignments[i] = orbit slot (in energy-sorted order) for mass i
    # mass_order[i] = original index of i-th most massive star
    # energy_order[j] = original index of j-th most bound orbit

    def assign_particle(i):
        mass_idx = mass_order[i]
        orbit_slot = final_assignments[i]
        orbit_idx = energy_order[orbit_slot]
        return mass_idx, orbit_idx

    mass_indices, orbit_indices = jax.vmap(assign_particle)(jnp.arange(N))

    # Reassign orbits
    positions_out = jnp.zeros_like(positions)
    velocities_out = jnp.zeros_like(velocities)

    positions_out = positions_out.at[mass_indices].set(positions[orbit_indices])
    velocities_out = velocities_out.at[mass_indices].set(velocities[orbit_indices])

    # Step 6: Rescale velocities to target virial ratio
    phi_out = _softened_potential(positions_out, masses, G, eps)
    U = 0.5 * jnp.sum(masses * phi_out)  # Total potential energy
    K = 0.5 * jnp.sum(masses * jnp.sum(velocities_out**2, axis=1))

    # Current virial ratio: Q = T/|V| (equilibrium at 0.5)
    Q_current = K / jnp.abs(U)

    # Scale velocities: v_new = v_old * sqrt(Q_target / Q_current)
    scale = jnp.sqrt(Q_target / (Q_current + 1e-10))
    velocities_out = velocities_out * scale

    # Step 7: Re-center on center of mass
    M_total = jnp.sum(masses)
    x_com = jnp.sum(masses[:, None] * positions_out, axis=0) / M_total
    v_com = jnp.sum(masses[:, None] * velocities_out, axis=0) / M_total

    positions_out = positions_out - x_com
    velocities_out = velocities_out - v_com

    return positions_out, velocities_out


# =============================================================================
# Public API: Subr+ (2008) Placeholder
# =============================================================================


def generate_mass_segregated_ic_subr(*args, **kwargs):
    """
    Generate mass-segregated ICs using Subr–Kroupa–Baumgardt (2008) method.

    This method generates a new DF-consistent set of positions and velocities
    with prescribed mass segregation, rather than reassigning orbits.

    NOT YET IMPLEMENTED.

    References:
        Subr, Kroupa & Baumgardt (2008), A&A, 487, 671

    Raises:
        NotImplementedError: This method is not yet implemented.
    """
    raise NotImplementedError(
        "Subr–Kroupa–Baumgardt (2008) mass segregation is not yet implemented. "
        "Use apply_mass_segregation_baumgardt() for orbit reassignment method."
    )


__all__ = [
    "mass_segregation_ratio_mst",
    "apply_mass_segregation_baumgardt",
    "generate_mass_segregated_ic_subr",
]
