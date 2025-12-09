# progenax/src/progenax/cluster/__init__.py
"""
Star cluster initial condition generator (v1.4 spec).

This module provides tools for generating realistic initial conditions for
star cluster simulations with optional mass segregation and fractal substructure.

Main Entry Point:
    generate_cluster_ic: Generate complete cluster IC from parameters

Data Classes:
    ClusterState: Immutable container for masses, positions, velocities
    SpatialStructureParams: Combined profile + structure parameters
    MassSegregationLayer: Baumgardt+2008 mass segregation parameters
    FractalLayer: Goodwin-Whitworth+2004 fractal substructure parameters

Example:
    >>> from progenax.cluster import (
    ...     generate_cluster_ic,
    ...     ClusterState,
    ...     SpatialStructureParams,
    ...     MassSegregationLayer,
    ... )
    >>> from progenax.imf import PowerLawIMF
    >>> import jax
    >>>
    >>> key = jax.random.PRNGKey(42)
    >>> imf = PowerLawIMF.kroupa()
    >>>
    >>> # Generate mass-segregated Plummer cluster
    >>> cluster = generate_cluster_ic(
    ...     key=key,
    ...     N_stars=1000,
    ...     M_total=1000.0,  # Msun
    ...     R_half=1.0,       # pc
    ...     imf_params=imf,
    ...     structure_params=SpatialStructureParams(
    ...         base_profile="plummer",
    ...         mass_segregation=MassSegregationLayer(lambda_seg=0.8),
    ...     ),
    ... )
    >>> print(f"N={cluster.N}, M={cluster.M_total:.0f} Msun")

Notes:
    - All code is JAX-native (jax.numpy, jax.lax.scan, etc.)
    - Mass segregation and fractal layers are mutually exclusive in v1
    - Diagnostics (Λ_MSR, Q parameter) are in progenax.diagnostics

References:
    Baumgardt, De Marchi & Kroupa (2008), ApJ 685, 247
    Goodwin & Whitworth (2004), A&A 413, 929
    Küpper et al. (2011), MNRAS 417, 2300 - McLuster
    Allison et al. (2009), ApJ 700, L99
"""

from progenax.cluster.core import (
    ClusterState,
    MassSegregationLayer,
    FractalLayer,
    SpatialStructureParams,
    generate_cluster_ic,
    sample_velocities_for_profile,
)

from progenax.cluster.mass_segregation import energy_sorted_segregation

# Legacy GW2004 implementation (deprecated - use FDF instead)
from progenax.cluster.fractal_gw_legacy import (
    generate_fractal_positions,
    rescale_fractal_to_target_radii,
    assign_velocities_and_virialize,
)

__all__ = [
    # Core API
    "ClusterState",
    "MassSegregationLayer",
    "FractalLayer",
    "SpatialStructureParams",
    "generate_cluster_ic",
    "sample_velocities_for_profile",
    # Mass segregation
    "energy_sorted_segregation",
    # Fractal substructure
    "generate_fractal_positions",
    "rescale_fractal_to_target_radii",
    "assign_velocities_and_virialize",
]
