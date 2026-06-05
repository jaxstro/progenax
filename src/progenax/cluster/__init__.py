# progenax/src/progenax/cluster/__init__.py
"""
Star cluster initial condition generator.

This module provides tools for generating realistic initial conditions for
star cluster simulations with optional mass segregation.

Turbulent/fractal substructure ICs (the gravoturbulent + fractal-density-field
pipeline) live in the experimental ``gravoturb_fdf`` package (follow-up paper),
not in released progenax. The old subsystem modules (``fdf``, ``fdf_density``,
``fdf_tail``, ``gravoturbulent``, ``fdf_config``, ``fdf_calibration``,
``fdf_hyperparams``, and the ``gravoturb`` package) were removed in the 2026-06
clean-room rewrite; their validated successors are in ``gravoturb_fdf``.

Main Entry Point:
    generate_cluster_ic: Generate complete cluster IC from parameters

Data Classes:
    ClusterState: Immutable container for masses, positions, velocities
    SpatialStructureParams: Combined profile + structure parameters
    MassSegregationLayer: Baumgardt+2008 mass segregation parameters

References:
    Baumgardt, De Marchi & Kroupa (2008), ApJ 685, 247
    Küpper et al. (2011), MNRAS 417, 2300 - McLuster
"""

from progenax.cluster.core import (
    ClusterState,
    MassSegregationLayer,
    SpatialStructureParams,
    generate_cluster_ic,
    sample_velocities_for_profile,
)

from progenax.cluster.mass_segregation import energy_sorted_segregation

# Physical constants
from progenax.cluster.constants import (
    G_KMS,
    C_S_DEFAULT,
    B_DEFAULT,
    BETA_KOLMOGOROV,
    BETA_BURGERS,
    SIGMA_V0_DEFAULT,
    ALPHA_LARSON,
)

# Turbulence physics (stays in core; used by EnvironmentIMF)
from progenax.cluster.turbulence import (
    sigma_ln_rho_from_mach,
    spectral_slope_from_mach,
    cloud_radius_from_density,
    larson_sigma_v,
    turbulent_mach_from_cloud,
    b_from_environment,
)

__all__ = [
    # Core API
    "ClusterState",
    "MassSegregationLayer",
    "SpatialStructureParams",
    "generate_cluster_ic",
    "sample_velocities_for_profile",
    # Mass segregation
    "energy_sorted_segregation",
    # Physical constants
    "G_KMS",
    "C_S_DEFAULT",
    "B_DEFAULT",
    "BETA_KOLMOGOROV",
    "BETA_BURGERS",
    "SIGMA_V0_DEFAULT",
    "ALPHA_LARSON",
    # Turbulence physics
    "sigma_ln_rho_from_mach",
    "spectral_slope_from_mach",
    "cloud_radius_from_density",
    "larson_sigma_v",
    "turbulent_mach_from_cloud",
    "b_from_environment",
]
