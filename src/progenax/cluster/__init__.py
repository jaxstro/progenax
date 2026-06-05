# progenax/src/progenax/cluster/__init__.py
"""
Star cluster initial condition generator (v2 with FDF).

This module provides tools for generating realistic initial conditions for
star cluster simulations with optional mass segregation and fractal substructure.

v2 Update: Fractal substructure now uses the differentiable Fractal Displacement
Field (FDF) method instead of the non-differentiable GW2004 tree algorithm.

Main Entry Point:
    generate_cluster_ic: Generate complete cluster IC from parameters

Data Classes:
    ClusterState: Immutable container for masses, positions, velocities
    SpatialStructureParams: Combined profile + structure parameters
    MassSegregationLayer: Baumgardt+2008 mass segregation parameters

FDF API (Fractal Displacement Field):
    FractalField: Frozen stochastic structure
    FractalDisplacementLayer: FDF parameter bundle
    generate_fractal_ic: Generate IC with FDF
    init_fractal_field: Initialize displacement field
    fractal_layer_from_D: Create FDF params from GW-style D

References:
    Baumgardt, De Marchi & Kroupa (2008), ApJ 685, 247
    Goodwin & Whitworth (2004), A&A 413, 929
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

# FDF API (v2 fractal substructure)
from progenax.cluster.fdf import (
    FractalField,
    FractalDisplacementLayer,
    generate_fractal_ic,
    init_fractal_field,
    compute_amplitudes,
    evaluate_displacement,
    apply_displacement,
    assign_fractal_velocities,
)

from progenax.cluster.fdf_calibration import (
    FDFCalibration,
    load_fdf_calibration,
    fractal_layer_from_D,
)

# FDF Density API (v3 fractal substructure via density field sampling)
from progenax.cluster.fdf_density import (
    FractalDensityLayer,
    TailSubstructureLayer,
    DensityField3D,
    generate_fractal_ic_density,
    init_turbulent_density_field,
    sample_positions_from_density,
    sample_positions_tail,
    density_layer_from_D,
)

# FDF Tail Selection (BM19-consistent)
from progenax.cluster.fdf_tail import (
    TailPMFResult,
    compute_tail_pmfs_bm19,
    sample_positions_tail_bm19,
)

# Physical constants
from progenax.cluster.constants import (
    G_KMS,
    C_S_DEFAULT,
    B_DEFAULT,
    BETA_KOLMOGOROV,
    BETA_BURGERS,
    SIGMA_V0_DEFAULT,
    ALPHA_LARSON,
    CHI_MIN,
    CHI_MAX,
)

# Turbulence physics
from progenax.cluster.turbulence import (
    sigma_ln_rho_from_mach,
    spectral_slope_from_mach,
    cloud_radius_from_density,
    larson_sigma_v,
    turbulent_mach_from_cloud,
    b_from_environment,
)

# FDF Hyperparameters
from progenax.cluster.fdf_hyperparams import (
    FDFDensityHyperparams,
    FDFDisplacementHyperparams,
    FDFUncalibratedHeuristics,
    FDF_DENSITY_DEFAULTS,
    FDF_DISPLACEMENT_DEFAULTS,
    FDF_HEURISTICS,
)

# Gravoturbulent Environment
from progenax.cluster.gravoturbulent import (
    GravoturbulentEnv,
    TailSelectionConfig,
    GRAVOTURBULENT_PRESETS,
    env_from_preset,
    tail_layer_from_env,
)

# Environment mapping and phenomenological helpers
from progenax.cluster.fdf_config import (
    env_to_fdf_layer,
    default_f_sub_for_cluster_type,
    tail_layer_from_cluster_type,
    f_sub_from_D,
    tail_layer_from_D,
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
    # FDF API
    "FractalField",
    "FractalDisplacementLayer",
    "generate_fractal_ic",
    "init_fractal_field",
    "compute_amplitudes",
    "evaluate_displacement",
    "apply_displacement",
    "assign_fractal_velocities",
    "FDFCalibration",
    "load_fdf_calibration",
    "fractal_layer_from_D",
    # FDF Density API (density field sampling)
    "FractalDensityLayer",
    "TailSubstructureLayer",
    "DensityField3D",
    "generate_fractal_ic_density",
    "init_turbulent_density_field",
    "sample_positions_from_density",
    "sample_positions_tail",
    "density_layer_from_D",
    # FDF Tail Selection (BM19-consistent)
    "TailPMFResult",
    "compute_tail_pmfs_bm19",
    "sample_positions_tail_bm19",
    # Physical constants
    "G_KMS",
    "C_S_DEFAULT",
    "B_DEFAULT",
    "BETA_KOLMOGOROV",
    "BETA_BURGERS",
    "SIGMA_V0_DEFAULT",
    "ALPHA_LARSON",
    "CHI_MIN",
    "CHI_MAX",
    # Turbulence physics
    "sigma_ln_rho_from_mach",
    "spectral_slope_from_mach",
    "cloud_radius_from_density",
    "larson_sigma_v",
    "turbulent_mach_from_cloud",
    "b_from_environment",
    # FDF Hyperparameters
    "FDFDensityHyperparams",
    "FDFDisplacementHyperparams",
    "FDFUncalibratedHeuristics",
    "FDF_DENSITY_DEFAULTS",
    "FDF_DISPLACEMENT_DEFAULTS",
    "FDF_HEURISTICS",
    # Gravoturbulent Environment
    "GravoturbulentEnv",
    "TailSelectionConfig",
    "GRAVOTURBULENT_PRESETS",
    "env_from_preset",
    "tail_layer_from_env",
    # Environment mapping
    "env_to_fdf_layer",
    # Phenomenological helpers
    "default_f_sub_for_cluster_type",
    "tail_layer_from_cluster_type",
    "f_sub_from_D",
    "tail_layer_from_D",
]
