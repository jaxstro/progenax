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
    FractalLayer: User-facing fractal params (D-based, uses FDF internally)

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
    FractalLayer,
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
    DensityField3D,
    generate_fractal_ic_density,
    init_turbulent_density_field,
    sample_positions_from_density,
    density_layer_from_D,
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
    "DensityField3D",
    "generate_fractal_ic_density",
    "init_turbulent_density_field",
    "sample_positions_from_density",
    "density_layer_from_D",
]
