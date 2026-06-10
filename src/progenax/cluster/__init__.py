# progenax/src/progenax/cluster/__init__.py
"""
Star cluster initial conditions: the unified multi-component equilibrium model
plus the primordial (non-equilibrium) mass-segregation generator.

Main Entry Points:
    MultiComponentCluster: N populations in ONE self-consistent shared
        potential, via two engines. Engine A (DF-defined,
        lowered-isothermal/LIMEPY family) — mass segregation
        (.from_mass_segregation), GC 1G/2G & halo+core (.from_components),
        IMF-driven (.from_imf). Engine B (density-defined, Eddington
        inversion) — Plummer/EFF/King density components in one shared
        potential (.from_density_profiles), optional per-component
        Osipkov-Merritt anisotropy. Differentiable; samples to ICResult.
    energy_sorted_segregation: Baumgardt+2008-style PRIMORDIAL segregation —
        energy-ordered orbit assignment (not an equilibrium construction; the
        fully-ordered state is a clean per-group equilibrium).

Smooth single-population ICs are built with ``progenax.build_spatial_ic``
(any SpatialProfile × VelocityDF). The legacy string-dispatch generator
(``generate_cluster_ic``/``ClusterState``) and the ``lambda_seg`` blend were
retired in the 2026-06 unified redesign (pre-launch, no backwards compat).

Turbulent/fractal substructure ICs live in the experimental ``gravoturb_fdf``
package (follow-up paper), not in released progenax.

References:
    Gieles & Zocchi (2015), MNRAS 454, 576 - LIMEPY family
    Baumgardt, De Marchi & Kroupa (2008), ApJ 685, 247
"""

from progenax.cluster.mass_segregation import energy_sorted_segregation
from progenax.cluster.multicomponent import MultiComponentCluster

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
    # Unified multi-component equilibrium model (Engine A: DF-defined
    # LIMEPY family; Engine B: density-defined Eddington inversion)
    "MultiComponentCluster",
    # Primordial (non-equilibrium) mass segregation
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
