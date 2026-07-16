"""gravoturb — gravoturbulent-1D + fractal-density-field (FDF) initial conditions.

Clean-room rewrite (2026-06) of the gravoturbulent + FDF IC subsystem, authored from
PDF-grounded theory (Burkhart & Mocz 2019; Parmentier & Pasquali 2020; Padoan &
Nordlund 2011; Federrath+2010; Heyer+2009; Kim & Ryu 2005).

**Status: EXPERIMENTAL — follow-up paper. NOT part of the initial progenax/jaxstro
release and NOT shipped in the progenax wheel.** This standalone package depends one-way
on ``progenax.cluster.turbulence``; nothing in released progenax imports it.

Layers
------
- ``theory``      : BM19 / PP20 / PN11 1D density-PDF theory (JAX-native, differentiable).
- ``realization`` : 3D realization — GRF + rank copula, dense-tail mask, star placement.
- ``diagnostics`` : CW04 Q substructure metric (numpy/scipy, non-differentiable).
- ``inference``   : differentiable predicted-statistics inference (blackjax NUTS).
- ``cluster``     : the end-to-end natal-parameters → N-body IC builder + typed specs.
- ``validation``  : AC1–AC17 + AC-IC acceptance scripts that print real numbers.

Every formula is re-derived from the held PDFs and re-validated against committed,
printing acceptance scripts before it is believed. See
an internal clean-room spec (§8 authoritative).
"""

# Configure JAX for 64-bit precision BEFORE any JAX arrays are created (ecosystem
# convention; float64 is mandatory for scientific precision).
from jaxstro.jaxconfig import enable_high_precision as _enable_jax_hp

_enable_jax_hp()
del _enable_jax_hp

__version__ = "0.0.0.dev0"

from gravoturb.cluster import ClusterIC, build_cluster_ic
from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec

__all__ = [
    "ClusterIC",
    "CloudSpec",
    "CompositionSpec",
    "GeometrySpec",
    "VelocitySpec",
    "build_cluster_ic",
]
