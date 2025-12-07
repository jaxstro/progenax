"""Fractal substructure generation for star cluster ICs.

Implements fractal distribution functions for generating clumpy,
hierarchical initial conditions as observed in young star clusters.

References:
    Goodwin & Whitworth (2004) A&A 413, 929 - Fractal star cluster structure
    Kupper et al. (2011) MNRAS 417, 2300 - McLuster implementation
"""

from progenax.substructure.fractal import (
    generate_fractal_positions,
    apply_fractal_overlay_radial,
    apply_fractal_overlay_blend,
)

__all__ = [
    "generate_fractal_positions",
    "apply_fractal_overlay_radial",
    "apply_fractal_overlay_blend",
]
