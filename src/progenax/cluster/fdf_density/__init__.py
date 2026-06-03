"""FDF density-field cluster initial conditions.

Split out of the former monolithic ``fdf_density.py`` to satisfy the 500-LOC file
limit. The public API is unchanged — every symbol below remains importable from
``progenax.cluster.fdf_density`` and ``progenax.cluster``.

Layering (acyclic): ``density_field`` (data structures + FFT/grid helpers) ←
``field_init`` / ``sampling`` ← ``pipeline``.
"""

from .density_field import (
    FractalDensityLayer,
    TailSubstructureLayer,
    DensityField3D,
)
from .field_init import (
    init_turbulent_density_field,
    init_bm19_density_field,
)
from .sampling import (
    sample_positions_from_density,
    sample_positions_tail,
)
from .pipeline import (
    generate_fractal_ic_density,
    density_layer_from_D,
)

__all__ = [
    # Data structures
    "FractalDensityLayer",
    "TailSubstructureLayer",
    "DensityField3D",
    # Field operations
    "init_turbulent_density_field",
    "init_bm19_density_field",
    "sample_positions_from_density",
    "sample_positions_tail",
    # IC Generator
    "generate_fractal_ic_density",
    # Calibration (DEPRECATED - use env_to_fdf_layer() instead)
    "density_layer_from_D",
]
