"""Three-dimensional candidate geometry regression; one compiled configuration."""

import jax
import jax.numpy as jnp

from progenax.diagnostics.q_approx import q_approx_fast

_q_3d = jax.jit(
    lambda pos: q_approx_fast(
        pos, project_to_2d=False, nbins_per_dim=2, calibration=1.0
    )
)


def test_fast_q_preserves_three_dimensional_candidate_coordinates():
    """Catch a two-column sentinel in the Nx3 path against analytic tetrahedral distances."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    # All four nearest-neighbor distances are 1. The six pair distances are
    # three 1s and three sqrt(2)s; cluster radius cancels in this Q convention.
    # Include the documented arithmetic regularizers: squared-distance 1e-12
    # and the 1e-10 denominator offset. The bounding radius is sqrt(11)/4.
    nearest = jnp.sqrt(1.0 + 1e-12)
    mean_distance = (nearest + jnp.sqrt(2.0 + 1e-12)) / 2
    radius = jnp.sqrt(11.0) / 4
    expected = 3 * nearest / (2 * jnp.sqrt(jnp.pi) * (mean_distance + 1e-10 * radius))
    assert jnp.allclose(_q_3d(positions), expected, rtol=2e-14, atol=0.0)
