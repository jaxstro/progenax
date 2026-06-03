"""DensityField3D absolute normalization (audit minor: dV off-by-one).

init_turbulent_density_field and init_bm19_density_field computed the cell volume
as ``dV = (2*L_box/N)**3`` while the grid is ``linspace(-L, L, N)`` whose true spacing
is ``2*L/(N-1)``. Because the field is normalized by ``sum(rho)*dV``, integrating the
stored field with the *true* grid spacing returned ``(N/(N-1))**3`` (= 1.0484 at N=64)
instead of 1. The integral is a pure normalization identity (the turbulent realization
cancels), so the expected value is exact, not statistical.

The fix decouples the two uses of the spacing: ``dx`` (for the periodic FFT k-grid via
fftfreq) is left as ``2*L_box/N``; the integration volume ``dV`` uses the actual grid
spacing. This changes only the absolute normalization, not the spectrum or sampling.
"""

import jax
import jax.numpy as jnp

import progenax  # noqa: F401  (enables float64)
from progenax.cluster.fdf_density import (
    FractalDensityLayer,
    init_bm19_density_field,
    init_turbulent_density_field,
)


def _integral_with_true_spacing(field):
    dx = field.x_grid[1] - field.x_grid[0]   # actual grid spacing = 2L/(N-1)
    return float(jnp.sum(field.rho_grid) * dx**3)


def test_turbulent_density_field_integrates_to_one():
    key = jax.random.PRNGKey(0)
    layer = FractalDensityLayer(grid_size=64)
    field = init_turbulent_density_field(key, R_half=1.0, layer=layer)
    integral = _integral_with_true_spacing(field)
    assert abs(integral - 1.0) < 1e-6, (
        f"∫ρ dV = {integral:.5f}; ~1.048 indicates the dV off-by-one"
    )


def test_bm19_density_field_integrates_to_one():
    key = jax.random.PRNGKey(0)
    sigma_s_sq = 1.0
    alpha = 2.0
    s_t = (alpha - 0.5) * sigma_s_sq
    field = init_bm19_density_field(
        key, sigma_s_sq, s_t, alpha, grid_size=64, box_half_size=1.0
    )
    integral = _integral_with_true_spacing(field)
    assert abs(integral - 1.0) < 1e-6, (
        f"∫ρ dV = {integral:.5f}; ~1.048 indicates the dV off-by-one"
    )
