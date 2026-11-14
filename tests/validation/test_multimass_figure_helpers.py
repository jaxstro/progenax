"""Figure diagnostics must integrate the nonuniform production radial grid."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest


@pytest.mark.parametrize("radii", [[0.0, 1.0, 2.0, 4.0], [0.0, 1.0, 1.0, 2.0, 4.0]])
def test_half_mass_radius_uses_each_interval_width(radii):
    helper = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "scripts/validate_multimass_equilibrium.py"
        )
    )["_half_mass_r"]
    model = SimpleNamespace(
        _r_grid=jnp.array(radii),
        r_c=1.0,
        xi_grid=jnp.array([0.0, 4.0]),
        psi_grid=jnp.ones(2),
        W0=1.0,
        rescale_j=jnp.ones(1),
        g=1.0,
        r_t=4.0,
    )
    # Constant density: trapezoidal cumulative masses at 0,1,2,4 are
    # 0, 0.5, 3, 23. Linear inversion of half the total gives 2.85.
    # A repeated node adds exactly zero mass.
    assert helper(model, 0) == pytest.approx(2.85)
