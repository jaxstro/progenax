"""Unit tests for the spherical cluster-shape envelope (FDF spherical-IC pivot, build 1).

Separable log-space model: s_total(x) = s_turb(x) + ln rho_env(r), with rho_env from any progenax
SpatialProfile (.density). The envelope adds the centrally-concentrated SHAPE; s_turb carries the
SUBSTRUCTURE.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental

SHAPE = (32, 32, 32)
BOX = 1.0


def test_radius_grid_min_at_center_max_at_corner():
    """radius_grid is ~0 near the box center and ~sqrt(3)/2*box at the corners."""
    from gravoturb.realization.envelope import radius_grid

    r = np.asarray(radius_grid(SHAPE, box_size=BOX))
    assert r.shape == SHAPE
    assert r.min() < 0.05 * BOX  # a cell near the center
    np.testing.assert_allclose(r[0, 0, 0], np.sqrt(3) * BOX / 2, rtol=0.05)  # corner


def test_envelope_reproduces_plummer_profile_when_no_turbulence():
    """With s_turb=0, e^{s_total} azimuthally averages to the Plummer density profile."""
    from gravoturb.realization.envelope import apply_spherical_envelope, radius_grid

    from progenax import PlummerProfile

    prof = PlummerProfile(r_h=0.15)
    s_total = apply_spherical_envelope(jnp.zeros(SHAPE), prof, box_size=BOX)
    rho = np.asarray(jnp.exp(s_total))
    r = np.asarray(radius_grid(SHAPE, box_size=BOX))
    # with no turbulence, e^{s_total} == the analytic Plummer density cell-by-cell (exact)
    np.testing.assert_allclose(
        rho, np.asarray(prof.density(jnp.asarray(r))), rtol=1e-10
    )
    # and it is centrally concentrated
    assert rho[r < 0.1].mean() > 5 * rho[(r > 0.3) & (r < 0.4)].mean()


def test_envelope_is_separable():
    """s_total - ln rho_env(r) recovers s_turb exactly (envelope only adds a radial mean)."""
    from gravoturb.realization.envelope import apply_spherical_envelope, radius_grid

    from progenax import PlummerProfile

    prof = PlummerProfile(r_h=0.2)
    key = jax.random.PRNGKey(0)
    s_turb = 0.5 * jax.random.normal(key, SHAPE)
    s_total = apply_spherical_envelope(s_turb, prof, box_size=BOX)
    s_env = jnp.log(prof.density(radius_grid(SHAPE, box_size=BOX)))
    np.testing.assert_allclose(
        np.asarray(s_total - s_env), np.asarray(s_turb), atol=1e-8
    )


def test_envelope_differentiable_in_rh():
    """jax.grad of total enveloped mass-ish wrt the profile r_h is finite (differentiable shape)."""
    from gravoturb.realization.envelope import apply_spherical_envelope

    from progenax import PlummerProfile

    s_turb = jnp.zeros(SHAPE)

    def total(r_h):
        s_total = apply_spherical_envelope(
            s_turb, PlummerProfile(r_h=r_h), box_size=BOX
        )
        return jnp.sum(jnp.exp(s_total))

    g = float(jax.grad(total)(0.2))
    assert np.isfinite(g) and g != 0.0
