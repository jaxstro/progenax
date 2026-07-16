"""Unit tests for turbulent coherent velocities (FDF spherical-IC pivot, build 2).

A young cluster's stars inherit the local coherent velocity of the natal turbulent gas (Goodwin &
Whitworth 2004 — coherence, realized here as a turbulent velocity GRF rather than their fractal tree).
We build a 3-component velocity GRF with spectrum P_v(k) ∝ k^{-beta_v}, interpolate it to star
positions (coherent: nearby stars move together), then scale to a chosen virial ratio Q via the core
``virial_scale`` (Q ≡ T/|V|; 0.5 virial, <0.5 collapsing, 0.75 super-virial — G&W 2004).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental

SHAPE = (24, 24, 24)


def _neighbor_corr(v_field):
    """Pearson correlation between a velocity component and its one-cell shift along x (coherence)."""
    a = np.asarray(v_field)[..., 0]
    x0, x1 = a[:-1].ravel(), a[1:].ravel()
    return float(np.corrcoef(x0, x1)[0, 1])


def test_velocity_field_shape_and_spatial_coherence():
    """turbulent_velocity_field is (nx,ny,nz,3) and spatially COHERENT (neighbours correlated)."""
    from gravoturb.realization.turbulent_velocity import turbulent_velocity_field

    v = turbulent_velocity_field(SHAPE, beta_v=4.0, key=jax.random.PRNGKey(0))
    assert v.shape == SHAPE + (3,)
    assert _neighbor_corr(v) > 0.5  # coherent, not white noise


def test_steeper_spectrum_more_coherent():
    """Larger beta_v -> smoother (more large-scale-dominated) field -> higher neighbour correlation."""
    from gravoturb.realization.turbulent_velocity import turbulent_velocity_field

    shallow = turbulent_velocity_field(SHAPE, beta_v=2.0, key=jax.random.PRNGKey(1))
    steep = turbulent_velocity_field(SHAPE, beta_v=5.0, key=jax.random.PRNGKey(1))
    assert _neighbor_corr(steep) > _neighbor_corr(shallow)


def test_sample_velocities_interpolates_and_is_differentiable():
    """Stars sample the local field velocity (trilinear); differentiable in positions."""
    from gravoturb.realization.turbulent_velocity import (
        sample_turbulent_velocities,
        turbulent_velocity_field,
    )

    v = turbulent_velocity_field(SHAPE, beta_v=4.0, key=jax.random.PRNGKey(2))
    pos = jnp.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.3]])  # in [0, box)^3
    vs = sample_turbulent_velocities(pos, v, box_size=1.0)
    assert vs.shape == (2, 3)
    assert np.all(np.isfinite(np.asarray(vs)))
    g = jax.grad(lambda p: jnp.sum(sample_turbulent_velocities(p, v, 1.0)))(pos)
    assert np.all(np.isfinite(np.asarray(g)))


def test_virial_scale_achieves_target_Q():
    """After core virial_scale, the measured Q = T/|V| matches the target (envelope auto-accounted)."""
    from gravoturb.realization.turbulent_velocity import (
        sample_turbulent_velocities,
        turbulent_velocity_field,
    )
    from jaxstro.units import STELLAR

    from progenax import compute_kinetic_energy, compute_potential_energy, virial_scale

    key = jax.random.PRNGKey(3)
    pos = jax.random.uniform(key, (300, 3))
    m = jnp.ones(300)
    v = sample_turbulent_velocities(
        pos, turbulent_velocity_field(SHAPE, 4.0, jax.random.PRNGKey(4)), box_size=1.0
    )
    v_scaled = virial_scale(pos, v, m, Q_target=0.5, G=STELLAR.G)
    T = compute_kinetic_energy(v_scaled, m)
    V = compute_potential_energy(pos, m, G=STELLAR.G)
    assert abs(float(T / jnp.abs(V)) - 0.5) < 1e-3
