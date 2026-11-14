"""Approved exact local speed/direction averaging; radial draws unchanged."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxstro.numerics.integration import cumulative_trapz

from progenax.cluster._radial_cdf import (
    component_linear_quantile,
    inverse_cdf_second_moments,
)
from progenax.kinematics.eddington import _interp_uniform


def local_speed_squared(psi, energy, df):
    w = jnp.linspace(0.0, jnp.sqrt(2 * jnp.maximum(psi, 1e-12 * energy[-1])), 4096)
    density = jnp.maximum(w**2 * _interp_uniform(psi - w**2 / 2, energy, df), 0.0)
    cdf = cumulative_trapz(density, dx=w[1] - w[0])
    cdf = cdf / (cdf[-1] + 1e-30)
    second = inverse_cdf_second_moments(w, cdf[None, :])[0]
    return jnp.where(psi > 1e-6 * energy[-1], second, 0.0)


def knot_speed_squared(psi, energy, df):
    """Exact Gauss integration for the retained nonnegative linear DF table."""
    nodes, weights = (jnp.asarray(x) for x in np.polynomial.legendre.leggauss(4))
    e = jnp.concatenate(
        (jnp.array([psi]), jnp.clip(energy[::-1], 0, psi), jnp.array([0.0]))
    )
    delta = 2 * (psi - e)
    edges = jnp.where(delta > 0, jnp.sqrt(jnp.where(delta > 0, delta, 1.0)), 0.0)
    half = (edges[1:] - edges[:-1]) / 2
    w = (edges[1:] + edges[:-1])[:, None] / 2 + half[:, None] * nodes
    f = _interp_uniform(psi - w * w / 2, energy, df)
    weight = half[:, None] * weights * f
    return jnp.sum(weight * w**4) / jnp.sum(weight * w**2)


def _engine_b_speed_moment(
    model, key, labels, weights, G, *, stratified=False, knot_average=False
):
    """Conditional speed moment with bounded reverse storage."""
    state = model.engine_b
    count = labels.size
    _, kr, _, _, _ = jax.random.split(key, 5)
    u = jax.random.uniform(kr, (count,))
    if stratified:
        per_component = count // model.m_j.size
        u = (jnp.tile(jnp.arange(per_component), model.m_j.size) + u) / per_component
    radii = component_linear_quantile(u, labels, model._r_grid, model._cdf_j)
    psi = jnp.interp(radii, model._r_grid, state.Psi_grid)
    finite = jnp.isfinite(state.r_a_j[labels])
    ra = jnp.where(finite, state.r_a_j[labels], 1.0)
    factor = 1 / 3 + 2 / 3 * jnp.where(finite, 1 / (1 + (radii / ra) ** 2), 1.0)
    batch = 64
    pad = (-count) % batch
    arrays = jax.tree.map(
        lambda x: jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge"),
        (psi, labels, factor, weights),
    )
    arrays = (*arrays, jnp.arange(count + pad) < count)

    @jax.checkpoint
    def chunk(index):
        p, c, d, w, valid = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, index * batch, batch), arrays
        )
        second = jax.vmap(
            lambda pp, cc: (
                knot_speed_squared if knot_average else local_speed_squared
            )(pp, state.E_grid, state.f_j_grid[cc])
        )(p, c)
        return jnp.sum(jnp.where(valid, second * d * w, 0.0))

    return (
        G
        * jnp.sum(jax.lax.map(chunk, jnp.arange((count + pad) // batch)))
        / (4 * jnp.pi * state.mu)
    )
