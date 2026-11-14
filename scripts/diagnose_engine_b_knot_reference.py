"""Knot-aligned integration of the retained piecewise-linear DF reference."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_engine_b_population_moments import model_at


def reference(radius, nr=12000, theta=None):
    if theta is None:
        theta = jnp.array([0.6, 2.0, 0.8])
    model = model_at(theta, n_r=nr, r_a_j=jnp.array([radius, jnp.inf]))
    state = model.engine_b
    nodes, weights = (jnp.asarray(x) for x in np.polynomial.legendre.leggauss(4))
    if bool(jnp.any(state.f_j_grid < 0)):
        raise ValueError("Negative DF table requires additional zero-crossing knots")
    r = state.r_poisson
    parts = []
    for j in range(model.m_j.size):

        def local(psi):
            # Clamped interpolation adds constant-DF end segments.
            energies = jnp.concatenate(
                (
                    jnp.array([psi]),
                    jnp.clip(state.E_grid[::-1], 0, psi),
                    jnp.array([0.0]),
                )
            )
            edges = jnp.sqrt(jnp.maximum(2 * (psi - energies), 0))
            middle = (edges[1:] + edges[:-1]) / 2
            half = (edges[1:] - edges[:-1]) / 2
            w = middle[:, None] + half[:, None] * nodes
            f = jnp.interp(psi - w * w / 2, state.E_grid, state.f_j_grid[j])
            # Nonnegative table checked above: no interior zero crossing.
            weight = half[:, None] * weights * f
            return jnp.sum(weight * w**4) / jnp.sum(weight * w**2)

        v2 = jax.lax.map(local, jnp.maximum(state.Psi_poisson, 1e-12), batch_size=32)
        factor = 1 / 3 + 2 / 3 / (1 + (r / state.r_a_j[j]) ** 2)
        density = state.rho_j_poisson[j] * r**2
        parts.append(
            jnp.trapezoid(density * v2 * factor, r) / jnp.trapezoid(density, r)
        )
    pi = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    return jnp.sum(pi * jnp.stack(parts) * (model.m_j + 99999 * (pi @ model.m_j))) / (
        4 * jnp.pi * state.mu
    )


if __name__ == "__main__":
    assert jax.config.jax_enable_x64
    out = Path("docs/reviews/plummer-om-artifacts/engine-b-ra-knot-reference.json")
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for nr in (6000, 12000):
        start = time.perf_counter()
        h = 3e-4
        values = [float(reference(r, nr)) for r in (3 - h, 3.0, 3 + h)]
        rows.append(
            dict(
                n_r=nr,
                value=values[1],
                fd=(values[2] - values[0]) / (2 * h),
                seconds=time.perf_counter() - start,
            )
        )
        print(rows[-1], flush=True)
    out.write_text(
        json.dumps(
            dict(
                cases=rows,
                scope="Knot-aligned DF quadrature; central finite difference h=3e-4",
            ),
            indent=2,
        )
        + "\n"
    )
