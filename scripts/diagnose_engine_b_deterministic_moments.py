"""Radial-CDF interval quadrature with knot-aligned conditional DF moments."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_engine_b_population_moments import model_at
from prototype_engine_b_speed_average import knot_speed_squared


def moment(radius, order, n_grid=1000, n_r=6000, theta=None):
    if theta is None:
        theta = jnp.array([0.6, 2.0, 0.8])
    model = model_at(theta, r_a_j=jnp.array([radius, jnp.inf]), n_grid=n_grid, n_r=n_r)
    state = model.engine_b
    nodes, weights = (jnp.asarray(x) for x in np.polynomial.legendre.leggauss(order))
    r = model._r_grid
    positions = (
        (r[1:] + r[:-1])[:, None] / 2 + (r[1:] - r[:-1])[:, None] / 2 * nodes
    ).ravel()
    psi = jnp.interp(positions, r, state.Psi_grid)
    parts = []
    for j in range(model.m_j.size):

        @jax.checkpoint
        def local(p):
            return knot_speed_squared(p, state.E_grid, state.f_j_grid[j])

        speed = jax.lax.map(local, psi, batch_size=32)
        factor = 1 / 3 + 2 / 3 / (1 + (positions / state.r_a_j[j]) ** 2)
        avg = ((speed * factor).reshape((-1, order)) @ weights) / 2
        parts.append(jnp.sum(jnp.diff(model._cdf_j[j]) * avg))
    pi = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    return jnp.sum(pi * jnp.stack(parts) * (model.m_j + 99999 * (pi @ model.m_j))) / (
        4 * jnp.pi * state.mu
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--radial-nodes", type=int, choices=[1000, 2000, 4000], default=1000
    )
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    out = Path("docs/reviews/plummer-om-artifacts/engine-b-ra-deterministic-pilot.json")
    if args.radial_nodes != 1000:
        out = out.with_name(f"engine-b-ra-deterministic-pilot-{args.radial_nodes}.json")
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for order in (4, 8, 16):
        start = time.perf_counter()
        v, g = jax.block_until_ready(
            jax.jvp(
                lambda r: moment(r, order, args.radial_nodes),
                (jnp.array(3.0),),
                (jnp.array(1.0),),
            )
        )
        rows.append(
            dict(
                order=order,
                value=float(v),
                gradient=float(g),
                seconds=time.perf_counter() - start,
            )
        )
        out.write_text(
            json.dumps(
                dict(
                    cases=rows,
                    status="running",
                    scope="Radial CDF quadrature prototype",
                ),
                indent=2,
            )
            + "\n"
        )
        print(rows[-1], flush=True)
    out.write_text(
        json.dumps(
            dict(
                cases=rows,
                status="completed",
                scope="Radial CDF quadrature prototype; not production qualification",
            ),
            indent=2,
        )
        + "\n"
    )
