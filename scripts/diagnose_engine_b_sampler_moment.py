"""Speed-table moment weighted by the retained sampler's radial CDF.

Local inverse-speed-CDF integration is exact. Radial integration uses the
existing grid and trapezoid rule and is not claimed to be converged.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_engine_b_population_moments import model_at
from jaxstro.numerics.integration import cumulative_trapz

from progenax.cluster._radial_cdf import inverse_cdf_second_moments

OUT = Path("docs/reviews/plummer-om-artifacts/engine-b-sampler-radial-moment.json")


def moment(theta, subdivisions=1, speed_nodes=256):
    model = model_at(theta)
    state = model.engine_b
    fraction = jnp.arange(subdivisions) / subdivisions
    psi = jnp.concatenate(
        (
            (
                state.Psi_grid[:-1, None] + jnp.diff(state.Psi_grid)[:, None] * fraction
            ).reshape(-1),
            state.Psi_grid[-1:],
        )
    )
    component_moments = []
    for component in range(model.m_j.size):

        def local(potential):
            safe = jnp.maximum(potential, 1e-12 * state.E_grid[-1])
            w = jnp.linspace(0.0, jnp.sqrt(2 * safe), speed_nodes)
            f = jnp.interp(
                potential - w**2 / 2, state.E_grid, state.f_j_grid[component]
            )
            weight = jnp.maximum(w**2 * f, 0.0)
            cdf = cumulative_trapz(weight, dx=w[1] - w[0])
            cdf = cdf / (cdf[-1] + 1e-30)
            value = inverse_cdf_second_moments(w, cdf[None, :])[0]
            return jnp.where(potential > 1e-6 * state.E_grid[-1], value, 0.0)

        v2 = jax.lax.map(local, psi, batch_size=32)
        original = model._cdf_j[component]
        cdf = jnp.concatenate(
            (
                (original[:-1, None] + jnp.diff(original)[:, None] * fraction).reshape(
                    -1
                ),
                original[-1:],
            )
        )
        component_moments.append(
            jnp.sum(jnp.diff(cdf) * (v2[:-1] + v2[1:]) / 2)
            + cdf[0] * v2[0]
            + (1 - cdf[-1]) * v2[-1]
        )
    pi = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    mean_mass = pi @ model.m_j
    return jnp.sum(
        pi * jnp.stack(component_moments) * (model.m_j + 99999 * mean_mass)
    ) / (4 * jnp.pi * state.mu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdivisions", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument(
        "--speed-nodes", type=int, choices=[256, 512, 1024], default=256
    )
    args = parser.parse_args()
    global OUT
    if args.subdivisions != 1:
        OUT = OUT.with_name(
            f"engine-b-sampler-radial-moment-sub{args.subdivisions}.json"
        )
    if args.speed_nodes != 256:
        OUT = OUT.with_name(
            f"engine-b-sampler-moment-sub{args.subdivisions}-speed{args.speed_nodes}.json"
        )
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    value, gradient = jax.value_and_grad(
        lambda t: moment(t, args.subdivisions, args.speed_nodes)
    )(jnp.array([0.6, 2.0, 0.8]))
    jax.block_until_ready(gradient)
    record = dict(
        subdivisions=args.subdivisions,
        speed_nodes=args.speed_nodes,
        value=float(value),
        gradient=np.asarray(gradient).tolist(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="retained isotropic model; exact local inverse-speed-CDF moment; resolution recorded separately; trapezoid radial integration on unchanged 1000-node sampler grid, not convergence-qualified",
    )
    OUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
