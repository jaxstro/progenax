"""Deterministic speed-CDF moment diagnostic for the retained ra derivative."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from diagnose_engine_b_population_moments import df_speed_moment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speed-nodes", type=int, choices=[1024, 2048, 4096, 8192, 16384], default=1024
    )
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    root = Path("docs/reviews/plummer-om-artifacts")
    output = root / "engine-b-ra-speed-cdf-moment.json"
    if args.speed_nodes != 1024:
        output = output.with_name(
            f"engine-b-ra-speed-cdf-moment-{args.speed_nodes}.json"
        )
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    value, derivative = jax.block_until_ready(
        jax.value_and_grad(
            lambda radius: df_speed_moment(
                jnp.array([0.6, 2.0, 0.8]),
                n_w=args.speed_nodes,
                n_r=12000,
                sampler_table=True,
                r_a_j=jnp.array([radius, jnp.inf]),
            )
        )(jnp.array(3.0))
    )
    ref = json.loads((root / "engine-b-ra-reference-6400.json").read_text())["gradient"]
    result = dict(
        speed_nodes=args.speed_nodes,
        value=float(value),
        derivative=float(derivative),
        reference=ref,
        relative_discrepancy=abs(float(derivative) - ref) / abs(ref),
        seconds=time.perf_counter() - start,
        scope="Exact local piecewise-linear inverse speed CDF moments; prescribed-density radial quadrature, not exact sampler radial-CDF expectation",
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)
