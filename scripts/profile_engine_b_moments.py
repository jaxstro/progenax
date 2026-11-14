"""Same-workload Engine B JVP compilation-boundary profile at 1024 speeds."""

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from qualify_engine_b_population_moments import moment

from progenax.cluster import eddington_engine
from progenax.kinematics.eddington import sample_speed_from_f_table

OUT = Path("docs/reviews/plummer-om-artifacts/engine-b-moment-performance-batched.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, choices=[0, 32, 64, 128, 512], default=0
    )
    parser.add_argument("--scan", action="store_true")
    args = parser.parse_args()
    if args.scan and not args.batch_size:
        parser.error("--scan requires a nonzero --batch-size")
    global OUT
    OUT = OUT.with_name(
        "engine-b-moment-performance.json"
        if not args.batch_size
        else f"engine-b-moment-performance-batch{args.batch_size}.json"
    )
    if args.scan:
        OUT = OUT.with_name(OUT.stem + "-scan.json")
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    eddington_engine.sample_speed_from_f_table = functools.partial(
        sample_speed_from_f_table, n_w=1024
    )
    theta = jnp.array([0.6, 2.0, 0.8])
    direction = jnp.array([0.0, 0.0, 1.0])
    key = jax.random.PRNGKey(1)

    def evaluate(t, e, k):
        return jax.jvp(lambda x: moment(x, k), (t,), (e,))

    record = dict(
        scope="100000-star target, 4 realizations, 50000 draws/component, 1024 speeds; core-scale JVP; seed 1",
        batch_size=args.batch_size,
        explicit_scan=args.scan,
        modes={},
    )
    from progenax.cluster import sampling
    from progenax.kinematics.eddington import assign_om_directions

    def velocities(state, keys, direction_key, positions, radii, grid, labels, G, mass):
        psi = jnp.interp(radii, grid, state.Psi_grid)

        def one(item):
            return sample_speed_from_f_table(
                item[0], item[1], state.E_grid, state.f_j_grid[item[2]], n_w=1024
            )

        inputs = (keys, psi, labels)
        if args.scan:
            count = labels.shape[0]
            padding = (-count) % args.batch_size
            padded = jax.tree.map(
                lambda x: jnp.pad(
                    x, [(0, padding)] + [(0, 0)] * (x.ndim - 1), mode="edge"
                ),
                inputs,
            )
            chunks = jax.tree.map(
                lambda x: x.reshape((-1, args.batch_size) + x.shape[1:]), padded
            )

            def step(carry, chunk):
                return carry, jax.vmap(one)(chunk)

            _, speeds = jax.lax.scan(step, None, chunks)
            speeds = speeds.reshape(-1)[:count]
        else:
            speeds = jax.lax.map(one, inputs, batch_size=args.batch_size)
        speeds = jnp.sqrt(G * mass / (4 * jnp.pi * state.mu)) * speeds
        return assign_om_directions(
            direction_key, positions, speeds, state.r_a_j[labels]
        )

    reference = None
    if args.batch_size:
        sampling.engine_b_star_velocities = velocities
        baseline = json.loads(
            Path(
                "docs/reviews/plummer-om-artifacts/engine-b-moment-performance.json"
            ).read_text()
        )
        reference = np.array(baseline["modes"]["outer_jit"]["value_and_jvp"])
    modes = (
        [("outer_jit", jax.jit(evaluate))]
        if args.batch_size
        else [("existing", evaluate), ("outer_jit", jax.jit(evaluate))]
    )
    for name, fn in modes:
        start = time.perf_counter()
        result = jax.block_until_ready(fn(theta, direction, key))
        cold = time.perf_counter() - start
        warm = []
        for _ in range(3):
            start = time.perf_counter()
            result = jax.block_until_ready(fn(theta, direction, key))
            warm.append(time.perf_counter() - start)
        values = np.asarray(result)
        if reference is None:
            reference = values
        np.testing.assert_allclose(values, reference, rtol=1e-12, atol=1e-12)
        row = dict(
            first_call_seconds=cold,
            warm_seconds=warm,
            value_and_jvp=values.tolist(),
            parity_pass=True,
        )
        if name == "outer_jit":
            mem = fn.lower(theta, direction, key).compile().memory_analysis()
            row["compiled_temporary_bytes"] = mem.temp_size_in_bytes
        record["modes"][name] = row
        OUT.write_text(json.dumps(record, indent=2) + "\n")
        print(name, row, flush=True)


if __name__ == "__main__":
    main()
