"""Stage timings for the unchanged W0=7 moment qualification workload."""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from diagnose_population_moments import THETA, build_model

OUT = Path("docs/reviews/plummer-om-artifacts/population-moment-stage-cost.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output
    assert jax.config.jax_enable_x64
    if output.exists():
        raise FileExistsError(output)
    report = {
        "theta": [2.3, 0.4, 7.0],
        "seed": 1,
        "direction": "alpha",
        "x64": True,
        "target_stars": 100000,
        "realizations": 4,
        "draws_per_component": 25000,
        "timings": {},
        "memory_unit": "bytes; macOS ru_maxrss process high-water mark",
    }

    report["sampling_source_sha256"] = hashlib.sha256(
        Path("src/progenax/cluster/sampling.py").read_bytes()
    ).hexdigest()

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    def timed(name, fn):
        start = time.perf_counter()
        result = jax.block_until_ready(fn())
        report["timings"][name] = time.perf_counter() - start
        report["process_maxrss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save()
        print(name, report["timings"][name], flush=True)
        return result

    theta = THETA.at[2].set(7.0)
    tangent = jnp.array([1.0, 0.0, 0.0])
    prepare = eqx.filter_jit(lambda t, d: eqx.filter_jvp(build_model, (t,), (d,)))
    model, model_dot = timed("model_cold", lambda: prepare(theta, tangent))
    timed("model_warm", lambda: prepare(theta, tangent))

    def moments(m):
        values = m.estimate_population_moments(
            jax.random.PRNGKey(1),
            100000,
            G=1.0,
            n_realizations=4,
            draws_per_component=25000,
        )
        return jnp.stack(
            [
                values[k]
                for k in ("mean_mass", "mean_radius_squared", "mean_speed_squared")
            ]
        )

    sample = eqx.filter_jit(lambda m, d: eqx.filter_jvp(moments, (m,), (d,)))
    values, derivative = timed("sampling_cold", lambda: sample(model, model_dot))
    timed("sampling_warm", lambda: sample(model, model_dot))
    old = json.loads(OUT.with_name("population-moment-W0-7.json").read_text())[
        "batches"
    ][0]
    report.update(
        values=values.tolist(),
        derivative=derivative.tolist(),
        values_match=bool(np.allclose(values, old["values"], rtol=1e-12, atol=1e-12)),
        derivative_match=bool(
            np.allclose(
                derivative, np.array(old["derivatives"])[:, 0], rtol=1e-12, atol=1e-12
            )
        ),
    )
    save()
    print(json.dumps(report, indent=2), flush=True)
    assert report["values_match"] and report["derivative_match"]


if __name__ == "__main__":
    main()
