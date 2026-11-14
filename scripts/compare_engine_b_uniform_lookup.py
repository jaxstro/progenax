"""Matched retained-model reverse comparison; preserves production defaults."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import profile_engine_b_reduction as p
from diagnose_engine_b_uniform_lookup import speed

from progenax.kinematics.eddington import _interp_uniform as interpolate
from progenax.kinematics.eddington import sample_speed_from_f_table


def main():
    out = Path(
        "docs/reviews/plummer-om-artifacts/engine-b-uniform-energy-comparison.json"
    )
    assert not out.exists()
    assert jax.config.jax_enable_x64
    q = jnp.concatenate((jnp.linspace(-0.2, 2.2, 113), jnp.linspace(0.1, 2.0, 1000)))

    def test(t, fn):
        e = t[0] * jnp.linspace(0.1, 2.0, 1000)
        return fn(q, e, jnp.sin(e * t[1]))

    t = jnp.array([1.0, 0.7])
    for transform in (lambda f: f, jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            transform(lambda t: test(t, interpolate))(t),
            transform(lambda t: test(t, jnp.interp))(t),
            rtol=1e-12,
            atol=1e-12,
        )
    x = jnp.array([0.6, 2.0, 0.8])
    key = jax.random.PRNGKey(1)
    records = {}
    executables = {}
    for name, kernel in [("generic", sample_speed_from_f_table), ("uniform", speed)]:
        p.sample_speed_from_f_table = kernel
        start = time.perf_counter()
        executables[name] = (
            jax.jit(jax.value_and_grad(lambda t: p.moment(t, key))).lower(x).compile()
        )
        mem = executables[name].memory_analysis().temp_size_in_bytes
        assert mem < 32_000_000_000
        records[name] = {
            "compile_seconds": time.perf_counter() - start,
            "temporary_bytes": mem,
            "seconds": [],
        }
    ref = None
    for name in ["generic", "uniform", "uniform", "generic", "generic", "uniform"]:
        start = time.perf_counter()
        result = jax.block_until_ready(executables[name](x))
        elapsed = time.perf_counter() - start
        if ref is None:
            ref = result
        for a, b in zip(result, ref):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        records[name]["seconds"].append(elapsed)
        print(name, elapsed, flush=True)
        out.write_text(
            json.dumps(
                {
                    "records": records,
                    "order": [
                        "generic",
                        "uniform",
                        "uniform",
                        "generic",
                        "generic",
                        "uniform",
                    ],
                    "parity_pass": True,
                    "interpolation_value_jvp_reverse_pass": True,
                    "value": float(result[0]),
                    "gradient": result[1].tolist(),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
