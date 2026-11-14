"""Isolate index-sliced versus explicit-chunk checkpoint boundaries."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from profile_engine_b_reduction import moment

OUT = Path("docs/reviews/plummer-om-artifacts/engine-b-reverse-chunk-boundary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, choices=[64, 256], default=64)
    args = parser.parse_args()
    global OUT
    if args.batch_size == 256:
        OUT = OUT.with_name("engine-b-reverse-chunk256.json")
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    theta = jnp.array([0.6, 2.0, 0.8])
    key = jax.random.PRNGKey(1)
    reference = json.loads(
        Path(
            "docs/reviews/plummer-om-artifacts/engine-b-streamed-reverse.json"
        ).read_text()
    )
    result = {}
    for mode in ["chunk_map", "map"] if args.batch_size == 64 else ["chunk_map"]:
        fn = jax.jit(
            jax.value_and_grad(
                lambda t: moment(
                    t, key, execution=mode, checkpoint=True, batch=args.batch_size
                )
            )
        )
        start = time.perf_counter()
        compiled = fn.lower(theta).compile()
        compile_time = time.perf_counter() - start
        mem = compiled.memory_analysis().temp_size_in_bytes
        assert mem < 32_000_000_000
        row = dict(compile_seconds=compile_time, temporary_bytes=mem, seconds=[])
        result[mode] = row
        OUT.write_text(json.dumps(result, indent=2) + "\n")
        print(mode, "compiled", row, flush=True)
        for _ in range(3):
            start = time.perf_counter()
            v, g = jax.block_until_ready(compiled(theta))
            row["seconds"].append(time.perf_counter() - start)
            np.testing.assert_allclose(v, reference["value"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                g, reference["reference_forward"], rtol=1e-12, atol=1e-12
            )
            row.update(value=float(v), gradient=g.tolist(), parity_pass=True)
            OUT.write_text(json.dumps(result, indent=2) + "\n")
            print(mode, row["seconds"][-1], flush=True)


if __name__ == "__main__":
    main()
