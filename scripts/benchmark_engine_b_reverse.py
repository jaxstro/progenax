"""Matched full-JIT reverse AD: map chunk sums versus fori accumulation."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from profile_engine_b_reduction import moment

OUT = Path("docs/reviews/plummer-om-artifacts/engine-b-reverse-loop-comparison.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    global OUT
    if args.checkpoint:
        OUT = OUT.with_name("engine-b-reverse-loop-checkpoint.json")
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    theta = jnp.array([0.6, 2.0, 0.8])
    key = jax.random.PRNGKey(1)
    record = dict(
        scope="matched indexed-radial scalar moment; 64-draw chunks, 1024 speed nodes, four realizations, 100000 draws each; all three reverse derivatives; CPU float64",
        checkpoint=args.checkpoint,
        modes={},
    )
    reference = json.loads(
        Path(
            "docs/reviews/plummer-om-artifacts/engine-b-streamed-reverse.json"
        ).read_text()
    )
    compiled = {}

    def save():
        OUT.write_text(json.dumps(record, indent=2) + "\n")

    for mode in ["map", "fori"]:
        fn = jax.jit(
            jax.value_and_grad(
                lambda t: moment(t, key, execution=mode, checkpoint=args.checkpoint)
            )
        )
        start = time.perf_counter()
        executable = fn.lower(theta).compile()
        compile_seconds = time.perf_counter() - start
        mem = executable.memory_analysis()
        record["modes"][mode] = dict(
            compile_seconds=compile_seconds,
            temporary_bytes=mem.temp_size_in_bytes,
            argument_bytes=mem.argument_size_in_bytes,
            output_bytes=mem.output_size_in_bytes,
            warm_seconds=[],
        )
        save()
        print(mode, "compiled", record["modes"][mode], flush=True)
        # Leave at least half of the user's 64 GB machine for the OS and
        # allocations not represented by this compiler temporary estimate.
        if mem.temp_size_in_bytes > 32_000_000_000:
            record["modes"][mode]["execution"] = (
                "skipped: temporary estimate exceeds 32 GB safety budget"
            )
            save()
            continue
        start = time.perf_counter()
        v, g = jax.block_until_ready(executable(theta))
        first = time.perf_counter() - start
        np.testing.assert_allclose(v, reference["value"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            g, reference["reference_forward"], rtol=1e-12, atol=1e-12
        )
        record["modes"][mode].update(
            first_execution_seconds=first,
            value=float(v),
            gradient=g.tolist(),
            parity_pass=True,
        )
        compiled[mode] = executable
        save()
        print(mode, "first execution", first, flush=True)
    for order in [("map", "fori"), ("fori", "map"), ("map", "fori")]:
        for mode in order:
            if mode not in compiled:
                continue
            start = time.perf_counter()
            jax.block_until_ready(compiled[mode](theta))
            elapsed = time.perf_counter() - start
            record["modes"][mode]["warm_seconds"].append(elapsed)
            save()
            print(mode, "warm", elapsed, flush=True)
    record["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
