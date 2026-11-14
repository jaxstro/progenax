"""Matched finest-grid reference JVP benchmark; quadrature unchanged."""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_anisotropic_population_moments import THETA, reference

OUT = Path("docs/reviews/plummer-om-artifacts/anisotropic-reference-batching.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=[32, 64, 128])
    args = parser.parse_args()
    global OUT
    if args.batch:
        OUT = OUT.with_name(f"anisotropic-reference-batching-{args.batch}.json")
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    retained = json.loads(
        Path(
            "docs/reviews/plummer-om-artifacts/population-moment-anisotropic-mapped-pilot.json"
        ).read_text()
    )["references"][-1]
    report = dict(
        scope="same finest-grid model, quadrature, four directional JVPs; execution batching only",
        source_sha256=hashlib.sha256(
            Path("scripts/diagnose_anisotropic_population_moments.py").read_bytes()
        ).hexdigest(),
        cases=[],
    )
    for batch in [args.batch] if args.batch else [32, 128]:
        fn = lambda t: reference(t, 4, batch_size=batch)[0]
        evaluate = jax.jit(lambda t, e: jax.jvp(fn, (t,), (e,)))
        start = time.perf_counter()
        compiled = evaluate.lower(THETA, jnp.eye(4)[0]).compile()
        case = dict(
            batch_size=batch,
            compile_seconds=time.perf_counter() - start,
            temp_bytes=compiled.memory_analysis().temp_size_in_bytes,
            seconds=[],
            derivatives=[],
        )
        # Warm once, then measure all four directions with synchronization.
        jax.block_until_ready(compiled(THETA, jnp.eye(4)[0]))
        for e in jnp.eye(4):
            start = time.perf_counter()
            v, d = jax.block_until_ready(compiled(THETA, e))
            case["seconds"].append(time.perf_counter() - start)
            print(
                f"batch {batch} direction {len(case['seconds'])}: {case['seconds'][-1]:.3f} s",
                flush=True,
            )
            case["derivatives"].append(float(d))
        case["value"] = float(v)
        case["peak_process_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        case["value_relative_error"] = float(abs(v / retained["value"] - 1))
        case["derivative_relative_error"] = (
            abs(np.array(case["derivatives"]) / np.array(retained["derivative"]) - 1)
        ).tolist()
        report["cases"].append(case)
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
