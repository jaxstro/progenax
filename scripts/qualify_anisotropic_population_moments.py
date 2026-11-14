"""Approved eight-batch anisotropic precision check and seed-1 reverse replay."""

import argparse
import functools
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from diagnose_anisotropic_population_moments import THETA, model

import progenax.cluster.sampling as sampling
from progenax.profiles.limepy_tables import AnisoSpeedCDFTable

OUT = Path(
    "docs/reviews/plummer-om-artifacts/anisotropic-population-qualification.json"
)
REF = Path(
    "docs/reviews/plummer-om-artifacts/population-moment-anisotropic-mapped-pilot.json"
)


def moments(t, key, adjoint):
    return model(t, adjoint=adjoint).estimate_population_moments(
        key, 100000, 1.0, n_realizations=4, draws_per_component=25000
    )["mean_speed_squared"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-nodes", type=int, choices=[48, 96], default=48)
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    global OUT
    if args.production:
        args.q_nodes = 96
        OUT = OUT.with_name("anisotropic-population-production.json")
    elif args.q_nodes != 48:
        OUT = OUT.with_name(
            f"anisotropic-population-qualification-q{args.q_nodes}.json"
        )
        # Process-local trial: production class defaults remain untouched.
        sampling.AnisoSpeedCDFTable = SimpleNamespace(
            build=functools.partial(AnisoSpeedCDFTable.build, n_p=args.q_nodes)
        )
    if not args.production and args.q_nodes == 48:
        sampling.AnisoSpeedCDFTable = SimpleNamespace(
            build=functools.partial(AnisoSpeedCDFTable.build, n_p=48)
        )
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    retained = json.loads(REF.read_text())
    assert retained["reference_pass"] and retained["fit_pass"]
    ref = np.asarray(retained["references"][-1]["derivative"])
    error = np.asarray(retained["relative_reference_error"]) * abs(ref)
    record = dict(
        status="sampling",
        reference_file=str(REF),
        reference_sha256=hashlib.sha256(REF.read_bytes()).hexdigest(),
        protocol=dict(
            theta=THETA.tolist(),
            speed_table_q_nodes=args.q_nodes,
            production_api=args.production,
            fit_tol=1e-10,
            n_iter=60,
            target_stars=100000,
            draws_per_component=25000,
            realizations_per_batch=4,
            seeds=list(range(1, 9)),
            relative_discrepancy_gate=0.01,
            relative_se_gate=0.0025,
            consistency_gate="3 SE + retained reference error",
            reverse_rtol=0.003,
            reverse_atol=1e-12,
            scope="one anisotropic Engine A model; mean squared speed first derivatives; see production_api for execution path",
        ),
        batches=[],
    )

    def save():
        OUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    save()
    evaluate = jax.jit(
        lambda t, e, k: jax.jvp(
            lambda tt: moments(tt, k, diffrax.ForwardMode()), (t,), (e,)
        )
    )
    for seed in range(1, 9):
        cols = []
        times = []
        for e in jnp.eye(4):
            start = time.perf_counter()
            v, d = jax.block_until_ready(evaluate(THETA, e, jax.random.PRNGKey(seed)))
            times.append(time.perf_counter() - start)
            cols.append(float(d))
        record["batches"].append(
            dict(seed=seed, value=float(v), derivative=cols, seconds=times)
        )
        save()
        print("batch", seed, flush=True)
    x = np.array([b["derivative"] for b in record["batches"]])
    mean = x.mean(0)
    se = x.std(0, ddof=1) / np.sqrt(8)
    dis = abs(mean - ref)
    record.update(
        mean=mean.tolist(),
        relative_discrepancy=(dis / abs(ref)).tolist(),
        relative_se=(se / abs(ref)).tolist(),
        pass_cells=(
            (dis <= 0.01 * abs(ref))
            & (se <= 0.0025 * abs(ref))
            & (dis <= 3 * se + error)
        ).tolist(),
        status="reverse replay",
    )
    save()
    jax.clear_caches()
    reverse = jax.jit(
        jax.value_and_grad(lambda t: moments(t, jax.random.PRNGKey(1), None))
    )
    start = time.perf_counter()
    v, d = jax.block_until_ready(reverse(THETA))
    record["reverse"] = dict(
        value=float(v),
        derivative=d.tolist(),
        seconds=time.perf_counter() - start,
        value_pass=bool(
            np.isclose(v, record["batches"][0]["value"], rtol=1e-12, atol=1e-12)
        ),
        derivative_pass=np.isclose(
            d, record["batches"][0]["derivative"], rtol=0.003, atol=1e-12
        ).tolist(),
    )
    record["status"] = "completed"
    save()
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
