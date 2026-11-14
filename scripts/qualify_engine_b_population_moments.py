"""Approved eight-batch precision check for the retained Engine B model."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_engine_b_population_moments import model_at

ROOT = Path("docs/reviews/plummer-om-artifacts")
OUT = ROOT / "engine-b-population-qualification-production.json"
REF = ROOT / "engine-b-reference-convergence.json"


def moment(theta, key):
    return model_at(theta).estimate_population_moments(
        key, 100000, 1.0, n_realizations=4, draws_per_component=50000
    )["mean_speed_squared"]


def anisotropic_moment(theta, key):
    model = model_at(theta, r_a_j=jnp.array([3.0, jnp.inf]))
    return model.estimate_population_moments(
        key, 100000, 1.0, n_realizations=4, draws_per_component=50000
    )["mean_speed_squared"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speed-nodes",
        type=int,
        choices=[1024],
        default=1024,
        help="Production resolution; historical 256-node runs are retained artifacts.",
    )
    parser.add_argument("--anisotropic", action="store_true")
    args = parser.parse_args()
    global OUT, REF
    if args.anisotropic:
        OUT = ROOT / "engine-b-anisotropic-population-qualification.json"
        REF = ROOT / "engine-b-anisotropic-reference-convergence.json"
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    retained = json.loads(REF.read_text())
    assert retained["status"] == "completed"
    gradients = np.array([c["gradient"] for c in retained["cases"]])
    ref = gradients[-1]
    allowance = abs(gradients[-2] - gradients[-3]) + abs(gradients[-1] - gradients[-2])
    if args.anisotropic:
        fine = json.loads(
            (ROOT / "engine-b-anisotropic-reference-3200.json").read_text()
        )
        fd = json.loads((ROOT / "engine-b-anisotropic-reference-fd.json").read_text())
        assert all(fine["passed"]) and all(fd["passed"])
        assert all(all(c["passed"]) for c in retained["comparisons"][1:])
        ref = np.array(fine["gradient"])
        allowance = abs(ref - gradients[-1]) + abs(gradients[-1] - gradients[-2])
    else:
        assert all(all(c["passed"]) for c in retained["comparisons"])
    theta = jnp.array(retained["theta"])
    record = dict(
        status="running",
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        reference_sha256=hashlib.sha256(REF.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference=ref.tolist(),
        reference_allowance=allowance.tolist(),
        protocol=dict(
            speed_nodes=args.speed_nodes,
            execution=(
                "public anisotropic API; checkpointed map64"
                if args.anisotropic
                else "public production API; checkpointed map64; uniform lookup"
            ),
            theta=theta.tolist(),
            target_stars=100000,
            G=1.0,
            realizations=4,
            draws_per_component=50000,
            seeds=list(range(1, 9)),
            discrepancy_gate=0.01,
            relative_se_gate=0.0025,
            consistency="3 SE + sum of final absolute speed/radial refinement changes",
            scope=(
                "halo ra=3, isotropic core; three profile derivatives"
                if args.anisotropic
                else "retained isotropic Engine B; public API; three first derivatives"
            ),
        ),
        batches=[],
    )

    def save():
        OUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    save()

    def evaluate(t, e, k):
        fn = anisotropic_moment if args.anisotropic else moment
        return jax.jvp(lambda tt: fn(tt, k), (t,), (e,))

    for seed in range(1, 9):
        derivatives, times = [], []
        for direction in jnp.eye(3):
            start = time.perf_counter()
            value, derivative = jax.block_until_ready(
                evaluate(theta, direction, jax.random.PRNGKey(seed))
            )
            times.append(time.perf_counter() - start)
            derivatives.append(float(derivative))
        record["batches"].append(
            dict(seed=seed, value=float(value), derivative=derivatives, seconds=times)
        )
        save()
        print("batch", seed, flush=True)
    x = np.array([b["derivative"] for b in record["batches"]])
    mean = x.mean(0)
    se = x.std(0, ddof=1) / np.sqrt(8)
    discrepancy = abs(mean - ref)
    record.update(
        status="completed",
        mean=mean.tolist(),
        se=se.tolist(),
        relative_discrepancy=(discrepancy / abs(ref)).tolist(),
        relative_se=(se / abs(ref)).tolist(),
        accuracy_pass=(discrepancy <= 0.01 * abs(ref)).tolist(),
        precision_pass=(se <= 0.0025 * abs(ref)).tolist(),
        consistency_pass=(discrepancy <= 3 * se + allowance).tolist(),
    )
    save()
    print(json.dumps({k: v for k, v in record.items() if k != "batches"}), flush=True)
    if not all(
        all(record[name])
        for name in ("accuracy_pass", "precision_pass", "consistency_pass")
    ):
        raise SystemExit(
            "Engine B production precision qualification failed; artifact retained"
        )


if __name__ == "__main__":
    main()
