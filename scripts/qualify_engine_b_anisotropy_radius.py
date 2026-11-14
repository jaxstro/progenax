"""Retained r_a=3 halo: reference convergence and public moment derivative."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_engine_b_population_moments import df_speed_moment, model_at

ROOT = Path("docs/reviews/plummer-om-artifacts")
REFERENCE = ROOT / "engine-b-ra-reference.json"


def reference(radius, nw, nr):
    return df_speed_moment(
        jnp.array([0.6, 2.0, 0.8]), n_w=nw, n_r=nr, r_a_j=jnp.array([radius, jnp.inf])
    )


def moment(radius, key):
    model = model_at(jnp.array([0.6, 2.0, 0.8]), r_a_j=jnp.array([radius, jnp.inf]))
    return model.estimate_population_moments(
        key, 100000, 1.0, n_realizations=4, draws_per_component=50000
    )["mean_speed_squared"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--refine-6400", action="store_true")
    parser.add_argument(
        "--reference-nodes", type=int, choices=[6400, 12800, 25600], default=6400
    )
    parser.add_argument("--angular-average", action="store_true")
    parser.add_argument("--speed-average", action="store_true")
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--knot-average", action="store_true")
    args = parser.parse_args()
    if args.knot_average and not (args.stratified and args.speed_average):
        parser.error("knot averaging requires --stratified --speed-average")
    if args.stratified and not args.speed_average:
        parser.error("stratified requires --speed-average")
    if args.speed_average and (not args.campaign or args.angular_average):
        parser.error(
            "speed averaging requires --campaign and excludes --angular-average"
        )
    if args.angular_average and not args.campaign:
        parser.error("angular averaging requires --campaign")
    if args.campaign and args.refine_6400:
        parser.error("choose campaign or reference refinement")
    assert jax.config.jax_enable_x64
    output = ROOT / "engine-b-ra-qualification.json" if args.campaign else REFERENCE
    if args.refine_6400:
        output = ROOT / f"engine-b-ra-reference-{args.reference_nodes}.json"
    if args.speed_average or args.angular_average:
        from progenax.cluster import sampling

        if not hasattr(sampling, "_engine_b_speed_moment"):
            raise RuntimeError(
                "Historical stochastic experiment: use its recorded Git revision; production now uses deterministic quadrature"
            )
    if args.angular_average:
        from prototype_engine_b_angular_average import _engine_b_speed_moment

        from progenax.cluster import sampling

        sampling._engine_b_speed_moment = _engine_b_speed_moment
        output = ROOT / "engine-b-ra-angular-average-qualification.json"
    if args.speed_average:
        from prototype_engine_b_speed_average import _engine_b_speed_moment

        from progenax.cluster import sampling

        sampling._engine_b_speed_moment = _engine_b_speed_moment
        output = ROOT / "engine-b-ra-speed-average-qualification.json"
        if args.stratified:
            from functools import partial

            sampling._engine_b_speed_moment = partial(
                _engine_b_speed_moment, stratified=True, knot_average=args.knot_average
            )
            output = ROOT / "engine-b-ra-stratified-qualification.json"
    if args.knot_average:
        output = ROOT / "engine-b-ra-knot-qualification.json"
    if output.exists():
        raise FileExistsError(output)
    record = dict(
        status="running",
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        radius=3.0,
        angular_average=args.angular_average or args.speed_average,
        speed_average=args.speed_average,
        stratified=args.stratified,
        knot_average=args.knot_average,
        cases=[],
    )

    def save():
        output.write_text(json.dumps(record, indent=2) + "\n")

    save()
    if args.refine_6400:
        old = json.loads(REFERENCE.read_text())
        f = lambda r: reference(r, args.reference_nodes, 12000)
        start = time.perf_counter()
        value, gradient = jax.block_until_ready(jax.value_and_grad(f)(jnp.array(3.0)))
        h = 3e-4
        fd = float((f(3.0 + h) - f(3.0 - h)) / (2 * h))
        relative = abs(float(gradient) - old["cases"][-1]["gradient"]) / abs(
            float(gradient)
        )
        record.update(
            status="completed",
            n_w=args.reference_nodes,
            n_r=12000,
            value=float(value),
            gradient=float(gradient),
            fd=fd,
            relative_ad_fd=abs(float(gradient) - fd) / abs(fd),
            relative_gradient_change=relative,
            refinement_pass=relative <= 0.003,
            seconds=time.perf_counter() - start,
            previous_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        )
        save()
        print(record, flush=True)
        if not record["refinement_pass"] or record["relative_ad_fd"] > 0.003:
            raise SystemExit("6400-node reference failed; evidence retained")
        return
    if not args.campaign:
        for nw, nr in [
            (400, 6000),
            (800, 6000),
            (1600, 6000),
            (1600, 12000),
            (3200, 12000),
        ]:
            f = lambda r: reference(r, nw, nr)
            start = time.perf_counter()
            v, g = jax.block_until_ready(jax.value_and_grad(f)(jnp.array(3.0)))
            h = 3e-4
            fd = float((f(3.0 + h) - f(3.0 - h)) / (2 * h))
            row = dict(
                n_w=nw,
                n_r=nr,
                value=float(v),
                gradient=float(g),
                fd=fd,
                relative_ad_fd=abs(float(g) - fd) / abs(fd),
                seconds=time.perf_counter() - start,
            )
            record["cases"].append(row)
            save()
            print(row, flush=True)
        g = np.array([c["gradient"] for c in record["cases"]])
        record.update(
            status="completed",
            relative_changes=(abs(np.diff(g)) / abs(g[1:])).tolist(),
            ad_fd_pass=all(c["relative_ad_fd"] <= 0.003 for c in record["cases"]),
            final_refinement_pass=bool(
                abs(g[-1] - g[-2]) / abs(g[-1]) <= 0.003
                and abs(g[-2] - g[-3]) / abs(g[-2]) <= 0.003
            ),
        )
        save()
        if not (record["ad_fd_pass"] and record["final_refinement_pass"]):
            raise SystemExit("Reference qualification failed; evidence retained")
        return
    ref = json.loads(REFERENCE.read_text())
    assert ref["status"] == "completed" and ref["ad_fd_pass"]
    gs = [c["gradient"] for c in ref["cases"]]
    allowance = abs(gs[-1] - gs[-2]) + abs(gs[-2] - gs[-3])
    fine_path = ROOT / "engine-b-ra-reference-6400.json"
    if fine_path.exists():
        fine = json.loads(fine_path.read_text())
        assert (
            fine["status"] == "completed"
            and fine["refinement_pass"]
            and fine["relative_ad_fd"] <= 0.003
        )
        allowance = abs(fine["gradient"] - gs[-1]) + abs(gs[-2] - gs[-3])
        gs[-1] = fine["gradient"]
        record["refinement_sha256"] = hashlib.sha256(fine_path.read_bytes()).hexdigest()
    else:
        assert ref["final_refinement_pass"]
    if args.knot_average:
        knot_path = ROOT / "engine-b-ra-knot-reference.json"
        knot = json.loads(knot_path.read_text())
        gs[-1] = knot["cases"][-1]["fd"]
        allowance = abs(gs[-1] - knot["cases"][-2]["fd"])
        record["knot_reference_sha256"] = hashlib.sha256(
            knot_path.read_bytes()
        ).hexdigest()
    record.update(
        reference=gs[-1],
        allowance=allowance,
        reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        protocol=dict(
            target=100000,
            realizations=4,
            draws_per_component=50000,
            speed_nodes=None
            if args.knot_average
            else 4096
            if args.speed_average
            else 1024,
            seeds=list(range(1, 9)),
            discrepancy_gate=0.01,
            relative_se_gate=0.0025,
            consistency="3 SE + final speed/radial refinement",
        ),
    )
    for seed in range(1, 9):
        start = time.perf_counter()
        v, g = jax.block_until_ready(
            jax.jvp(
                lambda r: moment(r, jax.random.PRNGKey(seed)),
                (jnp.array(3.0),),
                (jnp.array(1.0),),
            )
        )
        record["cases"].append(
            dict(
                seed=seed,
                value=float(v),
                gradient=float(g),
                seconds=time.perf_counter() - start,
            )
        )
        save()
        print(record["cases"][-1], flush=True)
    g = np.array([c["gradient"] for c in record["cases"]])
    mean = g.mean()
    se = g.std(ddof=1) / np.sqrt(8)
    error = abs(mean - gs[-1])
    record.update(
        status="completed",
        mean=float(mean),
        se=float(se),
        relative_discrepancy=float(error / abs(gs[-1])),
        relative_se=float(se / abs(gs[-1])),
        accuracy_pass=bool(error <= 0.01 * abs(gs[-1])),
        precision_pass=bool(se <= 0.0025 * abs(gs[-1])),
        consistency_pass=bool(error <= 3 * se + allowance),
    )
    save()
    print(record, flush=True)
    if not all(
        record[k] for k in ("accuracy_pass", "precision_pass", "consistency_pass")
    ):
        raise SystemExit("Qualification failed; evidence retained")


if __name__ == "__main__":
    main()
