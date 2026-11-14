"""Frozen Engine A qualification of two approved scalar public observables.

A is squared total catalogue mass. B is twice kinetic energy per star. The
physical catalogue has independent component labels; neither observable is
recentered, virial-rescaled, or divided by the catalogue's total mass.
"""

import argparse
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from diagnose_population_moments import THETA, build_model

from progenax.cluster._radial_cdf import potential_interp
from progenax.profiles.limepy import lowered_exponential

N = 100_000
R = 4
G = 1.0
NAMES = ["total_mass_squared", "twice_kinetic_energy_per_star"]


def total_mass_squared(ic):
    """M^2, in squared model mass units."""
    return jnp.sum(ic.masses) ** 2


def twice_kinetic_energy_per_star(ic):
    """2T/N, in model mass times velocity-squared units."""
    return jnp.mean(ic.masses * jnp.sum(ic.velocities**2, axis=1))


def mixture_references(probability, mass, q, scale, n_stars):
    """Independent categorical identities with q=E[w_j^2 u^2 | j]."""
    mean_mass = probability @ mass
    mass_squared = n_stars * (probability @ mass**2) + (
        n_stars * (n_stars - 1) * mean_mass**2
    )
    kinetic = scale * jnp.sum(
        probability * mass * q * (mass + (n_stars - 1) * mean_mass)
    )
    return jnp.array([mass_squared, kinetic])


def reference_model(model, subdivisions=1, n_stars=N, gravity=G):
    """Radial quadrature plus analytic isotropic lowered-DF second moments.

    Independent of random inverse-CDF sampling. Shares the equilibrium model;
    does not independently verify the equilibrium solver. Refinement preserves
    the sampler's piecewise-linear radial density and interpolated potential.
    """
    r = model._r_grid
    pdf = model.engine_a.radial_pdf_j
    normalization = jnp.trapezoid(pdf, r, axis=1)
    fractions = jnp.arange(subdivisions) / subdivisions
    fine_r = (r[:-1, None] + jnp.diff(r)[:, None] * fractions).ravel()
    fine_pdf = (
        pdf[:, :-1, None] + jnp.diff(pdf, axis=1)[:, :, None] * fractions
    ).reshape(model.m_j.size, -1)
    r = jnp.concatenate((fine_r, r[-1:]))
    pdf = jnp.concatenate((fine_pdf, pdf[:, -1:]), axis=1)
    psi = potential_interp(
        r / model.r_c,
        model.xi_grid,
        model.engine_a.psi_raw,
        left=model.W0,
        right=0.0,
    )
    W = model.rescale_j[:, None] * jnp.maximum(psi, 0.0)
    safe = jnp.where(W > 0, W, 1.0)
    speed = jnp.where(
        W > 0,
        3
        * lowered_exponential(model.g + 2.5, safe)
        / lowered_exponential(model.g + 1.5, safe),
        0.0,
    )
    q = model.w_j**2 * jnp.trapezoid(pdf * speed, r, axis=1) / normalization
    probability = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    return mixture_references(
        probability,
        model.m_j,
        q,
        gravity / (9 * model.r_c * model.mu_tot),
        n_stars,
    )


def public_model(model, key):
    """Both scalar calls use the unchanged general public expectation API."""
    return jnp.stack(
        [
            model.estimate_expectation(observable, key, N, G=G, n_realizations=R)
            for observable in (total_mass_squared, twice_kinetic_energy_per_star)
        ]
    )


def conditional_model(model, key):
    """Public bounded method; the script-only candidate has been removed."""
    result = model.estimate_mass_squared_and_kinetic(
        key, N, G=G, n_realizations=R, draws_per_component=N // model.m_j.size
    )
    return jnp.stack([result[name] for name in NAMES])


def construct(theta, adjoint=diffrax.ForwardMode()):
    # Explicitly tighter fraction fit resolves theoretically zero mass sensitivities.
    # Production defaults and the physical population are unchanged.
    return build_model(theta, adjoint=adjoint, fit_tol=1e-10, n_iter=60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, choices=(1, 8), default=8)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument(
        "--compare",
        type=Path,
        help="Frozen same-seed candidate evidence for production transfer",
    )
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(f"Retain existing exposure: {output}")
    assert jax.config.jax_enable_x64
    report = {
        "status": "started",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in map(
                Path,
                [
                    __file__,
                    "scripts/diagnose_population_moments.py",
                    "src/progenax/cluster/sampling.py",
                    "src/progenax/cluster/multicomponent.py",
                    "src/progenax/cluster/_radial_cdf.py",
                    "src/progenax/profiles/limepy_tables.py",
                ],
            )
        },
        "protocol": {
            "scope": "retained isotropic Engine A model; bounded conditional candidate"
            if args.conditional
            else "retained isotropic Engine A model; two scalar uses of general API",
            "estimator": "public conditional A/B method; not general-score precision"
            if args.conditional
            else "unchanged public general score-plus-pathwise API",
            "observables": NAMES,
            "theta": THETA.tolist(),
            "parameters": ["alpha", "delta", "W0"],
            "target_stars": N,
            "realizations_per_batch": R,
            "seeds": list(range(1, args.batches + 1)),
            "n_components": 4,
            "g": 1.0,
            "fit_tol": 1e-10,
            "n_iter": 60,
            "G": G,
            "units": "retained G=1 model units; not pc/Myr velocities",
            "fd_steps": [1e-4, 1e-5],
            "reference_subdivisions": [1, 2, 4],
            "deterministic_relative_gate": 0.003,
            "relative_discrepancy_gate": 0.01,
            "relative_standard_error_gate": 0.0025,
            "consistency_standard_errors": 3,
            "mass_shape_identity_scaled_gate": 1e-8,
            "compiler_temporary_guard_bytes": 32_000_000_000,
            "precision": "float64",
            "reverse": args.reverse,
        },
        "references": [],
        "batches": [],
        "timings": {},
        "limitations": [
            "One model, two observables; not arbitrary-domain precision or higher AD.",
            "Reference shares the equilibrium constructor, not random inverse sampling.",
            "Reference error is empirical refinement/FD discrepancy, not a rigorous bound.",
            "Four catalogues per batch do not suppress whole-catalogue score variance.",
        ],
    }

    def save():
        report["process_high_water_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def timed(name, fn):
        start = time.perf_counter()
        result = jax.block_until_ready(fn())
        report["timings"][name] = time.perf_counter() - start
        save()
        print(name, report["timings"][name], flush=True)
        return result

    save()
    prepare = eqx.filter_jit(lambda t, d: eqx.filter_jvp(construct, (t,), (d,)))
    model, model_dot = timed(
        "constructor_first_jvp", lambda: prepare(THETA, jnp.eye(3)[0])
    )
    probability = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    report["component_mass"] = model.m_j.tolist()
    report["component_probability"] = probability.tolist()
    report["expected_occupancy"] = (N * probability).tolist()
    save()
    derivatives = []
    for level in (1, 2, 4):
        fn = lambda t: reference_model(construct(t), level)
        evaluate = jax.jit(lambda t: (fn(t), jax.jacfwd(fn)(t)))
        value, derivative = timed(f"reference_{level}", lambda: evaluate(THETA))
        report["references"].append(
            dict(
                subdivisions=level,
                values=value.tolist(),
                derivatives=derivative.tolist(),
            )
        )
        derivatives.append(np.asarray(derivative))
        save()
    ref = jax.jit(lambda t: reference_model(construct(t), 4))
    errors = [abs(derivatives[-1] - derivatives[-2])]
    for h in (1e-4, 1e-5):
        fd = np.asarray(
            jnp.stack(
                [
                    (ref(THETA + h * e) - ref(THETA - h * e)) / (2 * h)
                    for e in jnp.eye(3)
                ],
                axis=1,
            )
        )
        report[f"reference_fd_{h}"] = fd.tolist()
        errors.append(abs(fd - derivatives[-1]))
        save()
    rd = derivatives[-1]
    error = np.maximum.reduce(errors)
    cells = error <= 0.003 * abs(rd)
    # M depends only on the IMF. Do not divide by numerical residuals in delta/W0.
    zero_scale = np.asarray(THETA)[1:] / float(value[0])
    cells[0, 1:] = np.maximum(abs(rd[0, 1:]), error[0, 1:]) * zero_scale <= 1e-8
    report["reference_error_estimate"] = error.tolist()
    report["reference_pass"] = cells.tolist()
    save()
    if not cells.all():
        report["status"] = "reference gate failed; stochastic campaign withheld"
        save()
        raise SystemExit(1)

    estimator = conditional_model if args.conditional else public_model
    sample = eqx.filter_jit(
        lambda m, d, k: eqx.filter_jvp(
            lambda current: estimator(current, k), (m,), (d,)
        )
    )
    key = jax.random.PRNGKey(1)
    lowered = sample.lower(model, model_dot, key)
    compiled = timed("sampling_compile", lowered.compile)
    memory = compiled.compiled.memory_analysis()
    report["sampling_memory_analysis"] = {
        name: int(getattr(memory, name))
        for name in (
            "temp_size_in_bytes",
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
        )
    }
    save()
    if memory.temp_size_in_bytes > 32_000_000_000:
        report["status"] = "compiler temporary guard exceeded; execution withheld"
        save()
        raise SystemExit(1)
    for seed in range(1, args.batches + 1):
        columns = []
        for direction in range(3):
            model, model_dot = timed(
                f"constructor_{seed}_{direction}",
                lambda: prepare(THETA, jnp.eye(3)[direction]),
            )
            means, column = timed(
                f"batch_{seed}_{direction}",
                lambda: compiled(model, model_dot, jax.random.PRNGKey(seed)),
            )
            columns.append(column)
        report["batches"].append(
            dict(
                seed=seed,
                values=means.tolist(),
                derivatives=jnp.stack(columns, axis=1).tolist(),
            )
        )
        save()
    samples = np.asarray([b["derivatives"] for b in report["batches"]])
    report["derivative_mean"] = samples.mean(0).tolist()
    if args.batches == 8:
        mean = samples.mean(0)
        se = samples.std(0, ddof=1) / np.sqrt(8)
        discrepancy = abs(mean - rd)
        stat = (
            (discrepancy <= 0.01 * abs(rd))
            & (se <= 0.0025 * abs(rd))
            & (discrepancy <= 3 * se + error)
        )
        stat[0, 1:] = (
            np.maximum(abs(samples[:, 0, 1:]).max(0), error[0, 1:]) * zero_scale <= 1e-8
        )
        report.update(
            standard_error=se.tolist(),
            relative_discrepancy=np.divide(
                discrepancy, abs(rd), out=np.zeros_like(rd), where=abs(rd) > 0
            ).tolist(),
            statistical_pass=stat.tolist(),
        )
    if args.reverse:
        # Separate scalar outputs, with checkpointed constructor reverse AD.
        objective = lambda t, weight: (
            estimator(construct(t, adjoint=None), key) @ weight
        )
        reverse = jax.jit(jax.value_and_grad(objective))
        reverse_compiled = timed(
            "reverse_compile", lambda: reverse.lower(THETA, jnp.eye(2)[0]).compile()
        )
        reverse_memory = reverse_compiled.memory_analysis()
        report["reverse_temp_bytes"] = reverse_memory.temp_size_in_bytes
        save()
        if reverse_memory.temp_size_in_bytes <= 32_000_000_000:
            rows = []
            for i in range(2):
                _, row = timed(
                    f"reverse_{i}", lambda: reverse_compiled(THETA, jnp.eye(2)[i])
                )
                rows.append(row)
            reverse_d = np.asarray(rows)
            forward_d = np.asarray(report["batches"][0]["derivatives"])
            report["reverse_derivatives"] = reverse_d.tolist()
            report["reverse_forward_pass"] = np.isclose(
                reverse_d, forward_d, rtol=0.003, atol=1e-8
            ).tolist()
        else:
            report["reverse_status"] = (
                "compiler temporary guard exceeded; execution withheld"
            )
    if args.compare is not None:
        prior = json.loads(args.compare.read_text())
        for field in (
            "theta",
            "target_stars",
            "realizations_per_batch",
            "n_components",
            "g",
            "fit_tol",
            "n_iter",
            "G",
        ):
            if prior["protocol"][field] != report["protocol"][field]:
                raise ValueError(f"Transfer protocol differs in {field}")
        if not np.asarray(prior["statistical_pass"]).all():
            raise ValueError("Transfer baseline did not pass its precision gates")
        previous = {b["seed"]: b for b in prior["batches"]}
        parity = []
        for batch in report["batches"]:
            old = previous[batch["seed"]]
            parity.append(
                {
                    "seed": batch["seed"],
                    "values": bool(
                        np.allclose(
                            batch["values"], old["values"], rtol=1e-12, atol=1e-12
                        )
                    ),
                    "derivatives": bool(
                        np.allclose(
                            batch["derivatives"],
                            old["derivatives"],
                            rtol=1e-12,
                            atol=1e-12,
                        )
                    ),
                }
            )
        report["transfer_parity"] = parity
        report["transfer_baseline"] = str(args.compare)
        report["transfer_baseline_sha256"] = hashlib.sha256(
            args.compare.read_bytes()
        ).hexdigest()

    report["status"] = (
        "completed qualification"
        if args.batches == 8
        else "completed one-batch diagnostic; no precision verdict"
    )
    save()
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "reference_pass",
                    "derivative_mean",
                    "process_high_water_rss_bytes",
                ]
            },
            indent=2,
        ),
        flush=True,
    )
    gates = []
    if args.batches == 8:
        gates.append(bool(np.asarray(report["statistical_pass"]).all()))
    if args.reverse:
        gates.append(bool(np.asarray(report.get("reverse_forward_pass", False)).all()))
    if args.compare is not None:
        gates.append(
            all(b["values"] and b["derivatives"] for b in report["transfer_parity"])
        )
    report["all_requested_gates_pass"] = all(gates)
    save()
    if not all(gates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
