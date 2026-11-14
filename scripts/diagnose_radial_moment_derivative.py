"""Approved deterministic radial-quantile derivative diagnosis at retained truth."""

import hashlib
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_population_moments import THETA, build_model, timed

from progenax.cluster._radial_cdf import radial_quantile

OUT = Path("docs/reviews/plummer-om-artifacts/radial-moment-derivative.json")


def arrays(theta):
    model = build_model(theta)
    return (
        model._r_grid,
        model.engine_a.radial_pdf_j,
        model._cdf_j,
        jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30)),
    )


def exact_moments(grid, pdf):
    width = jnp.diff(grid)
    left = grid[:-1]
    numerator = jnp.sum(
        width
        * (
            pdf[:, :-1] * (left**2 + left * width + width**2 / 3)
            + jnp.diff(pdf, axis=1)
            * (left**2 / 2 + 2 * left * width / 3 + width**2 / 4)
        ),
        axis=1,
    )
    return numerator / jnp.trapezoid(pdf, grid, axis=1)


def main():
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(f"Retain existing exposure: {OUT}")
    report = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "theta": THETA.tolist(),
        "x64": True,
        "orders": [4, 8, 16],
        "scope": "same retained model; fixed-probability quantile JVP integrated over each component; no new acceptance threshold",
        "levels": [],
    }

    def save():
        OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    (base, jac), report["preparation_seconds"] = timed(
        "model arrays and Jacobian",
        lambda: jax.jit(lambda t: (arrays(t), jax.jacfwd(arrays)(t)))(THETA),
    )
    grid, pdf, cdf, pi = base
    moments = exact_moments(grid, pdf)
    derivatives = jnp.stack(
        [
            jax.jvp(exact_moments, (grid, pdf), (jac[0][..., d], jac[1][..., d]))[1]
            for d in range(3)
        ],
        axis=1,
    )
    reference = jac[3].T @ moments + pi @ derivatives
    report.update(
        component_moments=moments.tolist(),
        component_derivatives=derivatives.tolist(),
        reference_derivatives=reference.tolist(),
    )
    save()
    # Zero-width cells carry no probability. Keep the full grid in the inverse.
    cells = np.flatnonzero(np.asarray(jnp.diff(grid)) > 0)
    width = jnp.diff(grid)[cells]
    for order in (4, 8, 16):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        fraction = jnp.asarray((nodes + 1) / 2)
        dx = width[:, None] * fraction
        rho0 = pdf[:, cells]
        slope = (pdf[:, cells + 1] - rho0) / width
        rho = rho0[..., None] + slope[..., None] * dx
        probability = (
            cdf[:, cells, None] + rho0[..., None] * dx + 0.5 * slope[..., None] * dx**2
        ) / cdf[:, -1, None, None]
        integration_weights = (
            rho
            * width[None, :, None]
            * jnp.asarray(weights / 2)
            / cdf[:, -1, None, None]
        )
        probability = probability.reshape(4, -1)
        integration_weights = integration_weights.reshape(4, -1)

        def evaluate(direction):
            def component(p, density, cumulative, density_dot, cumulative_dot):
                return jax.jvp(
                    lambda g, f, c: jax.vmap(
                        radial_quantile, in_axes=(0, None, None, None)
                    )(p, g, f, c),
                    (grid, density, cumulative),
                    (jac[0][..., direction], density_dot, cumulative_dot),
                )

            radii, dr = jax.vmap(component)(
                probability, pdf, cdf, jac[1][..., direction], jac[2][..., direction]
            )
            conditional = jnp.sum(integration_weights * 2 * radii * dr, axis=1)
            return conditional, jac[3][:, direction] @ moments + pi @ conditional

        (conditional, total), seconds = timed(
            f"Gauss order {order}", lambda: jax.jit(jax.vmap(evaluate))(jnp.arange(3))
        )
        report["levels"].append(
            {
                "order": order,
                "seconds": seconds,
                "component_derivatives": conditional.T.tolist(),
                "mixture_derivatives": total.tolist(),
                "relative_discrepancy": (
                    jnp.abs(total - reference) / jnp.abs(reference)
                ).tolist(),
                "integrated_probability": integration_weights.sum(axis=1).tolist(),
                "endpoint_rounded_probability_weight": jnp.sum(
                    jnp.where(
                        (probability <= 0) | (probability >= 1), integration_weights, 0
                    ),
                    axis=1,
                ).tolist(),
            }
        )
        save()
    report["status"] = "completed"
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
