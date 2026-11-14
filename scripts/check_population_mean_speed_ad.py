"""Full IMF construction AD qualification; run from the repository root."""

# ruff: noqa: E402
import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import diffrax
import jax.numpy as jnp
from jaxstro.units import STELLAR

from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.imf import PowerLawIMF

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--risk-matrix", action="store_true")
args = parser.parse_args()
base = dict(W0=5.0, g=1.0, delta=0.4, ra=10.0, n_comp=2)
cases = (
    [
        dict(base, **change)
        for change in [
            {},
            {"W0": 7.0},
            {"g": 0.0},
            {"g": 0.5},
            {"delta": 0.0},
            {"delta": 0.6},
            {"ra": 4.0},
            {"ra": None},
            {"n_comp": 4},
        ]
    ]
    if args.risk_matrix
    else [dict(base, ra=None), base]
)
print(
    json.dumps(
        {
            "cases": cases,
            "x64": jax.config.jax_enable_x64,
            "derivative_criterion": 0.003,
            "fit_criterion": 1e-6,
        }
    ),
    flush=True,
)
for config in cases:
    ra = config["ra"]
    print(json.dumps({"configuration": config}), flush=True)

    def values(theta, adjoint=None):
        imf = PowerLawIMF([theta[0]], [], m_min=0.1, m_max=20.0)
        m = MultiComponentCluster.from_imf(
            imf,
            n_comp=config["n_comp"],
            W0=config["W0"],
            g=config["g"],
            delta=theta[1],
            m_range=(0.1, 20.0),
            r_a=ra,
            n_iter=60,
            xi_max=800.0,
            n_ode_points=1500,
            n_grid=600,
            adjoint=adjoint,
        )
        keys = jax.random.split(jax.random.PRNGKey(0), 3)

        def sample(key):
            ic = m.sample_cluster(key, 400, G=STELLAR.G)
            return jnp.mean(
                jnp.sqrt(jnp.sum(ic.velocities**2, axis=1) + 1e-30)
            ), ic.component_id

        speeds, labels = jax.vmap(sample)(keys)
        return speeds, m.residual, labels

    theta = jnp.array([2.3, config["delta"]])
    f = jax.jit(values)
    v, res, labels = f(theta)
    print(
        json.dumps({"ra": ra, "value": v.tolist(), "fit_residual": float(res)}),
        flush=True,
    )
    jac = []
    for name, mode, adj in [
        ("forward", jax.jacfwd, diffrax.ForwardMode()),
        ("reverse", jax.jacrev, None),
    ]:
        t = time.perf_counter()
        a = jax.jit(mode(lambda t: values(t, adj)[0]))(theta)
        a.block_until_ready()
        jac.append(a)
        print(
            json.dumps(
                {
                    "ra": ra,
                    "mode": name,
                    "jacobian": a.tolist(),
                    "cold_s": time.perf_counter() - t,
                }
            ),
            flush=True,
        )
    for h in [1e-4, 1e-5]:
        columns, flips, residuals = [], [], []
        for i in range(2):
            shift = jnp.eye(2)[i] * h
            lower = theta if i == 1 and config["delta"] == 0 else theta - shift
            width = h if i == 1 and config["delta"] == 0 else 2 * h
            plus, minus = f(theta + shift), f(lower)
            columns.append((plus[0] - minus[0]) / width)
            flips.append(jnp.sum(plus[2] != minus[2], axis=1).tolist())
            residuals.append([float(plus[1]), float(minus[1])])
        fd = jnp.stack(columns, axis=1)
        print(
            json.dumps(
                {
                    "ra": ra,
                    "h": h,
                    "component_flips": flips,
                    "perturbed_residuals": residuals,
                    "fd": fd.tolist(),
                    "forward_relative_error": (jac[0] / fd - 1).tolist(),
                    "reverse_relative_error": (jac[1] / fd - 1).tolist(),
                }
            ),
            flush=True,
        )
    jax.clear_caches()
