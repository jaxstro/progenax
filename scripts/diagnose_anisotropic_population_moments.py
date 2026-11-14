"""Approved anisotropic Engine A reference gate and four-realization pilot."""

import hashlib
import json
import subprocess
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from diagnose_population_moments import timed

from progenax.cluster._radial_cdf import potential_interp
from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.cluster.sampling import _estimate_population_moments
from progenax.imf.smooth import Maschberger
from progenax.profiles.limepy_multimass import _aniso_v2hat_scalar

OUT = Path("docs/reviews/plummer-om-artifacts/population-moment-anisotropic-mapped-pilot.json")
THETA = jnp.array([2.3, 0.4, 5.0, 10.0])


def model(t, *, adjoint=diffrax.ForwardMode()):
    alpha, delta, w0, ra = t
    return MultiComponentCluster.from_imf(
        Maschberger(alpha=alpha, m_min=0.1, m_max=20.0),
        4,
        w0,
        g=1.0,
        delta=delta,
        m_range=(0.1, 20.0),
        r_a=ra,
        eta=0.0,
        fit_tol=1e-10,
        n_iter=60,
        adjoint=adjoint,
    )


def reference(t, subdivisions, *, batch_size=32):
    m = model(t)
    r = m._r_grid
    pdf = m.engine_a.radial_pdf_j
    norm = jnp.trapezoid(pdf, r, axis=1)
    if subdivisions > 1:
        f = jnp.arange(subdivisions) / subdivisions
        rr = (r[:-1, None] + jnp.diff(r)[:, None] * f).reshape(-1)
        pp = (pdf[:, :-1, None] + jnp.diff(pdf, axis=1)[:, :, None] * f).reshape(4, -1)
        r = jnp.concatenate((rr, r[-1:]))
        pdf = jnp.concatenate((pp, pdf[:, -1:]), axis=1)
    psi = potential_interp(
        r / m.r_c, m.xi_grid, m.engine_a.psi_raw, left=m.W0, right=0.0
    )
    W = m.rescale_j[:, None] * jnp.maximum(psi, 0.0)
    p = r[None, :] / m.r_c / m.ra_hat_j[:, None]
    v2 = jax.lax.map(
        lambda wp: _aniso_v2hat_scalar(wp[0], wp[1], m.g),
        (W.reshape(-1), p.reshape(-1)),
        batch_size=batch_size,
    ).reshape(W.shape)
    q = m.w_j**2 * jnp.trapezoid(pdf * v2, r, axis=1) / norm
    pi = jax.nn.softmax(jnp.log(m.N_frac_j + 1e-30))
    mean_mass = pi @ m.m_j
    return jnp.sum(pi * q * (m.m_j + 99999 * mean_mass)) / (
        9 * m.r_c * m.mu_tot
    ), m.residual


def main():
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    record = dict(
        status="references running",
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol=dict(
            theta=THETA.tolist(),
            g=1.0,
            eta=0.0,
            fit_tol=1e-10,
            n_iter=60,
            target_stars=100000,
            draws_per_component=25000,
            realizations=4,
            seed=0,
            subdivisions=[1, 2, 4],
            fd_steps=[1e-4, 1e-5],
            relative_gate=0.003,
            reference="existing direct anisotropic speed quadrature; independent of speed-CDF table; shared density/ODE model",
            scope="diagnostic pilot; no stochastic precision verdict",
        ),
        references=[],
    )

    def save():
        OUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    save()
    for level in [1, 2, 4]:
        fn = lambda t: reference(t, level)[0]
        evaluate = jax.jit(lambda t, e: jax.jvp(fn, (t,), (e,)))
        columns, durations = [], []
        for direction in jnp.eye(4):
            (value, column), seconds = timed(
                f"reference {level} direction", lambda: evaluate(THETA, direction)
            )
            columns.append(column)
            durations.append(seconds)
        derivative = jnp.stack(columns)
        residual = jax.jit(lambda t: model(t).residual)(THETA)
        seconds = sum(durations)
        record["references"].append(
            dict(
                value=float(value),
                residual=float(residual),
                derivative=derivative.tolist(),
                seconds=seconds,
            )
        )
        save()
        jax.clear_caches()
    fn = jax.jit(lambda t: reference(t, 4)[0])
    errors = [
        abs(np.array(record["references"][-2]["derivative"]) - np.array(derivative))
    ]
    for h in [1e-4, 1e-5]:
        fd = jnp.stack(
            [(fn(THETA + h * e) - fn(THETA - h * e)) / (2 * h) for e in jnp.eye(4)]
        )
        record[f"fd_{h}"] = fd.tolist()
        errors.append(abs(np.array(fd) - np.array(derivative)))
        save()
    error = np.maximum.reduce(errors)
    record["relative_reference_error"] = (error / abs(np.array(derivative))).tolist()
    record["reference_pass"] = bool(np.all(error <= 0.003 * abs(np.array(derivative))))
    record["fit_pass"] = all(x["residual"] <= 1e-10 for x in record["references"])
    if not (record["reference_pass"] and record["fit_pass"]):
        record["status"] = "reference gate failed; sampling not started"
        save()
        return
    jax.clear_caches()

    def sample(t, key):
        return _estimate_population_moments(model(t), key, 100000, 1.0, 1, 25000)[
            "mean_speed_squared"
        ]

    evaluate = jax.jit(lambda t, e, k: jax.jvp(lambda tt: sample(tt, k), (t,), (e,)))
    record["pilot"] = []
    record["status"] = "pilot running"
    for key in jax.random.split(jax.random.PRNGKey(0), 4):
        columns = []
        times = []
        for e in jnp.eye(4):
            (value, d), seconds = timed(
                "pilot direction", lambda: evaluate(THETA, e, key)
            )
            columns.append(float(d))
            times.append(seconds)
        record["pilot"].append(
            dict(value=float(value), derivative=columns, seconds=times)
        )
        save()
    record["status"] = "pilot completed; precision unqualified"
    save()


if __name__ == "__main__":
    main()
