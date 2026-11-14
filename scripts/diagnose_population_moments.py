"""Approved four-realization Engine A moment diagnostic, not a precision gate."""

import argparse
import json
import time
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
from demo_delta_recovery import _moment_ratio

from progenax.cluster._radial_cdf import potential_interp, radial_second_moments
from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.cluster.sampling import (
    _expectation_with_score,
    _sample_cluster_arrays,
    _sample_cluster_arrays_given_components,
)
from progenax.imf.smooth import Maschberger
from progenax.profiles.limepy import lowered_exponential

N = 100_000
R = 4
THETA = jnp.array([2.3, 0.4, 5.0])
KEY = jax.random.PRNGKey(0)
OUT = Path("docs/reviews/plummer-om-artifacts/population-moment-forward-pilot.json")


def build_model(
    theta,
    *,
    g=1.0,
    n_components=4,
    adjoint=diffrax.ForwardMode(),
    fit_tol=1e-6,
    n_iter=30,
):
    alpha, delta, W0 = theta
    return MultiComponentCluster.from_imf(
        Maschberger(alpha=alpha, m_min=0.1, m_max=20.0),
        n_components,
        W0,
        g=g,
        delta=delta,
        m_range=(0.1, 20.0),
        adjoint=adjoint,
        fit_tol=fit_tol,
        n_iter=n_iter,
    )


def sampled(theta):
    model = build_model(theta)
    logp = jax.nn.log_softmax(jnp.log(model.N_frac_j + 1e-30))

    def draw(key):
        pos, vel, mass, _, labels = _sample_cluster_arrays(model, key, N, 1.0)
        values = jnp.stack(
            (
                jnp.mean(mass),
                jnp.mean(jnp.sum(pos**2, axis=1)),
                jnp.mean(jnp.sum(vel**2, axis=1)),
            )
        )
        return values, jnp.sum(logp[labels])

    values, scores = jax.lax.map(draw, jax.random.split(KEY, R))
    corrected = jax.vmap(_expectation_with_score, in_axes=(1, None))(values, scores)
    return jnp.concatenate((corrected, jnp.mean(values, axis=0))), values


def conditional_sampled(theta, key=KEY, *, analytic_radius=False):
    """Stratified conditional integration of the same N-star population moments.

    N/4 draws per component are integration samples, not a physical catalogue.
    The finite-N velocity normalization is averaged analytically over all labels.
    analytic_radius=True also integrates radius squared exactly. False retains
    the original exposure for reproduction; neither changes velocity sampling.
    """
    model = build_model(theta)
    pi = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    labels = jnp.repeat(jnp.arange(4), N // 4)
    mean_mass = jnp.dot(pi, model.m_j)
    stratified_mass = (N // 4) * jnp.sum(model.m_j)
    correction = (model.m_j + (N - 1) * mean_mass) / stratified_mass
    if analytic_radius:
        exact_radius2 = pi @ radial_second_moments(
            model._r_grid, model.engine_a.radial_pdf_j
        )

    def draw(key):
        pos, vel, _, _, _ = _sample_cluster_arrays_given_components(
            model, key, labels, 1.0
        )
        radius2 = jnp.mean(jnp.sum(pos**2, axis=1).reshape(4, -1), axis=1)
        speed2 = jnp.mean(jnp.sum(vel**2, axis=1).reshape(4, -1), axis=1)
        mean_radius2 = exact_radius2 if analytic_radius else pi @ radius2
        return jnp.stack((mean_mass, mean_radius2, pi @ (speed2 * correction)))

    values = jax.lax.map(draw, jax.random.split(key, R))
    return jnp.mean(values, axis=0), values


def reference(
    theta,
    *,
    analytic=False,
    subdivisions=1,
    g=1.0,
    n_components=4,
    fit_tol=1e-6,
    n_iter=30,
):
    model = build_model(
        theta, g=g, n_components=n_components, fit_tol=fit_tol, n_iter=n_iter
    )
    pi = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    r = model._r_grid
    pdf = model.engine_a.radial_pdf_j
    norm = jnp.trapezoid(pdf, r, axis=1)
    if analytic:
        h = jnp.diff(r)
        left = r[:-1]
        # Exact integral of r^2 times the sampler's linear radial PDF in each cell.
        segment = h * (
            pdf[:, :-1] * (left**2 + left * h + h**2 / 3)
            + jnp.diff(pdf, axis=1) * (left**2 / 2 + 2 * left * h / 3 + h**2 / 4)
        )
        r2 = jnp.sum(segment, axis=1) / norm
    else:
        r2 = jnp.trapezoid(pdf * r**2, r, axis=1) / norm
    if subdivisions > 1:
        fractions = jnp.arange(subdivisions) / subdivisions
        fine_r = (r[:-1, None] + jnp.diff(r)[:, None] * fractions).reshape(-1)
        fine_pdf = (
            pdf[:, :-1, None] + jnp.diff(pdf, axis=1)[:, :, None] * fractions
        ).reshape(n_components, -1)
        r = jnp.concatenate((fine_r, r[-1:]))
        pdf = jnp.concatenate((fine_pdf, pdf[:, -1:]), axis=1)
    psi = potential_interp(
        r / model.r_c, model.xi_grid, model.engine_a.psi_raw, left=model.W0, right=0.0
    )
    W = model.rescale_j[:, None] * jnp.maximum(psi, 0.0)
    if analytic:
        safe_W = jnp.where(W > 0, W, 1.0)
        v2hat = jnp.where(
            W > 0,
            3
            * lowered_exponential(model.g + 2.5, safe_W)
            / lowered_exponential(model.g + 1.5, safe_W),
            0.0,
        )
    else:
        v2hat = 3 * jax.vmap(jax.vmap(lambda w: _moment_ratio(w, model.g)))(W)
    q = model.w_j**2 * jnp.trapezoid(pdf * v2hat, r, axis=1) / norm
    mean_mass = jnp.dot(pi, model.m_j)
    speed = jnp.sum(pi * q * (model.m_j + (N - 1) * mean_mass)) / (
        9 * model.r_c * model.mu_tot
    )
    return jnp.stack((mean_mass, jnp.dot(pi, r2), speed))


def timed(label, fn):
    start = time.perf_counter()
    result = jax.block_until_ready(fn())
    elapsed = time.perf_counter() - start
    print(label, elapsed, flush=True)
    return result, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditional", action="store_true")
    args = parser.parse_args()
    if args.conditional:
        OUT = Path(
            "docs/reviews/plummer-om-artifacts/population-moment-conditional-pilot.json"
        )
        sampled = conditional_sampled
    assert jax.config.jax_enable_x64
    report = dict(
        n_stars=N,
        n_realizations=R,
        seed=0,
        theta=THETA.tolist(),
        observables=["mean_mass", "mean_radius_squared", "mean_speed_squared"],
        parameters=["alpha", "delta", "W0"],
        x64=True,
        limitations=[
            "Four realizations cannot establish bias or precision.",
            "Moment quadrature uses production radial density; independent of random inverse sampling, not of the equilibrium solver.",
            "Reference quadrature convergence remains open. No statistical gate.",
        ],
    )
    ref = jax.jit(reference)
    rv, report["reference_cold_seconds"] = timed("reference", lambda: ref(THETA))
    report["reference_values"] = rv.tolist()
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    deriv = jax.jit(jax.jacfwd(reference))
    rd, report["reference_ad_seconds"] = timed("reference AD", lambda: deriv(THETA))
    report["reference_derivatives"] = rd.tolist()
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    def directional(theta, tangent):
        def with_aux(t):
            means, values = sampled(t)
            return means, values

        return jax.jvp(with_aux, (theta,), (tangent,), has_aux=True)

    evaluate = jax.jit(directional)
    columns = []
    elapsed_columns = []
    for index in range(3):
        (means, column, values), elapsed = timed(
            f"sample JVP {index}", lambda: evaluate(THETA, jnp.eye(3)[index])
        )
        columns.append(column)
        elapsed_columns.append(elapsed)
    sd = jnp.stack(columns, axis=1)
    report["sample_direction_seconds"] = elapsed_columns
    report["ad_mode"] = (
        "ForwardMode ODE adjoint, sequential realizations and parameter JVPs"
    )
    if args.conditional:
        sd = jnp.concatenate((sd, sd), axis=0)
        report["estimator"] = (
            "conditional component averaging; 25000 integration samples per component per realization"
        )
    report.update(
        realization_values=values.tolist(),
        sample_derivatives=sd[:3].tolist(),
        pathwise_derivatives=sd[3:].tolist(),
        score_derivatives=(sd[:3] - sd[3:]).tolist(),
        sample_mean=values.mean(0).tolist(),
        realization_sd=values.std(0, ddof=1).tolist(),
    )
    for h in (1e-4, 1e-5):
        fd = jnp.stack(
            [(ref(THETA + h * e) - ref(THETA - h * e)) / (2 * h) for e in jnp.eye(3)],
            axis=1,
        )
        report[f"reference_fd_{h}"] = fd.tolist()
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
