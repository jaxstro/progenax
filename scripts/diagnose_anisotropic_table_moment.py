"""Exact uniform integral of the implemented bilinear inverse speed table."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from diagnose_anisotropic_population_moments import THETA, model

from progenax.cluster._radial_cdf import potential_interp
from progenax.profiles.limepy_tables import AnisoSpeedCDFTable

OUT = Path("docs/reviews/plummer-om-artifacts/anisotropic-table-moment.json")


def exact_moment(tab, W, p):
    safe = jnp.where(W > 0, W, 1.0)
    s = jnp.clip(jnp.sqrt(safe), tab.s_nodes[0], tab.s_nodes[-1])
    q = jnp.clip(jnp.log1p(p * p * safe), tab.q_nodes[0], tab.q_nodes[-1])
    i = jnp.clip(jnp.searchsorted(tab.s_nodes, s) - 1, 0, tab.s_nodes.size - 2)
    j = jnp.clip(jnp.searchsorted(tab.q_nodes, q) - 1, 0, tab.q_nodes.size - 2)
    ts = jnp.clip((s - tab.s_nodes[i]) / (tab.s_nodes[1] - tab.s_nodes[0]), 0.0, 1.0)
    tq = jnp.clip((q - tab.q_nodes[j]) / (tab.q_nodes[1] - tab.q_nodes[0]), 0.0, 1.0)
    rows = jax.lax.dynamic_slice(
        tab.cdf, (i, j, jnp.asarray(0, i.dtype)), (2, 2, tab.x_nodes.size)
    ).reshape(4, -1)
    knots = jnp.sort(rows.reshape(-1))
    x4 = jax.vmap(lambda c: jnp.interp(knots, c, tab.x_nodes))(rows)
    weights = jnp.array([(1 - ts) * (1 - tq), (1 - ts) * tq, ts * (1 - tq), ts * tq])
    x = weights @ x4
    integral = jnp.sum(
        jnp.diff(knots) * (x[:-1] ** 2 + x[:-1] * x[1:] + x[1:] ** 2) / 3
    )
    return jnp.where(W > 0, 2 * safe * integral, 0.0)


def reference(t, level, *, n_W=192, n_p=48, n_x=384):
    m = model(t)
    r = m._r_grid
    pdf = m.engine_a.radial_pdf_j
    norm = jnp.trapezoid(pdf, r, axis=1)
    if level > 1:
        f = jnp.arange(level) / level
        rr = (r[:-1, None] + jnp.diff(r)[:, None] * f).reshape(-1)
        pp = (pdf[:, :-1, None] + jnp.diff(pdf, axis=1)[:, :, None] * f).reshape(4, -1)
        r = jnp.concatenate((rr, r[-1:]))
        pdf = jnp.concatenate((pp, pdf[:, -1:]), axis=1)
    tab = AnisoSpeedCDFTable.build(
        jnp.max(m.rescale_j) * m.W0,
        jnp.maximum((m.r_t / m.r_c) / jnp.min(m.ra_hat_j), 1e-3),
        m.g,
        n_W=n_W,
        n_p=n_p,
        n_x=n_x,
    )
    psi = potential_interp(
        r / m.r_c, m.xi_grid, m.engine_a.psi_raw, left=m.W0, right=0.0
    )
    W = m.rescale_j[:, None] * jnp.maximum(psi, 0.0)
    p = r[None, :] / m.r_c / m.ra_hat_j[:, None]
    v2 = jax.lax.map(
        lambda wp: exact_moment(tab, *wp), (W.reshape(-1), p.reshape(-1)), batch_size=32
    ).reshape(W.shape)
    q = m.w_j**2 * jnp.trapezoid(pdf * v2, r, axis=1) / norm
    pi = jax.nn.softmax(jnp.log(m.N_frac_j + 1e-30))
    mm = pi @ m.m_j
    return jnp.sum(pi * q * (m.m_j + 99999 * mm)) / (9 * m.r_c * m.mu_tot)


def main():
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    report = dict(
        scope="exact piecewise-linear inverse-CDF speed moment; retained radial integration; ra derivative only",
        theta=THETA.tolist(),
        levels=[],
    )
    for level in [1, 2, 4]:
        f = jax.jit(
            lambda t: jax.jvp(
                lambda x: reference(x, level), (t,), (jnp.array([0.0, 0.0, 0.0, 1.0]),)
            )
        )
        start = time.perf_counter()
        v, d = jax.block_until_ready(f(THETA))
        report["levels"].append(
            dict(
                level=level,
                value=float(v),
                ra_derivative=float(d),
                seconds=time.perf_counter() - start,
            )
        )
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(report["levels"][-1], flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
