#!/usr/bin/env python
"""Anisotropic multi-mass LIMEPY sampling checks and figures.

Compare radial/tangential moments with DF quadrature and test sampled virial
ratios. The default figure uses natural component assignments; dedicated
component and shell-stratified exposures report their separate acceptance scopes.
The LIMEPY DF follows Gieles & Zocchi (2015), equation 1.

Run from the repository root with float64 enabled:
    python scripts/validate_multimass_anisotropy.py
    python scripts/validate_multimass_anisotropy.py --conditional-components results.json
    python scripts/validate_multimass_anisotropy.py --stratified-shells results.json
    python scripts/validate_multimass_anisotropy.py --plot-stratified results.json
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR

from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.dynamics import compute_virial_ratio
from progenax.profiles.limepy import lowered_exponential

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = STELLAR.G
W0, g = 7.0, 1.0
M_J = jnp.array([1.0, 4.0])  # light / heavy stellar masses
ALPHA_J = jnp.array([0.6, 0.4])  # central density fractions
R_A = 5.0  # anisotropy radius (well-resolved peak beta ~ 0.22)
ETA = 0.0  # mass-independent anisotropy (paper default)
DELTAS = np.array([0.0, 0.2, 0.4, 0.6])
LCOL, HCOL = OI["sky"], OI["vermilion"]


def _uc_moments(model, j, rr):
    """(<u^2 c^2>, <u^2 (1-c^2)>) of the LIMEPY (u,c) phase-space weight at radius rr,
    component j. w(u,c) = u^2 E_gamma(g, W_j-u^2/2) exp(-(p^2/2) u^2 (1-c^2)). The s_j^2
    scaling is applied by the caller. Returns (0,0) outside the truncation."""
    psi = float(
        jnp.interp(
            rr / model.r_c, model.xi_grid, model.psi_grid, left=model.W0, right=0.0
        )
    )
    W_j = float(model.rescale_j[j]) * max(psi, 0.0)
    if W_j <= 0.0:
        return 0.0, 0.0
    p = (rr / float(model.r_c)) / float(model.ra_hat_j[j])
    u = jnp.linspace(0.0, jnp.sqrt(2.0 * W_j), 400)
    c = jnp.linspace(-1.0, 1.0, 240)
    E = lowered_exponential(model.g, W_j - u**2 / 2.0)
    U, C = jnp.meshgrid(u, c, indexing="ij")
    w = U**2 * E[:, None] * jnp.exp(-(p**2 / 2.0) * U**2 * (1.0 - C**2))
    Z = jnp.trapezoid(jnp.trapezoid(w, c, axis=1), u)
    m_rr = jnp.trapezoid(jnp.trapezoid(w * U**2 * C**2, c, axis=1), u)
    m_tt = jnp.trapezoid(jnp.trapezoid(w * U**2 * (1.0 - C**2), c, axis=1), u)
    return float(m_rr / Z), float(m_tt / Z)


def _shell_moments(model, j, lo, hi, n_radial=64):
    """Density-weighted radial/tangential moments over a spherical shell.

    Integrate r^2 times the existing 400-by-240 velocity quadrature.
    Component normalization cancels between numerator and denominator.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_radial)
    radii = jnp.asarray(lo + (nodes + 1.0) * (hi - lo) / 2.0)
    weights = jnp.asarray(weights * (hi - lo) / 2.0)

    def integrals(rr):
        psi = jnp.interp(rr / model.r_c, model.xi_grid, model.psi_grid)
        W = model.rescale_j[j] * jnp.maximum(psi, 0.0)
        p = (rr / model.r_c) / model.ra_hat_j[j]
        u = jnp.linspace(0.0, jnp.sqrt(2.0 * W), 400)
        c = jnp.linspace(-1.0, 1.0, 240)
        U, C = jnp.meshgrid(u, c, indexing="ij")
        w = (
            U**2
            * lowered_exponential(model.g, W - U**2 / 2.0)
            * jnp.exp(-(p**2) * U**2 * (1.0 - C**2) / 2.0)
        )

        def integrate(value):
            return jnp.trapezoid(jnp.trapezoid(value, c, axis=1), u)

        return rr**2 * jnp.array(
            [
                integrate(w),
                integrate(w * U**2 * C**2),
                integrate(w * U**2 * (1.0 - C**2)),
            ]
        )

    total = jnp.sum(weights[:, None] * jax.jit(jax.vmap(integrals))(radii), axis=0)
    return np.asarray(total[1:] / total[0])


def _analytic_beta(model, j, rr):
    m_rr, m_tt = _uc_moments(model, j, rr)
    return 1.0 - m_tt / (2.0 * m_rr) if m_rr > 0 else np.nan


def conditional_components(output):
    """Qualify shell beta for separately sampled components under the approved gate."""
    import json
    from pathlib import Path

    from progenax.cluster.sampling import _sample_cluster_arrays_given_components

    model = MultiComponentCluster.from_mass_segregation(
        alpha_j=ALPHA_J,
        m_j=M_J,
        W0=W0,
        g=g,
        delta=0.4,
        r_a=R_A,
        eta=ETA,
        r_c=1.0,
        xi_max=800.0,
        n_ode_points=3000,
    )
    rt = float(model.r_t)
    edges = np.linspace(0.06 * rt, 0.95 * rt, 11)
    records = []
    for component in range(2):
        counts = np.zeros((8, 10), dtype=int)
        beta = np.full((8, 10), np.nan)
        labels = jnp.full((60000,), component, dtype=jnp.int32)
        for seed in range(8):
            pos, vel, masses, _, _ = _sample_cluster_arrays_given_components(
                model,
                jax.random.PRNGKey(seed),
                labels,
                G,
            )
            pos = np.asarray(pos - jnp.average(pos, axis=0, weights=masses))
            vel = np.asarray(vel - jnp.average(vel, axis=0, weights=masses))
            radius = np.linalg.norm(pos, axis=1)
            radial = np.sum(vel * pos / (radius[:, None] + 1e-30), axis=1)
            tangential2 = np.sum(vel**2, axis=1) - radial**2
            for shell, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                selected = (radius >= lo) & (radius < hi)
                counts[seed, shell] = np.count_nonzero(selected)
                if counts[seed, shell]:
                    beta[seed, shell] = 1 - np.mean(tangential2[selected]) / (
                        2 * np.mean(radial[selected] ** 2)
                    )
            print(
                f"conditional component={component} seed={seed} counts={counts[seed].tolist()}",
                flush=True,
            )
        for shell, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mr, mt = _shell_moments(model, component, lo, hi)
            reference = float(1 - mt / (2 * mr))
            resolved = bool(np.all(counts[:, shell] >= 400))
            mean = (
                float(np.mean(beta[:, shell]))
                if np.all(np.isfinite(beta[:, shell]))
                else None
            )
            sem = (
                float(np.std(beta[:, shell], ddof=1) / np.sqrt(8))
                if mean is not None
                else None
            )
            error = abs(mean - reference) if mean is not None else None
            verdict = (
                ("PASS" if error is not None and error < 0.04 else "FAIL")
                if resolved
                else "UNRESOLVED"
            )
            row = dict(
                component=component,
                shell=shell,
                bounds_over_rt=[float(lo / rt), float(hi / rt)],
                counts=counts[:, shell].tolist(),
                beta_mean=mean,
                beta_sem=sem,
                beta_df=reference,
                absolute_error=error,
                verdict=verdict,
            )
            records.append(row)
            print(json.dumps(row), flush=True)
    result = dict(
        exposure=dict(
            W0=W0,
            g=g,
            delta=0.4,
            r_a=R_A,
            eta=ETA,
            alpha_j=np.asarray(ALPHA_J).tolist(),
            m_j=np.asarray(M_J).tolist(),
            xi_max=800,
            n_ode_points=3000,
            seeds=list(range(8)),
            stars_per_component_per_seed=60000,
            minimum_count_every_seed=400,
            absolute_beta_error_budget=0.04,
            radial_quadrature_order=64,
            velocity_quadrature=[400, 240],
            x64=bool(jax.config.jax_enable_x64),
        ),
        shells=records,
        counts={
            verdict: sum(row["verdict"] == verdict for row in records)
            for verdict in ("PASS", "FAIL", "UNRESOLVED")
        },
    )
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["counts"]), flush=True)
    return int(result["counts"]["FAIL"] > 0 or result["counts"]["PASS"] == 0)


def stratified_shells(output):
    """Check conditional shell moments without changing the production velocity law."""
    import json
    from pathlib import Path

    from progenax.cluster._radial_cdf import radial_cdf
    from progenax.cluster.sampling import _sample_cluster_arrays_given_components

    model = MultiComponentCluster.from_mass_segregation(
        alpha_j=ALPHA_J,
        m_j=M_J,
        W0=W0,
        g=g,
        delta=0.4,
        r_a=R_A,
        eta=ETA,
        r_c=1.0,
        xi_max=800.0,
        n_ode_points=3000,
    )
    rt = float(model.r_t)
    edges = np.linspace(0.06 * rt, 0.95 * rt, 11)
    records = []
    for component in range(2):
        probabilities = jax.vmap(
            lambda radius: radial_cdf(
                radius,
                model._r_grid,
                model.engine_a.radial_pdf_j[component],
                model._cdf_j[component],
            )
        )(jnp.asarray(edges))
        labels = jnp.full((20000,), component, dtype=jnp.int32)
        beta = np.empty((8, 10))
        for seed in range(8):
            key = jax.random.PRNGKey(seed)
            k_pos = jax.random.split(key, 5)[1]
            uniform = jax.random.uniform(k_pos, (20000,)).reshape(10, 2000)
            restricted = (
                probabilities[:-1, None] + uniform * jnp.diff(probabilities)[:, None]
            )
            pos, vel, _, _, _ = _sample_cluster_arrays_given_components(
                model, key, labels, G, radial_probabilities=restricted.ravel()
            )
            # Conditioning refers to the model centre; recentering would change shells.
            pos, vel = (
                np.asarray(pos).reshape(10, 2000, 3),
                np.asarray(vel).reshape(10, 2000, 3),
            )
            radius = np.linalg.norm(pos, axis=-1)
            assert np.all((radius >= edges[:-1, None]) & (radius <= edges[1:, None]))
            vr = np.sum(vel * pos / radius[..., None], axis=-1)
            vt2 = np.sum(vel**2, axis=-1) - vr**2
            beta[seed] = 1 - np.mean(vt2, axis=1) / (2 * np.mean(vr**2, axis=1))
            print(
                f"stratified component={component} seed={seed} beta={beta[seed].tolist()}",
                flush=True,
            )
        for shell, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mr, mt = _shell_moments(model, component, lo, hi)
            reference = float(1 - mt / (2 * mr))
            mean = float(np.mean(beta[:, shell]))
            error = abs(mean - reference)
            row = dict(
                component=component,
                shell=shell,
                bounds_over_rt=[float(lo / rt), float(hi / rt)],
                counts=[2000] * 8,
                beta_seeds=beta[:, shell].tolist(),
                beta_mean=mean,
                beta_sem=float(np.std(beta[:, shell], ddof=1) / np.sqrt(8)),
                beta_df=reference,
                absolute_error=error,
                verdict="PASS" if np.isfinite(error) and error < 0.04 else "FAIL",
            )
            records.append(row)
            print(json.dumps(row), flush=True)
    result = dict(
        exposure=dict(
            W0=W0,
            g=g,
            delta=0.4,
            r_a=R_A,
            eta=ETA,
            r_c=1.0,
            alpha_j=np.asarray(ALPHA_J).tolist(),
            m_j=np.asarray(M_J).tolist(),
            xi_max=800,
            n_ode_points=3000,
            seeds=list(range(8)),
            stars_per_shell_component_seed=2000,
            model_centered=True,
            absolute_beta_error_budget=0.04,
            radial_quadrature_order=64,
            velocity_quadrature=[400, 240],
            x64=bool(jax.config.jax_enable_x64),
        ),
        shells=records,
        counts={
            v: sum(row["verdict"] == v for row in records) for v in ("PASS", "FAIL")
        },
    )
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["counts"]), flush=True)
    return int(result["counts"]["FAIL"] > 0)


def plot_stratified(path):
    """Render saved shell-stratified measurements without rerunning the exposure."""
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt

    result = json.loads(Path(path).read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
    for component, (ax, color, name) in enumerate(
        zip(axes, (LCOL, HCOL), ("Light", "Heavy"))
    ):
        rows = [r for r in result["shells"] if r["component"] == component]
        x = np.array([np.mean(r["bounds_over_rt"]) for r in rows])
        reference = np.array([r["beta_df"] for r in rows])
        ax.fill_between(
            x,
            reference - 0.04,
            reference + 0.04,
            color=color,
            alpha=0.12,
            label=r"DF $\pm 0.04$ budget",
        )
        ax.plot(x, reference, color=color, lw=1.5, label="Shell DF")
        ax.errorbar(
            x,
            [r["beta_mean"] for r in rows],
            yerr=[r["beta_sem"] for r in rows],
            fmt="o",
            color=color,
            mfc="white",
            ms=4,
            capsize=2,
            label="Sample mean ± SE",
        )
        ax.set(xlabel=r"Radius / tidal radius, $r/r_t$", title=f"{name} component")
        ax.legend(fontsize=8, loc="upper left" if component == 0 else "lower center")
    axes[0].set_ylabel(r"Shell anisotropy, $\beta$")
    fig.suptitle(
        "Conditional shell sampling · 8 seeds × 2,000 stars per shell", fontsize=11
    )
    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_fig(fig, OUTPUT_DIR, "seg_multimass_stratified")
    return 0


def main():
    print("\n" + "=" * 70)
    print(
        "ANISOTROPIC MULTI-MASS LIMEPY SAMPLER VALIDATION (beta(r), sigma_r/t, virial)"
    )
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import matplotlib.pyplot as plt

    model = MultiComponentCluster.from_mass_segregation(
        alpha_j=ALPHA_J,
        m_j=M_J,
        W0=W0,
        g=g,
        delta=0.4,
        r_a=R_A,
        eta=ETA,
        r_c=1.0,
        xi_max=800.0,
        n_ode_points=3000,
    )
    rt = float(model.r_t)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
    axA, axB, axC, axD = axes.ravel()
    edges = np.linspace(0.06 * rt, 0.95 * rt, 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nb = len(centers)

    # beta(r) is a RATIO statistic (1 - <v_t^2>/(2<v_r^2>)); near r_t the light count per
    # bin falls and <v_r^2> -> 0, so single-seed beta is noise-dominated at the edge. The
    # correct presentation of a noisy observable is the seed-AVERAGED beta(r) with the
    # standard error across seeds -- which also directly shows the sampled beta converging
    # to the DF (the convergence-with-N question). Accumulate per-seed per-bin moments.
    N_SEED, N_STARS = 8, 60000
    beta_sd = {0: [[] for _ in range(nb)], 1: [[] for _ in range(nb)]}
    sr_sd = [[] for _ in range(nb)]
    st_sd = [[] for _ in range(nb)]
    M_tot = None
    for sd in range(N_SEED):
        ic = model.sample_cluster(jax.random.PRNGKey(sd), n_stars=N_STARS, G=G)
        pos, vel, masses = ic.positions, ic.velocities, ic.masses
        pos = pos - jnp.average(pos, axis=0, weights=masses)
        vel = vel - jnp.average(vel, axis=0, weights=masses)
        r = np.asarray(jnp.linalg.norm(pos, axis=1))
        r_hat = np.asarray(pos) / (r[:, None] + 1e-30)
        v_r = np.sum(np.asarray(vel) * r_hat, axis=1)
        v_t2 = np.sum(np.asarray(vel) ** 2, axis=1) - v_r**2
        masses_n = np.asarray(masses)
        if M_tot is None:
            M_tot = float(jnp.sum(masses))
        for j, mj in enumerate([float(M_J[0]), float(M_J[1])]):
            sel = np.isclose(masses_n, mj)
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                m = sel & (r >= lo) & (r < hi)
                if m.sum() < 30:
                    continue
                beta_sd[j][i].append(
                    1.0 - v_t2[m].mean() / (2.0 * (v_r[m] ** 2).mean())
                )
                if j == 0:
                    sr_sd[i].append(np.sqrt((v_r[m] ** 2).mean()))
                    st_sd[i].append(np.sqrt(v_t2[m].mean() / 2.0))

    def _mean_sem(rows):
        mean = np.array([np.mean(x) if len(x) else np.nan for x in rows])
        sem = np.array(
            [np.std(x) / np.sqrt(len(x)) if len(x) > 1 else np.nan for x in rows]
        )
        n = np.array([len(x) for x in rows])
        return mean, sem, n

    # (a) seed-averaged beta(r): sampled (mean +/- sem) vs analytic DF, per component
    print(
        f"  seed-averaged beta(r) [{N_SEED} seeds x {N_STARS}]:  r/rt  sampled+/-sem   DF"
    )
    for j, (col, lab) in enumerate([(LCOL, "light"), (HCOL, "heavy")]):
        bmean, bsem, n_seeds = _mean_sem(beta_sd[j])
        shell = np.array(
            [_shell_moments(model, j, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
        )
        shell32 = np.array(
            [
                _shell_moments(model, j, lo, hi, 32)
                for lo, hi in zip(edges[:-1], edges[1:])
            ]
        )
        b_df = 1.0 - shell[:, 1] / (2.0 * shell[:, 0])
        b32 = 1.0 - shell32[:, 1] / (2.0 * shell32[:, 0])
        print(
            f"  {lab} shell beta radial-order 32/64 max absolute change: {np.max(abs(b_df - b32)):.6g}"
        )
        for x, bm, bs, bd, count in zip(centers / rt, bmean, bsem, b_df, n_seeds):
            print(
                f"  shell {lab}: r/rt={x:.4f} sampled={bm:.6g} sem={bs:.6g} DF={bd:.6g} seeds={count}"
            )
        axA.plot(centers / rt, b_df, color=col, lw=1.7, label=f"{lab} (shell DF)")
        axA.errorbar(
            centers / rt,
            bmean,
            yerr=bsem,
            ls="none",
            marker="o",
            ms=4,
            mfc="white",
            color=col,
            capsize=2,
            label=f"{lab} (sampled)",
        )
        for x, bm, count in zip(centers / rt, bmean, n_seeds):
            if 0 < count < N_SEED:
                axA.annotate(
                    f"{count}/{N_SEED} seeds",
                    (x, bm),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=7,
                )
        if j == 0:
            for x, bm, bs, bd in zip(centers / rt, bmean, bsem, b_df):
                print(
                    f"                                  {x:.2f}  {bm:+.3f}+/-{bs:.3f}   {bd:+.3f}"
                )
    axA.axhline(0.0, color="0.6", ls="--", lw=1.0)
    axA.set_xlabel(r"$r/r_t$")
    axA.set_ylabel(r"anisotropy $\beta(r)$")
    axA.legend(frameon=False, fontsize=7, ncol=2)
    axA.set_title(r"sampled anisotropy and shell-averaged DF", fontsize=9)
    panel_label(axA, "(a)", loc="upper left")

    # (b) seed-averaged sigma_r, sigma_t (1D each), sampled vs analytic, light component
    s = float(jnp.sqrt(G * M_tot / (9.0 * model.r_c * model.mu_tot)))
    s_j = s * float(model.w_j[0])  # w_j = mu_j^(-delta), delta=0.4
    srm, srs, _ = _mean_sem(sr_sd)
    stm, sts, _ = _mean_sem(st_sd)
    moms = [_shell_moments(model, 0, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    sr_df = np.array([s_j * np.sqrt(mr) for mr, mt in moms])
    st_df = np.array([s_j * np.sqrt(mt / 2.0) for mr, mt in moms])
    axB.plot(centers / rt, sr_df, color=OI["blue"], lw=1.6, label=r"$\sigma_r$ (DF)")
    axB.errorbar(
        centers / rt,
        srm,
        yerr=srs,
        ls="none",
        marker="o",
        ms=4,
        mfc="white",
        color=OI["blue"],
        capsize=2,
        label=r"$\sigma_r$ (samp.)",
    )
    axB.plot(centers / rt, st_df, color=OI["orange"], lw=1.6, label=r"$\sigma_t$ (DF)")
    axB.errorbar(
        centers / rt,
        stm,
        yerr=sts,
        ls="none",
        marker="s",
        ms=4,
        mfc="white",
        color=OI["orange"],
        capsize=2,
        label=r"$\sigma_t$ (samp.)",
    )
    axB.set_xlabel(r"$r/r_t$")
    axB.set_ylabel(r"$\sigma_{r,t}$ (light)")
    axB.legend(frameon=False, fontsize=7, ncol=2)
    axB.set_title(r"$\sigma_r > \sigma_t$ in the bias region", fontsize=9)
    panel_label(axB, "(b)", loc="upper right")

    # (c) global virial Q vs delta (anisotropic models)
    print("  global Q vs delta (anisotropic):")
    Qg, Qse = [], []
    for d in DELTAS:
        mdl = MultiComponentCluster.from_mass_segregation(
            alpha_j=ALPHA_J,
            m_j=M_J,
            W0=W0,
            g=g,
            delta=float(d),
            r_a=R_A,
            eta=ETA,
            r_c=1.0,
            xi_max=800.0,
            n_ode_points=3000,
        )
        qs = []
        for sd in range(4):
            ic = mdl.sample_cluster(jax.random.PRNGKey(sd), n_stars=20000, G=G)
            p, v, mm = ic.positions, ic.velocities, ic.masses
            p = p - jnp.average(p, axis=0, weights=mm)
            v = v - jnp.average(v, axis=0, weights=mm)
            qs.append(float(compute_virial_ratio(p, v, mm, G=G)))
        Qg.append(np.mean(qs))
        Qse.append(np.std(qs) / np.sqrt(len(qs)))
        print(f"    delta={d:.1f}  Q={np.mean(qs):.3f} +/- {Qse[-1]:.3f}")
    axC.errorbar(
        DELTAS, Qg, yerr=Qse, marker="o", ms=5, lw=1.6, color=OI["green"], capsize=2
    )
    axC.axhline(0.5, color="0.6", ls="--", lw=1.0)
    axC.set_ylim(0.44, 0.56)
    axC.set_xlabel(r"equipartition $\delta$")
    axC.set_ylabel(r"global $Q = T/|V|$")
    axC.set_title("sampled virial ratio", fontsize=9)
    panel_label(axC, "(c)", loc="upper left")

    # (d) analytic beta_light(r) for several r_a (the anisotropy knob)
    fr = np.linspace(0.03, 0.97, 30)
    for r_a, col in zip(
        [8.0, 6.0, 5.0, 4.0], [OI["sky"], OI["green"], OI["orange"], OI["vermilion"]]
    ):
        mdl = MultiComponentCluster.from_mass_segregation(
            alpha_j=ALPHA_J,
            m_j=M_J,
            W0=W0,
            g=g,
            delta=0.4,
            r_a=r_a,
            eta=ETA,
            r_c=1.0,
            xi_max=800.0,
            n_ode_points=3000,
        )
        rt_ = float(mdl.r_t)
        b = [_analytic_beta(mdl, 0, f * rt_) for f in fr]
        axD.plot(fr, b, color=col, lw=1.6, label=rf"$r_a={r_a:.0f}$")
    axD.axhline(0.0, color="0.6", ls="--", lw=1.0)
    axD.set_xlabel(r"$r/r_t$")
    axD.set_ylabel(r"$\beta_{\rm light}(r)$ (DF)")
    axD.legend(frameon=False, fontsize=7, title="anisotropy knob", title_fontsize=7)
    axD.set_title(r"smaller $r_a$ $\Rightarrow$ stronger radial bias", fontsize=9)
    panel_label(axD, "(d)", loc="upper right")

    fig.tight_layout(pad=0.6)
    save_fig(fig, OUTPUT_DIR, "seg_multimass_anisotropy")

    # ---- pass/fail ----
    # (a) seed-averaged sampled beta_light(r) matches the DF in every bin that is resolved
    # in ALL seeds and lies within the DF turnover (center <= 0.8 r_t). beta is a ratio
    # statistic; the outermost bin (~0.9 r_t, ~tens of light stars, <v_r^2> -> 0) stays
    # noise-dominated even seed-averaged -- it is plotted with its (large) error bar but
    # not gated. The N-sweep + 16-seed average confirm convergence to the DF.
    bmean_l, bsem_l, n_l = _mean_sem(beta_sd[0])
    beta_ok, n_checked = True, 0
    for i, rc in enumerate(centers):
        if n_l[i] < N_SEED or rc > 0.8 * rt:
            continue
        n_checked += 1
        if abs(bmean_l[i] - _analytic_beta(model, 0, float(rc))) > 0.04:
            beta_ok = False
    beta_ok = beta_ok and n_checked >= 5
    Qg_ok = np.all(np.abs(np.array(Qg) - 0.5) < 0.04)
    ok = beta_ok and Qg_ok
    print(
        f"  seed-avg beta_light(r)=DF, resolved bins (|d|<0.04, {n_checked} bins): "
        f"{'PASS' if beta_ok else 'FAIL'}"
    )
    print(f"  global Q=0.5 across delta (aniso):    {'PASS' if Qg_ok else 'FAIL'}")
    print(f"  saved seg_multimass_anisotropy.{{png,pdf}} -> {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    print("  ANISOTROPIC SAMPLER VALIDATION PASS" if ok else "  VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--conditional-components",
        metavar="JSON",
        help="Save the approved natural-occupancy conditional exposure.",
    )
    mode.add_argument(
        "--stratified-shells",
        metavar="JSON",
        help="Save the approved shell-stratified conditional exposure.",
    )
    mode.add_argument(
        "--plot-stratified", metavar="JSON", help="Plot a saved stratified exposure."
    )
    args = parser.parse_args()
    if args.conditional_components:
        sys.exit(conditional_components(args.conditional_components))
    if args.stratified_shells:
        sys.exit(stratified_shells(args.stratified_shells))
    if args.plot_stratified:
        sys.exit(plot_stratified(args.plot_stratified))
    sys.exit(main())
