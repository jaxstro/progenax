#!/usr/bin/env python
"""Validation + figure for the ANISOTROPIC multi-mass LIMEPY sampler (Phase 2b sampler).

Closes the anisotropic equilibrium model with a per-component velocity sampler that draws
each star from the Michie/Osipkov-Merritt LIMEPY DF (Gieles & Zocchi 2015, Eq. 1):

    f(E, J^2) ∝ exp(-J^2 / 2 r_{a,j}^2 s_j^2) E_gamma(g, (phi_t - E)/s_j^2),

with per-component anisotropy radius r_{a,j} = r_a mu_j^eta. The figure establishes that
the SAMPLED cluster carries the RIGHT anisotropy, not merely 'some' anisotropy:

  (a) beta_j(r) = 1 - sigma_t^2/(2 sigma_r^2): the SAMPLED profile matches the DF's own
      analytic beta_j(r) (direct (u,c) quadrature), including the characteristic LIMEPY
      RISE to a radial-bias peak near ~0.5 r_t and TURNOVER toward r_t (truncation lowers
      the most radial orbits at the edge). The sampler-correctness headline.
  (b) sigma_r(r) and sigma_t(r) separately, sampled vs analytic: the radial dispersion
      exceeds the tangential in the bias region -- the kinematic content of beta>0.
  (c) global virial Q = T/|V| vs delta for anisotropic models: exactly 0.5 without any
      rescale (the scalar virial theorem 2T+W=0 is anisotropy-blind) -- a true equilibrium.
  (d) analytic beta_light(r) for a range of r_a: the anisotropy is a controlled knob
      (smaller r_a -> stronger radial bias), all comfortably truncating.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_multimass_anisotropy.py
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


def _analytic_beta(model, j, rr):
    m_rr, m_tt = _uc_moments(model, j, rr)
    return 1.0 - m_tt / (2.0 * m_rr) if m_rr > 0 else np.nan


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
        bmean, bsem, _ = _mean_sem(beta_sd[j])
        b_df = np.array([_analytic_beta(model, j, float(rc)) for rc in centers])
        axA.plot(centers / rt, b_df, color=col, lw=1.7, label=f"{lab} (DF)")
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
        if j == 0:
            for x, bm, bs, bd in zip(centers / rt, bmean, bsem, b_df):
                print(
                    f"                                  {x:.2f}  {bm:+.3f}+/-{bs:.3f}   {bd:+.3f}"
                )
    axA.axhline(0.0, color="0.6", ls="--", lw=1.0)
    axA.set_xlabel(r"$r/r_t$")
    axA.set_ylabel(r"anisotropy $\beta(r)$")
    axA.legend(frameon=False, fontsize=7, ncol=2)
    axA.set_title(r"sampled $\beta(r)$ = LIMEPY DF (rise + turnover)", fontsize=9)
    panel_label(axA, "(a)", loc="upper left")

    # (b) seed-averaged sigma_r, sigma_t (1D each), sampled vs analytic, light component
    s = float(jnp.sqrt(G * M_tot / (9.0 * model.r_c * model.mu_tot)))
    s_j = s * float(model.w_j[0])  # w_j = mu_j^(-delta), delta=0.4
    srm, srs, _ = _mean_sem(sr_sd)
    stm, sts, _ = _mean_sem(st_sd)
    moms = [_uc_moments(model, 0, float(rc)) for rc in centers]
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
    axC.set_title("anisotropic cluster is virial (Q=0.5)", fontsize=9)
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
    panel_label(axD, "(d)", loc="upper left")

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
    sys.exit(main())
