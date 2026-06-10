#!/usr/bin/env python
"""Validation + figure for Engine B: prescribed-density shared-Psi Eddington equilibria.

Engine B (`MultiComponentCluster.from_density_profiles`) builds Plummer/EFF/King
density components into ONE shared self-consistent potential (single quadrature
pass, no ODE), Eddington-inverts each component's DF in that shared Psi
(optionally Osipkov-Merritt per component), and samples a true equilibrium with
NO external virial rescale. This script re-runs the Task 5/6 physics anchors as
a standalone PASS/FAIL gate + one 5-panel figure:

  (a) King A-vs-B: the SAME physical model by two INDEPENDENT engines (A:
      lowered-isothermal DF + coupled ODE; B: prescribed King density +
      Poisson quadrature + Eddington inversion) -- sampled sigma_1d(r) and
      radial CDF must agree. THE cross-engine trust anchor.
  (b) Plummer analytic DF: the inverter against BT2008 f(E) propto E^{7/2}
      (untruncated zero point) AND the exact truncated closed-form oracle.
  (c) Plummer halo + EFF core headline: theory Q_j = 0.5 (exact-quadrature
      oracle), sampled global Q = 0.5 unscaled, and the sampled per-component
      Q_j against the hybrid predict-the-offset quadrature expectation (the
      hard-truncated halo plateaus BELOW 0.5 -- verified physics, not bias).
  (d) Osipkov-Merritt: sampled beta_halo(r) tracks r^2/(r^2 + r_a^2).
  (e) Differentiability: AD == FD through the full build (3 parameters).

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_multicomponent_eddington.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR
from progenax import EFFProfile, KingProfile, PlummerProfile, compute_potential_energy
from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.dynamics.virial import _accelerations
from progenax.kinematics.eddington import eddington_invert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = STELLAR.G  # pc^3 Msun^-1 Myr^-2 -> lengths pc, velocities pc/Myr


# ---------------------------------------------------------------------------
# helpers (mirrors of tests/validation/test_engine_b_physics.py)
# ---------------------------------------------------------------------------


def _headline_model(**kw):
    """Plummer halo + EFF(gamma=5) core (a=0.8 is a REALIZABILITY constraint)."""
    cfg = dict(
        profiles=[PlummerProfile(r_h=2.0), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)],
        mass_fractions=jnp.array([0.6, 0.4]),
        m_j=jnp.array([0.5, 1.0]),
    )
    cfg.update(kw)
    return MultiComponentCluster.from_density_profiles(**cfg)


def _com_arrays(ic):
    p = np.asarray(ic.positions - jnp.average(ic.positions, axis=0, weights=ic.masses))
    v = np.asarray(ic.velocities - jnp.average(ic.velocities, axis=0, weights=ic.masses))
    return p, v, np.asarray(ic.masses)


def _sampled_component_Q(model, seed, n_stars):
    """Sampled per-component Q_j = T_j/|W_j| (Clausius in the total field).

    Accelerations come from the library's memory-bounded blocked kernel
    (progenax.dynamics.virial._accelerations, O(block*N)); the local numpy
    mirror existed only because the dense kernel used to OOM.
    """
    ic = model.sample_cluster(jax.random.PRNGKey(seed), n_stars=n_stars, G=G)
    p, v, mass = _com_arrays(ic)
    cid = np.asarray(ic.component_id)
    a = np.asarray(_accelerations(jnp.asarray(p), jnp.asarray(mass), G))
    T_i = 0.5 * mass * np.sum(v**2, axis=1)
    W_i = mass * np.sum(p * a, axis=1)
    return np.array([T_i[cid == j].sum() / abs(W_i[cid == j].sum())
                     for j in range(int(cid.max()) + 1)])


def _predicted_component_Q(model, n_w=400):
    """Exact-quadrature HYBRID expectation of the sampled estimator: rho_presc
    weights x DF speed moments x prescribed-total Clausius field (the
    continuum limit of what Engine B's hybrid sampler realizes)."""
    st = model.engine_b
    r, Psi = st.r_poisson, st.Psi_poisson
    dphi_dr = -st.dPsi_dr_poisson
    Psi_safe = jnp.maximum(Psi, 1e-12)

    def moments(Psi_r, f_row):
        w = jnp.linspace(0.0, jnp.sqrt(2.0 * Psi_r), n_w)
        f_at = jnp.maximum(jnp.interp(Psi_r - 0.5 * w**2, st.E_grid, f_row), 0.0)
        return jnp.trapezoid(w**2 * f_at, w), jnp.trapezoid(w**4 * f_at, w)

    out = []
    for j in range(st.f_j_grid.shape[0]):
        m0, m2 = jax.vmap(lambda P: moments(P, st.f_j_grid[j]))(Psi_safe)
        finite = jnp.isfinite(st.r_a_j[j])
        ra_safe = jnp.where(finite, st.r_a_j[j], 1.0)
        inv_st2 = jnp.where(finite, 1.0 / (1.0 + (r / ra_safe) ** 2), 1.0)
        v2 = (m2 / (m0 + 1e-300)) * (1.0 / 3.0 + (2.0 / 3.0) * inv_st2)
        rho_p = st.rho_j_poisson[j]
        T = jnp.trapezoid(0.5 * rho_p * v2 * 4.0 * jnp.pi * r**2, r)
        W = jnp.trapezoid(-rho_p * r * dphi_dr * 4.0 * jnp.pi * r**2, r)
        out.append(float(T / jnp.abs(W)))
    return np.array(out)


def _plummer_grids(a=1.0, rt=100.0, n=20000):
    """Analytic (unnormalized, G=1) Plummer inputs for the inverter."""
    r = jnp.linspace(1e-5, rt, n)
    x = r / a
    rho = (1.0 + x**2) ** (-2.5)
    drho = -5.0 * (r / a**2) * (1.0 + x**2) ** (-3.5)
    Mr = (4.0 * jnp.pi * a**3 / 3.0) * x**3 / (1.0 + x**2) ** 1.5
    M_tot = 4.0 * jnp.pi * a**3 / 3.0
    Phi = -M_tot / jnp.sqrt(r**2 + a**2)
    dPsi_dr = -Mr / r**2
    return r, rho, drho, Phi, dPsi_dr, M_tot


# ---------------------------------------------------------------------------
# sections (each prints its block and returns (rows, plot payload))
# ---------------------------------------------------------------------------


def section_king(ax):
    """(a) King A-vs-B: sampled sigma_1d(r) + radial KS."""
    print("\n[a] King A-vs-B (W0=5, r_c=1 pc): two independent engines")
    king = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)
    mB = MultiComponentCluster.from_density_profiles(
        [king], jnp.array([1.0]), m_j=jnp.array([1.0]))
    mA = MultiComponentCluster.from_components(
        alpha_j=jnp.array([1.0]), w_j=jnp.array([1.0]), m_j=jnp.array([1.0]),
        W0=5.0, g=1.0, r_c=1.0)

    N = 20000
    key = jax.random.PRNGKey(0)
    icA = mA.sample_cluster(key, n_stars=N, G=G)
    icB = mB.sample_cluster(key, n_stars=N, G=G)
    _, vA, _ = _com_arrays(icA)
    _, vB, _ = _com_arrays(icB)
    rA = np.asarray(jnp.linalg.norm(icA.positions, axis=1))
    rB = np.asarray(jnp.linalg.norm(icB.positions, axis=1))
    v2A, v2B = np.sum(vA**2, axis=1), np.sum(vB**2, axis=1)

    grid = np.sort(np.concatenate([rA, rB]))
    ks = float(np.max(np.abs(
        np.searchsorted(np.sort(rA), grid, side="right") / N
        - np.searchsorted(np.sort(rB), grid, side="right") / N)))

    edges = np.quantile(rA, np.linspace(0.05, 0.90, 7))
    centers, sigA, sigB = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sA = (rA >= lo) & (rA < hi)
        sB = (rB >= lo) & (rB < hi)
        centers.append(0.5 * (lo + hi))
        sigA.append(np.sqrt(v2A[sA].mean() / 3.0))
        sigB.append(np.sqrt(v2B[sB].mean() / 3.0))
    centers, sigA, sigB = map(np.array, (centers, sigA, sigB))
    max_dev = float(np.max(np.abs(sigB / sigA - 1.0)))
    del icA, icB, vA, vB, rA, rB, v2A, v2B, grid  # free the 2x20k samples
    print(f"    radial KS distance          = {ks:.4f}  (gate < 0.02)")
    print(f"    max |sigma_B/sigma_A - 1|   = {max_dev:.4f}  (gate < 0.02)")

    ax.plot(centers, sigA, color=OI["blue"], lw=1.7, marker="o", ms=4,
            label="Engine A (ODE + lowered DF)")
    ax.plot(centers, sigB, color=OI["vermilion"], lw=1.7, marker="s", ms=4,
            mfc="white", ls="--", label="Engine B (Eddington)")
    ax.set_xlabel(r"$r$ [pc]")
    ax.set_ylabel(r"$\sigma_{1d}$ [pc Myr$^{-1}$]")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(r"King W$_0$=5: A vs B $\sigma(r)$ overlay", fontsize=9)
    return [
        ("King A-vs-B radial KS", f"{ks:.4f}", "< 0.02", ks < 0.02),
        ("King A-vs-B max sigma dev", f"{max_dev:.4f}", "< 0.02", max_dev < 0.02),
    ]


def section_plummer(ax):
    """(b) Plummer analytic-DF oracles: E^{7/2} law + truncated closed form."""
    print("\n[b] Plummer analytic DF oracles (eddington_invert, G=1 model units)")
    a, rt = 1.0, 100.0
    r, rho, drho, Phi, dPsi_dr, M_tot = _plummer_grids(a, rt)

    # Untruncated zero point: Psi = -Phi, f(E) propto E^{7/2} exactly (BT2008).
    Psi_u = -Phi
    E_u, f_u = eddington_invert(r, rho, drho, Psi_u, dPsi_dr)
    Psi0_u = float(Psi_u[0])
    sel_u = (np.asarray(E_u) > 0.1 * Psi0_u) & (np.asarray(E_u) < 0.8 * Psi0_u)
    E_i, f_i = np.asarray(E_u)[sel_u], np.asarray(f_u)[sel_u]
    i_ref = len(E_i) // 2
    err_law = np.abs(f_i / f_i[i_ref] / (E_i / E_i[i_ref]) ** 3.5 - 1.0)
    max_law = float(np.max(err_law))

    # Truncated zero point: exact closed form INCLUDING the boundary term.
    Psi_t = Phi[-1] - Phi
    E_t, f_t = eddington_invert(r, rho, drho, Psi_t, dPsi_dr)
    c = float(M_tot) / np.sqrt(rt**2 + a**2)
    k = (a / float(M_tot)) ** 5
    E_n = np.asarray(E_t)
    b = E_n + c
    integral = (2.0 * b**3 * np.sqrt(E_n) - 2.0 * b**2 * E_n**1.5
                + 1.2 * b * E_n**2.5 - (2.0 / 7.0) * E_n**3.5)
    f_exact = (20.0 * k * integral + 5.0 * k * c**4 / np.sqrt(E_n)) / (
        np.sqrt(8.0) * np.pi**2)
    Psi0_t = float(Psi_t[0])
    sel_t = (E_n > 0.1 * Psi0_t) & (E_n < 0.8 * Psi0_t)
    err_cf = np.abs(np.asarray(f_t)[sel_t] / f_exact[sel_t] - 1.0)
    max_cf = float(np.max(err_cf))
    print(f"    max rel err vs E^3.5 law (untruncated zero pt) = {max_law:.2e}  (gate < 1e-3)")
    print(f"    max rel err vs truncated closed form           = {max_cf:.2e}  (gate < 1e-4)")

    ax.loglog(E_i, f_i, color=OI["blue"], lw=1.7, label=r"$f(E)$ (inverter)")
    ax.loglog(E_i, f_i[i_ref] * (E_i / E_i[i_ref]) ** 3.5, color=OI["orange"],
              lw=1.4, ls="--", label=r"$\propto E^{7/2}$ (BT2008)")
    ax.set_xlabel(r"$E$ (model units, $G=1$)")
    ax.set_ylabel(r"$f(E)$ (model units)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title(r"Plummer ergodic DF: $f \propto E^{7/2}$", fontsize=9)
    ins = ax.inset_axes([0.56, 0.12, 0.40, 0.32])
    ins.semilogy(E_i / Psi0_u, err_law, color=OI["blue"], lw=1.0)
    ins.semilogy(E_n[sel_t] / Psi0_t, err_cf, color=OI["green"], lw=1.0)
    ins.set_xlabel(r"$E/\Psi_0$", fontsize=6)
    ins.set_ylabel(r"|rel err|", fontsize=6)
    ins.tick_params(labelsize=5)
    return [
        ("Plummer f vs E^3.5 law (untrunc)", f"{max_law:.2e}", "< 1e-3", max_law < 1e-3),
        ("Plummer f vs trunc closed form", f"{max_cf:.2e}", "< 1e-4", max_cf < 1e-4),
    ]


def section_headline(ax_rho, ax_q):
    """(c) halo+core headline: theory Q_j, global Q, predict-the-offset Q_j."""
    print("\n[c] Headline mix: Plummer halo (r_h=2 pc) + EFF gamma=5 core (a=0.8 pc)")
    m = _headline_model()

    Qj = np.asarray(m.component_virial_ratios())
    dev_th = float(np.max(np.abs(Qj - 0.5)))
    print(f"    theory Q_j (DF-weighted oracle)  = [{Qj[0]:.5f}, {Qj[1]:.5f}]"
          f"  (gate 0.5 +- 3e-3)")

    ic = m.sample_cluster(jax.random.PRNGKey(0), n_stars=30000, G=G)
    p, v, mass = _com_arrays(ic)
    T = 0.5 * float(np.sum(mass * np.sum(v**2, axis=1)))
    # Library blocked kernel (O(block*N) memory) -- the numpy mirror is gone.
    V = float(compute_potential_energy(jnp.asarray(p), jnp.asarray(mass), G=G))
    Q_glob = T / abs(V)
    print(f"    sampled global Q (N=30k, unscaled) = {Q_glob:.4f}  (gate 0.5 +- 0.02)")
    del ic, p, v, mass  # free the 30k sample before the per-seed Q_j passes

    Q_pred = _predicted_component_Q(m)
    seeds = (1, 2, 3)
    Q_seeds = np.stack([_sampled_component_Q(m, sd, 16000) for sd in seeds])
    Q_meas = Q_seeds.mean(axis=0)
    Q_sem = Q_seeds.std(axis=0) / np.sqrt(len(seeds))
    dev_pred = float(np.max(np.abs(Q_meas - Q_pred)))
    print(f"    predicted hybrid Q_j  = [{Q_pred[0]:.4f}, {Q_pred[1]:.4f}]"
          f"  (halo plateaus BELOW 0.5: truncation-edge physics)")
    print(f"    sampled Q_j ({len(seeds)} seeds x 16k) = "
          f"[{Q_meas[0]:.4f} +- {Q_sem[0]:.4f}, {Q_meas[1]:.4f} +- {Q_sem[1]:.4f}]"
          f"  (gate |sampled - predicted| < 0.012)")

    # Panel 3: rho_DF,j vs rho_presc,j (truncation-consistent form, both comps).
    st = m.engine_b
    r = np.asarray(st.r_poisson)
    Psi = st.Psi_poisson
    n_w = 1200

    def rho_df_row(f_row):
        def m0(Psi_r):
            w = jnp.linspace(0.0, jnp.sqrt(2.0 * jnp.maximum(Psi_r, 1e-12)), n_w)
            f_at = jnp.maximum(jnp.interp(Psi_r - 0.5 * w**2, st.E_grid, f_row), 0.0)
            return jnp.trapezoid(w**2 * f_at, w)
        return 4.0 * np.pi * np.asarray(jax.vmap(m0)(Psi))

    for j, (col, lab) in enumerate([(OI["blue"], "halo"), (OI["vermilion"], "core")]):
        rho_df = rho_df_row(st.f_j_grid[j])
        rho_presc = np.asarray(st.rho_j_poisson[j])
        target = rho_presc - rho_presc[-1]  # the DF represents rho(Psi)-rho(0)
        sel = (r > 0.02) & (target > 0)
        ax_rho.loglog(r[sel], target[sel], color=col, lw=1.7,
                      label=rf"{lab} $\rho_{{\rm presc}} - \rho(r_t)$")
        ax_rho.loglog(r[sel][::40], rho_df[sel][::40], color=col, ls="none",
                      marker="o", ms=3, mfc="white",
                      label=rf"{lab} $\rho_{{\rm DF}}$")
    ax_rho.set_xlabel(r"$r$ [pc]")
    ax_rho.set_ylabel(r"$\hat\rho_j$ (model units, $M_{\rm tot}=1$)")
    ax_rho.legend(frameon=False, fontsize=6.5)
    ax_rho.set_title(r"DF integrates back to $\rho_{\rm presc}$ (trunc.-consistent)",
                     fontsize=9)

    # Panel 5: Q_j summary (theory, predicted-hybrid, sampled +- sem).
    x = np.arange(2)
    ax_q.axhline(0.5, color="0.6", ls="--", lw=1.0, label=r"virial $Q=0.5$")
    ax_q.plot(x - 0.18, Qj, ls="none", marker="^", ms=7, color=OI["green"],
              label="theory (DF oracle)")
    ax_q.plot(x, Q_pred, ls="none", marker="D", ms=6, mfc="white",
              color=OI["orange"], label="predicted hybrid")
    ax_q.errorbar(x + 0.18, Q_meas, yerr=Q_sem, ls="none", marker="o", ms=6,
                  color=OI["blue"], capsize=3,
                  label=rf"sampled ({len(seeds)} seeds $\times$ 16k)")
    ax_q.set_xticks(x)
    ax_q.set_xticklabels(["halo (Plummer)", "core (EFF)"])
    ax_q.set_xlim(-0.6, 1.6)
    ax_q.set_ylim(0.485, 0.515)
    ax_q.set_ylabel(r"$Q_j = T_j/|W_j|$")
    ax_q.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax_q.set_title("per-component virial: predict-the-offset", fontsize=9)
    return [
        ("Headline theory max|Q_j - 0.5|", f"{dev_th:.1e}", "< 3e-3", dev_th < 3e-3),
        ("Headline global Q (N=30k)", f"{Q_glob:.4f}", "0.5 +- 0.02",
         abs(Q_glob - 0.5) < 0.02),
        ("Headline max|Q_j - Q_pred|", f"{dev_pred:.4f}", "< 0.012", dev_pred < 0.012),
    ]


def section_om(ax):
    """(d) OM anisotropy: sampled beta_halo(r) vs r^2/(r^2 + r_a^2)."""
    print("\n[d] Osipkov-Merritt: r_a = 3 pc on the halo (core isotropic)")
    r_a = 3.0
    m = _headline_model(r_a_j=jnp.array([r_a, jnp.inf]))
    n_seeds = 4
    r_l, vr2_l, vt2_l, cid_l = [], [], [], []
    for sd in range(n_seeds):
        ic = m.sample_cluster(jax.random.PRNGKey(sd), n_stars=20000, G=G)
        pos, vel = np.asarray(ic.positions), np.asarray(ic.velocities)
        r = np.linalg.norm(pos, axis=1)
        v_r = np.sum(pos * vel, axis=1) / np.maximum(r, 1e-30)
        r_l.append(r)
        vr2_l.append(v_r**2)
        vt2_l.append(np.sum(vel**2, axis=1) - v_r**2)
        cid_l.append(np.asarray(ic.component_id))
    r = np.concatenate(r_l)
    vr2, vt2, cid = np.concatenate(vr2_l), np.concatenate(vt2_l), np.concatenate(cid_l)
    del r_l, vr2_l, vt2_l, cid_l

    halo = cid == 0
    edges = np.quantile(r[halo], np.linspace(0.05, 0.95, 9))
    centers, beta_s, beta_om = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = halo & (r >= lo) & (r < hi)
        centers.append(float(np.mean(r[s])))
        beta_s.append(1.0 - vt2[s].mean() / (2.0 * vr2[s].mean()))
        beta_om.append(float(np.mean(r[s] ** 2 / (r[s] ** 2 + r_a**2))))
    centers, beta_s, beta_om = map(np.array, (centers, beta_s, beta_om))
    max_dev = float(np.max(np.abs(beta_s - beta_om)))
    del r, vr2, vt2, cid, halo  # free the pooled 4x20k kinematics arrays
    print(f"    pooled {n_seeds} seeds x 20k stars; 8 interior halo bins")
    print(f"    max |beta_sampled - beta_OM|  = {max_dev:.4f}  (gate < 0.05)")

    rr = np.linspace(0.05, float(m.r_t), 200)
    ax.plot(rr, rr**2 / (rr**2 + r_a**2), color=OI["orange"], lw=1.7,
            label=r"OM: $r^2/(r^2 + r_a^2)$")
    ax.plot(centers, beta_s, ls="none", marker="o", ms=5, mfc="white",
            color=OI["blue"], label=rf"sampled halo ({n_seeds} seeds)")
    ax.axhline(0.0, color="0.6", ls="--", lw=1.0)
    ax.set_xlabel(r"$r$ [pc]")
    ax.set_ylabel(r"$\beta(r) = 1 - \sigma_t^2 / 2\sigma_r^2$")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title(rf"OM anisotropy realized ($r_a = {r_a:.0f}$ pc)", fontsize=9)
    return [("OM max |beta - OM curve|", f"{max_dev:.4f}", "< 0.05", max_dev < 0.05)]


def section_gradients():
    """(e) AD-vs-FD through the full Engine B build (3 Task-6 configs)."""
    print("\n[e] Gradients: AD vs central FD through the full build")
    from progenax.cluster.eddington_engine import build_engine_b_state

    n_r, n_e = 3000, 500
    king = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)

    def scalar(state):
        return jnp.mean(state.Psi_poisson) + jnp.mean(state.f_j_grid[0])

    def loss_rh(x):
        state, _ = build_engine_b_state(
            [PlummerProfile(r_h=x), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)],
            jnp.array([0.6, 0.4]), jnp.array([jnp.inf, jnp.inf]),
            None, 0.995, n_r, n_e)
        return scalar(state)

    def loss_t(t):
        state, _ = build_engine_b_state(
            [king, PlummerProfile(r_h=2.0)], jnp.stack([t, 1.0 - t]),
            jnp.array([3.0, jnp.inf]), None, 0.995, n_r, n_e)
        return scalar(state)

    def loss_ra(ra):
        state, _ = build_engine_b_state(
            [king, PlummerProfile(r_h=2.0)], jnp.array([0.5, 0.5]),
            jnp.stack([ra, jnp.inf]), None, 0.995, n_r, n_e)
        return scalar(state)

    rows = []
    for name, loss, x0 in (("halo r_h", loss_rh, 2.0),
                           ("mass fraction t", loss_t, 0.5),
                           ("r_a_j[0]", loss_ra, 3.0)):
        ad = float(jax.grad(loss)(jnp.asarray(x0)))
        h = 1e-4 * abs(x0)
        fd = (float(loss(jnp.asarray(x0 + h)))
              - float(loss(jnp.asarray(x0 - h)))) / (2.0 * h)
        rel = abs(ad - fd) / abs(fd)
        print(f"    {name:<16s} AD = {ad:+.8e}  FD = {fd:+.8e}  rel = {rel:.2e}")
        rows.append((f"grad {name} AD-vs-FD", f"{rel:.2e}", "< 1e-3", rel < 1e-3))
    return rows


def main():
    print("=" * 78)
    print("ENGINE B VALIDATION: prescribed-density shared-Psi Eddington equilibria")
    print("(units: STELLAR -- lengths pc, masses Msun, velocities pc/Myr)")
    print("=" * 78)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.6))
    (axA, axB, axC), (axD, axE, axF) = axes
    axF.set_axis_off()

    rows = []
    rows += section_king(axA)
    rows += section_plummer(axB)
    rows += section_headline(axC, axE)
    rows += section_om(axD)
    rows += section_gradients()

    for ax, tag in ((axA, "(a)"), (axB, "(b)"), (axC, "(c)"), (axD, "(d)"),
                    (axE, "(e)")):
        loc = "upper right" if ax in (axB, axE) else "upper left"
        panel_label(ax, tag, loc=loc)

    fig.tight_layout(pad=0.7)
    save_fig(fig, OUTPUT_DIR, "engine_b_eddington")

    print("\n" + "-" * 78)
    print(f"  {'CHECK':<36s} {'measured':>12s} {'gate':>12s}   status")
    print("-" * 78)
    all_ok = True
    for name, measured, gate, ok in rows:
        all_ok &= ok
        print(f"  {name:<36s} {measured:>12s} {gate:>12s}   {'PASS' if ok else 'FAIL'}")
    print("-" * 78)
    print(f"  saved {OUTPUT_DIR}/engine_b_eddington.{{png,pdf}}")
    print("=" * 78)
    print("  ENGINE B VALIDATION: ALL PASS" if all_ok
          else "  ENGINE B VALIDATION: FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
