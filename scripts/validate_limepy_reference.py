#!/usr/bin/env python
"""Ours-vs-reference-LIMEPY cross-validation (Engine A is faithful published LIMEPY).

Builds the SAME lowered-isothermal model (single-mass and GZ15 multimass, isotropic
and anisotropic) in BOTH codes and compares scale-invariant quantities:

  - converged central density fractions alpha_j (sum to 1);
  - density shape rho_j(r)/rho_j(0) on the reference's own r/r0 grid;
  - dispersion shape sigma_j(r)/sigma_j(0) (from the 3-D mean-square v2_j);
  - realized per-component mass fractions;
  - concentration c = log10(r_t/r0) and half-mass radius r_h/r0 (ours from the
    cumulative-mass profile; cross-checked against the reference's .rh).

Reference: the canonical numpy/scipy LIMEPY (Gieles & Zocchi 2015) at
``ref-repos/limepy`` (git ef2a479, v1.2-21). It does NOT run under the project
env (scipy 1.17: float ``nsteps`` at limepy.py:488 + incompatible dopri5 solout
API), so it is invoked in a pinned ephemeral subprocess
(``uv run --no-project --python 3.11 --with numpy==1.26.4 --with scipy==1.11.4``,
see scripts/_limepy_reference_worker.py) and its outputs are CACHED as .npz with
full provenance under ``validation/data/limepy_reference/`` (committed, so the
skip-if-absent parity test can run without the pinned env). ``--regen`` re-runs
the reference; otherwise the cache is reused.

m-bar convention: the reference is run with ``meanmassdef='central'`` so its mean
mass is the GZ15 eq-26 central-density-weighted m-bar = sum_j m_j alpha_j --
identical to progenax's ``bar_m``. No Peuten et al. (2017) eq 8-9 translation
(W0* = W0 (mbar*/mbar)^(2 delta)) is then needed: both codes share one (W0, mbar)
convention. Radial units also coincide: our xi = r/r_c uses the same -9 (King
radius r0) nondimensionalization as the reference's r/r0 (config 1 confirms).

Resolution note: parity is measured with a WELL-RESOLVED ours-side ODE grid
(xi_max ~ 3 x xi_t, n_ode_points >= 4000). The released default (xi_max=300,
n_ode_points=2000) has ~0.9% pure-interpolation error in the core (grid spacing
0.15 in xi) -- a grid artifact, not a physics disagreement (measured 2026-06-11).

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_limepy_reference.py
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_limepy_reference.py --regen
"""
import argparse
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from progenax.imf.smooth import Maschberger
from progenax.profiles.limepy import _aniso_density_scalar, limepy_density_hat
from progenax.profiles.limepy_multimass import (
    _aniso_v2hat_scalar,
    _bin_imf,
    find_alpha_for_masses,
)
from progenax.cluster.multicomponent import MultiComponentCluster

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "validation", "data", "limepy_reference")
OUTPUT_DIR = os.path.join(REPO_ROOT, "validation", "plots")
WORKER = os.path.join(REPO_ROOT, "scripts", "_limepy_reference_worker.py")

# The B2-demo IMF binning (demo_delta_recovery.py: Maschberger alpha=2.3,
# N_COMP=4, M_RANGE=(0.1, 20), delta=0.4, W0=5) via the SAME _bin_imf as from_imf.
IMF_N_COMP, IMF_M_RANGE, IMF_ALPHA = 4, (0.1, 20.0), 2.3

# Each config: the model in BOTH codes + the measured-then-frozen PASS gates.
# Gates were MEASURED FIRST (2026-06-11, this script's table), then frozen at
# ~3-10x the measured deviations: shapes measured at 5e-7..8.4e-5, alpha and
# mass fractions at 0..7.1e-5, c at 1.4e-5..1.5e-4, r_h at <=6.3e-6. Every gate
# is <=5e-4 -- 20x tighter than the 1% physics target. NEVER loosen to pass.
CONFIGS = [
    dict(name="single_g1", label="single-mass g=1 (King), W0=5",
         W0=5.0, g=1.0, mj=None, Mj=None, delta=None, eta=None, ra=None,
         xi_max=30.0, n_ode=8000,
         gates=dict(rho=2e-5, sig=2e-5, alpha=1e-8, mfrac=1e-8, c=5e-4, rh=1e-4)),
    dict(name="single_g15", label="single-mass g=1.5, W0=5",
         W0=5.0, g=1.5, mj=None, Mj=None, delta=None, eta=None, ra=None,
         xi_max=45.0, n_ode=8000,
         gates=dict(rho=5e-5, sig=2e-5, alpha=1e-8, mfrac=1e-8, c=5e-4, rh=1e-4)),
    dict(name="twocomp_iso", label="2-comp m=[0.3,1], f=[0.7,0.3], delta=0.5",
         W0=5.0, g=1.0, mj=[0.3, 1.0], Mj=[0.7, 0.3], delta=0.5, eta=None, ra=None,
         xi_max=30.0, n_ode=8000,
         gates=dict(rho=5e-5, sig=3e-4, alpha=5e-4, mfrac=5e-4, c=5e-4, rh=1e-4)),
    dict(name="imf4", label="4-comp Maschberger(2.3) [0.1,20], delta=0.4 (B2)",
         W0=5.0, g=1.0, mj="imf", Mj="imf", delta=0.4, eta=None, ra=None,
         xi_max=30.0, n_ode=8000,
         gates=dict(rho=5e-5, sig=3e-4, alpha=5e-4, mfrac=5e-4, c=5e-4, rh=1e-4)),
    dict(name="twocomp_ra_eta0", label="2-comp + ra=5 r0, eta=0 (aniso)",
         W0=5.0, g=1.0, mj=[0.3, 1.0], Mj=[0.7, 0.3], delta=0.5, eta=0.0, ra=5.0,
         xi_max=30.0, n_ode=6000,
         gates=dict(rho=5e-5, sig=3e-4, alpha=5e-4, mfrac=5e-4, c=5e-4, rh=1e-4)),
    dict(name="twocomp_ra_eta05", label="2-comp + ra=5 r0, eta=0.5 (aniso)",
         W0=5.0, g=1.0, mj=[0.3, 1.0], Mj=[0.7, 0.3], delta=0.5, eta=0.5, ra=5.0,
         xi_max=30.0, n_ode=6000,
         gates=dict(rho=5e-5, sig=3e-4, alpha=5e-4, mfrac=5e-4, c=5e-4, rh=1e-4)),
]

REPRESENTATIVE = "imf4"  # the figure's multimass config


def _imf_bins():
    imf = Maschberger(alpha=IMF_ALPHA, m_min=IMF_M_RANGE[0], m_max=IMF_M_RANGE[1])
    return _bin_imf(imf, IMF_N_COMP, IMF_M_RANGE)


def _ref_config_json(cfg):
    """The worker's config dict (concrete floats only; 'imf' bins resolved here)."""
    out = {"W0": cfg["W0"], "g": cfg["g"]}
    if cfg["mj"] is not None:
        mj, Mj = (cfg["mj"], cfg["Mj"]) if cfg["mj"] != "imf" else \
            tuple([float(x) for x in arr] for arr in _imf_bins())
        out.update(mj=list(mj), Mj=list(Mj), delta=cfg["delta"])
        if cfg["eta"] is not None:
            out["eta"] = cfg["eta"]
    if cfg["ra"] is not None:
        out["ra"] = cfg["ra"]
    return out


def load_or_generate_reference(cfg, regen=False):
    """Load the cached reference npz; (re)generate via the pinned subprocess if
    absent or --regen. The pytest parity test reads the SAME cache and skips if
    absent -- only this CLI ever shells out to uv."""
    path = os.path.join(CACHE_DIR, f"{cfg['name']}.npz")
    if regen or not os.path.exists(path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        ref_cfg = _ref_config_json(cfg)
        print(f"  [regen] running reference LIMEPY subprocess for {cfg['name']} ...")
        cmd = ["env", "-u", "VIRTUAL_ENV", "uv", "run", "--no-project",
               "--python", "3.11", "--with", "numpy==1.26.4",
               "--with", "scipy==1.11.4", "python", WORKER,
               json.dumps(ref_cfg), path]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        if res.returncode != 0:
            raise RuntimeError(
                f"reference worker failed for {cfg['name']}:\n{res.stderr[-2000:]}")
        print("    " + res.stdout.strip().splitlines()[-1])
    return np.load(path)


def build_ours(cfg):
    """Build the progenax (Engine A, JAX) model for a config.

    Multimass: find_alpha_for_masses on the SAME (xi_max, n_ode) grid, then
    from_mass_segregation / from_imf. Anisotropic models keep the default
    'table' density source for the solve (error budget |psi| <= 1e-4 W0,
    asserted in tests) -- the parity profiles below are evaluated with the
    exact quadrature oracle at the solved psi.
    """
    W0, g = cfg["W0"], cfg["g"]
    kw = dict(xi_max=cfg["xi_max"], n_ode_points=cfg["n_ode"])
    if cfg["mj"] is None:
        return MultiComponentCluster.from_components(
            alpha_j=jnp.array([1.0]), w_j=jnp.array([1.0]), m_j=jnp.array([1.0]),
            W0=W0, g=g, **kw)
    if cfg["mj"] == "imf":
        m_j, M_j = _imf_bins()
    else:
        m_j, M_j = jnp.asarray(cfg["mj"]), jnp.asarray(cfg["Mj"])
    ra_hat = cfg["ra"]  # ours r_a/r_c == reference ra/r0 (shared -9 normalization)
    eta = cfg["eta"] if cfg["eta"] is not None else 0.0
    alpha_j, residual = find_alpha_for_masses(
        m_j, M_j, W0, g, cfg["delta"], xi_max=cfg["xi_max"],
        n_points=cfg["n_ode"], ra_hat=ra_hat, eta=eta)
    model = MultiComponentCluster.from_mass_segregation(
        alpha_j=alpha_j, m_j=m_j, W0=W0, g=g, delta=cfg["delta"],
        r_a=ra_hat, eta=eta, r_c=1.0, **kw)
    return model


def ours_profiles(model, xi):
    """(rho_hat_j, sig_hat_j) on xi = r/r0: density and 1-D-dispersion shapes,
    both normalized to 1 at the centre, via the exact quadrature oracle at the
    solved psi(xi). Anisotropic components evaluate at p_j = xi/ra_hat_j."""
    psi = jnp.interp(xi, model.xi_grid, model.psi_grid, left=model.W0, right=0.0)
    rescale = model.rescale_j
    ra_j = model.ra_hat_j  # inf = isotropic (p_j -> 0)
    rho, sig = [], []
    for j in range(rescale.shape[0]):
        p = jnp.where(jnp.isfinite(ra_j[j]), xi / ra_j[j], 0.0)
        if bool(jnp.isfinite(ra_j[j])):
            rho_j = jax.vmap(lambda W, pp: _aniso_density_scalar(W, pp, model.g))(
                rescale[j] * psi, p)
            rho_0 = _aniso_density_scalar(rescale[j] * model.W0, jnp.asarray(0.0),
                                          model.g)
        else:
            rho_j = limepy_density_hat(rescale[j] * psi, model.g)
            rho_0 = limepy_density_hat(rescale[j] * model.W0, model.g)
        v2_j = jax.vmap(lambda W, pp: _aniso_v2hat_scalar(W, pp, model.g))(
            rescale[j] * psi, p)
        v2_0 = _aniso_v2hat_scalar(rescale[j] * model.W0, jnp.asarray(0.0), model.g)
        rho.append(rho_j / rho_0)
        sig.append(jnp.sqrt(v2_j / v2_0))
    return jnp.stack(rho), jnp.stack(sig)


def ours_mass_profile(model, n=6000):
    """(r_grid/r0, M(<r)/M_tot, realized mass fractions f_j) from the model's own
    density components (quadrature oracle), trapezoid cumulative mass."""
    xg = jnp.linspace(1e-5, float(model.r_t), n)
    rho_j, _ = ours_profiles(model, xg)
    w_j = model.alpha_j[:, None] * rho_j  # alpha_j rho_hat_j
    nu_j = jnp.trapezoid(w_j * xg[None, :] ** 2, xg, axis=1)
    f_j = nu_j / jnp.sum(nu_j)
    integ = jnp.sum(w_j, axis=0) * xg**2
    M = jnp.concatenate([jnp.zeros(1), jnp.cumsum(
        0.5 * (integ[1:] + integ[:-1]) * jnp.diff(xg))])
    return xg, M / M[-1], f_j


def compare_config(cfg, ref, model):
    """All scale-invariant deviations for one config -> dict (measured, gated)."""
    r0, rt_ref = float(ref["r0"]), float(ref["rt"])
    xi_ref = ref["r"] / r0
    sel = (xi_ref > 0.0) & (xi_ref < 0.999 * rt_ref / r0)
    xi = jnp.asarray(xi_ref[sel])

    rho_ours, sig_ours = ours_profiles(model, xi)
    rho_ref = ref["rhoj"][:, sel] / ref["rhoj"][:, 0:1]
    sig_ref = np.sqrt(ref["v2j"][:, sel] / ref["v2j"][:, 0:1])

    d_rho = float(jnp.max(jnp.abs(rho_ours - jnp.asarray(rho_ref))))
    d_sig = float(jnp.max(jnp.abs(sig_ours - jnp.asarray(sig_ref))))
    d_alpha = float(np.max(np.abs(np.asarray(model.alpha_j) - ref["alpha"])))

    xg, Mcum, f_ours = ours_mass_profile(model)
    f_ref = ref["Mj"] / ref["Mj"].sum()
    d_mfrac = float(np.max(np.abs(np.asarray(f_ours) - f_ref)))

    c_ours = float(jnp.log10(model.r_t / model.r_c))
    c_ref = np.log10(rt_ref / r0)
    d_c = abs(c_ours - c_ref)

    rh_ours = float(jnp.interp(0.5, Mcum, xg))
    rh_ref = float(ref["rh"]) / r0
    d_rh = abs(rh_ours - rh_ref) / rh_ref

    measured = dict(rho=d_rho, sig=d_sig, alpha=d_alpha, mfrac=d_mfrac,
                    c=d_c, rh=d_rh)
    return measured, dict(c_ours=c_ours, c_ref=c_ref, rh_ours=rh_ours,
                          rh_ref=rh_ref, alpha_ours=np.asarray(model.alpha_j),
                          alpha_ref=np.asarray(ref["alpha"]))


def make_figure(cfg, ref, model):
    """rho_j(r) and sigma_j(r) ours-vs-ref overlays + residual strips for the
    representative multimass config."""
    import matplotlib.pyplot as plt

    r0, rt_ref = float(ref["r0"]), float(ref["rt"])
    xi_ref = ref["r"] / r0
    sel = (xi_ref > 0.0) & (xi_ref < 0.999 * rt_ref / r0)
    xi = jnp.asarray(xi_ref[sel])
    rho_ours, sig_ours = ours_profiles(model, xi)
    rho_ref = ref["rhoj"][:, sel] / ref["rhoj"][:, 0:1]
    sig_ref = np.sqrt(ref["v2j"][:, sel] / ref["v2j"][:, 0:1])
    m_j = np.asarray(ref["mj"])
    cols = [OI["sky"], OI["green"], OI["orange"], OI["vermilion"]][:len(m_j)]

    fig, axes = plt.subplots(
        2, 2, figsize=(8.6, 5.6), sharex="col",
        gridspec_kw=dict(height_ratios=[3.0, 1.2], hspace=0.07, wspace=0.26))
    (axR, axS), (axRr, axSr) = axes
    x = np.asarray(xi)
    for j, col in enumerate(cols):
        axR.plot(x, rho_ref[j], color=col, lw=2.4, alpha=0.45,
                 label=rf"ref $m_j={m_j[j]:.2f}$")
        axR.plot(x, np.asarray(rho_ours[j]), color=col, lw=1.0, ls="--",
                 label="ours" if j == 0 else None)
        axRr.plot(x, np.asarray(rho_ours[j]) - rho_ref[j], color=col, lw=1.0)
        axS.plot(x, sig_ref[j], color=col, lw=2.4, alpha=0.45)
        axS.plot(x, np.asarray(sig_ours[j]), color=col, lw=1.0, ls="--")
        axSr.plot(x, np.asarray(sig_ours[j]) - sig_ref[j], color=col, lw=1.0)
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_ylim(1e-7, 2.0)
    axR.set_ylabel(r"$\rho_j(r)/\rho_j(0)$")
    axR.legend(fontsize=7, loc="lower left")
    panel_label(axR, "(a)", loc="upper right")
    axRr.set_xscale("log")
    axRr.axhline(0.0, color="0.6", lw=0.8, ls=":")
    axRr.set_xlabel(r"$r/r_0$"); axRr.set_ylabel("ours $-$ ref")
    axS.set_xscale("log")
    axS.set_ylabel(r"$\sigma_j(r)/\sigma_j(0)$")
    axS.set_ylim(0.0, 1.05)
    panel_label(axS, "(b)", loc="lower left")
    axSr.set_xscale("log")
    axSr.axhline(0.0, color="0.6", lw=0.8, ls=":")
    axSr.set_xlabel(r"$r/r_0$"); axSr.set_ylabel("ours $-$ ref")
    for ax in (axRr, axSr):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    save_fig(fig, OUTPUT_DIR, "limepy_reference_parity")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regen", action="store_true",
                    help="re-run the reference subprocess (pinned env) and "
                         "refresh the npz cache")
    args = ap.parse_args()

    print("\n" + "=" * 88)
    print("OURS-vs-REFERENCE-LIMEPY PARITY (canonical Gieles & Zocchi 2015 code, "
          "meanmassdef='central')")
    print("=" * 88)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_ok = True
    header = (f"  {'config':<18} {'quantity':<22} {'measured':>10} {'gate':>9} "
              f"{'status':>7}")
    rows = []
    for cfg in CONFIGS:
        ref = load_or_generate_reference(cfg, regen=args.regen)
        if not bool(ref["converged"]):
            print(f"  {cfg['name']}: reference model NOT converged -- FAIL")
            all_ok = False
            continue
        model = build_ours(cfg)
        measured, extra = compare_config(cfg, ref, model)
        labels = {
            "rho": "max|drho_j(r)| (shape)", "sig": "max|dsigma_j(r)| (shape)",
            "alpha": "max|dalpha_j|", "mfrac": "max|dM_j/M|",
            "c": "|dlog10(rt/r0)|", "rh": "|drh|/rh",
        }
        for k in ("rho", "sig", "alpha", "mfrac", "c", "rh"):
            ok = measured[k] <= cfg["gates"][k]
            all_ok &= ok
            rows.append(f"  {cfg['name']:<18} {labels[k]:<22} {measured[k]:>10.2e} "
                        f"{cfg['gates'][k]:>9.0e} {'PASS' if ok else 'FAIL':>7}")
        rows.append(f"  {'':<18} alpha ours={np.round(extra['alpha_ours'], 5)} "
                    f"ref={np.round(extra['alpha_ref'], 5)}; "
                    f"c={extra['c_ours']:.5f} vs {extra['c_ref']:.5f}; "
                    f"rh/r0={extra['rh_ours']:.5f} vs {extra['rh_ref']:.5f}")
        if cfg["name"] == REPRESENTATIVE:
            make_figure(cfg, ref, model)

    print(header)
    print("  " + "-" * 84)
    for row in rows:
        print(row)
    print("=" * 88)
    print(f"  figure: {OUTPUT_DIR}/limepy_reference_parity.{{png,pdf}}")
    print("  REFERENCE-LIMEPY PARITY PASS" if all_ok
          else "  REFERENCE-LIMEPY PARITY FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
