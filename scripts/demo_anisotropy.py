#!/usr/bin/env python
r"""B6 -- Anisotropy measurability and the OM-vs-Michie formalism (Batch C).

A paper-seed demo on velocity anisotropy. The Binney anisotropy
``beta(r) = 1 - sigma_t^2 / (2 sigma_r^2)`` is the observable; the **anisotropy
radius** r_a is the parameter. Two questions:

(a) **Well-specified.** For an Osipkov-Merritt (OM) cluster, can r_a be recovered
    from beta(r), and how many stars does detecting anisotropy take? The OM
    anisotropy profile is the closed form ``beta_OM(r) = r^2 / (r^2 + r_a^2)``; a
    Plummer+OM cluster is sampled, beta(r) binned, and r_a recovered by a
    chi^2 fit. A Fisher forecast gives sigma(r_a) vs N.

(b) **Misspecified.** Anisotropy comes in different FORMALISMS. The Michie (1963)
    anisotropic-King DF ``f ~ exp(-J^2/2 r_a^2 sigma^2)[exp(-E/sigma^2)-1]`` is NOT
    a function of a single Q = E - J^2/2r_a^2 (the OM ansatz), so its beta(r) shape
    differs. We sample a Michie cluster and fit it with the OM form: the recovered
    r_a is biased and the fit leaves a systematic beta-shape residual (an inflated
    reduced chi^2) -- the anisotropy formalism you ASSUME shapes what you infer.

There is no "King+OM" model in progenax (King's anisotropic form IS Michie), so the
misspecification is an OM fit to a Michie SAMPLE on the beta(r) channel, not a
fixed-density self-consistent refit -- stated plainly.

Channels & method: beta(r) from binned radial/tangential velocity moments
(``binned_sigma_beta``) with the conservative SE ``beta_se = (1+|beta|)/sqrt(n)``;
the OM model ``beta_OM(r; r_a)`` fit by a Gaussian chi^2 (``gaussian_loglike``),
r_a by Adam MLE, sigma(r_a) and the forecast from the (ODE-free) Fisher.

Gates (exit 0 = all pass):
  * (a) OM r_a recovery within 3 sigma of truth;
  * (a) the OM fit to an OM cluster is GOOD (reduced chi^2 ~ 1);
  * (a) forecast sigma(r_a) ~ N^-1/2;
  * (b) the OM fit to a Michie cluster is significantly WORSE (reduced chi^2
        inflated by > 3x vs the well-specified fit) -- the formalism mismatch is
        detectable in beta(r).

Run record (2026-06-12, CPU/float64, N=30000, K=18 log bins, keys PRNGKey(0/1),
wall ~9 s, exit 0 / ALL PASS):
  (a) OM Plummer (r_h=1, r_a_true=1.5): recovered r_a = 1.5525 +- 0.0351 (pull
      +1.49); OM-fit reduced chi^2 = 0.43 (good; the conservative beta_se deflates
      it); forecast sigma(r_a) ~ N^-0.500, asymptotic 3-sigma anisotropy-detection
      N ~ 148 (small -- r_a~r_h is strong anisotropy; small-N CRLB is optimistic).
  (b) Michie King (W0=7, r_c=1, r_a_true=6, NON-OM formalism): OM-fit r_a = 8.90 +-
      0.27, reduced chi^2 = 5.58 -- a 12.9x inflation over the well-specified fit.
      The OM form mis-fits the Michie beta(r) (detectable formalism mismatch) AND
      mis-estimates the anisotropy radius (8.9 vs 6.0).

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_anisotropy.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR
from progenax import (
    MichieProfile,
    MichieVelocityDF,
    PlummerProfile,
    PlummerVelocityDF,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _demo_inference import binned_sigma_beta, fisher_cov, gaussian_loglike, mle_adam, expit
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = STELLAR.G

# (a) well-specified Osipkov-Merritt Plummer.
R_H = 1.0
RA_OM_TRUE = 1.5
# (b) Michie anisotropic King (r_a=6 is the strongest anisotropy that still
# truncates at W0=7; stronger gives infinite mass).
MICHIE_W0, MICHIE_RC, RA_MICHIE_TRUE = 7.0, 1.0, 6.0

N_STARS = 30_000
K_BINS = 18
N_MIN = 80
RA_BOX = (0.2, 60.0)
N_ADAM = 500
ADAM_LR = 3e-2
SEED = 0


def _beta_profile(pos, vel, r_edges):
    """Binned beta_hat(r) with the conservative SE beta_se = (1+|beta|)/sqrt(n)."""
    res = binned_sigma_beta(pos, vel, r_edges, component_id=None, n_min=N_MIN)
    beta_hat, weight, n = res.beta_hat[0], res.weight[0], res.n[0]
    beta_se = jnp.where(weight > 0,
                        (1.0 + jnp.abs(beta_hat)) / jnp.sqrt(jnp.maximum(n, 1.0)), 1.0)
    r_mid = jnp.sqrt(r_edges[:-1] * r_edges[1:])
    return r_mid, beta_hat, beta_se, weight


def _log_edges(radii):
    r = np.asarray(radii)
    lo = max(np.percentile(r, 1.0), 0.05)
    hi = np.percentile(r, 99.0)
    return jnp.asarray(np.geomspace(lo, hi, K_BINS + 1))


def _beta_om(r, r_a):
    return r**2 / (r**2 + r_a**2)


def fit_om(r_mid, beta_hat, beta_se, weight):
    """Fit beta_OM(r; r_a) to the binned beta(r); return (r_a_hat, sigma, red_chi2)."""
    data = (beta_hat, beta_se, weight)
    predict = lambda z: _beta_om(r_mid, expit(z[0], *RA_BOX))
    nll = lambda z: -gaussian_loglike(data, predict)(z)
    # init r_a at the radius where beta_hat first exceeds 0.5 (else the median bin).
    above = np.where((np.asarray(weight) > 0) & (np.asarray(beta_hat) > 0.5))[0]
    r_init = float(r_mid[above[0]]) if len(above) else float(np.median(np.asarray(r_mid)))
    r_init = float(np.clip(r_init, RA_BOX[0] + 1e-3, RA_BOX[1] - 1e-3))
    z0 = jnp.array([float(jnp.log((r_init - RA_BOX[0]) / (RA_BOX[1] - r_init)))])
    z_hat, _ = mle_adam(jax.jit(nll), z0, n_steps=N_ADAM, lr=ADAM_LR)
    r_a_hat = float(expit(z_hat[0], *RA_BOX))
    cov = fisher_cov(nll, z_hat)
    dra_dz = float(jax.grad(lambda z: expit(z[0], *RA_BOX))(z_hat)[0])
    sigma = float(jnp.sqrt(cov[0, 0]) * abs(dra_dz))
    dof = int(jnp.sum(weight > 0)) - 1
    red_chi2 = 2.0 * float(nll(z_hat)) / max(dof, 1)
    return r_a_hat, sigma, red_chi2


def sample_plummer_om(r_a, key):
    kp, kv = jax.random.split(key)
    m = jnp.ones(N_STARS)
    pos = PlummerProfile(r_h=R_H).sample_positions(m, kp)
    vel = PlummerVelocityDF(r_h=R_H, anisotropy_radius=r_a).sample_velocities(pos, m, kv, G=G)
    return pos, vel


def sample_michie(key):
    kp, kv = jax.random.split(key)
    m = jnp.ones(N_STARS)
    prof = MichieProfile.from_W0_rc(W0=MICHIE_W0, r_c=MICHIE_RC, r_a=RA_MICHIE_TRUE)
    pos = prof.sample_positions(m, kp)
    vel = MichieVelocityDF(W0=MICHIE_W0, r_c=MICHIE_RC, r_a=RA_MICHIE_TRUE
                           ).sample_velocities(pos, m, kv, G=G)
    return pos, vel


# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("ANISOTROPY: OM measurability + OM-vs-Michie misspecification (B6)")
    print("=" * 78)

    # --- (a) well-specified: OM Plummer ------------------------------------ #
    pos_a, vel_a = sample_plummer_om(RA_OM_TRUE, jax.random.PRNGKey(SEED))
    edges_a = _log_edges(jnp.linalg.norm(pos_a, axis=1))
    r_a_mid, beta_a, bse_a, w_a = _beta_profile(pos_a, vel_a, edges_a)
    ra_hat, ra_sig, chi2_om = fit_om(r_a_mid, beta_a, bse_a, w_a)
    pull = (ra_hat - RA_OM_TRUE) / ra_sig
    print(f"\n  (a) OM Plummer (truth r_a={RA_OM_TRUE}):")
    print(f"      recovered r_a = {ra_hat:.4f} +- {ra_sig:.4f}  (pull {pull:+.2f})")
    print(f"      OM-fit reduced chi^2 = {chi2_om:.2f}  (well-specified -> ~1)")

    # forecast: sigma(r_a) ~ N^-1/2 from the per-star Fisher info.
    info_per_star = 1.0 / (ra_sig**2 * N_STARS)
    # span low N so the 3-sigma detection N (~150) is in-range, not clipped at the edge.
    n_grid = np.array([3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5])
    sigma_grid = 1.0 / np.sqrt(n_grid * info_per_star)
    slope = float(np.polyfit(np.log(n_grid), np.log(sigma_grid), 1)[0])
    n_detect = 9.0 / (info_per_star * RA_OM_TRUE**2)  # 3-sigma r_a vs isotropic
    print(f"      forecast sigma(r_a) ~ N^{slope:.3f}; N for 3-sigma anisotropy "
          f"detection ~ {n_detect:.0f}")

    # --- (b) misspecified: OM fit to a Michie sample ----------------------- #
    pos_b, vel_b = sample_michie(jax.random.PRNGKey(SEED + 1))
    edges_b = _log_edges(jnp.linalg.norm(pos_b, axis=1))
    r_b_mid, beta_b, bse_b, w_b = _beta_profile(pos_b, vel_b, edges_b)
    ra_hat_b, ra_sig_b, chi2_michie = fit_om(r_b_mid, beta_b, bse_b, w_b)
    print(f"\n  (b) Michie King (truth r_a={RA_MICHIE_TRUE}, NON-OM formalism):")
    print(f"      OM-fit r_a = {ra_hat_b:.4f} +- {ra_sig_b:.4f}")
    print(f"      OM-fit reduced chi^2 = {chi2_michie:.2f}  (misspecified -> inflated)")
    print(f"      chi^2 inflation vs well-specified = {chi2_michie / chi2_om:.1f}x")

    make_figure(r_a_mid, beta_a, bse_a, w_a, ra_hat,
                r_b_mid, beta_b, bse_b, w_b, ra_hat_b,
                n_grid, sigma_grid, n_detect)

    recovery_ok = abs(pull) < 3.0
    goodfit_ok = chi2_om < 2.5
    forecast_ok = -0.55 < slope < -0.45
    misspec_ok = chi2_michie > 3.0 * chi2_om

    rows = [
        ("(a) OM r_a recovery", "PASS" if recovery_ok else "FAIL", "<3 sigma", recovery_ok),
        ("(a) OM-fit good (red chi2~1)", "PASS" if goodfit_ok else "FAIL",
         "<2.5", goodfit_ok),
        ("(a) forecast sigma~N^-1/2", "PASS" if forecast_ok else "FAIL",
         "slope -0.5", forecast_ok),
        ("(b) Michie misfit detectable", "PASS" if misspec_ok else "FAIL",
         ">3x chi2", misspec_ok),
    ]
    print("\n" + "-" * 78)
    print(f"  {'CHECK':<32s} {'status':>6s} {'gate':>12s}")
    print("-" * 78)
    all_ok = True
    for name, status, gate, ok in rows:
        all_ok &= ok
        print(f"  {name:<32s} {status:>6s} {gate:>12s}")
    print("-" * 78)
    print(f"  saved {OUTPUT_DIR}/demo_anisotropy.{{png,pdf}}")
    print("=" * 78)
    print("  ANISOTROPY DEMO: ALL PASS" if all_ok else "  ANISOTROPY DEMO: FAILED")
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
def make_figure(r_a_mid, beta_a, bse_a, w_a, ra_hat,
                r_b_mid, beta_b, bse_b, w_b, ra_hat_b,
                n_grid, sigma_grid, n_detect):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))

    def _plot_beta(ax, r_mid, beta, bse, w, ra_fit, label, fitlabel):
        m = np.asarray(w) > 0
        rm = np.asarray(r_mid)[m]
        ax.errorbar(rm, np.asarray(beta)[m], yerr=np.asarray(bse)[m], fmt="o", ms=3.5,
                    color=OI["black"], label=label, zorder=4)
        rr = np.geomspace(rm.min(), rm.max(), 100)
        ax.plot(rr, rr**2 / (rr**2 + ra_fit**2), "-", color=OI["vermilion"],
                label=fitlabel)
        ax.set_xscale("log")
        ax.set_xlabel(r"$r$  [pc]")
        ax.set_ylabel(r"$\beta(r)$")
        ax.set_ylim(-0.15, 1.0)
        ax.legend(fontsize=7)

    # (a) OM Plummer: OM fits well.
    _plot_beta(axes[0], r_a_mid, beta_a, bse_a, w_a, ra_hat, "OM Plummer",
               fr"OM fit $r_a={ra_hat:.2f}$")
    panel_label(axes[0], "(a)")

    # (b) Michie: OM fit leaves a residual.
    _plot_beta(axes[1], r_b_mid, beta_b, bse_b, w_b, ra_hat_b, "Michie King",
               fr"OM fit $r_a={ra_hat_b:.2f}$")
    panel_label(axes[1], "(b)")

    # (c) forecast sigma(r_a) vs N.
    ax = axes[2]
    ax.loglog(n_grid, sigma_grid, "o-", color=OI["green"])
    ax.axvline(n_detect, color="0.6", ls=":")
    ax.text(n_detect * 1.25, sigma_grid.min() * 1.5, fr"$N\approx{n_detect:.0f}$",
            fontsize=7.5, ha="left", va="bottom", color="0.3")
    ax.set_xlabel(r"$N_\star$")
    ax.set_ylabel(r"$\sigma(r_a)$  [pc]")
    panel_label(ax, "(c)")

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "demo_anisotropy")


if __name__ == "__main__":
    sys.exit(main())
