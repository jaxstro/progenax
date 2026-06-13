#!/usr/bin/env python
r"""B4 -- Unresolved-binary mass-function distortion: f_b recovery (Batch C).

The headline binary demo. Unresolved binaries blend into a single photometric
source whose inferred mass is biased HIGH, distorting the observed stellar mass
function. This demo recovers the binary fraction f_b from that distortion, and
shows the central point: getting f_b right requires the correct Moe & Di Stefano
(2017) period-mass-ratio COUPLING, because the BLENDED (close) binaries are a
biased, high-q subset that an independent-(P,q) model mis-models.

Physics
-------
Each binary has a primary m1 (from the IMF), a mass ratio q (so m2 = q m1) and a
period P; the semimajor axis a follows Kepler's third law. A survey RESOLVES wide
pairs and BLENDS close ones (separation below the resolution limit, a < a_crit).

  * Resolved binary  -> two catalogue stars at m1 and m2.
  * Blended binary   -> ONE catalogue star at m_obs, where the Tout et al. (1996)
                        ZAMS relation gives L_obs = L(m1) + L(m2) and
                        m_obs = L^{-1}(L_obs) > m1 (the photometric blend mass).
  * Single star      -> one catalogue star at m1.

The Tout ZAMS mass-luminosity relation is imported from the **fluxax** sibling
(``fluxax.photometry``); progenax is decoupled from that private package, so the
demo guards the import (install locally: ``uv pip install -e ../fluxax --no-deps``).

Inference
---------
The observed mass function is a linear mixture in f_b,
``mu_k(f_b) = N_sys [ (1 - f_b) S_k + f_b B_k ]``, with the single template
``S_k`` (the IMF) and the binary template ``B_k`` (per-binary expected catalogue
stars after resolution + blending) precomputed from a large pool. The data are the
frozen binned counts of a Moe-coupled truth cluster; f_b is recovered by a
per-bin Poisson MLE (``poisson_loglike``), differentiable in f_b, with the Fisher
variance from ``fisher_cov``.

The coupling test (the punchline)
---------------------------------
The independent comparison uses the SAME pool with q SHUFFLED against P -- this
preserves the q and P marginals exactly and changes only the P-q correlation. The
truth is Moe-coupled; fitting f_b with the (correct) Moe template recovers it,
while fitting with the (wrong) independent template is BIASED, because the blended
short-period subset is more equal-mass (higher q) under Moe than an independent
model assumes, so it over/under-predicts the blend bump.

Gates (exit 0 = all pass):
  * mechanism: the blended subset's median q is higher under Moe than independent;
  * self-consistency: the Moe template at f_b_true matches the data (Poisson);
  * f_b recovery (Moe template) within 3 sigma of truth;
  * BIAS (headline): the independent-template f_b is biased by > 3 sigma.

Run record (2026-06-12, CPU/float64, N_sys=3e5, N_pool=6e5, a_crit=50 AU, Z=0.02,
f_b_true=0.5, key PRNGKey(0/1), wall ~10 s, exit 0 / ALL PASS):
  mechanism: blended-subset median q  Moe = 0.540  vs  independent = 0.479
    (Moe couples close/short-period binaries to higher q -> more equal-mass blends).
  f_b recovery: Moe template (correct) 0.5016 +- 0.0074 (pull +0.22); independent
    template (wrong) 0.4749 +- 0.0070 -> a -3.6 sigma, ~5% LOW bias. The wrong-model
    f_b error is a systematic; its significance grows as sqrt(N_sys) (only -1.5 sigma
    at N_sys=5e4, -3.6 sigma at the 3e5 survey scale here).
  self-consistency (Moe template @ truth): max|N_k-mu_k|/sqrt(mu_k) = 2.76 (< 4).

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_binary_mass_function.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR
from progenax.binaries import period_to_semimajor_axis
from progenax.binaries.eccentricity import MoeEccentricity
from progenax.imf import Maschberger
from progenax.imf.binary import MoeDiStefano2017Full, MoeJointOrbit, MoePeriod

try:
    from fluxax.photometry import inverse_zams_luminosity, zams_luminosity
except ImportError as exc:  # pragma: no cover - local-only demo dependency
    raise SystemExit(
        "B4 needs the Tout+1996 ZAMS relations from the (private) fluxax sibling.\n"
        "Install it editable:  uv pip install -e ../fluxax --no-deps"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _demo_inference import expit, fisher_cov, mle_adam, poisson_loglike
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = STELLAR.G
DAY = 86400.0 / STELLAR.time_scale_cgs  # day in code (Myr) time units

# --- configuration ---------------------------------------------------------- #
# The independent-model f_b error is a SYSTEMATIC (wrong P-q coupling): the
# fractional bias is ~fixed while its significance grows as sqrt(N_sys). N_sys is set
# to a representative cluster-survey scale where the systematic is unambiguous; the
# pool is larger so template noise stays below the data Poisson noise.
N_POOL = 600_000          # template pool (low template noise)
N_SYS = 300_000           # systems in the mock data cluster (survey scale)
F_B_TRUE = 0.5
Z_MET = 0.02              # metallicity for the Tout ZAMS photometry (solar)
A_CRIT_AU = 50.0          # blend pairs with separation a < 50 AU (survey resolution)
A_CRIT = A_CRIT_AU / 206265.0  # AU -> pc (the units of period_to_semimajor_axis here)
IMF = Maschberger(alpha=2.3, m_min=0.08, m_max=100.0)
JOINT = MoeJointOrbit(period=MoePeriod(),
                      massratio=MoeDiStefano2017Full(q_min=0.1),
                      eccentricity=MoeEccentricity())

M_BINS = np.geomspace(0.08, 160.0, 31)
FB_BOX = (0.05, 0.95)
N_ADAM = 400
ADAM_LR = 3e-2
SEED = 0


def _draw_pool(key, n):
    """m1 (IMF), and Moe-coupled (P_days, q) for n systems."""
    km, kj = jax.random.split(key)
    m1 = np.asarray(IMF.sample(km, n))
    P_days, q, _e = JOINT.sample(kj, jnp.asarray(m1))
    return m1, np.asarray(P_days), np.asarray(q)


def _semimajor_axis(m1, m2, P_days):
    return np.asarray(period_to_semimajor_axis(
        jnp.asarray(P_days) * DAY, jnp.asarray(m1 + m2), G))  # pc


# inverse_zams_luminosity uses jax.grad(zams_luminosity) internally (scalar-only),
# so map it over the luminosity array (per the fluxax docstring).
_inv_zams_L_vec = jax.vmap(inverse_zams_luminosity, in_axes=(0, None))


def _blend_mass(m1, m2):
    L = zams_luminosity(jnp.asarray(m1), Z_MET) + zams_luminosity(jnp.asarray(m2), Z_MET)
    return np.asarray(_inv_zams_L_vec(L, Z_MET))


def _binary_catalogue_masses(m1, q, P_days):
    """Observed catalogue masses of a pure-binary population (resolution+blend)."""
    m2 = q * m1
    a = _semimajor_axis(m1, m2, P_days)
    blended = a < A_CRIT
    m_obs = _blend_mass(m1, m2)
    resolved = ~blended
    return np.concatenate([m_obs[blended], m1[resolved], m2[resolved]]), blended


def _hist(masses):
    return np.histogram(masses, bins=M_BINS)[0].astype(float)


def build_templates(key):
    """Single template S_k (per single) and Moe / independent binary templates B_k."""
    m1, P_days, q = _draw_pool(key, N_POOL)
    S_k = _hist(m1) / N_POOL                                  # 1 star per single (IMF)
    moe_cat, moe_blended = _binary_catalogue_masses(m1, q, P_days)
    q_shuf = q[np.random.default_rng(SEED).permutation(N_POOL)]  # break P-q coupling
    ind_cat, ind_blended = _binary_catalogue_masses(m1, q_shuf, P_days)
    B_moe = _hist(moe_cat) / N_POOL
    B_ind = _hist(ind_cat) / N_POOL
    # mechanism diagnostic: median q of the BLENDED subset, Moe vs independent.
    q_blend_moe = float(np.median(q[moe_blended]))
    q_blend_ind = float(np.median(q_shuf[ind_blended]))
    return S_k, B_moe, B_ind, q_blend_moe, q_blend_ind


def build_data(key):
    """A Moe-coupled mock cluster at F_B_TRUE -> frozen observed-MF counts."""
    m1, P_days, q = _draw_pool(key, N_SYS)
    kbin = jax.random.PRNGKey(SEED + 7)
    is_binary = np.asarray(jax.random.uniform(kbin, (N_SYS,)) < F_B_TRUE)
    m2 = q * m1
    a = _semimajor_axis(m1, m2, P_days)
    blended = a < A_CRIT
    m_obs = _blend_mass(m1, m2)
    cat = [m1[~is_binary]]                                    # singles
    bb = is_binary & blended
    cat.append(m_obs[bb])                                     # blended binaries
    br = is_binary & ~blended
    cat.extend([m1[br], m2[br]])                              # resolved binaries
    return _hist(np.concatenate(cat))


# --------------------------------------------------------------------------- #
def _fit_fb(counts, S_k, B_k):
    """Poisson MLE of f_b for mu_k = N_sys[(1-f_b)S_k + f_b B_k]."""
    S = jnp.asarray(S_k)
    B = jnp.asarray(B_k)
    counts_j = jnp.asarray(counts)

    def predict_mu(z):
        fb = expit(z[0], *FB_BOX)
        return N_SYS * ((1.0 - fb) * S + fb * B)

    nll = lambda z: -poisson_loglike((counts_j, jnp.ones_like(counts_j)), predict_mu)(z)
    z0 = jnp.array([float(jnp.log((F_B_TRUE - FB_BOX[0]) / (FB_BOX[1] - F_B_TRUE)))])
    z_hat, _ = mle_adam(jax.jit(nll), z0, n_steps=N_ADAM, lr=ADAM_LR)
    fb_hat = float(expit(z_hat[0], *FB_BOX))
    cov = fisher_cov(nll, z_hat)
    dfb_dz = float(jax.grad(lambda z: expit(z[0], *FB_BOX))(z_hat)[0])
    sigma = float(jnp.sqrt(cov[0, 0]) * abs(dfb_dz))
    return fb_hat, sigma, predict_mu(z_hat)


def main():
    print("=" * 78)
    print("UNRESOLVED-BINARY MASS FUNCTION (B4): recover f_b; Moe coupling matters")
    print("=" * 78)
    print(f"\n  N_sys={N_SYS}, N_pool={N_POOL}, f_b_true={F_B_TRUE}, "
          f"a_crit={A_CRIT_AU} AU, Z={Z_MET}")

    S_k, B_moe, B_ind, q_blend_moe, q_blend_ind = build_templates(jax.random.PRNGKey(SEED))
    counts = build_data(jax.random.PRNGKey(SEED + 1))
    print(f"  blended-subset median q:  Moe = {q_blend_moe:.3f}   "
          f"independent = {q_blend_ind:.3f}")

    fb_moe, sig_moe, mu_moe = _fit_fb(counts, S_k, B_moe)
    fb_ind, sig_ind, mu_ind = _fit_fb(counts, S_k, B_ind)
    pull_moe = (fb_moe - F_B_TRUE) / sig_moe
    bias_ind = (fb_ind - F_B_TRUE) / sig_ind
    print(f"\n  f_b recovery:")
    print(f"    Moe template (correct):     {fb_moe:.4f} +- {sig_moe:.4f}  "
          f"(pull {pull_moe:+.2f})")
    print(f"    independent template (wrong): {fb_ind:.4f} +- {sig_ind:.4f}  "
          f"(bias {bias_ind:+.2f} sigma)")

    # self-consistency: Moe template at truth vs data (Poisson residual).
    mu_truth = N_SYS * ((1 - F_B_TRUE) * S_k + F_B_TRUE * np.asarray(B_moe))
    sc = float(np.max(np.abs(counts - mu_truth) / np.sqrt(np.maximum(mu_truth, 1.0))))
    print(f"  self-consistency (Moe@truth): max|N_k-mu_k|/sqrt(mu_k) = {sc:.2f}")

    make_figure(counts, S_k, B_moe, B_ind, mu_moe, mu_ind,
                fb_moe, sig_moe, fb_ind, sig_ind, q_blend_moe, q_blend_ind)

    mech_ok = q_blend_moe > q_blend_ind + 0.02
    selfcon_ok = sc < 4.0
    recovery_ok = abs(pull_moe) < 3.0
    bias_ok = abs(bias_ind) > 3.0

    rows = [
        ("mechanism: blended q (Moe>ind)", "PASS" if mech_ok else "FAIL",
         "coupling", mech_ok),
        ("self-consistency (Moe@truth)", "PASS" if selfcon_ok else "FAIL",
         "<4 sigma", selfcon_ok),
        ("f_b recovery (Moe template)", "PASS" if recovery_ok else "FAIL",
         "<3 sigma", recovery_ok),
        ("f_b BIAS (independent)", "PASS" if bias_ok else "FAIL",
         ">3 sigma", bias_ok),
    ]
    print("\n" + "-" * 78)
    print(f"  {'CHECK':<32s} {'status':>6s} {'gate':>14s}")
    print("-" * 78)
    all_ok = True
    for name, status, gate, ok in rows:
        all_ok &= ok
        print(f"  {name:<32s} {status:>6s} {gate:>14s}")
    print("-" * 78)
    print(f"  saved {OUTPUT_DIR}/demo_binary_mass_function.{{png,pdf}}")
    print("=" * 78)
    print("  BINARY MASS FUNCTION DEMO: ALL PASS" if all_ok
          else "  BINARY MASS FUNCTION DEMO: FAILED")
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
def make_figure(counts, S_k, B_moe, B_ind, mu_moe, mu_ind,
                fb_moe, sig_moe, fb_ind, sig_ind, q_blend_moe, q_blend_ind):
    import matplotlib.pyplot as plt

    cen = np.sqrt(M_BINS[:-1] * M_BINS[1:])
    dlogm = np.diff(np.log10(M_BINS))
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))

    # (a) the distorted observed MF: data + single & binary components.
    ax = axes[0]
    ax.step(cen, counts / dlogm, where="mid", color=OI["black"], label="observed")
    ax.plot(cen, N_SYS * (1 - F_B_TRUE) * S_k / dlogm, ":", color=OI["blue"],
            label="singles")
    ax.plot(cen, N_SYS * F_B_TRUE * np.asarray(B_moe) / dlogm, "--",
            color=OI["vermilion"], label="binaries (blended+resolved)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{\rm obs}$  [$M_\odot$]")
    ax.set_ylabel(r"$dN/d\log m$")
    ax.set_ylim(bottom=max(1.0, counts.max() / 3e3))
    ax.legend(fontsize=7)
    panel_label(ax, "(a)")

    # (b) the mechanism: blended-subset q, Moe (coupled, high-q) vs independent.
    ax = axes[1]
    ax.bar([0, 1], [q_blend_moe, q_blend_ind], 0.55,
           color=[OI["vermilion"], OI["sky"]])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Moe\n(coupled)", "independent\n(shuffled q)"])
    ax.set_ylabel(r"median $q$ of blended subset")
    ax.set_ylim(0, 1)
    panel_label(ax, "(b)")

    # (c) f_b recovery: Moe (unbiased) vs independent (biased).
    ax = axes[2]
    ax.axhline(F_B_TRUE, color="0.6", ls="--", label="truth")
    ax.errorbar([0], [fb_moe], yerr=[sig_moe], fmt="o", ms=6, color=OI["vermilion"],
                label="Moe (correct)")
    ax.errorbar([1], [fb_ind], yerr=[sig_ind], fmt="s", ms=6, color=OI["sky"],
                label="independent (wrong)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Moe", "independent"])
    ax.set_ylabel(r"recovered $f_b$")
    ax.set_xlim(-0.5, 1.5)
    ax.legend(fontsize=7)
    panel_label(ax, "(c)")

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "demo_binary_mass_function")


if __name__ == "__main__":
    sys.exit(main())
