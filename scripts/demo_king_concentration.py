#!/usr/bin/env python
r"""B11 -- King concentration from star counts: (W0, r_c) recovery (Batch C).

The methods showcase for the NEW Poisson number-density channel
(``_demo_inference.binned_number_density`` / ``poisson_loglike`` /
``poisson_fisher_information``). Every other demo in this series (B2, B3, and the
anisotropy demo B6) recovers structure from the *velocity* channel sigma(r); this
one recovers a single-population King model's concentration from the **radial star
counts alone** -- the observable a photometric survey actually delivers.

Physics
-------
A King (1966) lowered-isothermal model is fixed by two structural parameters:

  * ``W0`` -- the dimensionless central potential (concentration). It sets the
    SHAPE of the projected/3-D number-density profile in units of the core radius,
    and the truncation ratio r_t / r_c (the King concentration c = log10(r_t/r_c)).
  * ``r_c`` -- the core radius, the physical SCALE at which the profile breaks.

So binned radial counts $N_k$ constrain BOTH: the shape pins ``W0`` and the
physical break radius pins ``r_c``. The forward model is Engine A (a single
King-limit LIMEPY component, ``g=1``): the coupled Poisson ODE gives psi(r/r_c),
and the King volume density is the closed form
``n_hat(r) = E_gamma(g + 3/2, W(r))`` (``limepy_density_hat``), differentiable in
(W0, r_c) through ``psi_grid``. Beyond r_t the potential W <= 0 and the density is
exactly 0 -- the hard truncation that pins the outer profile.

Likelihood
----------
The expected counts are ``mu_k(W0, r_c) = N_obs * p_k``, with ``p_k`` the fraction
of the model's enclosed number in radial bin k (number-weighted integral of
``4 pi r^2 n_hat(r)`` over the bin, normalized over the binned range). The data are
the FROZEN counts; gradients flow through the model only. Per-bin Poisson
log-likelihood ``sum_k (N_k log mu_k - mu_k)`` -- the right model for low-occupancy
OUTER bins, where the truncation (hence W0) is most constrained. Fisher information
is the reverse-mode Poisson form ``J^T diag(1/mu) J`` (``poisson_fisher_information``):
the Engine A diffrax solve carries a ``custom_vjp``, so ``jax.hessian`` would crash
(same reason B2 uses Gauss-Newton).

Recovered: ``(W0, r_c)``; reported: the King concentration ``c = log10(r_t/r_c)``
at the MLE (the model's c(W0) is validated against King 1966 Table II on the
[King validation page]).

No external rescale; the gates ARE the contract (exit 0 = all pass):
  * self-consistency: predict(truth) matches the data within 4 sigma Poisson per bin;
  * MLE recovery: |theta_hat - theta_true| < 3 sigma_hat for both parameters;
  * plateau: the loss tail is converged;
  * Fisher PD: the information matrix is positive definite.

Run record (2026-06-12, CPU/float64, N=30000, K=20 log bins [0.10, 16.81] pc,
keys PRNGKey(0)/(1), wall ~29 s, exit 0 / ALL PASS):
  self-consistency max|N_k - mu_k|/sqrt(mu_k) = 2.51 (< 4).
  W0 : 6.000 -> 6.0275 +- 0.0239 (pull +1.15)
  r_c: 1.000 -> 0.9952 +- 0.0102 (pull -0.47)
  3 dispersed inits -> identical loss -202978.0336 (robust optimum).
  King concentration c = log10(r_t/r_c): MLE 1.263 vs truth 1.255 (King 1966 Tab. II).
  Fisher rho(W0, r_c) = -0.911 -- the count channel alone is strongly W0-r_c
  DEGENERATE (a more concentrated model with a smaller core mimics the same count
  profile); both marginals are still tight, but the velocity channel (B2/B3) is
  what breaks the correlation.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_king_concentration.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR

from progenax import KingProfile
from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.profiles.limepy import limepy_density_hat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _demo_inference import (
    binned_number_density,
    constrained_cov,
    expit,
    mle_adam,
    poisson_fisher_information,
    poisson_loglike,
)
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = STELLAR.G

# --- truth + fit configuration ---------------------------------------------- #
W0_TRUE, RC_TRUE = 6.0, 1.0  # GC-like concentration (c ~ 1.3); core radius 1 pc
N_STARS = 30_000
SEED = 0

K_BINS = 20  # log-spaced radial bins
R_LO = 0.10  # inner edge [pc] (excludes the flat-core centre, ~0.1% mass)
N_FINE = 3000  # predict-side radial integration grid

W0_BOX = (3.5, 7.5)  # from_components diffrax ODE hits max_steps above ~8
RC_BOX = (0.3, 3.0)  # (B2 used (3,8) for W0_true=5; truth 6 here, capped at 7.5)

N_INITS = 3
N_ADAM = 400
ADAM_LR = 3e-2
SELFCON_NSIG = 4.0  # self-consistency gate: |N_k - mu_k|/sqrt(mu_k)
RECOVERY_NSIG = 3.0


def _king_model(W0, r_c):
    """Single King-limit (g=1) Engine A component; differentiable in (W0, r_c)."""
    one = jnp.array([1.0])
    return MultiComponentCluster.from_components(
        alpha_j=one, w_j=one, m_j=one, W0=W0, g=1.0, r_c=r_c
    )


def _theta_of_z(z):
    return expit(z[0], *W0_BOX), expit(z[1], *RC_BOX)


def _dtheta_dz(z):
    """Diagonal Jacobian of the box reparametrization (for the delta method)."""
    s0, s1 = jax.nn.sigmoid(z[0]), jax.nn.sigmoid(z[1])
    return jnp.array(
        [
            (W0_BOX[1] - W0_BOX[0]) * s0 * (1.0 - s0),
            (RC_BOX[1] - RC_BOX[0]) * s1 * (1.0 - s1),
        ]
    )


# --------------------------------------------------------------------------- #
# Forward model: expected counts mu_k(W0, r_c)
# --------------------------------------------------------------------------- #
def make_predict_counts(r_edges, n_obs):
    """Differentiable expected-count predictor on FIXED bin edges.

    p_k = (enclosed number in bin k) / (enclosed over the binned range); the
    enclosed profile is the cumulative trapezoid of 4 pi r^2 n_hat(r) on a fixed
    fine grid. Differentiable in z through the Engine A psi_grid."""
    r_grid = jnp.linspace(float(r_edges[0]), float(r_edges[-1]), N_FINE)

    def predict_mu(z):
        W0, r_c = _theta_of_z(z)
        m = _king_model(W0, r_c)
        W_r = jnp.interp(r_grid, m.xi_grid * r_c, m.psi_grid, left=W0, right=0.0)
        n_hat = limepy_density_hat(W_r, 1.0)  # King volume density (g=1)
        integrand = 4.0 * jnp.pi * r_grid**2 * n_hat
        incr = 0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(r_grid)
        cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(incr)])  # enclosed(r) on r_grid
        enc_edges = jnp.interp(r_edges, r_grid, cum)
        p_k = (enc_edges[1:] - enc_edges[:-1]) / cum[-1]
        return n_obs * p_k

    return predict_mu


# --------------------------------------------------------------------------- #
# Mock data
# --------------------------------------------------------------------------- #
def build_truth_data():
    """Sample the truth King, freeze the log-spaced bin edges + counts."""
    model = _king_model(W0_TRUE, RC_TRUE)
    ic = model.sample_cluster(jax.random.PRNGKey(SEED), n_stars=N_STARS, G=G)
    radii = jnp.linalg.norm(ic.positions, axis=1)
    r_t = float(model.xi_grid[-1] * RC_TRUE)  # outer ODE radius (psi -> 0 at r_t)
    r_hi = float(jnp.max(radii))
    r_edges = jnp.geomspace(R_LO, min(r_hi, r_t) * 1.001, K_BINS + 1)
    counts = binned_number_density(ic.positions, r_edges)
    n_obs = float(jnp.sum(counts))
    return r_edges, counts, n_obs


def dispersed_inits(key):
    z_true = jnp.array(
        [
            float(jnp.log((W0_TRUE - W0_BOX[0]) / (W0_BOX[1] - W0_TRUE))),
            float(jnp.log((RC_TRUE - RC_BOX[0]) / (RC_BOX[1] - RC_TRUE))),
        ]
    )
    noise = jax.random.normal(key, (N_INITS - 1, 2)) * 0.3
    return jnp.concatenate([z_true[None, :], z_true[None, :] + noise], axis=0)


def run_mle(negloglike, key):
    inits = dispersed_inits(key)
    nll_jit = jax.jit(negloglike)
    finals, traces = [], []
    for z0 in inits:
        z_hat, trace = mle_adam(nll_jit, z0, n_steps=N_ADAM, lr=ADAM_LR)
        finals.append(z_hat)
        traces.append(trace)
    losses = [float(nll_jit(z)) for z in finals]
    i_best = int(np.argmin(losses))
    return finals[i_best], traces[i_best], losses


def plateau_ok(trace):
    t = np.asarray(trace)
    total = t[0] - t[-1]
    tail = t[int(0.9 * len(t))] - t[-1]
    return tail < 0.01 * total if total > 0 else True


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(r_edges, counts, predict_mu, z_hat, theta_hat, sigma_theta, cov):
    import matplotlib.pyplot as plt

    r_edges_np = np.asarray(r_edges)
    r_mid = np.sqrt(r_edges_np[:-1] * r_edges_np[1:])
    n_obs = float(jnp.sum(counts))
    mu_hat = np.asarray(predict_mu(z_hat))
    mu_true = np.asarray(
        make_predict_counts(r_edges, n_obs)(_z_of_theta(W0_TRUE, RC_TRUE))
    )
    counts_np = np.asarray(counts)
    shell = 4.0 / 3.0 * np.pi * (r_edges_np[1:] ** 3 - r_edges_np[:-1] ** 3)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    # (a) number-density profile: data counts/shell-volume vs fit + truth.
    ax = axes[0]
    ax.errorbar(
        r_mid,
        counts_np / shell,
        yerr=np.sqrt(counts_np) / shell,
        fmt="o",
        ms=3.5,
        color=OI["black"],
        label="counts",
        zorder=4,
    )
    ax.plot(r_mid, mu_hat / shell, "-", color=OI["vermilion"], label="MLE")
    ax.plot(r_mid, mu_true / shell, "--", color=OI["blue"], label="truth")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r$  [pc]")
    ax.set_ylabel(r"number density  [stars pc$^{-3}$]")
    ax.legend()
    panel_label(ax, "(a)")

    # (b) 2-sigma Fisher ellipse in (W0, r_c).
    ax = axes[1]
    th = np.linspace(0, 2 * np.pi, 200)
    circ = np.stack([np.cos(th), np.sin(th)])
    L = np.linalg.cholesky(np.asarray(cov))
    ell = theta_hat[:, None] + 2.0 * L @ circ
    ax.plot(ell[0], ell[1], "-", color=OI["purple"], label=r"$2\sigma$ Fisher")
    ax.scatter(
        [theta_hat[0]],
        [theta_hat[1]],
        marker="o",
        color=OI["vermilion"],
        zorder=5,
        label="MLE",
    )
    ax.scatter(
        [W0_TRUE],
        [RC_TRUE],
        marker="*",
        s=90,
        color=OI["blue"],
        zorder=5,
        label="truth",
    )
    ax.set_xlabel(r"$W_0$")
    ax.set_ylabel(r"$r_c$  [pc]")
    ax.legend()
    panel_label(ax, "(b)")

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "demo_king_concentration")


def _z_of_theta(W0, r_c):
    return jnp.array(
        [
            float(jnp.log((W0 - W0_BOX[0]) / (W0_BOX[1] - W0))),
            float(jnp.log((r_c - RC_BOX[0]) / (RC_BOX[1] - r_c))),
        ]
    )


def _concentration(W0, r_c):
    """King c = log10(r_t / r_c) at concrete (W0, r_c) via the self-consistent r_t."""
    prof = KingProfile.from_W0_rc(W0=float(W0), r_c=float(r_c))
    return float(jnp.log10(prof.r_t / r_c))


# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("KING CONCENTRATION FROM STAR COUNTS (B11): recover (W0, r_c)")
    print("(units: STELLAR -- lengths pc; Poisson number-density channel)")
    print("=" * 78)

    r_edges, counts, n_obs = build_truth_data()
    print(
        f"\n  truth W0={W0_TRUE}, r_c={RC_TRUE} pc; N={N_STARS}, "
        f"binned N_obs={n_obs:.0f} over {K_BINS} log bins "
        f"[{float(r_edges[0]):.2f}, {float(r_edges[-1]):.2f}] pc"
    )

    predict_mu = make_predict_counts(r_edges, n_obs)
    negloglike = lambda z: (
        -poisson_loglike((counts, jnp.ones_like(counts)), predict_mu)(z)
    )

    # self-consistency at truth (before the optimizer).
    z_true = _z_of_theta(W0_TRUE, RC_TRUE)
    mu_true = np.asarray(predict_mu(z_true))
    resid = (np.asarray(counts) - mu_true) / np.sqrt(np.maximum(mu_true, 1.0))
    selfcon = float(np.max(np.abs(resid)))
    selfcon_ok = selfcon < SELFCON_NSIG
    print(
        f"\n  self-consistency: max|N_k - mu_k|/sqrt(mu_k) = {selfcon:.2f}  "
        f"(gate < {SELFCON_NSIG})"
    )

    z_hat, trace, losses = run_mle(negloglike, jax.random.PRNGKey(SEED + 1))
    W0_hat, rc_hat = _theta_of_z(z_hat)
    theta_hat = jnp.array([W0_hat, rc_hat])

    F_z = poisson_fisher_information(predict_mu, z_hat)
    cov = constrained_cov(F_z, _dtheta_dz(z_hat))
    sigma_theta = jnp.sqrt(jnp.diag(cov))

    truth = jnp.array([W0_TRUE, RC_TRUE])
    pulls = (theta_hat - truth) / sigma_theta
    print(f"\n  losses (per init): {[round(x, 4) for x in losses]}")
    print(
        f"  {'param':>6s} {'truth':>8s} {'theta_hat':>12s} {'sigma':>10s} {'pull':>8s}"
    )
    names = ["W0", "r_c"]
    for i in range(2):
        print(
            f"  {names[i]:>6s} {float(truth[i]):>8.3f} {float(theta_hat[i]):>12.4f} "
            f"{float(sigma_theta[i]):>10.4f} {float(pulls[i]):>8.2f}"
        )

    c_hat = _concentration(float(W0_hat), float(rc_hat))
    c_true = _concentration(W0_TRUE, RC_TRUE)
    print(
        f"\n  King concentration c = log10(r_t/r_c): MLE {c_hat:.3f}  vs truth {c_true:.3f}"
    )
    print(
        f"  Fisher rho(W0, r_c) = {float(cov[0, 1] / jnp.sqrt(cov[0, 0] * cov[1, 1])):.3f}"
    )

    make_figure(
        r_edges,
        counts,
        predict_mu,
        z_hat,
        np.asarray(theta_hat),
        np.asarray(sigma_theta),
        cov,
    )

    recovery_ok = bool(jnp.all(jnp.abs(pulls) < RECOVERY_NSIG))
    plat_ok = plateau_ok(trace)
    fisher_ok = bool(jnp.all(jnp.linalg.eigvalsh(cov) > 0))

    rows = [
        (
            "self-consistency at truth",
            "PASS" if selfcon_ok else "FAIL",
            f"< {SELFCON_NSIG} sigma",
            selfcon_ok,
        ),
        (
            "MLE recovery (both params)",
            "PASS" if recovery_ok else "FAIL",
            f"< {RECOVERY_NSIG} sigma",
            recovery_ok,
        ),
        ("loss plateau", "PASS" if plat_ok else "FAIL", "tail<1%", plat_ok),
        ("Fisher covariance PD", "PASS" if fisher_ok else "FAIL", "PD", fisher_ok),
    ]

    print("\n" + "-" * 78)
    print(f"  {'CHECK':<30s} {'status':>6s} {'gate':>16s}")
    print("-" * 78)
    all_ok = True
    for name, status, gate, ok in rows:
        all_ok &= ok
        print(f"  {name:<30s} {status:>6s} {gate:>16s}")
    print("-" * 78)
    print(f"  saved {OUTPUT_DIR}/demo_king_concentration.{{png,pdf}}")
    print("=" * 78)
    print(
        "  KING CONCENTRATION DEMO: ALL PASS"
        if all_ok
        else "  KING CONCENTRATION DEMO: FAILED"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
