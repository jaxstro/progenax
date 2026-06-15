r"""B12 -- The binary-inflated dynamical mass: a "confidently wrong" virial mass.

Unresolved binaries inflate a stellar system's line-of-sight velocity dispersion,
biasing the **virial / dynamical mass high**. In low-dispersion systems
(ultra-faint dwarfs, low-mass GCs) this is a *large fractional* effect that has
driven real debates about M/L ratios and dark-matter content. This demo shows:

  (1) the bias is real -- ``sigma_obs > sigma_true`` and ``M_naive`` is inflated;
  (2) a **dispersion-only** analysis cannot remove it -- the ``(sigma_true, f_b)``
      problem is rank-1 degenerate (one number can't separate two parameters);
  (3) a **differentiable joint recovery** from the non-Gaussian *wings* of the
      velocity distribution returns an unbiased dynamical mass, with a
      Fisher/CRLB forecast vs sample size ``N`` and RV precision ``eps``.

Forward model: one isotropic single-population cluster gives each star a LOS COM
velocity ``v_COM ~ N(0, sigma_true^2)``. A fraction ``f_b`` are unresolved Moe &
Di Stefano (2017) binaries whose observed velocity is the ZAMS-flux-weighted SB2
blend ``v_obs = v_COM + Delta`` (``Delta`` from the sigma-independent kernel
K_orb). Per-star RV noise ``N(0, eps^2)`` is added to every star.

This is the kinematic companion to the B4 unresolved-binary mass-function demo
(B4 measures f_b photometrically; B12 measures the dynamical mass kinematically).

Gates (CLI exits 0 iff all pass):
  1. Bias       -- sigma_obs > sigma_true; M_naive/M_true > 1.10 at f_b=0.5.
  2. Degeneracy -- dispersion-only Fisher is rank-1 (near-singular).
  3. Recovery   -- joint (sigma_true, f_b) MLE within 3sigma; M_dyn unbiased;
                   full-distribution Fisher well-conditioned.
  4. eps-floor  -- bias-removal degrades monotonically as RV precision worsens.
  5. Null       -- f_b=0 gives no bias and recovered f_b ~ 0.
  6. AD-vs-FD   -- the mixture Jacobian d mu / d z matches finite differences.

Demo only: scripts/ + docs/ (no src/progenax change). Units: ALL velocities km/s.

References:
  Moe & Di Stefano (2017) ApJS 230, 15  -- the P-q-e binary statistics.
  Tout et al. (1996) MNRAS 281, 257     -- the ZAMS mass-luminosity relation.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import progenax  # noqa: F401  -- enables float64 at import

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _demo_binaries import (  # noqa: E402
    build_korb_kernel,
    dyn_mass_ratio,
    predict_vlos_counts,
    sample_blend_velocities,
    _kernel_std,
)
from _demo_inference import (  # noqa: E402
    constrained_cov,
    expit,
    logit,
    mle_adam,
    poisson_fisher_information,
    poisson_loglike,
)
from _plotstyle import OI, apply_pub_style, panel_label, save_fig  # noqa: E402

apply_pub_style()

OUTPUT_DIR = "validation/plots"

# --- configuration ---------------------------------------------------------- #
# UFD-like regime: a low-dispersion, metal-poor system where unresolved binaries
# are a large fractional contaminant of the velocity dispersion.
SIGMA_TRUE = 5.0          # true cluster LOS velocity dispersion [km/s]
Z_MET = 1e-3              # metallicity for the Tout ZAMS photometry (metal-poor)
F_B_TRUE = 0.5            # unresolved binary fraction
N_STARS = 1500            # RV stars in the mock survey
EPS_KMS = 1.0             # per-star RV measurement precision [km/s]
R_H_PC = 30.0             # half-mass radius [pc] (sets the virial-mass scale)

# Binned-likelihood velocity grid: wide enough (~+/-10 sigma_obs) to hold the
# non-Gaussian binary wings that break the degeneracy.
V_EDGES = np.linspace(-60.0, 60.0, 121)   # 120 bins of 1 km/s
N_POOL = 200_000          # K_orb template pool (low template noise)
KORB_GRID_MAX = 150.0     # K_orb grid half-width [km/s] (must span the wings)
KORB_N_GRID = 601

# Parameter boxes for the bounded MLE (logit/expit reparametrization).
SIGMA_BOX = (0.5, 30.0)
FB_BOX = (0.0, 0.95)
N_ADAM = 600
ADAM_LR = 3e-2
SEED = 0


def build_mock_vlos(key, f_b=F_B_TRUE, sigma_true=SIGMA_TRUE, eps=EPS_KMS,
                    n_stars=N_STARS, Z=Z_MET):
    r"""Mock observed LOS velocities of a binary-contaminated cluster [km/s].

    ``n_b = round(f_b * n_stars)`` stars are unresolved binaries: their observed
    velocity is the cluster COM draw plus the flux-weighted blend ``Delta`` (drawn
    fresh from the Moe+ZAMS machinery, NOT from the histogram kernel). Every star
    then gets independent ``N(0, eps^2)`` RV measurement noise.

    Returns ``v_obs`` of shape ``(n_stars,)``.
    """
    k_com, k_binary, k_noise = jax.random.split(key, 3)
    v_com = sigma_true * jax.random.normal(k_com, (n_stars,))

    n_b = int(round(f_b * n_stars))
    if n_b > 0:
        delta = sample_blend_velocities(k_binary, n_b, Z=Z)
        delta = jnp.concatenate([jnp.asarray(delta), jnp.zeros(n_stars - n_b)])
    else:
        delta = jnp.zeros(n_stars)

    noise = eps * jax.random.normal(k_noise, (n_stars,))
    return v_com + delta + noise


# --------------------------------------------------------------------------- #
# Gate 1 -- the bias exists
# --------------------------------------------------------------------------- #
def gate1_bias(key, var_korb, f_b_grid=None, n_real=20, eps=EPS_KMS):
    r"""Naive dynamical-mass bias ``M_naive/M_true`` vs binary fraction ``f_b``.

    For each ``f_b`` the mock is rebuilt ``n_real`` times; ``sigma_obs`` is the
    sample std of ``v_obs`` and ``M_ratio = (sigma_obs/sigma_true)^2`` (virial
    ``M ~ sigma^2`` at fixed ``r_h``). The measured points are compared to the
    analytic variance budget ``M = 1 + (eps^2 + f_b Var(K_orb))/sigma_true^2``.
    Gate passes iff the dispersion is inflated and the mass is biased > 10 % high
    at ``f_b = 0.5``.

    Returns ``(passed, info)`` with ``info`` carrying the curve for the figure.
    """
    if f_b_grid is None:
        f_b_grid = np.linspace(0.0, 0.7, 8)   # includes 0.5

    keys = jax.random.split(key, len(f_b_grid) * n_real).reshape(
        len(f_b_grid), n_real, 2
    )
    m_ratio_mean = np.zeros(len(f_b_grid))
    m_ratio_se = np.zeros(len(f_b_grid))
    for i, f_b in enumerate(f_b_grid):
        ratios = np.array([
            float(dyn_mass_ratio(
                jnp.std(build_mock_vlos(keys[i, j], f_b=float(f_b))), SIGMA_TRUE))
            for j in range(n_real)
        ])
        m_ratio_mean[i] = ratios.mean()
        m_ratio_se[i] = ratios.std() / np.sqrt(n_real)

    # Headline at f_b = 0.5 (or nearest grid point).
    i50 = int(np.argmin(np.abs(f_b_grid - 0.5)))
    m_ratio_50 = m_ratio_mean[i50]
    sigma_obs_50 = SIGMA_TRUE * np.sqrt(m_ratio_50)
    passed = bool(sigma_obs_50 > SIGMA_TRUE and m_ratio_50 > 1.10)

    m_ratio_pred = 1.0 + (eps ** 2 + f_b_grid * var_korb) / SIGMA_TRUE ** 2
    info = dict(f_b_grid=f_b_grid, m_ratio_mean=m_ratio_mean,
                m_ratio_se=m_ratio_se, m_ratio_50=m_ratio_50,
                sigma_obs_50=sigma_obs_50, m_ratio_pred=m_ratio_pred)
    _plot_bias(info)
    return passed, info


def _plot_bias(info):
    """M_naive/M_true vs f_b with the unbiased line and the f_b=0.5 marker."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.axhline(1.0, color="0.5", ls="--", lw=1.0, label="unbiased ($M_{\\rm naive}=M_{\\rm true}$)")
    ax.plot(info["f_b_grid"], info["m_ratio_pred"], color=OI["black"], lw=1.2,
            ls="-", alpha=0.7, label=r"variance budget $1+(\epsilon^2+f_b\,{\rm Var}\,K_{\rm orb})/\sigma_{\rm true}^2$")
    ax.errorbar(info["f_b_grid"], info["m_ratio_mean"], yerr=info["m_ratio_se"],
                marker="o", ls="none", color=OI["blue"], capsize=2,
                label="measured (mock $\\sigma_{\\rm obs}$)")
    ax.scatter([0.5], [info["m_ratio_50"]], color=OI["vermilion"], zorder=5,
               label=f"$f_b=0.5$: {info['m_ratio_50']:.2f}$\\times$ bias")
    ax.set_xlabel("binary fraction $f_b$")
    ax.set_ylabel(r"$M_{\rm naive}/M_{\rm true} = (\sigma_{\rm obs}/\sigma_{\rm true})^2$")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    panel_label(ax, "B12")
    save_fig(fig, OUTPUT_DIR, "demo_binary_dynamical_mass_bias")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Gate 2 -- a dispersion-only analysis is rank-1 degenerate
# --------------------------------------------------------------------------- #
def gate2_dispersion_degeneracy(var_korb, eps=EPS_KMS):
    r"""The dispersion-only Fisher for ``(sigma_true, f_b)`` is rank-1.

    A dispersion-only analysis compresses the data to ONE number,
    ``sigma_obs(sigma_true, f_b) = sqrt(sigma_true^2 + f_b Var(K_orb) + eps^2)``.
    Its 2x2 Fisher is the outer product ``J J^T`` of the single summary's
    gradient -- exactly rank 1 (one eigenvalue is machine-zero). Gate passes iff
    it is near-singular (condition number > 1e8 / smallest eig < 1e-8 * largest).

    Returns ``(passed, info)``.
    """
    theta0 = jnp.array([SIGMA_TRUE, F_B_TRUE])

    def sigma_obs_pred(theta):
        return jnp.sqrt(theta[0] ** 2 + theta[1] * var_korb + eps ** 2)

    J = jax.grad(sigma_obs_pred)(theta0)          # (2,)
    # Single scalar summary with SE se: F = (1/se^2) J J^T; rank is se-independent.
    F_disp = jnp.outer(J, J)
    eigs = jnp.linalg.eigvalsh(F_disp)            # ascending; [~0, |J|^2]
    largest = float(eigs[-1])
    smallest = float(eigs[0])
    cond = largest / max(smallest, np.finfo(float).tiny)
    passed = bool(smallest < 1e-8 * largest or cond > 1e8)

    info = dict(J=np.asarray(J), eigs=np.asarray(eigs), cond=cond,
                smallest=smallest, largest=largest)
    return passed, info


# --------------------------------------------------------------------------- #
# Gate 3 -- joint recovery from the wings is unbiased + full-rank
# --------------------------------------------------------------------------- #
def _predict_mu_factory(korb_grid, korb, n_stars, eps):
    """Bounded predicted-counts closure ``predict_mu(z)`` over unconstrained z.

    ``sigma = expit(z[0]; SIGMA_BOX)``, ``f_b = expit(z[1]; FB_BOX)`` keep the
    optimizer in-bounds; ``predict_vlos_counts`` supplies the differentiable
    binned single+binary mixture.
    """
    korb = jnp.asarray(korb)

    def predict_mu(z):
        sigma = expit(z[0], *SIGMA_BOX)
        f_b = expit(z[1], *FB_BOX)
        return predict_vlos_counts(sigma, f_b, n_stars, V_EDGES, korb_grid, korb, eps)

    return predict_mu


def recover_sigma_fb(v_obs, korb_grid, korb, eps=EPS_KMS, fb_guess=0.3):
    r"""Joint Poisson MLE of ``(sigma_true, f_b)`` from a binned ``v_obs`` sample.

    Bins ``v_obs`` over ``V_EDGES``, fits the differentiable mixture by Adam in
    the unconstrained ``z`` space, then maps the Poisson Fisher information back
    to ``(sigma, f_b)`` space via the delta method. Reused by the eps-floor sweep
    and the null gate.

    Returns a dict with ``sigma_hat, fb_hat, sigma_err, fb_err, cov, F, cond,
    z_hat, counts``.
    """
    v_obs = np.asarray(v_obs)
    n_stars = v_obs.shape[0]
    counts = jnp.asarray(np.histogram(v_obs, bins=V_EDGES)[0].astype(float))
    weight = jnp.ones_like(counts)

    predict_mu = _predict_mu_factory(korb_grid, korb, n_stars, eps)

    sigma_guess = float(np.clip(v_obs.std(), SIGMA_BOX[0] + 1e-3, SIGMA_BOX[1] - 1e-3))
    z0 = jnp.array([logit(sigma_guess, *SIGMA_BOX), logit(fb_guess, *FB_BOX)])

    nll = lambda z: -poisson_loglike((counts, weight), predict_mu)(z)
    z_hat, _trace = mle_adam(nll, z0, n_steps=N_ADAM, lr=ADAM_LR)

    sigma_hat = float(expit(z_hat[0], *SIGMA_BOX))
    fb_hat = float(expit(z_hat[1], *FB_BOX))

    # Fisher in z-space -> (sigma, f_b)-space via the expit Jacobian diagonal.
    F = poisson_fisher_information(predict_mu, z_hat, weight)
    dsig = float(jax.grad(lambda zi: expit(zi, *SIGMA_BOX))(z_hat[0]))
    dfb = float(jax.grad(lambda zi: expit(zi, *FB_BOX))(z_hat[1]))
    cov = np.asarray(constrained_cov(F, jnp.array([dsig, dfb])))
    eigs = np.linalg.eigvalsh(np.asarray(F))
    cond = float(eigs[-1] / max(eigs[0], np.finfo(float).tiny))

    return dict(
        sigma_hat=sigma_hat, fb_hat=fb_hat,
        sigma_err=float(np.sqrt(cov[0, 0])), fb_err=float(np.sqrt(cov[1, 1])),
        cov=cov, F=np.asarray(F), cond=cond, z_hat=np.asarray(z_hat),
        counts=np.asarray(counts),
    )


def gate3_recovery(key, korb_grid, korb, var_korb, eps=EPS_KMS):
    r"""Joint recovery (Gate 3a) + full-rank Fisher and constraint figure (3b).

    Gate 3a: the recovered ``sigma_hat`` is within ``3 sigma_err`` of the truth
    and the recovered ``M_dyn`` bias ``(sigma_hat/sigma_true)^2`` is ~1.
    Gate 3b: the full-distribution Poisson Fisher is well-conditioned (cond < 1e6)
    -- the non-Gaussian wings broke the dispersion-only rank-1 degeneracy.
    """
    v_obs = build_mock_vlos(key, f_b=F_B_TRUE, eps=eps)
    sigma_obs = float(np.asarray(v_obs).std())
    rec = recover_sigma_fb(v_obs, korb_grid, korb, eps=eps)

    m_ratio_rec = (rec["sigma_hat"] / SIGMA_TRUE) ** 2
    within_3sig = abs(rec["sigma_hat"] - SIGMA_TRUE) < 3.0 * rec["sigma_err"]
    mass_unbiased = abs(m_ratio_rec - 1.0) < 0.10
    passed_3a = bool(within_3sig and mass_unbiased)
    passed_3b = bool(rec["cond"] < 1e6)

    info = dict(sigma_obs=sigma_obs, var_korb=var_korb, eps=eps,
                m_ratio_rec=m_ratio_rec, **rec)
    _plot_constraint(info)
    return passed_3a, passed_3b, info


def _cov_ellipse(ax, mean, cov, nsig=1.0, **kw):
    """Add an ``nsig``-sigma covariance ellipse centered at ``mean`` to ``ax``."""
    from matplotlib.patches import Ellipse

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * nsig * np.sqrt(np.maximum(vals, 0.0))
    ax.add_patch(Ellipse(mean, width, height, angle=angle, fill=False, **kw))


def _plot_constraint(info):
    """Dispersion-only degenerate ridge vs the tight full-distribution ellipse."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    # Dispersion-only ridge: every (sigma, f_b) with the SAME sigma_obs.
    fb_line = np.linspace(FB_BOX[0], FB_BOX[1], 300)
    sig_ridge = np.sqrt(np.clip(
        info["sigma_obs"] ** 2 - info["eps"] ** 2 - fb_line * info["var_korb"],
        0.0, None))
    keep = sig_ridge > 0
    ax.plot(sig_ridge[keep], fb_line[keep], color=OI["orange"], lw=1.8,
            label="dispersion-only ridge (rank-1)")

    # Full-distribution constraint: 1- and 2-sigma ellipses at the joint MLE.
    mean = (info["sigma_hat"], info["fb_hat"])
    for nsig, a in ((1.0, 0.9), (2.0, 0.45)):
        _cov_ellipse(ax, mean, info["cov"], nsig=nsig, color=OI["blue"],
                     lw=1.6, alpha=a)
    ax.scatter([info["sigma_hat"]], [info["fb_hat"]], color=OI["blue"], s=20,
               zorder=5, label="joint MLE (full distribution)")
    ax.scatter([SIGMA_TRUE], [F_B_TRUE], color=OI["vermilion"], marker="*",
               s=120, zorder=6, label="truth")

    ax.set_xlabel(r"$\sigma_{\rm true}$ [km/s]")
    ax.set_ylabel(r"binary fraction $f_b$")
    ax.set_xlim(SIGMA_TRUE - 5.0 * info["sigma_err"] - 0.5,
                info["sigma_obs"] + 0.5)
    ax.set_ylim(FB_BOX[0], FB_BOX[1])
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    panel_label(ax, "B12")
    save_fig(fig, OUTPUT_DIR, "demo_binary_dynamical_mass_constraint")
    plt.close(fig)
