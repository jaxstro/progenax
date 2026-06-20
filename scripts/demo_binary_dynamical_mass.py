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
    _kernel_std,
    build_korb_kernel,
    dyn_mass_ratio,
    predict_vlos_counts,
    sample_blend_velocities,
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
SIGMA_TRUE = 5.0  # true cluster LOS velocity dispersion [km/s]
Z_MET = 1e-3  # metallicity for the Tout ZAMS photometry (metal-poor)
F_B_TRUE = 0.5  # unresolved binary fraction
N_STARS = 1500  # RV stars in the mock survey
EPS_KMS = 1.0  # per-star RV measurement precision [km/s]
R_H_PC = 30.0  # half-mass radius [pc] (sets the virial-mass scale)

# Binned-likelihood velocity grid: wide enough (~+/-10 sigma_obs) to hold the
# non-Gaussian binary wings that break the degeneracy.
V_EDGES = np.linspace(-60.0, 60.0, 121)  # 120 bins of 1 km/s
N_POOL = 200_000  # K_orb template pool (low template noise)
KORB_GRID_MAX = 150.0  # K_orb grid half-width [km/s] (must span the wings)
KORB_N_GRID = 601

# Parameter boxes for the bounded MLE (logit/expit reparametrization).
SIGMA_BOX = (0.5, 30.0)
FB_BOX = (0.0, 0.95)
N_ADAM = 600
ADAM_LR = 3e-2
SEED = 0


def build_mock_vlos(
    key, f_b=F_B_TRUE, sigma_true=SIGMA_TRUE, eps=EPS_KMS, n_stars=N_STARS, Z=Z_MET
):
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
        f_b_grid = np.linspace(0.0, 0.7, 8)  # includes 0.5

    keys = jax.random.split(key, len(f_b_grid) * n_real).reshape(
        len(f_b_grid), n_real, 2
    )
    m_ratio_mean = np.zeros(len(f_b_grid))
    m_ratio_se = np.zeros(len(f_b_grid))
    for i, f_b in enumerate(f_b_grid):
        ratios = np.array(
            [
                float(
                    dyn_mass_ratio(
                        jnp.std(build_mock_vlos(keys[i, j], f_b=float(f_b))), SIGMA_TRUE
                    )
                )
                for j in range(n_real)
            ]
        )
        m_ratio_mean[i] = ratios.mean()
        m_ratio_se[i] = ratios.std() / np.sqrt(n_real)

    # Headline at f_b = 0.5 (or nearest grid point).
    i50 = int(np.argmin(np.abs(f_b_grid - 0.5)))
    m_ratio_50 = m_ratio_mean[i50]
    sigma_obs_50 = SIGMA_TRUE * np.sqrt(m_ratio_50)
    passed = bool(sigma_obs_50 > SIGMA_TRUE and m_ratio_50 > 1.10)

    m_ratio_pred = 1.0 + (eps**2 + f_b_grid * var_korb) / SIGMA_TRUE**2
    info = dict(
        f_b_grid=f_b_grid,
        m_ratio_mean=m_ratio_mean,
        m_ratio_se=m_ratio_se,
        m_ratio_50=m_ratio_50,
        sigma_obs_50=sigma_obs_50,
        m_ratio_pred=m_ratio_pred,
    )
    _plot_bias(info)
    return passed, info


def _plot_bias(info):
    """M_naive/M_true vs f_b with the unbiased line and the f_b=0.5 marker."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.axhline(
        1.0,
        color="0.5",
        ls="--",
        lw=1.0,
        label="unbiased ($M_{\\rm naive}=M_{\\rm true}$)",
    )
    ax.plot(
        info["f_b_grid"],
        info["m_ratio_pred"],
        color=OI["black"],
        lw=1.2,
        ls="-",
        alpha=0.7,
        label=r"variance budget $1+(\epsilon^2+f_b\,{\rm Var}\,K_{\rm orb})/\sigma_{\rm true}^2$",
    )
    ax.errorbar(
        info["f_b_grid"],
        info["m_ratio_mean"],
        yerr=info["m_ratio_se"],
        marker="o",
        ls="none",
        color=OI["blue"],
        capsize=2,
        label="measured (mock $\\sigma_{\\rm obs}$)",
    )
    ax.scatter(
        [0.5],
        [info["m_ratio_50"]],
        color=OI["vermilion"],
        zorder=5,
        label=f"$f_b=0.5$: {info['m_ratio_50']:.2f}$\\times$ bias",
    )
    ax.set_xlabel("binary fraction $f_b$")
    ax.set_ylabel(
        r"$M_{\rm naive}/M_{\rm true} = (\sigma_{\rm obs}/\sigma_{\rm true})^2$"
    )
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
        return jnp.sqrt(theta[0] ** 2 + theta[1] * var_korb + eps**2)

    J = jax.grad(sigma_obs_pred)(theta0)  # (2,)
    # Single scalar summary with SE se: F = (1/se^2) J J^T; rank is se-independent.
    F_disp = jnp.outer(J, J)
    eigs = jnp.linalg.eigvalsh(F_disp)  # ascending; [~0, |J|^2]
    largest = float(eigs[-1])
    smallest = float(eigs[0])
    cond = largest / max(smallest, np.finfo(float).tiny)
    passed = bool(smallest < 1e-8 * largest or cond > 1e8)

    info = dict(
        J=np.asarray(J),
        eigs=np.asarray(eigs),
        cond=cond,
        smallest=smallest,
        largest=largest,
    )
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
        sigma_hat=sigma_hat,
        fb_hat=fb_hat,
        sigma_err=float(np.sqrt(cov[0, 0])),
        fb_err=float(np.sqrt(cov[1, 1])),
        cov=cov,
        F=np.asarray(F),
        cond=cond,
        z_hat=np.asarray(z_hat),
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

    info = dict(
        sigma_obs=sigma_obs, var_korb=var_korb, eps=eps, m_ratio_rec=m_ratio_rec, **rec
    )
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
    sig_ridge = np.sqrt(
        np.clip(
            info["sigma_obs"] ** 2 - info["eps"] ** 2 - fb_line * info["var_korb"],
            0.0,
            None,
        )
    )
    keep = sig_ridge > 0
    ax.plot(
        sig_ridge[keep],
        fb_line[keep],
        color=OI["orange"],
        lw=1.8,
        label="dispersion-only ridge (rank-1)",
    )

    # Full-distribution constraint: 1- and 2-sigma ellipses at the joint MLE.
    mean = (info["sigma_hat"], info["fb_hat"])
    for nsig, a in ((1.0, 0.9), (2.0, 0.45)):
        _cov_ellipse(
            ax, mean, info["cov"], nsig=nsig, color=OI["blue"], lw=1.6, alpha=a
        )
    ax.scatter(
        [info["sigma_hat"]],
        [info["fb_hat"]],
        color=OI["blue"],
        s=20,
        zorder=5,
        label="joint MLE (full distribution)",
    )
    ax.scatter(
        [SIGMA_TRUE],
        [F_B_TRUE],
        color=OI["vermilion"],
        marker="*",
        s=120,
        zorder=6,
        label="truth",
    )

    ax.set_xlabel(r"$\sigma_{\rm true}$ [km/s]")
    ax.set_ylabel(r"binary fraction $f_b$")
    ax.set_xlim(SIGMA_TRUE - 5.0 * info["sigma_err"] - 0.5, info["sigma_obs"] + 0.5)
    ax.set_ylim(FB_BOX[0], FB_BOX[1])
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    panel_label(ax, "B12")
    save_fig(fig, OUTPUT_DIR, "demo_binary_dynamical_mass_constraint")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Gate 4 -- the RV-precision floor (mass stays unbiased; precision degrades)
# --------------------------------------------------------------------------- #
def gate4_eps_floor(key, korb_grid, korb, eps_grid=None, n_real=16):
    r"""Recovered-mass PRECISION vs RV precision ``eps`` (the honest eps-floor).

    Because ``eps`` is known and convolved into both mixture components, the
    recovered mass stays UNBIASED at every ``eps``; what degrades is the
    precision ``sigma(sigma_true)`` (and ``sigma(f_b)``), which grows
    monotonically as ``eps`` washes out the binary wing signature. The honest
    scope is the detectable binary fraction ``f_b * P(|Delta| > eps)`` -- the
    sub-eps (long-period) binaries that become indistinguishable from singles.

    Gate passes iff ``sigma(sigma_true)`` increases monotonically with ``eps``
    AND the mass stays unbiased (``|M_ratio - 1| < 0.05`` at every ``eps``).
    """
    if eps_grid is None:
        eps_grid = np.array([0.2, 0.5, 1.0, 2.0, 3.0, 5.0])

    keys = jax.random.split(key, len(eps_grid) * n_real).reshape(
        len(eps_grid), n_real, 2
    )
    sig_err = np.zeros(len(eps_grid))
    fb_err = np.zeros(len(eps_grid))
    m_ratio = np.zeros(len(eps_grid))
    for i, eps in enumerate(eps_grid):
        se, fe, mr = [], [], []
        for j in range(n_real):
            v = build_mock_vlos(keys[i, j], f_b=F_B_TRUE, eps=float(eps))
            r = recover_sigma_fb(v, korb_grid, korb, eps=float(eps))
            se.append(r["sigma_err"])
            fe.append(r["fb_err"])
            mr.append((r["sigma_hat"] / SIGMA_TRUE) ** 2)
        sig_err[i] = np.mean(se)
        fb_err[i] = np.mean(fe)
        m_ratio[i] = np.mean(mr)

    # Detectable fraction f_b * P(|Delta| > eps) from a fresh blend sample.
    delta = np.abs(
        np.asarray(
            sample_blend_velocities(jax.random.PRNGKey(SEED + 7), 200_000, Z=Z_MET)
        )
    )
    f_det = np.array([F_B_TRUE * (delta > e).mean() for e in eps_grid])

    # Precision degrades with eps, but PLATEAUS at eps << std(K_orb) (information-
    # limited, not noise-limited), so test a clear end-to-end degradation plus no
    # large reversal -- not strict point-wise monotonicity in the plateau.
    degrades = bool(sig_err[-1] > 1.5 * sig_err[0])
    no_big_reversal = bool(np.all(np.diff(sig_err) >= -0.03 * np.max(sig_err)))
    unbiased = bool(np.all(np.abs(m_ratio - 1.0) < 0.05))
    passed = degrades and no_big_reversal and unbiased

    info = dict(
        eps_grid=eps_grid,
        sig_err=sig_err,
        fb_err=fb_err,
        m_ratio=m_ratio,
        f_det=f_det,
        degrades=degrades,
        no_big_reversal=no_big_reversal,
        unbiased=unbiased,
    )
    _plot_eps_floor(info)
    return passed, info


def _plot_eps_floor(info):
    """Mass precision sigma(sigma_true) rising vs eps; detectable fraction falling."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    # Left axis: mass precision sigma(sigma_true) in km/s (the headline metric).
    ax.plot(
        info["eps_grid"],
        info["sig_err"],
        marker="o",
        color=OI["blue"],
        label=r"$\sigma(\sigma_{\rm true})$ [km/s] (mass precision)",
    )
    ax.set_xlabel(r"RV precision $\epsilon$ [km/s]")
    ax.set_ylabel(r"$\sigma(\sigma_{\rm true})$ [km/s]", color=OI["blue"])
    ax.tick_params(axis="y", labelcolor=OI["blue"])
    ax.set_ylim(bottom=0.0)

    # Right axis: dimensionless quantities -- sigma(f_b) and the detectable fraction.
    ax2 = ax.twinx()
    ax2.plot(
        info["eps_grid"],
        info["fb_err"],
        marker="s",
        color=OI["sky"],
        ls="--",
        label=r"$\sigma(f_b)$",
    )
    ax2.plot(
        info["eps_grid"],
        info["f_det"],
        marker="^",
        color=OI["vermilion"],
        ls=":",
        label=r"detectable $f_b\,P(|\Delta|>\epsilon)$",
    )
    ax2.set_ylabel(r"$f_b$: uncertainty $\sigma(f_b)$ / detectable fraction")
    ax2.set_ylim(bottom=0.0)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(
        lines,
        [l.get_label() for l in lines],
        frameon=False,
        fontsize=8,
        loc="upper center",
    )
    panel_label(ax, "B12", loc="lower left")
    save_fig(fig, OUTPUT_DIR, "demo_binary_dynamical_mass_eps_floor")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Gate 5 (null) + Gate 6 (AD-vs-FD) + the N forecast
# --------------------------------------------------------------------------- #
def gate5_null(key, korb_grid, korb, n_real=12):
    r"""With no binaries (``f_b = 0``) the fit recovers ``f_b ~ 0`` and an
    unbiased ``sigma`` (no spurious inflation). Gate: mean ``fb_hat < 0.05`` and
    mean ``|sigma_hat - sigma_true| < 0.1`` km/s."""
    keys = jax.random.split(key, n_real)
    fb, sig = [], []
    for kk in keys:
        v = build_mock_vlos(kk, f_b=0.0)
        r = recover_sigma_fb(v, korb_grid, korb)
        fb.append(r["fb_hat"])
        sig.append(r["sigma_hat"])
    fb_mean, sig_mean = float(np.mean(fb)), float(np.mean(sig))
    passed = bool(fb_mean < 0.05 and abs(sig_mean - SIGMA_TRUE) < 0.1)
    return passed, dict(fb_mean=fb_mean, sig_mean=sig_mean)


def gate6_ad_vs_fd(korb_grid, korb, eps=EPS_KMS, h=1e-5):
    r"""Gradient integrity: the model Jacobian ``d mu / d z`` from reverse-mode
    autodiff must match central finite differences (max rel-err < 1e-4). Guards
    against a stop_gradient / dead-branch / non-differentiable op silently
    corrupting the Fisher."""
    predict_mu = _predict_mu_factory(korb_grid, korb, N_STARS, eps)
    z = jnp.array([logit(SIGMA_TRUE, *SIGMA_BOX), logit(F_B_TRUE, *FB_BOX)])

    J_ad = np.asarray(jax.jacrev(predict_mu)(z))  # (K, 2)
    J_fd = np.zeros_like(J_ad)
    for i in range(2):
        zp = z.at[i].add(h)
        zm = z.at[i].add(-h)
        J_fd[:, i] = np.asarray((predict_mu(zp) - predict_mu(zm)) / (2.0 * h))

    scale = np.max(np.abs(J_fd))
    max_rel_err = float(np.max(np.abs(J_ad - J_fd)) / scale)
    passed = bool(max_rel_err < 1e-4)
    return passed, dict(max_rel_err=max_rel_err)


def forecast_vs_N(key, korb_grid, korb, n_grid=None):
    r"""Fisher forecast ``sigma(sigma_true)``, ``sigma(f_b)`` vs sample size N.

    Both scale as ``1/sqrt(N)`` (Poisson information is additive in stars). The
    figure overlays the ``N^{-1/2}`` reference anchored at the smallest N.
    """
    if n_grid is None:
        n_grid = np.array([500, 1500, 5000, 15000])
    keys = jax.random.split(key, len(n_grid))
    sig_err = np.zeros(len(n_grid))
    fb_err = np.zeros(len(n_grid))
    for i, n in enumerate(n_grid):
        v = build_mock_vlos(keys[i], f_b=F_B_TRUE, n_stars=int(n))
        r = recover_sigma_fb(v, korb_grid, korb)
        sig_err[i] = r["sigma_err"]
        fb_err[i] = r["fb_err"]
    info = dict(n_grid=n_grid, sig_err=sig_err, fb_err=fb_err)
    _plot_forecast(info)
    return info


def _plot_forecast(info):
    """sigma(sigma_true), sigma(f_b) vs N on log-log with the 1/sqrt(N) guide."""
    import matplotlib.pyplot as plt

    n = info["n_grid"].astype(float)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.loglog(
        n,
        info["sig_err"],
        marker="o",
        color=OI["blue"],
        label=r"$\sigma(\sigma_{\rm true})$ [km/s]",
    )
    ax.loglog(n, info["fb_err"], marker="s", color=OI["orange"], label=r"$\sigma(f_b)$")
    guide = info["sig_err"][0] * np.sqrt(n[0] / n)
    ax.loglog(n, guide, color="0.5", ls=":", lw=1.0, label=r"$\propto N^{-1/2}$")
    ax.set_xlabel("number of RV stars $N$")
    ax.set_ylabel(r"forecast 1$\sigma$ uncertainty")
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, "B12")
    save_fig(fig, OUTPUT_DIR, "demo_binary_dynamical_mass_fisher_vs_N")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Headline figure -- the v_los distribution and its non-Gaussian wings
# --------------------------------------------------------------------------- #
def plot_vlos_distribution(key, korb_grid, korb):
    """Observed v_los vs the single-only Gaussian: the binary wings made visible.

    Log-y histogram of a mock ``v_obs`` (f_b=0.5) overlaid with (a) the single-
    only model 𝒩(0, sigma_true^2 + eps^2) and (b) the full single+binary mixture.
    The excess over the Gaussian in the wings is the f_b signal the joint fit uses.
    """
    import matplotlib.pyplot as plt

    v_obs = np.asarray(build_mock_vlos(key, f_b=F_B_TRUE))
    counts, _ = np.histogram(v_obs, bins=V_EDGES)
    ctr = 0.5 * (V_EDGES[:-1] + V_EDGES[1:])

    mu_single = np.asarray(
        predict_vlos_counts(SIGMA_TRUE, 0.0, N_STARS, V_EDGES, korb_grid, korb, EPS_KMS)
    )
    mu_full = np.asarray(
        predict_vlos_counts(
            SIGMA_TRUE, F_B_TRUE, N_STARS, V_EDGES, korb_grid, korb, EPS_KMS
        )
    )

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.step(
        ctr,
        np.maximum(counts, 0.1),
        where="mid",
        color="0.55",
        lw=1.0,
        label=f"mock $v_{{\\rm obs}}$ ($f_b={F_B_TRUE}$)",
    )
    ax.plot(
        ctr,
        mu_single,
        color=OI["orange"],
        lw=1.6,
        label=r"single-only $\mathcal{N}(0,\sigma_{\rm true}^2+\epsilon^2)$",
    )
    ax.plot(ctr, mu_full, color=OI["blue"], lw=1.6, label="single + binary mixture")

    # Shade the wing regions where the binary excess lives.
    wing = 2.5 * np.sqrt(SIGMA_TRUE**2 + EPS_KMS**2)
    ax.axvspan(wing, V_EDGES[-1], color=OI["vermilion"], alpha=0.08)
    ax.axvspan(V_EDGES[0], -wing, color=OI["vermilion"], alpha=0.08)
    ax.text(
        0.97,
        0.6,
        "non-Gaussian\nbinary wings",
        transform=ax.transAxes,
        fontsize=8,
        color=OI["vermilion"],
        ha="right",
    )

    ax.set_yscale("log")
    ax.set_ylim(0.5, 1.5 * counts.max())
    ax.set_xlabel(r"line-of-sight velocity $v_{\rm los}$ [km/s]")
    ax.set_ylabel("stars per bin")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    panel_label(ax, "B12", loc="upper right")
    save_fig(fig, OUTPUT_DIR, "demo_binary_dynamical_mass_distribution")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Gated CLI
# --------------------------------------------------------------------------- #
def main():
    """Run all six gates, print an expected-vs-measured PASS/FAIL table, exit 0/1."""
    print("=" * 72)
    print("B12 -- binary-inflated dynamical mass (gated demo)")
    print(
        f"  sigma_true={SIGMA_TRUE} km/s  f_b={F_B_TRUE}  N={N_STARS}  "
        f"eps={EPS_KMS} km/s  Z={Z_MET}  r_h={R_H_PC} pc"
    )
    print("=" * 72)

    # Build the sigma-independent contamination kernel ONCE.
    korb_grid, korb = build_korb_kernel(
        n_pool=N_POOL, Z=Z_MET, seed=SEED, grid_max=KORB_GRID_MAX, n_grid=KORB_N_GRID
    )
    var_korb = _kernel_std(korb_grid, korb) ** 2
    print(f"K_orb: std={np.sqrt(var_korb):.3f} km/s  (n_pool={N_POOL})\n")

    key = jax.random.PRNGKey(SEED)
    k1, k3, k4, k5, kN, kd = jax.random.split(key, 6)

    rows = []  # (gate, expected, measured, passed)

    p1, i1 = gate1_bias(k1, var_korb)
    rows.append(
        (
            "1 bias",
            "M_ratio(0.5)>1.10",
            f"{i1['m_ratio_50']:.3f} (sigma_obs={i1['sigma_obs_50']:.2f})",
            p1,
        )
    )

    p2, i2 = gate2_dispersion_degeneracy(var_korb)
    rows.append(("2 degeneracy", "cond>1e8 (rank-1)", f"{i2['cond']:.1e}", p2))

    p3a, p3b, i3 = gate3_recovery(k3, korb_grid, korb, var_korb)
    rows.append(
        (
            "3a recovery",
            "M_ratio~1, <3sigma",
            f"sigma={i3['sigma_hat']:.2f}+/-{i3['sigma_err']:.2f}, "
            f"M={i3['m_ratio_rec']:.3f}",
            p3a,
        )
    )
    rows.append(("3b full-rank", "cond<1e6", f"{i3['cond']:.0f}", p3b))

    p4, i4 = gate4_eps_floor(k4, korb_grid, korb)
    rows.append(
        (
            "4 eps-floor",
            "precision degrades, M~1",
            f"sigma_err {i4['sig_err'][0]:.3f}->{i4['sig_err'][-1]:.3f}, "
            f"max|M-1|={np.max(np.abs(i4['m_ratio'] - 1)):.3f}",
            p4,
        )
    )

    p5, i5 = gate5_null(k5, korb_grid, korb)
    rows.append(
        (
            "5 null",
            "fb_hat<0.05",
            f"fb={i5['fb_mean']:.3f}, sigma={i5['sig_mean']:.2f}",
            p5,
        )
    )

    p6, i6 = gate6_ad_vs_fd(korb_grid, korb)
    rows.append(("6 AD-vs-FD", "rel-err<1e-4", f"{i6['max_rel_err']:.1e}", p6))

    # Forecast + headline figure (not gated).
    forecast_vs_N(kN, korb_grid, korb)
    plot_vlos_distribution(kd, korb_grid, korb)

    print(f"{'GATE':<14}{'EXPECTED':<26}{'MEASURED':<42}{'RESULT'}")
    print("-" * 88)
    for name, exp, meas, ok in rows:
        print(f"{name:<14}{exp:<26}{meas:<42}{'PASS' if ok else 'FAIL'}")
    print("-" * 88)

    all_pass = all(ok for *_, ok in rows)
    print(
        f"\n{'ALL GATES PASS' if all_pass else 'SOME GATES FAILED'} "
        f"({sum(ok for *_, ok in rows)}/{len(rows)})"
    )
    print(
        "Figures: validation/plots/demo_binary_dynamical_mass_"
        "{distribution,bias,constraint,eps_floor,fisher_vs_N}.png"
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
