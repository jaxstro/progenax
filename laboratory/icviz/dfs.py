"""ICViz velocity-DF figures.

F3  ``king-lowered-maxwellian`` — what "lowering" means: the King speed
    distribution vs the pure Maxwellian it truncates, at three well depths.
F4  ``plummer-dispersion-oracles`` — sampled sigma_r(r) against the closed
    form AND projected sigma_los(R) against the Dejonghe (1987) Eq. 43
    (3 pi / 64) oracle, with Poisson residuals.
F5  ``beta-anisotropy`` — realized beta(r) for OM-Plummer / OM-EFF against
    the exact OM identity, plus Michie sitting below its OM ceiling.
F10 ``eddington-triptych`` — the inversion pipeline rho(Psi) -> d2rho/dPsi2
    -> f(E) for Plummer, with the NUMERICAL inverter's output dots on the
    closed-form 24 sqrt(2)/(7 pi^3) E^(7/2) law.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxstro.units import STELLAR

from progenax import (
    EFFProfile,
    EFFVelocityDF,
    MichieProfile,
    MichieVelocityDF,
    PlummerProfile,
    PlummerVelocityDF,
    project_dispersion,
)

from .style import polish_axes, setup_style

SEED = 11
_N = 200_000
_G = STELLAR.G


# ---------------------------------------------------------------------------
# F3 — the lowered Maxwellian
# ---------------------------------------------------------------------------


def build_king_lowered_maxwellian() -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)

    # After unit normalization the Maxwellian reference is ONE W-independent
    # curve (the e^W amplitude cancels) — draw it once, grey.
    v_full = np.linspace(0, 5.4, 500)
    maxwellian = v_full**2 * np.exp(-(v_full**2) / 2)
    max_norm = np.trapezoid(maxwellian, v_full)
    ax.plot(
        v_full, maxwellian / max_norm, color="#8A8A8A", lw=1.0, ls=(0, (3, 2)),
        label="pure Maxwellian",
    )

    colors = {1.0: "#E9C46A", 3.0: "#F4A261", 6.0: "#E76F51"}
    for W, color in colors.items():
        v_esc = np.sqrt(2.0 * W)
        v = np.linspace(0, v_esc, 400)
        # Same normalization as the reference (the e^W amplitude divides out):
        # f_lowered/f_Maxwell = (e^{-v^2/2} - e^{-W}) / e^{-v^2/2} at each v.
        lowered = v**2 * (np.exp(-(v**2) / 2) - np.exp(-W)) / max_norm
        ax.plot(v, lowered, color=color, label=rf"$W = {W:g}$")
        ax.axvline(v_esc, color=color, lw=0.6, alpha=0.45, ymax=0.14)

    ax.annotate(
        r"cut at $v_{\rm esc} = \sqrt{2W}$",
        xy=(np.sqrt(12.0), 0.075), xytext=(4.05, 0.22), fontsize=7.2,
        color="#555555",
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
    )
    ax.set_xlim(0, 5.4)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$v / \sigma_0$")
    ax.set_ylabel(r"$f(v \mid W)$  (Maxwellian-normalized)")
    ax.legend(frameon=False, loc="upper right")
    polish_axes(ax)
    return fig


# ---------------------------------------------------------------------------
# F4 — Plummer dispersion + the Dejonghe projected oracle
# ---------------------------------------------------------------------------


def build_plummer_dispersion_oracles() -> plt.Figure:
    setup_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 3.8),
        constrained_layout=True,
        sharex="col",
        height_ratios=[2.6, 1.0],
    )
    M, r_h = 400.0, 1.0
    prof = PlummerProfile(r_h=r_h)
    a = float(prof.a)
    df = PlummerVelocityDF(r_h=r_h)
    key = jax.random.PRNGKey(SEED)
    k_pos, k_vel = jax.random.split(key)
    masses = jnp.full(_N, M / _N)
    pos = prof.sample_positions(masses, k_pos)
    vel = df.sample_velocities(pos, masses, k_vel, G=_G)

    radii = np.array(jnp.linalg.norm(pos, axis=1))
    r_hat = np.array(pos) / radii[:, None]
    v_r = np.einsum("ij,ij->i", np.array(vel), r_hat)

    # (a) sigma_r(r): binned sampled vs closed form ------------------------
    bins = np.geomspace(0.05, 8.0, 26)
    centers = np.sqrt(bins[1:] * bins[:-1])
    idx = np.digitize(radii, bins) - 1
    sig_hat, sig_err = np.full(len(centers), np.nan), np.full(len(centers), np.nan)
    for i in range(len(centers)):
        sel = idx == i
        n = int(sel.sum())
        if n >= 50:
            sig_hat[i] = np.std(v_r[sel])
            sig_err[i] = sig_hat[i] / np.sqrt(2 * (n - 1))
    sig_true = np.sqrt(_G * M / (6.0 * np.sqrt(centers**2 + a**2)))

    ax, ax_res = axes[0][0], axes[1][0]
    ax.plot(centers, sig_true, color="#355C7D", label="closed form")
    good = ~np.isnan(sig_hat)
    ax.errorbar(
        centers[good], sig_hat[good], yerr=sig_err[good],
        fmt="o", ms=2.6, lw=0.7, color="#355C7D", alpha=0.55,
        label=r"sampled ($N = 2{\times}10^5$)",
    )
    ax.set_xscale("log")
    ax.set_ylabel(r"$\sigma_r(r)$  [pc/Myr]")
    ax.set_title(r"intrinsic: $\sigma_r^2 = GM / 6\sqrt{r^2+a^2}$", fontsize=8.4)
    ax.legend(frameon=False, fontsize=6.6)
    polish_axes(ax)
    resid = sig_hat[good] / sig_true[good] - 1.0
    band = sig_err[good] / sig_true[good]
    ax_res.axhline(0, color="#9A9A9A", lw=0.6)
    ax_res.fill_between(centers[good], -band, band, color="#355C7D", alpha=0.15, lw=0)
    ax_res.plot(centers[good], resid, "o", ms=2.2, color="#355C7D", alpha=0.75)
    ax_res.set_xscale("log")
    ax_res.set_ylim(-0.05, 0.05)
    ax_res.set_xlabel(r"$r$  [pc]")
    ax_res.set_ylabel("resid.")
    polish_axes(ax_res, grid_axis="y")

    # (b) sigma_los(R): project_dispersion vs Dejonghe Eq. 43 ---------------
    R = jnp.geomspace(0.05, 6.0, 40)
    pj = project_dispersion(prof, None, R, M, _G)
    sig_los = np.array(pj.sigma_los)
    oracle = np.sqrt((3.0 * np.pi / 64.0) * _G * M / np.sqrt(a**2 + np.array(R) ** 2))

    ax, ax_res = axes[0][1], axes[1][1]
    ax.plot(np.array(R), oracle, color="#2A9D8F", label="Dejonghe (1987) Eq. 43")
    ax.plot(
        np.array(R), sig_los, "o", ms=2.6, color="#2A9D8F", alpha=0.55,
        label="`project_dispersion`",
    )
    ax.set_xscale("log")
    ax.set_ylabel(r"$\sigma_{\rm los}(R)$  [pc/Myr]")
    ax.set_title(r"projected: $\sigma_{\rm los}^2 = \frac{3\pi}{64}\, GM/\sqrt{a^2+R^2}$", fontsize=8.4)
    ax.legend(frameon=False, fontsize=6.6)
    polish_axes(ax)
    resid = sig_los / oracle - 1.0
    ax_res.axhline(0, color="#9A9A9A", lw=0.6)
    ax_res.plot(np.array(R), resid, "o", ms=2.2, color="#2A9D8F", alpha=0.75)
    ax_res.set_xscale("log")
    ax_res.set_ylim(-2e-5, 2e-5)
    ax_res.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_res.set_xlabel(r"$R$  [pc]")
    polish_axes(ax_res, grid_axis="y")

    return fig


# ---------------------------------------------------------------------------
# F5 — anisotropy: OM identity + Michie below its ceiling
# ---------------------------------------------------------------------------


def _binned_beta(pos: np.ndarray, vel: np.ndarray, bins: np.ndarray):
    radii = np.linalg.norm(pos, axis=1)
    r_hat = pos / radii[:, None]
    v_r = np.einsum("ij,ij->i", vel, r_hat)
    v_t_vec = vel - v_r[:, None] * r_hat
    v_t2 = np.einsum("ij,ij->i", v_t_vec, v_t_vec)  # BOTH tangential components
    centers = np.sqrt(bins[1:] * bins[:-1])
    idx = np.digitize(radii, bins) - 1
    beta = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        sel = idx == i
        if int(sel.sum()) >= 200:
            # beta = 1 - sigma_t^2 / (2 sigma_r^2) with 2-component sigma_t^2
            beta[i] = 1.0 - np.mean(v_t2[sel]) / (2.0 * np.mean(v_r[sel] ** 2))
    return centers, beta


def build_beta_anisotropy() -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.5), constrained_layout=True)
    key = jax.random.PRNGKey(SEED + 1)
    M = 400.0

    # OM-Plummer (r_a = 2) and OM-EFF (r_a = 3): realized beta vs identity.
    cases = [
        ("Plummer + OM, $r_a = 2$", PlummerProfile(r_h=1.0),
         PlummerVelocityDF(r_h=1.0, anisotropy_radius=2.0), 2.0, "#355C7D"),
        (r"EFF $\gamma=5$ + OM, $r_a = 3$", EFFProfile(a=0.766, gamma=5.0, r_t=12.0),
         EFFVelocityDF(a=0.766, gamma=5.0, r_t=12.0, anisotropy_radius=3.0), 3.0,
         "#2A9D8F"),
    ]
    bins = np.geomspace(0.08, 9.0, 22)
    for label, prof, df, r_a, color in cases:
        key, k_pos, k_vel = jax.random.split(key, 3)
        masses = jnp.full(_N, M / _N)
        pos = prof.sample_positions(masses, k_pos)
        vel = df.sample_velocities(pos, masses, k_vel, G=_G)
        centers, beta = _binned_beta(np.array(pos), np.array(vel), bins)
        r_fine = np.geomspace(0.08, 9.0, 300)
        ax.plot(r_fine, r_fine**2 / (r_fine**2 + r_a**2), color=color, lw=1.2)
        good = ~np.isnan(beta)
        ax.plot(centers[good], beta[good], "o", ms=3.0, color=color, alpha=0.65,
                label=label)

    # Michie: realized beta sits BELOW its OM ceiling (lowered model).
    key, k_pos, k_vel = jax.random.split(key, 3)
    mich_prof = MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0)
    mich_df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0)
    masses = jnp.full(_N, M / _N)
    pos = mich_prof.sample_positions(masses, k_pos)
    vel = mich_df.sample_velocities(pos, masses, k_vel, G=_G)
    bins_m = np.geomspace(0.3, float(mich_prof.r_t) * 0.75, 20)
    centers, beta = _binned_beta(np.array(pos), np.array(vel), bins_m)
    r_fine = np.geomspace(0.3, float(mich_prof.r_t) * 0.75, 300)
    ax.plot(r_fine, r_fine**2 / (r_fine**2 + 8.0**2), color="#6C5B7B", lw=1.0,
            ls=(0, (4, 2)), label="OM ceiling $r_a = 8$")
    good = ~np.isnan(beta)
    ax.plot(centers[good], beta[good], "s", ms=3.0, color="#6C5B7B", alpha=0.65,
            label=r"Michie $W_0=7$, $r_a=8 r_c$ (lowered)")

    ax.set_xscale("log")
    ax.set_ylim(-0.12, 1.0)
    ax.axhline(0.0, color="#B9B9B9", lw=0.5, zorder=0)
    ax.set_xlabel(r"$r$  [pc]")
    ax.set_ylabel(r"$\beta(r) = 1 - \sigma_t^2 / 2\sigma_r^2$")
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")
    polish_axes(ax)
    return fig


# ---------------------------------------------------------------------------
# F10 — the Eddington inversion pipeline, with the numerical inverter's dots
# ---------------------------------------------------------------------------


def build_eddington_triptych() -> plt.Figure:
    from progenax.kinematics.eddington import eddington_invert

    setup_style()
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(7.6, 2.8), constrained_layout=True
    )
    # Dimensionless Plummer: G = M = a = 1 (Dejonghe 1987 units).
    n = 2000
    r = np.geomspace(1e-3, 300.0, n)
    psi = 1.0 / np.sqrt(1.0 + r**2)
    rho = (3.0 / (4.0 * np.pi)) * psi**5

    ax1.plot(psi, rho, color="#355C7D")
    ax1.set_xlabel(r"$\Psi$")
    ax1.set_ylabel(r"$\rho(\Psi) = \frac{3}{4\pi}\Psi^5$")
    ax1.set_title("density as a function of potential", fontsize=8.2)

    d2 = (3.0 / (4.0 * np.pi)) * 20.0 * psi**3
    ax2.plot(psi, d2, color="#6C5B7B")
    ax2.set_xlabel(r"$\Psi$")
    ax2.set_ylabel(r"$\mathrm{d}^2\rho/\mathrm{d}\Psi^2 = \frac{15}{\pi}\Psi^3$")
    ax2.set_title("the Abel-kernel ingredient", fontsize=8.2)

    # Numerical inverter on the ANALYTIC (rho, Psi) pair -> dots on the law.
    r_j = jnp.asarray(r)
    psi_j = jnp.asarray(psi)
    rho_j = jnp.asarray(rho)
    drho_dr = jnp.gradient(rho_j, r_j)
    dpsi_dr = -r_j * (1.0 + r_j**2) ** (-1.5)
    E_grid, f_grid = eddington_invert(r_j, rho_j, drho_dr, psi_j, dpsi_dr)
    E = np.array(E_grid)
    f_num = np.array(f_grid)
    f_true = (24.0 * np.sqrt(2.0) / (7.0 * np.pi**3)) * E**3.5

    ax3.plot(E, f_true, color="#2A9D8F", label=r"$\frac{24\sqrt{2}}{7\pi^3}\,\mathcal{E}^{7/2}$")
    step = max(1, len(E) // 28)
    ax3.plot(E[::step], f_num[::step], "o", ms=2.8, color="#2A9D8F", alpha=0.6,
             label="`eddington_invert`")
    ax3.set_xlabel(r"$\mathcal{E}$")
    ax3.set_ylabel(r"$f(\mathcal{E})$")
    ax3.set_title("the inverted DF (numerical dots on the law)", fontsize=8.2)
    ax3.legend(frameon=False, fontsize=6.8, loc="upper left")

    for ax in (ax1, ax2, ax3):
        polish_axes(ax)
    return fig
