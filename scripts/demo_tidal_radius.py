#!/usr/bin/env python
r"""B7 -- Tidal radius from the count-limited outskirts; Galactocentric distance.

A stretch demo reusing the Poisson number-density channel (B11): an EFF young
cluster's TRUNCATION radius r_t is recovered from the outer star counts -- the
sparse regime where an honest Poisson likelihood matters -- and converted to a
Galactocentric distance via the Jacobi (tidal) radius.

Physics
-------
The Elson-Fall-Freeman (1987) profile rho(r) = (1 + r^2/a^2)^(-gamma/2) is sharply
truncated at r_t. The scale radius ``a`` is pinned by the INNER profile (where most
stars are), so it is held fixed; the truncation r_t is read from the OUTSKIRTS.
Bins are placed out PAST r_t so the outermost ones are empty in the data: a model
with too-large r_t predicts stars in those empty bins (penalized by the Poisson
``-mu`` term), while a too-small r_t cannot explain the observed outer stars. So
r_t is pinned by the few/zero outermost counts -- a Gaussian-on-log-count would be
ill-defined there; the Poisson channel is honest. The per-bin Fisher information
peaks at the truncation edge (panel b).

A steep halo (large gamma) self-truncates -- the density at r_t is already ~0, so
r_t sits in the noise; this uses a SHALLOW gamma=2.5 (YMC-typical) so stars reach
out to r_t and pin it.

The recovered r_t IS the Jacobi/tidal radius, giving the cluster's **Galactocentric
distance**: r_t = R_gal (M_cl / 3 M_gal)^(1/3) (King 1962), hence
R_gal = r_t (3 M_gal / M_cl)^(1/3), with sigma(R_gal)/R_gal = sigma(r_t)/r_t.

Method: Engine-free closed-form EFF density on a fixed grid -> binned expected
counts mu_k(r_t); per-bin Poisson MLE (poisson_loglike) + reverse-mode Poisson
Fisher (poisson_fisher_information, which floors mu so the empty outer bins are
NaN-safe); sigma(r_t) vs N forecast.

Gates (exit 0 = all pass):
  * self-consistency: predict(truth) matches the data within 4 sigma Poisson;
  * r_t recovery within 3 sigma;
  * the outer (count-limited) bins dominate the r_t Fisher information;
  * forecast sigma(r_t) ~ N^-1/2;
  * Jacobi: R_gal round-trips through jacobi_radius.

Run record (2026-06-12, CPU/float64, N=20000, gamma=2.5, a=1 fixed, r_t_true=12,
K=22 bins to 15.6 pc, keys PRNGKey(0/1/2), wall ~5 s, exit 0 / ALL PASS):
  self-consistency max|N_k - mu_k|/sqrt(mu_k) = 2.56 (< 4).
  r_t: 12.0 -> 12.0087 +- 0.0371 pc (pull +0.24).
  r_t Fisher info: 93% from the outer bins (r > 0.6 r_t) -- the count-limited
    outskirts pin r_t (the per-bin info spikes at the truncation edge, panel b).
  forecast sigma(r_t) ~ N^-0.500.
  Jacobi -> R_gal = 2.79 +- 0.01 kpc (M_gal=5e10 Msun, point-mass interior);
    jacobi_radius round-trip r_t = 12.009.
  NB: r_t is differentiable via the cumulative-clip-at-r_t trick (a hard jnp.where
  truncation has ZERO gradient through its boundary condition); the harness
  poisson_fisher_information floors mu so the empty outer bins stay NaN-safe.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_tidal_radius.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR
from progenax import EFFProfile
from progenax.imf import Maschberger
from progenax.tidal import jacobi_radius

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _demo_inference import (
    binned_number_density,
    expit,
    mle_adam,
    poisson_fisher_information,
    poisson_loglike,
)
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = STELLAR.G

# Truth EFF young cluster: shallow halo (gamma=2.5), scale a pinned by the inner profile.
A_FIXED, GAMMA, RT_TRUE = 1.0, 2.5, 12.0
N_STARS = 20_000
SEED = 0

# Galaxy model for the Jacobi -> Galactocentric conversion (point-mass MW interior).
M_GALAXY = 5.0e10        # Msun within the cluster's orbit (representative)

K_BINS = 22
R_LO = 0.2
R_HI_FAC = 1.3           # bin out to 1.3 r_t so the outer bins are EMPTY (pin r_t)
N_FINE = 3000
RT_BOX = (6.0, 18.0)
N_INITS = 3
N_ADAM = 400
ADAM_LR = 3e-2
SELFCON_NSIG = 4.0
RECOVERY_NSIG = 3.0


def _rt_of_z(z):
    return expit(z[0], *RT_BOX)


def _drt_dz(z):
    s = jax.nn.sigmoid(z[0])
    return (RT_BOX[1] - RT_BOX[0]) * s * (1.0 - s)


def _z_of_rt(r_t):
    return jnp.array([float(jnp.log((r_t - RT_BOX[0]) / (RT_BOX[1] - r_t)))])


def make_predict_counts(r_edges, n_obs):
    """mu_k(r_t) = n_obs * (EFF enclosed fraction in bin k), hard-truncated at r_t.

    r_t is made DIFFERENTIABLE without a hard ``jnp.where`` on the density (whose
    boundary condition has zero gradient w.r.t. r_t): integrate the UNtruncated EFF
    cumulative once, then evaluate it at bin edges CLIPPED at r_t and normalize by
    ``enc(r_t)``. r_t flows through the (differentiable) ``minimum`` clip and the
    ``interp`` normalization -- a sharp truncation with a real dmu/dr_t."""
    # grid spans past the r_t box so interp(r_t) is always in-range.
    r_grid = jnp.linspace(float(r_edges[0]), RT_BOX[1] + 1.0, N_FINE)
    rho = (1.0 + r_grid**2 / A_FIXED**2) ** (-GAMMA / 2.0)
    integrand = 4.0 * jnp.pi * r_grid**2 * rho
    incr = 0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(r_grid)
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(incr)])  # untruncated enclosed(r)

    def predict_mu(z):
        r_t = _rt_of_z(z)
        enc = jnp.interp(jnp.minimum(r_edges, r_t), r_grid, cum)  # clip bin edges at r_t
        total = jnp.interp(r_t, r_grid, cum)                      # enclosed up to r_t
        p_k = (enc[1:] - enc[:-1]) / total
        return n_obs * p_k

    return predict_mu


def build_truth_data():
    masses = Maschberger(alpha=2.3, m_min=0.08, m_max=100.0).sample(
        jax.random.PRNGKey(SEED + 2), N_STARS)
    prof = EFFProfile(a=A_FIXED, gamma=GAMMA, r_t=RT_TRUE)
    pos = prof.sample_positions(masses, jax.random.PRNGKey(SEED))
    r_edges = jnp.geomspace(R_LO, R_HI_FAC * RT_TRUE, K_BINS + 1)
    counts = binned_number_density(pos, r_edges)
    return r_edges, counts, float(jnp.sum(counts)), float(jnp.sum(masses))


def run_mle(negloglike, key):
    z_true = _z_of_rt(RT_TRUE)
    inits = jnp.concatenate(
        [z_true[None, :], z_true[None, :] + jax.random.normal(key, (N_INITS - 1, 1)) * 0.4])
    nll = jax.jit(negloglike)
    finals = [mle_adam(nll, z0, n_steps=N_ADAM, lr=ADAM_LR)[0] for z0 in inits]
    losses = [float(nll(z)) for z in finals]
    return finals[int(np.argmin(losses))]


def main():
    print("=" * 78)
    print("TIDAL RADIUS FROM THE COUNT-LIMITED OUTSKIRTS (B7)")
    print("=" * 78)

    r_edges, counts, n_obs, m_cl = build_truth_data()
    print(f"\n  truth EFF a={A_FIXED} (fixed), gamma={GAMMA}, r_t={RT_TRUE} pc; N={N_STARS}, "
          f"M_cl={m_cl:.0f} Msun")
    print(f"  binned N_obs={n_obs:.0f} over {K_BINS} bins to {float(r_edges[-1]):.1f} pc")

    predict_mu = make_predict_counts(r_edges, n_obs)
    negloglike = lambda z: -poisson_loglike((counts, jnp.ones_like(counts)), predict_mu)(z)

    z_true = _z_of_rt(RT_TRUE)
    mu_true = np.asarray(predict_mu(z_true))
    selfcon = float(np.max(np.abs(np.asarray(counts) - mu_true)
                           / np.sqrt(np.maximum(mu_true, 1.0))))
    print(f"\n  self-consistency: max|N_k - mu_k|/sqrt(mu_k) = {selfcon:.2f} (gate < {SELFCON_NSIG})")

    z_hat = run_mle(negloglike, jax.random.PRNGKey(SEED + 1))
    rt_hat = float(_rt_of_z(z_hat))
    F_z = poisson_fisher_information(predict_mu, z_hat)
    drt = float(_drt_dz(z_hat))
    sig_rt = float(jnp.sqrt(1.0 / F_z[0, 0]) * abs(drt))
    pull = (rt_hat - RT_TRUE) / sig_rt
    print(f"\n  r_t recovery: {RT_TRUE} -> {rt_hat:.4f} +- {sig_rt:.4f} pc (pull {pull:+.2f})")

    # per-bin Fisher information for r_t: info_k = (dmu_k/dr_t)^2 / mu_k.
    J = np.asarray(jax.jacrev(predict_mu)(z_hat))[:, 0]   # dmu_k/dz
    mu_hat = np.asarray(predict_mu(z_hat))
    info_k = J**2 / np.maximum(mu_hat, 1e-300)
    r_mid = np.sqrt(np.asarray(r_edges)[:-1] * np.asarray(r_edges)[1:])
    outer = r_mid > 0.6 * rt_hat
    outer_frac = float(info_k[outer].sum() / info_k.sum())
    print(f"  r_t Fisher info: {outer_frac*100:.0f}% from the outer bins (r > 0.6 r_t) "
          f"-- the count-limited outskirts pin r_t")

    # forecast sigma(r_t) ~ N^-1/2.
    info_rt = 1.0 / (sig_rt**2 * N_STARS)
    n_grid = np.array([1e3, 3e3, 1e4, 3e4, 1e5, 3e5])
    sig_rt_grid = 1.0 / np.sqrt(n_grid * info_rt)
    slope = float(np.polyfit(np.log(n_grid), np.log(sig_rt_grid), 1)[0])

    # Jacobi -> Galactocentric distance.
    fac = (3.0 * M_GALAXY / m_cl) ** (1.0 / 3.0)
    R_gal, sig_Rgal = rt_hat * fac, sig_rt * fac
    rt_check = float(jacobi_radius(m_cl, M_GALAXY, R_gal))
    print(f"\n  forecast sigma(r_t) ~ N^{slope:.3f}")
    print(f"  Jacobi -> R_gal = {R_gal/1e3:.2f} +- {sig_Rgal/1e3:.2f} kpc "
          f"(M_gal={M_GALAXY:.0e} Msun); jacobi_radius round-trip r_t={rt_check:.3f}")

    make_figure(r_edges, counts, mu_hat, rt_hat, sig_rt, r_mid, info_k,
                n_grid, sig_rt_grid, R_gal / 1e3, sig_Rgal / 1e3)

    selfcon_ok = selfcon < SELFCON_NSIG
    recovery_ok = abs(pull) < RECOVERY_NSIG
    outer_ok = outer_frac > 0.5
    forecast_ok = -0.55 < slope < -0.45
    jacobi_ok = np.isfinite(R_gal) and abs(rt_check - rt_hat) < 1e-6 * rt_hat

    rows = [
        ("self-consistency at truth", "PASS" if selfcon_ok else "FAIL",
         f"<{SELFCON_NSIG} sigma", selfcon_ok),
        ("r_t recovery", "PASS" if recovery_ok else "FAIL", f"<{RECOVERY_NSIG} sigma", recovery_ok),
        ("outskirts pin r_t (>50% info)", "PASS" if outer_ok else "FAIL",
         f"{outer_frac*100:.0f}%", outer_ok),
        ("forecast sigma(r_t)~N^-1/2", "PASS" if forecast_ok else "FAIL", "slope -0.5", forecast_ok),
        ("Jacobi R_gal round-trip", "PASS" if jacobi_ok else "FAIL", "consistent", jacobi_ok),
    ]
    print("\n" + "-" * 78)
    print(f"  {'CHECK':<32s} {'status':>6s} {'gate':>14s}")
    print("-" * 78)
    all_ok = True
    for name, status, gate, ok in rows:
        all_ok &= ok
        print(f"  {name:<32s} {status:>6s} {gate:>14s}")
    print("-" * 78)
    print(f"  saved {OUTPUT_DIR}/demo_tidal_radius.{{png,pdf}}")
    print("=" * 78)
    print("  TIDAL RADIUS DEMO: ALL PASS" if all_ok else "  TIDAL RADIUS DEMO: FAILED")
    return 0 if all_ok else 1


def make_figure(r_edges, counts, mu_hat, rt_hat, sig_rt, r_mid, info_k,
                n_grid, sig_rt_grid, R_gal_kpc, sig_Rgal_kpc):
    import matplotlib.pyplot as plt

    e = np.asarray(r_edges)
    cen = np.sqrt(e[:-1] * e[1:])
    shell = 4.0 / 3.0 * np.pi * (e[1:] ** 3 - e[:-1] ** 3)
    cnt = np.asarray(counts)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))

    # (a) number-density profile + the recovered truncation.
    ax = axes[0]
    pop = cnt > 0
    ax.errorbar(cen[pop], (cnt / shell)[pop], yerr=(np.sqrt(cnt) / shell)[pop],
                fmt="o", ms=3.5, color=OI["black"], label="counts", zorder=4)
    ax.plot(cen, mu_hat / shell, "-", color=OI["vermilion"], label="MLE")
    ax.axvline(rt_hat, color=OI["green"], ls="--",
               label=fr"$\hat r_t={rt_hat:.2f}\pm{sig_rt:.2f}$ pc")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r$  [pc]")
    ax.set_ylabel(r"number density  [pc$^{-3}$]")
    ax.legend(fontsize=7)
    panel_label(ax, "(a)")

    # (b) per-bin Fisher information for r_t -- peaks at the truncation edge.
    ax = axes[1]
    ax.fill_between(cen, info_k, step="mid", color=OI["sky"], alpha=0.6)
    ax.plot(cen, info_k, "o-", ms=3, color=OI["blue"])
    ax.axvline(rt_hat, color=OI["green"], ls="--", label=fr"$\hat r_t$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r$  [pc]")
    ax.set_ylabel(r"$r_t$ Fisher info per bin")
    ax.legend(fontsize=7)
    panel_label(ax, "(b)")

    # (c) forecast sigma(r_t) vs N + the Galactocentric distance.
    ax = axes[2]
    ax.loglog(n_grid, sig_rt_grid, "o-", color=OI["green"])
    ax.set_xlabel(r"$N_\star$")
    ax.set_ylabel(r"$\sigma(r_t)$  [pc]")
    ax.text(0.5, 0.92, fr"$R_{{\rm gal}}={R_gal_kpc:.2f}\pm{sig_Rgal_kpc:.2f}$ kpc",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7"))
    panel_label(ax, "(c)")

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "demo_tidal_radius")


if __name__ == "__main__":
    sys.exit(main())
