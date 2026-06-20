#!/usr/bin/env python
"""
Cluster-builder convenience-API validation figures + expected-vs-measured tables.

Validates the thin, differentiable ``build_cluster`` layer (the 5 named aliases,
``matched_velocity_df``, ``RotationSpec``, ``ClusterParams``) against the physics
it inherits from the composable ``build_spatial_ic`` core. The script recomputes
the same quantities the unit/integration tests assert and prints expected-vs-
measured PASS/FAIL tables, so the figures are a faithful visualization layer over
already-verified physics (not a second, drift-prone implementation).

Checks (-> what they anchor):
  table_virial_per_alias()        Q = T/|V| ~ 0.5 for every alias (default Q=0.5
                                  virial-scales the IC) -> integration near-virial
  fig_density_recovery.png        sampled n(r) vs analytic rho(r) for Plummer +
                                  King + EFF (log-log, residual panel) -> profile
                                  density-recovery (cf. validate_{plummer,king,eff})
  fig_tidal_cut.png               radial mass profile with/without tidal_radius:
                                  sharp cut at r_t + zero-mass ghosts beyond it
                                  -> apply_tidal_truncation (S4 caveat)
  fig_rotation_Lz.png             L_z(omega) linearity (slope = Sigma m R^2) for
                                  solid-body + v_phi(R) for differential overlay
                                  -> apply_{solid_body,differential}_rotation (S3)
  fig_anisotropy_beta.png         beta(r) = 1 - sigma_t^2/(2 sigma_r^2) for an OM
                                  Plummer vs analytic r^2/(r^2 + r_a^2)
                                  -> matched_velocity_df anisotropy_radius threading

References:
    Plummer (1911), MNRAS 71, 460; King (1966), AJ 71, 64;
    Elson, Fall & Freeman (1987), ApJ 323, 54; Merritt (1985), AJ 90, 1027
    (Osipkov-Merritt anisotropy); Binney & Tremaine (2008), Galactic Dynamics.

Usage:
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_cluster_builders.py

Output:
    validation/plots/cluster_builders/*.png (+ .pdf)
"""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR

from progenax import (
    EFFProfile,
    KingProfile,
    PlummerProfile,
    RotationSpec,
    build_eff_cluster,
    build_king_cluster,
    build_limepy_cluster,
    build_michie_cluster,
    build_plummer_cluster,
    compute_kinetic_energy,
    compute_potential_energy,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots/cluster_builders"
G = STELLAR.G  # pc^3 Msun^-1 Myr^-2
N_VIRIAL = 5_000  # per-alias virial table
N_DENSITY = 60_000  # density recovery (good shell statistics)
N_TIDAL = 20_000  # tidal cut mass profile
N_ROT = 5_000  # rotation L_z linearity
N_DIFF = 40_000  # differential v_phi(R) curve
N_ANISO = 60_000  # OM beta(r)
SEED = 42


def _virial_Q(ic):
    """Q = T/|V| over the mass-bearing particles (zero-mass ghosts drop out of the
    mass-weighted energies automatically)."""
    T = compute_kinetic_energy(ic.velocities, ic.masses)
    V = compute_potential_energy(ic.positions, ic.masses, G=G)
    return float(T / jnp.abs(V))


# ============================================================================
# Table -- virial ratio Q for every alias (default Q=0.5 virial scaling)
# ============================================================================
def table_virial_per_alias():
    """Build N=5000 from each of the 5 aliases and check Q = T/|V| ~ 0.5.

    `build_cluster` virial-scales to Q=0.5 by default, so every alias (regardless
    of its underlying DF's *unscaled* Q) must land at 0.5. A 0.03 tolerance covers
    the residual sampling noise of the post-scale energy estimate at N=5000.
    """
    print("\n" + "=" * 64)
    print("TABLE: virial ratio Q = T/|V| per alias (expected 0.5, default scale)")
    print("=" * 64)

    key = jax.random.PRNGKey(SEED)
    m = jnp.ones(N_VIRIAL)
    builders = [
        ("plummer", lambda k: build_plummer_cluster(masses=m, r_h=1.0, key=k)),
        # gamma=5 / r_t>>a -> mild truncation -> faithful near-virial EFF Eddington IC
        (
            "eff",
            lambda k: build_eff_cluster(masses=m, a=1.0, gamma=5.0, r_t=15.0, key=k),
        ),
        ("king", lambda k: build_king_cluster(masses=m, W0=7.0, r_c=1.0, key=k)),
        (
            "michie",
            lambda k: build_michie_cluster(masses=m, W0=7.0, r_c=1.0, r_a=8.0, key=k),
        ),
        (
            "limepy",
            lambda k: build_limepy_cluster(masses=m, W0=5.0, g=1.0, r_c=1.0, key=k),
        ),
    ]

    tol = 0.03
    print(
        f"  {'alias':<10}{'expected Q':>12}{'measured Q':>12}{'|delta|':>10}{'pass':>8}"
    )
    all_pass = True
    for name, build in builders:
        key, sub = jax.random.split(key)
        Q = _virial_Q(build(sub))
        delta = abs(Q - 0.5)
        ok = delta < tol
        all_pass = all_pass and ok
        print(
            f"  {name:<10}{0.5:>12.3f}{Q:>12.4f}{delta:>10.4f}"
            f"{'PASS' if ok else 'FAIL':>8}"
        )
    print(f"  tolerance |Q-0.5| < {tol}  ->  {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ============================================================================
# Figure -- sampled number density vs analytic profile (Plummer, King, EFF)
# ============================================================================
def _shell_weighted_density(profile, bin_lo, bin_hi, n_sub=200):
    """Shell-volume-weighted analytic density over a radial bin:

        <rho>_shell = int_lo^hi rho(r) r^2 dr / int_lo^hi r^2 dr.

    This is the analytic counterpart of the SAMPLED shell density (counts /
    shell volume). Comparing the sampled shell density against rho(center)
    instead biases the outermost bins where rho varies steeply across a wide
    log-bin (e.g. King near r_t) — a binning artifact, not a sampler error.
    """
    rr = np.linspace(bin_lo, bin_hi, n_sub)
    rho = np.asarray(profile.density(jnp.asarray(rr)))
    w = rr**2
    return float(np.trapezoid(rho * w, rr) / np.trapezoid(w, rr))


def _density_recovery(profile, radii, r_lo, r_hi, n_bins, min_counts=50):
    """Sampled shell number-density (normalized) vs the SHELL-WEIGHTED analytic
    density (the matched estimator). Returns
    (centers, counts, rho_sampled_norm, rho_analytic_norm, rel_err, valid).

    The sampled density is normalized to the analytic curve by the median ratio
    over valid bins (shape comparison; the builder samples the profile's own
    inverse-CDF, so the *shape* is the test)."""
    bins = np.logspace(np.log10(r_lo), np.log10(r_hi), n_bins)
    centers = np.sqrt(bins[1:] * bins[:-1])

    hist, _ = np.histogram(radii, bins=bins)
    shell_v = 4.0 / 3.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    rho_samp = hist / (shell_v + 1e-30)

    rho_an = np.array(
        [
            _shell_weighted_density(profile, bins[i], bins[i + 1])
            for i in range(n_bins - 1)
        ]
    )
    rho0 = float(profile.density(jnp.array([r_lo]))[0])
    rho_an_n = rho_an / rho0

    valid = (hist > 0) & (rho_an > 0)
    # Fit the single global amplitude over the SAME well-sampled band (>= min_counts)
    # the PASS metric judges, not all valid bins (M1: removes a subtle coupling where a
    # noisy low-count tail bin perturbs the median amplitude and the well-bin residuals).
    well = (hist >= min_counts) & (rho_an > 0)
    band = well if bool(np.any(well)) else valid
    scale = np.median(rho_an_n[band] / (rho_samp[band] + 1e-30))
    rho_samp_n = rho_samp * scale
    rel = np.abs(rho_samp_n - rho_an_n) / (rho_an_n + 1e-30)
    return centers, hist, rho_samp_n, rho_an_n, rel, valid


def fig_density_recovery(output_dir):
    print("\n" + "=" * 64)
    print("FIG: sampled number-density vs analytic profile (Plummer/King/EFF)")
    print("=" * 64)

    m = jnp.ones(N_DENSITY)
    plummer = PlummerProfile(r_h=1.0)
    king = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
    eff = EFFProfile(a=1.0, gamma=4.0, r_t=30.0)

    cases = [
        (
            "Plummer",
            plummer,
            OI["blue"],
            lambda k: build_plummer_cluster(masses=m, r_h=1.0, key=k),
            0.05,
            4.0,
            26,
        ),
        (
            "King",
            king,
            OI["green"],
            lambda k: build_king_cluster(masses=m, W0=7.0, r_c=1.0, key=k),
            0.05,
            0.9 * float(king.r_t),
            26,
        ),
        (
            "EFF",
            eff,
            OI["vermilion"],
            lambda k: build_eff_cluster(masses=m, a=1.0, gamma=4.0, r_t=30.0, key=k),
            0.05,
            16.0,
            26,
        ),
    ]

    fig, (axA, axB) = plt.subplots(
        2,
        1,
        figsize=(4.2, 4.6),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.32},
    )

    # density-recovery tolerance: the sampled shell density is a finite-N Poisson
    # estimator; against the SHELL-WEIGHTED analytic density (the matched
    # estimator) the shape matches to ~20% per bin in the well-populated band.
    # Restrict the metric to bins with >= 50 counts so we judge shape recovery,
    # not deep-tail shot noise.
    tol = 0.20
    all_pass = True
    print(
        f"  {'profile':<9}{'bins(>=50)':>11}{'max rel':>10}{'med rel':>10}{'pass':>8}"
    )
    for label, prof, col, build, r_lo, r_hi, nb in cases:
        radii = np.asarray(
            jnp.linalg.norm(build(jax.random.PRNGKey(SEED)).positions, axis=1)
        )
        centers, counts, rho_s, rho_a, rel, valid = _density_recovery(
            prof, radii, r_lo, r_hi, nb
        )

        # robust metric: only bins with >=50 sampled counts (shape, not tail noise)
        well = valid & (counts >= 50)
        max_rel = float(np.max(rel[well]))
        med_rel = float(np.median(rel[well]))
        ok = max_rel < tol
        all_pass = all_pass and ok
        print(
            f"  {label:<9}{int(np.sum(well)):>11d}{max_rel:>10.3f}"
            f"{med_rel:>10.3f}{'PASS' if ok else 'FAIL':>8}"
        )

        # dense analytic curve (normalized to rho(r_lo)) + sampled shell points
        r_dense = np.logspace(np.log10(r_lo), np.log10(r_hi), 300)
        rho_dense = np.asarray(prof.density(jnp.asarray(r_dense)))
        rho_dense_n = rho_dense / float(prof.density(jnp.array([r_lo]))[0])
        axA.loglog(r_dense, rho_dense_n, "-", color=col, lw=1.6, label=label)
        axA.loglog(
            centers[valid], rho_s[valid], "o", color=col, ms=3.0, mec="none", alpha=0.6
        )
        axB.semilogx(
            centers[well], rel[well], "o", color=col, ms=3.0, mec="none", alpha=0.7
        )

    axA.set_ylabel(r"$\rho(r) / \rho(r_{\rm lo})$")
    axA.set_ylim(1e-5, 5)
    axA.set_xlabel(r"$r$ [pc]")
    axA.legend(
        loc="lower left", title="lines: analytic; points: sampled", title_fontsize=7.5
    )
    panel_label(axA, "(a)", loc="upper right")

    axB.axhline(0.0, color="0.6", lw=0.7)
    axB.axhspan(0.0, tol, color=OI["green"], alpha=0.12)
    axB.axhline(tol, color=OI["vermilion"], ls="--", lw=1.0, label=rf"tol $={tol:.2f}$")
    axB.set_xlabel(r"$r$ [pc]")
    axB.set_ylabel(r"$|\Delta\rho| / \rho$")
    axB.set_ylim(0, 0.45)
    axB.legend(loc="upper left")
    panel_label(axB, "(b)", loc="upper right")

    save_fig(fig, output_dir, "cluster_density_recovery")
    print(
        f"  density shape recovered to < {tol:.0%} per well-sampled bin  ->  "
        f"{'PASS' if all_pass else 'FAIL'}"
    )
    print("  saved cluster_density_recovery.{png,pdf}")
    return all_pass


# ============================================================================
# Figure -- tidal cut: radial mass profile with/without tidal_radius
# ============================================================================
def fig_tidal_cut(output_dir):
    print("\n" + "=" * 64)
    print("FIG: tidal cut -- radial mass profile with/without tidal_radius")
    print("=" * 64)

    r_t = 1.5
    key = jax.random.PRNGKey(SEED)
    m = jnp.ones(N_TIDAL)
    ic_full = build_plummer_cluster(masses=m, r_h=1.0, key=key)
    ic_cut = build_plummer_cluster(masses=m, r_h=1.0, key=key, tidal_radius=r_t)

    radii = np.asarray(jnp.linalg.norm(ic_cut.positions, axis=1))
    mass_full = np.asarray(ic_full.masses)
    mass_cut = np.asarray(ic_cut.masses)

    # --- physics checks ---
    outside = radii > r_t
    inside = radii <= r_t
    ghost_max_mass = float(np.max(mass_cut[outside])) if np.any(outside) else 0.0
    surv_min_mass = float(np.min(mass_cut[inside])) if np.any(inside) else 0.0
    ghosts_zero = ghost_max_mass == 0.0
    survivors_kept = surv_min_mass > 0.0
    # full build keeps every particle massive (no cut)
    full_intact = bool(np.all(mass_full > 0.0))
    # mass beyond r_t in the cut build is exactly zero (sharp, no leakage)
    leaked = float(np.sum(mass_cut[outside]))
    sharp = leaked == 0.0
    passed = ghosts_zero and survivors_kept and full_intact and sharp

    print(f"  {'quantity':<34}{'expected':>11}{'measured':>12}{'pass':>8}")
    print(
        f"  {'max ghost mass (r>r_t)':<34}{0.0:>11.1f}{ghost_max_mass:>12.3e}"
        f"{'PASS' if ghosts_zero else 'FAIL':>8}"
    )
    print(
        f"  {'min survivor mass (r<=r_t)':<34}{'>0':>11}{surv_min_mass:>12.3f}"
        f"{'PASS' if survivors_kept else 'FAIL':>8}"
    )
    print(
        f"  {'leaked mass beyond r_t':<34}{0.0:>11.1f}{leaked:>12.3e}"
        f"{'PASS' if sharp else 'FAIL':>8}"
    )
    print(
        f"  {'full build all massive':<34}{'True':>11}{str(full_intact):>12}"
        f"{'PASS' if full_intact else 'FAIL':>8}"
    )
    surv_frac = float(np.mean(inside))
    print(
        f"  survivor fraction (r<=r_t={r_t}): {surv_frac:.3f}  "
        f"({int(np.sum(inside))}/{N_TIDAL})"
    )

    # --- cumulative enclosed mass M(<r) ---
    order = np.argsort(radii)
    r_sorted = radii[order]
    cum_full = np.cumsum(mass_full[order])
    cum_cut = np.cumsum(mass_cut[order])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 3.2))

    axA.plot(r_sorted, cum_full, "-", color=OI["blue"], lw=1.8, label="no tidal cut")
    axA.plot(
        r_sorted,
        cum_cut,
        "-",
        color=OI["vermilion"],
        lw=1.8,
        label=rf"tidal cut $r_t={r_t}$",
    )
    axA.axvline(r_t, color="0.5", ls=":", lw=1.0)
    axA.text(r_t * 1.03, 0.05 * cum_full[-1], r"$r_t$", color="0.4", fontsize=9)
    axA.set_xlabel(r"$r$ [pc]")
    axA.set_ylabel(r"enclosed mass $M(<r)$ [$M_\odot$]")
    axA.set_xlim(0, 5)
    axA.legend(loc="lower right")
    panel_label(axA, "(a)", loc="upper left")

    # per-particle mass vs radius: survivors at m=1, ghosts at m=0 beyond r_t
    sub = slice(None, None, max(1, N_TIDAL // 4000))
    axB.plot(
        radii[sub],
        mass_cut[sub],
        "o",
        color=OI["vermilion"],
        ms=2.4,
        mec="none",
        alpha=0.4,
    )
    axB.axvline(r_t, color="0.5", ls=":", lw=1.0)
    axB.text(r_t * 1.04, 0.5, r"$r_t$", color="0.4", fontsize=9)
    axB.text(
        0.55,
        1.05,
        "survivors",
        color=OI["vermilion"],
        fontsize=8.5,
        transform=axB.get_yaxis_transform(),
        ha="left",
    )
    axB.text(2.4, 0.08, r"zero-mass ghosts", color="0.35", fontsize=8.5)
    axB.set_xlabel(r"$r$ [pc]")
    axB.set_ylabel(r"particle mass $m_i$ [$M_\odot$]")
    axB.set_xlim(0, 5)
    axB.set_ylim(-0.1, 1.25)
    panel_label(axB, "(b)", loc="upper right")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "cluster_tidal_cut")
    print(
        f"  sharp cut + zero-mass ghosts beyond r_t  ->  {'PASS' if passed else 'FAIL'}"
    )
    print("  saved cluster_tidal_cut.{png,pdf}")
    return passed


# ============================================================================
# Figure -- rotation: L_z(omega) linearity + differential v_phi(R)
# ============================================================================
def fig_rotation_Lz(output_dir):
    print("\n" + "=" * 64)
    print("FIG: rotation -- L_z(omega) linearity + differential v_phi(R)")
    print("=" * 64)

    key = jax.random.PRNGKey(SEED)
    m = jnp.ones(N_ROT)

    # base (no rotation) sample -> the analytic L_z slope is exactly Sigma m R^2,
    # where R is cylindrical radius (solid-body v_rot = omega x r, axis = z-hat,
    # so the added L_z(omega) = omega * sum m (x^2 + y^2)).
    ic0 = build_plummer_cluster(masses=m, r_h=1.0, key=key)
    x0, y0 = np.asarray(ic0.positions[:, 0]), np.asarray(ic0.positions[:, 1])
    masses0 = np.asarray(ic0.masses)
    Sigma_mR2 = float(np.sum(masses0 * (x0**2 + y0**2)))

    omegas = np.linspace(0.0, 0.4, 9)
    Lz = []
    for w in omegas:
        ic = build_plummer_cluster(masses=m, r_h=1.0, key=key, rotation=float(w))
        xx, yy = np.asarray(ic.positions[:, 0]), np.asarray(ic.positions[:, 1])
        vx, vy = np.asarray(ic.velocities[:, 0]), np.asarray(ic.velocities[:, 1])
        Lz.append(float(np.sum(np.asarray(ic.masses) * (xx * vy - yy * vx))))
    Lz = np.array(Lz)
    slope, intercept = np.polyfit(omegas, Lz, 1)
    slope_rel = abs(slope - Sigma_mR2) / Sigma_mR2
    # The rotation overlay leaves positions bit-identical, so L_z = L_z0 + omega*Sigma m R^2
    # is EXACT to float64 round-off -> a strong sentinel (measured ~5e-16, ~7 orders under
    # this bound). 1e-9 still catches any real regression (wrong axis / position perturbation)
    # while staying far above cross-arch round-off (M3).
    slope_pass = slope_rel < 1e-9

    print(
        f"  {'quantity':<28}{'expected':>14}{'measured':>14}{'rel err':>11}{'pass':>7}"
    )
    print(
        f"  {'L_z slope d L_z/d omega':<28}{Sigma_mR2:>14.3f}{slope:>14.3f}"
        f"{slope_rel:>11.2e}{'PASS' if slope_pass else 'FAIL':>7}"
    )
    print(
        f"  fit intercept (base sample L_z) = {intercept:.3f} "
        f"(small random residual, not a target)"
    )

    # differential rotation curve v_phi(R) = v_peak (R/R_peak) exp(1 - R/R_peak)
    v_peak, R_peak = 2.0, 1.0
    spec = RotationSpec(kind="differential", v_peak=v_peak, R_peak=R_peak)
    ic_d = build_plummer_cluster(
        masses=jnp.ones(N_DIFF), r_h=1.0, key=key, rotation=spec
    )
    pos_d, vel_d = np.asarray(ic_d.positions), np.asarray(ic_d.velocities)
    R = np.sqrt(pos_d[:, 0] ** 2 + pos_d[:, 1] ** 2)
    vphi = (pos_d[:, 0] * vel_d[:, 1] - pos_d[:, 1] * vel_d[:, 0]) / np.maximum(
        R, 1e-10
    )
    edges = np.linspace(0.2, 2.6, 9)
    Rmid, vmeas, verr, van = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (R >= lo) & (R < hi)
        n = int(np.sum(msk))
        rm = float(np.mean(R[msk]))
        Rmid.append(rm)
        vmeas.append(float(np.mean(vphi[msk])))
        verr.append(float(np.std(vphi[msk]) / np.sqrt(max(n, 1))))
        van.append(v_peak * (rm / R_peak) * np.exp(1.0 - rm / R_peak))
    vmeas, van, verr = np.array(vmeas), np.array(van), np.array(verr)
    # mean v_phi includes the base sample's random azimuthal component, so the bin
    # means carry std-error scatter; require the overlay curve to be recovered to
    # 15% in the well-populated inner band (R <= R_peak * 2.2).
    inner = np.array(Rmid) <= 2.2
    diff_rel_inner = float(
        np.max(np.abs(vmeas[inner] - van[inner]) / (van[inner] + 1e-9))
    )
    diff_pass = diff_rel_inner < 0.15
    print(
        f"  differential v_phi(R) overlay: max rel (R<=2.2) = {diff_rel_inner:.3f}  "
        f"(tol 0.15)  ->  {'PASS' if diff_pass else 'FAIL'}"
    )

    passed = slope_pass and diff_pass

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 3.2))

    axA.plot(
        omegas,
        Sigma_mR2 * omegas + intercept,
        "-",
        color=OI["black"],
        lw=1.6,
        label=r"$\Sigma m R^2\,\omega + b$",
    )
    axA.plot(
        omegas,
        Lz,
        "o",
        color=OI["blue"],
        ms=5,
        mfc="none",
        mew=1.3,
        label="measured $L_z$",
    )
    axA.set_xlabel(r"$\omega$ [rad Myr$^{-1}$]")
    axA.set_ylabel(r"net $L_z$ [$M_\odot$ pc$^2$ Myr$^{-1}$]")
    axA.legend(loc="upper left")
    axA.text(
        0.5,
        0.06,
        rf"slope $={slope:.1f}$ vs $\Sigma m R^2={Sigma_mR2:.1f}$"
        "\n"
        rf"(rel err ${slope_rel:.0e}$)",
        transform=axA.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5),
    )
    panel_label(axA, "(a)", loc="lower right")

    R_curve = np.linspace(0.05, 2.6, 200)
    v_curve = v_peak * (R_curve / R_peak) * np.exp(1.0 - R_curve / R_peak)
    axB.plot(
        R_curve,
        v_curve,
        "-",
        color=OI["black"],
        lw=1.6,
        label=r"$v_{\rm peak}\frac{R}{R_p}e^{1-R/R_p}$",
    )
    axB.errorbar(
        Rmid,
        vmeas,
        yerr=verr,
        fmt="s",
        color=OI["vermilion"],
        ms=5,
        capsize=2.5,
        lw=1.0,
        label=r"binned $\langle v_\phi\rangle$",
    )
    axB.axvline(R_peak, color="0.5", ls=":", lw=1.0)
    axB.text(R_peak * 1.04, 0.3, r"$R_{\rm peak}$", color="0.4", fontsize=8.5)
    axB.set_xlabel(r"cylindrical $R$ [pc]")
    axB.set_ylabel(r"$\langle v_\phi\rangle(R)$ [pc Myr$^{-1}$]")
    axB.set_xlim(0, 2.6)
    axB.set_ylim(0, None)
    axB.legend(loc="upper right")
    panel_label(axB, "(b)", loc="upper left")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "cluster_rotation_Lz")
    print(f"  rotation L_z + v_phi(R) recovered  ->  {'PASS' if passed else 'FAIL'}")
    print("  saved cluster_rotation_Lz.{png,pdf}")
    return passed


# ============================================================================
# Figure -- OM anisotropy: beta(r) vs analytic r^2/(r^2 + r_a^2)
# ============================================================================
def fig_anisotropy_beta(output_dir):
    print("\n" + "=" * 64)
    print("FIG: anisotropy beta(r) for OM Plummer vs r^2/(r^2 + r_a^2)")
    print("=" * 64)

    r_a = 0.8  # Osipkov-Merritt anisotropy radius [pc]
    key = jax.random.PRNGKey(SEED)
    m = jnp.ones(N_ANISO)
    # Q=None -> faithful UNSCALED OM equilibrium (a virial rescale would multiply
    # v_r and v_t uniformly and leave beta unchanged, but Q=None keeps the DF pure).
    ic = build_plummer_cluster(
        masses=m, r_h=1.0, key=key, anisotropy_radius=r_a, Q=None
    )
    pos, vel = np.asarray(ic.positions), np.asarray(ic.velocities)
    r = np.linalg.norm(pos, axis=1)
    rhat = pos / (r[:, None] + 1e-12)
    vr = np.sum(vel * rhat, axis=1)
    vt2 = np.sum(vel**2, axis=1) - vr**2

    edges = np.array([0.25, 0.45, 0.7, 1.0, 1.4, 2.0, 3.0])
    rmid, beta_meas, beta_an, beta_err = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (r >= lo) & (r < hi)
        n = int(np.sum(msk))
        rm = float(np.mean(r[msk]))
        s_r2 = float(np.mean(vr[msk] ** 2))
        s_t2 = float(np.mean(vt2[msk]))
        b = 1.0 - s_t2 / (2.0 * s_r2)
        rmid.append(rm)
        beta_meas.append(b)
        beta_an.append(rm**2 / (rm**2 + r_a**2))
        # bootstrap-free std-error on beta via per-particle delta-method scatter
        beta_err.append(0.5 * float(np.std(vt2[msk]) / s_r2) / np.sqrt(max(n, 1)))
    rmid = np.array(rmid)
    beta_meas = np.array(beta_meas)
    beta_an = np.array(beta_an)
    beta_err = np.array(beta_err)
    abs_dev = np.abs(beta_meas - beta_an)
    max_dev = float(np.max(abs_dev))
    tol = 0.05
    passed = max_dev < tol

    print(
        f"  {'r [pc]':>8}{'beta_meas':>12}{'beta_analytic':>15}{'|dev|':>9}{'pass':>7}"
    )
    for rm, bm, ba, ad in zip(rmid, beta_meas, beta_an, abs_dev):
        print(
            f"  {rm:>8.3f}{bm:>12.4f}{ba:>15.4f}{ad:>9.4f}"
            f"{'PASS' if ad < tol else 'FAIL':>7}"
        )
    print(
        f"  max |beta_meas - beta_analytic| = {max_dev:.4f}  (tol {tol})  "
        f"->  {'PASS' if passed else 'FAIL'}"
    )

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    r_curve = np.linspace(0.05, 3.0, 200)
    ax.plot(
        r_curve,
        r_curve**2 / (r_curve**2 + r_a**2),
        "-",
        color=OI["black"],
        lw=1.8,
        label=r"analytic $\beta(r)=r^2/(r^2+r_a^2)$",
    )
    ax.errorbar(
        rmid,
        beta_meas,
        yerr=beta_err,
        fmt="o",
        color=OI["vermilion"],
        ms=5,
        capsize=2.5,
        lw=1.0,
        label=rf"sampled ($N={N_ANISO:,}$)",
    )
    ax.axvline(r_a, color="0.5", ls=":", lw=1.0)
    ax.text(r_a * 1.04, 0.05, r"$r_a$", color="0.4", fontsize=9)
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel(r"$r$ [pc]")
    ax.set_ylabel(r"anisotropy $\beta(r) = 1 - \sigma_t^2 / (2\sigma_r^2)$")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc="lower right")
    ax.text(
        0.04,
        0.94,
        rf"max $|\Delta\beta|={max_dev:.3f}$",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "cluster_anisotropy_beta")
    print("  saved cluster_anisotropy_beta.{png,pdf}")
    return passed


def main():
    print("\n" + "=" * 70)
    print("PROGENAX CLUSTER-BUILDER CONVENIENCE-API VALIDATION")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "virial Q per alias (5)": table_virial_per_alias(),
        "density recovery (Plummer/King/EFF)": fig_density_recovery(OUTPUT_DIR),
        "tidal cut (mass profile + ghosts)": fig_tidal_cut(OUTPUT_DIR),
        "rotation L_z(omega) + v_phi(R)": fig_rotation_Lz(OUTPUT_DIR),
        "anisotropy beta(r) (OM Plummer)": fig_anisotropy_beta(OUTPUT_DIR),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print(
        "  ALL CLUSTER-BUILDER VALIDATION CHECKS PASS"
        if all_ok
        else "  SOME CLUSTER-BUILDER VALIDATION CHECKS FAILED"
    )
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/cluster_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
