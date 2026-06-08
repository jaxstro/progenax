#!/usr/bin/env python
"""Validate released-core cluster initial-condition generation + plots.

Validates the *live* progenax cluster-IC API against physical expectations:

1. ``build_spatial_ic`` — a Plummer IC at Q_target=0.5 is in virial equilibrium
   (measured Q = T/|V| ≈ 0.5) and bound; sampled density matches the analytic profile.
2. ``generate_two_component_cluster`` — a concentrated + an extended component give
   the expected radial separation (median r_A < median r_B).
3. ``energy_sorted_segregation`` — energy-ordered orbit assignment produces real mass
   segregation: Λ_MSR (the validated diagnostic) rises above the unsegregated value, and
   massive stars occupy more-bound orbits.

History: this script previously imported the removed ``progenax.cluster.fractal_gw_legacy``
(dead after the gravoturbulent-FDF clean-room rewrite moved fractal/turbulent ICs to the
experimental ``gravoturb_fdf`` package). Rewritten 2026-06-08 to target the live released-core
API only. Turbulent/fractal IC validation now lives in ``gravoturb_fdf/validation/``.

Usage:
    PYTHONPATH=src python scripts/validate_cluster_ic.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from jaxstro.units import STELLAR  # noqa: E402
from progenax import (  # noqa: E402
    PlummerProfile,
    PlummerVelocityDF,
    TwoComponentConfig,
    build_spatial_ic,
    compute_kinetic_energy,
    compute_potential_energy,
    generate_two_component_cluster,
)
from progenax.cluster.mass_segregation import energy_sorted_segregation  # noqa: E402
from progenax.diagnostics import compute_lambda_msr  # noqa: E402

G = STELLAR.G
EPS = 0.01  # pc softening for per-star binding energy
PLOT_DIR = Path(__file__).parent.parent / "validation" / "plots"


def _specific_potential(positions, masses, eps=EPS):
    """Per-star gravitational potential φ_i = −G Σ_j m_j / sqrt(r_ij² + eps²) (numpy)."""
    p = np.asarray(positions); m = np.asarray(masses)
    d2 = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1) + eps**2
    inv = 1.0 / np.sqrt(d2)
    np.fill_diagonal(inv, 0.0)
    return -G * (inv @ m)


# ----------------------------------------------------------------- 1. build_spatial_ic
def validate_build_spatial_ic(N=600):
    print("\n=== 1. build_spatial_ic — Plummer virial equilibrium ===")
    masses = jnp.ones(N)
    ic = build_spatial_ic(PlummerProfile(r_h=1.0), masses, PlummerVelocityDF(r_h=1.0),
                          key=jax.random.PRNGKey(0), G=G, Q=0.5)
    T = float(compute_kinetic_energy(ic.velocities, ic.masses))
    V = float(compute_potential_energy(ic.positions, ic.masses, G=G))
    Q = T / abs(V)
    phi = _specific_potential(ic.positions, ic.masses)
    E = 0.5 * np.sum(np.asarray(ic.velocities) ** 2, axis=1) + phi
    bound = float(np.mean(E < 0))
    ok = abs(Q - 0.5) < 0.05 and bound > 0.95
    print(f"  measured Q=T/|V| = {Q:.3f} (target 0.5);  bound fraction = {bound:.2%}  "
          f"{'PASS' if ok else 'FAIL'}")

    # plot: sampled radial density vs analytic Plummer
    r = np.linalg.norm(np.asarray(ic.positions) - np.asarray(ic.positions).mean(0), axis=1)
    edges = np.linspace(0, 4, 25); cen = 0.5 * (edges[1:] + edges[:-1])
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    dens = np.histogram(r, bins=edges)[0] / shell
    pl = np.asarray(PlummerProfile(r_h=1.0).density(jnp.asarray(cen)))
    iref = int(np.argmin(np.abs(cen - 1.0)))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cen, dens / dens[iref], "o-", label="sampled")
    ax.plot(cen, pl / pl[iref], "k--", label="analytic Plummer")
    ax.set_yscale("log"); ax.set_xlabel("r (pc)"); ax.set_ylabel("ρ(r)/ρ(r_h)")
    ax.set_title(f"build_spatial_ic: Plummer IC (Q={Q:.2f}, bound={bound:.0%})"); ax.legend()
    fig.savefig(PLOT_DIR / "cluster_ic_plummer_equilibrium.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return ok


# ----------------------------------------------------------------- 2. two-component
def validate_two_component(N=800):
    print("\n=== 2. generate_two_component_cluster — concentrated + extended ===")
    masses = jnp.ones(N)
    config = TwoComponentConfig(
        f_A=0.5,
        profile_A=PlummerProfile(r_h=0.5), profile_B=PlummerProfile(r_h=2.0),
        velocity_df_A=PlummerVelocityDF(r_h=0.5), velocity_df_B=PlummerVelocityDF(r_h=2.0),
    )
    mask = jnp.arange(N) < N // 2  # first half = component A
    pos, vel, m = generate_two_component_cluster(masses, config, key=jax.random.PRNGKey(1),
                                                 G=G, pop_mask=mask)
    pos = np.asarray(pos); mask = np.asarray(mask)
    r = np.linalg.norm(pos - pos.mean(0), axis=1)
    med_A, med_B = float(np.median(r[mask])), float(np.median(r[~mask]))
    ok = med_A < med_B
    print(f"  median r: component A (r_h=0.5) = {med_A:.2f} pc  <  B (r_h=2.0) = {med_B:.2f} pc  "
          f"{'PASS' if ok else 'FAIL'}")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(r[mask], bins=30, alpha=0.6, density=True, label="A (concentrated, r_h=0.5)")
    ax.hist(r[~mask], bins=30, alpha=0.6, density=True, label="B (extended, r_h=2.0)")
    ax.set_xlabel("r (pc)"); ax.set_ylabel("density"); ax.legend()
    ax.set_title("generate_two_component_cluster: radial separation of the two components")
    fig.savefig(PLOT_DIR / "cluster_ic_two_component.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return ok


# ----------------------------------------------------------------- 3. energy-sorted segregation
def validate_energy_sorted_segregation(N=500):
    print("\n=== 3. energy_sorted_segregation — Λ_MSR rises, massive stars more bound ===")
    from progenax.imf import PowerLawIMF

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2), 3)
    masses = PowerLawIMF.kroupa().sample(k1, N)
    # unsegregated orbit pool from a Plummer IC
    ic = build_spatial_ic(PlummerProfile(r_h=1.0), masses, PlummerVelocityDF(r_h=1.0),
                          key=k2, G=G, Q=0.5)
    pos_pool, vel_pool = ic.positions, ic.velocities

    def potential_fn(p):
        return jnp.asarray(_specific_potential(p, masses))

    m_seg, pos_seg, vel_seg = energy_sorted_segregation(k3, masses, pos_pool, vel_pool, potential_fn)

    lam_unseg, _ = compute_lambda_msr(np.asarray(pos_pool), np.asarray(masses),
                                      N_massive=20, N_random_samples=200, seed=1)
    lam_seg, _ = compute_lambda_msr(np.asarray(pos_seg), np.asarray(m_seg),
                                    N_massive=20, N_random_samples=200, seed=1)
    E = 0.5 * np.sum(np.asarray(vel_seg) ** 2, axis=1) + _specific_potential(pos_seg, m_seg)
    rho_mE, _ = spearmanr(np.asarray(m_seg), E)  # massive → more bound ⇒ negative
    ok = lam_seg > lam_unseg and lam_seg > 1.3 and rho_mE < -0.3
    print(f"  Λ_MSR: unsegregated {lam_unseg:.2f} → segregated {lam_seg:.2f};  "
          f"ρ(mass, E) = {rho_mE:+.2f}  {'PASS' if ok else 'FAIL'}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(["unsegregated", "energy-sorted"], [lam_unseg, lam_seg], color=["0.6", "crimson"])
    axes[0].axhline(1.0, ls=":", c="k"); axes[0].set_ylabel("Λ_MSR")
    axes[0].set_title("Λ_MSR rises after energy-sorted segregation")
    axes[1].scatter(np.asarray(m_seg), E, s=8, alpha=0.5)
    axes[1].set_xscale("log"); axes[1].set_xlabel("mass (M⊙)"); axes[1].set_ylabel("binding energy E")
    axes[1].set_title(f"massive stars more bound (ρ={rho_mE:+.2f})")
    fig.savefig(PLOT_DIR / "cluster_ic_energy_sorted_segregation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return ok


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print("Released-core cluster-IC validation (build_spatial_ic / two-component / segregation)")
    print("=" * 64)
    results = {
        "build_spatial_ic": validate_build_spatial_ic(),
        "two_component": validate_two_component(),
        "energy_sorted_segregation": validate_energy_sorted_segregation(),
    }
    print("\n" + "=" * 64)
    for k, v in results.items():
        print(f"  {k:<32} {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print("Overall:", "ALL PASS" if ok else "SOME FAILED")
    print("plots →", PLOT_DIR)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
