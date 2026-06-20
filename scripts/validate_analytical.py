#!/usr/bin/env python
"""
Analytical test-case validation figures.

Five publication-quality figures for progenax's exact-solution IC builders, anchored
to passing tests in ``tests/validation/test_analytical_physics.py`` (12 tests) and
printing expected-vs-measured PASS/FAIL against analytic invariants (Kepler's laws,
the figure-eight L=0 / closure, the simple-harmonic solution). The orbits are
integrated in-script with symplectic velocity-Verlet (the same scheme the tests use);
the ORACLE is always the analytic invariant, not the integrator.

Figures (-> what they validate):
  1. analytical_two_body.png    eccentric Kepler ellipse closes after one period;
                                E = -G m1 m2 / 2a and angular momentum conserved
  2. analytical_figure_eight.png  the Chenciner-Montgomery choreography: three equal
                                masses trace one figure-eight; L = 0, closes
  3. analytical_kepler_third.png  Kepler III T^2 ∝ a^3 for the eight planets (JPL
                                table) vs the observed sidereal periods; IAU mass ratios
  4. analytical_harmonic.png    x(t) = A cos(ωt+φ) exact vs integrated; E = ½ m ω² A²
  5. analytical_figure_eight_adversarial.png  the canonical IC closes (L=0) while a
                                plausible 3-fold *spatial* rotation fails closure + L≠0

Provenance: the figure-eight initial conditions are Chenciner & Montgomery (2000),
Ann. Math. 152, 881 (with Simó 2001 numerical coefficients) — PDF not held; the
planet data are JPL Horizons (J2000.0) with IAU 2009 mass ratios.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_analytical.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from progenax.analytical import (
    SOLAR_SYSTEM_PLANETS,
    get_planet,
    harmonic_oscillator,
    three_body_figure_eight,
    two_body_kepler,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G_PLANET = 39.478  # PLANETARY units (AU^3 Msun^-1 yr^-2)

# observed sidereal periods [yr] (the external oracle for Kepler III)
OBS_PERIOD_YR = {
    "Mercury": 0.2408,
    "Venus": 0.6152,
    "Earth": 1.0,
    "Mars": 1.881,
    "Jupiter": 11.862,
    "Saturn": 29.457,
    "Uranus": 84.02,
    "Neptune": 164.79,
}


# ============================================================================
# Symplectic velocity-Verlet (records the trajectory)
# ============================================================================
def _accel(pos, masses, G):
    diff = pos[None, :, :] - pos[:, None, :]
    r = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    r = jnp.where(jnp.eye(pos.shape[0], dtype=bool), jnp.inf, r)
    return G * jnp.sum(masses[None, :, None] * diff / r[:, :, None] ** 3, axis=1)


def _verlet_traj(pos0, vel0, masses, G, T, n):
    """Velocity-Verlet for time T in n steps; returns (positions[n,N,3], final p, v)."""
    dt = T / n

    def step(c, _):
        p, v = c
        v = v + 0.5 * dt * _accel(p, masses, G)
        p = p + dt * v
        v = v + 0.5 * dt * _accel(p, masses, G)
        return (p, v), p

    (pf, vf), traj = jax.lax.scan(step, (pos0, vel0), None, length=n)
    return traj, pf, vf


def _energy(pos, vel, masses, G):
    KE = 0.5 * jnp.sum(masses * jnp.sum(vel**2, axis=1))
    diff = pos[None, :, :] - pos[:, None, :]
    r = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    r = jnp.where(jnp.eye(pos.shape[0], dtype=bool), jnp.inf, r)
    PE = -0.5 * G * jnp.sum(masses[:, None] * masses[None, :] / r)
    return KE + PE


def _ang_mom(pos, vel, masses):
    return jnp.sum(masses[:, None] * jnp.cross(pos, vel), axis=0)


# ============================================================================
# Figure 1 -- two-body Kepler ellipse + conservation
# ============================================================================
def fig_two_body(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: two-body Kepler ellipse + energy/L conservation")
    print("=" * 60)
    G, M1, M2, a, e = 1.0, 1.0, 0.1, 1.0, 0.5
    ic = two_body_kepler(M1=M1, M2=M2, a=a, e=e, G=G)
    n = 120_000
    traj, pf, vf = _verlet_traj(ic.positions, ic.velocities, ic.masses, G, ic.period, n)
    closure = float(jnp.max(jnp.linalg.norm(pf - ic.positions, axis=1)))
    E0 = float(_energy(ic.positions, ic.velocities, ic.masses, G))
    E1 = float(_energy(pf, vf, ic.masses, G))
    L0 = _ang_mom(ic.positions, ic.velocities, ic.masses)
    L1 = _ang_mom(pf, vf, ic.masses)
    dE = abs((E1 - E0) / E0)
    dL = float(jnp.linalg.norm(L1 - L0))
    E_exact = -G * M1 * M2 / (2 * a)
    e_match = abs(E0 - E_exact) < 1e-12
    closed = closure < 1e-4
    cons = dE < 1e-5 and dL < 1e-6
    print(
        f"  E = {E0:.6f} (exact -G m1 m2/2a = {E_exact:.6f})  -> {'PASS' if e_match else 'FAIL'}"
    )
    print(
        f"  closure after 1 period = {closure:.2e} (tol 1e-4)  -> {'PASS' if closed else 'FAIL'}"
    )
    print(
        f"  dE/E = {dE:.2e} (tol 1e-5), d|L| = {dL:.2e} (tol 1e-6)  "
        f"-> {'PASS' if cons else 'FAIL'}"
    )

    tr = np.asarray(traj)  # (n, 2, 3)
    rel = tr[:, 1, :] - tr[:, 0, :]  # relative orbit (body2 - body1)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    axA.plot(rel[:, 0], rel[:, 1], "-", color=OI["blue"], lw=1.3)
    axA.plot(0, 0, "+", color=OI["black"], ms=10, mew=1.4, label="focus (COM)")
    axA.plot(
        rel[0, 0],
        rel[0, 1],
        "o",
        color=OI["vermilion"],
        ms=6,
        mec="white",
        label="start = end",
    )
    axA.set_xlabel(r"$x$ [code]")
    axA.set_ylabel(r"$y$ [code]")
    axA.set_aspect("equal")
    axA.legend(loc="upper right", fontsize=7)
    axA.text(
        0.04,
        0.05,
        rf"$e={e}$, closure $={closure:.0e}$",
        transform=axA.transAxes,
        fontsize=7.5,
        color="0.4",
    )
    panel_label(axA, "(a)", loc="upper left")

    # conservation bar (end-state deviations vs tolerance)
    axB.axhline(1.0, color="0.5", ls="--", lw=1.0, label="tolerance")
    bars = {
        r"$|E-E_{\rm exact}|$": abs(E0 - E_exact) / 1e-12,
        r"$\Delta E/E$": dE / 1e-5,
        r"$\Delta|L|$": dL / 1e-6,
        "closure": closure / 1e-4,
    }
    axB.bar(
        range(len(bars)),
        [max(v, 1e-3) for v in bars.values()],
        color=[OI["green"] if v < 1 else OI["vermilion"] for v in bars.values()],
        edgecolor="white",
    )
    axB.set_yscale("log")
    axB.set_ylim(1e-3, 2)
    axB.set_xticks(range(len(bars)))
    axB.set_xticklabels(list(bars.keys()), fontsize=7)
    axB.set_ylabel("measured / tolerance")
    axB.legend(loc="upper right", fontsize=7)
    panel_label(axB, "(b)", loc="upper left")

    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "analytical_two_body")
    print("  saved analytical_two_body.{png,pdf}")
    return e_match and closed and cons


# ============================================================================
# Figure 2 -- figure-eight choreography
# ============================================================================
def fig_figure_eight(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: three-body figure-eight choreography (L=0, closes)")
    print("=" * 60)
    ic = three_body_figure_eight(mass=1.0, scale=1.0, G=1.0)
    n = 200_000
    traj, pf, vf = _verlet_traj(
        ic.positions, ic.velocities, ic.masses, 1.0, ic.period, n
    )
    closure = float(jnp.max(jnp.linalg.norm(pf - ic.positions, axis=1)))
    L = float(jnp.linalg.norm(_ang_mom(ic.positions, ic.velocities, ic.masses)))
    closed = closure < 1e-6
    zeroL = L < 1e-10
    print(f"  |L| = {L:.2e} (tol 1e-10)  -> {'PASS' if zeroL else 'FAIL'}")
    print(
        f"  closure after 1 period = {closure:.2e} (tol 1e-6)  -> {'PASS' if closed else 'FAIL'}"
    )

    tr = np.asarray(traj)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    # the single shared figure-eight curve (all three bodies trace it)
    ax.plot(tr[:, 0, 0], tr[:, 0, 1], "-", color="0.6", lw=1.0, zorder=1)
    cols = [OI["blue"], OI["vermilion"], OI["green"]]
    for i, c in enumerate(cols):
        ax.plot(
            ic.positions[i, 0],
            ic.positions[i, 1],
            "o",
            color=c,
            ms=8,
            mec="white",
            mew=0.8,
            zorder=3,
            label=f"body {i + 1} (start)",
        )
    ax.plot(0, 0, "+", color=OI["black"], ms=9, mew=1.3, zorder=2)
    ax.set_xlabel(r"$x$ [code]")
    ax.set_ylabel(r"$y$ [code]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=6.8)
    ax.text(
        0.04,
        0.06,
        rf"$|L|={L:.0e}$" + "\n" + rf"closure $={closure:.0e}$",
        transform=ax.transAxes,
        fontsize=7.5,
        color="0.4",
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "analytical_figure_eight")
    print("  saved analytical_figure_eight.{png,pdf}")
    return closed and zeroL


# ============================================================================
# Figure 3 -- Kepler's third law across the solar system
# ============================================================================
def fig_kepler_third(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: Kepler III (T^2 ∝ a^3) for the eight planets + IAU mass ratios")
    print("=" * 60)
    names = [p["name"] for p in SOLAR_SYSTEM_PLANETS]
    a = np.array([float(p["a"]) for p in SOLAR_SYSTEM_PLANETS])
    Mp = np.array([float(p["M"]) for p in SOLAR_SYSTEM_PLANETS])
    T_kepler = 2 * np.pi * np.sqrt(a**3 / (G_PLANET * (1.0 + Mp)))  # yr
    T_obs = np.array([OBS_PERIOD_YR[n] for n in names])
    rel = np.abs(T_kepler - T_obs) / T_obs
    kep_pass = bool(np.max(rel) < 0.01)
    print(
        f"  Kepler III T(a) vs observed period: max rel = {np.max(rel):.2e} "
        f"(tol 1e-2)  -> {'PASS' if kep_pass else 'FAIL'}"
    )

    # IAU 2009 mass-ratio provenance
    iau = {"Earth": 332946.0, "Jupiter": 1047.35, "Saturn": 3497.9}
    mass_ok = True
    for nm, ref in iau.items():
        got = 1.0 / float(get_planet(nm)["M"])
        ok = abs(got - ref) / ref < 2e-3
        mass_ok &= ok
        print(f"  1/M_{nm} = {got:.1f} (IAU {ref})  -> {'PASS' if ok else 'FAIL'}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.3, 3.1))
    aa = np.logspace(np.log10(0.3), np.log10(35), 100)
    axA.loglog(
        aa**3,
        (2 * np.pi * np.sqrt(aa**3 / G_PLANET)) ** 2,
        "-",
        color=OI["black"],
        lw=1.4,
        label=r"$T^2 = \frac{4\pi^2}{GM_\odot}a^3$",
    )
    axA.loglog(
        a**3,
        T_obs**2,
        "o",
        color=OI["vermilion"],
        ms=6,
        mec="white",
        mew=0.6,
        label="planets (observed)",
    )
    for nm, ai, ti in zip(names, a, T_obs):
        axA.annotate(
            nm[:3],
            (ai**3, ti**2),
            textcoords="offset points",
            xytext=(4, -2),
            fontsize=5.5,
            color="0.4",
        )
    axA.set_xlabel(r"$a^3$ [AU$^3$]")
    axA.set_ylabel(r"$T^2$ [yr$^2$]")
    axA.legend(loc="upper left", fontsize=7)
    panel_label(axA, "(a)", loc="lower right")

    axB.bar(
        range(len(names)), np.maximum(rel, 1e-6), color=OI["blue"], edgecolor="white"
    )
    axB.axhline(1e-2, color=OI["vermilion"], ls="--", lw=1.2, label=r"$1\%$")
    axB.set_yscale("log")
    axB.set_ylim(1e-6, 3e-2)
    axB.set_xticks(range(len(names)))
    axB.set_xticklabels([n[:3] for n in names], rotation=40, ha="right", fontsize=6.5)
    axB.set_ylabel(r"rel. err  $T_{\rm Kepler}$ vs observed")
    axB.legend(loc="upper right", fontsize=7)
    panel_label(axB, "(b)", loc="upper left")

    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "analytical_kepler_third")
    print("  saved analytical_kepler_third.{png,pdf}")
    return kep_pass and mass_ok


# ============================================================================
# Figure 4 -- harmonic oscillator
# ============================================================================
def fig_harmonic(output_dir):
    print("\n" + "=" * 60)
    print("FIG 4: harmonic oscillator x(t)=A cos(ωt+φ) + energy")
    print("=" * 60)
    A_amp, omega, phase, m = 1.0, 1.0, 0.3, 1.0
    ic = harmonic_oscillator(
        amplitude=A_amp, omega=omega, phase=phase, mass=m, dimension="1D"
    )
    # analytic solution over one period; integrate the SHO ODE for comparison
    T = 2 * np.pi / omega
    t = np.linspace(0, T, 400)
    x_an = A_amp * np.cos(omega * t + phase)

    # integrate x'' = -omega^2 x with velocity-Verlet from the IC
    x0 = float(ic.positions[0, 0])
    v0 = float(ic.velocities[0, 0])
    n = 4000
    dt = T / n
    xs, vs = [x0], [v0]
    x, v = x0, v0
    for _ in range(n):
        v = v - 0.5 * dt * omega**2 * x
        x = x + dt * v
        v = v - 0.5 * dt * omega**2 * x
        xs.append(x)
        vs.append(v)
    t_int = np.linspace(0, T, n + 1)
    x_int = np.array(xs)

    ic_match = abs(x0 - A_amp * np.cos(phase)) < 1e-12
    x_int_on_grid = np.interp(t, t_int, x_int)
    max_dev = float(np.max(np.abs(x_int_on_grid - x_an)))
    E = 0.5 * m * np.array(vs) ** 2 + 0.5 * m * omega**2 * np.array(xs) ** 2
    E_exact = 0.5 * m * omega**2 * A_amp**2
    dE = float(np.max(np.abs(E - E_exact)) / E_exact)
    sol_ok = max_dev < 1e-3 and ic_match
    cons = dE < 1e-4
    print(
        f"  IC x(0)=A cos(phi): {x0:.6f} vs {A_amp * np.cos(phase):.6f}  "
        f"-> {'PASS' if ic_match else 'FAIL'}"
    )
    print(
        f"  integrated vs analytic x(t): max dev = {max_dev:.2e} (tol 1e-3)  "
        f"-> {'PASS' if max_dev < 1e-3 else 'FAIL'}"
    )
    print(
        f"  E = 0.5 m omega^2 A^2 = {E_exact:.4f}, max dE/E = {dE:.2e} (tol 1e-4)  "
        f"-> {'PASS' if cons else 'FAIL'}"
    )

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axA.plot(
        t,
        x_an,
        "-",
        color=OI["black"],
        lw=1.6,
        label=r"analytic $A\cos(\omega t+\phi)$",
    )
    axA.plot(
        t_int[::120],
        x_int[::120],
        "o",
        color=OI["vermilion"],
        ms=4,
        mfc="none",
        mew=1.1,
        label="Verlet",
    )
    axA.set_xlabel(r"$t$")
    axA.set_ylabel(r"$x(t)$")
    axA.legend(loc="upper right", fontsize=7)
    panel_label(axA, "(a)", loc="upper left")

    axB.plot(t_int, E, "-", color=OI["blue"], lw=1.4)
    axB.axhline(
        E_exact,
        color=OI["black"],
        ls="--",
        lw=1.0,
        label=rf"$\frac{{1}}{{2}}m\omega^2A^2={E_exact:.2f}$",
    )
    axB.set_xlabel(r"$t$")
    axB.set_ylabel(r"energy $E$")
    axB.set_ylim(E_exact * (1 - 1e-3), E_exact * (1 + 1e-3))
    axB.legend(loc="upper right", fontsize=7)
    axB.text(
        0.5,
        0.1,
        rf"max $\Delta E/E={dE:.0e}$",
        transform=axB.transAxes,
        ha="center",
        fontsize=7.5,
        color="0.4",
    )
    panel_label(axB, "(b)", loc="lower left")

    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "analytical_harmonic")
    print("  saved analytical_harmonic.{png,pdf}")
    return sol_ok and cons


# ============================================================================
# Figure 5 -- adversarial: canonical figure-eight vs a wrong (rotated) IC
# ============================================================================
def fig_figure_eight_adversarial(output_dir):
    print("\n" + "=" * 60)
    print("FIG 5: adversarial -- canonical figure-eight closes; a spun IC does not")
    print("=" * 60)
    ic = three_body_figure_eight(mass=1.0, scale=1.0, G=1.0)
    n = 200_000

    # canonical
    traj_c, pf_c, _ = _verlet_traj(
        ic.positions, ic.velocities, ic.masses, 1.0, ic.period, n
    )
    clos_c = float(jnp.max(jnp.linalg.norm(pf_c - ic.positions, axis=1)))
    L_c = float(jnp.linalg.norm(_ang_mom(ic.positions, ic.velocities, ic.masses)))

    # WRONG: a spurious net rotation -- add solid-body spin Omega*(z_hat x r) to the
    # velocities. This injects net angular momentum (L = Omega * I_zz != 0) and breaks
    # the choreography, so the orbit no longer closes. The classic IC mistake.
    omega_spin = 0.15
    zc = jnp.cross(jnp.array([0.0, 0.0, 1.0]), ic.positions)  # z_hat x r, per body
    vel_w = ic.velocities + omega_spin * zc
    traj_w, pf_w, _ = _verlet_traj(ic.positions, vel_w, ic.masses, 1.0, ic.period, n)
    clos_w = float(jnp.max(jnp.linalg.norm(pf_w - ic.positions, axis=1)))
    L_w = float(jnp.linalg.norm(_ang_mom(ic.positions, vel_w, ic.masses)))

    canon_ok = clos_c < 1e-6 and L_c < 1e-10
    wrong_fails = clos_w > 1e-2 or L_w > 1e-6
    print(
        f"  canonical:  |L|={L_c:.2e}, closure={clos_c:.2e}  -> "
        f"{'PASS (closes, L=0)' if canon_ok else 'FAIL'}"
    )
    print(
        f"  spurious-spin: |L|={L_w:.2e}, closure={clos_w:.2e}  -> "
        f"{'PASS (correctly fails)' if wrong_fails else 'FAIL (should not close)'}"
    )

    tc, tw = np.asarray(traj_c), np.asarray(traj_w)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True, sharey=True)
    for i, c in enumerate([OI["blue"], OI["vermilion"], OI["green"]]):
        axA.plot(tc[:, i, 0], tc[:, i, 1], "-", color=c, lw=0.8, alpha=0.8)
    axA.set_title(rf"canonical: $|L|={L_c:.0e}$, closes", fontsize=8.5)
    axA.set_xlabel(r"$x$")
    axA.set_ylabel(r"$y$")
    axA.set_aspect("equal")
    panel_label(axA, "(a)", loc="upper left")
    for i, c in enumerate([OI["blue"], OI["vermilion"], OI["green"]]):
        axB.plot(tw[:, i, 0], tw[:, i, 1], "-", color=c, lw=0.8, alpha=0.8)
    axB.set_title(rf"spurious spin: $|L|={L_w:.2f}\neq0$, drifts", fontsize=8.5)
    axB.set_xlabel(r"$x$")
    axB.set_aspect("equal")
    panel_label(axB, "(b)", loc="upper left")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "analytical_figure_eight_adversarial")
    print("  saved analytical_figure_eight_adversarial.{png,pdf}")
    return canon_ok and wrong_fails


def main():
    print("\n" + "=" * 70)
    print("PROGENAX ANALYTICAL TEST-CASE VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "Fig 1  two-body Kepler": fig_two_body(OUTPUT_DIR),
        "Fig 2  figure-eight choreography": fig_figure_eight(OUTPUT_DIR),
        "Fig 3  Kepler III (solar system)": fig_kepler_third(OUTPUT_DIR),
        "Fig 4  harmonic oscillator": fig_harmonic(OUTPUT_DIR),
        "Fig 5  figure-eight adversarial": fig_figure_eight_adversarial(OUTPUT_DIR),
    }
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print(
        "  ALL ANALYTICAL VALIDATION FIGURES PASS"
        if all_ok
        else "  SOME ANALYTICAL VALIDATION FIGURES FAILED"
    )
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/analytical_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
