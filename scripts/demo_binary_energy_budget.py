#!/usr/bin/env python
r"""B9 -- Binary energy budget: the primordial-binary energy reservoir (Batch C).

A methods + mini-science figure for ``binaries.diagnostics.binary_energy_budget``.
``build_binary_cluster`` virializes the *system COMs* to ``Q`` treating each binary
as a point mass (the McLuster scale-separation convention, Kuepper+2011 SA8); the
internal binary binding energy is a SEPARATE reservoir that ``Q`` never touches.
This demo makes that explicit and shows why the naive resolved virial ratio is
misleading.

Physics
-------
Two energy scales of a primordial-binary cluster:

  * ``W_com`` -- the cluster's bulk gravitational binding on the system COMs; the
    scale the cluster is virialized on (``Q_com = T_com/|W_com| ~ 0.5``).
  * ``E_internal = sum_b (-G m1 m2 / 2 a_b)`` -- the internal binary binding, set
    by periods + masses (vis-viva), INDEPENDENT of where the COM sits. The
    reservoir ``Q`` leaves alone.

``Q_resolved = T/|W|`` on the resolved stars MIXES the two scales and is NOT the
cluster's virial ratio (audit S10). A hard binary's INTERNAL virial is itself ~0.5
(time-averaged), so sampled at random orbital PHASES the resolved ratio scatters
around 0.5 (contaminated), rather than deflating monotonically with hardness. The
robust, gated statement is the energy SEPARATION: ``Q_com`` cleanly recovers 0.5
and ``|E_internal|`` dwarfs ``|W_com|``.

Young clusters (EFF), not GCs, are the natural home of a *primordial* binary
population -- they are the birth state, before dynamical processing ionizes the
soft binaries and hardens the rest. So the primary cluster here is an
Elson-Fall-Freeman (1987) young-cluster profile; a concentrated King (W0=7,
GC-like) appears only in the environment figure, as the SAME primordial population
laid into a more tightly bound birth potential (NOT a claim that GCs are born as
King models full of primordial binaries -- real GCs are old and processed; that is
the deferred N-body evolution arc).

Figures
-------
1. ``demo_binary_energy_budget`` -- EFF young cluster, two controlled sweeps:
   (a,b) binary HARDNESS (LogUniformPeriod band centre, hard->soft) and
   (c,d) binary FRACTION f_b. As binaries harden / f_b rises, ``|E_internal|``
   grows (panels b,d) while ``Q_com`` stays pinned at 0.5; ``Q_resolved`` scatters
   around 0.5 (panels a,c) -- the visible contamination is the point.
2. ``demo_binary_energy_budget_environment`` -- the SAME realistic Moe & Di Stefano
   (2017) population (same key -> identical E_internal) in a young EFF vs a
   concentrated King potential. The global reservoir FRACTION |E_internal|/|W_com|
   is larger in the puffy young cluster (smaller |W_com|) than in the dense GC --
   the binary energy store is relatively more important at birth in a young
   cluster. (A per-binary hard/soft statement -- |E_bind| vs local kT -- is
   related but distinct and not claimed here.)

No inference. The gates ARE the contract (exit 0 = all pass):
  * Q_com recovers the build target 0.5 at every sweep point (cluster virial intact
    on the COMs);
  * E_internal < 0 everywhere (bound binaries);
  * realized f_b = n_binaries / N_systems matches ConstantBinaryFraction within
    3 sigma Poisson on the IndependentCompanions sweeps;
  * |E_internal| > |W_com| at every sampled point (the reservoir dwarfs the cluster
    potential) -- Q_resolved itself is reported as a contaminated diagnostic, not gated;
  * environment: E_internal identical across EFF/King at the same key (controlled),
    and the reservoir fraction is larger for EFF than King.

Run record (2026-06-12, CPU/float64, N_SYSTEMS=2000, seeds from PRNGKey(0),
wall ~32 s, exit 0 / ALL PASS):
  HARDNESS (EFF, f_b=0.5): Q_com = 0.5000 at every logP in {1.5..5.5}; |E_internal|
    spans 1.22e6 (hard) -> 7.1e3 (soft); |W_com| ~ 6e2-1e3; Q_resolved scatters
    {0.464, 0.420, 0.512, 0.491, 0.327}.
  FRACTION (EFF, broad band): Q_com = 0.5000 for f_b in {0.1..1.0}; reservoir ratio
    {183, 292, 197, 351, 411}; n_bin tracks f_b*2000.
  ENVIRONMENT (Moe, same key): E_internal IDENTICAL 2.965e5 (EFF & King); |W_com|
    674 (EFF, r_h=5.32 pc) vs 995 (King W0=7, r_h=3.26 pc) -> reservoir 440 vs 298.
  Diagnostics: max|Q_resolved-Q_com| = 0.257; reservoir ratio in [11.6, 1918.6].

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_binary_energy_budget.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR
from progenax import (
    EFFProfile,
    EFFVelocityDF,
    KingProfile,
    KingVelocityDF,
    ThermalEccentricity,
)
from progenax.binaries import (
    IndependentCompanions,
    MoeCompanions,
    binary_energy_budget,
)
from progenax.binaries.period import LogUniformPeriod
from progenax.builders import Systems, build_binary_cluster
from progenax.imf import Maschberger
from progenax.imf.binary import ConstantBinaryFraction, FlatMassRatio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = STELLAR.G  # pc^3 Msun^-1 Myr^-2 -> lengths pc, masses Msun, velocities pc/Myr

# --- cluster + population configuration ------------------------------------- #
N_SYSTEMS = 2000          # system COMs (resolved stars up to ~2N at f_b=1)
SEED = 0

# EFF young-cluster profile (extended power-law halo; gamma ~ 2.5 is YMC-typical).
EFF_A, EFF_GAMMA, EFF_RT = 1.0, 2.5, 15.0
# King GC-like comparison (concentrated; r_t derived self-consistently from W0).
KING_W0, KING_RC = 7.0, 1.0

# Maschberger (smooth, differentiable) primary IMF over the stellar range.
PRIMARY_IMF = Maschberger(alpha=2.3, m_min=0.08, m_max=100.0)
Q_DIST = FlatMassRatio(q_min=0.3)
ECC_DIST = ThermalEccentricity()

# Hardness sweep (EFF, f_b=0.5): log10(P/day) band centres, hard -> soft.
LOGP_CENTERS = [1.5, 2.5, 3.5, 4.5, 5.5]
LOGP_BAND = 0.25          # half-width of the log-uniform period band [dex]
HARDNESS_FB = 0.5

# Binary-fraction sweep (EFF, fixed broad Opik period band).
FB_VALUES = [0.1, 0.3, 0.5, 0.7, 1.0]
FB_SWEEP_PERIOD = LogUniformPeriod(log_P_min=1.5, log_P_max=5.5)

# Gate tolerances.
Q_COM_TOL = 1.0e-2        # |Q_com - 0.5|
NSIG_POISSON = 3.0


def _eff():
    return EFFProfile(a=EFF_A, gamma=EFF_GAMMA, r_t=EFF_RT), \
        EFFVelocityDF(a=EFF_A, gamma=EFF_GAMMA, r_t=EFF_RT)


def _king():
    prof = KingProfile.from_W0_rc(W0=KING_W0, r_c=KING_RC)
    df = KingVelocityDF(W0=KING_W0, r_c=KING_RC)  # solves its own ODE; r_t derived
    return prof, df


def _indep(fbin, period_dist):
    return IndependentCompanions(
        binary_fraction=ConstantBinaryFraction(fbin),
        q_distribution=Q_DIST,
        period_distribution=period_dist,
        eccentricity_distribution=ECC_DIST,
    )


def _build(profile, df, companion_model, key):
    return build_binary_cluster(
        profile=profile,
        velocity_df=df,
        primary_imf=PRIMARY_IMF,
        companion_model=companion_model,
        target=Systems(N_SYSTEMS),
        key=key,
        units=STELLAR,
        Q=0.5,
    )


def _budget(ic):
    return binary_energy_budget(
        ic.positions, ic.velocities, ic.masses, ic.primordial_system_id, G=G
    )


def _half_mass_radius(ic):
    """Radius enclosing half the total mass (the cluster's physical size)."""
    r = np.asarray(jnp.linalg.norm(ic.positions, axis=1))
    m = np.asarray(ic.masses)
    order = np.argsort(r)
    cum = np.cumsum(m[order])
    half = 0.5 * cum[-1]
    return float(r[order][np.searchsorted(cum, half)])


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def hardness_sweep():
    """EFF, f_b=0.5: vary the period-band centre (hard -> soft)."""
    prof, df = _eff()
    out = []
    for i, c in enumerate(LOGP_CENTERS):
        period = LogUniformPeriod(log_P_min=c - LOGP_BAND, log_P_max=c + LOGP_BAND)
        ic = _build(prof, df, _indep(HARDNESS_FB, period), jax.random.PRNGKey(SEED + i))
        out.append(_budget(ic))
    return out


def fb_sweep():
    """EFF, fixed broad period band: vary the binary fraction."""
    prof, df = _eff()
    out = []
    for i, fb in enumerate(FB_VALUES):
        ic = _build(prof, df, _indep(fb, FB_SWEEP_PERIOD),
                    jax.random.PRNGKey(SEED + 100 + i))
        out.append((fb, _budget(ic)))
    return out


def environment_point():
    """SAME Moe population (same key -> identical E_internal) in EFF vs King."""
    moe = MoeCompanions()
    key = jax.random.PRNGKey(SEED + 200)
    eff_prof, eff_df = _eff()
    king_prof, king_df = _king()
    ic_eff = _build(eff_prof, eff_df, moe, key)
    ic_king = _build(king_prof, king_df, moe, key)
    return (_budget(ic_eff), _half_mass_radius(ic_eff)), \
        (_budget(ic_king), _half_mass_radius(ic_king))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_sweep_figure(hardness, fbs, moe_eff):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    logp = np.array(LOGP_CENTERS)

    qcom = np.array([float(b.Q_com) for b in hardness])
    qres = np.array([float(b.Q_resolved) for b in hardness])
    eint = np.array([abs(float(b.E_internal)) for b in hardness])
    wcom = np.array([abs(float(b.W_com)) for b in hardness])

    # (a) hardness: Q_com (flat ~0.5) vs Q_resolved (deflating).
    ax = axes[0, 0]
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
    ax.plot(logp, qcom, "o-", color=OI["blue"], label=r"$Q_{\rm com}$")
    ax.plot(logp, qres, "s-", color=OI["vermilion"], label=r"$Q_{\rm resolved}$")
    ax.set_xlabel(r"median $\log_{10}(P/{\rm day})$  (hard $\rightarrow$ soft)")
    ax.set_ylabel(r"virial ratio")
    ax.legend()
    panel_label(ax, "(a)")

    # (b) hardness: reservoir |E_internal| vs cluster |W_com|.
    ax = axes[0, 1]
    ax.semilogy(logp, eint, "o-", color=OI["green"], label=r"$|E_{\rm internal}|$")
    ax.semilogy(logp, wcom, "^-", color=OI["purple"], label=r"$|W_{\rm com}|$")
    ax.set_xlabel(r"median $\log_{10}(P/{\rm day})$")
    ax.set_ylabel(r"energy  [$M_\odot\,{\rm pc^2\,Myr^{-2}}$]")
    ax.legend()
    panel_label(ax, "(b)")

    fb = np.array([f for f, _ in fbs])
    qcom_f = np.array([float(b.Q_com) for _, b in fbs])
    qres_f = np.array([float(b.Q_resolved) for _, b in fbs])
    ratio_f = np.array([abs(float(b.E_internal)) / abs(float(b.W_com)) for _, b in fbs])

    # (c) f_b: Q_com vs Q_resolved.
    ax = axes[1, 0]
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
    ax.plot(fb, qcom_f, "o-", color=OI["blue"], label=r"$Q_{\rm com}$")
    ax.plot(fb, qres_f, "s-", color=OI["vermilion"], label=r"$Q_{\rm resolved}$")
    ax.set_xlabel(r"binary fraction $f_b$")
    ax.set_ylabel(r"virial ratio")
    ax.legend()
    panel_label(ax, "(c)")

    # (d) f_b: reservoir fraction.
    ax = axes[1, 1]
    ax.semilogy(fb, ratio_f, "o-", color=OI["orange"], label=r"$|E_{\rm internal}|/|W_{\rm com}|$")
    # Moe reference point (its realized f_b).
    moe_fb = moe_eff.n_binaries / N_SYSTEMS
    moe_ratio = abs(float(moe_eff.E_internal)) / abs(float(moe_eff.W_com))
    ax.scatter([moe_fb], [moe_ratio], marker="*", s=90, color=OI["black"], zorder=5,
               label=r"Moe \& Di Stefano (2017)")
    ax.set_xlabel(r"binary fraction $f_b$")
    ax.set_ylabel(r"reservoir fraction")
    ax.legend()
    panel_label(ax, "(d)")

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "demo_binary_energy_budget")


def make_environment_figure(eff, king):
    import matplotlib.pyplot as plt

    (b_eff, rh_eff), (b_king, rh_king) = eff, king
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    x = np.arange(2)
    labels = [f"EFF (young)\n$r_h={rh_eff:.2f}$ pc",
              f"King $W_0=7$ (GC)\n$r_h={rh_king:.2f}$ pc"]

    # (a) the two energy scales; E_internal is identical (same key), W_com differs.
    ax = axes[0]
    eint = [abs(float(b_eff.E_internal)), abs(float(b_king.E_internal))]
    wcom = [abs(float(b_eff.W_com)), abs(float(b_king.W_com))]
    ax.bar(x - 0.18, eint, 0.36, color=OI["green"], label=r"$|E_{\rm internal}|$")
    ax.bar(x + 0.18, wcom, 0.36, color=OI["purple"], label=r"$|W_{\rm com}|$")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"energy  [$M_\odot\,{\rm pc^2\,Myr^{-2}}$]")
    ax.legend()
    panel_label(ax, "(a)")

    # (b) reservoir fraction: larger in the puffy young cluster.
    ax = axes[1]
    ratio = [eint[0] / wcom[0], eint[1] / wcom[1]]
    ax.bar(x, ratio, 0.5, color=[OI["blue"], OI["vermilion"]])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$|E_{\rm internal}|/|W_{\rm com}|$")
    panel_label(ax, "(b)")

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "demo_binary_energy_budget_environment")
    return ratio


# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("BINARY ENERGY BUDGET (B9): the primordial-binary energy reservoir")
    print("(units: STELLAR -- lengths pc, masses Msun, velocities pc/Myr)")
    print("=" * 78)

    print(f"\n  EFF young cluster: a={EFF_A}, gamma={EFF_GAMMA}, r_t={EFF_RT} pc; "
          f"N_systems={N_SYSTEMS}")
    print(f"  Maschberger IMF alpha=2.3 [{PRIMARY_IMF.m_min}, {PRIMARY_IMF.m_max}] Msun\n")

    hardness = hardness_sweep()
    print("  HARDNESS sweep (EFF, f_b=0.5):")
    print(f"  {'logP':>6s} {'Q_com':>9s} {'Q_res':>9s} {'|E_int|':>11s} "
          f"{'|W_com|':>11s} {'n_bin':>6s}")
    for c, b in zip(LOGP_CENTERS, hardness):
        print(f"  {c:>6.2f} {float(b.Q_com):>9.4f} {float(b.Q_resolved):>9.4f} "
              f"{abs(float(b.E_internal)):>11.3e} {abs(float(b.W_com)):>11.3e} "
              f"{b.n_binaries:>6d}")

    fbs = fb_sweep()
    print("\n  BINARY-FRACTION sweep (EFF, broad period band):")
    print(f"  {'f_b':>6s} {'Q_com':>9s} {'Q_res':>9s} {'ratio':>11s} {'n_bin':>6s}")
    for fb, b in fbs:
        ratio = abs(float(b.E_internal)) / abs(float(b.W_com))
        print(f"  {fb:>6.2f} {float(b.Q_com):>9.4f} {float(b.Q_resolved):>9.4f} "
              f"{ratio:>11.3e} {b.n_binaries:>6d}")

    (b_eff, rh_eff), (b_king, rh_king) = environment_point()
    print("\n  ENVIRONMENT (same Moe population, same key):")
    print(f"    EFF  : r_h={rh_eff:.3f} pc  |E_int|={abs(float(b_eff.E_internal)):.3e} "
          f" |W_com|={abs(float(b_eff.W_com)):.3e}  n_bin={b_eff.n_binaries}")
    print(f"    King : r_h={rh_king:.3f} pc  |E_int|={abs(float(b_king.E_internal)):.3e} "
          f" |W_com|={abs(float(b_king.W_com)):.3e}  n_bin={b_king.n_binaries}")

    make_sweep_figure(hardness, fbs, b_eff)
    env_ratio = make_environment_figure((b_eff, rh_eff), (b_king, rh_king))

    # ---- gates ------------------------------------------------------------- #
    all_budgets = list(hardness) + [b for _, b in fbs] + [b_eff, b_king]
    q_com_ok = all(abs(float(b.Q_com) - 0.5) < Q_COM_TOL for b in all_budgets)
    bound_ok = all(float(b.E_internal) < 0.0 for b in all_budgets)

    # realized f_b vs ConstantBinaryFraction (3 sigma Poisson) on the f_b sweep.
    fb_ok = True
    for fb, b in fbs:
        exp = fb * N_SYSTEMS
        sig = np.sqrt(max(fb * (1.0 - fb) * N_SYSTEMS, 1.0))
        fb_ok &= abs(b.n_binaries - exp) < NSIG_POISSON * sig

    # The internal reservoir dwarfs the cluster potential at every sampled point
    # (|E_internal| > |W_com|) -- the robust headline. Q_resolved is NOT gated: a
    # hard binary's internal virial is itself ~0.5 (time-averaged), so at random
    # orbital PHASES the resolved ratio scatters around 0.5 rather than deflating
    # monotonically; it is reported as a contaminated diagnostic (!= Q_com).
    reservoir_ratios = [abs(float(b.E_internal)) / abs(float(b.W_com))
                        for b in all_budgets]
    reservoir_ok = all(rr > 1.0 for rr in reservoir_ratios)
    qres_contam = max(abs(float(b.Q_resolved) - float(b.Q_com)) for b in all_budgets)

    # environment: identical E_internal (same key) and larger reservoir in EFF.
    e_identical = abs(float(b_eff.E_internal) - float(b_king.E_internal)) <= \
        1e-6 * abs(float(b_eff.E_internal))
    env_ok = env_ratio[0] > env_ratio[1]

    print(f"\n  diagnostic: max |Q_resolved - Q_com| = {qres_contam:.3f}  "
          f"(contaminated; Q_resolved is NOT the cluster's virial ratio)")
    print(f"  diagnostic: reservoir ratio |E_int|/|W_com| in "
          f"[{min(reservoir_ratios):.1f}, {max(reservoir_ratios):.1f}]")

    rows = [
        ("Q_com ~ 0.5 (all points)", "PASS" if q_com_ok else "FAIL",
         f"< {Q_COM_TOL}", q_com_ok),
        ("E_internal < 0 (all bound)", "PASS" if bound_ok else "FAIL", "< 0", bound_ok),
        ("realized f_b ~ set f_b", "PASS" if fb_ok else "FAIL",
         f"{NSIG_POISSON:.0f}sigma Poisson", fb_ok),
        ("|E_internal| > |W_com| (all)", "PASS" if reservoir_ok else "FAIL",
         "reservoir", reservoir_ok),
        ("E_int(EFF)==E_int(King) @key", "PASS" if e_identical else "FAIL",
         "controlled", e_identical),
        ("reservoir EFF > King", "PASS" if env_ok else "FAIL",
         "environment", env_ok),
    ]

    print("\n" + "-" * 78)
    print(f"  {'CHECK':<34s} {'status':>6s} {'gate':>18s}")
    print("-" * 78)
    all_ok = True
    for name, status, gate, ok in rows:
        all_ok &= ok
        print(f"  {name:<34s} {status:>6s} {gate:>18s}")
    print("-" * 78)
    print(f"  saved {OUTPUT_DIR}/demo_binary_energy_budget{{,_environment}}.{{png,pdf}}")
    print("=" * 78)
    print("  BINARY ENERGY BUDGET DEMO: ALL PASS" if all_ok
          else "  BINARY ENERGY BUDGET DEMO: FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
