#!/usr/bin/env python
r"""Batch 8 -- cluster-builder convenience-API versatility demos.

A versatility showcase for the thin, differentiable ``build_cluster`` layer (the
5 named aliases, ``matched_velocity_df``, ``RotationSpec``, ``ClusterParams`` +
``build_cluster_from_params``). Where ``validate_cluster_builders.py`` proves the
physics is faithful with publication figures, THIS script demonstrates the API
SURFACE -- onboarding ergonomics, the one-call theta->ICResult inference forward
map, the generic profile engine, each modifier's physical signature, and the two
mass-spec paths (generative vs fixed-data) -- as a PRINT-ONLY gate.

Five demos (each PRINTS a physical readout + PASS/FAIL; main() exits 1 on any FAIL):

  1. Onboarding one-liner       build_plummer_cluster(n=1000, r_h=1.0, key) -> N, Q, r_h.
  2. theta->ICResult inference   build_cluster_from_params as the one-call forward map;
                                 the Gauss-Newton Fisher info / CRLB sigma(theta) for each
                                 differentiable knob (r_h, r_a, r_t, omega). This is the
                                 B-series physics-direct inference pattern collapsed to one
                                 call. r_t uses apply_tidal_truncation's STRAIGHT-THROUGH
                                 surrogate gradient (live, not FD-consistent) on a COUNT
                                 summary -- noted honestly (positions are r_t-invariant).
  3. All 5 profiles, one engine  loop build_cluster(profile, ...) over Plummer/EFF/King/
                                 Michie/LIMEPY -> measured Q + 10/50/90% Lagrangian radii.
  4. Each modifier's readout     anisotropy beta(r) vs r^2/(r^2+r_a^2); tidal surviving-mass
                                 fraction + massless ghosts; rotation net L_z > 0.
  5. Generative vs inference      n+imf (Kroupa generative draw -> mass-function summary)
                                 vs masses=<fixed array> (fixed-data path -> EXACT round-trip).

The Fisher is the Gauss-Newton form F = J^T J on a STANDARDIZED summary (residual
divided by a per-cell Poisson/sampling SE), reusing _demo_inference.fisher_information_gn
-- the same machinery the B-series demos use, here driven by the one-call
build_cluster_from_params forward map. For each knob the summary is chosen so the
gradient is LIVE (|F|>0):
  * r_h    -> sorted enclosed-radius profile (positions scale with r_h);
  * r_a    -> binned beta(r) (anisotropy lives in the velocities; OM-augmented DF);
  * omega  -> binned <v_phi>(R) (the solid-body rotation overlay);
  * r_t    -> binned surviving COUNTS per shell (mass/count summary; the straight-
              through surrogate makes d counts / d r_t live, NOT FD-consistent).

Gate (exit 0 = all pass): all five demos PASS.

Run record (2026-06-14, CPU/float64, key PRNGKey(0), exit 0 / ALL PASS):
  1. N=1000, Q=0.5000, r_h(empirical)=0.9754 pc.
  2. Fisher info / sigma per knob: r_h 30.32 / 0.182; r_a 13.29 / 0.274;
     omega 37.20 / 0.164 (all FD-consistent); r_t 1.85e9 / 2.3e-5 (straight-through
     surrogate on the surviving-mass-per-shell summary -- live, not FD-consistent).
  3. Q=0.5000 for all 5 families; Lagrangian radii monotone (e.g. Plummer
     0.398/0.993/2.881 pc; King 1.06/4.01/12.26 pc).
  4. anisotropy beta(r=1.18)=0.670 vs analytic 0.686 (|dev|=0.016); tidal surviving-
     mass fraction 0.707 with ghost mass beyond r_t = 0; rotation net L_z = +1.61e4.
  5. generative Kroupa draw: N=2000, mean mass 0.351 Msun (std 1.335>0), p10/50/90
     0.026/0.121/0.694 Msun; fixed-data masses= round-trip EXACT.

Usage:
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_cluster_builders.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

# float64 is enabled by `import progenax` below (jaxstro.jaxconfig.enable_high_precision);
# STELLAR.G is a plain Python float and no jnp array is created before that import, so no
# explicit jax.config.update is needed here (matches the sibling B-series demos).
from jaxstro.units import STELLAR
from progenax import (
    build_plummer_cluster,
    build_cluster,
    build_cluster_from_params,
    ClusterParams,
    PlummerProfile,
    EFFProfile,
    KingProfile,
    MichieProfile,
    LIMEPYProfile,
    PowerLawIMF,
    compute_kinetic_energy,
    compute_potential_energy,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _demo_inference import fisher_information_gn

G = STELLAR.G                       # pc^3 Msun^-1 Myr^-2
SEED = 0


# --------------------------------------------------------------------------- #
# Shared physical readouts
# --------------------------------------------------------------------------- #
def _virial_Q(ic):
    """Q = T/|V| over the mass-bearing particles (zero-mass ghosts drop out of the
    mass-weighted energies automatically)."""
    T = compute_kinetic_energy(ic.velocities, ic.masses)
    V = compute_potential_energy(ic.positions, ic.masses, G=G)
    return float(T / jnp.abs(V))


def _half_mass_radius(ic):
    """Empirical half-mass radius: the radius enclosing half the total mass."""
    r = np.asarray(jnp.linalg.norm(ic.positions, axis=1))
    m = np.asarray(ic.masses)
    order = np.argsort(r)
    cum = np.cumsum(m[order])
    half = 0.5 * cum[-1]
    return float(r[order][np.searchsorted(cum, half)])


def _lagrangian_radii(ic, fracs=(0.1, 0.5, 0.9)):
    """Lagrangian radii: the radii enclosing the given mass fractions."""
    r = np.asarray(jnp.linalg.norm(ic.positions, axis=1))
    m = np.asarray(ic.masses)
    order = np.argsort(r)
    cum = np.cumsum(m[order]) / np.sum(m)
    return [float(r[order][np.searchsorted(cum, f)]) for f in fracs]


# --------------------------------------------------------------------------- #
# Demo 1 -- onboarding one-liner
# --------------------------------------------------------------------------- #
def demo_onboarding():
    print("\n" + "=" * 74)
    print("DEMO 1 -- onboarding one-liner: build_plummer_cluster(n, r_h, key)")
    print("=" * 74)
    print("  >>> ic = build_plummer_cluster(n=1000, r_h=1.0, key=jax.random.PRNGKey(0))")

    n, r_h = 1000, 1.0
    ic = build_plummer_cluster(n=n, r_h=r_h, key=jax.random.PRNGKey(SEED))
    N = int(ic.masses.shape[0])
    Q = _virial_Q(ic)
    r_h_meas = _half_mass_radius(ic)

    n_ok = N == n
    q_ok = abs(Q - 0.5) < 0.03
    rh_ok = abs(r_h_meas - r_h) < 0.15        # finite-N sampling scatter of r_h(empirical)
    passed = n_ok and q_ok and rh_ok

    print(f"\n  N (particles)        = {N:5d}        (expected {n}, {'PASS' if n_ok else 'FAIL'})")
    print(f"  Q = T/|V|            = {Q:7.4f}      (expected 0.500, {'PASS' if q_ok else 'FAIL'})")
    print(f"  r_h (empirical)      = {r_h_meas:7.4f} pc   (expected {r_h:.3f}, "
          f"{'PASS' if rh_ok else 'FAIL'})")
    print(f"\n  one-liner -> equilibrium Plummer IC  ->  {'PASS' if passed else 'FAIL'}")
    return passed


# --------------------------------------------------------------------------- #
# Demo 2 -- differentiable theta->ICResult Fisher inference (the headline)
# --------------------------------------------------------------------------- #
def _enclosed_radius_summary(ic, n_q=24):
    """Sorted enclosed-radius profile: the n_q quantile radii of the (mass-weighted)
    radial distribution. r_h scales positions, so this vector is r_h-sensitive."""
    r = jnp.sort(jnp.linalg.norm(ic.positions, axis=1))
    qs = jnp.linspace(0.05, 0.95, n_q)
    idx = jnp.clip((qs * r.shape[0]).astype(int), 0, r.shape[0] - 1)
    return r[idx]


def _beta_summary(ic, edges):
    """Binned Binney anisotropy beta(r) = 1 - sigma_t^2/(2 sigma_r^2). r_a enters the
    velocities, so this vector is anisotropy_radius-sensitive."""
    pos, vel = ic.positions, ic.velocities
    r = jnp.linalg.norm(pos, axis=1)
    rhat = pos / (r[:, None] + 1e-12)
    vr = jnp.sum(vel * rhat, axis=1)
    vt2 = jnp.sum(vel * vel, axis=1) - vr * vr
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (r >= lo) & (r < hi)
        w = msk.astype(jnp.float64)
        n = jnp.sum(w)
        s_r2 = jnp.sum(w * vr * vr) / jnp.maximum(n, 1.0)
        s_t2 = jnp.sum(w * vt2) / jnp.maximum(n, 1.0)
        out.append(1.0 - s_t2 / (2.0 * jnp.maximum(s_r2, 1e-12)))
    return jnp.stack(out)


def _vphi_summary(ic, edges):
    """Binned mean azimuthal velocity <v_phi>(R). omega enters the velocities via the
    solid-body overlay, so this vector is omega-sensitive."""
    pos, vel = ic.positions, ic.velocities
    R = jnp.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    vphi = (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / jnp.maximum(R, 1e-10)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (R >= lo) & (R < hi)
        w = msk.astype(jnp.float64)
        out.append(jnp.sum(w * vphi) / jnp.maximum(jnp.sum(w), 1.0))
    return jnp.stack(out)


def _surviving_counts_summary(ic, edges):
    """Soft surviving-mass per radial shell (the count/mass channel). The tidal cut
    zeroes masses beyond r_t; the straight-through surrogate makes d(shell mass)/d r_t
    LIVE (positions are r_t-invariant, so a position summary would be dead)."""
    r = jnp.linalg.norm(ic.positions, axis=1)
    m = ic.masses
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        w = ((r >= lo) & (r < hi)).astype(jnp.float64)
        out.append(jnp.sum(w * m))
    return jnp.stack(out)


def _fisher_for_knob(summary_at, theta0, se_floor=1e-3):
    """Gauss-Newton Fisher info for a scalar knob from a standardized summary.

    ``summary_at(theta)`` -> (K,) summary vector. The standardized residual is the
    DEVIATION from the truth-point summary divided by a per-cell SE (a constant
    Poisson-like sqrt(mu) floor here -- the demo's point is the Fisher CONTENT/sign,
    not a calibrated CRLB), so r(theta0)=0 and F = J^T J = (d summary/d theta / se)^2
    summed. F = 1/sigma^2 (a scalar Fisher), sigma = 1/sqrt(F)."""
    s0 = np.asarray(summary_at(theta0))
    se = np.maximum(np.sqrt(np.abs(s0)), se_floor)         # per-cell scale (Poisson-like)
    se = jnp.asarray(se)
    s0j = jnp.asarray(s0)

    def residual(theta):
        # theta0 is the truth-point, so r(theta0)=0 by construction -> F = J^T J is the
        # exact expected (Gauss-Newton) information here, not a self-referential tautology.
        return (summary_at(theta[0]) - s0j) / se

    F = fisher_information_gn(residual, jnp.array([theta0]))
    info = float(F[0, 0])
    sigma = float(1.0 / np.sqrt(info)) if info > 0 else float("inf")
    return info, sigma


def demo_inference_fisher():
    print("\n" + "=" * 74)
    print("DEMO 2 -- differentiable theta->ICResult Fisher inference (the headline)")
    print("=" * 74)
    print("  build_cluster_from_params is the ONE-CALL forward map theta -> ICResult;")
    print("  fisher_information_gn (Gauss-Newton J^T J) gives Fisher info / CRLB sigma(theta)")
    print("  for each differentiable knob. r_t uses the apply_tidal_truncation straight-")
    print("  through surrogate gradient on a COUNT summary (live, NOT FD-consistent).")

    key = jax.random.PRNGKey(SEED)
    m = jnp.ones(400)
    beta_edges = jnp.array([0.25, 0.45, 0.7, 1.0, 1.4, 2.0, 3.0])
    vphi_edges = jnp.linspace(0.2, 2.6, 9)
    tidal_edges = jnp.linspace(0.3, 3.0, 12)

    # r_h: profile scale -> positions. Summary = sorted enclosed-radius profile.
    def fwd_rh(r_h):
        ic = build_cluster_from_params(
            ClusterParams(profile=PlummerProfile(r_h=r_h)), masses=m, key=key)
        return _enclosed_radius_summary(ic)

    # r_a (anisotropy): OM DF -> velocities. Summary = binned beta(r). Q=None keeps the
    # pure OM equilibrium (a virial rescale leaves beta unchanged, but Q=None is cleaner).
    def fwd_ra(r_a):
        ic = build_cluster_from_params(
            ClusterParams(profile=PlummerProfile(r_h=1.0), anisotropy_radius=r_a, Q=None),
            masses=m, key=key)
        return _beta_summary(ic, beta_edges)

    # omega (rotation): solid-body overlay -> velocities. Summary = binned <v_phi>(R).
    def fwd_omega(omega):
        ic = build_cluster_from_params(
            ClusterParams(profile=PlummerProfile(r_h=1.0), rotation=omega), masses=m, key=key)
        return _vphi_summary(ic, vphi_edges)

    # r_t (tidal): straight-through surrogate -> masses (positions r_t-invariant!).
    # Summary = binned surviving mass per shell (the count/mass channel).
    def fwd_rt(r_t):
        ic = build_cluster_from_params(
            ClusterParams(profile=PlummerProfile(r_h=1.0), tidal_radius=r_t), masses=m, key=key)
        return _surviving_counts_summary(ic, tidal_edges)

    knobs = [
        ("r_h    (profile scale)",   fwd_rh,    1.0, "FD-consistent"),
        ("r_a    (OM anisotropy)",   fwd_ra,    0.7, "FD-consistent"),
        ("omega  (solid rotation)",  fwd_omega, 0.3, "FD-consistent"),
        ("r_t    (tidal cut)",       fwd_rt,    1.5, "straight-through"),
    ]

    print(f"\n  {'knob':<26}{'theta0':>8}{'Fisher info':>14}{'sigma(theta)':>14}"
          f"{'grad kind':>17}{'':>6}")
    print("  " + "-" * 84)
    all_ok = True
    for label, fwd, theta0, gradkind in knobs:
        info, sigma = _fisher_for_knob(fwd, theta0)
        ok = np.isfinite(info) and info > 0 and np.isfinite(sigma)
        all_ok = all_ok and ok
        print(f"  {label:<26}{theta0:>8.2f}{info:>14.4e}{sigma:>14.4e}"
              f"{gradkind:>17}{('PASS' if ok else 'FAIL'):>6}")
    print("  " + "-" * 84)
    print("  NOTE: r_t's Fisher uses the apply_tidal_truncation STRAIGHT-THROUGH surrogate")
    print("        gradient on a surviving-mass-per-shell summary (positions are r_t-")
    print("        invariant, so a position summary would be dead). The surrogate is LIVE")
    print("        but DELIBERATELY not FD-consistent -- it is a teeth-tested channel, not")
    print("        a calibrated CRLB. r_h/r_a/omega are FD-consistent (grad-audit registry).")
    print(f"\n  every knob's Fisher info finite + positive  ->  {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# --------------------------------------------------------------------------- #
# Demo 3 -- all 5 profiles via the generic engine
# --------------------------------------------------------------------------- #
def demo_all_profiles():
    print("\n" + "=" * 74)
    print("DEMO 3 -- all 5 profiles via the generic build_cluster engine")
    print("=" * 74)
    print("  loop build_cluster(profile, masses, key) over the 5 families -> Q + Lagrangian radii")

    n = 4000
    m = jnp.ones(n)
    profiles = [
        ("Plummer", PlummerProfile(r_h=1.0)),
        ("EFF",     EFFProfile(a=1.0, gamma=5.0, r_t=15.0)),   # gamma=5/mild trunc -> near-virial
        ("King",    KingProfile.from_W0_rc(W0=7.0, r_c=1.0)),
        ("Michie",  MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0)),
        ("LIMEPY",  LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0)),
    ]

    print(f"\n  {'profile':<9}{'Q=T/|V|':>10}{'r_10%':>9}{'r_50%':>9}{'r_90%':>9}"
          f"{'monotone':>10}{'':>7}")
    print("  " + "-" * 64)
    all_ok = True
    key = jax.random.PRNGKey(SEED)
    for name, prof in profiles:
        key, sub = jax.random.split(key)
        ic = build_cluster(prof, masses=m, key=sub)
        Q = _virial_Q(ic)
        r10, r50, r90 = _lagrangian_radii(ic)
        q_ok = abs(Q - 0.5) < 0.04
        mono_ok = (r10 < r50 < r90) and (r10 > 0.0)
        ok = q_ok and mono_ok
        all_ok = all_ok and ok
        print(f"  {name:<9}{Q:>10.4f}{r10:>9.4f}{r50:>9.4f}{r90:>9.4f}"
              f"{str(mono_ok):>10}{('PASS' if ok else 'FAIL'):>7}")
    print("  " + "-" * 64)
    print(f"\n  all 5 families near-virial (|Q-0.5|<0.04) + monotone radii  ->  "
          f"{'PASS' if all_ok else 'FAIL'}")
    return all_ok


# --------------------------------------------------------------------------- #
# Demo 4 -- each modifier with a physical readout
# --------------------------------------------------------------------------- #
def demo_modifiers():
    print("\n" + "=" * 74)
    print("DEMO 4 -- each modifier with a physical readout")
    print("=" * 74)

    key = jax.random.PRNGKey(SEED)

    # --- anisotropy: measured beta at a mid radius vs analytic r^2/(r^2+r_a^2) ---
    r_a = 0.8
    ic_a = build_plummer_cluster(masses=jnp.ones(40_000), r_h=1.0, key=key,
                                 anisotropy_radius=r_a, Q=None)
    pos, vel = np.asarray(ic_a.positions), np.asarray(ic_a.velocities)
    r = np.linalg.norm(pos, axis=1)
    rhat = pos / (r[:, None] + 1e-12)
    vr = np.sum(vel * rhat, axis=1)
    vt2 = np.sum(vel ** 2, axis=1) - vr ** 2
    lo, hi = 1.0, 1.4                                   # mid radius shell
    msk = (r >= lo) & (r < hi)
    rm = float(np.mean(r[msk]))
    beta_meas = 1.0 - float(np.mean(vt2[msk])) / (2.0 * float(np.mean(vr[msk] ** 2)))
    beta_an = rm ** 2 / (rm ** 2 + r_a ** 2)
    aniso_dev = abs(beta_meas - beta_an)
    aniso_ok = aniso_dev < 0.05
    print(f"\n  anisotropy (r_a={r_a}):")
    print(f"    beta_meas(r={rm:.3f}) = {beta_meas:.4f}   analytic r^2/(r^2+r_a^2) = "
          f"{beta_an:.4f}   |dev| = {aniso_dev:.4f}  ({'PASS' if aniso_ok else 'FAIL'})")

    # --- tidal: surviving-mass fraction + massless ghosts beyond r_t ---
    r_t = 1.5
    n_t = 20_000
    ic_t = build_plummer_cluster(masses=jnp.ones(n_t), r_h=1.0, key=key, tidal_radius=r_t)
    rt_r = np.asarray(jnp.linalg.norm(ic_t.positions, axis=1))
    mt = np.asarray(ic_t.masses)
    inside = rt_r <= r_t
    surv_frac = float(np.sum(mt) / n_t)                # total surviving mass / N
    ghost_mass = float(np.sum(mt[~inside]))            # mass beyond r_t (must be exactly 0)
    ghosts_massless = ghost_mass == 0.0
    surv_ok = (0.0 < surv_frac < 1.0) and ghosts_massless
    print(f"\n  tidal (r_t={r_t}):")
    print(f"    surviving-mass fraction = {surv_frac:.4f}   ghost mass beyond r_t = "
          f"{ghost_mass:.3e}   (massless: {ghosts_massless}, {'PASS' if surv_ok else 'FAIL'})")

    # --- rotation: net L_z > 0 ---
    omega = 0.3
    ic_r = build_plummer_cluster(masses=jnp.ones(5_000), r_h=1.0, key=key, rotation=omega)
    x, y = np.asarray(ic_r.positions[:, 0]), np.asarray(ic_r.positions[:, 1])
    vx, vy = np.asarray(ic_r.velocities[:, 0]), np.asarray(ic_r.velocities[:, 1])
    Lz = float(np.sum(np.asarray(ic_r.masses) * (x * vy - y * vx)))
    rot_ok = Lz > 0.0
    print(f"\n  rotation (omega={omega}):")
    print(f"    net L_z = {Lz:.4f} Msun pc^2 Myr^-1   (> 0: {rot_ok}, "
          f"{'PASS' if rot_ok else 'FAIL'})")

    passed = aniso_ok and surv_ok and rot_ok
    print(f"\n  anisotropy beta(r) + tidal cut + rotation L_z all physical  ->  "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


# --------------------------------------------------------------------------- #
# Demo 5 -- generative vs inference (fixed-data) mass-spec paths
# --------------------------------------------------------------------------- #
def demo_generative_vs_inference():
    print("\n" + "=" * 74)
    print("DEMO 5 -- generative (n+imf) vs inference (masses=) mass-spec paths")
    print("=" * 74)

    key = jax.random.PRNGKey(SEED)

    # --- generative: n + Kroupa IMF -> sampled mass function ---
    n = 2000
    imf = PowerLawIMF.kroupa()
    ic_gen = build_plummer_cluster(n=n, r_h=1.0, imf=imf, key=key)
    mg = np.asarray(ic_gen.masses)
    p10, p50, p90 = np.percentile(mg, [10, 50, 90])
    gen_ok = (mg.shape[0] == n) and (mg.std() > 0.0) and bool(np.all(mg > 0.0))
    print(f"\n  generative: build_plummer_cluster(n={n}, r_h=1.0, imf=PowerLawIMF.kroupa())")
    print(f"    N sampled        = {mg.shape[0]}    (expected {n})")
    print(f"    mean mass        = {mg.mean():.4f} Msun   (std {mg.std():.4f} > 0 -> IMF sampled)")
    print(f"    min / max mass   = {mg.min():.4f} / {mg.max():.4f} Msun")
    print(f"    p10 / p50 / p90  = {p10:.4f} / {p50:.4f} / {p90:.4f} Msun")
    print(f"    sampled mass function -> {'PASS' if gen_ok else 'FAIL'}")

    # --- inference / fixed-data: masses=<fixed array> -> EXACT round-trip ---
    m_fixed = jnp.linspace(0.5, 3.0, 256)
    ic_fix = build_plummer_cluster(masses=m_fixed, r_h=1.0, key=key)
    roundtrip = bool(jnp.all(ic_fix.masses == m_fixed))
    fix_ok = roundtrip and (ic_fix.masses.shape == m_fixed.shape)
    print(f"\n  inference: build_plummer_cluster(masses=<fixed 256-array>, r_h=1.0)")
    print(f"    masses round-trip EXACTLY (jnp.all(ic.masses == input)) = {roundtrip}")
    print(f"    shape {tuple(ic_fix.masses.shape)} == input {tuple(m_fixed.shape)}   "
          f"-> {'PASS' if fix_ok else 'FAIL'}")

    passed = gen_ok and fix_ok
    print(f"\n  generative draw + fixed-data exact round-trip  ->  "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


# --------------------------------------------------------------------------- #
# main -- the gate
# --------------------------------------------------------------------------- #
def main():
    print("\n" + "=" * 74)
    print("PROGENAX CLUSTER-BUILDER CONVENIENCE-API -- VERSATILITY DEMOS (Batch 8)")
    print("=" * 74)

    results = {
        "1. onboarding one-liner":            demo_onboarding(),
        "2. theta->ICResult Fisher inference": demo_inference_fisher(),
        "3. all 5 profiles, one engine":      demo_all_profiles(),
        "4. each modifier readout":           demo_modifiers(),
        "5. generative vs inference paths":   demo_generative_vs_inference(),
    }

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 74)
    print("  ALL CLUSTER-BUILDER DEMOS PASS" if all_ok
          else "  SOME CLUSTER-BUILDER DEMOS FAILED")
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
