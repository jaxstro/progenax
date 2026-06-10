#!/usr/bin/env python
"""Staged peak-RSS memory profile for progenax cluster construction.

Permanent memory-evidence artifact (Batch A, Task 4): each stage runs in its
OWN subprocess so the OS peak-RSS counter (`resource.ru_maxrss`) attributes
cleanly to that stage's workload -- no JAX-buffer or allocator carry-over
between stages. Every stage has a quantitative PASS gate on peak RSS.

Platform note: `ru_maxrss` is reported in BYTES on Darwin (macOS) but in
KILOBYTES on Linux; `_peak_rss_gb` guards on `sys.platform`.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/profile_cluster_memory.py
    # Known-failing stage allowed (until Task 5 lands the speed-CDF tables):
    env -u VIRTUAL_ENV uv run --no-sync python scripts/profile_cluster_memory.py \
        --allow-fail limepy_df_aniso
    # Single stage (also the subprocess re-entry mechanism):
    env -u VIRTUAL_ENV uv run --no-sync python scripts/profile_cluster_memory.py \
        --stage virial_pe

Exits nonzero iff any stage whose gate is not marked --allow-fail exceeds it.
--allow-fail covers GATE EXCESS only (a real measurement over its gate); a
stage that crashes (nonzero subprocess exit or no result line) always FAILs.
"""
import argparse
import resource
import subprocess
import sys

# stage name -> (N, gate_gb)
STAGES = {
    "import": (None, 1.0),
    "engineA_iso": (100_000, 3.0),
    "engineA_aniso": (100_000, 4.0),
    "engineB_halo_core": (100_000, 4.0),
    "virial_pe": (20_000, 2.0),
    "group_virial": (20_000, 2.0),
    "limepy_df_aniso": (20_000, 3.0),
}
_RESULT_PREFIX = "PEAK_RSS_GB"


def _peak_rss_gb() -> float:
    """Process-lifetime peak RSS in GB (ru_maxrss: bytes on Darwin, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    return rss / 1e9


def _plummer_ic(n: int):
    """Shared N-body fixture: equal-mass Plummer IC with G from STELLAR."""
    import jax
    import jax.numpy as jnp
    from jaxstro.units import STELLAR
    from progenax import PlummerProfile, PlummerVelocityDF, build_spatial_ic

    masses = jnp.ones(n)
    ic = build_spatial_ic(PlummerProfile(r_h=1.0), masses,
                          PlummerVelocityDF(r_h=1.0),
                          key=jax.random.PRNGKey(0), G=STELLAR.G)
    return ic, masses, STELLAR.G


def run_stage(stage: str) -> None:
    """Run one stage's workload in THIS process and print its peak RSS."""
    n, _ = STAGES[stage]

    if stage == "import":
        import progenax  # noqa: F401
    else:
        import jax
        import jax.numpy as jnp
        from jaxstro.units import STELLAR
        key = jax.random.PRNGKey(0)

        if stage in ("engineA_iso", "engineA_aniso"):
            from progenax import MultiComponentCluster
            ra = {"ra_hat_j": jnp.array([10.0, 10.0])} if stage == "engineA_aniso" else {}
            model = MultiComponentCluster.from_components(
                alpha_j=jnp.array([0.5, 0.5]), w_j=jnp.array([1.0, 0.6]),
                m_j=jnp.array([0.8, 0.8]), W0=7.0, g=1.0, r_c=1.0, **ra)
            ic = model.sample_cluster(key, n, G=STELLAR.G)
            jax.block_until_ready(ic.velocities)

        elif stage == "engineB_halo_core":
            from progenax import EFFProfile, MultiComponentCluster, PlummerProfile
            model = MultiComponentCluster.from_density_profiles(
                [PlummerProfile(r_h=2.0), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)],
                mass_fractions=jnp.array([0.6, 0.4]), m_j=jnp.array([0.5, 1.0]))
            ic = model.sample_cluster(key, n, G=STELLAR.G)
            jax.block_until_ready(ic.velocities)

        elif stage == "virial_pe":
            from progenax import compute_potential_energy
            ic, masses, G = _plummer_ic(n)
            pe = compute_potential_energy(ic.positions, masses, G=G)
            pe.block_until_ready()

        elif stage == "group_virial":
            from progenax.dynamics import mass_group_masks, per_group_virial_ratio
            ic, masses, G = _plummer_ic(n)
            masks = mass_group_masks(masses, 4)
            q_j = per_group_virial_ratio(ic.positions, ic.velocities, masses, G,
                                         masks)
            q_j.block_until_ready()

        elif stage == "limepy_df_aniso":
            from progenax.kinematics.limepy_df import LIMEPYVelocityDF
            from progenax.profiles.limepy import LIMEPYProfile
            prof = LIMEPYProfile.from_W0_rc(W0=7.0, g=1.0, r_c=1.0)
            df = LIMEPYVelocityDF(W0=7.0, g=1.0, r_c=1.0, r_a=10.0)
            masses = jnp.ones(n)
            pos = prof.sample_positions(masses, key)
            vel = df.sample_velocities(pos, masses, jax.random.PRNGKey(1),
                                       G=STELLAR.G)
            vel.block_until_ready()

    print(f"{_RESULT_PREFIX} {_peak_rss_gb():.3f}", flush=True)


def _run_subprocess(stage: str):
    """Run one stage as a subprocess; return (measured_gb | None, error_text).

    A result line only counts if the subprocess exited 0; a nonzero exit or a
    missing result line means the stage CRASHED (measured is None).
    """
    cmd = [sys.executable, __file__, "--stage", stage]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if line.startswith(_RESULT_PREFIX):
                return float(line.split()[1]), ""
    err = proc.stderr.strip()[-1500:] or "no result line in stage output"
    return None, f"exit code {proc.returncode}: {err}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Staged peak-RSS memory profile with PASS gates.")
    parser.add_argument("--stage", choices=sorted(STAGES),
                        help="run a single stage in-process (subprocess re-entry)")
    parser.add_argument("--allow-fail", action="append", default=[],
                        metavar="STAGE", choices=sorted(STAGES),
                        help="mark this stage's gate ALLOWED-FAIL on gate "
                             "excess only; stage crashes always FAIL "
                             "(repeatable)")
    args = parser.parse_args()

    if args.stage:
        run_stage(args.stage)
        return 0

    print("=" * 72)
    print("CLUSTER MEMORY PROFILE (peak RSS per stage, subprocess-isolated)")
    print(f"platform: {sys.platform} (ru_maxrss in "
          f"{'bytes' if sys.platform == 'darwin' else 'kilobytes'})")
    print("=" * 72)

    rows = []
    any_fatal = False
    for stage, (n, gate_gb) in STAGES.items():
        measured, err = _run_subprocess(stage)
        if measured is None:
            # Crashes ALWAYS fail; --allow-fail only covers gate excess.
            status = "FAIL"
            print(f"  {stage}: stage ERROR\n{err}", flush=True)
        else:
            ok = measured < gate_gb
            status = ("PASS" if ok
                      else "ALLOWED-FAIL" if stage in args.allow_fail else "FAIL")
        any_fatal |= status == "FAIL"
        rows.append((stage, n, gate_gb, measured, status))
        m_str = f"{measured:.2f}" if measured is not None else "ERROR"
        print(f"  {stage:<20} measured {m_str:>6} GB  (gate < {gate_gb:.1f} GB)"
              f"  {status}", flush=True)

    print("-" * 72)
    print(f"  {'stage':<20} {'N':>8} {'gate [GB]':>10} {'measured [GB]':>14} "
          f"{'status':>13}")
    for stage, n, gate_gb, measured, status in rows:
        n_str = f"{n:,}" if n else "-"
        m_str = f"{measured:.2f}" if measured is not None else "ERROR"
        print(f"  {stage:<20} {n_str:>8} {gate_gb:>10.1f} {m_str:>14} "
              f"{status:>13}")
    print("=" * 72)
    print("  MEMORY GATES PASS" if not any_fatal
          else "  MEMORY GATES FAILED (non-allowed gate excess or stage crash)")
    return 1 if any_fatal else 0


if __name__ == "__main__":
    sys.exit(main())
