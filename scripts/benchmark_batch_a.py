#!/usr/bin/env python
"""Wall-clock + peak-RSS benchmark: Batch A kernels vs the pre-batch code.

Measures BOTH code versions live -- nothing hardcoded. The "before" version is
the commit Batch A branched from, checked out into a read-only git worktree;
each case runs in its OWN subprocess with PYTHONPATH pointing at either the
worktree's ``src/`` (before) or this repo's ``src/`` (after), so the same venv
serves both and peak RSS attributes cleanly per case.

Cases (kernels under test; IC/position construction is NOT timed):
  * ``compute_potential_energy`` on an equal-mass Plummer IC
      - timing: N in {2,000, 8,000} both versions; N = 20,000 after-only
        (the dense kernel at N = 20,000 allocates ~33 GB -- timed under swap
        pressure the number is meaningless, so it is reported as
        memory-infeasible instead of timed)
      - memory: N in {8,000, 20,000} both versions
  * anisotropic ``LIMEPYVelocityDF.sample_velocities`` (W0=7, g=1, r_c=1, r_a=10)
      - timing: N in {2,000, 10,000} both versions; N = 20,000 after-only;
        plus the after-version's exact-quadrature oracle
        (``speed_method="quadrature"``) at N = 10,000
      - memory: N = 20,000 both versions

Timing protocol: 1 warm-up call, then median of 5 timed calls
(``time.perf_counter`` around call + ``block_until_ready``), eager mode,
default CPU threading (identical env for both versions).

Sanity gate (memory only; timings are reported, never gated):
after-version ``compute_potential_energy`` peak RSS at N = 8,000 must be
>= 5x below the before-version's.

Outputs:
  * ``validation/benchmark_batch_a.json``      -- all measured numbers + metadata
  * ``validation/plots/performance_memory.{png,pdf}``
  * ``validation/plots/performance_walltime.{png,pdf}``

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/benchmark_batch_a.py
    # keep the pre-batch worktree around for inspection:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/benchmark_batch_a.py --keep-worktree
"""

import argparse
import datetime
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOT_DIR = REPO / "validation" / "plots"
JSON_PATH = REPO / "validation" / "benchmark_batch_a.json"
PRE_REF = "0dd1cd9"  # commit Batch A branched from (pre-batch kernels)
WORKTREE = Path("/tmp/progenax-prebatch")
TIME_REPS = 5
_RESULT_PREFIX = "BENCH_RESULT "

# (case_id, kind, workload, N, version, extra) -- the full measurement matrix.
# version: "before" (worktree src) | "after" (repo src).
CASES = [
    ("time_pe_2000_before", "time", "pe", 2_000, "before", None),
    ("time_pe_2000_after", "time", "pe", 2_000, "after", None),
    ("time_pe_8000_before", "time", "pe", 8_000, "before", None),
    ("time_pe_8000_after", "time", "pe", 8_000, "after", None),
    ("time_pe_20000_after", "time", "pe", 20_000, "after", None),
    ("time_df_2000_before", "time", "df", 2_000, "before", None),
    ("time_df_2000_after", "time", "df", 2_000, "after", None),
    ("time_df_10000_before", "time", "df", 10_000, "before", None),
    ("time_df_10000_after", "time", "df", 10_000, "after", None),
    ("time_df_20000_after", "time", "df", 20_000, "after", None),
    ("time_df_10000_quad", "time", "df", 10_000, "after", "quadrature"),
    ("mem_pe_8000_before", "mem", "pe", 8_000, "before", None),
    ("mem_pe_8000_after", "mem", "pe", 8_000, "after", None),
    ("mem_pe_20000_before", "mem", "pe", 20_000, "before", None),
    ("mem_pe_20000_after", "mem", "pe", 20_000, "after", None),
    ("mem_df_20000_before", "mem", "df", 20_000, "before", None),
    ("mem_df_20000_after", "mem", "df", 20_000, "after", None),
]


def _peak_rss_gb() -> float:
    """Process-lifetime peak RSS in GB (ru_maxrss: bytes on Darwin, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    return rss / 1e9


def run_case(kind: str, workload: str, n: int, src: str, extra) -> None:
    """Run one measurement case in THIS process and print its RESULT line."""
    import progenax  # resolved via PYTHONPATH=src (parent sets it)

    assert progenax.__file__.startswith(src), (
        f"progenax resolved to {progenax.__file__}, expected under {src}"
    )

    import jax
    import jax.numpy as jnp
    from jaxstro.units import STELLAR

    if workload == "pe":
        from progenax import (
            PlummerProfile,
            PlummerVelocityDF,
            build_spatial_ic,
            compute_potential_energy,
        )

        masses = jnp.ones(n)
        ic = build_spatial_ic(
            PlummerProfile(r_h=1.0),
            masses,
            PlummerVelocityDF(r_h=1.0),
            key=jax.random.PRNGKey(0),
            G=STELLAR.G,
        )
        pos = ic.positions

        def call():
            compute_potential_energy(pos, masses, G=STELLAR.G).block_until_ready()

    elif workload == "df":
        from progenax.kinematics.limepy_df import LIMEPYVelocityDF
        from progenax.profiles.limepy import LIMEPYProfile

        kw = {"speed_method": extra} if extra else {}
        prof = LIMEPYProfile.from_W0_rc(W0=7.0, g=1.0, r_c=1.0)
        df = LIMEPYVelocityDF(W0=7.0, g=1.0, r_c=1.0, r_a=10.0, **kw)
        masses = jnp.ones(n)
        pos = prof.sample_positions(masses, jax.random.PRNGKey(0))

        def call():
            df.sample_velocities(
                pos, masses, jax.random.PRNGKey(1), G=STELLAR.G
            ).block_until_ready()

    else:  # pragma: no cover - registry-guarded
        raise SystemExit(f"unknown workload {workload}")

    if kind == "mem":
        call()
        out = {"peak_rss_gb": round(_peak_rss_gb(), 3)}
    else:
        call()  # warm-up (not recorded)
        walls = []
        for _ in range(TIME_REPS):
            t0 = time.perf_counter()
            call()
            walls.append(time.perf_counter() - t0)
        out = {
            "wall_s_median": round(statistics.median(walls), 4),
            "wall_s_all": [round(w, 4) for w in walls],
        }
    print(_RESULT_PREFIX + json.dumps(out), flush=True)


def _run_subprocess(case_id, kind, workload, n, version, extra, worktree):
    """Run one case as a subprocess; return (result_dict | None, error_text)."""
    src = str((worktree if version == "before" else REPO) / "src")
    cmd = [sys.executable, __file__, "--case", case_id]
    env = dict(os.environ, PYTHONPATH=src)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if line.startswith(_RESULT_PREFIX):
                return json.loads(line[len(_RESULT_PREFIX) :]), ""
    err = proc.stderr.strip()[-1500:] or "no result line"
    return None, f"exit code {proc.returncode}: {err}"


def _ensure_worktree(worktree: Path) -> None:
    if (worktree / "src" / "progenax").is_dir():
        return
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), PRE_REF],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )


def _remove_worktree(worktree: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def _label(workload, n):
    name = "potential energy" if workload == "pe" else "aniso LIMEPY DF sampling"
    return f"{name} N={n:,}"


def make_figures(results: dict) -> None:
    """Publication figures from the measured results (Okabe-Ito, house style)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import matplotlib.pyplot as plt
    import numpy as np
    from _plotstyle import OI, apply_pub_style, panel_label, save_fig

    apply_pub_style()
    r = results

    # --- performance_memory: grouped bars, before vs after, log y ----------
    stages = [("pe", 8_000), ("pe", 20_000), ("df", 20_000)]
    before = [r[f"mem_{w}_{n}_before"]["peak_rss_gb"] for w, n in stages]
    after = [r[f"mem_{w}_{n}_after"]["peak_rss_gb"] for w, n in stages]
    tick = {"pe": "potential energy", "df": "aniso DF sampling"}
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    b1 = ax.bar(
        x - 0.18, before, 0.36, color=OI["vermilion"], label=f"pre-batch ({PRE_REF})"
    )
    b2 = ax.bar(x + 0.18, after, 0.36, color=OI["blue"], label="Batch A")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 200)
    ax.set_xticks(x, [f"{tick[w]}\n$N = {n:,}$" for w, n in stages], fontsize=8)
    ax.set_ylabel("peak RSS [GB]")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    save_fig(fig, PLOT_DIR, "performance_memory")

    # --- performance_walltime: median wall-clock vs N, two panels ----------
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    pe_ns = [2_000, 8_000]
    axa.plot(
        pe_ns,
        [r[f"time_pe_{n}_before"]["wall_s_median"] for n in pe_ns],
        "o-",
        color=OI["vermilion"],
        label=f"pre-batch ({PRE_REF})",
    )
    pe_ns_after = [2_000, 8_000, 20_000]
    axa.plot(
        pe_ns_after,
        [r[f"time_pe_{n}_after"]["wall_s_median"] for n in pe_ns_after],
        "s-",
        color=OI["blue"],
        label="Batch A (blocked)",
    )
    axa.annotate(
        f"N=20,000 pre-batch:\nmemory-infeasible\n"
        f"({r['mem_pe_20000_before']['peak_rss_gb']:.1f} GB peak RSS)",
        xy=(0.97, 0.05),
        xycoords="axes fraction",
        ha="right",
        fontsize=7.5,
        color=OI["vermilion"],
    )
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlabel("$N$ stars")
    axa.set_ylabel("median wall-clock [s]")
    axa.legend(loc="upper left")
    panel_label(axa, "(a)", loc="lower left")

    df_ns = [2_000, 10_000]
    axb.plot(
        df_ns,
        [r[f"time_df_{n}_before"]["wall_s_median"] for n in df_ns],
        "o-",
        color=OI["vermilion"],
        label="pre-batch (quadrature)",
    )
    df_ns_after = [2_000, 10_000, 20_000]
    axb.plot(
        df_ns_after,
        [r[f"time_df_{n}_after"]["wall_s_median"] for n in df_ns_after],
        "s-",
        color=OI["blue"],
        label="Batch A (table)",
    )
    axb.plot(
        [10_000],
        [r["time_df_10000_quad"]["wall_s_median"]],
        "D",
        color=OI["green"],
        label="Batch A quadrature oracle",
    )
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.margins(y=0.2)
    axb.set_xlabel("$N$ stars")
    axb.set_ylabel("median wall-clock [s]")
    axb.legend(loc="upper left")
    panel_label(axb, "(b)", loc="lower right")
    fig.tight_layout()
    save_fig(fig, PLOT_DIR, "performance_walltime")
    print(f"  wrote {PLOT_DIR}/performance_memory.(png|pdf)")
    print(f"  wrote {PLOT_DIR}/performance_walltime.(png|pdf)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch A wall-clock + memory benchmark (before vs after)."
    )
    parser.add_argument(
        "--case", help="run a single case in-process (subprocess re-entry)"
    )
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="keep the pre-batch worktree after the run",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="re-render the figures from the existing JSON (no re-measurement)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="PREFIX",
        help="run only cases whose id starts with PREFIX "
        "(repeatable); results MERGE into the existing "
        "JSON so unaffected cases keep their numbers",
    )
    args = parser.parse_args()

    if args.figures_only:
        results = json.loads(JSON_PATH.read_text())["results"]
        make_figures(results)
        return 0

    if args.case:
        matches = [c for c in CASES if c[0] == args.case]
        if not matches:
            raise SystemExit(f"unknown case: {args.case}")
        _, kind, workload, n, version, extra = matches[0]
        src = str((WORKTREE if version == "before" else REPO) / "src")
        run_case(kind, workload, n, src, extra)
        return 0

    print("=" * 76)
    print("BATCH A BENCHMARK (subprocess-isolated; before = pre-batch worktree)")
    print(f"before ref: {PRE_REF}   reps: median of {TIME_REPS} (1 warm-up)")
    print("=" * 76)
    cases = [
        c for c in CASES if not args.only or any(c[0].startswith(p) for p in args.only)
    ]
    if args.only and not cases:
        raise SystemExit(f"--only {args.only} matches no cases")
    # Merge mode: start from the existing JSON so unaffected cases persist.
    results = (
        json.loads(JSON_PATH.read_text())["results"]
        if args.only and JSON_PATH.exists()
        else {}
    )
    failed = []
    _ensure_worktree(WORKTREE)
    try:
        for case_id, kind, workload, n, version, extra in cases:
            res, err = _run_subprocess(
                case_id, kind, workload, n, version, extra, WORKTREE
            )
            if res is None:
                failed.append(case_id)
                print(f"  {case_id:<24} ERROR\n{err}", flush=True)
                continue
            res["measured"] = datetime.date.today().isoformat()
            results[case_id] = res
            val = (
                f"{res['peak_rss_gb']:8.2f} GB"
                if kind == "mem"
                else f"{res['wall_s_median']:8.3f} s "
            )
            print(f"  {case_id:<24} {val}", flush=True)
    finally:
        if not args.keep_worktree:
            _remove_worktree(WORKTREE)

    if failed:
        print(f"  BENCHMARK FAILED: {failed}")
        return 1

    # Sanity gate: blocked PE at N=8k must be >= 5x below the dense kernel.
    if not {"mem_pe_8000_before", "mem_pe_8000_after"} <= results.keys():
        raise SystemExit(
            "--only merge mode requires an existing full-run JSON "
            f"({JSON_PATH}) carrying the mem_pe_8000 gate cases -- run the "
            "full benchmark once first."
        )
    ratio = (
        results["mem_pe_8000_before"]["peak_rss_gb"]
        / results["mem_pe_8000_after"]["peak_rss_gb"]
    )
    gate_ok = ratio >= 5.0
    print("-" * 76)
    print(
        f"  memory gate: PE N=8k before/after = {ratio:.1f}x "
        f"(require >= 5x)  {'PASS' if gate_ok else 'FAIL'}"
    )

    payload = {
        "meta": {
            "date": datetime.date.today().isoformat(),
            "platform": sys.platform,
            "machine": platform.machine(),
            "before_ref": PRE_REF,
            "time_reps": TIME_REPS,
        },
        "results": results,
        "memory_gate_5x": gate_ok,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {JSON_PATH}")
    make_figures(results)
    print("=" * 76)
    print("  BENCHMARK PASS" if gate_ok else "  BENCHMARK FAILED (memory gate)")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
