"""Task 1.6 staleness gate (the ratchet): the committed ``validation/data/
test_dashboard.json`` must match a fresh IN-PROCESS regeneration. This is what
makes the dashboard non-drifting by construction — edit a test, change a registry,
or move a validate-script, and this gate goes RED until the JSON is regenerated.

Mirrors ``grad_audit/test_json_fresh.py``: regenerate via ``build_dashboard()``,
semantic-diff vs the committed JSON — EXACT on the discrete/structural fields
(module test counts, registry status, gate flags, validation-script exit codes,
durations/line-coverage status), rtol-tolerant on the only floats (per-module
``line_cov`` percentages). The ``generated_utc`` timestamp is IGNORED (it always
differs; we pass a fixed sentinel so even a literal compare of that field would be
benign).

INTROSPECTION-ONLY / COST: ``build_dashboard`` calls ``collect_test_inventory()``,
which shells out to ``pytest --collect-only`` (~10-35s). That is the ONLY heavy
part — it does NOT run the suite or ``--cov``. Measured wall-time here is >15s, so
this test is ``@pytest.mark.slow``: it re-collects the whole suite, which is full-
gate work, not fast-inner-loop work. The FAST gate stays fast; the FULL gate
catches dashboard drift.

_RTOL is for the line-coverage floats. There are NO measured floats yet
(``line_coverage`` is ``"not-measured"``, every per-module ``line_cov`` is
``None``), so the float path is forward-looking for Phase 2. It is implemented now
so the gate is ready the moment coverage is wired in. Same-process regeneration
re-parses the SAME committed ``coverage.json`` byte-for-byte, so the deltas will be
exactly zero; 1e-6 is an ample float64 round-trip margin. If Phase 2's CI ever
regenerates coverage on a different arch, widen to the measured max (do NOT loosen
blindly — a large reldiff on a same-suite parse is a real drift, not noise).
"""
import json
import math
import subprocess
import time

import pytest

from scripts.build_test_dashboard import _DASHBOARD_JSON, _REPO_ROOT, build_dashboard

# Stamped verbatim into generated_utc by build_dashboard; the gate IGNORES this
# field, so any fixed value works. A literal sentinel makes the intent explicit.
_FIXED_TS = "1970-01-01T00:00:00+00:00"

# rtol for the line-coverage floats (per-module line_cov %). See module docstring.
_RTOL = 1e-6

# Anti-vacuous floor (mirrors grad-audit's _MIN_ROWS): a semantic diff over two
# silently-emptied censuses passes trivially. Pin a sane minimum so a collapsed
# inventory is caught as drift, not waved through. The live census has 40+ module
# buckets and ~1500 collected tests; these floors carry ample margin while staying
# count-agnostic (they needn't track the exact totals).
_MIN_MODULES = 30
_MIN_TESTS = 1000

_TIER_KEYS = ("unit", "integration", "validation")


def _total_tests(dashboard: dict) -> int:
    return sum(
        m[tier] for m in dashboard["modules"].values() for tier in _TIER_KEYS
    )


def _diff_floats(label: str, cv, fv, drift: list) -> None:
    """Compare two line_cov values: None==None exact; floats rtol-tolerant."""
    if cv is None or fv is None:
        if cv != fv:
            drift.append(f"{label}: committed={cv!r} fresh={fv!r}")
        return
    cv, fv = float(cv), float(fv)
    c_fin, f_fin = math.isfinite(cv), math.isfinite(fv)
    if c_fin != f_fin:
        drift.append(f"{label}: finiteness changed committed={cv!r} fresh={fv!r}")
    elif not c_fin:
        if repr(cv) != repr(fv):
            drift.append(f"{label}: non-finite kind changed committed={cv!r} fresh={fv!r}")
    else:
        denom = max(abs(cv), abs(fv), 1e-30)
        if abs(cv - fv) / denom > _RTOL:
            drift.append(
                f"{label}: committed={cv:.6e} fresh={fv:.6e} "
                f"(reldiff={abs(cv - fv) / denom:.2e} > rtol={_RTOL:.0e})"
            )


def _diff_modules(committed: dict, fresh: dict, drift: list) -> None:
    """Per-module: EXACT on tier counts, rtol on line_cov, exact module-set."""
    cmods, fmods = committed["modules"], fresh["modules"]
    cset, fset = set(cmods), set(fmods)
    if cset != fset:
        drift.append(
            f"module set drift: only committed={sorted(cset - fset)} "
            f"only fresh={sorted(fset - cset)}"
        )
    for module in sorted(cset & fset):
        c, f = cmods[module], fmods[module]
        for tier in _TIER_KEYS:
            if c[tier] != f[tier]:
                drift.append(
                    f"modules[{module!r}][{tier!r}]: committed={c[tier]!r} fresh={f[tier]!r}"
                )
        _diff_floats(f"modules[{module!r}]['line_cov']", c["line_cov"], f["line_cov"], drift)


def _head_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.mark.slow  # re-collects the whole suite (~10-35s); FULL-gate work, see docstring.
def test_committed_dashboard_matches_fresh_regeneration():
    committed = json.loads((_REPO_ROOT / _DASHBOARD_JSON).read_text())
    t0 = time.perf_counter()
    fresh = build_dashboard(_FIXED_TS)
    elapsed = time.perf_counter() - t0
    print(f"\n[dashboard staleness gate] in-process regeneration took {elapsed:.1f}s")

    # Anti-vacuous floor FIRST: a semantic diff over two emptied censuses passes
    # vacuously, so pin that neither side collapsed before trusting the diff.
    for side, dash in (("committed", committed), ("fresh", fresh)):
        n_mod, n_test = len(dash["modules"]), _total_tests(dash)
        assert n_mod >= _MIN_MODULES and n_test >= _MIN_TESTS, (
            f"{side} dashboard too small (modules={n_mod}, tests={n_test}; "
            f"floors modules>={_MIN_MODULES}, tests>={_MIN_TESTS}) — the census "
            f"looks emptied, not merely drifted."
        )

    drift: list[str] = []

    # IGNORE generated_utc (always differs); EXACT-compare every other top-level
    # block except `modules`, which has its own float-aware comparator. registries,
    # gate, line_coverage, durations, validation_scripts are all discrete/structural
    # JSON (status strings, ints, bools, exit codes) -> a plain == is the gate.
    _diff_modules(committed, fresh, drift)
    for block in ("registries", "gate", "line_coverage", "durations", "validation_scripts"):
        if committed.get(block) != fresh.get(block):
            drift.append(
                f"{block} drift:\n    committed={committed.get(block)!r}"
                f"\n    fresh={fresh.get(block)!r}"
            )

    # A new/removed top-level key (other than the ignored generated_utc) is drift.
    cdash_keys = set(committed) - {"generated_utc"}
    fdash_keys = set(fresh) - {"generated_utc"}
    if cdash_keys != fdash_keys:
        drift.append(
            f"top-level key drift: only committed={sorted(cdash_keys - fdash_keys)} "
            f"only fresh={sorted(fdash_keys - cdash_keys)}"
        )

    assert not drift, (
        "dashboard staleness drift (regenerate + recommit "
        f"{_DASHBOARD_JSON} if intended):\n  " + "\n  ".join(drift)
    )

    # Provenance check (conditional, Phase 2+): once line coverage is MEASURED, the
    # committed coverage carries a git_sha; a stale coverage.json (measured on an old
    # tree) must not pass silently. We do NOT re-run --cov — we only read the stamp.
    # When not-measured (now), there is no stamp to check -> skip.
    line_cov = committed["line_coverage"]
    if line_cov.get("status") == "measured":
        committed_sha = line_cov["coverage_provenance"]["git_sha"]
        assert committed_sha == _head_sha(), (
            f"stale coverage.json: provenance git_sha={committed_sha} != HEAD={_head_sha()} "
            f"— regenerate the FULL-suite coverage at the current commit."
        )
