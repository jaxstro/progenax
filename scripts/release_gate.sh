#!/usr/bin/env bash
# Heavy, periodic two-tier RELEASE GATE. The FAST continuous gate (scripts/check.sh)
# already runs the four registry ratchets + the FAST dashboard-freshness teeth + the
# coverage-floor READ inside `pytest -m "not slow"`. THIS script is the HEAVY tier: it
# re-measures FULL-suite coverage, refreshes the observability artifacts, regenerates
# coverage.json + the dashboard at HEAD, runs the floor + freshness tests (incl. the @slow
# full-regeneration match), and asserts the 4-part conjunction:
#
#   registries_full  AND  line_cov_measured >= line_cov_floor  AND  dashboard_fresh
#       AND  full_suite_green
#
# Run from the repo root. Exits non-zero on FAIL.
#
# progenax notes (vs the fluxax template this is adapted from):
#   - The dashboard CLI is progenax's: --stamp-coverage RAW_COV (writes coverage.json),
#     --emit (writes test_dashboard.json), --render (writes the MyST page). It has NO
#     fluxax-style --write/--timestamp/--selector/--git-sha/--measured-utc/--note flags;
#     selector/git_sha/measured_utc are derived INSIDE --stamp-coverage from HEAD.
#   - Coverage measures src/progenax; the gate floor is 90 (build_test_dashboard._LINE_COV_FLOOR).
#   - The committed coverage file is validation/data/coverage.json, the dashboard is
#     validation/data/test_dashboard.json, the freshness test is
#     tests/validation/test_dashboard_fresh.py.
#   - Test invocation uses the XLA thread caps + pytest-xdist -n auto (progenax/CLAUDE.md).
#   - There is NO ruff/mypy step (progenax configures neither; CI lints nothing) — see check.sh.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

RUN="env -u VIRTUAL_ENV uv run --no-sync"
XLA_CAPS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
RAW_COV="/tmp/progenax_release_cov.json"

echo "== release-gate: sync (--all-extras; best-effort, see note) =="
# Full provisioning so the WHOLE suite collects + runs (some validation tests need the
# diagnostics/experimental deps). Best-effort: a fresh `uv sync` re-resolves the lock and
# currently hits a PRE-EXISTING requires-python wall (pyproject ">=3.10" vs jaxstro ">=3.11";
# a one-line pyproject fix, OUT OF SCOPE here, that also reds CI's `uv sync --locked`). The
# already-provisioned .venv satisfies the suite; every step below uses `--no-sync`. WARN and
# proceed if sync fails; once requires-python is tightened the WARN disappears.
env -u VIRTUAL_ENV uv sync --all-extras || \
    echo "WARNING: uv sync --all-extras failed (pre-existing requires-python mismatch); using the existing .venv via --no-sync."

echo "== release-gate: FULL-suite coverage (heavy step, ~14-36 min) =="
# The FULL suite (NOT -m "not slow"): a partial run understates coverage and would game the
# floor. --cov-branch matches the committed artifact's branch coverage. DESELECT the
# dashboard-freshness meta-tests: they assert the committed coverage.json is already fresh,
# which is CIRCULAR during the very run that produces the new coverage. They test scripts/
# behavior, not src/progenax, so excluding them does not affect the coverage number; they
# run in the post-regeneration verification step further down.
env -u VIRTUAL_ENV XLA_FLAGS="$XLA_CAPS" uv run --no-sync \
    pytest tests/unit tests/integration tests/validation \
    --cov=progenax --cov-branch --cov-report=json:"${RAW_COV}" \
    -n auto -p no:cacheprovider \
    --deselect tests/validation/test_dashboard_fresh.py
FULL_SUITE_GREEN=true   # set only because the line above did NOT abort (set -e)

echo "== release-gate: refresh observability artifacts (validation_runs + durations) =="
# durations.json: per-module slowest test over the FULL suite (release-grade refresh of the
# fast-tier snapshot check.sh commits). Reuses the same bucketing as the dashboard census.
env -u VIRTUAL_ENV XLA_FLAGS="$XLA_CAPS" uv run --no-sync \
    pytest tests/unit tests/integration tests/validation -n auto --durations=0 -q \
    > /tmp/progenax_full_durations.txt 2>&1 || true
$RUN python - <<'PY'
import json, re
from pathlib import Path
root = Path.cwd()
text = Path("/tmp/progenax_full_durations.txt").read_text()
TIERS = {"unit", "integration", "validation"}
def node_to_module(node_id):
    parts = Path(node_id.split("::", 1)[0]).parts
    if len(parts) < 3 or parts[0] != "tests" or parts[1] not in TIERS:
        return None
    rest = parts[2:]
    return rest[0] if len(rest) >= 2 else Path(rest[0]).stem
line_re = re.compile(r"^\s*([\d.]+)s\s+(call|setup|teardown)\s+(\S+::\S+)\s*$")
node_secs = {}
for ln in text.splitlines():
    m = line_re.match(ln)
    if m:
        node_secs[m.group(3)] = node_secs.get(m.group(3), 0.0) + float(m.group(1))
modules = {}
for node, secs in node_secs.items():
    mod = node_to_module(node)
    if mod is None:
        continue
    if mod not in modules or secs > modules[mod]["seconds"]:
        modules[mod] = {"slowest_test": node, "seconds": round(secs, 2)}
out = {
    "_note": ("FULL-suite per-module slowest test (seconds = call+setup+teardown), "
              "refreshed by scripts/release_gate.sh. Read by "
              "scripts/build_test_dashboard.py::read_durations."),
    "modules": dict(sorted(modules.items())),
}
Path("validation/data/durations.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"refreshed durations.json with {len(modules)} module buckets")
PY
# validation_runs.json: real exit code of every scripts/validate_*.py.
$RUN python - <<'PY'
import json, os, subprocess, time
from pathlib import Path
root = Path.cwd()
env = dict(os.environ)
env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
results = {}
for s in sorted((root / "scripts").glob("validate_*.py")):
    t0 = time.time()
    try:
        p = subprocess.run(
            ["env", "-u", "VIRTUAL_ENV", "uv", "run", "--no-sync", "python", str(s)],
            cwd=str(root), env=env, capture_output=True, text=True, timeout=900)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    results[s.name] = rc
    print(f"{s.name}: exit={rc} ({time.time()-t0:.1f}s)", flush=True)
Path("validation/data/validation_runs.json").write_text(
    json.dumps(results, indent=2, sort_keys=True) + "\n")
print(f"refreshed validation_runs.json ({len(results)} scripts)")
PY

echo "== release-gate: stamp coverage.json + regen dashboard.json + page at HEAD =="
$RUN python scripts/build_test_dashboard.py --stamp-coverage "${RAW_COV}"
$RUN python scripts/build_test_dashboard.py --emit --render

echo "== release-gate: floor + freshness pytest tests (now fresh) =="
# Fast given freshly-regenerated artifacts. The @slow full-regeneration match re-collects
# the suite to semantic-diff the dashboard; run it explicitly here (drop -m filter).
env -u VIRTUAL_ENV XLA_FLAGS="$XLA_CAPS" uv run --no-sync \
    pytest -p no:cacheprovider \
    tests/validation/api_coverage/test_api_coverage.py::test_line_coverage_above_floor \
    tests/validation/test_dashboard_fresh.py -q

echo "== release-gate: assert the 4-part conjunction =="
# Read the regenerated dashboard's gate block and assert all four parts. full_suite_green is
# injected from THIS shell run (the dashboard records None — it is introspection-only).
$RUN python - "$FULL_SUITE_GREEN" <<'PY'
import json, sys
from pathlib import Path
full_suite_green = sys.argv[1] == "true"
dash = json.loads(Path("validation/data/test_dashboard.json").read_text())
gate = dash["gate"]
registries_full = bool(gate["registries_full"])
line_cov = gate["line_cov_measured"]
floor = gate["line_cov_floor"]
line_cov_ok = line_cov is not None and float(line_cov) >= float(floor)
# dashboard_fresh: the freshness pytest step above passed (set -e would have aborted on a
# drift), so by here the committed dashboard matched a fresh regeneration.
dashboard_fresh = True
ok = registries_full and line_cov_ok and dashboard_fresh and full_suite_green
verdict = "PASS" if ok else "FAIL"
cov_str = f"{line_cov:.2f}" if line_cov is not None else "None"
print(
    f"RELEASE GATE: registries_full={registries_full} "
    f"line_cov={cov_str}(>={floor}) "
    f"dashboard_fresh={dashboard_fresh} "
    f"full_suite_green={full_suite_green} -> {verdict}"
)
sys.exit(0 if ok else 1)
PY

echo "RELEASE GATE PASSED"
