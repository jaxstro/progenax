#!/usr/bin/env bash
# Local released-core measurement: one full coverage/duration run, followed by
# artifact-consumer and serial timing checks. Remote CI authorization is separate.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="env -u VIRTUAL_ENV uv run --no-sync"
XLA_CAPS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
WORKERS="${PROGENAX_RELEASE_WORKERS:-3}"
case "$WORKERS" in
    1|2|3|4) ;;
    *) echo "PROGENAX_RELEASE_WORKERS must be 1, 2, 3 or 4" >&2; exit 2 ;;
esac
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/progenax-release.XXXXXX")"
RAW_COV="$RUN_DIR/coverage-raw.json"
FULL_LOG="$RUN_DIR/full-tests.log"
export JAX_ENABLE_X64=true PROGENAX_STRICT_REFS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_DIR/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
echo "Release logs: $RUN_DIR"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "Release gate failed (exit $rc); retained logs: $RUN_DIR" >&2; fi' EXIT

# Same source/test/config scope used by the existing coverage freshness guard.
# Generated coverage/dashboard/duration records are outputs, not these inputs.
COVERAGE_INPUTS=(
    src/progenax/ tests/unit/ tests/integration/ tests/validation/
    tests/conftest.py tests/__init__.py conftest.py pyproject.toml
    pytest.ini .coveragerc setup.cfg tox.ini
)
MEASURED_SHA="$(git -c gc.auto=0 -c maintenance.auto=false rev-parse --verify HEAD)"
assert_frozen_inputs() {
    local current untracked
    current="$(git -c gc.auto=0 -c maintenance.auto=false rev-parse --verify HEAD)"
    if [ "$current" != "$MEASURED_SHA" ]; then
        echo "HEAD changed during measurement; do not stamp this run as another commit" >&2
        return 1
    fi
    if ! git -c gc.auto=0 -c maintenance.auto=false diff --quiet "$MEASURED_SHA" -- "${COVERAGE_INPUTS[@]}" ||
       ! git -c gc.auto=0 -c maintenance.auto=false diff --cached --quiet "$MEASURED_SHA" -- "${COVERAGE_INPUTS[@]}"; then
        echo "Commit/freeze released-core source, tests and configuration before measurement" >&2
        return 1
    fi
    untracked="$(git -c gc.auto=0 -c maintenance.auto=false ls-files --others --exclude-standard -- "${COVERAGE_INPUTS[@]}")"
    if [ -n "$untracked" ]; then
        echo "Untracked released-core measurement inputs remain:" >&2
        printf '%s\n' "$untracked" >&2
        return 1
    fi
}
assert_frozen_inputs

if [ "${PROGENAX_RELEASE_SKIP_SYNC:-0}" = "1" ]; then
    echo "Using the installed environment without syncing or removing optional dependencies"
else
    env -u VIRTUAL_ENV uv sync --locked --extra dev --extra diagnostics --extra viz
fi

# Refuse missing caches before a validation CLI can provision a foreign Python
# environment. No --regen or reference download is part of this release gate.
$RUN python - <<'PY'
import subprocess
from pathlib import Path
import numpy as np
from scripts.build_test_dashboard import RELEASED_VALIDATION_SCRIPTS
root = Path.cwd()
tracked = set(subprocess.check_output(["git", "ls-files", "scripts/*.py"], text=True).splitlines())
missing_scripts = [name for name in RELEASED_VALIDATION_SCRIPTS
                   if f"scripts/{name}" not in tracked or not (root / "scripts" / name).is_file()]
cache = root / "validation/data/limepy_reference"
required = [cache / f"{name}.npz" for name in (
    "single_g1", "single_g15", "twocomp_iso", "imf4", "twocomp_ra_eta0", "twocomp_ra_eta05")]
required.append(cache / "absolute/level-2-base.npz")
required += [cache / "absolute" / f"level-2-{name}_{sign}_{step}.npz"
             for name in ("alpha", "delta", "W0")
             for sign in ("plus", "minus") for step in (1e-4, 1e-5)]
missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
if missing_scripts or missing:
    raise SystemExit(f"Release prerequisites missing; no regeneration attempted: scripts={missing_scripts}, caches={missing}")
for path in required:
    with np.load(path, allow_pickle=False) as cached:
        if not cached.files:
            raise SystemExit(f"Empty reference cache: {path.relative_to(root)}")
print(f"Preflight: {len(RELEASED_VALIDATION_SCRIPTS)} core validation scripts and {len(required)} cached references")
PY

# Refresh the existing registry under its unchanged per-case criteria. A hazard
# aborts here; neither tolerances nor statuses are rewritten to obtain a pass.
if [ "${PROGENAX_RELEASE_SKIP_GRAD_REFRESH:-0}" = "1" ]; then
    echo "Reusing gradient JSON provisionally; independent serial freshness and primary hazard checks remain required"
else
    env XLA_FLAGS="$XLA_CAPS" $RUN python scripts/audit_gradients.py \
        2>&1 | tee "$RUN_DIR/gradient-refresh.log"
fi
assert_frozen_inputs

echo "== release-gate: one full released-core coverage and duration measurement =="
# Defer only artifact consumers. The two fresh_audit consumers run together in
# one serial post-phase, so their session fixture regenerates once, not per worker.
env -u VIRTUAL_ENV XLA_FLAGS="$XLA_CAPS" uv run --no-sync \
    pytest tests/unit tests/integration tests/validation \
    --cov=progenax --cov-branch --cov-report=json:"${RAW_COV}" \
    --durations=0 -n "$WORKERS" --dist loadscope -p no:cacheprovider \
    --deselect tests/validation/test_dashboard_fresh.py \
    --deselect tests/validation/api_coverage/test_api_coverage.py::test_line_coverage_above_floor \
    --deselect tests/validation/grad_audit/test_audit_script.py \
    --deselect tests/validation/grad_audit/test_json_fresh.py \
    2>&1 | tee "$FULL_LOG"
assert_frozen_inputs

# Parse durations from the same successful primary measurement; never rerun the
# full suite merely to obtain timings, and never suppress a failed pytest exit.
$RUN python - "$FULL_LOG" <<'PY'
import json, re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
def node_to_module(node_id):
    parts = Path(node_id.split("::", 1)[0]).parts
    if len(parts) < 3 or parts[0] != "tests" or parts[1] not in {"unit", "integration", "validation"}:
        return None
    rest = parts[2:]
    return rest[0] if len(rest) >= 2 else Path(rest[0]).stem
line_re = re.compile(r"^\s*([\d.]+)s\s+(call|setup|teardown)\s+(\S+::\S+)\s*$")
node_secs = {}
for line in text.splitlines():
    match = line_re.match(line)
    if match:
        node_secs[match.group(3)] = node_secs.get(match.group(3), 0.0) + float(match.group(1))
if not node_secs:
    raise SystemExit("Primary pytest log contains no duration rows")
modules = {}
for node, seconds in node_secs.items():
    module = node_to_module(node)
    if module is not None and (module not in modules or seconds > modules[module]["seconds"]):
        modules[module] = {"slowest_test": node, "seconds": round(seconds, 2)}
out = {"_note": "Primary full released-core coverage run; deferred artifact consumers excluded. Seconds sum call+setup+teardown.",
       "modules": dict(sorted(modules.items()))}
Path("validation/data/durations.json").write_text(json.dumps(out, indent=2) + "\n")
PY

# Exact canonical core list, not a filename discovery of local research scripts.
$RUN python - "$RUN_DIR" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path
from scripts.build_test_dashboard import RELEASED_VALIDATION_SCRIPTS
root = Path.cwd()
env = dict(os.environ)
env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
results = {}
for name in RELEASED_VALIDATION_SCRIPTS:
    started = time.monotonic()
    log = Path(sys.argv[1]) / (name + ".log")
    with log.open("w") as stream:
        try:
            process = subprocess.run([sys.executable, str(root / "scripts" / name)],
                cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    results[name] = code
    print(f"{name}: exit={code} ({time.monotonic()-started:.1f}s)", flush=True)
Path("validation/data/validation_runs.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
PY
assert_frozen_inputs

echo "== release-gate: stamp measured coverage and regenerate dashboard =="
# HEAD and all scoped inputs are still the frozen measurement commit.
$RUN python scripts/build_test_dashboard.py --stamp-coverage "${RAW_COV}"
$RUN python scripts/build_test_dashboard.py --emit --render

echo "== release-gate: serial artifact-consumer verification =="
env -u VIRTUAL_ENV XLA_FLAGS="$XLA_CAPS" uv run --no-sync \
    pytest -n 0 -p no:cacheprovider \
    tests/validation/api_coverage/test_api_coverage.py::test_line_coverage_above_floor \
    tests/validation/test_dashboard_fresh.py \
    tests/validation/grad_audit/test_audit_script.py \
    tests/validation/grad_audit/test_json_fresh.py \
    tests/unit/profiles/test_limepy_tables.py::TestTableBackedSolver::test_table_solve_is_faster -q \
    2>&1 | tee "$RUN_DIR/artifact-checks.log"
assert_frozen_inputs
FULL_SUITE_GREEN=true

echo "== release-gate: assert the 5-part conjunction =="
# Read the regenerated dashboard's gate block and assert all five parts. full_suite_green is
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
# validation_scripts_pass: every scripts/validate_*.py exited 0 in this run's refresh. A
# broken validate script (API drift, failed assertion) reds the release gate — it no longer
# sits invisibly under green (the Phase-3 gap).
val_scripts = dash.get("validation_scripts", {})
def _code(v):
    return v.get("exit") if isinstance(v, dict) else v
# Recorded JSON statuses must be explicit integer exit code 0. A timeout,
# unknown/missing value, float, or bool is not a successful process exit.
val_fails = sorted(k for k, v in val_scripts.items() if not (type(_code(v)) is int and _code(v) == 0))
validation_scripts_pass = bool(val_scripts) and not val_fails
if val_fails:
    print("  WARN: failing validate scripts:", ", ".join(val_fails))
ok = (registries_full and line_cov_ok and dashboard_fresh and full_suite_green
      and validation_scripts_pass)
verdict = "PASS" if ok else "FAIL"
cov_str = f"{line_cov:.2f}" if line_cov is not None else "None"
print(
    f"RELEASE GATE: registries_full={registries_full} "
    f"line_cov={cov_str}(>={floor}) "
    f"dashboard_fresh={dashboard_fresh} "
    f"full_suite_green={full_suite_green} "
    f"validation_scripts_pass={validation_scripts_pass}({len(val_scripts)} scripts) -> {verdict}"
)
sys.exit(0 if ok else 1)
PY

echo "RELEASE GATE PASSED"
