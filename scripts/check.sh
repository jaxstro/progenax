#!/usr/bin/env bash
# Local mirror of the dormant GitHub Actions gate (Actions minutes are exhausted;
# both workflows are disabled_manually). Run from the repo root. Any failure aborts
# (set -e). This is the FAST continuous gate — keep it fast (a few minutes); the
# HEAVY full-suite coverage re-measure + dashboard regen + 4-part conjunction live in
# scripts/release_gate.sh, NOT here.
#
# progenax differs from the sibling fluxax/jaxstro gates:
#   1. Lint: progenax now declares [tool.ruff]/[tool.mypy] in pyproject (jaxstro-matched)
#      and its `dev` extra carries ruff/mypy. We run `ruff check`, `ruff format --check`,
#      and `mypy src/progenax` as HARD steps before the test tier (mirroring fluxax/jaxstro).
#   2. The test invocation uses the XLA thread caps + pytest-xdist `-n auto`
#      (progenax/CLAUDE.md "Quick Commands"): the multimass-LIMEPY equilibrium tests
#      make the serial suite slow, so we cap XLA threads and shard with xdist.
#   3. The wheel depends on the PUBLIC jaxstro sibling (local path source),
#      not PyPI, so the clean-venv smoke installs the jaxstro checkout alongside the
#      wheel (mirrors CI's wheel-smoke job) and imports progenax + progenax.diagnostics
#      (the R9 lazy-scipy carve: the clean venv has no scipy, so diagnostics must
#      import lazily).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
RUN="env -u VIRTUAL_ENV uv run --no-sync"
XLA="XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

echo "== lock-check =="
# `uv lock --check` asserts uv.lock is in sync with pyproject.toml. This is a HARD step
# (set -e aborts on failure), mirroring CI's `uv sync --locked`. pyproject's
# `requires-python = ">=3.11"` matches the jaxstro path-dependency's `>=3.11` floor, so the
# lock resolves cleanly across the whole declared range. (Earlier this was best-effort WARN
# because pyproject declared ">=3.10", which jaxstro could not satisfy; that floor is fixed.)
env -u VIRTUAL_ENV uv lock --check

echo "== sync (extra dev) =="
# HARD step (set -e aborts on failure): with requires-python tightened to ">=3.11" the lock
# re-resolves cleanly at the active interpreter. Released-core syncs ONLY `--extra dev`:
# the OED-demo unit tests (tests/unit/test_demo_oed*.py) are informax-bound and import optax
# (experimental) / matplotlib (via jaxstroviz, undeclared) — they `pytest.importorskip(...)`
# those deps, so they SKIP cleanly on a dev-only env rather than coupling released-core to
# the experimental/viz extras. Run them under `--extra experimental` (+ a jaxstroviz checkout
# for the CLI smokes) when you want the OED demos exercised.
env -u VIRTUAL_ENV uv sync --extra dev

echo "== lint: ruff check =="
$RUN ruff check src/progenax tests scripts
echo "== lint: ruff format --check =="
$RUN ruff format --check src/progenax tests scripts
echo "== lint: mypy =="
$RUN mypy src/progenax

echo "== fast test tier (-m 'not slow', xdist, XLA-capped) =="
# This is the FAST tier of the two-tier gate. It runs:
#   - the four registry ratchets (tests/validation/{grad_audit,api_coverage,
#     physics_registry,provenance_registry}/),
#   - the FAST dashboard-freshness teeth (tests/validation/test_dashboard_fresh.py:
#     ::test_coverage_provenance_teeth_fire_on_stale_src + ::test_committed_page_
#     matches_render_of_committed_json — fast git+json/render checks; the full-suite
#     regeneration match is @slow and runs only in release_gate.sh),
#   - the coverage-floor READ (api_coverage/test_api_coverage.py::
#     test_line_coverage_above_floor, which reads the COMMITTED
#     validation/data/coverage.json — it does NOT re-measure coverage).
# It must stay fast: NO @slow test, NO `--cov` re-measure here.
env -u VIRTUAL_ENV XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    uv run --no-sync pytest tests/unit tests/integration tests/validation \
    -m "not slow" -n auto -q

echo "== wheel-smoke =="
env -u VIRTUAL_ENV uv build --wheel -o dist/
rm -rf /tmp/progenax-clean
env -u VIRTUAL_ENV uv venv /tmp/progenax-clean
# progenax's wheel depends on the PUBLIC jaxstro sibling, a local path source (see
# [tool.uv.sources]) NOT on PyPI. In a clean venv that source does not apply, so PyPI
# resolution would fail. Install the jaxstro local checkout alongside the wheel so the
# smoke honestly verifies `import progenax` from the built artifact. We also import
# progenax.diagnostics: the clean venv has no scipy, so a regression to an eager
# numpy/scipy import (audit R9) would red here.
env -u VIRTUAL_ENV uv pip install --python /tmp/progenax-clean/bin/python \
    /Users/anna/projects/jaxstro-dev/jaxstro \
    dist/*.whl
/tmp/progenax-clean/bin/python -c \
    "import progenax; import progenax.diagnostics; print(progenax.__name__, 'imports clean (no scipy in venv)')"

echo "ALL LOCAL GATES PASSED"
