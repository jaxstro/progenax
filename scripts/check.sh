#!/usr/bin/env bash
# Fast local released-core checks. Remote CI authorization is separate.
# The full coverage/measurement gate is scripts/release_gate.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="env -u VIRTUAL_ENV uv run --no-sync"
WORKERS="${PROGENAX_RELEASE_WORKERS:-3}"
case "$WORKERS" in
    1|2|3|4) ;;
    *) echo "PROGENAX_RELEASE_WORKERS must be 1, 2, 3 or 4" >&2; exit 2 ;;
esac

echo "== lock-check (offline) =="
env -u VIRTUAL_ENV uv lock --check --offline
if [ "${PROGENAX_RELEASE_SKIP_SYNC:-0}" = "1" ]; then
    echo "Using installed environment without syncing optional dependencies"
else
    env -u VIRTUAL_ENV uv sync --locked --extra dev
fi

# Lint released tiers and tracked public helpers. Local research scripts and the
# old extinction demo import experimental Gravoturb or private fluxax.
LINT_INPUTS=(
    src/progenax tests/unit tests/integration tests/validation
    tests/conftest.py tests/__init__.py docs/website/scripts laboratory
)
while IFS= read -r -d '' path; do
    case "$path" in
        *gravoturb*|scripts/demo_extinction_bands.py|scripts/sync_docs_tracker.py) continue ;;
    esac
    LINT_INPUTS+=("$path")
done < <(git -c gc.auto=0 -c maintenance.auto=false ls-files -z -- 'scripts/*.py')

echo "== lint: ruff check =="
$RUN ruff check --no-fix "${LINT_INPUTS[@]}"
echo "== lint: ruff format --check =="
$RUN ruff format --check "${LINT_INPUTS[@]}"
echo "== lint: mypy =="
$RUN mypy src/progenax

echo "== fast released-core tests =="
env -u VIRTUAL_ENV XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    uv run --no-sync pytest tests/unit tests/integration tests/validation \
    -m "not slow" -n "$WORKERS" --dist loadscope -q

echo "== wheel import smoke =="
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/progenax-check.XXXXXX")"
trap 'rm -rf "$SMOKE_ROOT"' EXIT
env -u VIRTUAL_ENV uv build --wheel -o "$SMOKE_ROOT/dist/"
env -u VIRTUAL_ENV uv venv "$SMOKE_ROOT/venv"
# The public jaxstro sibling supplies the editable local source dependency.
# JAX depends on SciPy: this checks declared-core imports, not SciPy absence.
env -u VIRTUAL_ENV uv pip install --python "$SMOKE_ROOT/venv/bin/python" \
    ../jaxstro \
    "$SMOKE_ROOT"/dist/*.whl
"$SMOKE_ROOT/venv/bin/python" -c \
    "import progenax; import progenax.diagnostics; print(progenax.__name__, 'wheel imports with declared core dependencies')"

echo "ALL LOCAL GATES PASSED"
