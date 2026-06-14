"""Build the progenax test/validation dashboard.

This is the generator behind the website's ``test-dashboard`` page and the
staleness gate (Task 1.6). It is a plain SCRIPT, not core library code: it shells
out to ``pytest --collect-only`` and parses node ids, which is acceptable here
(it is NOT bound by the JAX-native constraint that governs ``src/progenax``).

Task 1.2 implements ``collect_test_inventory()`` — the per-module test census
across the three tiers (unit / integration / validation). Later tasks (1.3-1.5)
add line coverage, registry status, durations, and JSON/MyST emission.

Direct invocation (``python scripts/build_test_dashboard.py``) puts only
``scripts/`` on ``sys.path``; ``pyproject.toml`` sets
``pythonpath=["src","src/experimental"]`` which does NOT include the repo root,
so ``import tests.*`` / ``import scripts.*`` would ImportError. We mirror the
bootstrap in ``scripts/audit_gradients.py`` and insert the repo root first.
"""
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The three released-core tiers, in dashboard order.
_TIERS = ("unit", "integration", "validation")
_TIER_DIRS = tuple(f"tests/{tier}" for tier in _TIERS)

# pytest-cov keys each covered file by its repo-relative path; the released
# package lives under this prefix. We only report coverage for these files.
_SRC_PREFIX = "src/progenax/"


def _node_id_to_module_tier(node_id: str) -> tuple[str, str] | None:
    """Map a pytest node id to ``(module, tier)`` or ``None`` if outside the tiers.

    Node id forms (the part before ``::`` is the file path):

    - ``tests/unit/builders/test_x.py::test_name`` -> module ``builders``, tier ``unit``
      (a test inside a tier SUBDIRECTORY: module is the first subdir component).
    - ``tests/unit/test_protocols.py::test_name`` -> module ``test_protocols``, tier ``unit``
      (a test file DIRECTLY under the tier: module is the file stem — deterministic,
      and avoids collapsing several unrelated top-level files into one ``_root`` bucket).
    """
    file_path = node_id.split("::", 1)[0]
    parts = Path(file_path).parts
    if len(parts) < 3 or parts[0] != "tests" or parts[1] not in _TIERS:
        return None
    tier = parts[1]
    # parts[2:] is everything below the tier dir, ending in the test file.
    rest = parts[2:]
    if len(rest) >= 2:
        module = rest[0]  # first subdirectory under the tier
    else:
        module = Path(rest[0]).stem  # file directly under the tier -> file stem
    return module, tier


def collect_test_inventory() -> dict[str, dict[str, int]]:
    """Collect per-module test counts across the three tiers.

    Runs ``pytest --collect-only -q`` over ``tests/unit tests/integration
    tests/validation`` in a subprocess and parses the node ids.

    Returns ``{module: {"unit": n, "integration": n, "validation": n}}`` with all
    three tier keys present (zero-filled) for every module that has any tests.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *_TIER_DIRS, "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    inventory: dict[str, dict[str, int]] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" not in line:
            continue  # skip blank lines and the "N tests collected" summary
        mapped = _node_id_to_module_tier(line)
        if mapped is None:
            continue
        module, tier = mapped
        counts = inventory.setdefault(module, {t: 0 for t in _TIERS})
        counts[tier] += 1
    if not inventory:
        # No node ids parsed at all -> the collect failed (e.g. a collection error).
        # Surface it loudly rather than silently returning an empty census.
        raise RuntimeError(
            "pytest --collect-only produced no node ids; collection likely failed.\n"
            f"return code: {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout tail:\n{proc.stdout[-2000:]}"
        )
    return inventory


def _file_key_to_module(file_key: str) -> str:
    """Map a pytest-cov ``files`` key to a per-file module name, or ``""`` to skip.

    pytest-cov keys are repo-relative paths, e.g. ``src/progenax/builders.py`` or
    ``src/progenax/analytical/base.py``. The module name is the path RELATIVE to
    ``src/progenax/`` with the ``.py`` suffix dropped:

    - ``src/progenax/builders.py``        -> ``builders``
    - ``src/progenax/analytical/base.py`` -> ``analytical/base``
    - ``src/progenax/__init__.py``        -> ``__init__``

    Per-FILE granularity (the plan permits this). Top-level files collapse to a
    bare name (``builders``), which lines up with how ``collect_test_inventory``
    buckets top-level test files; nested files keep their package path so distinct
    modules never collapse together. Files outside ``src/progenax/`` (e.g. tests,
    conftest) return ``""`` and are dropped by the caller.
    """
    if not file_key.startswith(_SRC_PREFIX):
        return ""
    rel = file_key[len(_SRC_PREFIX):]
    if rel.endswith(".py"):
        rel = rel[: -len(".py")]
    return rel


def load_coverage(path: str) -> dict:
    """Parse a pytest-cov ``coverage.json`` into dashboard-friendly coverage.

    Reads the top-level ``totals.percent_covered`` and each ``files`` entry's
    ``summary.percent_covered``, mapping every ``src/progenax/<...>.py`` file to a
    per-file module name (see :func:`_file_key_to_module`). Non-``src/progenax``
    files are ignored.

    The committed ``validation/data/coverage.json`` (Task 1.5) is a pytest-cov
    JSON with an ADDED top-level ``coverage_provenance`` block
    ``{"selector": ..., "git_sha": ...}`` that the Phase-2 floor gate reads. A raw
    ``--cov`` run does NOT have it, so it is OPTIONAL: passed through when present,
    ``None`` when absent.

    Returns::

        {
            "total_percent": float,            # totals.percent_covered
            "per_module": {module: float},     # src/progenax/<...> only
            "coverage_provenance": dict | None,
        }
    """
    data = json.loads(Path(path).read_text())
    per_module: dict[str, float] = {}
    for file_key, file_data in data.get("files", {}).items():
        module = _file_key_to_module(file_key)
        if not module:
            continue
        per_module[module] = float(file_data["summary"]["percent_covered"])
    return {
        "total_percent": float(data["totals"]["percent_covered"]),
        "per_module": per_module,
        "coverage_provenance": data.get("coverage_provenance"),
    }


# grad-audit row statuses are COMPUTED by grad_audit/core.py::_classify into exactly
# three values (see its docstring + AuditResult.status comment): "clean" and
# "known-limitation" are benign/expected (FD-consistent, or a deliberately annotated
# known_blocked/known_zero edge); "hazard" is the non-benign value — a silently zeroed
# or FD-inconsistent gradient. The HAZARD RULE here mirrors that: a row is a hazard iff
# its status is NOT one of the benign values. (Equivalently: status == "hazard", but we
# express it as "not benign" so a NEW non-benign status string can never slip through as
# safe.) The committed JSON should carry zero hazards in a healthy tree.
_GRAD_AUDIT_BENIGN_STATUSES = frozenset({"clean", "known-limitation"})

# The three registries that Phases 2/4/5 will build. Until then read_registry_status
# returns a not-built placeholder for each (the grad-audit block is the only real one).
_NOT_BUILT_REGISTRIES = ("api_coverage", "physics_validation", "provenance")

_GRAD_AUDIT_JSON = "validation/data/grad_audit_results.json"


def read_registry_status() -> dict:
    """Summarize the validation registries for the dashboard (introspection-only).

    Imports the frozen-literal ``tests.validation.grad_audit.manifest`` (a pure
    module with NO pytest-collection side effects — safe to import here; it must
    NEVER run the suite) and parses the COMMITTED grad-audit results JSON. It does
    NOT run the audit, pytest, or any registry test.

    The grad-audit block reports, from the manifest literals: the AUDITED symbol
    count and the per-category EXEMPT_* histogram (from ``SYMBOL_CATEGORY``), the
    ``MUST_AUDIT`` size, and from the committed JSON the row count, the per-status
    histogram, and a derived ``hazards`` count (rows whose status is NOT a benign
    value — see ``_GRAD_AUDIT_BENIGN_STATUSES``).

    The other three registries do not exist yet -> ``{"status": "not-built"}``
    placeholders (Phases 2/4/5 replace them).

    Returns ``{"differentiability": {...}, "api_coverage": {...}, ...}``.
    """
    from collections import Counter

    from tests.validation.grad_audit.manifest import MUST_AUDIT, SYMBOL_CATEGORY

    category_hist = Counter(SYMBOL_CATEGORY.values())
    audited = category_hist.pop("AUDITED", 0)
    exempt = {cat: n for cat, n in sorted(category_hist.items())}

    rows = json.loads((_REPO_ROOT / _GRAD_AUDIT_JSON).read_text())
    status_hist = dict(Counter(r["status"] for r in rows))
    hazards = sum(
        n for status, n in status_hist.items()
        if status not in _GRAD_AUDIT_BENIGN_STATUSES
    )

    differentiability = {
        "status": "built",
        "audited": audited,
        "exempt": exempt,
        "must_audit": len(MUST_AUDIT),
        "json_rows": len(rows),
        "hazards": hazards,
        "status_histogram": status_hist,
    }
    result = {"differentiability": differentiability}
    for reg in _NOT_BUILT_REGISTRIES:
        result[reg] = {"status": "not-built"}
    return result


def read_durations(path: str = "validation/data/durations.json") -> dict:
    """Read the COMMITTED per-module slowest-test durations artifact (no run).

    The generator NEVER runs ``pytest --durations`` itself — that is a slow,
    manual, FULL-suite step (deferred to Phase 3 re-profiling). This reader only
    parses a committed artifact: a JSON ``{"modules": {module: {"slowest_test":
    str, "seconds": float}}}`` map. If the file is ABSENT it returns
    ``{"status": "not-measured"}`` so the dashboard can render an honest
    "not measured yet" cell without triggering any measurement.

    Returns the ``modules`` mapping (``{module: {slowest_test, seconds}}``) when
    present, else ``{"status": "not-measured"}``.
    """
    p = _REPO_ROOT / path
    if not p.exists():
        return {"status": "not-measured"}
    return json.loads(p.read_text())["modules"]


def read_validation_scripts(
    path: str = "validation/data/validation_runs.json",
) -> dict:
    """Map every ``scripts/validate_*.py`` to its last recorded exit code (no run).

    Globs the 23 ``scripts/validate_*.py`` files and, for each, reports the exit
    code recorded in the COMMITTED ``{script_name: exit_code}`` artifact at
    ``path`` (else ``"unknown"``). The generator does NOT run the scripts — they
    are too slow; a separate ``--run-validations`` flag (Task 1.5 / future)
    refreshes the artifact. Enumerating from the glob (not from the artifact keys)
    means a newly-added validate script shows up immediately as ``"unknown"``.

    Returns ``{script_name: exit_code_or_"unknown"}`` for all 23 scripts.
    """
    p = _REPO_ROOT / path
    recorded = json.loads(p.read_text()) if p.exists() else {}
    scripts = sorted((_REPO_ROOT / "scripts").glob("validate_*.py"))
    return {s.name: recorded.get(s.name, "unknown") for s in scripts}


if __name__ == "__main__":
    # Minimal entrypoint for Task 1.2 (the --emit / --render CLI lands in Task 1.5).
    inv = collect_test_inventory()
    total = sum(t for m in inv.values() for t in m.values())
    print(f"collected {total} tests across {len(inv)} modules")
