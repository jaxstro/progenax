"""Build the progenax test/validation dashboard.

This is the generator behind the website's ``test-dashboard`` page and the
staleness gate (Task 1.6). It is a plain SCRIPT, not core library code: it shells
out to ``pytest --collect-only`` and parses node ids, which is acceptable here
(it is NOT bound by the JAX-native constraint that governs ``src/progenax``).

Task 1.2 implements ``collect_test_inventory()`` — the per-module test census
across the three tiers (unit / integration / validation). Tasks 1.3-1.4 add line
coverage, registry status, durations, and validation-script readers. Task 1.5
assembles them into a timestamped dashboard dict (:func:`build_dashboard`),
stamps + writes the committed JSON (``--emit``), injects coverage provenance
(:func:`write_coverage_json`, the Phase-2 ``coverage.json`` stamp path), and
renders the MyST matrix page (``--render``, delegated to
:mod:`scripts._dashboard_render` to keep this file under the 500-LOC cap).

Direct invocation (``python scripts/build_test_dashboard.py``) puts only
``scripts/`` on ``sys.path``; ``pyproject.toml`` sets
``pythonpath=["src","src/experimental"]`` which does NOT include the repo root,
so ``import tests.*`` / ``import scripts.*`` would ImportError. We mirror the
bootstrap in ``scripts/audit_gradients.py`` and insert the repo root first.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The committed line-coverage artifact (FULL-suite, stamped with provenance) and
# the committed timestamped dashboard JSON, plus the rendered MyST page. All three
# are relative to the repo root; the dashboard's ``line_coverage`` block is derived
# from ``coverage.json`` IF it exists (else a not-measured placeholder).
_COVERAGE_JSON = "validation/data/coverage.json"
_DASHBOARD_JSON = "validation/data/test_dashboard.json"
_DASHBOARD_PAGE = "docs/website/50-validation/test-dashboard.md"

# Phase-2 line-coverage floor (ratchet-up-only); echoed into the gate block.
_LINE_COV_FLOOR = 90

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

    ``per_file_lines`` retains the raw ``covered_lines`` + ``num_statements`` for
    each file (keyed the same as ``per_module``) so :func:`rollup_coverage_by_dir`
    can fold them up to a STATEMENT-WEIGHTED per-directory percentage (M1) — the
    census buckets by directory, not by file.

    Returns::

        {
            "total_percent": float,            # totals.percent_covered
            "per_module": {module: float},     # per-FILE percent, src/progenax/<...>
            "per_file_lines": {module: {"covered_lines": int, "num_statements": int}},
            "coverage_provenance": dict | None,
        }
    """
    data = json.loads(Path(path).read_text())
    per_module: dict[str, float] = {}
    per_file_lines: dict[str, dict[str, int]] = {}
    for file_key, file_data in data.get("files", {}).items():
        module = _file_key_to_module(file_key)
        if not module:
            continue
        summary = file_data["summary"]
        per_module[module] = float(summary["percent_covered"])
        per_file_lines[module] = {
            "covered_lines": int(summary["covered_lines"]),
            "num_statements": int(summary["num_statements"]),
        }
    return {
        "total_percent": float(data["totals"]["percent_covered"]),
        "per_module": per_module,
        "per_file_lines": per_file_lines,
        "coverage_provenance": data.get("coverage_provenance"),
    }


def rollup_coverage_by_dir(coverage: dict) -> dict[str, float]:
    """Fold per-file line counts up to a statement-weighted per-DIRECTORY percent.

    ``load_coverage`` keys coverage per FILE (``profiles/plummer``,
    ``analytical/base``), but :func:`collect_test_inventory` buckets the census by
    the top-level DIRECTORY (``profiles``, ``analytical``) — or, for a source file
    sitting directly under ``src/progenax/``, by the bare file stem (``builders``),
    which matches how a top-level test file is bucketed. So the per-module join in
    :func:`build_dashboard` needs coverage keyed by that SAME top-level component.

    For each per-file entry we take its first path component as the rollup key and
    sum ``covered_lines`` / ``num_statements`` into that bucket, then report
    ``100 * sum(covered) / sum(statements)`` per bucket. Buckets with zero
    statements are dropped (avoids a 0/0). Non-``src/progenax`` files never enter
    ``per_file_lines``, so they cannot appear here.

    Example: ``profiles/plummer`` 80/100 + ``profiles/king`` 40/100 ->
    ``profiles`` = 100*(120/200) = 60.0; ``builders`` 30/50 -> ``builders`` = 60.0.
    """
    buckets: dict[str, dict[str, int]] = {}
    for module, lines in coverage.get("per_file_lines", {}).items():
        top = module.split("/", 1)[0]  # first path component = census key
        acc = buckets.setdefault(top, {"covered_lines": 0, "num_statements": 0})
        acc["covered_lines"] += lines["covered_lines"]
        acc["num_statements"] += lines["num_statements"]
    return {
        top: 100.0 * acc["covered_lines"] / acc["num_statements"]
        for top, acc in buckets.items()
        if acc["num_statements"] > 0
    }


def write_coverage_json(
    raw_cov_path: str,
    out_path: str,
    selector: str,
    git_sha: str,
    total_percent: float | None = None,
    measured_utc: str | None = None,
) -> None:
    """Stamp a raw pytest-cov JSON with provenance and write it to ``out_path``.

    This is the EXACT path Phase-2 Task 2.2 uses to produce the committed
    ``validation/data/coverage.json``: it reads a raw ``--cov-report=json`` file,
    injects a top-level ``coverage_provenance`` block::

        {"selector": <full-suite selector>, "git_sha": <HEAD sha>,
         "total_percent": <totals.percent_covered>, "measured_utc": <UTC stamp>}

    and writes the merged JSON. The fields:

    - ``selector`` records which suite produced the numbers (the floor gate refuses
      a partial ``-m "not slow"`` run, which understates coverage).
    - ``git_sha`` is the commit the coverage was MEASURED at; the staleness gate uses
      it for a SRC-based freshness check (``git diff <git_sha> HEAD -- src/progenax``)
      without re-running the ~14-min ``--cov``.
    - ``total_percent`` is the headline ``totals.percent_covered`` lifted to the
      provenance block so the floor gate reads ONE field (it does not have to know
      pytest-cov's nested ``totals`` schema); defaults to ``totals.percent_covered``.
    - ``measured_utc`` is when the measurement was taken (human-facing provenance);
      defaults to ``datetime.now(timezone.utc)``. ``datetime.now`` is fine here — this
      is a plain SCRIPT, not a workflow.

    The resulting file round-trips through :func:`load_coverage` with
    ``coverage_provenance`` populated (it is ``None`` in a raw, unstamped file).
    """
    data = json.loads(Path(raw_cov_path).read_text())
    if total_percent is None:
        total_percent = float(data["totals"]["percent_covered"])
    if measured_utc is None:
        measured_utc = datetime.now(timezone.utc).isoformat()
    data["coverage_provenance"] = {
        "selector": selector,
        "git_sha": git_sha,
        "total_percent": total_percent,
        "measured_utc": measured_utc,
    }
    Path(out_path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command at the repo root and return the CompletedProcess.

    A thin wrapper so the coverage-provenance stamping (``--stamp-coverage``) can
    read the current HEAD sha, mirroring the ``--collect-only`` subprocess pattern.
    NOT ``check=True`` — callers inspect ``returncode`` (the freshness gate needs to
    distinguish a non-ancestor sha from a clean diff).
    """
    return subprocess.run(
        ["git", *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )


def _stamp_coverage(raw_cov_path: str) -> None:
    """Stamp ``raw_cov_path`` as the committed FULL-suite ``coverage.json``.

    Records ``selector="full"`` (the value the floor gate requires) and the current
    HEAD sha, lifting ``total_percent`` + ``measured_utc`` from the raw file / clock.
    """
    git_sha = _git("rev-parse", "HEAD").stdout.strip()
    out_path = _REPO_ROOT / _COVERAGE_JSON
    write_coverage_json(raw_cov_path, str(out_path), selector="full", git_sha=git_sha)
    total = json.loads(out_path.read_text())["coverage_provenance"]["total_percent"]
    print(f"wrote {out_path.relative_to(_REPO_ROOT)} (selector=full, "
          f"sha={git_sha[:8]}, total={total:.2f}%)")


# grad-audit row statuses are COMPUTED by grad_audit/core.py::_classify into exactly
# three values (see its docstring + AuditResult.status comment): "clean" and
# "known-limitation" are benign/expected (FD-consistent, or a deliberately annotated
# known_blocked/known_zero edge); "hazard" is the non-benign value — a silently zeroed
# or FD-inconsistent gradient. The HAZARD RULE here mirrors that: a row is a hazard iff
# its status is NOT one of the benign values. (Equivalently: status == "hazard", but we
# express it as "not benign" so a NEW non-benign status string can never slip through as
# safe.) The committed JSON should carry zero hazards in a healthy tree.
_GRAD_AUDIT_BENIGN_STATUSES = frozenset({"clean", "known-limitation"})

# The registries that Phase 5 will build. api_coverage (Task 2.2) and
# physics_validation (Task 4.2) are now built; provenance stays a not-built
# placeholder until Phase 5.
_NOT_BUILT_REGISTRIES = ("provenance",)

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

    The remaining registry (provenance) does not exist yet ->
    ``{"status": "not-built"}`` placeholder (Phase 5 replaces it).

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

    # CONTRACT: every BUILT registry block MUST emit `full: bool` — the per-registry
    # all-clear that `gate.registries_full` aggregates over (build_dashboard).
    # not-built placeholders OMIT `full`, so registries_full stays False until all 4
    # registries are present AND full. For grad-audit, "full" == hazard-free (0
    # hazards): MUST_AUDIT manifest-coverage is enforced SEPARATELY by
    # grad_audit/test_manifest_coverage.py, so it is not re-litigated here.
    differentiability = {
        "status": "built",
        "full": hazards == 0,
        "audited": audited,
        "exempt": exempt,
        "must_audit": len(MUST_AUDIT),
        "json_rows": len(rows),
        "hazards": hazards,
        "status_histogram": status_hist,
    }

    # api-coverage registry (Task 2.2): the frozen-literal manifest partitions every
    # progenax.__all__ symbol into SYMBOL_TESTS / EXEMPT / UNTESTED. We import the pure
    # manifest module (NO pytest-collection side effects — same contract as the grad-audit
    # manifest) and report the partition sizes. CONTRACT: `full` == zero UNTESTED holes
    # (every public symbol either has an asserting test or is justified EXEMPT). With holes
    # still open it is False, holding registries_full False until Task 2.3 closes them.
    from tests.validation.api_coverage.manifest import EXEMPT, SYMBOL_TESTS, UNTESTED

    api_coverage = {
        "status": "built",
        "full": len(UNTESTED) == 0,
        "symbol_tests": len(SYMBOL_TESTS),
        "exempt": len(EXEMPT),
        "untested": len(UNTESTED),
    }

    # physics-validation registry (Task 4.2): the frozen-literal manifest partitions
    # every model among MODEL_INVARIANTS (asserting-test-backed physics invariants),
    # EXEMPT_NON_MODEL (utilities/containers/distributions/helpers/analytical ICs),
    # EXEMPT_NON_EQUILIBRIUM_MODEL (reference-parity / uniform-density carves), and
    # UNTESTED_MODELS (real holes). We import the pure manifest module (NO pytest-collection
    # side effects — same contract as the grad-audit / api-coverage manifests) and report the
    # partition sizes. CONTRACT: `full` == zero UNTESTED holes (every operational model has
    # an asserting validation invariant or is a documented exempt carve); True at Task 4.1
    # (0 holes), so this registry no longer holds registries_full down — only the not-built
    # provenance registry (Phase 5) does.
    from tests.validation.physics_registry.manifest import (
        EXEMPT_NON_EQUILIBRIUM_MODEL,
        EXEMPT_NON_MODEL,
        MODEL_INVARIANTS,
        UNTESTED_MODELS,
    )

    physics_validation = {
        "status": "built",
        "full": len(UNTESTED_MODELS) == 0,
        "models": len(MODEL_INVARIANTS),
        "exempt_non_model": len(EXEMPT_NON_MODEL),
        "exempt_non_equilibrium": len(EXEMPT_NON_EQUILIBRIUM_MODEL),
        "untested": len(UNTESTED_MODELS),
    }

    result = {
        "differentiability": differentiability,
        "api_coverage": api_coverage,
        "physics_validation": physics_validation,
    }
    for reg in _NOT_BUILT_REGISTRIES:
        result[reg] = {"status": "not-built"}  # no `full` -> holds registries_full False
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
    data = json.loads(p.read_text())
    if "modules" not in data:
        # ABSENT (handled above) is honest "not measured"; a committed-but-malformed
        # artifact lacking "modules" is a broken artifact -> surface it loudly
        # (mirrors collect_test_inventory's empty-collect RuntimeError) rather than
        # a bare KeyError from data["modules"].
        raise RuntimeError(
            f"durations artifact {path} lacks the required top-level 'modules' key "
            f"(found keys: {sorted(data)}); the committed artifact is malformed."
        )
    return data["modules"]


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


def _line_coverage_block() -> dict:
    """The dashboard ``line_coverage`` block: parsed committed coverage or a stub.

    Derived from the COMMITTED ``validation/data/coverage.json`` IF it exists
    (written by a FULL-suite ``--cov`` run + :func:`write_coverage_json`), else
    ``{"status": "not-measured"}``. The full-suite ``--cov`` run is a documented
    manual step (~13-15 min, Phase-2 Task 2.1) — ``build_dashboard`` NEVER triggers
    it; it only reads what is already committed.

    The block ALWAYS carries a ``status`` key so downstream consumers (the
    ``line_cov_measured`` gate flag, the renderer, and the staleness gate's
    provenance teeth) can key off ONE field (C1): ``"measured"`` when the parsed
    coverage is present, ``"not-measured"`` when absent. Without this, ``measured``
    was never emitted and the gate's ``status == "measured"`` provenance branch was
    dead.
    """
    p = _REPO_ROOT / _COVERAGE_JSON
    if not p.exists():
        return {"status": "not-measured"}
    return {"status": "measured", **load_coverage(str(p))}


def build_dashboard(timestamp: str) -> dict:
    """Assemble the timestamped dashboard dict from the introspection readers.

    Unions the per-module test census (:func:`collect_test_inventory`) with
    per-module line coverage (``None`` where coverage is not measured), the
    registry status (:func:`read_registry_status`), the line-coverage block
    (:func:`_line_coverage_block`), durations (:func:`read_durations`), and
    validation-script exit codes (:func:`read_validation_scripts`), plus a release
    ``gate`` summary.

    ``timestamp`` is stamped verbatim into ``generated_utc`` (the CLI passes a real
    UTC time; the staleness gate IGNORES this field). Everything else is
    deterministic from committed artifacts + node-id collection, so the gate can
    regenerate and semantic-diff cheaply.
    """
    inventory = collect_test_inventory()
    registries = read_registry_status()
    line_coverage = _line_coverage_block()

    # Per-module line coverage, keyed to match the census buckets. The census
    # buckets by top-level DIRECTORY (`profiles`), so we roll the per-FILE coverage
    # up to a statement-weighted per-directory percentage (M1). Empty (-> None
    # everywhere) when coverage is not measured.
    per_module_cov = rollup_coverage_by_dir(line_coverage)
    modules: dict[str, dict] = {}
    for module, counts in sorted(inventory.items()):
        modules[module] = {
            "unit": counts["unit"],
            "integration": counts["integration"],
            "validation": counts["validation"],
            "line_cov": per_module_cov.get(module),
        }

    # line_cov_measured is the total percent when MEASURED, else None — keyed off the
    # single `status` field the block always carries (C1), so it stays consistent
    # with the renderer and the gate's provenance teeth.
    line_cov_measured = (
        line_coverage.get("total_percent")
        if line_coverage.get("status") == "measured"
        else None
    )
    registries_full = all(
        block.get("status") == "built" and block.get("full") is True
        for block in registries.values()
    )

    return {
        "generated_utc": timestamp,
        "modules": modules,
        "registries": registries,
        "line_coverage": line_coverage,
        "durations": read_durations(),
        "validation_scripts": read_validation_scripts(),
        "gate": {
            "registries_full": registries_full,
            "line_cov_floor": _LINE_COV_FLOOR,
            "line_cov_measured": line_cov_measured,
            "full_suite_green": None,
        },
    }


def _emit(out_path: Path) -> None:
    """Stamp the UTC time, build the dashboard, and write the committed JSON.

    ``datetime.now`` is fine here — this is a plain script, NOT a workflow. The
    JSON is pretty-printed with stable key order so diffs stay clean and the
    staleness gate is deterministic.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    dashboard = build_dashboard(timestamp)
    out_path.write_text(json.dumps(dashboard, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path.relative_to(_REPO_ROOT)} ({timestamp})")


def _render(json_path: Path, page_path: Path) -> None:
    """Render the committed dashboard JSON to the MyST matrix page."""
    from scripts._dashboard_render import render_dashboard_page

    dashboard = json.loads(json_path.read_text())
    page_path.write_text(render_dashboard_page(dashboard))
    print(f"wrote {page_path.relative_to(_REPO_ROOT)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit", action="store_true",
        help="build + write the timestamped dashboard JSON (validation/data/).",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="render the committed dashboard JSON to the MyST matrix page.",
    )
    parser.add_argument(
        "--stamp-coverage", metavar="RAW_COV_JSON", default=None,
        help="stamp a raw pytest-cov JSON (from a FULL-suite --cov run) with "
             "provenance (selector=full, HEAD sha, total_percent, measured_utc) "
             "and write it to validation/data/coverage.json.",
    )
    args = parser.parse_args(argv)

    if args.stamp_coverage:
        _stamp_coverage(args.stamp_coverage)
    if args.emit:
        _emit(_REPO_ROOT / _DASHBOARD_JSON)
    if args.render:
        _render(_REPO_ROOT / _DASHBOARD_JSON, _REPO_ROOT / _DASHBOARD_PAGE)
    if not (args.emit or args.render or args.stamp_coverage):
        inv = collect_test_inventory()
        total = sum(t for m in inv.values() for t in m.values())
        print(f"collected {total} tests across {len(inv)} modules")


if __name__ == "__main__":
    main()
