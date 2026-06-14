"""Tests for the test-dashboard generator (scripts/build_test_dashboard.py).

Task 1.2: the inventory collector reads per-module test counts across the three
tiers (unit / integration / validation) by parsing ``pytest --collect-only``.

Task 1.4: introspection-only readers for registry status, per-module durations,
and validation-script exit codes. NONE of these run the suite, ``pytest
--durations``, or the ``validate_*.py`` scripts — they only import frozen-literal
modules and parse COMMITTED artifacts (so the Task-1.6 staleness gate stays cheap).
"""
from pathlib import Path

from scripts.build_test_dashboard import (
    build_dashboard,
    collect_test_inventory,
    load_coverage,
    read_durations,
    read_registry_status,
    read_validation_scripts,
    write_coverage_json,
)

_FIXTURE = str(
    Path(__file__).parent / "fixtures" / "coverage_sample.json"
)
_DURATIONS_FIXTURE = str(
    Path(__file__).parent / "fixtures" / "durations_sample.json"
)
_VALRUNS_FIXTURE = str(
    Path(__file__).parent / "fixtures" / "validation_runs_sample.json"
)


def test_inventory_has_modules_and_counts():
    inv = collect_test_inventory()  # {module: {"unit": n, "integration": n, "validation": n}}
    assert "builders" in inv
    assert inv["builders"]["unit"] > 0
    assert sum(t for m in inv.values() for t in m.values()) > 1000


def test_load_coverage_parses_totals_and_per_module():
    cov = load_coverage(_FIXTURE)
    assert 0 <= cov["total_percent"] <= 100
    assert cov["per_module"]            # non-empty mapping of module -> percent
    # only src/progenax files become modules; non-src files are ignored.
    assert "builders" in cov["per_module"]
    assert "analytical/base" in cov["per_module"]
    assert "tests/conftest" not in cov["per_module"]
    # coverage_provenance passes through when present (None when absent)
    assert "coverage_provenance" in cov


# --- Task 1.4: read_registry_status -----------------------------------------

def test_registry_status_grad_audit_built_others_not():
    status = read_registry_status()
    # The one real registry (grad-audit) is built from frozen literals + JSON.
    ga = status["differentiability"]
    assert ga["status"] == "built"
    assert ga["audited"] > 0
    assert ga["must_audit"] > 0
    assert ga["json_rows"] > 0
    # hazards is an int (0 in a clean state) derived from the status histogram.
    assert isinstance(ga["hazards"], int)
    assert ga["hazards"] == 0
    # The histogram only carries the grad-audit status vocabulary.
    assert set(ga["status_histogram"]) <= {"clean", "known-limitation", "hazard"}
    # audited count is consistent with the AUDITED bucket of SYMBOL_CATEGORY.
    assert sum(ga["exempt"].values()) + ga["audited"] == 114
    # The other three registries do not exist yet -> not-built placeholders.
    for reg in ("api_coverage", "physics_validation", "provenance"):
        assert status[reg] == {"status": "not-built"}


# --- Task 1.4: read_durations -----------------------------------------------

def test_durations_parses_committed_artifact():
    dur = read_durations(path=_DURATIONS_FIXTURE)
    # slowest-per-module schema: each module -> {slowest_test, seconds}
    assert dur["cluster"]["seconds"] == 42.7
    assert "test_multimass_equilibrium_physics" in dur["cluster"]["slowest_test"]
    assert dur["builders"]["seconds"] == 3.1


def test_durations_absent_returns_not_measured():
    dur = read_durations(path="validation/data/__nope_durations__.json")
    assert dur == {"status": "not-measured"}


# --- Task 1.4: read_validation_scripts --------------------------------------

def test_validation_scripts_enumerates_all_23_with_exit_codes():
    runs = read_validation_scripts(path=_VALRUNS_FIXTURE)
    # All 23 scripts/validate_*.py are enumerated.
    assert len(runs) == 23
    assert all(name.startswith("validate_") and name.endswith(".py") for name in runs)
    # Recorded exit codes come through; unrecorded scripts are "unknown".
    assert runs["validate_plummer.py"] == 0
    assert runs["validate_eff.py"] == 1
    assert runs["validate_tidal.py"] == "unknown"


def test_validation_scripts_absent_artifact_all_unknown():
    runs = read_validation_scripts(path="validation/data/__nope_valruns__.json")
    assert len(runs) == 23
    assert all(code == "unknown" for code in runs.values())


# --- Task 1.5: write_coverage_json (provenance injection) -------------------

def test_write_coverage_json_injects_provenance(tmp_path):
    """The Phase-2 path: read a raw pytest-cov json, inject a top-level
    coverage_provenance block, write the merged json, and confirm it round-trips
    through load_coverage with provenance now populated.
    """
    out = tmp_path / "coverage.json"
    write_coverage_json(
        raw_cov_path=_FIXTURE,
        out_path=str(out),
        selector="tests/unit tests/integration tests/validation",
        git_sha="deadbeef",
    )
    cov = load_coverage(str(out))
    # round-trips: same totals + per-module mapping as the raw fixture.
    raw = load_coverage(_FIXTURE)
    assert cov["total_percent"] == raw["total_percent"]
    assert cov["per_module"] == raw["per_module"]
    # ...but now provenance is populated (the raw fixture has None).
    assert raw["coverage_provenance"] is None
    assert cov["coverage_provenance"] == {
        "selector": "tests/unit tests/integration tests/validation",
        "git_sha": "deadbeef",
    }


# --- Task 1.5: build_dashboard ----------------------------------------------

def test_build_dashboard_has_all_blocks():
    dash = build_dashboard("2026-01-01T00:00:00Z")
    # generated_utc is stamped EXACTLY as passed (staleness gate ignores it).
    assert dash["generated_utc"] == "2026-01-01T00:00:00Z"
    for key in ("modules", "registries", "line_coverage", "durations",
                "validation_scripts", "gate"):
        assert key in dash, f"missing top-level block: {key}"


def test_build_dashboard_modules_merge_inventory():
    dash = build_dashboard("2026-01-01T00:00:00Z")
    builders = dash["modules"]["builders"]
    assert builders["unit"] > 0
    for tier in ("unit", "integration", "validation"):
        assert tier in builders
    # line_cov is present (float when coverage measured, else None).
    assert "line_cov" in builders


def test_build_dashboard_registry_and_gate_state():
    dash = build_dashboard("2026-01-01T00:00:00Z")
    # api_coverage registry is not built yet in Phase 1.
    assert dash["registries"]["api_coverage"]["status"] == "not-built"
    # gate: registries are NOT all full (3 of 4 not built) and the floor is 90.
    assert dash["gate"]["registries_full"] is False
    assert dash["gate"]["line_cov_floor"] == 90
    assert dash["gate"]["full_suite_green"] is None
