"""Tests for the test-dashboard generator (scripts/build_test_dashboard.py).

Task 1.2: the inventory collector reads per-module test counts across the three
tiers (unit / integration / validation) by parsing ``pytest --collect-only``.

Task 1.4: introspection-only readers for registry status, per-module durations,
and validation-script exit codes. NONE of these run the suite, ``pytest
--durations``, or the ``validate_*.py`` scripts — they only import frozen-literal
modules and parse COMMITTED artifacts (so the Task-1.6 staleness gate stays cheap).
"""
from pathlib import Path

import pytest

from scripts.build_test_dashboard import (
    build_dashboard,
    collect_test_inventory,
    load_coverage,
    read_durations,
    read_registry_status,
    read_validation_scripts,
    rollup_coverage_by_dir,
    write_coverage_json,
)

_FIXTURE = str(
    Path(__file__).parent / "fixtures" / "coverage_sample.json"
)
_DURATIONS_FIXTURE = str(
    Path(__file__).parent / "fixtures" / "durations_sample.json"
)
_DURATIONS_NO_MODULES_FIXTURE = str(
    Path(__file__).parent / "fixtures" / "durations_no_modules.json"
)
_VALRUNS_FIXTURE = str(
    Path(__file__).parent / "fixtures" / "validation_runs_sample.json"
)
_ROLLUP_FIXTURE = str(
    Path(__file__).parent / "fixtures" / "coverage_rollup_sample.json"
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
    # per-file line counts are retained for the directory rollup (M1).
    assert cov["per_file_lines"]["builders"] == {
        "covered_lines": 90,
        "num_statements": 100,
    }
    assert cov["per_file_lines"]["analytical/base"] == {
        "covered_lines": 30,
        "num_statements": 40,
    }


def test_rollup_coverage_by_dir_is_weighted_by_statements():
    """The per-module join key in ``build_dashboard`` is the census DIRECTORY
    (``profiles``, ``imf``), but ``load_coverage`` keys per FILE
    (``profiles/plummer``). The rollup folds per-file line counts up to the
    top-level directory as a STATEMENT-WEIGHTED percentage (M1), so the join key
    lines up with the census key.

    Fixture: profiles/plummer.py 80/100, profiles/king.py 40/100 ->
    profiles = 100*(80+40)/(100+100) = 60.0 (weighted, NOT the 60.0 mean by luck:
    builders.py 30/50 = 60.0 too, a top-level single-file module that still works).
    """
    cov = load_coverage(_ROLLUP_FIXTURE)
    rolled = rollup_coverage_by_dir(cov)
    # Two files under one directory -> one weighted bucket under the census key.
    assert rolled["profiles"] == pytest.approx(60.0)  # 120/200
    # A top-level single-file module keeps working (keyed by its bare stem).
    assert rolled["builders"] == pytest.approx(60.0)  # 30/50
    # Non-src files (tests/conftest) never enter the rollup.
    assert "tests" not in rolled
    assert "conftest" not in rolled


# --- Task 1.4: read_registry_status -----------------------------------------

def test_registry_status_all_four_built():
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
    # Every BUILT registry block must emit a `full: bool` flag (I2 contract);
    # grad-audit is full iff it is hazard-free (0 hazards -> True).
    assert ga["full"] is True
    # The histogram only carries the grad-audit status vocabulary.
    assert set(ga["status_histogram"]) <= {"clean", "known-limitation", "hazard"}
    # audited count is consistent with the AUDITED bucket of SYMBOL_CATEGORY.
    # 122 = 114 base + 5 ZAMS stellar relations (P3 ZAMS-migration) + 3 dispersion
    # forward models (Phase 0: jeans_dispersion, project_dispersion; Phase 0.5:
    # df_moment_dispersion; all AUDITED).
    assert sum(ga["exempt"].values()) + ga["audited"] == 122

    # api-coverage is BUILT and FULL (Task 2.3 closed all 6 UNTESTED holes): it
    # partitions __all__ into SYMBOL_TESTS / EXEMPT / UNTESTED, and UNTESTED is now empty.
    api = status["api_coverage"]
    assert api["status"] == "built"
    assert api["full"] is True  # all 6 holes closed (Task 2.3)
    assert api["untested"] == 0
    # The three partition sizes sum to the full __all__ (122 = 114 base + 5 ZAMS
    # + 3 dispersion forward models: jeans_dispersion, project_dispersion,
    # df_moment_dispersion).
    assert api["symbol_tests"] + api["exempt"] + api["untested"] == 122

    # physics-validation is BUILT and FULL (Task 4.2): the manifest partitions every
    # model into MODEL_INVARIANTS / EXEMPT_NON_MODEL / EXEMPT_NON_EQUILIBRIUM_MODEL /
    # UNTESTED_MODELS, and UNTESTED_MODELS is empty (0 holes as of Task 4.1).
    phys = status["physics_validation"]
    assert phys["status"] == "built"
    assert phys["full"] is True  # 0 untested holes
    assert phys["models"] > 0
    assert phys["untested"] == 0

    # provenance is BUILT and FULL (Task 5.2): the frozen-literal manifest ports every
    # cited constant from docs/provenance-ledger.md and carries an UNPROVENANCED hole list
    # that the 2026-06 audit left empty (0 fabricated values). This is the LAST registry —
    # built+full here flips gate.registries_full to True.
    prov = status["provenance"]
    assert prov["status"] == "built"
    assert prov["full"] is True  # 0 unprovenanced holes
    assert prov["constants"] > 0
    assert prov["unprovenanced"] == 0
    assert prov["allowlist_modules"] > 0


def test_registries_full_flag_gates_on_built_and_full():
    """``gate["registries_full"]`` is the all-clear flag: every registry must be
    BUILT and ``full is True``. As of Task 5.2 ALL FOUR registries (differentiability
    + api-coverage + physics-validation + provenance) are built+full, so the gate's
    registry condition is now MET and the aggregate flag is True. This is the milestone:
    the last registry (provenance) became built+full, flipping registries_full.
    """
    dash = build_dashboard("2026-01-01T00:00:00Z")
    regs = dash["registries"]
    # All four registries are built+full -> the gate's registry condition is satisfied.
    assert regs["differentiability"]["full"] is True
    assert regs["api_coverage"]["full"] is True
    assert regs["physics_validation"]["full"] is True
    assert regs["provenance"]["full"] is True
    assert dash["gate"]["registries_full"] is True
    built_and_full = {
        name for name, b in regs.items()
        if b.get("status") == "built" and b.get("full") is True
    }
    assert built_and_full == {
        "differentiability", "api_coverage", "physics_validation", "provenance"
    }


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


def test_durations_missing_modules_key_raises_runtime_error():
    """A committed durations artifact lacking the top-level ``modules`` key is a
    malformed artifact, not an absent one — surface it loudly (M2), not as a bare
    KeyError from ``json.loads(...)["modules"]``.
    """
    with pytest.raises(RuntimeError, match="modules"):
        read_durations(path=_DURATIONS_NO_MODULES_FIXTURE)


# --- Task 1.4: read_validation_scripts --------------------------------------

def test_validation_scripts_enumerates_all_24_with_exit_codes():
    runs = read_validation_scripts(path=_VALRUNS_FIXTURE)
    # All 24 scripts/validate_*.py are enumerated (24 = 23 + validate_zams.py, ZAMS migration).
    assert len(runs) == 24
    assert all(name.startswith("validate_") and name.endswith(".py") for name in runs)
    # Recorded exit codes come through; unrecorded scripts are "unknown".
    assert runs["validate_plummer.py"] == 0
    assert runs["validate_eff.py"] == 1
    assert runs["validate_tidal.py"] == "unknown"


def test_validation_scripts_absent_artifact_all_unknown():
    runs = read_validation_scripts(path="validation/data/__nope_valruns__.json")
    assert len(runs) == 24
    assert all(code == "unknown" for code in runs.values())


# --- Task 1.5: write_coverage_json (provenance injection) -------------------

def test_write_coverage_json_injects_provenance(tmp_path):
    """The Phase-2 path: read a raw pytest-cov json, inject a top-level
    coverage_provenance block, write the merged json, and confirm it round-trips
    through load_coverage with provenance now populated.

    Task 2.2 extended the provenance block to also carry ``total_percent`` (lifted
    from ``totals.percent_covered`` so the floor gate reads ONE field) and a
    ``measured_utc`` stamp. When not passed, both default (total from the totals,
    measured_utc from the clock).
    """
    out = tmp_path / "coverage.json"
    write_coverage_json(
        raw_cov_path=_FIXTURE,
        out_path=str(out),
        selector="full",
        git_sha="deadbeef",
        measured_utc="2026-06-14T12:00:00+00:00",  # pin for a deterministic assert
    )
    cov = load_coverage(str(out))
    # round-trips: same totals + per-module mapping as the raw fixture.
    raw = load_coverage(_FIXTURE)
    assert cov["total_percent"] == raw["total_percent"]
    assert cov["per_module"] == raw["per_module"]
    # ...but now provenance is populated (the raw fixture has None).
    assert raw["coverage_provenance"] is None
    prov = cov["coverage_provenance"]
    assert prov["selector"] == "full"
    assert prov["git_sha"] == "deadbeef"
    assert prov["measured_utc"] == "2026-06-14T12:00:00+00:00"
    # total_percent in the stamp defaults to totals.percent_covered (the headline).
    assert prov["total_percent"] == raw["total_percent"]

    # And the defaulted measured_utc path: a real UTC isoformat stamp is injected.
    out2 = tmp_path / "coverage2.json"
    write_coverage_json(_FIXTURE, str(out2), selector="full", git_sha="cafef00d")
    prov2 = load_coverage(str(out2))["coverage_provenance"]
    assert prov2["measured_utc"].endswith("+00:00")  # UTC-stamped
    assert prov2["total_percent"] == raw["total_percent"]


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
    # api_coverage registry is BUILT and FULL as of Task 2.3 (all holes closed).
    assert dash["registries"]["api_coverage"]["status"] == "built"
    assert dash["registries"]["api_coverage"]["full"] is True
    # physics_validation registry is BUILT and FULL as of Task 4.2 (0 untested holes).
    assert dash["registries"]["physics_validation"]["status"] == "built"
    assert dash["registries"]["physics_validation"]["full"] is True
    # provenance registry is BUILT and FULL as of Task 5.2 (0 unprovenanced holes).
    assert dash["registries"]["provenance"]["status"] == "built"
    assert dash["registries"]["provenance"]["full"] is True
    # gate: ALL FOUR registries are built+full as of Task 5.2, so registries_full is
    # now True (the milestone). The line-coverage floor is 90.
    assert dash["gate"]["registries_full"] is True
    assert dash["gate"]["line_cov_floor"] == 90
    assert dash["gate"]["full_suite_green"] is None
