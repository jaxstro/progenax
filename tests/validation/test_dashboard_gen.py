"""Tests for the test-dashboard generator (scripts/build_test_dashboard.py).

Task 1.2: the inventory collector reads per-module test counts across the three
tiers (unit / integration / validation) by parsing ``pytest --collect-only``.
"""
from pathlib import Path

from scripts.build_test_dashboard import collect_test_inventory, load_coverage

_FIXTURE = str(
    Path(__file__).parent / "fixtures" / "coverage_sample.json"
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
