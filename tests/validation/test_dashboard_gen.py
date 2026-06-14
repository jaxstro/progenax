"""Tests for the test-dashboard generator (scripts/build_test_dashboard.py).

Task 1.2: the inventory collector reads per-module test counts across the three
tiers (unit / integration / validation) by parsing ``pytest --collect-only``.
"""
from scripts.build_test_dashboard import collect_test_inventory


def test_inventory_has_modules_and_counts():
    inv = collect_test_inventory()  # {module: {"unit": n, "integration": n, "validation": n}}
    assert "builders" in inv
    assert inv["builders"]["unit"] > 0
    assert sum(t for m in inv.values() for t in m.values()) > 1000
