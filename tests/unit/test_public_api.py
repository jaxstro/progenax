# tests/unit/test_public_api.py
"""Public-API surface guards (exports resolve; doc-advertised symbols exist)."""
import progenax


def test_energy_sorted_segregation_is_top_level_export():
    assert "energy_sorted_segregation" in progenax.__all__
    assert hasattr(progenax, "energy_sorted_segregation")
