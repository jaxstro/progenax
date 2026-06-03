# tests/unit/test_public_api.py
"""Public-API surface guards (exports resolve; doc-advertised symbols exist)."""
import progenax


def test_energy_sorted_segregation_is_top_level_export():
    assert "energy_sorted_segregation" in progenax.__all__
    assert hasattr(progenax, "energy_sorted_segregation")


def test_doc_advertised_analytical_symbols_exist():
    """Guard against phantom doc refs (audit found harmonic_oscillator_1d): every analytical
    name advertised in the docs must resolve on the package."""
    import progenax
    advertised = [
        "two_body_kepler", "three_body_figure_eight", "earth_sun_2body",
        "solar_system_inner_4", "solar_system_full", "harmonic_oscillator",
    ]
    missing = [n for n in advertised if not hasattr(progenax, n)]
    assert not missing, f"doc-advertised symbols missing from progenax: {missing}"
