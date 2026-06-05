"""P0 severance gates — released progenax core must not depend on the gravoturb-FDF
subsystem that is moving to experimental / being deleted.

These tests pin the 3 documented core->subsystem links being severed in P0:
  1. the `fractal_gw_legacy` top-level re-export (Task 0.2),
  2. the legacy-FDF / FractalLayer branch in generate_cluster_ic (Task 0.3),
  3. birth_environment reaching turbulence via fdf_config (Task 0.4),
plus the importer-free invariant (Task 0.5).
"""

import pytest

pytestmark = pytest.mark.experimental


# ── Task 0.2: link #1 — fractal_gw_legacy must not be re-exported from progenax ──
def test_no_fractal_gw_legacy_reexport():
    """The deprecated GW2004 fractal functions are not part of the released API."""
    import progenax

    for name in (
        "generate_fractal_positions",
        "rescale_fractal_to_target_radii",
        "assign_velocities_and_virialize",
    ):
        assert not hasattr(progenax, name), (
            f"progenax.{name} is still re-exported; the legacy GW2004 fractal is "
            "retired to experimental and must not be in the released top-level API."
        )
        assert name not in getattr(progenax, "__all__", []), (
            f"{name} is still in progenax.__all__"
        )


# ── Task 0.3: link #2 — the FractalLayer / fractal= branch is removed from core ──
def test_no_fractal_layer_in_released_api():
    """FractalLayer is retired from the released cluster API (Option A)."""
    import progenax.cluster as cluster

    assert not hasattr(cluster, "FractalLayer"), (
        "progenax.cluster.FractalLayer still exists; the D-stub fractal layer is "
        "retired to experimental gravoturb_fdf."
    )
    assert "FractalLayer" not in getattr(cluster, "__all__", [])


def test_spatial_structure_params_has_no_fractal_field():
    """SpatialStructureParams exposes only base_profile + mass_segregation."""
    import dataclasses

    from progenax.cluster import SpatialStructureParams

    field_names = {f.name for f in dataclasses.fields(SpatialStructureParams)}
    assert "fractal" not in field_names, (
        f"SpatialStructureParams still has a `fractal` field: {field_names}"
    )
    assert "mass_segregation" in field_names  # mass-seg stays in released core


# ── Task 0.4: link #3 — birth_environment imports turbulence directly, not fdf_config ──
def test_birth_environment_imports_turbulence_directly():
    """The released EnvironmentIMF reaches turbulence relations via the core
    turbulence module, not via the to-be-retired cluster.fdf_config re-exports."""
    import inspect

    from progenax.imf.environment import birth_environment

    src = inspect.getsource(birth_environment)
    assert "progenax.cluster.fdf_config" not in src, (
        "birth_environment still imports from cluster.fdf_config; it must import "
        "turbulence relations directly from progenax.cluster.turbulence."
    )
