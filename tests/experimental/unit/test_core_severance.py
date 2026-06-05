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
