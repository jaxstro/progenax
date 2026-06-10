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


def test_no_spatial_structure_params_in_released_api():
    """The string-dispatch SpatialStructureParams (which once carried the fractal
    field) was retired wholesale in the 2026-06 unified redesign — the strongest
    form of the Task-0.3 severance. Mass segregation stays in released core via
    energy_sorted_segregation + MultiComponentCluster.from_mass_segregation."""
    import progenax.cluster as cluster

    assert not hasattr(cluster, "SpatialStructureParams")
    assert hasattr(cluster, "energy_sorted_segregation")
    assert hasattr(cluster, "MultiComponentCluster")


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


# ── Task 0.5: importer-free invariant — no released-core module imports the subsystem ──
# The gravoturb-FDF subsystem modules still EXIST (deleted in P5), but released
# progenax must not import them, so P5 deletion is a pure file removal.
_SUBSYSTEM_FILES = {
    "fdf.py", "fdf_tail.py", "gravoturbulent.py", "fdf_config.py",
    "fdf_calibration.py", "fdf_hyperparams.py", "fractal_gw_legacy.py",
}
_SUBSYSTEM_TOKENS = {
    "fdf", "fdf_density", "fdf_tail", "gravoturbulent", "fdf_config",
    "fdf_calibration", "fdf_hyperparams", "fractal_gw_legacy", "gravoturb",
}


def _is_subsystem_module(modname):
    if not modname:
        return False
    parts = modname.replace("progenax.", "").split(".")
    return any(p in _SUBSYSTEM_TOKENS for p in parts)


def _in_subsystem_file(path):
    s = str(path)
    return ("/gravoturb/" in s or "/fdf_density/" in s
            or path.name in _SUBSYSTEM_FILES)


def test_no_core_module_imports_subsystem_at_module_level():
    """AST-scan released progenax: no core module imports a subsystem module
    at module level (the prerequisite for safe wholesale deletion in P5)."""
    import ast
    import pathlib

    import progenax

    root = pathlib.Path(progenax.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        if _in_subsystem_file(path):
            continue  # subsystem files may import each other
        tree = ast.parse(path.read_text())
        for node in tree.body:  # module level only
            if isinstance(node, ast.ImportFrom) and _is_subsystem_module(node.module):
                offenders.append(f"{path.name}:{node.lineno} from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_subsystem_module(alias.name):
                        offenders.append(f"{path.name}:{node.lineno} import {alias.name}")

    assert not offenders, (
        "Released-core modules still import the gravoturb-FDF subsystem at module "
        f"level (must be importer-free before P5 deletion):\n" + "\n".join(offenders)
    )


def test_importing_progenax_pulls_in_no_subsystem_module():
    """Runtime guard: `import progenax` must not load any condemned subsystem module
    (the strongest evidence P5 deletion is a pure file removal)."""
    import sys

    import progenax  # noqa: F401

    leaked = [
        m for m in sys.modules
        if _is_subsystem_module(m) and "gravoturb_fdf" not in m
    ]
    assert not leaked, f"import progenax leaked subsystem modules: {leaked}"
