import subprocess
import sys


def test_missing_scipy_diagnostic_modules_give_actionable_error():
    """Catch eager diagnostic imports and loss of the actionable missing-module error.

    JAX 0.11 requires scipy>=1.15 and uses scipy.linalg for LAPACK setup.
    Blocking the entire SciPy package tests a broken JAX installation. Block
    only the graph/distance modules required by the exact diagnostics, then
    verify import, JAX-native Q, and the exact diagnostic's error boundary.
    """
    code = (
        "import sys;"
        "sys.modules['scipy.sparse.csgraph'] = None;"
        "sys.modules['scipy.spatial.distance'] = None;"
        "import progenax.diagnostics as d;"  # must import fine
        "from progenax.diagnostics import q_approx;"  # JAX-native: must work
        "import jax.numpy as jnp, jax;"
        "q_approx(jax.random.normal(jax.random.PRNGKey(0), (50, 3)));"
        "exc = None\n"
        "try:\n"
        "    d.compute_q_parameter(jax.random.normal(jax.random.PRNGKey(0), (50, 3)))\n"
        "except ImportError as e:\n"
        "    exc = e\n"
        "assert exc is not None and 'diagnostics' in str(exc), exc\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
