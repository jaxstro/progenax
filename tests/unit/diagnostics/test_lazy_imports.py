import subprocess
import sys


def test_diagnostics_import_without_scipy_gives_actionable_error():
    """Audit R9: in an env without scipy, importing the subpackage must NOT
    crash at import time; the JAX-native q_approx must still work; and calling
    an exact (scipy-backed) function must raise an ImportError naming the
    [diagnostics] extra."""
    code = (
        "import sys;"
        "sys.modules['scipy'] = None; sys.modules['scipy.sparse'] = None;"
        "sys.modules['scipy.sparse.csgraph'] = None; sys.modules['scipy.spatial'] = None;"
        "sys.modules['scipy.spatial.distance'] = None;"
        "import progenax.diagnostics as d;"           # must import fine
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
