"""FDF stub-calibration warning + version surfacing (audit M9).

`load_fdf_calibration()` returns an uncalibrated v1 stub (chi~D identity), so
FDF-generated cluster statistics (Q_CW, sigma_Sigma/Sigma) are not yet calibrated
to Goodwin-Whitworth. M9: surface the calibration `version` and emit a warning so
stub output is not mistaken for calibrated results — including via the top-level
`generate_cluster_ic` fractal path.
"""

import pytest

import progenax  # noqa: F401  (enables float64 at import)
from progenax.cluster.fdf_calibration import load_fdf_calibration


def test_calibration_version_is_surfaced():
    """The calibration object exposes its version string."""
    cal = load_fdf_calibration()
    assert cal.version == "v1_stub_uncalibrated"


def test_load_fdf_calibration_warns_naming_version():
    """Loading the stub warns and names the version."""
    with pytest.warns(UserWarning, match=r"v1_stub_uncalibrated"):
        load_fdf_calibration()
