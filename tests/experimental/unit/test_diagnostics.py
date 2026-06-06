import numpy as np, pytest
from gravoturb_fdf.inference.diagnostics import compute_hmc_diagnostics

pytestmark = pytest.mark.experimental


def test_diagnostics_on_iid_gaussian_chains():
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((4, 1000, 3))          # 4 well-mixed chains
    diag = compute_hmc_diagnostics(
        positions, divergences=np.zeros((4, 1000), bool),
        energy=rng.standard_normal((4, 1000)),
        tree_depth=np.full((4, 1000), 5), max_tree_depth=10,
        param_names=["M", "alpha", "beta"])
    assert np.all(diag["r_hat"] < 1.01)                    # iid -> R-hat ~ 1
    assert np.all(diag["ess_bulk"] > 400)
    assert np.all(diag["ess_tail"] > 400)
    assert diag["divergence_rate"] == 0.0
    assert 0.3 < diag["bfmi"] < 3.0
    assert diag["max_tree_depth_saturation"] == 0.0        # depth 5 < 10
    assert diag["passed"] is True


def test_diagnostics_flags_stuck_chains():
    positions = np.concatenate([np.zeros((2, 500, 1)), np.ones((2, 500, 1))], axis=0)
    diag = compute_hmc_diagnostics(positions, param_names=["x"])
    assert np.any(diag["r_hat"] > 1.1)                     # disagreeing chains
    assert diag["passed"] is False
