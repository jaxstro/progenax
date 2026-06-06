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


def test_diagnostics_saturation_and_low_bfmi_each_fail():
    # Lock the two AC19-critical gates that the iid test does NOT exercise (both pass there):
    # tree-depth saturation FIRING, and a low E-BFMI. Positions are well-mixed so R-hat/ESS
    # pass and each gate is isolated as the sole cause of passed=False.
    rng = np.random.default_rng(3)
    positions = rng.standard_normal((4, 1000, 2))

    # (a) every step at the doubling ceiling -> saturation 1.0 -> fail
    sat = compute_hmc_diagnostics(
        positions, tree_depth=np.full((4, 1000), 10), max_tree_depth=10,
        param_names=["M", "alpha"])
    assert sat["max_tree_depth_saturation"] == 1.0
    assert sat["passed"] is False

    # (b) random-walk energy (slow diffusion vs marginal variance) -> low E-BFMI -> fail
    walk_energy = np.cumsum(rng.standard_normal((4, 1000)), axis=1)
    low = compute_hmc_diagnostics(positions, energy=walk_energy, param_names=["M", "alpha"])
    assert low["bfmi"] < 0.3
    assert low["passed"] is False
