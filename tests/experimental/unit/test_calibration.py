"""f_sub → Q calibration driver (spec §3.8 / §8 steps 4-6; AC7 headline).

For each substructure fraction f_sub, realize FDF fields, sample N⋆ stars (N_tail from
p_tail∝wρ, rest from ρ), and measure the CW04 Q with the AC5-validated estimator. The
driver returns Q ensembles per f_sub; the headline is the monotone-decreasing Q(f_sub)
with realization scatter (a forward calibration — not an inversion of Q to FBM params).
"""

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.experimental

PARAMS = dict(mach=8.0, b=0.5, alpha=1.8, beta=3.5)


def test_measure_q_ensemble_shape_finite():
    """measure_q_ensemble returns n_real finite Q values in a physical range."""
    from gravoturb_fdf.validation.calibration import measure_q_ensemble

    q = measure_q_ensemble(
        **PARAMS,
        f_sub=0.4,
        n_stars=400,
        n_real=5,
        shape=(48, 48, 48),
        key=jax.random.PRNGKey(0),
    )
    assert q.shape == (5,)
    assert np.all(np.isfinite(q))
    assert np.all((q > 0.1) & (q < 1.5))


def test_q_vs_fsub_monotone_decreasing():
    """Mean Q decreases with f_sub (more dense-tail stars → more substructure)."""
    from gravoturb_fdf.validation.calibration import q_vs_fsub

    res = q_vs_fsub(
        **PARAMS,
        f_sub_values=(0.1, 0.4, 0.7),
        n_stars=400,
        n_real=6,
        shape=(48, 48, 48),
        key=jax.random.PRNGKey(1),
    )
    qm = res["q_mean"]
    assert qm[0] > qm[1] > qm[2]  # strictly decreasing


def test_q_vs_fsub_physical_range_and_struct():
    """Result carries f_sub/q_mean/q_std arrays; means in CW04 substructured band."""
    from gravoturb_fdf.validation.calibration import q_vs_fsub

    res = q_vs_fsub(
        **PARAMS,
        f_sub_values=(0.0, 0.3, 0.6),
        n_stars=400,
        n_real=6,
        shape=(48, 48, 48),
        key=jax.random.PRNGKey(2),
    )
    assert res["f_sub"].shape == res["q_mean"].shape == res["q_std"].shape == (3,)
    assert np.all((res["q_mean"] > 0.4) & (res["q_mean"] < 0.8))
    assert np.all(res["q_std"] >= 0.0)
