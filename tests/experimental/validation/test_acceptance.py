"""Assert the committed AC printing scripts report PASS (AC1-AC5, AC8, AC9).

The scripts in gravoturb_fdf.validation.acceptance print expected-vs-measured tables;
these tests assert their PASS verdicts so "validated" is backed by fresh output.
"""

import pytest

pytestmark = [pytest.mark.experimental, pytest.mark.validation]

from gravoturb_fdf.validation import acceptance  # noqa: E402


def test_ac1_ac2_bm19():
    assert acceptance.ac1_ac2_bm19()["passed"]


def test_ac3_ac4_zeta():
    assert acceptance.ac3_ac4_zeta()["passed"]


def test_ac5_q():
    assert acceptance.ac5_q()["passed"]


def test_ac6_cornerstone():
    # 64³×4 for test speed; the script main() runs the full 128³×8 ensemble.
    assert acceptance.ac6_cornerstone(shape=(64, 64, 64), n_real=4)["passed"]


def test_ac7_q_calibration():
    # 48³×4 smoke for test speed; the script main() runs the 64³ smoke.
    assert acceptance.ac7_q_calibration(shape=(48, 48, 48), n_real=4, n_stars=400)["passed"]


def test_ac8_ac9_grads():
    assert acceptance.ac8_ac9_grads()["passed"]


def test_ac11_xi_s_vs_oracle():
    # 48³×4 for test speed (rel_tol=0.03 covers the smaller-grid noise); the script
    # main() runs the full 64³×8 ensemble at rel_tol=0.02.
    res = acceptance.ac11_xi_s_vs_oracle(shape=(48, 48, 48), n_real=4, rel_tol=0.03)
    assert res["passed"]


def test_ac11b_rank_copula_equivalence():
    # 48³×4 for test speed; physical rank/mass-conserving copula xi_s vs prediction.
    res = acceptance.ac11b_rank_copula_equivalence(shape=(48, 48, 48), n_real=4, rel_tol=0.03)
    assert res["passed"]


def test_ac12_limber_projection_vs_oracle():
    # Asserts max ABSOLUTE error max|w_pred-w_meas| (the robust metric; relative error divides
    # the flat ~0.007 residual by w->w_floor at the outer bin and spuriously explodes). Uses
    # the SAME deterministic config as main() (48³×48, seed=0 -> max|dw|~0.008 < abs_tol 0.03)
    # since the residual is cosmic-variance-noisy across seeds even at n_real=48. ~15s.
    res = acceptance.ac12_limber_projection_vs_oracle()
    assert res["passed"]


def test_ac13_cic_vs_oracle():
    # 48³×10 smoke for test speed; the Cox relation + P(N) are tight at any n_real, the
    # Route-A linear moment is cosmic-variance-limited at small n_real -> theory_tol loosened
    # (the script main() runs the full 48³×24 ensemble where Route A reaches ~2.5%).
    res = acceptance.ac13_cic_vs_oracle(shape=(48, 48, 48), n_real=10, c=4,
                                        cox_tol=0.06, theory_tol=0.18, l1_tol=0.12)
    assert res["passed"]
