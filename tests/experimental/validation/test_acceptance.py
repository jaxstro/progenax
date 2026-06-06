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
    # 40³×12 for test speed; Limber-projected analytic 2-pt vs projected-realization 2-pt.
    res = acceptance.ac12_limber_projection_vs_oracle(shape=(40, 40, 40), n_real=12)
    assert res["passed"]
