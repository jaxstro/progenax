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


def test_ac8_ac9_grads():
    assert acceptance.ac8_ac9_grads()["passed"]
