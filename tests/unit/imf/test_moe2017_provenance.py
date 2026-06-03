"""Pin the Moe+2017 binary-model constants (audit minor: stale moe2017 docstring).

The moe2017() docstring advertised a stale binary-fraction fit (a=0.0416, b=1.3925),
but the code uses the quadratic-logit fit (a=-0.2799, b=1.4170, c=0.4755) via
DifferentiableBinaryFraction.from_moe2017(). The model is correct; the docstring lied.

These tests pin the *actual* runtime constants so the corrected docstring cannot drift
from the code again.

Provenance: Moe & Di Stefano (2017), ApJS 230, 15, Table 13 (binary fraction) and the
mass-ratio slope gamma(m) = c + d*log10(m).
"""

import pytest

import progenax  # noqa: F401  (enables float64)
from progenax.imf.differentiable_binary import (
    DifferentiableBinaryFraction,
    DifferentiableBinaryModel,
)


def test_from_moe2017_binary_fraction_constants():
    bf = DifferentiableBinaryFraction.from_moe2017()
    assert float(bf.a) == pytest.approx(-0.2799)
    assert float(bf.b) == pytest.approx(1.4170)
    assert float(bf.c) == pytest.approx(0.4755)


def test_moe2017_model_constants():
    model = DifferentiableBinaryModel.moe2017()
    # Binary fraction = quadratic logit from from_moe2017()
    assert float(model.binary_fraction.a) == pytest.approx(-0.2799)
    assert float(model.binary_fraction.b) == pytest.approx(1.4170)
    assert float(model.binary_fraction.c) == pytest.approx(0.4755)
    # Mass-ratio slope gamma(m) = c + d*log10(m)
    assert float(model.gamma_intercept) == pytest.approx(0.1907)
    assert float(model.gamma_slope) == pytest.approx(-0.7521)
