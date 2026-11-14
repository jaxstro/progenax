"""Algebraic reference-adapter checks; no physical IMF qualification."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from validate_limepy_reference import absolute_moments_from_reference


def _reference():
    return dict(
        mj=np.array([1.0, 2.0]),
        Mj=np.array([3.0, 6.0]),
        Kj=np.array([9.0, 18.0]),
        M=9.0,
        G=9.0,
        r0=3.0,
        s2=3.0,
        converged=True,
        fit_residual_max_abs=0.0,
        numerics="{}",
        provenance="{}",
    )


def test_absolute_kinetic_adapter_matches_finite_label_enumeration():
    """Three labels only check finite-N algebra and the three-dimensional scale."""
    from itertools import product

    actual = absolute_moments_from_reference(
        _reference(), n_stars=3, gravity=2.0, r_c=4.0
    )
    values = []
    for labels in product([1.0, 2.0], repeat=3):
        mass = np.asarray(labels)
        total = mass.sum()
        # q=2, mu=1, G/(9 rc mu)=1/18; physical speeds scale with total mass.
        values.append([total**2, np.mean(mass * 2 * total / 18)])
    np.testing.assert_allclose(actual["values"], np.mean(values, axis=0), rtol=1e-15)
    np.testing.assert_array_equal(actual["q"], [2.0, 2.0])
    assert actual["mu_tot"] == 1.0


def test_reference_physical_scaling_preserves_dimensionless_absolute_moments():
    ref = _reference()
    expected = absolute_moments_from_reference(ref)
    ref.update(
        G=ref["G"] * 2,
        M=ref["M"] * 5,
        Mj=ref["Mj"] * 5,
        r0=ref["r0"] * 3,
        s2=ref["s2"] * 10 / 3,
        Kj=ref["Kj"] * 50 / 3,
    )
    actual = absolute_moments_from_reference(ref)
    np.testing.assert_allclose(actual["values"], expected["values"], rtol=1e-15)


def test_absolute_comparison_exposes_uniform_speed_error():
    ref = _reference()
    expected = absolute_moments_from_reference(ref)
    ref["Kj"] *= 4
    actual = absolute_moments_from_reference(ref)
    assert actual["values"][0] == expected["values"][0]
    assert actual["values"][1] == 4 * expected["values"][1]


@pytest.mark.parametrize("field", ["Kj", "s2", "fit_residual_max_abs", "numerics"])
def test_shape_only_cache_cannot_supply_absolute_comparison(field):
    ref = _reference()
    del ref[field]
    with pytest.raises(ValueError, match="metadata missing"):
        absolute_moments_from_reference(ref)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("Kj", np.array([np.nan, 18.0]), "finite and positive"),
        ("s2", 0.0, "finite and positive"),
        ("Mj", np.array([3.0]), "must align"),
        ("converged", False, "did not converge"),
        ("fit_residual_max_abs", np.nan, "residual must"),
    ],
)
def test_invalid_reference_state_rejected(field, value, message):
    ref = _reference()
    ref[field] = value
    with pytest.raises(ValueError, match=message):
        absolute_moments_from_reference(ref)
