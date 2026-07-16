"""PP20 magnification factor zeta(p) — physics tests.

Parmentier & Pasquali 2020 (ApJ 903, 56), Eq. 6 (embedded in Eq. 9), verified
against the held PDF (docs/core-papers/Parmentier_2020_ApJ_903_56.pdf):

    zeta(p) = (3 - p)^{3/2} / [ (3^{3/2}/2) (2 - p) ]

The published constant "2.6" is the rounded value of 3^{3/2}/2 = 2.598076...; the
exact constant is fixed by the physical top-hat lower limit zeta(0) = 1 and makes
zeta(1.5) = sqrt(2) exactly. Valid on 0 <= p < 2; diverges only at p -> 2.
There is NO pole at p = 1.3 (a previously-caught transcription fabrication).
"""

import math

import jax
import pytest

pytestmark = pytest.mark.experimental


# ── Task 1.5: analytic zeta(p) (PP20 Eq. 6) — AC3 anchors ──
@pytest.mark.parametrize(
    "p,expected",
    [
        (0.0, 1.0),  # top-hat lower limit (exact)
        (1.0, 1.0887),  # 2^{2.5}/3^{1.5}
        (1.5, math.sqrt(2.0)),  # exact sqrt(2)
        (1.67, 1.79),  # PP20 anchor
    ],
)
def test_zeta_analytic_anchors(p, expected):  # AC3
    from gravoturb.theory.dense_gas_sfr import magnification_factor

    val = float(magnification_factor(p))
    assert val == pytest.approx(expected, rel=1e-3)


def test_zeta_no_spurious_pole_at_1p3():
    """zeta is finite and smooth across p=1.3 (the fabricated-pole region)."""
    from gravoturb.theory.dense_gas_sfr import magnification_factor

    vals = [float(magnification_factor(p)) for p in (1.25, 1.29, 1.30, 1.31, 1.35)]
    assert all(math.isfinite(v) and 1.0 < v < 2.0 for v in vals)
    # monotone increasing through 1.3 (no blow-up)
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_zeta_diverges_only_at_2():
    from gravoturb.theory.dense_gas_sfr import magnification_factor

    assert float(magnification_factor(1.99)) > 5.0  # large near p=2
    assert math.isfinite(float(magnification_factor(1.9)))


def test_zeta_increases_with_p_and_grad_positive():  # AC8 (partial)
    from gravoturb.theory.dense_gas_sfr import magnification_factor

    g = jax.grad(magnification_factor)(1.5)
    assert float(g) > 0.0  # steeper profile -> higher magnification


# ── Task 1.6: cored numerical zeta + direct-field zeta_FDF — AC4 ──
def test_zeta_with_core_approaches_powerlaw_as_core_shrinks():
    """Cored profile rho ~ [1+(r/r_c)^2]^{-p/2}: as r_c/R -> 0 it approaches a
    pure power law, so numerical zeta -> analytic zeta(p)."""
    from gravoturb.theory.dense_gas_sfr import (
        magnification_factor,
        magnification_factor_with_core,
    )

    p = 1.5
    analytic = float(magnification_factor(p))
    cuspy = float(magnification_factor_with_core(p, r_c_over_R=1e-3))
    assert cuspy == pytest.approx(analytic, rel=0.05)


def test_zeta_with_core_top_hat_limit():
    """Large core (r_c >> R) -> nearly uniform -> zeta -> 1."""
    from gravoturb.theory.dense_gas_sfr import magnification_factor_with_core

    assert float(
        magnification_factor_with_core(1.5, r_c_over_R=100.0)
    ) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("p", [0.5, 1.0, 1.5])
def test_zeta_from_field_matches_analytic_powerlaw(p):  # AC4
    """Direct-field estimator on a sampled pure power-law sphere matches
    analytic zeta(p) within a few percent for p < 1.7."""
    import numpy as np
    from gravoturb.theory.dense_gas_sfr import magnification_factor, zeta_from_field

    # Radial power-law rho ~ (r/R)^{-p} on shells; weights = shell volumes.
    r = np.linspace(1e-3, 1.0, 40_000)
    rho = r ** (-p)
    w = 4.0 * np.pi * r**2  # dV ~ r^2 dr (uniform dr -> constant factor cancels)
    val = float(zeta_from_field(rho, w))
    assert val == pytest.approx(float(magnification_factor(p)), rel=0.03)
