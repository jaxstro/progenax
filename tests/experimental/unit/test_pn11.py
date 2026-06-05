"""PN11 classical critical density for collapse — physics tests.

Padoan & Nordlund 2011 (ApJ 730, 40), hydrodynamic (non-magnetized) case,
verified against the held PDF (docs/core-papers/Padoan_2011_ApJ_730_40.pdf):

    Eq. 8   rho_cr/rho_0 = 0.067 theta^{-2} alpha_vir M^2
    Eq. 11  with theta = 0.35  ->  rho_cr/rho_0 = 0.547 alpha_vir M^2

This is a clearly-labelled CLASSICAL ALTERNATIVE to the BM19 transition density,
NOT the default path. (Prior code used a prefactor 0.242, ~2.3x too small.)
"""

import math

import jax
import pytest

pytestmark = pytest.mark.experimental


def test_pn11_prefactor_is_0p547():
    """0.067 * theta^{-2} with theta=0.35 = 0.547 (PN11 Eq. 11), not 0.242."""
    from gravoturb_fdf.theory.pn11 import critical_overdensity_pn11

    val = float(critical_overdensity_pn11(mach=1.0, alpha_vir=1.0))  # theta=0.35 default
    assert val == pytest.approx(0.067 * 0.35**-2, rel=1e-6)
    assert val == pytest.approx(0.547, abs=1e-3)


def test_s_crit_pn11_known_value():
    """s_crit = ln(0.547 alpha_vir M^2); alpha_vir=1, M=10 -> ln(54.7)."""
    from gravoturb_fdf.theory.pn11 import s_crit_pn11

    val = float(s_crit_pn11(mach=10.0, alpha_vir=1.0))
    assert val == pytest.approx(math.log(0.067 * 0.35**-2 * 100.0), abs=1e-9)


def test_s_crit_pn11_monotonic():
    from gravoturb_fdf.theory.pn11 import s_crit_pn11

    # rises with Mach (M^2) and with virial parameter
    assert float(s_crit_pn11(20.0, 1.0)) > float(s_crit_pn11(10.0, 1.0))
    assert float(s_crit_pn11(10.0, 2.0)) > float(s_crit_pn11(10.0, 1.0))


def test_s_crit_pn11_mach_squared_grad():
    from gravoturb_fdf.theory.pn11 import s_crit_pn11

    # d s_crit / d M = 2/M  -> at M=10, grad = 0.2
    g = float(jax.grad(lambda m: s_crit_pn11(m, 1.0))(10.0))
    assert g == pytest.approx(0.2, rel=1e-6)
