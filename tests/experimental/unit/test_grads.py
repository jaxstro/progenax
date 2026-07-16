"""Differentiability of the gravoturb 1D theory — AC8 (grad signs) + AC9 (FD vs autodiff).

Public differentiable entry points: sigma_s_squared, f_dense_bm19_full,
magnification_factor (and bm19_icdf, checked in test_pdf). All must be smooth and
grad-correct in the cloud parameters; float64 is enabled at package import.
"""

import math

import jax
import pytest

pytestmark = pytest.mark.experimental


def _central_fd(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


# ── AC8: gradient signs ──
def test_grad_sign_sigma_s_squared_positive():
    from gravoturb.theory.density_pdf import sigma_s_squared

    assert (
        float(jax.grad(lambda m: sigma_s_squared(m, 0.4))(5.0)) > 0.0
    )  # widens with Mach


def test_grad_sign_f_dense_decreases_with_mach_and_alpha():
    from gravoturb.theory.density_pdf import f_dense_bm19_full

    dM = float(jax.grad(lambda m: f_dense_bm19_full(m, 1.0 / 3, 1.8))(8.0))
    da = float(jax.grad(lambda a: f_dense_bm19_full(8.0, 1.0 / 3, a))(1.8))
    assert dM < 0.0  # higher Mach -> less dense gas
    assert da < 0.0  # steeper tail  -> less dense gas


def test_grad_sign_zeta_decreases_with_alpha():
    """zeta rises with p but p = 3/alpha, so d zeta/d alpha < 0."""
    from gravoturb.theory.density_pdf import pdf_slope_to_radial
    from gravoturb.theory.dense_gas_sfr import magnification_factor

    zeta_of_alpha = lambda a: magnification_factor(pdf_slope_to_radial(a))
    assert float(jax.grad(zeta_of_alpha)(2.0)) < 0.0  # alpha=2 -> p=1.5


# ── AC9: finite-difference vs autodiff agreement (rel err < 1e-4) ──
@pytest.mark.parametrize("mach", [3.0, 5.0, 12.0])
def test_fd_vs_autodiff_sigma_s_squared(mach):
    from gravoturb.theory.density_pdf import sigma_s_squared

    f = lambda m: float(sigma_s_squared(m, 0.45))
    ad = float(jax.grad(lambda m: sigma_s_squared(m, 0.45))(mach))
    assert ad == pytest.approx(_central_fd(f, mach), rel=1e-4)


@pytest.mark.parametrize("alpha", [1.6, 1.8, 2.2])
def test_fd_vs_autodiff_f_dense_in_alpha(alpha):
    from gravoturb.theory.density_pdf import f_dense_bm19_full

    f = lambda a: float(f_dense_bm19_full(6.0, 0.5, a))
    ad = float(jax.grad(lambda a: f_dense_bm19_full(6.0, 0.5, a))(alpha))
    assert ad == pytest.approx(_central_fd(f, alpha), rel=1e-4)


@pytest.mark.parametrize("mach", [4.0, 9.0])
def test_fd_vs_autodiff_f_dense_in_mach(mach):
    from gravoturb.theory.density_pdf import f_dense_bm19_full

    f = lambda m: float(f_dense_bm19_full(m, 0.4, 1.7))
    ad = float(jax.grad(lambda m: f_dense_bm19_full(m, 0.4, 1.7))(mach))
    assert ad == pytest.approx(_central_fd(f, mach), rel=1e-4)


@pytest.mark.parametrize("p", [0.5, 1.0, 1.5, 1.8])
def test_fd_vs_autodiff_magnification(p):
    from gravoturb.theory.dense_gas_sfr import magnification_factor

    f = lambda x: float(magnification_factor(x))
    ad = float(jax.grad(magnification_factor)(p))
    assert ad == pytest.approx(_central_fd(f, p), rel=1e-4)


# ── guard regions: gradients stay finite near the alpha->1 and p->2 singularities ──
def test_grads_finite_near_guards():
    from gravoturb.theory.density_pdf import f_dense_bm19_full
    from gravoturb.theory.dense_gas_sfr import magnification_factor

    g_fdense = float(jax.grad(lambda a: f_dense_bm19_full(6.0, 0.5, a))(1.05))
    g_zeta = float(jax.grad(magnification_factor)(1.95))
    assert math.isfinite(g_fdense)
    assert math.isfinite(g_zeta)
