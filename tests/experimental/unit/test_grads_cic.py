"""Phase 4 (AC14): gradient validation of the differentiable predicted-statistics layer.

A correct forward value does NOT imply a correct gradient -- and Phases 5-6 (Fisher, HMC)
ride entirely on jax.grad being right. Per research-workflow:gradient-validation: float64
(on at package import), autodiff vs CENTRAL finite-difference with an h-sweep (rel < 1e-5),
and finiteness at domain edges (alpha->2 fat-tail boundary, small R, beta extremes). The
contentious analytic beta path (Decision #3, no soft-sort) gets its own checks here and a
paired-CRN-vs-simulator check in AC14.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental

_BASE = dict(mach=5.0, b=0.4, alpha=2.5, beta=3.0)
_INIT = {"mach": 5.0, "b": 0.4, "alpha": 2.5, "beta": 3.0}


def _grad_check(f, x, ad, hs=(1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6)):
    """Min relative error between autodiff ``ad`` and central FD over an h-sweep.

    Sweeping h guards against a single step falsely failing (truncation at large h,
    round-off at small h). If ``ad`` were wrong, NO h would match (the FD converges to the
    true derivative across a range of h), so min-over-h < tol is a strong check."""
    rels = []
    for h in hs:
        fd = (f(x + h) - f(x - h)) / (2.0 * h)
        rels.append(abs(ad - fd) / (abs(ad) + abs(fd) + 1e-30))
    return min(rels)


def _scalar_and_grad(which, stat_fn):
    """Return (f_float, ad) for varying parameter ``which`` through ``stat_fn(**params)``."""

    def f(val):
        p = {**_BASE, which: val}
        return float(stat_fn(p))

    ad = float(jax.grad(lambda val: stat_fn({**_BASE, which: val}))(_INIT[which]))
    return f, ad


@pytest.mark.parametrize("which", ["mach", "b", "alpha", "beta"])
def test_gradcheck_cell_averaged_xi_rho(which):
    """Route-A CIC clustering term xi_bar_rho(R): autodiff vs central FD in each param."""
    from gravoturb_fdf.theory.cic import cell_averaged_xi_rho

    shape, R = (24, 24, 24), 2.0
    stat = lambda p: cell_averaged_xi_rho(
        shape, p["beta"], R, p["mach"], p["b"], p["alpha"], n_max=14
    )
    f, ad = _scalar_and_grad(which, stat)
    assert _grad_check(f, _INIT[which], ad) < 1e-5


@pytest.mark.parametrize("which", ["mach", "b", "alpha", "beta"])
def test_gradcheck_cic_variance(which):
    """sigma^2_N(R) = N_bar + N_bar^2 xi_bar: autodiff vs central FD in each param."""
    from gravoturb_fdf.theory.cic import cell_averaged_xi_rho, cic_variance

    shape, R, n_bar = (24, 24, 24), 2.0, 20.0
    stat = lambda p: cic_variance(
        n_bar,
        cell_averaged_xi_rho(
            shape, p["beta"], R, p["mach"], p["b"], p["alpha"], n_max=14
        ),
    )
    f, ad = _scalar_and_grad(which, stat)
    assert _grad_check(f, _INIT[which], ad) < 1e-5


@pytest.mark.parametrize("which", ["mach", "b", "alpha", "beta"])
def test_gradcheck_smoothed_log_variance(which):
    """Route-B sigma_s^2(R) (sets p_R): autodiff vs central FD in each param."""
    from gravoturb_fdf.theory.cic import smoothed_log_variance

    shape, R = (24, 24, 24), 2.0
    stat = lambda p: smoothed_log_variance(
        shape, p["beta"], R, p["mach"], p["b"], p["alpha"], n_max=14
    )
    f, ad = _scalar_and_grad(which, stat)
    assert _grad_check(f, _INIT[which], ad) < 1e-5


@pytest.mark.parametrize("which", ["mach", "b", "alpha", "beta"])
def test_gradcheck_count_distribution_tail(which):
    """A compound-Poisson P(N) functional (P(N>25), the over-dense cells): autodiff vs FD."""
    from gravoturb_fdf.theory.cic import count_distribution

    shape, R, n_bar = (20, 20, 20), 2.0, 12.0
    N = jnp.arange(0, 160)
    mask = (N > 25).astype(float)

    def stat(p):
        pN = count_distribution(
            N,
            n_bar,
            shape,
            p["beta"],
            R,
            p["mach"],
            p["b"],
            p["alpha"],
            n_max=10,
            n_s=1024,
        )
        return jnp.sum(mask * pN)

    f, ad = _scalar_and_grad(which, stat)
    assert _grad_check(f, _INIT[which], ad) < 1e-5


@pytest.mark.parametrize("alpha", [2.0, 2.05, 2.5])
def test_grad_finite_at_fat_tail_boundary(alpha):
    """Gradients finite (no NaN/inf) at the alpha->2 fat-tail boundary where <rho^2> diverges
    -- the cell scale R must keep the moment + its gradient finite (Decision #1)."""
    from gravoturb_fdf.theory.cic import cell_averaged_xi_rho

    shape, R = (24, 24, 24), 2.0
    g = jax.grad(lambda a: cell_averaged_xi_rho(shape, 3.0, R, 5.0, 0.4, a, n_max=14))(
        alpha
    )
    assert np.isfinite(float(g))


@pytest.mark.parametrize("R", [0.5, 1.0, 8.0])
@pytest.mark.parametrize("beta", [1.5, 3.0, 4.0])
def test_grad_finite_small_R_and_beta_extremes(R, beta):
    """beta-gradient finite across the smoothing scale R and beta range (the analytic
    beta path through ifftn(k^{-beta}) -- Decision #3, validated not just nonzero but finite)."""
    from gravoturb_fdf.theory.cic import cell_averaged_xi_rho

    shape = (24, 24, 24)
    g = jax.grad(
        lambda be: cell_averaged_xi_rho(shape, be, R, 5.0, 0.4, 2.5, n_max=14)
    )(beta)
    assert np.isfinite(float(g))
