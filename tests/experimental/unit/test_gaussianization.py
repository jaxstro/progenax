"""Unit tests for gravoturb_fdf.theory.gaussianization.

Phase 1 of the differentiable predicted-statistics layer. The copula map
``s = T(g) = bm19_icdf(Phi(g)) - log<e^s>`` carries the BM19 marginal onto a unit
Gaussian field; its Hermite expansion gives the analytic log-density 2-point ``xi_s``.

Grounding: internal design note
(Coles & Jones 1991 Eq 21 mean-1 shift; Eq 30 exp-case 2-pt; Eq 17 moments).
"""

from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def test_s_of_g_lognormal_limit_recovers_gaussian_moments():
    """Large-alpha (tail-negligible) BM19 => s ~ Normal(-sigma_s^2/2, sigma_s^2)."""
    from gravoturb_fdf.theory.bm19 import sigma_s_squared
    from gravoturb_fdf.theory.gaussianization import s_of_g

    mach, b, alpha = 5.0, 0.4, 6.0
    sig2 = float(sigma_s_squared(mach, b))
    g = jax.random.normal(jax.random.PRNGKey(0), (400_000,))
    s = s_of_g(g, mach, b, alpha)
    assert float(jnp.mean(s)) == pytest.approx(-0.5 * sig2, abs=0.02)
    assert float(jnp.var(s)) == pytest.approx(sig2, rel=0.03)


def test_s_of_g_mean_density_unity():
    """rho0 convention: population <e^s> = 1 via the analytic shift = log<e^s>.

    Uses alpha=3 (finite second moment) so the sample mean of rho is well behaved.
    """
    from gravoturb_fdf.theory.gaussianization import s_of_g

    mach, b, alpha = 5.0, 0.4, 3.0
    g = jax.random.normal(jax.random.PRNGKey(1), (400_000,))
    s = s_of_g(g, mach, b, alpha)
    assert float(jnp.mean(jnp.exp(s))) == pytest.approx(1.0, rel=3e-2)


def test_s_of_g_monotone_in_g():
    """T = F^{-1} o Phi is monotone increasing in g."""
    from gravoturb_fdf.theory.gaussianization import s_of_g

    g = jnp.linspace(-4.0, 4.0, 500)
    s = s_of_g(g, 5.0, 0.4, 2.0)
    assert bool(jnp.all(jnp.diff(s) > 0.0))


def test_s_of_g_differentiable_in_params():
    """grad of a permutation-invariant functional of s wrt mach is finite (no NaN)."""
    from gravoturb_fdf.theory.gaussianization import s_of_g

    g = jax.random.normal(jax.random.PRNGKey(2), (2000,))

    def loss(mach):
        return jnp.var(s_of_g(g, mach, 0.4, 2.0))

    grad = float(jax.grad(loss)(5.0))
    assert np.isfinite(grad)


# --- Task 1.2: Hermite coefficients c_n = <T(g) He_n(g)> -----------------------


def test_hermite_coefficients_linear_map_only_c1():
    """T(g) = sigma g + const  =>  c_0=const, c_1=sigma, c_{n>=2}=0 (to machine eps)."""
    from gravoturb_fdf.theory.gaussianization import hermite_coefficients

    sigma, const = 1.3, 0.7
    c = np.asarray(hermite_coefficients(lambda g: sigma * g + const, n_max=6))
    assert c[0] == pytest.approx(const, abs=1e-10)
    assert c[1] == pytest.approx(sigma, abs=1e-10)
    assert np.allclose(c[2:], 0.0, atol=1e-10)


def test_hermite_coefficients_exp_map_generating_function():
    """T(g)=exp(sigma g)  =>  c_n = sigma^n e^{sigma^2/2}  (C&J 1991 Eq 17 structure)."""
    from gravoturb_fdf.theory.gaussianization import hermite_coefficients

    sigma, n_max = 0.5, 6
    c = np.asarray(hermite_coefficients(lambda g: jnp.exp(sigma * g), n_max=n_max))
    n = np.arange(n_max + 1)
    expected = sigma**n * np.exp(0.5 * sigma**2)
    assert np.allclose(c, expected, rtol=1e-6)


def test_hermite_exp_map_variance_identity():
    """Var[e^{sigma g}] = sum_{n>=1} c_n^2/n! = e^{sigma^2}(e^{sigma^2}-1)."""
    from gravoturb_fdf.theory.gaussianization import hermite_coefficients

    sigma = 0.5
    c = np.asarray(hermite_coefficients(lambda g: jnp.exp(sigma * g), n_max=14))
    fact = np.array([factorial(k) for k in range(len(c))], dtype=float)
    var_series = float(np.sum(c[1:] ** 2 / fact[1:]))
    expected = np.exp(sigma**2) * (np.exp(sigma**2) - 1.0)
    assert var_series == pytest.approx(expected, rel=1e-5)


def test_bm19_hermite_coefficients_lognormal_limit_variance():
    """Large alpha (tail negligible): sum_{n>=1} c_n^2/n! ~ sigma_s^2 = Var[s]."""
    from gravoturb_fdf.theory.bm19 import sigma_s_squared
    from gravoturb_fdf.theory.gaussianization import bm19_hermite_coefficients

    mach, b, alpha = 5.0, 0.4, 6.0
    sig2 = float(sigma_s_squared(mach, b))
    c = np.asarray(bm19_hermite_coefficients(mach, b, alpha, n_max=10))
    fact = np.array([factorial(k) for k in range(len(c))], dtype=float)
    var_series = float(np.sum(c[1:] ** 2 / fact[1:]))
    assert var_series == pytest.approx(sig2, rel=0.05)


def test_bm19_hermite_coefficients_differentiable():
    """c_n(mach,b,alpha) differentiable: grad of c_2 wrt mach is finite."""
    from gravoturb_fdf.theory.gaussianization import bm19_hermite_coefficients

    def c2(mach):
        return bm19_hermite_coefficients(mach, 0.4, 2.0, n_max=4)[2]

    assert np.isfinite(float(jax.grad(c2)(5.0)))


# --- Task 1.3: the Gaussianization 2-point series xi_s(rho_g; c_n) -------------


def test_gaussianized_xi_zero_at_rho0_and_variance_at_rho1():
    """xi_s(0)=0; xi_s(1)=sum_{n>=1} c_n^2/n! = Var[s]."""
    from gravoturb_fdf.theory.gaussianization import (
        bm19_hermite_coefficients,
        gaussianized_xi,
    )

    c = bm19_hermite_coefficients(5.0, 0.4, 2.0, n_max=12)
    fact = np.array([factorial(k) for k in range(len(c))], dtype=float)
    var = float(np.sum(np.asarray(c)[1:] ** 2 / fact[1:]))
    assert float(gaussianized_xi(0.0, c)) == pytest.approx(0.0, abs=1e-12)
    assert float(gaussianized_xi(1.0, c)) == pytest.approx(var, rel=1e-10)


def test_gaussianized_xi_monotone_nondecreasing_in_rho():
    """All c_n^2/n! >= 0 => xi_s nondecreasing on rho in [0,1]."""
    from gravoturb_fdf.theory.gaussianization import (
        bm19_hermite_coefficients,
        gaussianized_xi,
    )

    c = bm19_hermite_coefficients(5.0, 0.4, 2.0, n_max=12)
    rho = jnp.linspace(0.0, 1.0, 50)
    xi = gaussianized_xi(rho, c)
    assert bool(jnp.all(jnp.diff(xi) >= -1e-12))


def test_gaussianized_xi_exp_map_matches_coles_jones_eq30():
    """Series with exp-map c_n reproduces e^{s^2}(e^{s^2 rho} - 1) = Coles & Jones
    1991 Eq (30) (the lognormal 1+xi=exp[Xi], up to the <Y>^2 normalization)."""
    from gravoturb_fdf.theory.gaussianization import (
        gaussianized_xi,
        hermite_coefficients,
    )

    sigma = 0.5
    c = hermite_coefficients(lambda g: jnp.exp(sigma * g), n_max=20)
    rho = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    xi = gaussianized_xi(rho, c)
    expected = np.exp(sigma**2) * (np.exp(sigma**2 * np.asarray(rho)) - 1.0)
    assert np.allclose(np.asarray(xi), expected, rtol=1e-6)


def test_gaussianized_xi_differentiable_in_params():
    """xi_s(rho*; theta) differentiable in (mach,b,alpha): finite grad wrt mach."""
    from gravoturb_fdf.theory.gaussianization import (
        bm19_hermite_coefficients,
        gaussianized_xi,
    )

    def xi_at(mach):
        c = bm19_hermite_coefficients(mach, 0.4, 2.0, n_max=8)
        return gaussianized_xi(0.5, c)

    assert np.isfinite(float(jax.grad(xi_at)(5.0)))
