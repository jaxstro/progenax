"""Unit tests for the EXACT BM19 DENSITY 2-pt Hermite coefficients (A-new1, Piece 1).

The density map is ``rho = e^s = e^{T(g)}``; its probabilists'-Hermite coefficients are

    d_n = <exp(T(g)) He_n(g)> ,   T(g) = s_of_g(g; M,b,alpha) .

These feed ``gaussianized_xi(rho_g, d)`` to give the EXACT density 2-pt (the Mehler
bivariate-Hermite expansion ``xi_rho(r) = sum_{n>=1} d_n^2/n! rho_g(r)^n``), as opposed to
the lognormal-limit ``expm1(xi_s)``. Two derived invariants:
  - d_0 = <e^s> = 1  (mean density; the rho0 convention enforced in s_of_g)
  - xi_rho(0) = sum_{n>=1} d_n^2/n! = Var(rho)  (density variance)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def test_density_hermite_shape_and_mean_density_unity():
    """Returns (n_max+1,); d_0 = <e^s> = 1 (mean density, rho0 convention)."""
    from gravoturb.theory.log_correlations import bm19_density_hermite_coefficients

    n_max = 8
    d = bm19_density_hermite_coefficients(8.0, 0.4, 2.5, n_max)
    assert d.shape == (n_max + 1,)
    assert float(d[0]) == pytest.approx(1.0, abs=1e-3)


def test_density_variance_matches_measured_field():
    """sum_{n>=1} d_n^2/n! (predicted Var(rho)) matches measured Var(exp(s)) within a few %.

    The predicted ``xi_rho(0)`` is the *infinite-resolution* density variance (converged in
    n_max to ~0.1% by n_max=14). The measured finite-grid Var(exp(s)) of the
    mass-conserving copula approaches it monotonically from below as the grid under-resolves
    less of the fat power-law tail -- a DETERMINISTIC (no cosmic-variance) resolution effect:
    rel = +8.8% (48^3), +7.0% (64^3), +5.0% (96^3), +3.8% (128^3). We test at 96^3 with a 6%
    budget that reflects this documented finite-grid residual, NOT a model error. (The amplitude
    residual is the known forecast-grade term; the keystone observable for beta is the SLOPE,
    tested separately.)
    """
    from gravoturb.realization.gaussian_field import gaussian_random_field
    from gravoturb.realization.copula import mass_conserving_copula_field
    from gravoturb.theory.log_correlations import bm19_density_hermite_coefficients

    mach, b, alpha = 8.0, 0.4, 2.5
    shape = (96, 96, 96)
    d = bm19_density_hermite_coefficients(mach, b, alpha, n_max=14)
    n = jnp.arange(d.shape[0])
    # sum_{n>=1} d_n^2 / n!  (drop the n=0 mean term) = xi_rho(0) = Var(rho)
    var_pred = float(jnp.sum((d**2 / jnp.exp(jax.scipy.special.gammaln(n + 1.0)))[1:]))

    g = gaussian_random_field(shape, beta=3.0, key=jax.random.PRNGKey(0))
    s = np.asarray(mass_conserving_copula_field(g, mach, b, alpha))
    var_meas = float(np.var(np.exp(s)))

    # measured converges UP toward the analytic limit; residual is finite-grid tail resolution.
    assert var_pred == pytest.approx(var_meas, rel=0.06)
    assert var_pred > var_meas  # analytic = infinite-resolution limit, measured < it


def test_density_hermite_differentiable_in_mach():
    """jax.grad of sum(d_n) wrt mach is finite and nonzero."""
    from gravoturb.theory.log_correlations import bm19_density_hermite_coefficients

    g = jax.grad(lambda m: jnp.sum(bm19_density_hermite_coefficients(m, 0.4, 2.5, 8)))(
        8.0
    )
    assert np.isfinite(float(g))
    assert float(g) != 0.0
