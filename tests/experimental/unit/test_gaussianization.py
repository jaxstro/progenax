"""Unit tests for gravoturb_fdf.theory.gaussianization.

Phase 1 of the differentiable predicted-statistics layer. The copula map
``s = T(g) = bm19_icdf(Phi(g)) - log<e^s>`` carries the BM19 marginal onto a unit
Gaussian field; its Hermite expansion gives the analytic log-density 2-point ``xi_s``.

Grounding: docs/plans/2026-06-05-gaussianization-formula-verification.md
(Coles & Jones 1991 Eq 21 mean-1 shift; Eq 30 exp-case 2-pt; Eq 17 moments).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def test_s_of_g_lognormal_limit_recovers_gaussian_moments():
    """Large-alpha (tail-negligible) BM19 => s ~ Normal(-sigma_s^2/2, sigma_s^2)."""
    from gravoturb_fdf.theory.gaussianization import s_of_g
    from gravoturb_fdf.theory.bm19 import sigma_s_squared

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
