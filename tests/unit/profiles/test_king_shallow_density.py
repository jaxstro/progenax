"""Independent low-potential King density checks, including cancellation failure."""

import jax
import jax.numpy as jnp
import numpy as np

from progenax.profiles._spherical_poisson import regular_center_state
from progenax.profiles.king import _king_poisson_rhs, king_lowered_maxwellian_density

_density_and_grad = jax.jit(
    jax.vmap(
        lambda w: (
            king_lowered_maxwellian_density(w),
            jax.grad(king_lowered_maxwellian_density)(w),
        )
    )
)
_curvature = jax.jit(
    lambda w: _king_poisson_rhs(1e-6, regular_center_state(w, 1e-6), (w,))[1]
)


def test_density_and_derivative_match_rescaled_velocity_integral():
    """Catch negative/cancelled densities near W=0 and incorrect series coefficients."""
    w = np.concatenate((np.logspace(-10, -1, 10), [0.099999, 0.100001, 0.3]))
    nodes, weights = np.polynomial.legendre.leggauss(64)
    t = (nodes + 1) / 2
    weights = weights / 2
    # v=sqrt(2W)*t rescales the shrinking velocity domain to [0,1].
    # expm1 retains the small lowered-Maxwellian difference independently.
    integral = np.sum(weights * t**2 * np.expm1(w[:, None] * (1 - t**2)), axis=1)
    expected = 4 / np.sqrt(np.pi) * w**1.5 * integral
    derivative = 1.5 * expected / w + 4 / np.sqrt(np.pi) * w**1.5 * np.sum(
        weights * t**2 * (1 - t**2) * np.exp(w[:, None] * (1 - t**2)), axis=1
    )
    value, grad = map(np.asarray, _density_and_grad(jnp.asarray(w)))
    # Fixed 64-node smooth quadrature has near-roundoff error here; the closed
    # form just above the series switch loses ~1e-14 relatively by subtraction.
    np.testing.assert_allclose(value, expected, rtol=5e-13, atol=0)
    np.testing.assert_allclose(grad, derivative, rtol=5e-13, atol=0)


def test_shallow_central_source_is_not_disabled_by_absolute_density_cutoff():
    """Catch rho0>1e-10 suppressing all gravity at the valid shallow W0=1e-4."""
    # rho'/rho ~ 5/(2W0), so the finite-start correction is ~3.4e-7.
    np.testing.assert_allclose(_curvature(1e-4), -3.0, atol=5e-7, rtol=0)
