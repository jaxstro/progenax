"""EFF & King mass-CDF quadrature accuracy (audit finding M5).

M5: eff.py and king.py built the cumulative mass as ``jnp.cumsum(integrand) * dr``
(a 1st-order left/right-Riemann sum) while the comment claimed a "trapezoid
approximation". The audit measured a ~6.3e-3 total-mass error for EFF vs ~1e-6 for a
true cumulative trapezoid on the same grid. The biased CDF skews the sampled radial
distribution.

Oracle: an independent fine-grid (200k-point) cumulative-trapezoid reference built
from the same analytic integrand, plus a measured order-of-accuracy check
(left-Riemann -> p~1, trapezoid -> p~2).
"""

import jax.numpy as jnp
import numpy as np
import pytest

import progenax  # noqa: F401  (enables float64)
from progenax.profiles.eff import EFFProfile
from progenax.profiles.king import KingProfile, king_lowered_maxwellian_density


def _fine_trapezoid_cdf(r_fine, f_fine, r_eval):
    """Reference CDF: cumulative trapezoid of f on a fine grid, evaluated at r_eval."""
    dr = r_fine[1] - r_fine[0]
    M = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(0.5 * (f_fine[1:] + f_fine[:-1])) * dr]
    )
    cdf = M / M[-1]
    return jnp.interp(r_eval, r_fine, cdf)


def _eff_reference_cdf(a, gamma, r_t, r_eval, n_fine=200_001):
    rf = jnp.linspace(0.0, r_t, n_fine)
    f = 4.0 * jnp.pi * rf**2 * jnp.power(1.0 + (rf / a) ** 2, -gamma / 2.0)
    return _fine_trapezoid_cdf(rf, f, r_eval)


def _king_reference_cdf(profile, r_eval, n_fine=200_001):
    """Reconstruct King's integrand at fine resolution from the profile's own
    ODE-interpolated psi (same density formula the constructor uses)."""
    rf = jnp.linspace(0.0, profile.r_t, n_fine)
    xi_local = rf / profile.r_c
    psi = jnp.interp(xi_local, profile.xi_grid, profile.psi_grid,
                     left=profile.W0, right=0.0)
    rho0 = king_lowered_maxwellian_density(profile.W0)
    rho = jnp.where(rho0 > 1e-10,
                    king_lowered_maxwellian_density(psi) / rho0, 0.0)
    rho = jnp.where(rf <= profile.r_t, rho, 0.0)
    f = 4.0 * jnp.pi * rf**2 * rho
    return _fine_trapezoid_cdf(rf, f, r_eval)


def test_eff_cdf_matches_fine_trapezoid_reference():
    prof = EFFProfile(a=1.0, gamma=3.0, r_t=10.0, n_grid=1000)
    ref = _eff_reference_cdf(1.0, 3.0, 10.0, prof._r_grid)
    max_err = float(jnp.max(jnp.abs(prof._cdf_grid - ref)))
    assert max_err < 1e-4, f"EFF CDF max error {max_err:.2e} (left-Riemann gives ~6e-3)"


def test_king_cdf_matches_fine_trapezoid_reference():
    prof = KingProfile.from_W0_rc(7.0, 1.0, n_grid=1000)
    ref = _king_reference_cdf(prof, prof._r_grid)
    max_err = float(jnp.max(jnp.abs(prof._cdf_grid - ref)))
    assert max_err < 1e-4, f"King CDF max error {max_err:.2e} (left-Riemann gives ~e-3)"


def test_eff_cdf_quadrature_is_second_order():
    """Measured order of accuracy at a fixed radius: left-Riemann ~1, trapezoid ~2."""
    r_test = jnp.asarray(1.0)  # EFF scale radius a
    ref = float(_eff_reference_cdf(1.0, 3.0, 10.0, r_test))

    def err(n):
        p = EFFProfile(a=1.0, gamma=3.0, r_t=10.0, n_grid=n)
        cdf_at_r = float(jnp.interp(r_test, p._r_grid, p._cdf_grid))
        return abs(cdf_at_r - ref)

    e1, e2 = err(250), err(1000)  # 4x refinement
    p_measured = np.log(e1 / e2) / np.log(4.0)
    assert p_measured > 1.6, (
        f"measured order p={p_measured:.2f} (trapezoid ~2, left-Riemann ~1); "
        f"errors {e1:.2e} -> {e2:.2e}"
    )
