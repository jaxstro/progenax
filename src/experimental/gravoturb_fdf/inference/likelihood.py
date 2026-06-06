r"""Gaussian likelihood on the gravoturb_fdf data vector (Milestone 1).

Data vector ``d(theta) = [P_s(k_i) band-powers, sigma^2_N(c_j) CIC variances]``, all analytic
and differentiable in ``theta = (mach, b, alpha, beta)``. The CIC block uses the cubic-cell
box window (matching square-cell counting). The Gaussian log-likelihood

    log L = -1/2 (d(theta) - data)^T  Cinv  (d(theta) - data)

uses the mock precision ``Cinv`` (Hartlap-corrected, fixed at fiducial theta -- Decision #4),
so there is no log|C(theta)| term to bias the fit. Differentiable in theta for HMC (Phase 6)
and exact at the MAP on noiseless data. JAX-native.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from gravoturb_fdf.inference.covariance import power_spectrum_bandpowers
from gravoturb_fdf.theory.cic import cell_averaged_xi_rho, cic_variance
from gravoturb_fdf.theory.projection import box_window_sq_grid


def data_vector(
    theta: Float[Array, " 4"],
    shape: tuple[int, int, int],
    k_edges: Float[Array, " kp1"],
    cell_sizes: tuple[int, ...],
    n_bar: Float[Array, ""],
    n_max: int = 14,
    n_quad: int = 256,
) -> Float[Array, " d"]:
    r"""Predicted data vector ``[P_s(k_1..k_K), sigma^2_N(c_1..c_m)]`` at ``theta``.

    ``theta = (mach, b, alpha, beta)``. The 2-pt block is the log-density power-spectrum
    band-powers (the beta-carrier); the CIC block is ``sigma^2_N = N_bar + N_bar^2 xi_bar_rho``
    at each cubic cell size ``c`` (box window). Differentiable in theta."""
    mach, b, alpha, beta = theta
    _, P, _ = power_spectrum_bandpowers(shape, beta, mach, b, alpha, k_edges, n_max, n_quad)
    cic = [
        cic_variance(
            n_bar,
            cell_averaged_xi_rho(shape, beta, float(c), mach, b, alpha, n_max=n_max,
                                 n_quad=n_quad, w2=box_window_sq_grid(shape, c)),
        )
        for c in cell_sizes
    ]
    return jnp.concatenate([P, jnp.stack(cic)])


def gaussian_loglike(
    data: Float[Array, " d"],
    theta: Float[Array, " 4"],
    precision: Float[Array, " d d"],
    **cfg,
) -> Float[Array, ""]:
    r"""Gaussian log-likelihood ``-1/2 r^T Cinv r``, ``r = d(theta) - data``.

    ``precision`` = fixed mock ``Cinv`` (Hartlap-corrected). Maximal (=0) at theta_true on
    noiseless data; differentiable in theta. ``cfg`` are the :func:`data_vector` keyword args."""
    resid = data_vector(theta, **cfg) - data
    return -0.5 * resid @ (precision @ resid)
