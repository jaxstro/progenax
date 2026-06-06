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
from gravoturb_fdf.theory.cic import cell_averaged_xi_rho, cic_variance, count_distribution
from gravoturb_fdf.theory.pdf import bm19_volume_pdf
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


def count_loglike(
    count_hist: Float[Array, " nmax"],
    theta: Float[Array, " 4"],
    shape: tuple[int, int, int],
    cell_size: int,
    n_bar: Float[Array, ""],
    n_max: int = 14,
    n_quad: int = 256,
    n_s: int = 1024,
    s_max: float = 40.0,
    floor: float = 1e-300,
) -> Float[Array, ""]:
    r"""1-pt compound-Poisson count log-likelihood ``sum_N hist[N] log P(N|theta)`` (Milestone 2).

    Treats the cells as i.i.d. draws from the count distribution ``P(N|theta)``
    (:func:`count_distribution`, cubic-cell box window); the spatial correlation is the separate
    2-pt block. ``count_hist[N]`` = number of observed cells with count N (N = 0..len-1). This is
    the STAR-LEVEL, shot-noise-included observable whose high-N tail pins alpha (the density-PDF
    tail slope). Differentiable in ``theta = (mach, b, alpha, beta)``; the ``log`` is floored to
    stay finite where P(N) underflows in the deep tail."""
    mach, b, alpha, beta = theta
    n = count_hist.shape[0]
    pN = count_distribution(
        jnp.arange(n), n_bar, shape, beta, float(cell_size), mach, b, alpha,
        n_max=n_max, n_quad=n_quad, w2=box_window_sq_grid(shape, cell_size),
        n_s=n_s, s_max=s_max,
    )
    pN = pN / jnp.sum(pN)  # normalize over the support [0, nmax] -> conditional P(N|N<=nmax);
    # removes a theta-dependent truncation bias on alpha (the tail param) for finite nmax.
    return jnp.sum(count_hist * jnp.log(jnp.clip(pN, floor, None)))


def density_pdf_loglike(
    s_hist: Float[Array, " nb"],
    s_centers: Float[Array, " nb"],
    theta: Float[Array, " 4"],
    floor: float = 1e-300,
) -> Float[Array, ""]:
    r"""1-pt log-density-PDF log-likelihood ``sum_bins hist[s] log p_BM19(s;M,b,alpha)`` (M2).

    The faithful, ALPHA-sensitive observable: the BM19 volume density PDF (lognormal body +
    power-law tail). Stars don't carry alpha at realistic sampling (the tail is washed out by
    cell-smoothing and shot noise), so alpha needs a GAS-density tracer's 1-pt PDF (extinction /
    column-density / dust maps) -- this block represents that probe. ``s_hist[i]`` = (relative)
    count of density samples in bin ``i`` centred at ``s_centers[i]``; ``p_BM19`` is normalized
    over the bins. Constrains (sigma_s^2 = ln(1+(b M)^2), alpha); pairs with the stellar CIC +
    band-powers (sigma_s^2, beta) for a joint (M, alpha, beta) inference. Differentiable in theta."""
    mach, b, alpha, _beta = theta
    p = bm19_volume_pdf(s_centers, mach, b, alpha)
    p = p / jnp.trapezoid(p, s_centers)  # normalize over the observed support
    return jnp.sum(s_hist * jnp.log(jnp.clip(p, floor, None)))
