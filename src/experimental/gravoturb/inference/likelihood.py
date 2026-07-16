r"""Gaussian likelihood on the gravoturb data vector (Milestone 1).

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

from gravoturb.inference.covariance import power_spectrum_bandpowers
from gravoturb.theory.counts_in_cells import (
    cell_averaged_xi_rho,
    cic_variance,
    count_distribution,
    predict_log_count_variance,
)
from gravoturb.theory.density_cdf import log_density_pdf
from gravoturb.theory.projection import box_window_sq_grid


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


def log_count_variance_loglike(
    measured_v: Float[Array, ""],
    theta: Float[Array, " 4"],
    shape: tuple[int, int, int],
    cell_size: int,
    n_bar: Float[Array, ""],
    var_v: Float[Array, ""],
    n_max: int = 14,
    n_quad: int = 256,
    n_s: int = 1024,
    s_max: float = 40.0,
) -> Float[Array, ""]:
    r"""Gaussian log-likelihood on the tail-robust CIC log-count variance (the sigma_s^2 -> mach block).

    Compares the measured ``Var_cells[log_plus(N)]`` to the analytic
    :func:`~gravoturb.theory.counts_in_cells.predict_log_count_variance` at ``theta``; ``var_v`` is the fixed
    (fiducial-mock) estimator variance (Decision #4 mock-precision pattern, no log|C| term). Replaces
    ``count_loglike`` in the inference path: tail-robust, so it does not bias mach high. Differentiable
    in ``theta = (mach, b, alpha, beta)`` (carries mach; alpha/beta enter only weakly via P(N)/R).
    """
    mach, b, alpha, beta = theta
    pred = predict_log_count_variance(
        n_bar, shape, beta, float(cell_size), mach, b, alpha,
        n_max=n_max, n_quad=n_quad, n_s=n_s, s_max=s_max,
        w2=box_window_sq_grid(shape, cell_size),
    )
    return -0.5 * (pred - measured_v) ** 2 / var_v


def density_pdf_loglike(
    s_hist: Float[Array, " nb"],
    s_centers: Float[Array, " nb"],
    theta: Float[Array, " 4"],
    floor: float = 1e-300,
) -> Float[Array, ""]:
    r"""1-pt log-density-PDF log-likelihood ``sum_bins hist[s] log p_BM19(s;M,b,alpha)`` (M2).

    A 1-pt density-PDF fit over the WHOLE BM19 PDF (lognormal body + power-law tail). CONTRACT:
    ``s_hist``/``s_centers`` must be on the same rho_0 = volume-mean convention as the model (the
    ``log_density_pdf`` s-axis, i.e. <e^s>=1) -- histogram a field with the empirical mean-1 shift
    (``rank_copula_field`` already returns shifted s). NOTE: for inferring the TAIL slope alpha,
    prefer :func:`tail_exceedance_loglike` (the POT block): fitting the full infinite-tail PDF to a
    finite (truncated) field biases alpha high, whereas the POT exceedance fit is exact,
    shift-immune, and geometry-free (see AC16). The robust convergent cross-check on the tail mass
    is ``dense_mass_fraction`` (AC17, Option B). This block remains a body+tail diagnostic.
    ``s_hist[i]`` = count in bin ``i`` centred at ``s_centers[i]``; ``p_BM19`` normalized over the
    bins. Differentiable in theta."""
    mach, b, alpha, _beta = theta
    p = log_density_pdf(s_centers, mach, b, alpha)
    p = p / jnp.trapezoid(p, s_centers)  # normalize over the observed support
    return jnp.sum(s_hist * jnp.log(jnp.clip(p, floor, None)))


def tail_exceedance_loglike(
    exc_counts: Float[Array, " nb"],
    exc_edges: Float[Array, " nbp1"],
    theta: Float[Array, " 4"],
    s_thr: float,
    s_max: float,
    floor: float = 1e-300,
) -> Float[Array, ""]:
    r"""Peaks-over-threshold log-likelihood for the BM19 tail slope alpha (Phase 6 finish).

    Above the transition s_t the BM19 density PDF is ``C e^{-alpha s}``; it is memoryless, so
    the gas log-density exceedances ``x = s - s_thr in [0, L]`` (``L = s_max - s_thr``) above any
    fixed threshold ``s_thr >= s_t`` are a TRUNCATED EXPONENTIAL of rate alpha:

        p(x) = alpha e^{-alpha x} / (1 - e^{-alpha L}),    0 <= x <= L.

    The per-bin model probability for the s-space edges ``exc_edges`` (bins ``[s_i, s_{i+1}]``,
    ``x_i = s_i - s_thr``) is the truncated-exponential CDF difference, evaluated in log-space
    with ``-expm1`` (never ``1 - exp``) so the normalizer ``1 - e^{-alpha L}`` and the per-bin
    factor ``1 - e^{-alpha (x_{i+1}-x_i)}`` stay accurate (and grads finite) as ``alpha*L -> 0``::

        log P_i = -alpha x_i + log(-expm1(-alpha dx_i)) - log(-expm1(-alpha L))
        loglike = sum_i exc_counts[i] * log P_i.

    Properties exploited by the inference (see the design doc): (1) EXACT, not asymptotic -- the
    lognormal normalization cancels, so alpha is read off the tail independent of sigma_s^2(mach,b),
    breaking the mach-alpha degeneracy; (2) SHIFT-IMMUNE -- ``s_thr`` and ``s_max`` are measured on
    the same field as ``exc_counts``, so a global s-shift cancels in ``x`` (no mean-1 shift needed);
    (3) GEOMETRY-FREE -- purely marginal counts, no ``rho_g(r;beta)`` grid, hence no cross-grid
    forward bias. ``s_thr`` and ``s_max`` are data-derived constants; only ``theta[2]=alpha`` is
    differentiated (``theta`` is passed whole for interface uniformity with the other blocks). The
    top bin must close at ``s_max`` -- an open-to-infinity bin reintroduces the infinite-tail bias.
    """
    alpha = theta[2]
    x = exc_edges - s_thr  # exceedance-space edges in [0, L]
    L = s_max - s_thr
    x_lo = x[:-1]
    dx = x[1:] - x[:-1]
    log_p = (-alpha * x_lo) + jnp.log(-jnp.expm1(-alpha * dx)) - jnp.log(-jnp.expm1(-alpha * L))
    log_p = jnp.clip(log_p, jnp.log(floor), None)  # guard empty/underflowing bins
    return jnp.sum(exc_counts * log_p)
