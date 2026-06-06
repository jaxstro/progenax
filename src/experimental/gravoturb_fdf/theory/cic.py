r"""Counts-in-cells (CIC): the differentiable stellar observable for gravoturb_fdf.

The simulator places stars proportional to the LINEAR density rho (field/sampling.py:
``p_smooth proportional to rho``) -- a Cox / doubly-stochastic Poisson process with
intensity ``lambda(x) = n_bar * rho_tilde(x)``, ``rho_tilde = rho/<rho>`` (mean 1). The
law of total variance then gives the standard counts-in-cells variance (design doc Sec.4)

    sigma^2_N(R) = N_bar + N_bar^2 * xi_bar(R) ,    xi_bar(R) = Var(rho_tilde_cell) ,

where ``xi_bar(R)`` is the cell-averaged 2-point of the *linear* density (NOT the log-
density xi_s, which is the separate beta-carrier block). The cell scale ``R`` regularizes
the alpha<=2 fat tail: cell-averaging IS a smoothing, so ``Var(rho_tilde_cell)`` is finite
for finite R even though ``<rho^2>`` diverges (Decision #1: R, no hard density cap).

Route A (Anna 2026-06-05): the moment ``xi_bar(R)`` is computed from the EXACT marginal-
induced linear-rho Gaussianization series ``xi_rho(r) = sum_{n>=1} (d_n^2/n!) rho_g(r)^n``
with ``d_n = <rho_tilde(g) He_n(g)>`` -- the SAME Hermite machinery as xi_s, fed the linear
map ``exp(s_of_g)`` instead of ``s_of_g``. The single watch item (vs xi_s) is that the heavy
tail makes the series converge slower; convergence in (n_max, n_quad) is measured against the
oracle (AC13), mirroring AC11. The count distribution P(N) uses Route B (theory/cic Task 3.3).

JAX-native; differentiable in (mach, b, alpha, beta).
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from gravoturb_fdf.theory.gaussianization import (
    bm19_hermite_coefficients,
    gaussianized_xi,
    hermite_coefficients,
    s_of_g,
)
from gravoturb_fdf.theory.pdf import bm19_volume_pdf
from gravoturb_fdf.theory.projection import (
    _kmag_grid,
    gaussian_correlation_grid,
    top_hat_window,
)


def _windowed_series_variance(
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    R: Float[Array, ""],
    coeffs: Float[Array, " n"],
    window,
) -> Float[Array, ""]:
    r"""Windowed (cell-averaged) variance of a Gaussianization series at scale ``R`` (cells).

    Builds the autocovariance grid ``xi(r) = sum_{n>=1}(coeffs_n^2/n!) rho_g(r;beta)^n`` (a
    valid PSD autocovariance: powers of a correlation are PSD) and returns
    ``(1/N) sum_{k!=0} FFT[xi](k) W(kR)^2`` -- the variance of the mapped field smoothed at
    R, with the k=0 mode excluded (cell-to-cell fluctuations about the mean). Differentiable
    in ``beta`` (via rho_g) and in ``coeffs`` (via theta). Shared by the linear-rho clustering
    term and the smoothed log-density variance (same machinery, different copula map).
    """
    rho_g = gaussian_correlation_grid(shape, beta)
    xi = gaussianized_xi(rho_g, coeffs)
    power = jnp.fft.fftn(xi).real
    kmag = _kmag_grid(shape)
    w2 = jnp.where(kmag > 0, window(kmag * R) ** 2, 0.0)
    return jnp.sum(power * w2) / power.size


def linear_hermite_coefficients(
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    n_max: int,
    n_quad: int = 256,
) -> Float[Array, " n"]:
    r"""Hermite coefficients ``d_n = <rho_tilde(g) He_n(g)>`` of the LINEAR mean-1 density.

    Route A: the linear map ``rho_tilde = exp(s_of_g)`` (``<rho_tilde>=1`` by the rho0
    convention) fed through the SAME quadrature as :func:`bm19_hermite_coefficients`. The
    n>=1 coefficients carry the linear 2-point ``xi_rho(r) = sum (d_n^2/n!) rho_g(r)^n``;
    ``d_0 = <rho_tilde> ~ 1`` is dropped by :func:`gaussianized_xi`. Differentiable in
    (mach, b, alpha). The heavy tail (alpha<=2) makes ``sum d_n^2/n! = Var(rho)`` diverge in
    the continuum, so unsmoothed ``xi_rho(0)`` is quadrature-dependent -- but the cell-
    averaged (smoothed) ``xi_bar_rho(R)`` suppresses the high-k tail and stays robust.
    """
    return hermite_coefficients(
        lambda g: jnp.exp(s_of_g(g, mach, b, alpha)), n_max, n_quad
    )


def cell_averaged_xi_rho(
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    R: Float[Array, ""],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    n_max: int = 16,
    n_quad: int = 256,
    window=top_hat_window,
) -> Float[Array, ""]:
    r"""CIC clustering term ``xi_bar_rho(R) = Var(rho_tilde smoothed at scale R)`` (Route A).

    Builds the linear-density autocovariance grid ``xi_rho(r) = sum_{n>=1}(d_n^2/n!)
    rho_g(r;beta)^n`` (a valid PSD autocovariance: powers of a correlation are PSD), then
    returns its windowed variance ``(1/N) sum_{k!=0} P_rho(k) W(kR)^2`` with
    ``P_rho = FFT[xi_rho]``. The k=0 mode is excluded (cell-to-cell fluctuations about the
    mean). ``R`` in grid cells, shared with the 2-pt window and the CIC cell (Decision #1).
    Differentiable in (mach, b, alpha) via ``d_n`` and in ``beta`` via ``rho_g``.
    """
    d = linear_hermite_coefficients(mach, b, alpha, n_max, n_quad)
    return _windowed_series_variance(shape, beta, R, d, window)


def smoothed_log_variance(
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    R: Float[Array, ""],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    n_max: int = 16,
    n_quad: int = 256,
    window=top_hat_window,
) -> Float[Array, ""]:
    r"""Exact smoothed LOG-density variance ``sigma_s^2(R) = Var(s smoothed at R)`` (Route B).

    The log-map analog of :func:`cell_averaged_xi_rho`: the windowed variance of the
    log-density 2-point ``xi_s(r) = sum_{n>=1}(c_n^2/n!) rho_g(r;beta)^n``,
    ``c_n = <s_of_g He_n>``. ``-> sigma_s_squared(mach,b)`` as ``R->0`` and decreases with R.
    This sets the effective Mach of the reduced-variance BM19 smoothed PDF (:func:`smoothed_pdf`).
    Differentiable in (mach, b, alpha, beta).
    """
    c = bm19_hermite_coefficients(mach, b, alpha, n_max, n_quad)
    return _windowed_series_variance(shape, beta, R, c, window)


def effective_mach(
    sigma_s_sq_R: Float[Array, ""], b: Float[Array, ""]
) -> Float[Array, ""]:
    r"""Effective Mach number reproducing a target log-variance: ``ln(1+b^2 M_eff^2) =
    sigma_s^2(R)`` -> ``M_eff = sqrt(exp(sigma_s^2(R)) - 1)/b`` (BM19 Eq.1 inverted).

    Lets the smoothed BM19 PDF be re-used at the reduced variance ``sigma_s^2(R)`` while
    keeping (b, alpha): smoothing shrinks the lognormal width and pulls s_t inward
    self-consistently. ``M_eff -> mach`` as ``R->0``. ``expm1`` keeps small-R accuracy.
    """
    return jnp.sqrt(jnp.expm1(sigma_s_sq_R)) / b


def smoothed_pdf(
    s: Float[Array, " m"],
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    R: Float[Array, ""],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    n_max: int = 16,
    n_quad: int = 256,
    window=top_hat_window,
) -> Float[Array, " m"]:
    r"""Route B smoothed-density volume PDF ``p_R(s)`` (reduced-variance BM19).

    The cell-averaged (scale-R) log-density follows a BM19 PDF at the reduced log-variance
    ``sigma_s^2(R)`` (:func:`smoothed_log_variance`), i.e. ``bm19_volume_pdf`` evaluated at
    the effective Mach ``M_eff(R)`` with (b, alpha) preserved. ``int p_R ds = 1``; as
    ``R->0`` it recovers the full unsmoothed BM19 PDF. This sources the compound-Poisson
    count distribution P(N) (Task 3.3). **Approximation:** keeping the BM19 tail shape at the
    reduced variance models the smoothed tail by its variance reduction only; smoothing also
    suppresses the rarest peaks beyond that, so the high-s tail is an over-estimate at large R
    (documented; the moment xi_bar(R) for sigma^2_N uses the exact Route A series instead).
    Differentiable in (mach, b, alpha, beta).
    """
    sigma_s_sq_R = smoothed_log_variance(shape, beta, R, mach, b, alpha, n_max, n_quad, window)
    mach_eff = effective_mach(sigma_s_sq_R, b)
    return bm19_volume_pdf(s, mach_eff, b, alpha)


def cic_variance(
    n_bar: Float[Array, ""], xi_bar: Float[Array, ""]
) -> Float[Array, ""]:
    r"""Counts-in-cells variance ``sigma^2_N = N_bar + N_bar^2 xi_bar`` (design Sec.4).

    ``N_bar`` is the mean count per cell (survey-set); ``xi_bar`` is the cell-averaged
    linear-density 2-point ``Var(rho_tilde_cell)`` at the cell scale R (from
    :func:`cell_averaged_xi_rho`). ``xi_bar=0`` recovers the Poisson floor (var=mean);
    ``xi_bar>0`` is the clustering over-dispersion. Differentiable in both arguments.
    """
    return n_bar + n_bar**2 * xi_bar
