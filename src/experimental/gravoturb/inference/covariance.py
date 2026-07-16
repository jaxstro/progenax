r"""Data-vector covariance for the gravoturb Fisher/likelihood (Milestone 1).

Data vector d(theta) = [log-density power-spectrum band-powers P_s(k_i), CIC variance
sigma^2_N(R)]. Band-powers (Anna 2026-06-05) have the textbook diagonal Gaussian covariance
``Cov[P_i,P_j] = delta_ij * 2 P_s(k_i)^2 / N_modes(k_i)`` -- exact for a Gaussian field,
scalable with survey volume, differentiable in theta. The analytic covariance is validated
against the realization mock covariance (Hartlap-corrected) in ``mock_covariance``.

Convention: P_s(k) = FFT[xi_s grid](k) = E[|delta_k|^2 / N] (so (1/N) sum_k P_s = Var(s)).
The measured periodogram ``measured_bandpowers`` uses the same ``|fft(s-<s>)|^2 / N``.

JAX-native analytic path; numpy only in the mock/measurement helpers (validation).
"""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from gravoturb.theory.log_correlations import (
    bm19_density_hermite_coefficients,
    bm19_hermite_coefficients,
    gaussianized_xi,
)
from gravoturb.theory.projection import (
    _kmag_grid,
    gaussian_correlation_grid,
    limber_project_grid,
    limber_project_slab,
)


def _bin_by_kmag(values, kmag, k_edges):
    r"""Radially bin a grid quantity by |k| into ``k_edges``: returns (centers, mean, count).

    The bin masks depend only on ``kmag``/``k_edges`` (static), so the per-bin means are
    differentiable in whatever ``values`` depends on. k=0 (DC) is excluded by k_edges[0]>0."""
    centers, means, counts = [], [], []
    for lo, hi in zip(k_edges[:-1], k_edges[1:]):
        mask = (kmag >= lo) & (kmag < hi)
        n = jnp.sum(mask)
        centers.append(jnp.sum(jnp.where(mask, kmag, 0.0)) / n)
        means.append(jnp.sum(jnp.where(mask, values, 0.0)) / n)
        counts.append(n)
    return jnp.stack(centers), jnp.stack(means), jnp.stack(counts)


def power_spectrum_grid(
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    n_max: int = 14,
    n_quad: int = 256,
) -> Float[Array, " nx ny nz"]:
    r"""Analytic log-density power spectrum ``P_s(k) = FFT[xi_s grid]`` (>= 0).

    ``xi_s(r) = sum_{n>=1}(c_n^2/n!) rho_g(r;beta)^n`` is a valid PSD autocovariance, so its
    FFT is a non-negative power spectrum. Differentiable in (mach,b,alpha,beta)."""
    rho_g = gaussian_correlation_grid(shape, beta)
    c = bm19_hermite_coefficients(mach, b, alpha, n_max, n_quad)
    xi_s = gaussianized_xi(rho_g, c)
    return jnp.fft.fftn(xi_s).real


def power_spectrum_bandpowers(
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    k_edges: Float[Array, " kp1"],
    n_max: int = 14,
    n_quad: int = 256,
) -> tuple[Float[Array, " k"], Float[Array, " k"], Float[Array, " k"]]:
    r"""Radially-binned log-density power spectrum: returns (k_centers, P_s band-powers,
    N_modes per bin). The 2-pt block of the Fisher data vector; differentiable in theta."""
    P = power_spectrum_grid(shape, beta, mach, b, alpha, n_max, n_quad)
    kmag = _kmag_grid(shape)
    return _bin_by_kmag(P, kmag, k_edges)


def gaussian_bandpower_covariance(
    P: Float[Array, " k"], n_modes: Float[Array, " k"]
) -> Float[Array, " k k"]:
    r"""DIAGNOSTIC diagonal Gaussian band-power covariance ``2 P_s^2 / N_modes``.

    Exact for a Gaussian field, but **underestimates the true covariance 2-20x** for the
    non-Gaussian (lognormal-like) log-density field -- the excess grows toward small scales
    (connected trispectrum + mode coupling). The Fisher/likelihood therefore use the MOCK
    covariance (Anna 2026-06-05); this is kept as the documented comparison (see
    test_gaussian_bandpower_covariance_underestimates_mock)."""
    return jnp.diag(2.0 * P**2 / n_modes)


def mock_covariance(rows):
    r"""Sample covariance of stacked data-vector rows (numpy, ddof=1) -- the realization mock
    covariance that captures the full non-Gaussian + cross-block structure. Fixed at the
    fiducial theta for the Fisher/HMC (Decision #4)."""
    return np.cov(np.asarray(rows), rowvar=False, ddof=1)


def hartlap_factor(n_real, n_data):
    r"""Anderson-Hartlap unbiased-inverse factor ``(n_real - n_data - 2)/(n_real - 1)``.

    The naive inverse of a sample covariance is biased; multiplying by this factor debiases
    the precision matrix. Requires ``n_real > n_data + 2``; -> 1 as n_real -> inf."""
    return (n_real - n_data - 2.0) / (n_real - 1.0)


def mock_precision(rows):
    r"""Hartlap-corrected precision matrix ``C^{-1}`` from mock data-vector rows (numpy)."""
    rows = np.asarray(rows)
    n_real, n_data = rows.shape
    return hartlap_factor(n_real, n_data) * np.linalg.inv(mock_covariance(rows))


def _kmag_grid_2d(shape: tuple[int, int]) -> Float[Array, " ny nx"]:
    r"""2-D isotropic |k| grid (kx = fftfreq(n)*n per axis), matching
    :func:`gravoturb.validation.measure.measure_angular_bandpowers_2d` so the predicted
    and measured angular band-powers share an identical |k| convention."""
    ny, nx = shape
    ky = jnp.fft.fftfreq(ny) * ny
    kx = jnp.fft.fftfreq(nx) * nx
    return jnp.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)


def _angular_bandpowers_from_xi_rho_2d(xi_Sigma, k_edges):
    r"""Bin the 2-D projected autocovariance into angular band-powers via Wiener-Khinchin.

    ``P_Sigma(k) = fft2(xi_Sigma)(k)`` is the EXACT discrete 2-D power spectrum for the
    same periodogram convention as ``measure_angular_bandpowers_2d`` (derived normalization
    constant = 1; see _v1b_limber_predictor module docstring), provided ``xi_Sigma`` is the
    true autocovariance of the projected map. Returns (k_centers, band-powers, N_modes)."""
    P_2d = jnp.fft.fft2(xi_Sigma).real
    kmag = _kmag_grid_2d(xi_Sigma.shape)
    return _bin_by_kmag(P_2d, kmag, k_edges)


def _xi_rho_grid(shape, beta, mach, b, alpha, n_max, n_quad):
    r"""3-D EXACT BM19 density 2-point ``xi_rho(r) = sum_{n>=1} d_n^2/n! rho_g(r)^n``.

    Uses the DENSITY Hermite coefficients ``d_n = <e^s He_n(g)>`` (the exact Mehler
    expansion), NOT the lognormal-limit ``expm1(xi_s)`` -- the only change vs a naive build
    (A-new1). Differentiable in (beta, mach, b, alpha)."""
    rho_g = gaussian_correlation_grid(shape, beta)
    d = bm19_density_hermite_coefficients(mach, b, alpha, n_max, n_quad)
    return gaussianized_xi(rho_g, d)


def angular_bandpowers_2d_limber(
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    depth: Float[Array, ""],
    k_edges: Float[Array, " kp1"],
    n_max: int = 14,
    n_quad: int = 256,
) -> tuple[Float[Array, " k"], Float[Array, " k"], Float[Array, " k"]]:
    r"""Predicted 2-D projected-DENSITY angular band-powers via the analytic Limber chain.

    Walks the validated forward model (A-new1):
      rho_g(beta) -> d_n (EXACT BM19 density Hermite) -> xi_rho (Mehler, NOT expm1)
      -> ``limber_project_slab`` (depth-L LOS projection) -> ``fft2`` (Wiener-Khinchin,
      derived normalization = 1) -> radially binned by 2-D |k| into ``k_edges``.

    The density 2-point is the exact ``sum_{n>=1} d_n^2/n! rho_g^n`` (copula-faithful);
    this predicts the observable slope to <=0.06 across M in {4,8,16} (vs the lognormal-limit
    ``expm1(xi_s)`` which biases the slope by M). Returns (k_centers, band-powers, N_modes)
    mirroring :func:`power_spectrum_bandpowers`. Differentiable in (beta, mach, b, alpha) and
    the ``depth`` nuisance. This is the CLUSTERING predictor only; compose with
    :func:`add_poisson_shot` for the count observable."""
    xi_rho = _xi_rho_grid(shape, beta, mach, b, alpha, n_max, n_quad)
    xi_Sigma = limber_project_slab(xi_rho, depth, los_axis=2)
    return _angular_bandpowers_from_xi_rho_2d(xi_Sigma, k_edges)


def _angular_bandpowers_from_xi_rho_full(
    shape, beta, mach, b, alpha, k_edges, n_max=14, n_quad=256
):
    r"""Full-depth periodic-projection variant of :func:`angular_bandpowers_2d_limber`
    (uses ``limber_project_grid`` directly). At ``depth = n_los`` the slab predictor reduces
    to this exactly; kept as the reduction-limit reference for tests."""
    xi_rho = _xi_rho_grid(shape, beta, mach, b, alpha, n_max, n_quad)
    xi_Sigma = limber_project_grid(xi_rho, los_axis=2)
    return _angular_bandpowers_from_xi_rho_2d(xi_Sigma, k_edges)


def add_poisson_shot(
    bandpowers: Float[Array, " k"],
    n_bar_sky: Float[Array, ""],
    depth: Float[Array, ""],
) -> Float[Array, " k"]:
    r"""Compose the clustering band-powers into the validated COUNT observable.

    ``P_N(k) = (n_bar_sky / depth)^2 * P_clustering(k) + n_bar_sky``. The clustering term is
    in units of the unit-mean column density Sigma = sum_los rho (mean = depth); the count map
    N = n_bar_sky * Sigma/<Sigma> rescales its band-power by ``(n_bar_sky/depth)^2``, and the
    Poisson white-noise floor is the mean count per sky cell ``n_bar_sky`` (_v1b shot block).
    Kept separate from the clustering predictor so the two pieces stay composable."""
    return (n_bar_sky / depth) ** 2 * bandpowers + n_bar_sky


def measured_bandpowers(s, shape, k_edges):
    r"""Measured periodogram band-powers ``<|fft(s-<s>)|^2 / N>`` of a realization field ``s``,
    binned by |k| into ``k_edges`` (numpy; the mock oracle for ``power_spectrum_bandpowers``)."""
    f = np.asarray(s, dtype=float)
    f = f - f.mean()
    pk = np.abs(np.fft.fftn(f)) ** 2 / f.size
    kmag = np.asarray(_kmag_grid(shape))
    out = np.zeros(len(k_edges) - 1)
    ke = np.asarray(k_edges)
    for i, (lo, hi) in enumerate(zip(ke[:-1], ke[1:])):
        mask = (kmag >= lo) & (kmag < hi)
        out[i] = pk[mask].mean()
    return out
