r"""Projection layer: the analytic Gaussian correlation rho_g(r; beta), smoothing@R,
and (later) the Limber 3D->2D projection for the differentiable predicted statistics.

rho_g(r) is the normalized real-space autocovariance of the Gaussian field with power
spectrum P(k) = k^{-beta} (DC = 0) on the simulator's grid -- the beta-carrier of the
2-point. It enters the Gaussianization series xi_s(r) = sum_n (c_n^2/n!) rho_g(r)^n, so
beta flows analytically (P(k) = exp(-beta ln k) is smooth in beta). Matches
gravoturb_fdf.field.gaussian_random_field's spectrum, so the analytic rho_g equals the
ensemble-mean of gravoturb_fdf.validation.measure.gaussian_correlation_measured.

JAX-native; differentiable in beta.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float


def _kmag_grid(shape: tuple[int, int, int]) -> Float[Array, " nx ny nz"]:
    r"""Isotropic wavenumber magnitude |k| on the full grid (integer modes), matching
    ``gaussian_random_field`` (kx = fftfreq(n)*n)."""
    kx = jnp.fft.fftfreq(shape[0]) * shape[0]
    ky = jnp.fft.fftfreq(shape[1]) * shape[1]
    kz = jnp.fft.fftfreq(shape[2]) * shape[2]
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    return jnp.sqrt(KX**2 + KY**2 + KZ**2)


def gaussian_correlation_grid(
    shape: tuple[int, int, int], beta: Float[Array, ""]
) -> Float[Array, " nx ny nz"]:
    r"""Normalized autocovariance rho_g(r) for P(k) = k^{-beta} (DC=0) on the grid.

    rho_g = IFFT[P] / IFFT[P](0); the normalization cancels, so rho_g[0,0,0] = 1 and the
    result is independent of FFT/amplitude conventions (it equals the ensemble-mean of the
    measured autocovariance of ``gaussian_random_field``). Differentiable in ``beta`` via
    a grad-safe ``where`` at the DC mode.
    """
    kmag = _kmag_grid(shape)
    kmag_safe = jnp.where(kmag > 0, kmag, 1.0)  # avoid 0^{-beta} / ln(0) in the dead branch
    power = jnp.where(kmag > 0, kmag_safe ** (-beta), 0.0)
    xi = jnp.fft.ifftn(power).real
    return xi / xi[0, 0, 0]
