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


def top_hat_window(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    r"""Fourier transform of a real-space spherical top-hat (radius R, ``x = k R``):
    ``W(x) = 3 (sin x - x cos x) / x^3``, ``W(0) = 1``. Grad-safe at x=0."""
    x_safe = jnp.where(x > 0, x, 1.0)
    w = 3.0 * (jnp.sin(x_safe) - x_safe * jnp.cos(x_safe)) / x_safe**3
    return jnp.where(x > 0, w, 1.0)


def gaussian_window(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    r"""Gaussian smoothing window ``W(x) = exp(-x^2/2)``, ``x = k R``; ``W(0) = 1``."""
    return jnp.exp(-0.5 * x**2)


def smoothed_variance_fraction(
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    R: Float[Array, ""],
    window=top_hat_window,
) -> Float[Array, ""]:
    r"""Fraction of Gaussian variance retained after smoothing at scale ``R`` (cells):
    ``sigma_g^2(R)/sigma_g^2(0) = sum_k P(k) W(kR)^2 / sum_k P(k)``, ``P(k)=k^{-beta}``
    (DC=0). ->1 as R->0, decreasing in R. The single scale R shared by the 2-pt window
    and the CIC cell. Differentiable in ``beta`` and ``R``.
    """
    kmag = _kmag_grid(shape)
    kmag_safe = jnp.where(kmag > 0, kmag, 1.0)
    power = jnp.where(kmag > 0, kmag_safe ** (-beta), 0.0)
    w = window(kmag * R)
    return jnp.sum(power * w**2) / jnp.sum(power)


def limber_project_grid(
    xi_3d: Float[Array, " nx ny nz"], los_axis: int = 2
) -> Float[Array, " nx ny"]:
    r"""Discrete Limber projection: the 2-D projected autocovariance from the 3-D
    autocovariance, summing over the line-of-sight axis. For a column field
    ``Sigma = sum_los f`` the exact periodic identity is
    ``xi_Sigma(r_perp) = N_los * sum_{dlos} xi_f(r_perp, dlos)``.

    Field-agnostic: the choice of which field is projected (linear column density Sigma=∫ρ
    vs log-density s) is a Task-3 / CIC decision; this operator just projects a given 2-pt.
    """
    n_los = xi_3d.shape[los_axis]
    return n_los * jnp.sum(xi_3d, axis=los_axis)


def limber_project_radial(
    xi_fn, r_perp: Float[Array, " ..."], half_depth: Float[Array, ""], n_nodes: int = 512
) -> Float[Array, " ..."]:
    r"""Continuous Limber projection ``w(r_perp) = int_{-L}^{L} xi_fn(sqrt(r_perp^2+l^2)) dl``
    by fixed-node trapezoid quadrature over the line-of-sight depth ``[-half_depth, +half_depth]``.

    ``xi_fn`` maps a separation array to the 3-D correlation. Differentiable in ``r_perp`` and
    any parameters ``xi_fn`` closes over (the depth/distance enters as a nuisance, FK10 §3.5).
    """
    r_perp = jnp.asarray(r_perp)
    ell = jnp.linspace(-half_depth, half_depth, n_nodes)
    sep = jnp.sqrt(r_perp[..., None] ** 2 + ell**2)
    return jnp.trapezoid(xi_fn(sep), ell, axis=-1)
