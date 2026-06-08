r"""Analytic log+ projected-band-power inference of the turbulence slope beta (Phase 2).

The forward model is the analytic projected LOG-density band-power times a calibrate-once transfer:

    mu(beta, mach) = A_s(beta, mach) * T_fixed(k)

where ``A_s`` walks the analytic chain (log-density Hermite c_n, NOT the density d_n):

    rho_g(beta)  [P(k)=k^-beta]                         gaussian_correlation_grid
      -> c_n = <s_of_g He_n>                            bm19_hermite_coefficients
      -> xi_s(r) = sum_{n>=1} c_n^2/n! rho_g(r)^n       gaussianized_xi
      -> xi_Sigma(r_perp) = Limber_slab(xi_s; depth)    limber_project_slab
      -> P_s(k) = bin( fft2(xi_Sigma) )                 _angular_bandpowers_from_xi_rho_2d

and ``T_fixed(k)`` is a beta-independent per-bin multiplicative transfer calibrated ONCE at a fixed
fiducial theta (truth-independent). Phase-0 (validation/_d03,_d05) established that the log+ observable
makes this transfer beta-stable (~5% CV) while keeping the beta-RESPONSE purely analytic -- avoiding
the emulated-slope fragility that mis-calibrated the rank-G approach (whose transfer is beta-dependent,
39-91% CV). The observable this predicts the mean of is
``measure_angular_bandpowers_2d(log_plus(project_counts_los(counts)), k_edges)``.

JAX-native; differentiable in beta (and mach) through the analytic chain. ``T`` is data (a constant).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gravoturb_fdf.inference.covariance import _angular_bandpowers_from_xi_rho_2d
from gravoturb_fdf.theory.gaussianization import bm19_hermite_coefficients, gaussianized_xi
from gravoturb_fdf.theory.projection import gaussian_correlation_grid, limber_project_slab


def analytic_logdensity_bandpowers(
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    depth: Float[Array, ""],
    k_edges: Float[Array, " kp1"],
    n_max: int = 14,
    n_quad: int = 256,
) -> Float[Array, " k"]:
    r"""Analytic projected LOG-density angular band-powers ``A_s`` (no transfer, no shot).

    Uses the log-density Hermite coefficients ``c_n`` (the copula map ``s`` itself), so the 2-point is
    ``xi_s = sum_{n>=1} c_n^2/n! rho_g^n`` -- the near-Gaussian log-density field whose band-power slope
    carries beta with gain ~1 (Phase-0 D01). Differentiable in (beta, mach, b, alpha) and depth."""
    rho_g = gaussian_correlation_grid(shape, beta)
    c = bm19_hermite_coefficients(mach, b, alpha, n_max, n_quad)
    xi_s = gaussianized_xi(rho_g, c)
    xi_Sigma = limber_project_slab(xi_s, depth, los_axis=2)
    _kc, P, _nm = _angular_bandpowers_from_xi_rho_2d(xi_Sigma, k_edges)
    return P


def calibrate_transfer(
    fiducial_rows: Float[Array, " r k"], a_s_fid: Float[Array, " k"]
) -> Float[Array, " k"]:
    r"""Calibrate the per-bin transfer ``T(k) = <observable band-power>_fid / A_s(theta_fid)``.

    ``fiducial_rows`` are measured log+ observable band-power vectors from ``n_real`` simulator
    realizations at a FIXED fiducial theta (independent of any trial truth -> SBC-valid); ``a_s_fid``
    is the analytic :func:`analytic_logdensity_bandpowers` at the same fiducial. By construction
    ``a_s_fid * T == mean(rows)``, so ``mu(theta_fid) = A_s(theta_fid) * T`` matches the observable
    mean at the fiducial; Phase-0 (D05) showed ``T`` is beta-stable to ~5%, so the beta-response stays
    analytic. Pure (no simulator import); the caller supplies the truth-independent fiducial rows."""
    return jnp.mean(fiducial_rows, axis=0) / a_s_fid


def predict_logp_bandpowers(
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    depth: Float[Array, ""],
    k_edges: Float[Array, " kp1"],
    transfer: Float[Array, " k"],
    n_max: int = 14,
    n_quad: int = 256,
) -> Float[Array, " k"]:
    r"""Forward model ``mu(beta, mach) = A_s(beta, mach) * transfer`` for the log+ observable.

    ``transfer`` is the calibrate-once, beta-independent per-bin transfer ``T_fixed(k)`` (length
    ``len(k_edges)-1``); it is a constant w.r.t. the inferred parameters, so the beta-response stays
    purely analytic. Differentiable in (beta, mach, ...)."""
    return analytic_logdensity_bandpowers(
        shape, beta, mach, b, alpha, depth, k_edges, n_max, n_quad
    ) * transfer


def precompute_a_s_table(
    shape: tuple[int, int, int],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    depth: Float[Array, ""],
    k_edges: Float[Array, " kp1"],
    beta_nodes: Float[Array, " n"],
    n_max: int = 14,
    n_quad: int = 256,
) -> Float[Array, " n k"]:
    r"""Precompute the analytic ``A_s(beta)`` band-power table on ``beta_nodes`` (at fixed mach,...).

    A one-time precompute so NUTS interpolates instead of re-running the 3-D FFT chain per leapfrog.
    Crucially ``A_s`` is a SMOOTH, DETERMINISTIC function of beta (no Monte-Carlo noise), so
    interpolating this table preserves the beta-slope -- the opposite of emulating a noisy simulation
    mean (the v2h failure). Returns shape ``(len(beta_nodes), len(k_edges)-1)``."""
    rows = [
        analytic_logdensity_bandpowers(shape, bn, mach, b, alpha, depth, k_edges, n_max, n_quad)
        for bn in beta_nodes
    ]
    return jnp.stack(rows, axis=0)


def interp_logp_bandpowers(
    beta: Float[Array, ""],
    beta_nodes: Float[Array, " n"],
    a_s_table: Float[Array, " n k"],
    transfer: Float[Array, " k"],
) -> Float[Array, " k"]:
    r"""Differentiable forward model from the precomputed table: ``interp(A_s; beta) * transfer``.

    Per-bin linear interpolation of the smooth analytic ``A_s`` table at ``beta`` (differentiable in
    ``beta`` via :func:`jax.numpy.interp`), times the fixed transfer. Because the table is noise-free
    and smooth, the interpolated beta-response matches the direct model to the grid rate (<<1% on a
    ~100-node grid) -- see ``test_analytic_emulator_matches_direct_model``."""
    a_s = jax.vmap(lambda col: jnp.interp(beta, beta_nodes, col), in_axes=1, out_axes=0)(a_s_table)
    return a_s * transfer


def logp_loglike(
    data: Float[Array, " k"],
    beta: Float[Array, ""],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    depth: Float[Array, ""],
    shape: tuple[int, int, int],
    k_edges: Float[Array, " kp1"],
    transfer: Float[Array, " k"],
    precision: Float[Array, " k k"],
    n_max: int = 14,
    n_quad: int = 256,
) -> Float[Array, ""]:
    r"""Gaussian log-likelihood of the log+ band-powers: ``-1/2 r^T Cinv r``, ``r = mu(beta) - data``.

    ``mu`` is :func:`predict_logp_bandpowers`; ``precision`` is the FIXED-fiducial Hartlap-corrected
    inverse covariance (truth-independent -> no log|C| term -> SBC-valid). Phase-0 D02 showed the log+
    band-powers are near-Gaussian (skew->0), justifying the Gaussian form. Differentiable in ``beta``
    (and ``mach``, ...); the data-misfit is stationary at the truth on noiseless data."""
    mu = predict_logp_bandpowers(shape, beta, mach, b, alpha, depth, k_edges, transfer, n_max, n_quad)
    r = mu - data
    return -0.5 * r @ (precision @ r)
