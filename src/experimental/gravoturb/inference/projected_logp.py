r"""Analytic log+ projected-band-power inference of the turbulence slope beta (Phase 2).

The forward model is the analytic projected LOG-density band-power times a calibrate-once transfer:

    mu(beta, mach) = A_s(beta, mach) * T_fixed(k)

where ``A_s`` walks the analytic chain (log-density Hermite c_n, NOT the density d_n):

    rho_g(beta)  [P(k)=k^-beta]                         gaussian_correlation_grid
      -> c_n = <s_of_g He_n>                            log_density_hermite_coefficients
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
from jax.scipy.special import gammaln
from jaxtyping import Array, Float

from gravoturb.inference.covariance import _angular_bandpowers_from_xi_rho_2d, _xi_rho_grid
from jaxstro.numerics.quadrature import gauss_hermite_nodes, hermite_coefficients

from gravoturb.theory.log_correlations import (
    log_density_hermite_coefficients,
    gaussianized_xi,
)
from gravoturb.theory.projection import gaussian_correlation_grid, limber_project_slab


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
    c = log_density_hermite_coefficients(mach, b, alpha, n_max, n_quad)
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


# ---------------------------------------------------------------------------------------------------
# Analytic Poisson-shot transfer (N-agnostic forward model for the log+ observable)
# ---------------------------------------------------------------------------------------------------
def _log_plus_counts(n_counts: Float[Array, " m"], n_bar_sky: Float[Array, ""]) -> Float[Array, " m"]:
    r"""Neyrinck Eq.2 log+ on integer counts ``N``: ln(N/nbar) for N>nbar else N/nbar-1 (mean count)."""
    ratio = n_counts / n_bar_sky
    return jnp.where(n_counts > n_bar_sky, jnp.log(jnp.where(ratio > 0, ratio, 1.0)), ratio - 1.0)


def _poisson_logp_moments(
    g_nodes: Float[Array, " q"],
    n_bar_sky: Float[Array, ""],
    s_sigma: Float[Array, ""],
    n_count_max: int,
):
    r"""Conditional log+ mean ``m(g)`` and variance ``v(g)`` of ``N|Sigma ~ Poisson(nbar_sky*Sigma/L)``.

    The projected density is modelled lognormal: ``Sigma/L = exp(s_sigma*g - s_sigma^2/2)`` (mean 1),
    so ``lambda(g) = n_bar_sky * exp(s_sigma*g - s_sigma^2/2)``. Returns ``(m, v)`` over the quadrature
    nodes via the Poisson sum N=0..n_count_max (differentiable in n_bar_sky, s_sigma). High-|g| nodes
    carry ~e^{-g^2/2} weight, so truncating the Poisson tail there is negligible."""
    lam = n_bar_sky * jnp.exp(s_sigma * g_nodes - 0.5 * s_sigma**2)        # (q,)
    n = jnp.arange(n_count_max + 1, dtype=jnp.float64)                     # (m,)
    log_pmf = n[None, :] * jnp.log(lam[:, None]) - lam[:, None] - gammaln(n + 1.0)[None, :]
    pmf = jnp.exp(log_pmf)                                                 # (q, m)
    lp = _log_plus_counts(n, n_bar_sky)                                    # (m,)
    m = jnp.sum(pmf * lp[None, :], axis=1)
    e2 = jnp.sum(pmf * (lp**2)[None, :], axis=1)
    return m, e2 - m**2


def logp_shot_components(
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    depth: Float[Array, ""],
    k_edges: Float[Array, " kp1"],
    n_bar_3d: Float[Array, ""],
    n_max: int = 14,
    n_quad: int = 256,
    n_count_max: int = 800,
) -> tuple[Float[Array, " k"], Float[Array, ""]]:
    r"""Analytic clustering band-powers ``P_clust(k)`` and white shot floor ``W_shot`` for log+ counts.

    Conditional-independence split ``P_A(k) = P_clust(k) + W_shot`` (see the projected-beta-inference
    theory page). ``n_bar_3d`` is the (known) mean stars per 3-D cell; mean projected count
    ``n_bar_sky = n_bar_3d * depth``. The projected-density 2-pt is analytic
    (``xi_Sigma = Limber[xi_rho]``); its marginal is modelled lognormal (the one approximation). The
    Poisson-smoothed log map ``m(Sigma(g))`` is Mehler-expanded through the UNDERLYING Gaussian
    correlation ``rho_g = ln(1 + xi_Sigma/L^2)/s_sigma^2`` (lognormal copula), reusing
    :func:`gaussianized_xi`. Fully differentiable in (beta, mach, ...). At high ``n_bar`` reduces to
    the lognormal-copula log predictor (``m -> ln``, ``W_shot -> 0``)."""
    xi_rho = _xi_rho_grid(shape, beta, mach, b, alpha, n_max, n_quad)
    xi_Sigma = limber_project_slab(xi_rho, depth, los_axis=2)
    L = depth
    sigma_sigma2 = xi_Sigma[0, 0]                                          # projected-density variance
    s_sigma2 = jnp.log1p(sigma_sigma2 / L**2)                             # underlying Gaussian log-var
    s_sigma = jnp.sqrt(s_sigma2)
    rho_g_und = jnp.log1p(xi_Sigma / L**2) / s_sigma2                     # underlying Gaussian corr (=1 @ 0)
    n_bar_sky = n_bar_3d * L

    # clustering: Hermite coeffs of the Poisson-smoothed log map, Mehler-expanded in rho_g_und
    a = hermite_coefficients(
        lambda g: _poisson_logp_moments(g, n_bar_sky, s_sigma, n_count_max)[0], n_max, n_quad
    )
    xi_clust = gaussianized_xi(rho_g_und, a)
    _kc, P_clust, _nm = _angular_bandpowers_from_xi_rho_2d(xi_clust, k_edges)

    # white shot floor: <Var(log+ N | Sigma)>_Sigma via the same Gauss-Hermite rule
    g_nodes, weights = gauss_hermite_nodes(n_quad)
    _m, v = _poisson_logp_moments(g_nodes, n_bar_sky, s_sigma, n_count_max)
    W_shot = jnp.sum(weights * v)
    return P_clust, W_shot


def predict_logp_bandpowers_shot(
    shape: tuple[int, int, int],
    beta: Float[Array, ""],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    depth: Float[Array, ""],
    k_edges: Float[Array, " kp1"],
    n_bar_3d: Float[Array, ""],
    n_max: int = 14,
    n_quad: int = 256,
    n_count_max: int = 800,
) -> Float[Array, " k"]:
    r"""N-agnostic analytic forward model ``mu(k) = P_clust(k) + W_shot`` for the log+ observable.

    The analytic Poisson-shot transfer (:func:`logp_shot_components`): no fitted, beta-dependent
    transfer -> the beta-response stays analytic at any stellar density. Differentiable in
    (beta, mach, ...)."""
    P_clust, W_shot = logp_shot_components(
        shape, beta, mach, b, alpha, depth, k_edges, n_bar_3d, n_max, n_quad, n_count_max
    )
    return P_clust + W_shot
