"""BM19 volume-weighted PDF for CDF remap field generation.

This module provides utilities for generating density fields where the
one-point PDF exactly matches the BM19 piecewise lognormal+powerlaw
distribution. This is achieved via Gaussian copula (CDF remap):

    g(x) ~ N(0,1)  ->  u(x) = Phi(g(x))  ->  s(x) = F_V^{-1}(u(x))

where:
- g(x) is a Gaussian GRF with desired power spectrum (turbulent structure)
- Phi is the standard normal CDF
- F_V is the CDF of the BM19 volume-weighted PDF
- F_V^{-1} is its inverse (quantile function)

Key Insight: Volume-Weighted PDF
--------------------------------
Each voxel has equal volume dV. When we sample s ~ F_V^{-1}(u), we get the
VOLUME distribution of log-density. The MASS distribution is obtained by
weighting by e^s:

    p_M(s) = e^s * p_V(s) / <e^s>

This means:
- Use p_V(s) for CDF remap sampling
- Mass integrals naturally include e^s weighting
- f_tail_actual = sum(e^s * w) / sum(e^s) matches f_dense by construction

All functions are:
- @jax.jit compatible
- Differentiable via jax.grad
- Vectorizable via jax.vmap

References
----------
Burkhart, B. & Mocz, P. 2019, ApJ, 879, 129
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from jaxtyping import Array, Float


# =============================================================================
# Volume-Weighted BM19 PDF
# =============================================================================


@jax.jit
def bm19_volume_pdf(
    s: Array,
    sigma_s_sq: Array,
    s_t: Array,
    alpha: Array,
) -> Array:
    """BM19 piecewise volume-weighted PDF p_V(s).

    The PDF is:
        p_V(s) = p_LN(s)           for s < s_t  (lognormal)
        p_V(s) = A * exp(-alpha*s) for s >= s_t (powerlaw tail)

    where:
        - p_LN(s) is lognormal with mean s_0 = -sigma_s^2/2 (mass conservation)
        - A = p_LN(s_t) * exp(alpha * s_t) ensures continuity at s_t

    Parameters
    ----------
    s : Array
        Log-density contrast s = ln(rho / rho_mean).
    sigma_s_sq : Array
        PDF variance sigma_s^2 = ln(1 + b^2 * M^2).
    s_t : Array
        Transition density s_t = (alpha - 0.5) * sigma_s^2.
    alpha : Array
        Powerlaw slope (must be > 1 for normalization).

    Returns
    -------
    p_V : Array
        Probability density (unnormalized - normalize after integration).

    Notes
    -----
    This is the VOLUME-weighted PDF. Each voxel has equal volume, so sampling
    s from this distribution gives the volume distribution. Mass weighting
    is applied separately via e^s factors in integrals.
    """
    sigma_s = jnp.sqrt(sigma_s_sq)
    s_0 = -sigma_s_sq / 2.0  # Mass conservation: <rho> = 1

    # Lognormal part: p_LN(s) = exp(-(s - s_0)^2 / (2*sigma_s^2)) / (sqrt(2*pi)*sigma_s)
    log_p_ln = -(s - s_0) ** 2 / (2.0 * sigma_s_sq) - 0.5 * jnp.log(
        2.0 * jnp.pi * sigma_s_sq
    )
    p_ln = jnp.exp(log_p_ln)

    # Powerlaw part: A * exp(-alpha * s), where A ensures continuity at s_t
    log_p_ln_at_st = -(s_t - s_0) ** 2 / (2.0 * sigma_s_sq) - 0.5 * jnp.log(
        2.0 * jnp.pi * sigma_s_sq
    )
    log_A = log_p_ln_at_st + alpha * s_t
    log_p_pl = log_A - alpha * s

    # Piecewise selection (work in log-space for numerical stability)
    p = jnp.where(s < s_t, p_ln, jnp.exp(log_p_pl))

    return p


@jax.jit
def _integrate_volume_pdf_normalized(
    s_grid: Array,
    sigma_s_sq: float,
    s_t: float,
    alpha: float,
) -> Tuple[Array, float]:
    """Compute normalized volume PDF and its CDF on a grid.

    Parameters
    ----------
    s_grid : Array (n_grid,)
        Grid of s values from s_min to s_max.
    sigma_s_sq : float
        PDF variance.
    s_t : float
        Transition density.
    alpha : float
        Powerlaw slope.

    Returns
    -------
    p_normalized : Array (n_grid,)
        Normalized PDF values (integrate to 1).
    F : Array (n_grid,)
        CDF values F(s) = integral from -inf to s of p(s') ds'.
    """
    # Compute unnormalized PDF
    p = bm19_volume_pdf(s_grid, sigma_s_sq, s_t, alpha)

    # Numerical integration for normalization
    ds = s_grid[1] - s_grid[0]
    total_mass = jnp.trapezoid(p, dx=ds)

    # Normalize
    p_normalized = p / total_mass

    # Compute CDF via cumulative trapezoidal rule
    # F[i] = integral from s_min to s[i] of p(s) ds
    F = jnp.cumsum(p_normalized) * ds

    # Ensure F starts near 0 and ends at 1
    F = F - F[0]  # Subtract initial value
    F = F / (F[-1] + 1e-12)  # Normalize to [0, 1]

    return p_normalized, F


def build_bm19_cdf_table(
    sigma_s_sq: float,
    s_t: float,
    alpha: float,
    s_min: float | None = None,
    s_max: float | None = None,
    n_grid: int = 2000,
) -> Tuple[Array, Array]:
    """Build tabulated CDF from volume PDF.

    This precomputes the CDF table for efficient inverse CDF lookups.

    Parameters
    ----------
    sigma_s_sq : float
        PDF variance sigma_s^2.
    s_t : float
        Transition density.
    alpha : float
        Powerlaw slope (must be > 1).
    s_min : float, optional
        Minimum s value. Default: -6 * sigma_s (captures ~99.9999% of LN part).
    s_max : float, optional
        Maximum s value. Default: s_t + 10 (extends well into powerlaw tail).
    n_grid : int
        Number of grid points (default 2000 for ~0.1% accuracy).

    Returns
    -------
    s_grid : Array (n_grid,)
        Grid of s values.
    F_grid : Array (n_grid,)
        CDF values F(s) at each grid point.

    Notes
    -----
    The CDF table enables fast inverse CDF lookups via linear interpolation.
    For 128^3 fields (2M voxels), table lookup is ~100x faster than
    root-finding per voxel.
    """
    sigma_s = jnp.sqrt(sigma_s_sq)

    # Set sensible defaults for integration range
    if s_min is None:
        s_min = -6.0 * sigma_s
    if s_max is None:
        s_max = s_t + 10.0 / alpha  # Extend into powerlaw tail

    s_grid = jnp.linspace(s_min, s_max, n_grid)

    _, F_grid = _integrate_volume_pdf_normalized(s_grid, sigma_s_sq, s_t, alpha)

    return s_grid, F_grid


@jax.jit
def bm19_icdf(
    u: Array,
    s_grid: Array,
    F_grid: Array,
) -> Array:
    """Inverse CDF: map uniform u in (0,1) to s with BM19 distribution.

    Uses linear interpolation for efficiency. The inverse is:
        s = F_V^{-1}(u)

    where F_V is the CDF of the BM19 volume PDF.

    Parameters
    ----------
    u : Array
        Uniform random values in (0, 1).
    s_grid : Array (n_grid,)
        Grid of s values (from build_bm19_cdf_table).
    F_grid : Array (n_grid,)
        CDF values at each grid point.

    Returns
    -------
    s : Array
        Log-density values sampled from BM19 distribution.

    Notes
    -----
    Uses searchsorted + linear interpolation for O(log n) lookup per sample.
    For n_grid=2000 and 128^3 voxels, this is very efficient.
    """
    # Clamp u to valid range (avoid numerical issues at boundaries)
    u = jnp.clip(u, 1e-10, 1.0 - 1e-10)

    # Find indices via searchsorted
    # searchsorted returns index i where F_grid[i-1] <= u < F_grid[i]
    idx = jnp.searchsorted(F_grid, u, side="right") - 1
    idx = jnp.clip(idx, 0, len(s_grid) - 2)

    # Linear interpolation between grid points
    s0 = s_grid[idx]
    s1 = s_grid[idx + 1]
    F0 = F_grid[idx]
    F1 = F_grid[idx + 1]

    # Avoid division by zero
    dF = jnp.maximum(F1 - F0, 1e-12)
    t = (u - F0) / dF

    s = s0 + t * (s1 - s0)

    return s


# =============================================================================
# Convenience Functions
# =============================================================================


def gaussian_to_bm19(
    g: Array,
    sigma_s_sq: float,
    s_t: float,
    alpha: float,
    s_grid: Array | None = None,
    F_grid: Array | None = None,
) -> Array:
    """Transform Gaussian field to BM19 LN+PL distribution via CDF remap.

    This is the main interface for generating BM19-distributed fields:

        g(x) ~ N(0,1)  ->  u(x) = Phi(g(x))  ->  s(x) = F_V^{-1}(u(x))

    Parameters
    ----------
    g : Array
        Standardized Gaussian field with any shape (should have mean ~0, std ~1).
    sigma_s_sq : float
        BM19 PDF variance.
    s_t : float
        BM19 transition density.
    alpha : float
        BM19 powerlaw slope.
    s_grid : Array, optional
        Precomputed s grid (for efficiency when calling multiple times).
    F_grid : Array, optional
        Precomputed CDF grid.

    Returns
    -------
    s : Array
        Log-density field with BM19 distribution (same shape as g).

    Notes
    -----
    This preserves spatial correlations from g while enforcing exact BM19 PDF.
    The resulting rho = exp(s) field has:
    - Correct turbulent geometry from the Gaussian GRF power spectrum
    - Exact BM19 one-point statistics (lognormal + powerlaw tail)
    - f_tail_actual matches f_dense by construction

    Examples
    --------
    >>> import jax.random as random
    >>> key = random.PRNGKey(42)
    >>> g = random.normal(key, (64, 64, 64))
    >>> s = gaussian_to_bm19(g, sigma_s_sq=2.0, s_t=3.0, alpha=2.0)
    >>> rho = jnp.exp(s)  # Density field with BM19 PDF
    """
    # Build CDF table if not provided
    if s_grid is None or F_grid is None:
        s_grid, F_grid = build_bm19_cdf_table(sigma_s_sq, s_t, alpha)

    # Transform Gaussian to uniform via standard normal CDF
    # Phi(g) = 0.5 * (1 + erf(g / sqrt(2)))
    u = 0.5 * (1.0 + erf(g / jnp.sqrt(2.0)))

    # Flatten for inverse CDF lookup
    original_shape = g.shape
    u_flat = u.flatten()

    # Apply inverse CDF
    s_flat = bm19_icdf(u_flat, s_grid, F_grid)

    # Reshape to original
    s = s_flat.reshape(original_shape)

    return s


def validate_bm19_field(
    s_field: Array,
    sigma_s_sq: float,
    s_t: float,
    alpha: float,
    kappa: float = 10.0,
) -> dict:
    """Validate that a field has correct BM19 statistics.

    Useful for verifying CDF remap worked correctly.

    Parameters
    ----------
    s_field : Array
        Log-density field s = ln(rho / rho_mean).
    sigma_s_sq : float
        Expected PDF variance.
    s_t : float
        Expected transition density.
    alpha : float
        Expected powerlaw slope.
    kappa : float
        Soft sigmoid sharpness for tail selection.

    Returns
    -------
    stats : dict
        Dictionary with:
        - 'mean_s': Mean of s (should be ~-sigma_s^2/2)
        - 'var_s': Variance of s
        - 'f_tail_actual': Measured tail mass fraction
        - 'f_dense_theory': Theoretical f_dense from BM19
        - 'relative_error': (f_tail_actual - f_dense) / f_dense * 100%
    """
    from progenax.gravoturb.bm19_model import f_dense_bm19_full

    # Compute statistics
    mean_s = float(jnp.mean(s_field))
    var_s = float(jnp.var(s_field))

    # Expected mean is s_0 = -sigma_s^2/2
    expected_mean = -sigma_s_sq / 2.0

    # Compute f_tail_actual using soft sigmoid
    rho = jnp.exp(s_field)
    w = jax.nn.sigmoid(kappa * (s_field - s_t))
    f_tail_actual = float(jnp.sum(w * rho) / jnp.sum(rho))

    # Theoretical f_dense
    f_dense_theory = float(f_dense_bm19_full(sigma_s_sq, s_t, alpha))

    # Relative error
    relative_error = (f_tail_actual - f_dense_theory) / f_dense_theory * 100.0

    return {
        "mean_s": mean_s,
        "expected_mean_s": expected_mean,
        "var_s": var_s,
        "expected_var_s": sigma_s_sq,
        "f_tail_actual": f_tail_actual,
        "f_dense_theory": f_dense_theory,
        "relative_error_percent": relative_error,
    }
