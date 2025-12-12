"""Parmentier & Pasquali 2020 dense-gas SFR framework.

Implements the magnification factor zeta for predicting dense-gas star
formation efficiency from density profile geometry.

All functions are:
- @jax.jit compatible
- Differentiable via jax.grad
- Vectorizable via jax.vmap

Key Equations
-------------
- PP20 Eq. 6: zeta(p) = (3 - p) / (2.6 - 2*p)^(3/2)
- PP20 Eq. 8: zeta with finite core via numerical integration
- zeta_fdf_direct: Freefall-weighted zeta measurement from 3D field

Domain Warnings
---------------
The analytic zeta(p) formula (PP20 Eq. 6) has a singularity at p = 1.3.

- p in [0, 1): Reliable
- p in [1, 1.3): Use with caution
- p >= 1.3: Singularity; use zeta_fdf_direct() instead

For typical alpha values (alpha <= 3, i.e., p >= 1), always prefer
zeta_fdf_direct() which measures zeta from the actual density field.

References
----------
Parmentier, G. & Pasquali, A. 2020, ApJ, 903, 56
Tan, J. C., Krumholz, M. R., & McKee, C. F. 2006, ApJL, 641, L121
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# =============================================================================
# Analytic Magnification Factor
# =============================================================================


@jax.jit
def magnification_factor(p: Array) -> Array:
    """PP20 Eq. 6: Analytic zeta(p) for pure power-law profiles.

    zeta(p) = (3 - p) / (2.6 - 2*p)^(3/2)

    For density profiles rho ~ r^(-p), zeta quantifies the SFR boost
    compared to uniform density (top-hat). Centrally concentrated profiles
    have zeta > 1 because inner regions have shorter freefall times.

    Parameters
    ----------
    p : Array
        Radial density profile slope (rho ~ r^{-p}).

        **For reliable results, use only with p < 1.0.**

        For p >= 1 (typical in star formation), use zeta_fdf_direct()
        instead.

    Returns
    -------
    zeta : Array
        Magnification factor (clamped to >= 1).

    Warnings
    --------
    DOMAIN WARNING: This formula is only valid for p < 1.3.

    - p in [0, 1): Physically meaningful, zeta(p) > 1
    - p in [1, 1.3): Mathematically defined but unreliable
    - p >= 1.3: Singularity (denominator -> 0); produces arbitrary values
    - p >= 2: Undefined and should NEVER be used

    For p >= 1 (i.e., alpha <= 3), always use zeta_fdf_direct() instead.

    Notes
    -----
    This function is retained for reference and sanity checks at small p,
    but should NOT be used in production for typical alpha values.

    References
    ----------
    Parmentier & Pasquali 2020, ApJ, 903, 56, Equation 6
    """
    # Clamp denominator to avoid division by zero near p = 1.3
    # Note: This produces arbitrary values for p >= 1.3; see docstring warning.
    denom = jnp.maximum(2.6 - 2.0 * p, 1e-6)
    zeta = (3.0 - p) / denom**1.5

    # Clamp to physical range: zeta >= 1 (top-hat is minimum)
    return jnp.maximum(zeta, 1.0)


@jax.jit
def magnification_factor_with_core(
    p: Array,
    r_c_over_R: Array,
    n_radial_points: int = 100,
) -> Array:
    """PP20 zeta with finite central core via numerical integration.

    For profiles:
        rho(r) = rho_c / [1 + (r/r_c)^2]^(p/2)

    which transitions from rho ~ rho_c for r << r_c to rho ~ r^(-p) for r >> r_c.

    This avoids the singularity at p = 1.3 by integrating over a realistic
    cored profile.

    The magnification factor is computed by numerical integration:

        zeta = [int_0^R rho(r)^(3/2) 4*pi*r^2 dr] / [M * <rho>^(1/2)]

    where M = int_0^R rho(r) 4*pi*r^2 dr is the total mass.

    Parameters
    ----------
    p : Array
        Profile slope (0 to ~2.5).
    r_c_over_R : Array
        Core radius as fraction of outer radius (0 to 1).
    n_radial_points : int
        Integration resolution (default 100).

        n_radial_points = 100 is sufficient for ~1% accuracy for typical
        p and r_c/R. Increase if higher precision is needed.

    Returns
    -------
    zeta : Array
        Magnification factor (>= 1).

    Notes
    -----
    Limiting cases:
    - r_c/R -> 0: approaches pure power-law zeta(p) for p < 2
    - r_c/R -> 1: approaches zeta = 1 (uniform density)
    - p -> 0: zeta = 1 regardless of r_c/R

    References
    ----------
    Parmentier & Pasquali 2020, ApJ, 903, 56, Equations 7-8
    Tan, Krumholz & McKee 2006, ApJL, 641, L121
    """
    # Dimensionless radial grid (r/R)
    x = jnp.linspace(0.01, 1.0, n_radial_points)
    dx = x[1] - x[0]

    # Core radius in dimensionless units
    x_c = jnp.maximum(r_c_over_R, 1e-4)  # Avoid division by zero

    # Density profile: rho/rho_c = [1 + (x/x_c)^2]^(-p/2)
    # Using softened form for numerical stability
    rho_normalized = jnp.power(1.0 + (x / x_c) ** 2, -p / 2.0)

    # Volume element: 4*pi*x^2*dx (in dimensionless units)
    dV = 4.0 * jnp.pi * x**2 * dx

    # Mass integral: M ~ int rho dV
    mass_integrand = rho_normalized * dV
    total_mass = jnp.sum(mass_integrand)

    # Mean density: <rho> = M / V, where V = (4/3)*pi*R^3 = 4*pi/3 in units of R
    volume = 4.0 * jnp.pi / 3.0
    mean_rho = total_mass / volume

    # SFR-weighted integral: int rho^(3/2) dV
    # (since SFR ~ rho/t_ff ~ rho^(3/2))
    sfr_integrand = jnp.power(rho_normalized, 1.5) * dV
    sfr_weighted = jnp.sum(sfr_integrand)

    # Top-hat reference: SFR_tophat ~ M * <rho>^(1/2)
    tophat_sfr = total_mass * jnp.sqrt(mean_rho)

    # Magnification factor
    zeta = sfr_weighted / tophat_sfr

    # Ensure zeta >= 1 (top-hat is minimum)
    return jnp.maximum(zeta, 1.0)


# =============================================================================
# Direct FDF Measurement (PRIMARY METHOD)
# =============================================================================


@jax.jit
def zeta_fdf_direct(
    rho_grid: Float[Array, "Nx Ny Nz"],
    tail_weights: Float[Array, "Nx Ny Nz"],
) -> Float[Array, ""]:
    """Measure ζ_FDF directly from a 3D density field.

    ζ_FDF is the ratio between the star formation rate in the actual
    structured dense tail and the SFR if that same mass were uniformly
    distributed at its mean density.

    This is the PRIMARY method for computing ζ. It bypasses the
    power-law assumption and measures the actual geometric boost from
    the 3D density field.

    Uses the soft-weight formula:

        ζ_FDF = [Σ w · ρ^(3/2)] / [M_tail · √ρ̄_tail]

    where:
    - w(x) are the soft tail weights (sigmoid) in [0, 1]
    - M_tail = Σ(w · ρ) is the weighted tail mass
    - ρ̄_tail = M_tail / Σ(w) is the weighted mean tail density

    Parameters
    ----------
    rho_grid : Array, shape (Nx, Ny, Nz)
        3D density field.
    tail_weights : Array, shape (Nx, Ny, Nz)
        Soft tail mask from compute_tail_pmfs_bm19().
        Values in [0, 1] where 1 = fully in tail, 0 = not in tail.

    Returns
    -------
    zeta_fdf : Array, scalar
        Measured magnification factor (>= 1).

    Notes
    -----
    The soft weights ensure differentiability. No hard threshold is used
    in this function.

    For p >= 1 (i.e., alpha <= 3), this is the recommended method over
    the analytic magnification_factor().

    The formula measures the actual freefall-weighted SFR boost relative
    to a uniform (top-hat) distribution of the same tail mass.
    """
    # dV cancels between numerator and denominator, so we work with
    # dimensionless sums over the grid. For non-uniform grids, would
    # need to pass dV explicitly.

    # Weighted tail mass: M_tail = sum(w * rho)
    M_tail = jnp.sum(tail_weights * rho_grid)

    # Weighted tail volume: V_tail = sum(w)
    V_tail = jnp.sum(tail_weights)

    # Avoid division by zero for empty tail
    V_tail_safe = jnp.maximum(V_tail, 1e-10)
    M_tail_safe = jnp.maximum(M_tail, 1e-10)

    # Weighted mean tail density: rho_tail_mean = M_tail / V_tail
    rho_tail_mean = M_tail_safe / V_tail_safe

    # SFR-weighted numerator: sum(w * rho^(3/2))
    sfr_weighted = jnp.sum(tail_weights * jnp.power(rho_grid, 1.5))

    # Top-hat reference: SFR_tophat = M_tail * rho_tail_mean^(1/2)
    tophat_sfr = M_tail_safe * jnp.sqrt(rho_tail_mean)

    # Avoid division by zero
    tophat_sfr_safe = jnp.maximum(tophat_sfr, 1e-10)

    # Magnification factor
    zeta = sfr_weighted / tophat_sfr_safe

    # Ensure zeta >= 1 (top-hat is minimum)
    return jnp.maximum(zeta, 1.0)


# =============================================================================
# SFR Prediction
# =============================================================================


@jax.jit
def sfr_per_dense_gas(
    zeta: Array,
    epsilon_ff_int: Array = 0.01,
    t_ff_dg: Array = 1.0,
) -> Array:
    """PP20 SFR per unit dense gas mass.

    SFR/M_dg = zeta * epsilon_ff_int / t_ff_dg

    Parameters
    ----------
    zeta : Array
        Magnification factor (from magnification_factor or zeta_fdf_direct).
    epsilon_ff_int : Array
        Intrinsic SFE per freefall time (default 0.01 = 1%).
        This is the "efficiency per freefall" from turbulent fragmentation
        theory.
    t_ff_dg : Array
        Mean freefall time of dense gas [Myr] (default 1.0).
        Typical values: 0.1-0.5 Myr for dense cores.

    Returns
    -------
    sfr_per_mdg : Array
        SFR per unit M_dg [Myr^{-1}].

    Notes
    -----
    The PP20 framework predicts that:
    - Higher zeta (more centrally concentrated) -> higher SFR/M_dg
    - The magnification factor accounts for the geometry-dependent
      boost in SFR due to shorter freefall times in dense regions

    References
    ----------
    Parmentier & Pasquali 2020, ApJ, 903, 56
    """
    return zeta * epsilon_ff_int / t_ff_dg
