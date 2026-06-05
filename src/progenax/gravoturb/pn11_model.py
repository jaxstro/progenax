"""PN11/FK12 gravoturbulent framework (Padoan-Nordlund / Federrath-Klessen).

This module implements the classical Padoan & Nordlund (2011) / Federrath &
Klessen (2012) critical density framework for gravoturbulent star formation.

All functions are:
- @jax.jit compatible
- Differentiable via jax.grad
- Vectorizable via jax.vmap

When to Use PN11 vs BM19
------------------------
**PN11** (this module):
- When you need the PN11 integral-scale critical density (theta parameter)
- When you have cloud surface density (Sigma) information
- For comparison with classical literature

**BM19** (bm19_model.py):
- Default for most applications (fewer free parameters)
- When powerlaw tail matters (full piecewise PDF)
- When s_t should be derived from PDF shape

Key Equations
-------------
- s_crit = ln(0.067 * theta^-2 * alpha_vir * M^2)  (PN11 Eq. 8)
- alpha_vir = alpha_0 * (Sigma_0 / Sigma)
- f_dense = (1/2) * erfc[(s_crit - sigma_s^2/2) / (sqrt(2) * sigma_s)]

References
----------
Padoan, P. & Nordlund, A. 2011, ApJ, 730, 40
Federrath, C. & Klessen, R. S. 2012, ApJ, 761, 156
Heyer, M. & Dame, T. M. 2015, ARA&A, 53, 583
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import erfc
from jaxtyping import Array, Float


# =============================================================================
# Result Container
# =============================================================================


class PN11Result(NamedTuple):
    """Container for PN11 pipeline outputs.

    All fields correspond to the PN11/FK12 derivation chain.

    Attributes
    ----------
    sigma_s : Array
        PDF width sqrt(sigma_s_sq)
    sigma_s_sq : Array
        PDF variance from Federrath+2010
    alpha_vir : Array
        Virial parameter from Heyer-Dame relation
    s_crit : Array
        Critical log-density (PN11 Eq.)
    f_dense : Array
        Self-gravitating fraction (pure lognormal erfc)
    f_sub : Array
        Substructure fraction = eta * f_dense
    """

    sigma_s: Float[Array, "..."]
    sigma_s_sq: Float[Array, "..."]
    alpha_vir: Float[Array, "..."]
    s_crit: Float[Array, "..."]
    f_dense: Float[Array, "..."]
    f_sub: Float[Array, "..."]


# =============================================================================
# PN11/FK12 Core Functions
# =============================================================================


@jax.jit
def s_crit_pn11(
    mach: Array,
    alpha_vir: Array,
    theta: Array = 0.35,
) -> Array:
    """Padoan & Nordlund (2011) critical density threshold (PN11 Eq. 8).

    s_crit = ln(0.067 * theta^-2 * alpha_vir * M^2)

    The critical density marks the transition to self-gravitating gas; gas with
    s > s_crit is expected to collapse. PN11 derive it by equating the
    Bonnor-Ebert mass to the mass of a post-shock layer (their Eq. 7), giving the
    numerical coefficient 0.067 (PN11 Eq. 8).

    Parameters
    ----------
    mach : Array
        Turbulent (3D rms) Mach number.
    alpha_vir : Array
        Virial parameter (from alpha_vir_from_sigma or known).
    theta : Array
        Turbulence integral-scale factor: the integral scale is theta * L_cloud
        with theta <= 1. PN11 adopt theta = 0.35 (Wang & George 2002 correction;
        PN11 p.3), giving prefactor 0.067 * 0.35^-2 = 0.547 (PN11 Eq. 11).

    Returns
    -------
    s_crit : Array
        Critical log-density.

    Notes
    -----
    Larger theta (larger turbulence driving scale) -> LOWER critical density:
    the prefactor 0.067 * theta^-2 decreases with theta (theta is in the
    denominator). This is the opposite sense to the KM05/FK12 phi_x parameter.

    This is the PN11 parametrization (Eq. 8), distinct from the KM05/FK12 form
    s_crit = ln((pi^2/5) * phi_x^2 * alpha_vir * M^2) used by swindlax and the
    rosen-burkhart-swindle companion paper. PN11 (p.3) report the KM05 numerical
    coefficient as phi_x = 1.12.

    References
    ----------
    Padoan & Nordlund 2011, ApJ, 730, 40, Eq. 8 and Eq. 11
    """
    prefactor = 0.067 * theta**-2.0
    return jnp.log(prefactor * alpha_vir * mach**2)


@jax.jit
def f_dense_pn11(sigma_s_sq: Array, s_crit: Array) -> Array:
    """PN11-style dense gas fraction using pure lognormal erfc.

    f_dense = (1/2) * erfc[(s_crit - sigma_s^2/2) / (sqrt(2) * sigma_s)]

    This formula is exact for a pure lognormal PDF. For comparison with
    BM19's piecewise approach, see bm19_model.f_dense_bm19_full().

    Parameters
    ----------
    sigma_s_sq : Array
        PDF variance (from sigma_s_squared or equivalent).
    s_crit : Array
        Critical density from s_crit_pn11().

    Returns
    -------
    f_dense : Array
        Self-gravitating mass fraction.

    Notes
    -----
    Comparison with BM19:
    - PN11 f_dense depends on: M, alpha_vir, theta (via s_crit)
    - BM19 f_dense depends on: M, alpha (powerlaw slope)

    PN11 uses pure lognormal; BM19 includes powerlaw tail contribution.
    """
    sigma_s = jnp.sqrt(sigma_s_sq)
    u = (s_crit - sigma_s_sq / 2.0) / (jnp.sqrt(2.0) * sigma_s)
    return 0.5 * erfc(u)


@jax.jit
def alpha_vir_from_sigma(
    Sigma: Array,
    alpha_0: Array = 2.0,
    Sigma_0: Array = 85.0,
) -> Array:
    """Virial parameter from surface density (Heyer & Dame 2015).

    alpha_vir = alpha_0 * (Sigma_0 / Sigma)

    Parameters
    ----------
    Sigma : Array
        Cloud surface density [M_sun/pc^2].
    alpha_0 : Array
        Reference virial parameter (default 2.0 from Heyer & Dame 2015).
    Sigma_0 : Array
        Reference surface density [M_sun/pc^2] (default 85.0).

    Returns
    -------
    alpha_vir : Array
        Virial parameter.

    Notes
    -----
    The virial parameter alpha_vir = 2 * E_kin / |E_grav| measures how
    bound a cloud is:
    - alpha_vir < 1: Gravitationally bound
    - alpha_vir ~ 1-2: Marginally bound (typical GMCs)
    - alpha_vir > 2: Unbound

    The inverse scaling with Sigma comes from observations that denser
    clouds are more gravitationally bound (lower alpha_vir).

    Scatter in this relation is significant (factor ~2).

    References
    ----------
    Heyer & Dame 2015, ARA&A 53, 583
    """
    return alpha_0 * (Sigma_0 / Sigma)


@jax.jit
def sigma_s_squared(mach: Array, b: Array = 0.4) -> Array:
    """PDF variance from turbulence (Federrath+2010 Eq. 14).

    sigma_s^2 = ln(1 + b^2 * M^2)

    Parameters
    ----------
    mach : Array
        Turbulent Mach number (sigma_v / c_s).
    b : Array
        Driving parameter (0.3-1.0; default 0.4 for mixed driving).

    Returns
    -------
    sigma_s_sq : Array
        Log-density variance.

    Notes
    -----
    This is the same formula used by BM19. Shared here for convenience
    so users don't need to import from both modules.

    References
    ----------
    Federrath et al. 2010, A&A 512, A81, Eq. 14
    """
    return jnp.log(1.0 + b**2 * mach**2)


# =============================================================================
# Main Pipeline
# =============================================================================


def pn11_pipeline(
    mach: Array,
    Sigma: Array,
    eta_survive: Array = 0.6,
    b: Array = 0.4,
    theta: Array = 0.35,
    alpha_0: Array = 2.0,
    Sigma_0: Array = 85.0,
) -> PN11Result:
    """Complete PN11/FK12 calculation: cloud parameters -> f_sub.

    Derivation chain:
        (Sigma, M) -> sigma_s -> alpha_vir -> s_crit -> f_dense -> f_sub

    Parameters
    ----------
    mach : Array
        Turbulent Mach number.
    Sigma : Array
        Cloud surface density [M_sun/pc^2].
    eta_survive : Array
        Feedback survival efficiency (0-1; default 0.6).
    b : Array
        Driving parameter (default 0.4).
    theta : Array
        Turbulence integral-scale factor (default 0.35; PN11 Eq. 8).
    alpha_0 : Array
        Reference virial parameter (default 2.0).
    Sigma_0 : Array
        Reference surface density (default 85.0).

    Returns
    -------
    PN11Result
        NamedTuple with all intermediate and output quantities.

    Notes
    -----
    PN11 vs BM19 key differences:

    1. PN11 uses s_crit (parameterized); BM19 uses s_t (derived from PDF)
    2. PN11 requires Sigma (surface density); BM19 does not
    3. PN11 includes theta (integral-scale factor); BM19 has fewer parameters
    4. PN11 uses pure lognormal erfc; BM19 uses full piecewise integral

    Examples
    --------
    >>> result = pn11_pipeline(mach=10.0, Sigma=100.0, eta_survive=0.6)
    >>> print(f"f_dense = {result.f_dense:.3f}")
    >>> print(f"f_sub = {result.f_sub:.3f}")

    >>> # Vectorized over surface densities
    >>> import jax.numpy as jnp
    >>> Sigmas = jnp.array([50.0, 100.0, 200.0])
    >>> results = jax.vmap(lambda S: pn11_pipeline(10.0, S))(Sigmas)
    >>> print(f"f_dense = {results.f_dense}")
    """
    return _pn11_pipeline_jit(mach, Sigma, eta_survive, b, theta, alpha_0, Sigma_0)


@jax.jit
def _pn11_pipeline_jit(
    mach: Array,
    Sigma: Array,
    eta_survive: Array,
    b: Array,
    theta: Array,
    alpha_0: Array,
    Sigma_0: Array,
) -> PN11Result:
    """JIT-compiled core of pn11_pipeline."""
    # Step 1: PDF width (Federrath+2010)
    sigma_s_sq = sigma_s_squared(mach, b)
    sigma_s = jnp.sqrt(sigma_s_sq)

    # Step 2: Virial parameter (Heyer & Dame 2015)
    alpha_vir = alpha_vir_from_sigma(Sigma, alpha_0, Sigma_0)

    # Step 3: Critical density (PN11 Eq. 8)
    s_crit = s_crit_pn11(mach, alpha_vir, theta)

    # Step 4: Self-gravitating fraction (pure lognormal)
    f_dense = f_dense_pn11(sigma_s_sq, s_crit)

    # Step 5: Substructure fraction
    f_sub = eta_survive * f_dense

    return PN11Result(
        sigma_s=sigma_s,
        sigma_s_sq=sigma_s_sq,
        alpha_vir=alpha_vir,
        s_crit=s_crit,
        f_dense=f_dense,
        f_sub=f_sub,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "PN11Result",
    "s_crit_pn11",
    "f_dense_pn11",
    "alpha_vir_from_sigma",
    "sigma_s_squared",
    "pn11_pipeline",
]
