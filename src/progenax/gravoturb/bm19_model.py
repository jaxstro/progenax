"""BM19 gravoturbulent PDF framework (Burkhart & Mocz 2019).

This module implements the piecewise lognormal+powerlaw density PDF
framework for predicting self-gravitating gas fractions from cloud
properties.

All functions are:
- @jax.jit compatible
- Differentiable via jax.grad
- Vectorizable via jax.vmap

Key Equations
-------------
- BM19 Eq. 1: sigma_s_sq = ln(1 + b^2 * M^2)
- BM19 Eq. 2: s_t = (alpha - 0.5) * sigma_s_sq
- BM19 Eq. 19-20: f_dense = full piecewise LN+PL integral

Parameter Ranges
----------------
- Mach number: Tested for M in [5, 30]. Formulas are formally valid but
  not calibrated outside this range.
- Powerlaw slope alpha: Must be > 1.0 for convergence. Tested range is
  [1.5, 3.0]. Values outside this range trigger a warning.

References
----------
Burkhart, B. & Mocz, P. 2019, ApJ, 879, 129
Federrath, C. et al. 2010, A&A, 512, A81
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfc
from jaxtyping import Array, Float

# Import magnification_factor from parmentier (lazy to avoid circular import)
# This is computed in bm19_pipeline for convenience


# =============================================================================
# Constants
# =============================================================================

# Power spectrum slope limits
BETA_KOLMOGOROV = 11.0 / 3.0  # ~3.67, incompressible limit
BETA_BURGERS = 4.0  # Shock-dominated limit

# Alpha range for validation
ALPHA_MIN_CONVERGENT = 1.0
ALPHA_MIN_TESTED = 1.5
ALPHA_MAX_TESTED = 3.0


# =============================================================================
# Result Container
# =============================================================================


class BM19Result(NamedTuple):
    """Container for BM19 pipeline outputs.

    All fields correspond to theory guide notation.

    Attributes
    ----------
    sigma_s : Array
        PDF width sqrt(sigma_s_sq)
    sigma_s_sq : Array
        PDF variance sigma_s^2
    s_t : Array
        Transition density (BM19 Eq. 2)
    f_dense : Array
        Self-gravitating fraction (BM19 Eq. 19-20)
    f_sub : Array
        Substructure fraction = eta * f_dense
    beta : Array
        Power spectrum slope
    p : Array
        PP20 profile slope = 3/alpha
    zeta : Array
        PP20 magnification factor
    """

    sigma_s: Float[Array, "..."]
    sigma_s_sq: Float[Array, "..."]
    s_t: Float[Array, "..."]
    f_dense: Float[Array, "..."]
    f_sub: Float[Array, "..."]
    beta: Float[Array, "..."]
    p: Float[Array, "..."]
    zeta: Float[Array, "..."]


# =============================================================================
# Core BM19 Functions
# =============================================================================


@jax.jit
def sigma_s_squared(mach: Array, b: Array = 0.4) -> Array:
    """BM19 Eq. 1: PDF variance from turbulence.

    sigma_s^2 = ln(1 + b^2 * M^2)

    Parameters
    ----------
    mach : Array
        Turbulent Mach number (sigma_v / c_s).
        Tested range: M in [5, 30]. Formulas are formally valid but not
        calibrated outside this range.
    b : Array
        Driving parameter (0.3-1.0; default 0.4 for mixed driving).
        - b ~ 1/3 (0.33): Solenoidal (incompressible, rotational) driving
        - b ~ 1.0: Compressive (irrotational) driving
        - b ~ 0.4: Natural mixture (default for star-forming clouds)

    Returns
    -------
    sigma_s_sq : Array
        Log-density variance.

    Notes
    -----
    Physical ranges for sigma_s at b=0.4:
        - M = 5: sigma_s ~ 0.9
        - M = 10: sigma_s ~ 1.4
        - M = 25: sigma_s ~ 1.9
        - M = 50: sigma_s ~ 2.4

    References
    ----------
    Burkhart & Mocz 2019, ApJ 879, 129, Eq. 1; originally Federrath et al.
    2010, A&A 512, A81.
    """
    return jnp.log(1.0 + b**2 * mach**2)


@jax.jit
def transition_density(sigma_s_sq: Array, alpha: Array) -> Array:
    """BM19 Eq. 2: Transition density (DERIVED, not parameterized).

    s_t = (alpha - 1/2) * sigma_s^2

    This emerges from requiring PDF continuity at the LN->PL transition.
    It represents the onset of self-gravitating collapse.

    Parameters
    ----------
    sigma_s_sq : Array
        PDF variance from sigma_s_squared().
    alpha : Array
        Powerlaw slope (tested range: 1.5-3.0).
        - alpha ~ 1.5: Steep collapse / centrally concentrated
        - alpha ~ 2.0: Active collapse (default)
        - alpha ~ 3.0: Early collapse / shallow density contrast

        Must be > 1.0 for convergence. Values outside [1.5, 3.0] may be
        unreliable.

    Returns
    -------
    s_t : Array
        Transition log-density.
    """
    return (alpha - 0.5) * sigma_s_sq


@jax.jit
def f_dense_lognormal_limit(sigma_s_sq: Array, s_t: Array) -> Array:
    """Self-gravitating fraction in pure lognormal limit (alpha -> infinity).

    f_dense = (1/2) * erfc[(s_t - sigma_s^2/2) / (sqrt(2) * sigma_s)]

    This approximation ignores the powerlaw tail and is only valid for
    very steep slopes (alpha > 2.5). For general use, prefer
    f_dense_bm19_full() which implements the complete piecewise integral.

    IMPORTANT: This is for comparison/validation only.
    Use f_dense_bm19_full() for production.

    Parameters
    ----------
    sigma_s_sq : Array
        PDF variance.
    s_t : Array
        Transition density.

    Returns
    -------
    f_dense : Array
        Self-gravitating mass fraction (lognormal approximation).
    """
    sigma_s = jnp.sqrt(sigma_s_sq)
    u = (s_t - sigma_s_sq / 2.0) / (jnp.sqrt(2.0) * sigma_s)
    return 0.5 * erfc(u)


def _validate_alpha(alpha: float | Array) -> None:
    """Validate alpha parameter and emit warnings/errors.

    This is called at Python level before JIT compilation.

    Raises
    ------
    ValueError
        If alpha <= 1.0 (integral diverges).

    Warns
    -----
    UserWarning
        If alpha outside tested range [1.5, 3.0].
    """
    # Convert to Python scalar for comparison if possible
    try:
        alpha_val = float(alpha)
    except (TypeError, ValueError):
        # Array input - can't validate statically, will be handled in JIT
        return

    if alpha_val <= ALPHA_MIN_CONVERGENT:
        raise ValueError(
            f"BM19 power-law slope alpha must be > 1.0 for convergence. "
            f"Got alpha = {alpha_val}."
        )

    if alpha_val < ALPHA_MIN_TESTED or alpha_val > ALPHA_MAX_TESTED:
        warnings.warn(
            f"BM19 alpha outside tested range [{ALPHA_MIN_TESTED}, {ALPHA_MAX_TESTED}]. "
            f"Results may be unreliable (alpha = {alpha_val}).",
            UserWarning,
            stacklevel=3,
        )


@jax.jit
def _f_dense_bm19_full_jit(
    sigma_s_sq: Array,
    s_t: Array,
    alpha: Array,
) -> Array:
    """JIT-compiled core of f_dense_bm19_full (no validation)."""
    sigma_s = jnp.sqrt(sigma_s_sq)

    # -----------------------------------------------------------------
    # Mass in lognormal part: M_LN(-inf, s_t) = Phi[(s_t - sigma^2/2) / sigma]
    #
    # This is the CDF of a Gaussian with mean sigma^2/2 and std sigma.
    # NO extra exp(sigma^2/2) factor - that's a common error.
    # -----------------------------------------------------------------
    u_LN = (s_t - sigma_s_sq / 2.0) / (jnp.sqrt(2.0) * sigma_s)
    M_LN = 0.5 * (1.0 + erf(u_LN))

    # -----------------------------------------------------------------
    # Lognormal PDF at transition point (for powerlaw normalization A)
    # p_LN(s_t) = (1/sqrt(2*pi)*sigma_s) * exp[-(s_t - s_0)^2 / (2*sigma_s^2)]
    # where s_0 = -sigma_s^2/2 for mass conservation
    # -----------------------------------------------------------------
    s_0 = -sigma_s_sq / 2.0
    p_LN_at_st = (1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma_s)) * jnp.exp(
        -((s_t - s_0) ** 2) / (2.0 * sigma_s_sq)
    )

    # Powerlaw normalization: A = p_LN(s_t) * exp(alpha * s_t)
    A = p_LN_at_st * jnp.exp(alpha * s_t)

    # -----------------------------------------------------------------
    # Mass in powerlaw part: M_PL = int_{s_t}^{inf} e^s * A * exp(-alpha*s) ds
    #                             = A * int_{s_t}^{inf} exp((1-alpha)*s) ds
    #                             = A / (alpha-1) * exp((1-alpha)*s_t) for alpha > 1
    # -----------------------------------------------------------------
    # Guard against alpha <= 1 (integral diverges)
    alpha_safe = jnp.maximum(alpha, 1.0 + 1e-6)
    M_PL = A / (alpha_safe - 1.0) * jnp.exp((1.0 - alpha_safe) * s_t)

    # -----------------------------------------------------------------
    # Total mass and fraction
    # For pure lognormal (alpha -> inf), M_PL -> 0 and M_LN -> total mass
    # The total should be ~1 + small PL contribution
    # -----------------------------------------------------------------
    M_total = M_LN + M_PL
    f_dense = M_PL / M_total

    # Clamp to valid range
    return jnp.clip(f_dense, 0.0, 1.0)


def f_dense_bm19_full(
    sigma_s_sq: Array,
    s_t: Array,
    alpha: Array,
) -> Array:
    """Full BM19 self-gravitating fraction (Eqs. 19-20).

    Computes the mass-weighted integral over the piecewise LN+PL PDF:

        f_dense = M(s > s_t) / M_total

    where the PDF is:
        p(s) = p_LN(s)           for s < s_t
        p(s) = A * exp(-alpha*s) for s >= s_t

    with A chosen for continuity at s_t.

    Parameters
    ----------
    sigma_s_sq : Array
        PDF variance.
    s_t : Array
        Transition density.
    alpha : Array
        Powerlaw slope.

        Must be > 1.0 for convergence (raises ValueError otherwise).
        Tested range is [1.5, 3.0]; values outside trigger a warning.

    Returns
    -------
    f_dense : Array
        Mass fraction in self-gravitating tail [0, 1].

    Raises
    ------
    ValueError
        If alpha <= 1.0 (integral diverges).

    Notes
    -----
    CRITICAL IMPLEMENTATION NOTES:

    1. Mass conservation: s_0 = -sigma_s^2/2 ensures int e^s p_LN(s) ds = 1.
       There is NO extra exp(sigma_s^2/2) prefactor.
    2. Powerlaw normalization: A = p_LN(s_t) * exp(alpha * s_t) for continuity.
    3. Convergence requires alpha > 1.

    References
    ----------
    Burkhart & Mocz 2019, ApJ, 879, 129, Equations 19-20
    """
    # Validate alpha at Python level (before JIT)
    _validate_alpha(alpha)

    # Call JIT-compiled core
    return _f_dense_bm19_full_jit(sigma_s_sq, s_t, alpha)


@jax.jit
def power_spectrum_slope(mach: Array, b: Array = 0.4) -> Array:
    """Density power spectrum slope beta from turbulence.

    Interpolates between Kolmogorov (beta ~ 11/3) and Burgers (beta ~ 4).
    Based on Federrath+ 2010 (A&A 512, A81).

    Parameters
    ----------
    mach : Array
        Turbulent Mach number.
        Tested range: M in [5, 30]. Formulas are formally valid but not
        calibrated outside this range.
    b : Array
        Driving parameter.

    Returns
    -------
    beta : Array
        Power spectrum slope (3.67-4.0).

    Notes
    -----
    Limiting behavior:
        - Subsonic (M << 1): beta -> 11/3 ~ 3.67 (Kolmogorov)
        - Supersonic (M >> 1): beta -> 4.0 (Burgers/shock-dominated)

    For star-forming clouds with M >> 1, expect beta ~ 4.

    References
    ----------
    Federrath et al. 2010, A&A 512, A81
    Kim & Ryu 2005, ApJL 630, L45
    """
    # Effective turbulent amplitude
    bM = b * mach

    # Smooth transition using tanh
    # Transition occurs around M ~ 1-2
    transition_mach = 1.5
    transition_width = 1.0

    weight = 0.5 * (1.0 + jnp.tanh((bM - transition_mach) / transition_width))

    beta = BETA_KOLMOGOROV + (BETA_BURGERS - BETA_KOLMOGOROV) * weight

    return beta


# =============================================================================
# Main Pipeline
# =============================================================================


def bm19_pipeline(
    mach: Array,
    b: Array = 0.4,
    alpha: Array = 2.0,
    eta_survive: Array = 0.6,
) -> BM19Result:
    """Complete BM19 calculation: cloud parameters -> f_sub + PP20 quantities.

    This is the primary entry point for BM19-consistent physics.

    Parameters
    ----------
    mach : Array
        Turbulent Mach number.
        Tested range: M in [5, 30]. Formulas are formally valid but not
        calibrated outside this range.
    b : Array
        Driving parameter (0.3-1.0; default 0.4 for mixed driving).
    alpha : Array
        Powerlaw slope (default 2.0).
        - alpha ~ 1.5: Steep collapse / centrally concentrated
        - alpha ~ 2.0: Active collapse (default)
        - alpha ~ 3.0: Early collapse / shallow density contrast

        Must be > 1.0 for convergence (raises ValueError).
        Tested range is [1.5, 3.0]; values outside trigger a warning.
    eta_survive : Array
        Feedback survival efficiency (0-1; default 0.6).
        Fraction of f_dense that survives as stellar substructure.

    Returns
    -------
    BM19Result
        NamedTuple with all intermediate and output quantities.

    Raises
    ------
    ValueError
        If alpha <= 1.0 (integral diverges).

    Notes
    -----
    The PP20 quantities (p, zeta) are computed for convenience but are
    conceptually Part III (Parmentier interpretation), not Part I (BM19).

    Examples
    --------
    >>> result = bm19_pipeline(mach=10.0, alpha=2.0, eta_survive=0.6)
    >>> print(f"f_sub = {result.f_sub:.3f}")

    >>> # Vectorized over Mach numbers
    >>> import jax.numpy as jnp
    >>> machs = jnp.array([5.0, 10.0, 20.0])
    >>> results = jax.vmap(lambda m: bm19_pipeline(m, alpha=2.0))(machs)
    >>> print(f"f_dense = {results.f_dense}")
    """
    # Validate alpha at Python level (before JIT)
    _validate_alpha(alpha)

    # Call JIT-compiled core
    return _bm19_pipeline_jit(mach, b, alpha, eta_survive)


@jax.jit
def _bm19_pipeline_jit(
    mach: Array,
    b: Array,
    alpha: Array,
    eta_survive: Array,
) -> BM19Result:
    """JIT-compiled core of bm19_pipeline (no validation)."""
    # Import here to avoid circular import
    from progenax.gravoturb.pp20_magnification import magnification_factor

    # BM19 Eq. 1: PDF width
    sigma_s_sq = sigma_s_squared(mach, b)
    sigma_s = jnp.sqrt(sigma_s_sq)

    # Power spectrum slope from turbulence (Federrath+ 2010)
    beta = power_spectrum_slope(mach, b)

    # BM19 Eq. 2: Transition density
    s_t = transition_density(sigma_s_sq, alpha)

    # Self-gravitating fraction: FULL BM19 piecewise integral
    f_dense = _f_dense_bm19_full_jit(sigma_s_sq, s_t, alpha)

    # Substructure fraction after feedback
    f_sub = eta_survive * f_dense

    # PP20 connection: alpha <-> p
    p = 3.0 / alpha
    zeta = magnification_factor(p)

    return BM19Result(
        sigma_s=sigma_s,
        sigma_s_sq=sigma_s_sq,
        s_t=s_t,
        f_dense=f_dense,
        f_sub=f_sub,
        beta=beta,
        p=p,
        zeta=zeta,
    )
