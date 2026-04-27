"""Parmentier & Pasquali (2020) magnification factor ζ(p) and dense-gas SFR.

Reference: Parmentier, G. & Pasquali, A. 2020, ApJ, 903, 56
           (arXiv:2009.10652; "A New Parameterization of the Star
           Formation Rate–Dense Gas Mass Relation: Embracing Gas Density
           Gradients").

Physical meaning
================

The magnification factor ζ quantifies how much the *star formation rate*
of a centrally-concentrated cloud is boosted relative to a uniform-density
("top-hat") cloud of the same mass and outer radius. The boost arises
because the local free-fall time t_ff ∝ ρ^(-1/2) is shortest where the
density is highest, so cloud regions whose density exceeds the cloud-mean
contribute disproportionately to the cloud-integrated SFR
(SFR_local ∝ ρ / t_ff_local ∝ ρ^(3/2)).

Definition (PP20 Eq. 1, equivalently Eq. 8 of Parmentier 2019):

    ζ ≡ SFR_clump / SFR_TH                                          (1)

where SFR_TH is the SFR of the same clump if its gas were redistributed
uniformly at the cloud mean density ⟨ρ⟩. Writing the local SFR as
ε_ff,int · ρ / t_ff(ρ) and integrating over the clump volume V_R:

         ∫_{V_R} ρ^(3/2) dV
    ζ = ──────────────────────                                      (2)
            M · ⟨ρ⟩^(1/2)

where M is the clump's total gas mass. In this form ζ(0) ≡ 1 by
construction (a uniform clump is its own top-hat reference) and ζ
increases with the steepness of the density profile.

Closed-form derivation for a pure power-law profile
===================================================

For ρ(r) = ρ_R (r/R)^(-p), with the clump filling a sphere of outer
radius R, the integrals in Eq. (2) admit closed forms whenever
0 ≤ p < 2 (the upper bound is the divergence threshold of the SFR
integrand ∫ r^(2 - 3p/2) dr; physically, p = 2 is the singular-isothermal
profile where the central density runs away).

Step 1 — total mass:

    M = 4π ρ_R R^p ∫_0^R r^{2-p} dr = 4π ρ_R R^3 / (3 - p)            (3)

Step 2 — mean density:

    ⟨ρ⟩ = M / V = 3 ρ_R / (3 - p)                                    (4)

Step 3 — SFR integral (∫ ρ^(3/2) dV):

    ∫ ρ^(3/2) dV = 4π ρ_R^{3/2} R^{3p/2} ∫_0^R r^{2 - 3p/2} dr
                 = 8π ρ_R^{3/2} R^3 / [3 (2 - p)]                    (5)

Step 4 — top-hat reference (M · √⟨ρ⟩):

    M · ⟨ρ⟩^{1/2} = 4π √3 ρ_R^{3/2} R^3 / (3 - p)^{3/2}              (6)

Step 5 — combine (5) / (6):

    ζ(p) = 2 (3 - p)^{3/2} / [3^{3/2} (2 - p)]                       (7)

This is the canonical analytic form implemented in
:func:`magnification_factor` below. PP20 Eq. 6 quotes the same result
in the equivalent form

    ζ(p) = (3 - p)^{3/2} / [2.6 · (2 - p)]                           (PP20-6)

where "2.6" is a numerical approximation to 3^{3/2}/2 = 2.598. Equations
(7) and (PP20-6) agree to 0.08% across the physical domain; tests in
``progenax/tests/unit/physics/test_pp20_zeta_canonical.py`` lock this
equivalence.

Spot values (compare PP20 Fig. 1):

    p = 0    →  ζ = 1                       (top-hat)
    p = 1    →  ζ = 2 · 2^{3/2} / 3^{3/2}  ≈ 1.0887
    p = 1.5  →  ζ = √2                      ≈ 1.4142
    p = 1.67 →  ζ ≈ 1.79                    (Kainulainen+2014 median)
    p → 2    →  ζ → ∞

Numerical safety
================

The closed-form ζ(p) diverges at p = 2. To keep gradients well-behaved
under HMC/NUTS and to avoid pathological broadcasts in vectorised
forward calls, we clip p to [0, P_MAX] with P_MAX = 1.95. PP20 Fig. 1
adopts the same convention (the analytic curve is plotted only up to
p ≈ 1.95; beyond that the paper switches to a numerically-integrated
profile with a finite constant-density core, i.e.
:func:`magnification_factor_with_core` below).

Module contents
===============

- :func:`magnification_factor`: PP20 Eq. 6 in the canonical analytic
  form — for pure power-law profiles ρ(r) ∝ r^(-p).
- :func:`magnification_factor_with_core`: PP20 Eq. 7-8 via numerical
  integration — for cored profiles ρ(r) ∝ [1+(r/r_c)^2]^(-p/2).
- :func:`zeta_fdf_direct`: Freefall-weighted ζ measured directly from a
  3D density field (no power-law assumption); see Burkhart 2018 / 2021.
- :func:`sfr_per_dense_gas`: PP20 SFR per unit dense-gas mass.

All functions are JAX-native: @jax.jit compatible, differentiable via
jax.grad, vectorisable via jax.vmap.

Historical note
===============

A pre-2026 transcription of PP20 Eq. 6 in this module read
``(3 - p) / (2.6 - 2*p)^(3/2)`` — a typo in which the constant 2.6 had
been moved *inside* the 3/2 power and the (3 - p) factor had lost its
3/2 exponent. The buggy form had a spurious singularity at p = 1.3 that
was rationalised in docstrings as a "domain limit"; in fact PP20 Eq. 6
is well-behaved over the full 0 ≤ p < 2 domain, and the only true
singularity is at p = 2 (singular isothermal collapse). The same bug
existed in ``swindlax.physics.density_gradient`` and was fixed there
first; see the hardening report at
``papers/rosen-burkhart-swindle-2026/reviews/audit-2026-04-28-pp20-validation.md``.
Verified 2026-04-28 against PP20 Eq. 6 (page 2) and Kainulainen+2014
ζ(1.67) ≈ 1.79.

References
----------
Parmentier, G. & Pasquali, A. 2020, ApJ, 903, 56
Tan, J. C., Krumholz, M. R., & McKee, C. F. 2006, ApJL, 641, L121
Kainulainen, J., Federrath, C., & Henning, T. 2014, Science, 344, 183
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# Numerical-safety clip: ζ(p) diverges as p → 2 (singular isothermal
# collapse). P_MAX < 2 keeps the function differentiable for HMC/NUTS
# and matches PP20 Fig. 1, which caps the analytic curve before p=2.
P_MAX = 1.95


# =============================================================================
# Analytic Magnification Factor
# =============================================================================


@jax.jit
def magnification_factor(p: Array) -> Array:
    r"""Compute the PP20 magnification factor ζ(p) for power-law profiles.

    Closed-form analytic expression for a pure power-law density profile
    ρ(r) ∝ r^(-p), 0 ≤ p < 2:

    .. math::

        \zeta(p) \;=\; \frac{2\,(3-p)^{3/2}}{3^{3/2}\,(2-p)}

    equivalent to Parmentier & Pasquali 2020 Eq. 6:

    .. math::

        \zeta(p) \;=\; \frac{(3-p)^{3/2}}{2.6\,(2-p)}

    See the module docstring for the integral derivation. The two forms
    agree to 0.08% across the physical 0 ≤ p < 2 domain — PP20's "2.6"
    is a numerical approximation to 3^(3/2)/2 = 2.598; this implementation
    uses the unrounded analytic form.

    Parameters
    ----------
    p : Array
        Radial density-profile slope for ρ(r) ∝ r^(-p). Values are
        clipped to [0, P_MAX] = [0, 1.95]. The clip protects HMC/NUTS
        gradients against the p → 2 divergence (singular isothermal
        collapse) and matches PP20 Fig. 1, which caps the analytic curve
        before p = 2.

    Returns
    -------
    zeta : Array
        Magnification factor ζ(p) ≥ 1, with ζ(0) = 1 (uniform sphere)
        and ζ increasing monotonically with p over the clipped domain.

    Notes
    -----
    Spot values:

    - ζ(0)    = 1                         (top-hat reference)
    - ζ(1)    = 2·2^(3/2) / 3^(3/2) ≈ 1.0887
    - ζ(1.5)  = √2                  ≈ 1.4142
    - ζ(1.67) ≈ 1.79                (Kainulainen+2014 observational anchor)

    For cored profiles or for ζ measured directly from a 3D density field,
    see :func:`magnification_factor_with_core` and :func:`zeta_fdf_direct`.

    References
    ----------
    Parmentier & Pasquali 2020, ApJ, 903, 56, Equation 6
    """
    p_safe = jnp.clip(p, 0.0, P_MAX)
    return 2.0 * (3.0 - p_safe) ** 1.5 / (3.0**1.5 * (2.0 - p_safe))


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
