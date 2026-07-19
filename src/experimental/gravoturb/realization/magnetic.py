r"""Magnetized-turbulence scalar chain (P1a; ADR-0060).

Magnetic support suppresses the gravoturbulent density-PDF width. The primary physical knob is
the mass-to-flux ratio ``mu_phi`` (ADR-0060); the mean field ``B0``, Alfven Mach number ``M_A``,
and plasma ``beta_0`` are all *derived* from ``mu_phi`` and the cloud at the profile (half-mass)
scale, then fed into the Molina et al. 2012 magnetic density-variance law.

Physics (verified against papers in ``docs/core-papers/``):

- Magnetic width (Molina et al. 2012, intermediate B ∝ ρ^{1/2} case; Federrath & Klessen 2012
  Eqs. 4-5, ``Federrath_2012_ApJ_761_156.pdf``):

  .. math:: \sigma_s^2 = \ln\!\Big(1 + b^2 \mathcal{M}_s^2 \frac{\beta_0}{\beta_0+1}\Big)

  reducing to the hydro law :func:`gravoturb.theory.density_pdf.sigma_s_squared` as β₀→∞.

- Plasma beta (F&K12, explicit): β₀ = P_th/P_mag = 2 c_s²/v_A² = 2 (M_A/M_s)².

- Mass-to-flux inversion at the half-mass scale. With flux Φ = B₀ π r_h², critical value
  (M/Φ)_crit = c_Φ/√G (Padoan & Nordlund 2011 Eq. 16, coeff. from Tomisaka et al. 1988; ADR-0059),
  and mean density ρ₀ = M/((4/3)π r_h³), the chain reduces analytically to

  .. math:: \beta_0 = 6\pi^2 c_\Phi^2 \mu_\Phi^2 \frac{r_h c_s^2}{G M}.

All functions are UNIT-AGNOSTIC and JAX-native/differentiable: pass a consistent (G, mass, length,
velocity) set — β₀ and σ_s² come out dimensionless. The km/s → pc/Myr conversion of c_s happens at
the spec/builder boundary, not here (mirrors ``VelocitySpec.mode='physical'``). Gauss/μG conversion
of B₀ for RMHD export is a later phase (L3/P1b); B₀ here is in the caller's dynamical units.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

# Critical mass-to-flux coefficient: (M/Phi)_crit = C_PHI / sqrt(G).
# Padoan & Nordlund 2011 (ApJ 730, 40) Eq. 16, numerical coefficient from Tomisaka et al. 1988
# (see also Nakano & Nakamura 1978). Verified against docs/core-papers/Padoan_2011_ApJ_730_40.pdf.
# ADR-0059.
C_PHI = 0.17

_FourPi = 4.0 * jnp.pi


def sigma_s_squared_magnetic(
    mach: Float[Array, ""], b: Float[Array, ""], beta0: Float[Array, ""]
) -> Float[Array, ""]:
    r"""Magnetic density-PDF width σ_s² = ln(1 + b²ℳ² β₀/(β₀+1)) (Molina 2012 / F&K12 Eq. 4).

    Recovers the hydro :func:`gravoturb.theory.density_pdf.sigma_s_squared` as ``beta0`` → ∞
    (weak field), and is strictly suppressed below it for finite β₀ (magnetic cushioning).
    """
    factor = beta0 / (beta0 + 1.0)
    return jnp.log(1.0 + (b * mach) ** 2 * factor)


def plasma_beta(
    mach: Float[Array, ""], mach_alfven: Float[Array, ""]
) -> Float[Array, ""]:
    r"""Plasma beta β₀ = 2 (ℳ_A/ℳ_s)² = P_th/P_mag (Federrath & Klessen 2012, explicit)."""
    return 2.0 * (mach_alfven / mach) ** 2


def mean_density(mass: Float[Array, ""], radius: Float[Array, ""]) -> Float[Array, ""]:
    r"""Mean mass density in a sphere, ρ = M / ((4/3)π R³)."""
    return mass / (_FourPi / 3.0 * radius**3)


def mean_field_strength(
    mu_phi: Float[Array, ""],
    mass: Float[Array, ""],
    radius: Float[Array, ""],
    G: Float[Array, ""],
    c_phi: float = C_PHI,
) -> Float[Array, ""]:
    r"""Mean field B₀ from the mass-to-flux ratio, B₀ = M√G / (μ_Φ c_Φ π R²).

    Inverts μ_Φ ≡ (M/Φ)/(M/Φ)_crit with Φ = B₀ π R² and (M/Φ)_crit = c_Φ/√G. Returned in the
    caller's dynamical units (Gauss/μG conversion is a later phase).
    """
    return mass * jnp.sqrt(G) / (mu_phi * c_phi * jnp.pi * radius**2)


def alfven_speed(b0: Float[Array, ""], rho: Float[Array, ""]) -> Float[Array, ""]:
    r"""Alfven speed v_A = B₀ / √(4πρ)."""
    return b0 / jnp.sqrt(_FourPi * rho)


def alfven_mach(
    mach: Float[Array, ""], c_s: Float[Array, ""], v_alfven: Float[Array, ""]
) -> Float[Array, ""]:
    r"""Alfven Mach number ℳ_A = σ_v / v_A with σ_v = ℳ_s c_s (isothermal)."""
    return mach * c_s / v_alfven


def beta_from_mass_to_flux(
    mu_phi: Float[Array, ""],
    mach: Float[Array, ""],
    c_s: Float[Array, ""],
    m_half: Float[Array, ""],
    r_h: Float[Array, ""],
    G: Float[Array, ""],
    c_phi: float = C_PHI,
) -> Float[Array, ""]:
    r"""Composed μ_Φ → β₀ chain at the profile (half-mass) scale.

    B₀ → ρ₀ → v_A → ℳ_A → β₀, using ``m_half`` = mass within r_h and ρ₀ = mean density there.
    Analytically equals 6π²c_Φ²μ_Φ² r_h c_s²/(G m_half) (see module docstring).
    """
    b0 = mean_field_strength(mu_phi, m_half, r_h, G, c_phi)
    rho0 = mean_density(m_half, r_h)
    v_a = alfven_speed(b0, rho0)
    m_a = alfven_mach(mach, c_s, v_a)
    return plasma_beta(mach, m_a)
