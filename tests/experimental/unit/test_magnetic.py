"""Unit tests for the gravoturb magnetized-turbulence scalar chain (P1a; ADR-0060).

The hydro density-PDF width is sigma_s^2 = ln(1 + b^2 M^2) (Burkhart & Mocz 2019). Magnetic
support suppresses it via the Molina et al. 2012 / Federrath & Klessen 2012 factor
beta/(beta+1), with the plasma beta beta_0 = 2 (M_A / M_s)^2 (F&K12, explicit). The primary
physical knob is the mass-to-flux ratio mu_phi (ADR-0060); B_0, M_A, beta_0 are DERIVED from
mu_phi + the cloud at the profile (half-mass) scale, using the critical constant
c_phi = 0.17/sqrt(G) (PN11 Eq. 16, Tomisaka et al. 1988; ADR-0059).

Verified against papers in hand: docs/core-papers/Molina_2012_arXiv_1203.2117.pdf (Eq. for the
intermediate B ∝ ρ^{1/2} case), Federrath_2012_ApJ_761_156.pdf (Eqs. 4-5, plasma-beta def),
Padoan_2011_ApJ_730_40.pdf (Eq. 16, critical mass-to-flux).

All core functions are UNIT-AGNOSTIC: pass a consistent (G, mass, length, velocity) set and
beta_0/sigma_s^2 come out dimensionless. The km/s -> pc/Myr conversion for c_s happens at the
spec/builder boundary, not here (mirrors VelocitySpec.mode='physical').
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gravoturb.theory.density_pdf import sigma_s_squared  # hydro reference

pytestmark = pytest.mark.experimental

# Critical mass-to-flux coefficient c_phi in (M/Phi)_crit = c_phi / sqrt(G) (ADR-0059).
_C_PHI = 0.17


def _beta0_closed_form(mu_phi, mach, c_s, m_half, r_h, G, c_phi=_C_PHI):
    """Independent analytic reduction of the mu_phi -> beta_0 chain (test oracle).

    With Phi = B_0 * pi r_h^2, (M/Phi) = mu_phi * c_phi/sqrt(G), rho_0 = m_half/((4/3)pi r_h^3),
    v_A^2 = B_0^2/(4 pi rho_0), and beta_0 = 2 c_s^2 / v_A^2, the whole chain reduces to
        beta_0 = 6 pi^2 c_phi^2 mu_phi^2 r_h c_s^2 / (G m_half).
    """
    return 6.0 * np.pi**2 * c_phi**2 * mu_phi**2 * r_h * c_s**2 / (G * m_half)


# --------------------------------------------------------------------------- #
# 1. Magnetic sigma_s^2 = ln(1 + b^2 M^2 beta/(beta+1))
# --------------------------------------------------------------------------- #
def test_magnetic_sigma_s_recovers_hydro_as_beta_to_infinity():
    from gravoturb.realization.magnetic import sigma_s_squared_magnetic

    mach, b = 10.0, 0.4
    hydro = sigma_s_squared(mach, b)
    mag = sigma_s_squared_magnetic(mach, b, beta0=1e10)
    assert jnp.abs(mag - hydro) < 1e-6


def test_magnetic_sigma_s_suppressed_below_hydro_for_finite_beta():
    from gravoturb.realization.magnetic import sigma_s_squared_magnetic

    mach, b = 10.0, 0.4
    hydro = sigma_s_squared(mach, b)
    mag = sigma_s_squared_magnetic(mach, b, beta0=1.0)
    assert mag < hydro  # magnetic pressure cushions density fluctuations


def test_magnetic_sigma_s_analytic_anchor():
    from gravoturb.realization.magnetic import sigma_s_squared_magnetic

    # b=0.4, M=10, beta0=1 -> factor=1/2, arg=1+0.16*100*0.5=9, ln 9 = 2.1972245773...
    val = sigma_s_squared_magnetic(10.0, 0.4, beta0=1.0)
    assert jnp.abs(val - jnp.log(9.0)) < 1e-12


# --------------------------------------------------------------------------- #
# 2. plasma beta = 2 (M_A / M_s)^2   (F&K12, explicit)
# --------------------------------------------------------------------------- #
def test_plasma_beta_definition():
    from gravoturb.realization.magnetic import plasma_beta

    # M_s=10, M_A=5 -> beta = 2*(5/10)^2 = 0.5
    assert jnp.abs(plasma_beta(10.0, 5.0) - 0.5) < 1e-12


# --------------------------------------------------------------------------- #
# 3. mu_phi -> B_0 -> v_A -> M_A -> beta_0 chain (profile/half-mass scale)
# --------------------------------------------------------------------------- #
def test_mean_density_uniform_sphere():
    from gravoturb.realization.magnetic import mean_density

    # m=1, r=1 -> 1/((4/3)pi) = 3/(4pi)
    assert jnp.abs(mean_density(1.0, 1.0) - 3.0 / (4.0 * np.pi)) < 1e-12


def test_beta_from_mass_to_flux_matches_closed_form():
    from gravoturb.realization.magnetic import beta_from_mass_to_flux

    cases = [
        # (mu_phi, mach, c_s, m_half, r_h, G)
        (1.0, 10.0, 1.0, 1.0, 1.0, 1.0),
        (2.0, 8.0, 0.3, 500.0, 1.5, 4.5e-3),   # STELLAR-ish G, c_s already in pc/Myr
        (0.5, 12.0, 0.2, 1e4, 2.0, 4.5e-3),
    ]
    for mu_phi, mach, c_s, m_half, r_h, G in cases:
        got = beta_from_mass_to_flux(mu_phi, mach, c_s, m_half, r_h, G)
        want = _beta0_closed_form(mu_phi, mach, c_s, m_half, r_h, G)
        assert jnp.abs(got - want) / want < 1e-10, (mu_phi, got, want)


def test_beta_scales_as_mu_phi_squared():
    from gravoturb.realization.magnetic import beta_from_mass_to_flux

    args = dict(mach=10.0, c_s=0.3, m_half=1e3, r_h=1.5, G=4.5e-3)
    b1 = beta_from_mass_to_flux(1.0, **args)
    b2 = beta_from_mass_to_flux(2.0, **args)
    assert jnp.abs(b2 / b1 - 4.0) < 1e-9  # weaker field (higher mu_phi) -> higher beta


def test_full_chain_recovers_hydro_as_mu_phi_large():
    from gravoturb.realization.magnetic import beta_from_mass_to_flux, sigma_s_squared_magnetic

    mach, b = 10.0, 0.4
    beta0 = beta_from_mass_to_flux(1e6, mach=mach, c_s=0.3, m_half=1e3, r_h=1.5, G=4.5e-3)
    mag = sigma_s_squared_magnetic(mach, b, beta0)
    assert jnp.abs(mag - sigma_s_squared(mach, b)) < 1e-5


# --------------------------------------------------------------------------- #
# 4. Differentiability (the critical gate): AD through mu_phi vs finite difference
# --------------------------------------------------------------------------- #
def test_sigma_s_magnetic_grad_through_mu_phi_matches_fd():
    from gravoturb.realization.magnetic import beta_from_mass_to_flux, sigma_s_squared_magnetic

    mach, b = 10.0, 0.4

    def sigma_of_mu(mu_phi):
        beta0 = beta_from_mass_to_flux(mu_phi, mach=mach, c_s=0.3, m_half=1e3, r_h=1.5, G=4.5e-3)
        return sigma_s_squared_magnetic(mach, b, beta0)

    mu0 = 1.3
    ad = jax.grad(sigma_of_mu)(mu0)
    h = 1e-4
    fd = (sigma_of_mu(mu0 + h) - sigma_of_mu(mu0 - h)) / (2 * h)
    assert jnp.isfinite(ad)
    assert ad > 0.0  # higher mu_phi (weaker field) -> larger sigma_s^2 (toward hydro)
    assert jnp.abs(ad - fd) / jnp.abs(fd) < 1e-5
