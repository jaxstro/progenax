"""PN11 classical critical density for star formation (hydrodynamic case).

Padoan & Nordlund 2011, ApJ 730, 40, Section 2 (non-magnetized / HD limit),
verified against the held PDF (docs/core-papers/Padoan_2011_ApJ_730_40.pdf):

    Eq. 8   rho_cr/rho_0 = 0.067 theta^{-2} alpha_vir M^2
    Eq. 11  with theta = 0.35  ->  rho_cr/rho_0 = 0.547 alpha_vir M^2

theta = 0.35 is the turbulence integral-scale fraction adopted by PN11 (their
Section 2, after Wang & George 2002). This is a clearly-labelled CLASSICAL
ALTERNATIVE to the BM19 transition density s_t, not the default path for the
gravoturb pipeline. The full MHD critical density (PN11 Eq. 18) is not
implemented here.

JAX-native and differentiable in (mach, alpha_vir).
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

# PN11 adopted turbulence integral-scale fraction (their Section 2).
THETA_PN11 = 0.35


def virial_parameter(
    mass: Float[Array, ""],
    radius: Float[Array, ""],
    sigma_v: Float[Array, ""],
    G: float,
) -> Float[Array, ""]:
    r"""Cloud virial parameter alpha_vir = 5 sigma_v^2 R / (G M) (Bertoldi & McKee 1992).

    The dimensionless ratio of (twice the) turbulent kinetic to gravitational energy for a
    uniform sphere; ``alpha_vir = 1`` is virial equilibrium, ``< 1`` bound/collapsing,
    ``>> 1`` unbound. Pass the same unit system through ``(mass, radius, sigma_v, G)``
    (e.g. M_sun, pc, km/s with G = 4.3009e-3 pc (km/s)^2 M_sun^-1).

    Grounding (Heyer et al. 2009, ApJ 699, 1092): Galactic GMCs follow
    ``v_0 = sigma_v/R^{1/2} ∝ Sigma^{1/2}``, which makes alpha_vir ≈ const (~1-2) across
    surface density — i.e. clouds self-regulate to virial equilibrium. This is the
    correct grounding; the older code's ``alpha_vir ∝ Sigma^{-1}`` assumed Larson's
    (refuted) constant v_0 and is NOT used. ``critical_overdensity_pn11`` still accepts
    ``alpha_vir`` directly, so this helper is optional.

    JAX-native, differentiable.
    """
    return 5.0 * sigma_v**2 * radius / (G * mass)


def critical_overdensity_pn11(
    mach: Float[Array, ""],
    alpha_vir: Float[Array, ""],
    theta: float = THETA_PN11,
) -> Float[Array, ""]:
    r"""Critical overdensity rho_cr/rho_0 for collapse (PN11 Eq. 8).

    .. math:: \rho_\mathrm{cr}/\rho_0 = 0.067\,\theta^{-2}\,\alpha_\mathrm{vir}\,\mathcal{M}^2

    With theta = 0.35 the prefactor is 0.067 * 0.35^{-2} = 0.547 (PN11 Eq. 11).
    """
    return 0.067 * theta**-2 * alpha_vir * mach**2


def s_crit_pn11(
    mach: Float[Array, ""],
    alpha_vir: Float[Array, ""],
    theta: float = THETA_PN11,
) -> Float[Array, ""]:
    r"""Critical log-density s_crit = ln(rho_cr/rho_0) (PN11 Eq. 8).

    .. math:: s_\mathrm{crit} = \ln\!\left(0.067\,\theta^{-2}\,\alpha_\mathrm{vir}\,\mathcal{M}^2\right)

    The classical-alternative analogue of the BM19 transition density s_t.
    """
    return jnp.log(critical_overdensity_pn11(mach, alpha_vir, theta))
