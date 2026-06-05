"""PN11 classical critical density for star formation (hydrodynamic case).

Padoan & Nordlund 2011, ApJ 730, 40, Section 2 (non-magnetized / HD limit),
verified against the held PDF (docs/core-papers/Padoan_2011_ApJ_730_40.pdf):

    Eq. 8   rho_cr/rho_0 = 0.067 theta^{-2} alpha_vir M^2
    Eq. 11  with theta = 0.35  ->  rho_cr/rho_0 = 0.547 alpha_vir M^2

theta = 0.35 is the turbulence integral-scale fraction adopted by PN11 (their
Section 2, after Wang & George 2002). This is a clearly-labelled CLASSICAL
ALTERNATIVE to the BM19 transition density s_t, not the default path for the
gravoturb_fdf pipeline. The full MHD critical density (PN11 Eq. 18) is not
implemented here.

JAX-native and differentiable in (mach, alpha_vir).
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

# PN11 adopted turbulence integral-scale fraction (their Section 2).
THETA_PN11 = 0.35


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
