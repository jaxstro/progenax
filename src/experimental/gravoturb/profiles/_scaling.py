r"""Shared scale-setting helpers for the gas-envelope profiles (ADR-0067).

Both :class:`~gravoturb.profiles.bonnor_ebert.BonnorEbertProfile` and
:class:`~gravoturb.profiles.polytrope.PolytropeProfile` are ``r_h``-primary: the ODE is
solved once in dimensionless ``xi``, the half-mass point ``xi_h`` is located, and the
physical scale follows as ``r_0 = r_h / xi_h``. That inversion is shared here.
"""

import jax.numpy as jnp
from jaxstro.numerics.interpolation import monotone_cubic_interp
from jaxstro.numerics.rootfinding import monotone_inverse_interp

_BISECT_STEPS = 40


def interp_flat(x, y, x_new):
    """PCHIP interpolation accepting an ``x_new`` of any shape (e.g. a 3-D grid)."""
    shape = x_new.shape
    return monotone_cubic_interp(x, y, x_new.reshape(-1)).reshape(shape)


def strictly_increasing_prefix(m) -> int:
    """Length of the leading run of ``m`` that is strictly increasing.

    The enclosed-mass table saturates near the outer edge of a soft-equation-of-state
    polytrope: ``dm/dxi = xi^2 theta^n``, and with ``theta -> 0`` raised to a large power
    the outermost shells contribute less mass than the ODE's own ``atol``. Those cells are
    genuinely flat to machine precision -- correct physics, not solver error -- so the
    half-mass inversion must simply stay out of them rather than pretend they are
    monotone.
    """
    d = jnp.diff(jnp.asarray(m))
    bad = jnp.where(d <= 0.0)[0]
    return int(bad[0]) + 1 if bad.size > 0 else int(jnp.asarray(m).shape[0])


def half_mass_xi(xi, m, *, context: str):
    """Locate ``xi_h`` where the enclosed mass reaches half the total.

    Inverts on the strictly-increasing prefix of ``m`` (see
    :func:`strictly_increasing_prefix`), brackets with the LINEAR
    ``monotone_inverse_interp``, then bisects against the PCHIP ``m(xi)`` so ``xi_h`` is
    not left first-order accurate while the rest of the solve is Tsit5-accurate
    (ADR-0067).

    Raises:
        ValueError: if the half-mass point falls outside the usable range. That means the
            ODE domain is mis-sized, and ``monotone_inverse_interp`` would silently clamp
            to an endpoint -- ADR-0067 requires asserting instead.
    """
    xi = jnp.asarray(xi)
    m = jnp.asarray(m)
    m_total = m[-1]
    m_half = 0.5 * m_total

    stop = strictly_increasing_prefix(m)
    xi_usable, m_usable = xi[:stop], m[:stop]

    if not (float(m_usable[0]) < float(m_half) < float(m_usable[-1])):
        raise ValueError(
            f"half-mass point m={float(m_half):.6e} lies outside the strictly-increasing "
            f"range [{float(m_usable[0]):.6e}, {float(m_usable[-1]):.6e}] "
            f"({stop}/{m.shape[0]} usable points) for {context}. The ODE domain is "
            "mis-sized; refusing to clamp."
        )

    xi_lo = monotone_inverse_interp(xi_usable, m_usable, m_half)
    dxi = xi[1] - xi[0]
    lo, hi = xi_lo - dxi, xi_lo + dxi
    for _ in range(_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        too_small = monotone_cubic_interp(xi, m, mid.reshape(1))[0] < m_half
        lo = jnp.where(too_small, mid, lo)
        hi = jnp.where(too_small, hi, mid)
    return 0.5 * (lo + hi)
