r"""Shared scale-setting helpers for the gas-envelope profiles (ADR-0067, ADR-0069).

Both :class:`~gravoturb.profiles.bonnor_ebert.BonnorEbertProfile` and
:class:`~gravoturb.profiles.polytrope.PolytropeProfile` are ``r_h``-primary: the ODE is
solved once in dimensionless ``xi``, the half-mass point ``xi_h`` is located, and the
physical scale follows as ``r_0 = r_h / xi_h``. That inversion is shared here, and it has
to be correct in the GRADIENT as well as the value -- ``r_0`` multiplies into every
density, so a wrong ``d xi_h / d(params)`` corrupts the whole profile's derivatives.

**Why not plain bisection.** Bisection builds ``xi_h`` purely from arithmetic on the
bracket endpoints; the mass table enters only through the comparison ``m(mid) < m_half``,
which is a hard threshold with zero derivative. The forward value is excellent and
``d xi_h / dm`` is silently **zero**. That is the classic zero-gradient trap.

**Why not plain linear inversion.** ``monotone_inverse_interp`` *is* differentiable --
it computes ``t = (y - y0)/(y1 - y0)``, which carries the cotangent -- but it is only
first-order accurate in ``dxi``, while the rest of the solve is Tsit5-accurate.

**What we do instead.** Bisect for the value under ``stop_gradient``, then take a single
Newton step which carries the gradient by the implicit function theorem:

    xi_h = xi_b - (m(xi_b) - m_half) / (dm/dxi)(xi_b)

At convergence the correction is ~0, so the forward value stays bisection-accurate, while
differentiating the expression yields the correct implicit derivative
``d xi_h/dp = -(dm/dp) / (dm/dxi)``. The Newton derivative is the ODE's own analytic
``dm/dxi`` (``xi^2 e^{-psi}`` or ``xi^2 theta^n`` -- i.e. ``4 pi r^2 rho``), never a
finite difference.

This also removes the earlier ``strictly_increasing_prefix`` step: bisection needs only
non-decreasing ``m``, so the flat outer tail of a soft-equation-of-state polytrope (where
``dm/dxi = xi^2 theta^n`` falls below the solver's ``atol``) is harmless rather than
something to slice away with a concrete ``int()`` that would block tracing.
"""

import jax
import jax.numpy as jnp
from jaxstro.numerics.checks import try_concrete_bool
from jaxstro.numerics.interpolation import monotone_cubic_interp

# 60 halvings shrink any bracket in this problem far below float64 resolution.
_BISECT_STEPS = 60


def interp_flat(x, y, x_new):
    """PCHIP interpolation accepting an ``x_new`` of any shape (e.g. a 3-D grid)."""
    shape = x_new.shape
    return monotone_cubic_interp(x, y, x_new.reshape(-1)).reshape(shape)


def require(predicate, message: str) -> None:
    """Raise when ``predicate`` is concretely false; stay silent under tracing.

    Validation must not force concretization: ``float(r_h) <= 0`` would make the whole
    constructor untraceable and so undifferentiable. ``try_concrete_bool`` returns
    ``None`` for a tracer, which is the ecosystem's idiom for "cannot check here".
    """
    if try_concrete_bool(jnp.asarray(predicate)) is False:
        raise ValueError(message)


def _bisect(xi, m, m_half):
    """Locate ``m(xi) = m_half`` by bisection. Value only -- no usable derivative."""

    def step(bracket, _):
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        too_small = monotone_cubic_interp(xi, m, mid.reshape(1))[0] < m_half
        return (jnp.where(too_small, mid, lo), jnp.where(too_small, hi, mid)), None

    (lo, hi), _ = jax.lax.scan(step, (xi[0], xi[-1]), None, length=_BISECT_STEPS)
    return 0.5 * (lo + hi)


def half_mass_xi(xi, m, dm):
    """Locate ``xi_h`` where the enclosed mass reaches half the total.

    Args:
        xi: dimensionless radius grid.
        m: enclosed mass on that grid (non-decreasing).
        dm: the ODE's analytic ``dm/dxi`` on that grid.

    Returns:
        ``xi_h``, accurate to bisection precision and differentiable in ``m``/``dm``
        through the implicit function theorem (see the module docstring).
    """
    xi = jnp.asarray(xi)
    m = jnp.asarray(m)
    m_half = 0.5 * m[-1]

    xi_b = jax.lax.stop_gradient(_bisect(xi, m, m_half))

    m_at = monotone_cubic_interp(xi, m, xi_b.reshape(1))[0]
    dm_at = monotone_cubic_interp(xi, dm, xi_b.reshape(1))[0]
    return xi_b - (m_at - m_half) / dm_at
