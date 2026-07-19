r"""Differentiable Lane-Emden solver core for gas-cloud envelopes (ADR-0065, ADR-0067).

Two branches of the same self-gravitating-sphere problem, kept as two right-hand sides
rather than one parametrized family (ADR-0065):

+-----------+---------------------------------------+--------------------------------+
|           | isothermal (Bonnor-Ebert)             | polytropic (P = K rho^gamma)   |
+===========+=======================================+================================+
| ODE       | psi'' + (2/xi) psi' = e^{-psi}        | theta'' + (2/xi) theta' = -th^n|
| origin    | psi(0) = 0, psi'(0) = 0               | theta(0) = 1, theta'(0) = 0    |
| density   | rho = rho_c e^{-psi}                  | rho = rho_c theta^n            |
| mass      | m(xi) = xi^2 psi'                     | m(xi) = -xi^2 theta'           |
| edge      | none -- set by external pressure      | xi_1, the first zero of theta  |
+-----------+---------------------------------------+--------------------------------+

The isothermal case is the n -> infinity *singular limit* of the polytropic one, not a
large value of ``n``: under the change of variables the source term ``theta^n`` becomes
``e^{-psi}``. Evaluating ``theta^n`` at large n is both numerically hopeless and
non-differentiable in n, which is why the two live side by side here.

Numerics are borrowed, not hand-rolled (ADR-0067): diffrax ``Tsit5`` with
``PIDController(rtol=1e-8, atol=1e-10)``, exactly the idiom already used by
``progenax.profiles.king`` / ``michie`` / ``limepy``. Gradients are known to flow
through this solve (see ``king.py:300``).

**Origin start.** The ODEs are singular at xi=0 (the ``2/xi`` term), so integration
starts at ``XI_0 = 1e-6`` seeded with the series expansion rather than the bare
boundary condition. For the isothermal branch, substituting
``psi = a2 xi^2 + a4 xi^4`` into the ODE gives ``6 a2 = 1`` and ``20 a4 = -a2``, i.e.

    psi(xi) = xi^2/6 - xi^4/120 + O(xi^6),      psi'(xi) = xi/3 - xi^3/30 + O(xi^5)

and the polytropic branch has the analogous ``theta = 1 - xi^2/6 + n xi^4/120``.

JAX-native and differentiable in ``n``; ``xi_max`` and ``n_points`` are STATIC (they
size the output grid), matching the King convention.
"""

from typing import NamedTuple

import diffrax
import jax.numpy as jnp
import optimistix
from jaxtyping import Array, Float

# Integration starts here rather than at the singular origin (see module docstring).
XI_0 = 1e-6

# diffrax settings -- identical to progenax.profiles.king (ADR-0067).
_RTOL = 1e-8
_ATOL = 1e-10
_DT0 = 1e-4
_MAX_STEPS = 100_000


class LaneEmdenSolution(NamedTuple):
    """Tabulated Lane-Emden solution on a dimensionless radius grid.

    Attributes:
        xi: dimensionless radius, strictly increasing, starting at ``XI_0``.
        y: ``psi`` (isothermal) or ``theta`` (polytropic).
        dy: the first derivative of ``y`` with respect to ``xi``.
        m: dimensionless enclosed mass -- ``xi^2 psi'`` or ``-xi^2 theta'``. Strictly
            increasing wherever the density is positive, so it is a legitimate input to
            ``jaxstro.numerics.rootfinding.monotone_inverse_interp``.
    """

    xi: Float[Array, " n"]
    y: Float[Array, " n"]
    dy: Float[Array, " n"]
    m: Float[Array, " n"]


def _isothermal_rhs(xi, state, args):
    """``psi'' + (2/xi) psi' = e^{-psi}`` as a first-order system."""
    psi, u = state
    return jnp.array([u, jnp.exp(-psi) - 2.0 * u / xi])


def _polytropic_rhs(xi, state, args):
    """``theta'' + (2/xi) theta' = -theta^n`` as a first-order system.

    ``theta`` is floored at zero inside the power: past the first zero the polytropic
    source term is not physical, and ``theta^n`` for non-integer ``n`` would be NaN.
    Callers should terminate at ``xi_1`` (see :func:`polytrope_xi1`) rather than rely on
    the floored continuation.
    """
    (n,) = args
    theta, u = state
    theta_pos = jnp.maximum(theta, 0.0)
    return jnp.array([u, -(theta_pos**n) - 2.0 * u / xi])


def _isothermal_y0():
    """Series-seeded state at ``XI_0``: ``psi = xi^2/6 - xi^4/120``."""
    xi = XI_0
    psi = xi**2 / 6.0 - xi**4 / 120.0
    dpsi = xi / 3.0 - xi**3 / 30.0
    return jnp.array([psi, dpsi])


def _polytropic_y0(n):
    """Series-seeded state at ``XI_0``: ``theta = 1 - xi^2/6 + n xi^4/120``."""
    xi = XI_0
    theta = 1.0 - xi**2 / 6.0 + n * xi**4 / 120.0
    dtheta = -xi / 3.0 + n * xi**3 / 30.0
    return jnp.array([theta, dtheta])


def _solve(term, y0, xi_max, n_points, args):
    """Run the shared diffrax solve and return ``(xi, y, dy)``."""
    saveat = diffrax.SaveAt(ts=jnp.linspace(XI_0, xi_max, n_points))
    solution = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0=XI_0,
        t1=xi_max,
        dt0=_DT0,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=_RTOL, atol=_ATOL),
        max_steps=_MAX_STEPS,
    )
    return solution.ts, solution.ys[:, 0], solution.ys[:, 1]


def solve_isothermal(xi_max: float, n_points: int = 2000) -> LaneEmdenSolution:
    r"""Solve the isothermal Lane-Emden equation for a Bonnor-Ebert sphere.

    Args:
        xi_max: dimensionless truncation radius. Unlike a polytrope this is a genuine
            physical input -- an isothermal sphere has no zero, so its edge is set by the
            confining external pressure (ADR-0066).
        n_points: size of the output grid (STATIC; sets the ``linspace`` length).

    Returns:
        :class:`LaneEmdenSolution` with ``y = psi`` and ``m = xi^2 psi'``.
    """
    xi, psi, dpsi = _solve(
        diffrax.ODETerm(_isothermal_rhs), _isothermal_y0(), xi_max, n_points, None
    )
    return LaneEmdenSolution(xi=xi, y=psi, dy=dpsi, m=xi**2 * dpsi)


def solve_polytrope(n, xi_max: float, n_points: int = 2000) -> LaneEmdenSolution:
    r"""Solve the polytropic Lane-Emden equation of index ``n``.

    Args:
        n: polytropic index (traced/differentiable). Relates to the adiabatic index by
            ``n = 1/(gamma - 1)``.
        xi_max: upper limit of integration. For ``n < 5`` the physical solution ends at
            ``xi_1 < xi_max`` (see :func:`polytrope_xi1`); beyond that the returned
            ``theta`` is the floored continuation and is not physical.
        n_points: size of the output grid (STATIC).

    Returns:
        :class:`LaneEmdenSolution` with ``y = theta`` and ``m = -xi^2 theta'``.
    """
    n = jnp.asarray(n, dtype=float)
    xi, theta, dtheta = _solve(
        diffrax.ODETerm(_polytropic_rhs), _polytropic_y0(n), xi_max, n_points, (n,)
    )
    return LaneEmdenSolution(xi=xi, y=theta, dy=dtheta, m=-(xi**2) * dtheta)


def polytrope_xi1(n, xi_search_max: float = 50.0) -> Float[Array, ""]:
    r"""Locate ``xi_1``, the first zero of ``theta`` -- the polytrope's outer edge.

    Uses ``diffrax.Event`` with an optimistix root finder rather than an ``argmin`` over
    a grid: the event root is differentiable in ``n`` through the implicit function
    theorem, whereas ``argmin`` has no gradient at all (ADR-0067).

    Only defined for ``n < 5``. At ``n >= 5`` the solution is strictly positive and the
    sphere has infinite extent, so no event fires and the integration runs to
    ``xi_search_max``; callers must reject that regime (``PolytropeProfile`` requires
    ``gamma > 1.2``).

    Args:
        n: polytropic index (traced/differentiable).
        xi_search_max: give-up radius for the search (STATIC).

    Returns:
        Scalar ``xi_1``.
    """
    n = jnp.asarray(n, dtype=float)

    def _theta_is_zero(t, y, args, **kwargs):
        return y[0]

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_polytropic_rhs),
        diffrax.Tsit5(),
        t0=XI_0,
        t1=xi_search_max,
        dt0=_DT0,
        y0=_polytropic_y0(n),
        args=(n,),
        stepsize_controller=diffrax.PIDController(rtol=_RTOL, atol=_ATOL),
        event=diffrax.Event(
            _theta_is_zero, root_finder=optimistix.Newton(rtol=_RTOL, atol=_ATOL)
        ),
        max_steps=_MAX_STEPS,
    )
    return solution.ts[-1]
