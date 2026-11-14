# progenax/src/progenax/profiles/limepy.py
"""
General-g LIMEPY lowered-isothermal density (Gieles & Zocchi 2015).

This module generalizes the King (1966) lowered-Maxwellian to the continuous
truncation-parameter family of Gieles & Zocchi (2015, MNRAS 454, 576; "LIMEPY"),
of which Woolley (g=0), King (g=1), and Wilson (g=2) are integer members. The
single dimensionless density needed for the self-consistent Poisson solve is

    I_rho(W) = E_gamma(g + 3/2, W),

where E_gamma is the "lowered exponential" of their Eq. 2,

    E_gamma(a, x) = e^x P(a, x),   P(a, x) = gamma(a, x) / Gamma(a),

for a > 0, with E_gamma(0, x) = exp(x) on positive x. P is the regularized
lower incomplete gamma function. The implementation supplies a custom JVP
using a Poisson-mixture representation for the shape derivative; native JAX
gammainc alone does not provide the required derivative in both arguments.
The truncation index g therefore uses that explicit derivative rule.

Index note (verified against the held PDF, MNRAS 454, 576, p. 578): the density
uses g + 3/2, given DIRECTLY by the main-text Eq. 8,

    I_rho = (2/sqrt pi) int_0^W dk k^{1/2} E_gamma(g, W - k) = E_gamma(g + 3/2, W),

which follows from the E_gamma convolution identity (Eq. D11); an alternative
fractional-calculus derivation is in Appendix B. (There is NO "g + 1/2" misprint
in the main text -- Eqs. 8 and 11 both print g + 3/2 in the held PDF; an earlier
provenance note claiming a main-text typo was mistaken.) The 2018 erratum's
Eqs. 20/21 carry the same E_gamma(g + 3/2, W0) normalization. Two cross-checks:
  1. E_gamma(5/2, W) expands exactly to the King volume density
     e^W erf(sqrt W) - (2/sqrt pi) sqrt(W) (1 + 2W/3)  -> g=1 corner.
  2. The erratum (Eqs. 20/21) normalization denominator is E_gamma(g + 3/2, W0).

References:
    Gieles, M. & Zocchi, A. (2015), MNRAS, 454, 576 (Eqs. 1-9, App. B, D).
    Gieles & Zocchi (2018), MNRAS, 474, 3997 (erratum: Eqs. 20, 21, 41).
    King, I. R. (1966), AJ, 71, 64 (the g=1 corner).
"""

from functools import partial
from typing import Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero
from jaxstro.numerics import gauss_legendre_nodes
from jaxtyping import Array, ArrayLike, Bool, Float, PRNGKeyArray

from progenax.profiles._spherical_poisson import (
    poisson_second_derivative,
    regular_center_state,
)
from progenax.profiles.king import _find_tidal_radius, _require_tidal_crossing


@jax.custom_jvp
def lowered_exponential(
    a: Float[Array, ""], W: Float[Array, "..."]
) -> Float[Array, "..."]:
    """The lowered exponential E_gamma(a, W) of Gieles & Zocchi (2015) Eq. 2:

        E_gamma(a, W) = e^W              for a = 0   (the Woolley branch),
                      = e^W P(a, W)      for a > 0,

    with P(a, W) = gammainc(a, W) the regularized lower incomplete gamma function.
    For W <= 0 the function is 0 (no stars above the escape energy). The argument
    is clamped to W >= 0 before `gammainc` so the backward pass never sees a
    negative argument.

    The density uses a = g + 3/2 >= 3/2 (always the a > 0 branch); the velocity DF
    uses a = g, which is 0 for the Woolley (g=0) corner — hence the a = 0 branch.

    At a=0, the a derivative is the one-sided derivative from a>0.
    Interior values carry derivatives in both a and W.
    """
    return _lowered_exponential_value(a, W)


def _lowered_exponential_value(a, W):
    W_pos = jnp.where(W > 0.0, W, 0.0)
    a_safe = jnp.where(a > 0.0, a, 1.0)
    positive = jnp.exp(W_pos) * jax.scipy.special.gammainc(a_safe, W_pos)
    value = jnp.where(a > 0.0, positive, jnp.exp(W_pos))
    return jnp.where(W > 0.0, value, 0.0)


def _exp1_positive(x):
    """E1 for positive x, keeping the continued fraction away from x=0.

    The convergent small-x series has remainder below 1/(33*33!) on [0,1].
    Safe arguments keep inactive vectorized branches bounded.
    """
    small = jnp.minimum(x, 1.0)

    def term(k, state):
        power, total = state
        power = power * (-small) / k
        return power, total + power / k

    _, total = jax.lax.fori_loop(1, 33, term, (jnp.ones_like(x), jnp.zeros_like(x)))
    series = -jnp.euler_gamma - jnp.log(small) - total
    large = jax.scipy.special.exp1(jnp.maximum(x, 1.0))
    return jnp.where(x <= 1.0, series, large)


@partial(lowered_exponential.defjvp, symbolic_zeros=True)
def _lowered_exponential_jvp(primals, tangents):
    a, W = primals
    da, dW = tangents
    value = _lowered_exponential_value(a, W)
    tangent = jnp.zeros_like(value)
    if not isinstance(dW, SymbolicZero):
        tangent = jax.jvp(lambda w: _lowered_exponential_value(a, w), (W,), (dW,))[1]
    if not isinstance(da, SymbolicZero):
        # dP(a,W)/da at a=0+ is -E1(W), from 1/Gamma(a)~a in
        # the defining upper incomplete-gamma integral.
        safe = jnp.where(W > 0.0, W, 1.0)

        def boundary(_):
            e1 = _exp1_positive(safe)
            return jnp.where(W > 0.0, -jnp.exp(safe) * e1, 0.0)

        derivative = jax.lax.cond(
            a == 0.0,
            boundary,
            lambda _: jax.jvp(
                lambda aa: _lowered_exponential_value(aa, W), (a,), (jnp.ones_like(a),)
            )[1],
            operand=None,
        )
        tangent = tangent + da * derivative
    return value, tangent


def limepy_density_hat(
    W: Float[Array, "..."], g: Float[Array, ""]
) -> Float[Array, "..."]:
    """Unnormalized isotropic LIMEPY density I_rho(W) = E_gamma(g + 3/2, W).

    This is the dimensionless Poisson source (before normalization to the central
    value). At g=1 it equals the King volume density
    `king_lowered_maxwellian_density(W)` to float64 precision.

    Args:
        W: Dimensionless potential (scalar or array). W <= 0 returns 0.
        g: Truncation parameter (g=0 Woolley, g=1 King, g=2 Wilson; continuous).
           Finite extent also depends on W0 and anisotropy. The necessary
           condition g < 3.5 is not sufficient (Gieles & Zocchi 2015, Sec. 3.1.3).

    Returns:
        I_rho(W), the unnormalized LIMEPY volume density. Differentiable in (W, g).
    """
    return lowered_exponential(g + 1.5, W)


# ==============================================================================
# Anisotropic (Michie exponential angular-momentum) density — stable 1-D speed quadrature
# ==============================================================================

# Fixed setup constants; physical inputs remain differentiable.
_ANGLE_NODES, _ANGLE_WEIGHTS = gauss_legendre_nodes(64)
_ANGLE_T = (_ANGLE_NODES + 1.0) / 2.0
_ANGLE_WEIGHTS = _ANGLE_WEIGHTS / 2.0
_SPEED_NODES, _SPEED_WEIGHTS = gauss_legendre_nodes(128)
_SPEED_T = (_SPEED_NODES + 1.0) / 2.0
_SPEED_WEIGHTS = _SPEED_WEIGHTS / 2.0


def _angle_integral_T(beta: Float[Array, "..."]) -> Float[Array, "..."]:
    """Angular integral T(beta) = integral exp[-beta(1-c^2)] dc over [-1, 1].

    Here c is the velocity-direction cosine and beta = p^2 u^2 / 2 >= 0,
    with p = r/r_a and dimensionless speed u. Symmetry reduces the domain
    to [0, 1]. The change c = 1 - t/[1 + beta(1-t)] resolves the narrow
    angular peak at large beta using one smooth 64-node Gauss-Legendre rule.
    At the isotropic limit, T(0) = 2 and T'(0) = -4/3.
    """
    b = jnp.asarray(beta)[..., None]
    denominator = 1.0 + b * (1.0 - _ANGLE_T)
    y = _ANGLE_T / denominator
    return 2.0 * jnp.sum(
        _ANGLE_WEIGHTS * (1.0 + b) / denominator**2 * jnp.exp(-b * y * (2.0 - y)),
        axis=-1,
    )


def _aniso_density_scalar(
    W: Float[Array, ""], p: Float[Array, ""], g: Float[Array, ""]
) -> Float[Array, ""]:
    """Scalar anisotropic density rho_hat(W, p, g), p = r / r_a (unnormalized).

    The velocity-space integral of the LIMEPY DF with the Michie exponential angular-momentum anisotropy,
    reduced to a 1-D speed quadrature by closing the angle integral into T(beta):

        rho_hat(W, p, g) = int_0^{sqrt(2W)} u^2 E_gamma(g, W - u^2/2) T(p^2 u^2/2) du.

    At p=0, T=2 and this is sqrt(2 pi) times the isotropic limepy_density_hat (the
    same proportionality michie_density uses); at g=1 it equals michie_density(W, p).
    A 128-node Gauss-Legendre speed rule excludes the escape endpoint.
    Returns 0 for W <= 0.
    """
    W_pos = jnp.maximum(W, 1e-12)
    u = jnp.sqrt(2.0 * W_pos) * _SPEED_T
    E = lowered_exponential(g, W_pos - u**2 / 2.0)
    integrand = u**2 * E * _angle_integral_T(p**2 * u**2 / 2.0)
    rho = jnp.sqrt(2.0 * W_pos) * jnp.sum(_SPEED_WEIGHTS * integrand)
    return jnp.where(W > 0.0, rho, 0.0)


def limepy_density_aniso_hat(
    W: Float[Array, "..."], p: Float[Array, ""], g: Float[Array, ""]
) -> Float[Array, "..."]:
    """Anisotropic (Michie exponential angular-momentum) LIMEPY density rho_hat(W, p, g), p = r/r_a (scalar).

    Broadcasts over W (scalar p, g). Unnormalized (consistent constant across W, p);
    normalize by the central value rho_hat(W0, 0, g) as in the Poisson solve. At p=0
    it reduces to the isotropic limepy density (after normalization); at g=1 it equals
    `michie_density`. Differentiable in (W, p, g): p carries the anisotropy radius
    gradient, g the truncation-index gradient.
    """
    W_arr = jnp.asarray(W)
    flat = jnp.atleast_1d(W_arr).reshape(-1)
    out = jax.vmap(lambda w: _aniso_density_scalar(w, p, g))(flat)
    return out.reshape(W_arr.shape) if W_arr.ndim > 0 else out[0]


# Paired (W, p) -> rho vmap, for evaluating the anisotropic density on a grid where
# both the potential psi(xi) and the radius p = xi/ra_hat vary together.
_aniso_density_vec = jax.vmap(_aniso_density_scalar, in_axes=(0, 0, None))


# ==============================================================================
# Self-consistent Poisson solve (general-g; generalizes solve_king_profile)
# ==============================================================================


def _poisson_d2(xi, dpsi_dxi, rho_tilde):
    """King-radius Poisson curvature with its regular spherical center limit."""
    return poisson_second_derivative(xi, dpsi_dxi, rho_tilde)


def _limepy_poisson_rhs_iso(xi, y, args):
    """Isotropic LIMEPY Poisson RHS (fast path). args = (W0, g). At g=1 == King."""
    W0, g = args
    psi, dpsi_dxi = y[0], y[1]
    rho0 = limepy_density_hat(W0, g)
    rho_tilde = jnp.where(rho0 > 1e-300, limepy_density_hat(psi, g) / rho0, 0.0)
    return jnp.array([dpsi_dxi, _poisson_d2(xi, dpsi_dxi, rho_tilde)])


def _limepy_poisson_rhs_aniso(xi, y, args):
    """Anisotropic (Michie exponential angular-momentum) LIMEPY Poisson RHS. args = (W0, g, ra_hat). The source
    depends on xi explicitly through p = xi/ra_hat (as in solve_michie). At g=1 ==
    Michie-King. Central normalization uses the isotropic (p=0) central density."""
    W0, g, ra_hat = args
    psi, dpsi_dxi = y[0], y[1]
    p = xi / ra_hat
    rho0 = _aniso_density_scalar(W0, 0.0, g)
    rho_tilde = jnp.where(rho0 > 1e-300, _aniso_density_scalar(psi, p, g) / rho0, 0.0)
    return jnp.array([dpsi_dxi, _poisson_d2(xi, dpsi_dxi, rho_tilde)])


def solve_limepy_profile(
    W0: ArrayLike,
    g: ArrayLike,
    ra_hat: ArrayLike | None = None,
    xi_max: float = 300.0,
    n_points: int = 2000,
    *,
    adjoint=None,
) -> Tuple[
    Float[Array, "n_points"], Float[Array, "n_points"], Float[Array, "n_points"]
]:
    """Solve the general-g (optionally anisotropic) LIMEPY Poisson equation (diffrax).

    Integrates W(xi) from the centre (W=W0, dW/dxi=0) outward to the truncation
    radius where W -> 0. Generalizes `solve_king_profile` with the continuous
    truncation parameter g (g=0 Woolley, g=1 King, g=2 Wilson) and an optional
    Michie exponential angular-momentum radial anisotropy radius. At g=1: isotropic reproduces King,
    anisotropic reproduces the Michie-King model.

    JIT/grad-safe in (W0, g, ra_hat): n_points and xi_max are static; W0, g, ra_hat
    may be tracers and the ODE -> W(xi) path carries their gradients.

    Args:
        W0: Dimensionless central potential.
        g: Truncation parameter. Finite extent requires g < 3.5 and also
            depends on W0 and anisotropy; verify an actual zero crossing.
        ra_hat: Anisotropy radius in core-radius units, r_a/r_c. None (default) =
            isotropic (uses the fast isotropic density). Finite values add radial
            anisotropy (more extended; too-small ra_hat -> no finite tidal radius).
        xi_max: Max dimensionless radius (anisotropic models are more extended;
            pass a larger value, e.g. 800).
        n_points: Output grid size.

    Returns:
        3-tuple ``(xi_grid, psi_clamped, psi_raw)`` (mirrors ``solve_king_profile``):

        - ``xi_grid``: dimensionless radii.
        - ``psi_clamped``: ``max(psi, 0)`` -- the physical potential W(xi) >= 0
          used for density / CDF / mu / virial.
        - ``psi_raw``: the UNCLAMPED ODE solution (negative past the zero-crossing).
          Feed ``psi_raw`` to ``_find_tidal_radius`` so the crossing interpolation
          carries d(xi_t)/dW0 -- the clamp would zero psi at the crossing node and
          kill that gradient (audit Task 1.2b pattern). The forward xi_t value is
          identical either way; only the gradient differs.

    Use adjoint=None for checkpointed reverse AD (default), diffrax.ForwardMode()
    for forward AD, or diffrax.DirectAdjoint() for both. Keep this choice
    static under JIT; it does not change the physical equations or tolerances.
    """
    isotropic = ra_hat is None  # static (Python) -> picks the RHS at trace time
    rhs = _limepy_poisson_rhs_iso if isotropic else _limepy_poisson_rhs_aniso
    args = (W0, g) if isotropic else (W0, g, ra_hat)

    xi_span = (1e-6, xi_max)
    y0 = regular_center_state(W0, xi_span[0])
    term = diffrax.ODETerm(rhs)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-10)
    saveat = diffrax.SaveAt(ts=jnp.linspace(xi_span[0], xi_span[1], n_points))

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=xi_span[0],
        t1=xi_span[1],
        dt0=1e-4,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        adjoint=diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint,
        max_steps=100000,
    )
    xi_grid = solution.ts
    psi_end = solution.ys[-1, 0]  # W at xi_max (negative once truncated)
    psi_raw = solution.ys[:, 0]  # UNCLAMPED W(xi) (negative past r_t)
    psi_grid = jnp.maximum(psi_raw, 0.0)

    # Non-truncation guard (concrete inputs only; skipped under tracing). If W(xi_max)
    # has not crossed ~0, the model does not truncate within the ODE domain and r_t
    # would be silently PINNED to xi_max*r_c — a wrong tidal radius, and hence a wrong
    # mass CDF and DF velocity scale (mu / the virial scale become grid-dependent).
    # A positive finite-domain endpoint does not determine whether a larger domain
    # would establish a crossing, or classify the model's total mass or stability.
    if (
        isinstance(W0, (int, float))
        and isinstance(g, (int, float))
        and (isotropic or isinstance(ra_hat, (int, float)))
        and float(psi_end) > 1e-3 * W0
    ):
        if isotropic:
            raise ValueError(
                f"LIMEPY model (W0={W0}, g={g}) does not truncate within "
                f"xi_max={xi_max} (W(xi_max)={float(psi_end):.3g} > 0): r_t would be "
                f"silently PINNED to the domain edge (a wrong tidal radius). Increase "
                f"xi_max — the true xi_t can be large (e.g. W0=9, g=2 needs xi_t~2100). "
                f"Note g >= 3.5 models have no finite extent at all (Gieles & Zocchi "
                f"2015, Sec. 3.1.3)."
            )
        raise ValueError(
            f"Anisotropic LIMEPY model (W0={W0}, g={g}, r_a/r_c={ra_hat}) does not "
            f"truncate within xi_max={xi_max} (W(xi_max)={float(psi_end):.3f} > 0): the "
            f"finite domain has not established a tidal crossing. Check resolution "
            f"and a larger integration domain, and assess the anisotropy before "
            f"classifying the model's extent."
        )
    return xi_grid, psi_grid, psi_raw


# ==============================================================================
# LIMEPYProfile (SpatialProfile implementation; generalizes KingProfile)
# ==============================================================================


class LIMEPYProfile(eqx.Module):
    """General-g LIMEPY (lowered-isothermal) spherical density profile.

    Implements the SpatialProfile protocol. Generalizes `KingProfile` with a
    continuous truncation parameter g (g=0 Woolley, g=1 King, g=2 Wilson) and an
    optional Michie exponential angular-momentum radial anisotropy radius r_a. Corners: g=1
    isotropic == KingProfile; g=1 anisotropic == MichieProfile.

    The CDF is precomputed at construction for inverse-transform position sampling.

    Attributes:
        W0: Dimensionless central potential.
        g: Truncation parameter; g < 3.5 is necessary but not sufficient for finite extent.
        r_c: Core (King) radius [length units].
        r_a: Anisotropy radius [length units]; inf for the isotropic model.
        r_t: Truncation radius [length units], where W(r) -> 0.
        xi_grid, psi_grid: ODE solution W(xi) on a dimensionless grid.
        is_aniso: static flag selecting the anisotropic density path.
        _r_grid, _cdf_grid: precomputed mass CDF for sampling.
        r_t_is_pinned: Traced diagnostic for a stored potential with no nonpositive
            node. ``from_W0_rc`` rejects unresolved crossings, including under JIT;
            manual precomputed-grid construction retains this diagnostic.
    """

    W0: Float[Array, ""]
    g: Float[Array, ""]
    r_c: Float[Array, ""]
    r_a: Float[Array, ""]
    r_t: Float[Array, ""]
    xi_grid: Float[Array, "n_points"]
    psi_grid: Float[Array, "n_points"]
    _r_grid: Float[Array, "n_grid"]
    _cdf_grid: Float[Array, "n_grid"]
    r_t_is_pinned: Bool[Array, ""]
    is_aniso: bool = eqx.field(static=True)

    def __init__(
        self, W0, g, r_c, r_t, xi_grid, psi_grid, r_a=None, n_grid: int = 1000
    ):
        is_aniso = r_a is not None
        W0_arr = jnp.asarray(W0, dtype=jnp.float64)
        g_arr = jnp.asarray(g, dtype=jnp.float64)
        r_c_arr = jnp.asarray(r_c, dtype=jnp.float64)
        r_a_arr = jnp.asarray(jnp.inf if r_a is None else r_a, dtype=jnp.float64)
        r_t_arr = jnp.asarray(r_t, dtype=jnp.float64)
        xi_grid_arr = jnp.asarray(xi_grid, dtype=jnp.float64)
        psi_grid_arr = jnp.asarray(psi_grid, dtype=jnp.float64)

        # Sqrt-stretched grid r = r_t * u^2 (as in KingProfile / EFFProfile, audit R4/S2):
        # a linear grid under-resolves the core when r_t >> r_c (W0=12, g=1: r_t ~ 548 r_c,
        # so a 1000-pt linear grid has ~1 node inside 0.5 r_c -> ~+49% core-mass error).
        # u^2 concentrates nodes near the centre where the mass is; smooth in r_t, so the
        # sampling stays differentiable in r_t.
        r_grid = r_t_arr * jnp.linspace(0.0, 1.0, n_grid) ** 2
        xi_local = r_grid / r_c_arr
        psi_vals = jnp.interp(
            xi_local, xi_grid_arr, psi_grid_arr, left=W0_arr, right=0.0
        )

        if is_aniso:
            rho0 = _aniso_density_scalar(W0_arr, jnp.asarray(0.0), g_arr)
            rho_vals = _aniso_density_vec(psi_vals, r_grid / r_a_arr, g_arr)
            rho_grid = jnp.where(rho0 > 1e-300, rho_vals / rho0, 0.0)
        else:
            rho0 = limepy_density_hat(W0_arr, g_arr)
            rho_grid = jnp.where(
                rho0 > 1e-300, limepy_density_hat(psi_vals, g_arr) / rho0, 0.0
            )
        rho_grid = jnp.where(r_grid <= r_t_arr, rho_grid, 0.0)

        integrand = 4.0 * jnp.pi * r_grid**2 * rho_grid
        # Non-uniform trapezoid: the sqrt-stretched grid has per-interval widths, so weight
        # each interval by its own diff(r_grid) (a scalar dr would be wrong here).
        M_cum = jnp.concatenate(
            [
                jnp.zeros(1, dtype=integrand.dtype),
                jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(r_grid)),
            ]
        )
        cdf_grid = M_cum / (M_cum[-1] + 1e-30)

        # Traced diagnostic (mirrors KingProfile.r_t_is_pinned): the clamped psi_grid
        # has a node <= 0 iff the model truncated within the ODE domain. If none does,
        # r_t was pinned to the grid boundary — a wrong tidal radius. Self-consistent
        # factory construction rejects an unresolved crossing; the manual
        # precomputed-grid constructor retains this diagnostic without that check.
        r_t_is_pinned = jnp.logical_not(jnp.any(psi_grid_arr <= 0.0))

        object.__setattr__(self, "W0", W0_arr)
        object.__setattr__(self, "g", g_arr)
        object.__setattr__(self, "r_c", r_c_arr)
        object.__setattr__(self, "r_a", r_a_arr)
        object.__setattr__(self, "r_t", r_t_arr)
        object.__setattr__(self, "xi_grid", xi_grid_arr)
        object.__setattr__(self, "psi_grid", psi_grid_arr)
        object.__setattr__(self, "is_aniso", is_aniso)
        object.__setattr__(self, "_r_grid", r_grid)
        object.__setattr__(self, "_cdf_grid", cdf_grid)
        object.__setattr__(self, "r_t_is_pinned", r_t_is_pinned)

    @classmethod
    def from_W0_rc(
        cls,
        W0: float,
        g: float,
        r_c: float,
        r_a: float | None = None,
        xi_max: float = 300.0,
        n_ode_points: int = 2000,
        n_grid: int = 1000,
        *,
        adjoint=None,
    ) -> "LIMEPYProfile":
        """Self-consistent constructor: r_t derived from where W(xi) -> 0.

        r_a=None (default) is the isotropic model; a finite r_a adds Michie exponential angular-momentum radial
        anisotropy (more extended -- pass a larger xi_max, e.g. 800).

        JIT/grad-safe in (W0, g, r_a): the fixed (xi_max, n_ode_points) keep array
        sizes static, so the parameters may be tracers and profile-shape functionals
        carry their gradients. The truncation radius r_t is the interpolated
        zero-crossing of the UNCLAMPED psi, so d(r_t)/dW0 flows (as in KingProfile).

        The raw solution must be finite, start positive and bracket an actual
        tidal zero. An unresolved domain is rejected eagerly and under JIT.
        This does not classify its physical extent, mass or stability, and does
        not constrain the manual precomputed-grid constructor.

        Use adjoint=None for checkpointed reverse AD (default), diffrax.ForwardMode()
        for forward AD, or diffrax.DirectAdjoint() for both. Keep this choice
        static under JIT; it does not change the physical equations or tolerances.
        """
        ra_hat = None if r_a is None else r_a / r_c
        xi_grid, psi_grid, psi_raw = solve_limepy_profile(
            W0,
            g,
            ra_hat=ra_hat,
            xi_max=xi_max,
            n_points=n_ode_points,
            adjoint=adjoint,
        )
        # Feed UNCLAMPED psi_raw so d(r_t)/dW0 flows (clamped psi zeros the
        # crossing node's gradient). Forward r_t is the interpolated crossing.
        xi_t = _find_tidal_radius(xi_grid, psi_raw)
        r_t = r_c * xi_t
        prof = cls(
            W0=W0,
            g=g,
            r_c=r_c,
            r_t=r_t,
            xi_grid=xi_grid,
            psi_grid=psi_grid,
            r_a=r_a,
            n_grid=n_grid,
        )
        _require_tidal_crossing(psi_raw)
        return prof

    def sample_positions(
        self, masses: Float[Array, "N"], key: PRNGKeyArray
    ) -> Float[Array, "N 3"]:
        """Sample isotropic positions via the precomputed inverse mass CDF."""
        N = len(masses)
        key, sub = jax.random.split(key)
        radii = self._sample_radii(sub, N)
        key, sub = jax.random.split(key)
        theta = jnp.arccos(1.0 - 2.0 * jax.random.uniform(sub, shape=(N,)))
        key, sub = jax.random.split(key)
        phi = 2.0 * jnp.pi * jax.random.uniform(sub, shape=(N,))
        x = radii * jnp.sin(theta) * jnp.cos(phi)
        y = radii * jnp.sin(theta) * jnp.sin(phi)
        z = radii * jnp.cos(theta)
        return jnp.stack([x, y, z], axis=1)

    def _sample_radii(self, key: PRNGKeyArray, N: int) -> Float[Array, "N"]:
        u = jax.random.uniform(key, shape=(N,))
        r_sampled = jnp.interp(u, self._cdf_grid, self._r_grid)
        return jnp.clip(r_sampled, 0.0, self.r_t)

    def density(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized radial density rho(r)/rho_0 (general-g; anisotropic if r_a set)."""
        r_arr = jnp.asarray(r)
        psi_vals = jnp.interp(
            r_arr / self.r_c, self.xi_grid, self.psi_grid, left=self.W0, right=0.0
        )
        if self.is_aniso:
            rho0 = _aniso_density_scalar(self.W0, jnp.asarray(0.0), self.g)
            flat_psi = jnp.atleast_1d(psi_vals).reshape(-1)
            flat_p = jnp.atleast_1d(r_arr / self.r_a).reshape(-1)
            rho = (_aniso_density_vec(flat_psi, flat_p, self.g) / rho0).reshape(
                r_arr.shape
            )
        else:
            rho0 = limepy_density_hat(self.W0, self.g)
            rho = jnp.where(
                rho0 > 1e-300, limepy_density_hat(psi_vals, self.g) / rho0, 0.0
            )
        return jnp.where(r_arr <= self.r_t, rho, 0.0)

    def characteristic_radius(self) -> Float[Array, ""]:
        """Truncation radius r_t (the model's outer scale)."""
        return self.r_t


__all__ = [
    "lowered_exponential",
    "limepy_density_hat",
    "limepy_density_aniso_hat",
    "solve_limepy_profile",
    "LIMEPYProfile",
]
