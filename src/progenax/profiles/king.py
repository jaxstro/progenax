# progenax/src/progenax/profiles/king.py
"""
King (1966) density and potential profiles.

Implements the King (1966) dynamical model for star clusters with tidal cutoff,
widely used for globular cluster simulations. This module contains the core
profile functions (density, potential, cumulative mass) and the KingProfile
class implementing the SpatialProfile protocol.

The King model is characterized by:
- Lowered Maxwellian distribution function
- Tidal truncation at radius r_t
- Concentration parameter W0 (dimensionless central potential)

References:
    King, I. R. (1966), "The Structure of Star Clusters. III. Some Simple
    Dynamical Models", AJ, 71, 64

    Binney & Tremaine (2008), "Galactic Dynamics" (2nd ed.), Section 4.3

Notes:
    - W0 is the dimensionless central potential; larger W0 = more centrally
      concentrated. Galactic globular clusters span W0 ~ 5-9.
    - King (1966) Table II tabulates models up to W0 = 15; as W0 increases the
      model approaches the singular isothermal sphere (Binney & Tremaine 2008).
"""

import math
import warnings
from typing import SupportsFloat, Tuple, cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, PRNGKeyArray

from progenax.numerics import require_positive
from progenax.profiles._spherical_poisson import (
    poisson_second_derivative,
    regular_center_state,
)


def _is_concrete(x) -> bool:
    """True iff x is a concrete value (codebase idiom; cf. _auto_ode_domain)."""
    try:
        float(x)
        return True
    except (
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerArrayConversionError,
        TypeError,
    ):
        return False


# ==============================================================================
# King lowered-Maxwellian density (Poisson source)
# ==============================================================================


def king_lowered_maxwellian_density(W: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    King (1966) lowered-Maxwellian dimensionless *volume* density rho_hat(W).

        rho_hat(W) = e^W erf(sqrt(W)) - (2/sqrt(pi)) sqrt(W) (1 + 2W/3)

    This is the velocity-space integral of the King DF f(E) ∝ (e^{E/sigma^2} - 1):

        rho(W) ∝ int_0^{sqrt(2W)} v^2 (e^{W - v^2/2} - 1) dv,

    and equals that direct integral divided by sqrt(pi/2). It is the
    3-D *volume* density that appears in the King Poisson equation. The King
    expression erf(sqrt(W)) - (2/sqrt(pi)) sqrt(W) e^{-W} is the Maxwell speed
    cumulative probability at escape speed, not a projected spatial density.
    Surface density requires line-of-sight integration of the volume density.

    Gradient-safe at W=0: rho_hat(0)=0 and d(rho_hat)/dW(0)=0; clamp before
    sqrt/exp so the backward pass never differentiates sqrt at 0.

    Args:
        W: Dimensionless potential (scalar or array). W <= 0 returns 0.

    Returns:
        rho_hat(W), the (unnormalized) King volume density.

    References:
        King (1966), AJ, 71, 64
        Binney & Tremaine (2008), "Galactic Dynamics", 2nd ed., Eq. 4.131
    """
    W_pos = jnp.where(W > 0.0, W, 1.0)  # never feed 0/negative to sqrt
    sqrt_W = jnp.sqrt(W_pos)
    rho_pos = jnp.exp(W_pos) * jax.scipy.special.erf(sqrt_W) - (
        2.0 / jnp.sqrt(jnp.pi)
    ) * sqrt_W * (1.0 + 2.0 * W_pos / 3.0)
    # The closed form cancels terms of order sqrt(W) to obtain W^(5/2).
    # Its relative roundoff grows as eps/W^2. Integrating the DF series gives
    # rho = W^(5/2)/Gamma(7/2) sum_k W^k / (7/2)_k. On [0,0.1], the first
    # omitted term after k=11 is below 1.1e-23 of the leading term.
    small_W = jnp.where(W_pos <= 0.1, W_pos, 0.1)
    term = jnp.ones_like(small_W)
    series = term
    for k in range(1, 12):
        term = term * small_W / (2.5 + k)
        series = series + term
    rho_series = 8.0 / (15.0 * jnp.sqrt(jnp.pi)) * small_W**2.5 * series
    rho_pos = jnp.where(W_pos <= 0.1, rho_series, rho_pos)
    return jnp.where(W > 0.0, rho_pos, 0.0)


# ==============================================================================
# King Profile ODE Solution (Poisson Equation)
# ==============================================================================


def _king_poisson_rhs(
    xi: ArrayLike, y: Float[Array, "2"], args: tuple
) -> Float[Array, "2"]:
    """
    Right-hand side of King's dimensionless Poisson equation.

    The King (1966) model satisfies:
        d^2 psi/d xi^2 + (2/xi) d psi/d xi = -9 rho_tilde(psi)

    where xi = r/r_c is dimensionless radius and rho_tilde(psi) is the dimensionless
    density from integrating the King distribution function.

    We convert to first-order system:
        y[0] = psi(xi)
        y[1] = d psi/d xi

    Then:
        dy[0]/d xi = y[1]
        dy[1]/d xi = -9 rho_tilde(psi) - (2/xi) y[1]

    Args:
        xi: Dimensionless radius xi = r/r_c
        y: State vector [psi, d psi/d xi]
        args: (W0,) - concentration parameter

    Returns:
        Derivative [d psi/d xi, d^2 psi/d xi^2]

    References:
        King (1966), AJ, 71, 64, Eq. 9-10
        Binney & Tremaine (2008), "Galactic Dynamics", Section 4.3.2
    """
    (W0,) = args
    xi = jnp.asarray(xi)
    psi, dpsi_dxi = y[0], y[1]

    # Dimensionless density = lowered-Maxwellian volume density rho_hat(psi),
    # normalized to 1 at the center (psi=W0).
    rho0 = king_lowered_maxwellian_density(W0)
    safe_rho0 = jnp.where(rho0 > 0.0, rho0, 1.0)
    rho_tilde = jnp.where(
        rho0 > 0.0, king_lowered_maxwellian_density(psi) / safe_rho0, 0.0
    )

    # King's Poisson equation in standard nondimensional form (King 1966;
    # Binney & Tremaine 2008, Eq. 4.131): with xi = r/r_c and r_c the King core
    # radius r_0 = sqrt(9 sigma^2 / 4 pi G rho_0), the RHS carries a factor of 9.
    # Regularity gives lim (2/xi) W' = 2 W''(0), hence W''(0) = -3.
    d2psi_dxi2 = poisson_second_derivative(xi, dpsi_dxi, rho_tilde)

    return jnp.array([dpsi_dxi, d2psi_dxi2])


def _auto_ode_domain(W0: ArrayLike) -> Tuple[float, int]:
    """Default King ODE integration domain ``(xi_max, n_points)`` sized from ``W0``.

    The tidal radius grows super-exponentially with concentration (King 1966
    Table II: xi_t = r_t/r_c is ~4.7 at W0=3, ~131 at W0=9, ~2272 at W0=15), so a
    fixed domain pins high-W0 models to the integration boundary and
    under-estimates the concentration. This envelope keeps ~1.6-1.8x margin above
    xi_t for W0 up to ~16, while reproducing the historical default
    (xi_max=300, n_points=2000) for W0 <= ~9.5 (so typical globular-cluster models
    are unchanged). ``n_points`` scales with the domain to hold the core resolution
    (~0.15 in xi) roughly fixed.

    Sizing the domain needs a *concrete* ``W0`` (it sets static array sizes). When
    ``W0`` is a JAX tracer -- i.e. the caller is jitting or differentiating
    *through W0* -- we fall back to the fixed default ``(300, 2000)``. That keeps
    ``from_W0_rc`` JIT-able and differentiable in ``W0`` exactly as before (the ODE
    -> psi -> density -> CDF path carries dpsi/dW0); auto-sizing applies to ordinary
    eager construction, which is where high-concentration models are built. Under
    traced ``W0`` only the historical W0 <= ~10 range is in-domain at the fixed
    default -- pass an explicit ``xi_max`` for traced high-W0 work.
    """
    try:
        # W0 is a real scalar here; cast documents that for the type checker
        # (ArrayLike nominally includes complex). Runtime float() is unchanged and
        # raises on a tracer, which is caught below.
        w = float(cast(SupportsFloat, W0))
    except (
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerArrayConversionError,
        TypeError,
    ):
        return 300.0, 2000  # traced W0: array sizes cannot depend on a tracer
    xi_max = max(300.0, 10.0 ** (0.21 * w + 0.45))
    n_points = int(max(2000, math.ceil(xi_max / 0.15)))
    return xi_max, n_points


def solve_king_profile(
    W0: ArrayLike,
    xi_max: float = 300.0,
    n_points: int = 2000,
    *,
    adjoint=None,
):
    """
    Solve King's Poisson equation numerically using diffrax.

    Integrates from xi=1e-6 outward through the tidal boundary. The regular
    central series initializes W=W0-3 xi^2/2 and W'= -3 xi. The omitted terms
    are O(xi^4) and O(xi^3), respectively; the grid and return shapes are unchanged.

    Boundary conditions (King 1966, Eq. 10):
        psi(0) = W0  (central potential)
        d psi/d xi|_0 = 0  (symmetry at center)

    Args:
        W0: King concentration parameter
        xi_max: Maximum dimensionless radius to integrate to
        n_points: Number of points in output grid

    Returns:
        3-tuple ``(xi_grid, psi_clamped, psi_raw)``:

        - ``xi_grid``: dimensionless radii (``solution.ts``).
        - ``psi_clamped``: ``max(psi, 0)``, truncated at the tidal radius. This
          is the physical potential used for density / CDF / mu / virial.
        - ``psi_raw``: the unclamped ODE potential, negative beyond the
          tidal boundary. Use it for cubic root interpolation and implicit
          parameter derivatives; clipping removes crossing information.

    References:
        King (1966), AJ, 71, 64
        Binney & Tremaine (2008), Section 4.3.2

    Note:
        JIT-compatible when ``n_points`` (and ``xi_max``) are static: they set the
        ``linspace`` size and are closed over, so ``jax.jit(solve_king_profile)(W0)``
        traces fine (W0 may be a tracer). Uses Tsit5 (Runge-Kutta 5th order) from
        diffrax for robustness.

    Use adjoint=None for checkpointed reverse AD (default), diffrax.ForwardMode()
    for forward AD, or diffrax.DirectAdjoint() for both. Keep this choice
    static under JIT; it does not change the physical equations or tolerances.
    """
    # Initial conditions
    xi_span = (1e-6, xi_max)
    y0 = regular_center_state(W0, xi_span[0])

    # Create ODE term
    term = diffrax.ODETerm(_king_poisson_rhs)

    # Use Tsit5 (adaptive Runge-Kutta)
    solver = diffrax.Tsit5()

    # Adaptive step size controller
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-10)

    # Save at specified points
    saveat = diffrax.SaveAt(ts=jnp.linspace(xi_span[0], xi_span[1], n_points))

    # Solve ODE
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=xi_span[0],
        t1=xi_span[1],
        dt0=1e-4,
        y0=y0,
        args=(W0,),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        adjoint=diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint,
        max_steps=100000,
    )

    xi_grid = solution.ts
    psi_raw = solution.ys[:, 0]  # Extract psi(xi), UNCLAMPED (negative past r_t)

    # Ensure psi >= 0 (truncate at tidal radius where psi -> 0)
    psi_clamped = jnp.maximum(psi_raw, 0.0)

    return xi_grid, psi_clamped, psi_raw


def _has_tidal_crossing(psi_raw: Float[Array, "n_points"]) -> Bool[Array, ""]:
    """Whether a finite raw potential starts positive and brackets a tidal zero."""
    return (
        jnp.all(jnp.isfinite(psi_raw))
        & (psi_raw[0] > 0.0)
        & jnp.any((psi_raw[:-1] > 0.0) & (psi_raw[1:] <= 0.0))
    )


def _raise_if_unresolved_tidal_crossing(invalid) -> None:
    # A vmapped cond may execute both branches. Recheck the callback argument
    # instead of treating callback execution itself as evidence of failure.
    if bool(invalid):
        raise ValueError(
            "Self-consistent profile requires a finite raw potential with a "
            "positive centre and an actual tidal crossing. The finite ODE "
            "domain has not established that crossing; check xi_max, "
            "n_ode_points and any anisotropy."
        )


def _require_tidal_crossing(psi_raw: Float[Array, "n_points"]) -> None:
    """Reject an unresolved factory solve, including derivative-only JIT outputs.

    The ordered effect retains the check when AD discards all primal values.
    No derivative of the potential or its tidal root is modified.
    """
    invalid = jax.lax.stop_gradient(~_has_tidal_crossing(psi_raw))
    if _is_concrete(invalid):
        _raise_if_unresolved_tidal_crossing(invalid)
    else:
        jax.lax.cond(
            invalid,
            lambda flag: jax.debug.callback(
                _raise_if_unresolved_tidal_crossing, flag, ordered=True
            ),
            lambda flag: None,
            invalid,
        )


def _find_tidal_radius(
    xi_grid: Float[Array, "n_points"],
    psi_grid: Float[Array, "n_points"],
) -> Float[Array, ""]:
    """Locate the first zero of an unclamped, decreasing potential.

    Radii must increase strictly. A monotone cubic uses shared nodal slopes,
    making the derivative continuous as a simple root crosses a saved node.
    Root derivatives follow the implicit equation, including slope dependence.
    A flat multiple root is outside this differentiable contract.

    No crossing returns the last radius; a nonpositive first potential returns
    the first radius. Supply negative potentials beyond the tidal boundary to
    retain its parameter sensitivity; a clipped zero plateau loses that information.
    """
    x, y = xi_grid, psi_grid
    h = jnp.diff(x)
    d = jnp.diff(y) / h
    same = d[:-1] * d[1:] > 0
    a = jnp.where(same, d[:-1], 1.0)
    b = jnp.where(same, d[1:], 1.0)
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    interior = jnp.where(same, (w1 + w2) / (w1 / a + w2 / b), 0.0)
    slopes = jnp.concatenate((d[:1], interior, d[-1:]))
    crossed = y <= 0
    j = jnp.clip(jnp.argmax(crossed), 1, len(x) - 1)
    i = j - 1
    width = x[j] - x[i]

    def polynomial(t):
        return (
            (2 * t**3 - 3 * t * t + 1) * y[i]
            + (t**3 - 2 * t * t + t) * width * slopes[i]
            + (-2 * t**3 + 3 * t * t) * y[j]
            + (t**3 - t * t) * width * slopes[j]
        )

    def solve(f, guess):
        del guess

        def step(_, bracket):
            lo, hi = bracket
            mid = (lo + hi) / 2
            positive = f(mid) > 0
            return jnp.where(positive, mid, lo), jnp.where(positive, hi, mid)

        lo, hi = jax.lax.fori_loop(0, 48, step, (jnp.array(0.0), jnp.array(1.0)))
        return (lo + hi) / 2

    # The selected polynomial, including its nodal slopes, carries parameter AD.
    # A no-crossing input uses an inactive well-conditioned linear polynomial.
    active = jnp.any(crossed) & (y[0] > 0)

    def f(t):
        return jnp.where(active, polynomial(t), 0.5 - t)

    t = jax.lax.custom_root(
        f, jnp.array(0.5), solve, lambda lin, rhs: rhs / lin(jnp.array(1.0))
    )
    return jnp.where(active, x[i] + width * t, jnp.where(y[0] <= 0, x[0], x[-1]))


# ==============================================================================
# KingProfile Class (SpatialProfile implementation)
# ==============================================================================


class KingProfile(eqx.Module):
    """
    King (1966) spherical density profile.

    Implements SpatialProfile protocol for IC assembly.

    The CDF is precomputed at initialization for efficient sampling.

    Attributes:
        W0: King concentration parameter (dimensionless)
        r_c: Core radius [length units]
        r_t: Tidal (truncation) radius [length units]
        xi_grid: Pre-computed dimensionless radii from ODE solver
        psi_grid: Pre-computed dimensionless potential from ODE solver
        _r_grid: Precomputed radial grid for CDF interpolation
        _cdf_grid: Precomputed CDF values on grid

    References:
        King (1966), AJ, 71, 64

    Examples:
        # Recommended: Use from_W0_rc for self-consistent model
        >>> profile = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)

        # Or manually with pre-computed ODE solution
        >>> xi_grid, psi_grid, _ = solve_king_profile(W0=7.0)
        >>> profile = KingProfile(W0=7.0, r_c=1.0, r_t=10.0,
        ...                       xi_grid=xi_grid, psi_grid=psi_grid)
        >>> masses = jnp.ones(100)
        >>> key = jax.random.PRNGKey(42)
        >>> positions = profile.sample_positions(masses, key)
    """

    W0: Float[Array, ""]
    r_c: Float[Array, ""]
    r_t: Float[Array, ""]
    xi_grid: Float[Array, "n_points"]
    psi_grid: Float[Array, "n_points"]
    _r_grid: Float[Array, "n_grid"]
    _cdf_grid: Float[Array, "n_grid"]
    # Diagnostic (audit J4): True iff psi never crossed 0 within the ODE domain,
    # so r_t was pinned to the grid boundary (a wrong tidal radius). Traced bool.
    r_t_is_pinned: Bool[Array, ""]

    def __init__(
        self,
        W0: ArrayLike,
        r_c: ArrayLike,
        r_t: ArrayLike,
        xi_grid: Float[Array, "n_points"],
        psi_grid: Float[Array, "n_points"],
        n_grid: int = 1000,
    ):
        """
        Initialize King profile with precomputed CDF.

        Args:
            W0: King concentration parameter (typical 1-12)
            r_c: Core radius [length units]
            r_t: Tidal radius [length units]
            xi_grid: Pre-computed dimensionless radii from solve_king_profile()
            psi_grid: Pre-computed dimensionless potential from solve_king_profile()
            n_grid: Number of grid points for CDF interpolation (default: 1000)
        """
        W0_arr = jnp.asarray(W0, dtype=jnp.float64)
        r_c_arr = jnp.asarray(r_c, dtype=jnp.float64)
        r_t_arr = jnp.asarray(r_t, dtype=jnp.float64)
        xi_grid_arr = jnp.asarray(xi_grid, dtype=jnp.float64)
        psi_grid_arr = jnp.asarray(psi_grid, dtype=jnp.float64)

        # Build radial grid for CDF — sqrt-stretched (r = r_t * u^2): spacing
        # dr ∝ sqrt(r) concentrates points in the core. A LINEAR grid leaves
        # <10 points inside the core at W0 >= 9 (xi_t grows super-exponentially:
        # 131 r_c at W0=9, 548 at W0=12), giving +18%..+270% core-mass errors
        # (audit R4, measured). Smooth in r_t -> differentiable in W0/r_c.
        u_grid = jnp.linspace(0.0, 1.0, n_grid)
        r_grid = r_t_arr * u_grid**2
        xi_grid_local = r_grid / r_c_arr

        # Compute density on grid via interpolation of ODE solution
        psi_vals = jnp.interp(
            xi_grid_local, xi_grid_arr, psi_grid_arr, left=W0_arr, right=0.0
        )

        # King density: rho(r)/rho_0 = rho_hat(psi) / rho_hat(W0)
        # (lowered-Maxwellian volume density; see king_lowered_maxwellian_density)
        rho0 = king_lowered_maxwellian_density(W0_arr)
        rho_grid = jnp.where(
            rho0 > 0.0,
            king_lowered_maxwellian_density(psi_vals)
            / jnp.where(rho0 > 0.0, rho0, 1.0),
            0.0,
        )

        # Truncate at tidal radius
        rho_grid = jnp.where(r_grid <= r_t_arr, rho_grid, 0.0)

        # Integrand: 4*pi*r^2*rho(r)
        integrand = 4.0 * jnp.pi * r_grid**2 * rho_grid

        # Cumulative mass via the NON-UNIFORM trapezoid rule (2nd-order): the
        # sqrt-stretched grid has variable spacing, so the per-interval width
        # diff(r_grid) must weight each trapezoid (a single dr would mis-integrate).
        M_cum = jnp.concatenate(
            [
                jnp.zeros(1, dtype=integrand.dtype),
                jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(r_grid)),
            ]
        )

        # Normalize to [0, 1] for CDF
        cdf_grid = M_cum / (M_cum[-1] + 1e-30)

        # r_t pinning diagnostic (audit J4): if psi(xi) never reaches 0 within
        # the solved domain there is no crossing, so _find_tidal_radius fell back
        # to xi_grid[-1] (the boundary) — a wrong tidal radius. Traced bool.
        r_t_is_pinned = jnp.logical_not(jnp.any(psi_grid_arr <= 0.0))

        # r_t consistency (audit S1): the direct constructor accepts an arbitrary
        # r_t. For concrete, non-pinned inputs, warn if r_t deviates from the
        # c(W0) tidal radius r_c*xi_t by >5% — a non-self-consistent, non-
        # equilibrium model. from_W0_rc derives r_t, so it never trips this.
        if _is_concrete(r_t_arr) and _is_concrete(W0_arr) and not bool(r_t_is_pinned):
            xi_t = _find_tidal_radius(xi_grid_arr, psi_grid_arr)
            r_t_consistent = float(r_c_arr * xi_t)
            if abs(float(r_t_arr) - r_t_consistent) > 0.05 * r_t_consistent:
                warnings.warn(
                    f"KingProfile r_t={float(r_t_arr):.4g} is inconsistent with the "
                    f"c(W0={float(W0_arr):.2f}) tidal radius r_c*xi_t="
                    f"{r_t_consistent:.4g} (>5% deviation): this builds a NON-self-"
                    f"consistent, non-equilibrium King model. Use "
                    f"KingProfile.from_W0_rc(W0, r_c) to derive a consistent r_t.",
                    UserWarning,
                    stacklevel=2,
                )

        # Store using object.__setattr__ (future-proof Equinox pattern)
        object.__setattr__(self, "W0", W0_arr)
        object.__setattr__(self, "r_c", r_c_arr)
        object.__setattr__(self, "r_t", r_t_arr)
        object.__setattr__(self, "xi_grid", xi_grid_arr)
        object.__setattr__(self, "psi_grid", psi_grid_arr)
        object.__setattr__(self, "_r_grid", r_grid)
        object.__setattr__(self, "_cdf_grid", cdf_grid)
        object.__setattr__(self, "r_t_is_pinned", r_t_is_pinned)

    @classmethod
    def from_W0_rc(
        cls,
        W0: float,
        r_c: float,
        xi_max: float | None = None,
        n_ode_points: int | None = None,
        n_grid: int = 1000,
        *,
        adjoint=None,
    ) -> "KingProfile":
        """
        Create self-consistent King profile where r_t is derived from W0.

        This is the RECOMMENDED constructor. The tidal radius is computed
        from where the potential psi(xi) crosses zero, ensuring a physically
        self-consistent King model.

        Args:
            W0: King concentration parameter (typical 3-12; supported to ~15)
            r_c: Core radius [length units]
            xi_max: Maximum dimensionless radius for ODE integration. If None
                (default), sized automatically from W0 (see _auto_ode_domain) so
                high-concentration models integrate to psi->0 instead of pinning.
            n_ode_points: Number of ODE solution points. If None (default), sized
                automatically alongside xi_max to hold core resolution fixed.
            n_grid: Number of grid points for CDF interpolation (default: 1000)

        Returns:
            KingProfile with self-consistent r_t derived from W0

        The raw ODE solution must be finite, start positive and bracket an actual
        tidal zero. An unresolved domain is rejected eagerly and under JIT.
        This factory check does not constrain the manual precomputed-grid constructor.

        Examples:
            >>> profile = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
            >>> print(f"Tidal radius: {profile.r_t:.2f}")

        Use adjoint=None for checkpointed reverse AD (default), diffrax.ForwardMode()
        for forward AD, or diffrax.DirectAdjoint() for both. Keep this choice
        static under JIT; it does not change the physical equations or tolerances.
        """
        require_positive(W0, "KingProfile W0")
        require_positive(r_c, "KingProfile r_c")
        auto_xi_max, auto_n_points = _auto_ode_domain(W0)
        if xi_max is None:
            xi_max = auto_xi_max
        if n_ode_points is None:
            n_ode_points = auto_n_points
        # Use the unclamped potential for the root and its parameter derivatives;
        # density and CDF evaluation use the nonnegative stored potential.
        xi_grid, psi_grid, psi_raw = solve_king_profile(
            W0,
            xi_max=xi_max,
            n_points=n_ode_points,
            adjoint=adjoint,
        )
        xi_t = _find_tidal_radius(xi_grid, psi_raw)
        r_t = r_c * xi_t
        prof = cls(
            W0=W0,
            r_c=r_c,
            r_t=r_t,
            xi_grid=xi_grid,
            psi_grid=psi_grid,
            n_grid=n_grid,
        )
        # Preserve the existing concrete ValueError. The effectful factory check
        # also survives consumers that discard every primal profile field in AD.
        if _is_concrete(prof.r_t_is_pinned) and bool(prof.r_t_is_pinned):
            raise ValueError(
                f"King ODE domain too small: psi(xi) never reached 0 within "
                f"xi_max={float(xi_max):.1f} for W0={float(W0):.2f}, so r_t is "
                f"PINNED to the grid boundary (a wrong tidal radius). Increase "
                f"xi_max (and n_ode_points), or omit them to auto-size. For "
                f"traced/jit W0 pass an explicit xi_max sized for the largest W0."
            )
        _require_tidal_crossing(psi_raw)
        return prof

    def sample_positions(
        self,
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "N 3"]:
        """
        Sample particle positions from King density profile.

        Uses precomputed CDF for efficient inverse transform sampling.

        Args:
            masses: Particle masses [mass units]. Note: Only the array
                length is used to determine N; mass values are not used
                for position sampling in this profile.
            key: JAX random key for reproducible sampling

        Returns:
            Particle positions [length units]
        """
        N = len(masses)

        # Sample radii via precomputed inverse CDF
        key, subkey = jax.random.split(key)
        radii = self._sample_radii(subkey, N)

        # Isotropic angles
        key, subkey = jax.random.split(key)
        theta = jnp.arccos(1.0 - 2.0 * jax.random.uniform(subkey, shape=(N,)))
        key, subkey = jax.random.split(key)
        phi = 2.0 * jnp.pi * jax.random.uniform(subkey, shape=(N,))

        # Convert to Cartesian
        x = radii * jnp.sin(theta) * jnp.cos(phi)
        y = radii * jnp.sin(theta) * jnp.sin(phi)
        z = radii * jnp.cos(theta)

        return jnp.stack([x, y, z], axis=1)

    def _sample_radii(self, key: PRNGKeyArray, N: int) -> Float[Array, "N"]:
        """
        Sample radii from precomputed CDF via inverse transform.

        Args:
            key: JAX random key
            N: Number of particles to sample

        Returns:
            Radii following King profile [length units]
        """
        # Generate uniform random numbers
        u = jax.random.uniform(key, shape=(N,))

        # Inverse CDF: interpolate to find r where CDF = u
        r_sampled = jnp.interp(u, self._cdf_grid, self._r_grid)

        # Ensure r <= r_t (strict truncation)
        r_sampled = jnp.clip(r_sampled, 0.0, self.r_t)

        return r_sampled

    def density(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """
        Unnormalized density profile rho(r)/rho_0.

        The King density is the lowered-Maxwellian *volume* density,
        normalized to 1 at the centre:
            rho(r)/rho_0 = rho_hat(psi(r)) / rho_hat(W0)
        with rho_hat(W) = e^W erf(sqrt(W)) - (2/sqrt(pi)) sqrt(W) (1 + 2W/3)
        (king_lowered_maxwellian_density; BT2008 Eq. 4.131), and psi(r) the
        dimensionless potential from interpolating the ODE solution. This is the
        3-D volume density. The Maxwell speed cumulative probability is a
        different velocity-space integral; spatial surface density requires a
        line-of-sight projection of this volume density.

        This method returns the unnormalized form (rho_0=1), useful for
        plotting and analysis with jaxstroviz.

        Args:
            r: Radial distances [length units]. Can be any shape.

        Returns:
            Unnormalized density at each radius (same shape as input)
        """
        xi = r / self.r_c
        psi_vals = jnp.interp(xi, self.xi_grid, self.psi_grid, left=self.W0, right=0.0)

        rho0 = king_lowered_maxwellian_density(self.W0)
        rho = jnp.where(
            rho0 > 0.0,
            king_lowered_maxwellian_density(psi_vals)
            / jnp.where(rho0 > 0.0, rho0, 1.0),
            0.0,
        )

        # Truncate at tidal radius
        return jnp.where(r <= self.r_t, rho, 0.0)

    def characteristic_radius(self) -> Float[Array, ""]:
        """
        Return characteristic radius (tidal radius for King).

        Returns:
            Tidal radius [length units]
        """
        return self.r_t


__all__ = ["KingProfile", "solve_king_profile"]
