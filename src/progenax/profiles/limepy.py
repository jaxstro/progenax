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

and P is the regularized lower incomplete gamma function (`jax.scipy.special.
gammainc`), which is differentiable in BOTH arguments — so the truncation index
g (entering as a = g + 3/2) carries gradients.

Index note (verified against the source, not assumed): the closed form uses
g + 3/2, established in the paper's Appendix B (Eqs. B4-B9) via the E_gamma
convolution identity (Eq. D11). The main-text Eqs. 8/11 print "g + 1/2"; this is
a typesetting slip relative to the appendix and the released `limepy` code. The
2018 erratum's corrected Eqs. 20/21 carry the same g + 3/2 normalization
(E_gamma(g + 3/2, W0)). Three independent confirmations agree:
  1. E_gamma(5/2, W) expands exactly to the King volume density
     e^W erf(sqrt W) - (2/sqrt pi) sqrt(W) (1 + 2W/3)  -> g=1 corner.
  2. The erratum normalization denominator is E_gamma(g + 3/2, W0).
  3. The convolution Eq. D11 with b = 3/2 lifts the index by exactly 3/2.

References:
    Gieles, M. & Zocchi, A. (2015), MNRAS, 454, 576 (Eqs. 1-9, App. B, D).
    Gieles & Zocchi (2018), MNRAS, 474, 3997 (erratum: Eqs. 20, 21, 41).
    King, I. R. (1966), AJ, 71, 64 (the g=1 corner).
"""

from typing import Tuple

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def lowered_exponential(
    a: Float[Array, ""], W: Float[Array, "..."]
) -> Float[Array, "..."]:
    """The lowered exponential E_gamma(a, W) = e^W P(a, W) for a > 0 (Eq. 2).

    P(a, W) = gammainc(a, W) is the regularized lower incomplete gamma function.
    For W <= 0 the function is 0 (gamma(a, 0) = 0); we clamp the argument before
    `gammainc` so the backward pass never sees a negative argument.

    This is the a > 0 branch only (a = g + 3/2 >= 3/2 for every density use). The
    a = 0 Woolley branch E_gamma(0, x) = e^x is needed only for the DF itself, not
    for the density, and is intentionally not handled here.

    Differentiable in both a and W (carries d/dg via a = g + 3/2).
    """
    W_pos = jnp.where(W > 0.0, W, 0.0)
    val = jnp.exp(W_pos) * jax.scipy.special.gammainc(a, W_pos)
    return jnp.where(W > 0.0, val, 0.0)


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
           The density is finite-extent for g <= 3.5.

    Returns:
        I_rho(W), the unnormalized LIMEPY volume density. Differentiable in (W, g).
    """
    return lowered_exponential(g + 1.5, W)


# ==============================================================================
# Self-consistent Poisson solve (general-g; generalizes solve_king_profile)
# ==============================================================================


def _limepy_poisson_rhs(xi: float, y: Float[Array, "2"], args: tuple) -> Float[Array, "2"]:
    """RHS of the LIMEPY dimensionless Poisson equation (Gieles & Zocchi Eq. 5):

        d^2W/dxi^2 + (2/xi) dW/dxi = -9 rho_hat(W; g),

    with rho_hat normalized to 1 at the centre (W = W0) and xi = r/r_s the
    King-radius-scaled radius (the same factor-of-9 nondimensionalization as the
    King solver; LIMEPY uses King's r_s by construction). State y = [W, dW/dxi].

    At g=1 this is identical to `_king_poisson_rhs`. The xi=0 singularity is
    handled by L'Hopital (lim (2/xi)dW/dxi = 0 since dW/dxi(0)=0).
    """
    W0, g = args
    psi, dpsi_dxi = y[0], y[1]

    rho0 = limepy_density_hat(W0, g)
    rho_tilde = jnp.where(rho0 > 1e-300, limepy_density_hat(psi, g) / rho0, 0.0)

    d2psi_dxi2 = jnp.where(
        xi > 1e-6,
        -9.0 * rho_tilde - (2.0 / xi) * dpsi_dxi,
        -9.0 * rho_tilde,  # centre guard (dW/dxi(0)=0)
    )
    return jnp.array([dpsi_dxi, d2psi_dxi2])


def solve_limepy_profile(
    W0: float, g: float, xi_max: float = 300.0, n_points: int = 2000
) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"]]:
    """Solve the general-g LIMEPY Poisson equation with diffrax (Tsit5).

    Integrates W(xi) from the centre (W=W0, dW/dxi=0) outward to the truncation
    radius where W -> 0. Identical structure to `solve_king_profile`, with the
    continuous truncation parameter g (g=0 Woolley, g=1 King, g=2 Wilson). At
    g=1 it reproduces the King solution.

    JIT/grad-safe in (W0, g): n_points and xi_max are static (they set the output
    grid size); W0 and g may be tracers, and the ODE -> W(xi) path carries
    dW/dW0 and dW/dg (the latter through gammainc's a-derivative).

    Args:
        W0: Dimensionless central potential (King's W0 = Gieles & Zocchi's phi0).
        g: Truncation parameter (continuous; finite extent for g <= 3.5).
        xi_max: Max dimensionless radius for integration.
        n_points: Output grid size.

    Returns:
        (xi_grid, psi_grid): dimensionless radius and potential W(xi) (>= 0).
    """
    y0 = jnp.array([W0, 0.0])
    xi_span = (1e-6, xi_max)
    term = diffrax.ODETerm(_limepy_poisson_rhs)
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
        args=(W0, g),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100000,
    )
    xi_grid = solution.ts
    psi_grid = jnp.maximum(solution.ys[:, 0], 0.0)
    return xi_grid, psi_grid


__all__ = ["lowered_exponential", "limepy_density_hat", "solve_limepy_profile"]
