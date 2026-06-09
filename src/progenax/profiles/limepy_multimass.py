"""
Multi-mass LIMEPY coupled equilibrium (Gieles & Zocchi 2015, Section 2.2).

A first-principles, differentiable, multi-mass lowered-isothermal model: one shared
self-consistent potential phi(r) with per-component velocity scales set by the
equipartition parameter delta. Heavier components get a smaller velocity scale, hence a
deeper effective well and central concentration -- mass segregation AS A GENUINE
EQUILIBRIUM (each component individually virial, Q_j ~ 0.5), unlike the lambda_seg blend.

Three layers (see docs/plans/2026-06-09-multimass-limepy-equilibrium-design.md):
  A. solve_multimass_limepy(alpha_j, ...) -- one coupled Poisson solve given the central
     density fractions alpha_j (this module; pure physics, no iteration).
  B. find_alpha_for_masses(m_j, M_j, ...) -- fixed-iteration eigenvalue solve for the
     alpha_j that reproduce target masses (this module).
  C. MultiMassLIMEPY -- user-facing model + IMF binning + segregated-IC sampling.

Isotropic (delta only); per-component anisotropy (eta) is a planned extension (Phase 2b).

References:
    Gieles, M. & Zocchi, A. (2015), MNRAS, 454, 576 (Eqs. 24-29, Section 4.1).
"""

from typing import Tuple

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from progenax.profiles.limepy import limepy_density_hat


def _multimass_density_sources(
    psi: Float[Array, ""],
    rescale: Float[Array, "n_comp"],
    W0: Float[Array, ""],
    g: Float[Array, ""],
) -> Float[Array, "n_comp"]:
    """Per-component normalized densities rho_hat_j at potential psi (Eq. 29).

    rescale_j = mu_j^(2 delta); rho_hat_j = limepy_density_hat(rescale_j psi, g)
    / limepy_density_hat(rescale_j W0, g), each normalized to 1 at the centre.
    """
    num = limepy_density_hat(rescale * psi, g)
    den = limepy_density_hat(rescale * W0, g)
    return jnp.where(den > 1e-300, num / den, 0.0)


def solve_multimass_limepy(
    alpha_j: Float[Array, "n_comp"],
    m_j: Float[Array, "n_comp"],
    W0: float,
    g: float,
    delta: float,
    xi_max: float = 300.0,
    n_points: int = 2000,
) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"], Float[Array, "n_comp n_points"]]:
    """Solve the multi-mass coupled LIMEPY Poisson equation (Layer A).

    Given central density fractions alpha_j (sum to 1) and component masses m_j, solve

        (1/xi^2) d/dxi(xi^2 dpsi/dxi) = -9 sum_j alpha_j rho_hat_j(psi),

    with the shared King-radius (-9) nondimensionalization. The per-component rescaling
    is mu_j^(2 delta), mu_j = m_j / bar_m, bar_m = sum_j m_j alpha_j (central
    density-weighted mean mass, Eq. 26). At delta=0 every rescaling is 1 and the source
    is (sum alpha_j) rho_hat = rho_hat -- identical to solve_limepy_profile.

    JIT/grad-safe in (alpha_j, m_j, W0, g, delta): n_points, xi_max static.

    Returns:
        (xi_grid, psi_grid, rho_j_grid): dimensionless radius, shared potential W(xi)>=0,
        and per-component normalized densities rho_hat_j(xi), shape (n_comp, n_points).
    """
    alpha_j = jnp.asarray(alpha_j)
    m_j = jnp.asarray(m_j)
    bar_m = jnp.sum(m_j * alpha_j)
    mu_j = m_j / bar_m
    rescale = mu_j ** (2.0 * delta)  # mu_j^(2 delta) per component

    def rhs(xi, y, args):
        psi, dpsi = y[0], y[1]
        rho_j = _multimass_density_sources(psi, rescale, W0, g)
        rho_tot = jnp.sum(alpha_j * rho_j)
        d2 = jnp.where(
            xi > 1e-6, -9.0 * rho_tot - (2.0 / xi) * dpsi, -9.0 * rho_tot
        )
        return jnp.array([dpsi, d2])

    y0 = jnp.array([W0, 0.0])
    xi_span = (1e-6, xi_max)
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=xi_span[0],
        t1=xi_span[1],
        dt0=1e-4,
        y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(xi_span[0], xi_span[1], n_points)),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=100000,
    )
    xi_grid = solution.ts
    psi_grid = jnp.maximum(solution.ys[:, 0], 0.0)

    # Per-component densities on the grid (vmap over the grid), truncated at psi -> 0.
    rho_j_grid = jax.vmap(
        lambda p: _multimass_density_sources(p, rescale, W0, g), out_axes=1
    )(psi_grid)
    rho_j_grid = jnp.where(psi_grid[None, :] > 0.0, rho_j_grid, 0.0)
    return xi_grid, psi_grid, rho_j_grid


def _realized_fractions(
    alpha_j: Float[Array, "n_comp"],
    m_j: Float[Array, "n_comp"],
    W0: float,
    g: float,
    delta: float,
    xi_max: float,
    n_points: int,
) -> Float[Array, "n_comp"]:
    """Realized mass fractions f_j' = alpha_j nu_j / sum_k alpha_k nu_k from a coupled
    solve, nu_j = int rho_hat_j xi^2 dxi (the per-component dimensionless mass)."""
    xi, _, rho_j = solve_multimass_limepy(alpha_j, m_j, W0, g, delta, xi_max, n_points)
    nu_j = jnp.trapezoid(rho_j * xi**2, xi, axis=1)
    M_real = alpha_j * nu_j
    return M_real / (jnp.sum(M_real) + 1e-300)


def find_alpha_for_masses(
    m_j: Float[Array, "n_comp"],
    M_j: Float[Array, "n_comp"],
    W0: float,
    g: float,
    delta: float,
    n_iter: int = 30,
    xi_max: float = 300.0,
    n_points: int = 2000,
) -> Tuple[Float[Array, "n_comp"], Float[Array, ""]]:
    """Find the central density fractions alpha_j that reproduce target masses M_j (Layer B).

    Fixed-point iteration on the realized mass fractions, with the stabilized update of
    Gieles & Zocchi (2015, Section 4.1) -- NOT Gunn & Griffin's linear M_j/M_j' (which
    diverges for wide mass functions):

        alpha_j <- alpha_j sqrt(f_j / f_j'),   renormalize sum_j alpha_j = 1,

    f_j = M_j / sum M (target), f_j' = realized fraction. Run as a FIXED-length
    jax.lax.scan (never while_loop) so the whole solve is differentiable in (M_j, delta,
    g, W0). Starts from alpha_j = f_j.

    Args:
        m_j: component representative masses. M_j: target mass per component.
        W0, g, delta: model parameters. n_iter: fixed iteration count.
        xi_max, n_points: ODE grid (static).

    Returns:
        (alpha_j, residual): converged central density fractions (sum to 1, positive)
        and the final fractional residual max_j |f_j' - f_j| (reported, never branched on).
    """
    m_j = jnp.asarray(m_j)
    f_target = jnp.asarray(M_j) / jnp.sum(jnp.asarray(M_j))

    def step(alpha, _):
        f_real = _realized_fractions(alpha, m_j, W0, g, delta, xi_max, n_points)
        alpha_new = alpha * jnp.sqrt(f_target / (f_real + 1e-300))
        alpha_new = alpha_new / jnp.sum(alpha_new)
        return alpha_new, None

    alpha_final, _ = jax.lax.scan(step, f_target, None, length=n_iter)
    f_real = _realized_fractions(alpha_final, m_j, W0, g, delta, xi_max, n_points)
    residual = jnp.max(jnp.abs(f_real - f_target))
    return alpha_final, residual


__all__ = ["solve_multimass_limepy", "find_alpha_for_masses"]
