"""
Multi-mass LIMEPY coupled equilibrium (Gieles & Zocchi 2015, Section 2.2).

A first-principles, differentiable, multi-mass lowered-isothermal model: one shared
self-consistent potential phi(r) with per-component velocity scales set by the
equipartition parameter delta. Heavier components get a smaller velocity scale, hence a
deeper effective well and central concentration -- mass segregation AS A GENUINE
EQUILIBRIUM (each component individually virial, Q_j ~ 0.5), unlike the lambda_seg blend.

Three layers (see docs/plans/2026-06-09-multimass-limepy-equilibrium-design.md):
  A. solve_multicomponent_limepy(alpha_j, rescale_j, ...) -- one coupled Poisson solve
     given central density fractions alpha_j and DIRECT per-component scales (Engine A
     core); solve_multimass_limepy is its thin mass-segregation wrapper.
  B. find_alpha_for_masses(m_j, M_j, ...) -- fixed-iteration eigenvalue solve for the
     alpha_j that reproduce target masses (this module).
  C. The user-facing model is progenax.cluster.multicomponent.MultiComponentCluster;
     this module keeps its grid/density/sampling helpers.

Isotropic and per-component anisotropic (ra_hat_j; mass path: r_{a,j} = r_a mu_j^eta).

References:
    Gieles, M. & Zocchi, A. (2015), MNRAS, 454, 576 (Eqs. 24-29, Section 4.1).
"""

from typing import Tuple

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax.profiles.limepy import (
    _aniso_density_scalar,
    _angle_integral_T,
    limepy_density_hat,
    lowered_exponential,
)


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


_N_U_VIR = 256  # speed quadrature for the anisotropic v^2 moment


def _grid_density_components(psi_arr, xi_arr, rescale, W0, g, ra_hat_j, is_aniso):
    """Per-component normalized densities on a grid, shape (n_comp, len(psi_arr)).

    Isotropic: rho_hat_j = limepy_density_hat(rescale_j psi)/norm. Anisotropic
    (finite per-component ra_hat_j): the anisotropic density at p_j = xi/ra_hat_j,
    normalized to the central (p=0) value. xi_arr is the dimensionless radius r/r_c.
    """
    if is_aniso:
        rho0_j = jax.vmap(lambda res: _aniso_density_scalar(res * W0, jnp.asarray(0.0), g))(rescale)

        def at(xi, psi):
            p_j = xi / ra_hat_j
            rho = jax.vmap(lambda res, pj: _aniso_density_scalar(res * psi, pj, g))(rescale, p_j)
            return jnp.where(rho0_j > 1e-300, rho / rho0_j, 0.0)

        return jax.vmap(at, out_axes=1)(xi_arr, psi_arr)
    return jnp.where(
        psi_arr[None, :] > 0.0,
        jax.vmap(lambda p: _multimass_density_sources(p, rescale, W0, g), out_axes=1)(psi_arr),
        0.0,
    )


def _aniso_v2hat_scalar(W, p, g):
    """Dimensionless 3-D mean-square speed <u^2> of the anisotropic speed distribution
    at potential W and anisotropy p (in units of s_j^2). Since the speed |v|=s_j u is
    angle-independent, <u^2> = int u^4 E_gamma(g,W-u^2/2) T(p^2 u^2/2) du /
    int u^2 E_gamma(...) T(...) du, with T the tangential-angle integral (= isotropic
    moment when p -> 0)."""
    W_pos = jnp.maximum(W, 1e-12)
    u = jnp.linspace(0.0, jnp.sqrt(2.0 * W_pos), _N_U_VIR)
    E = lowered_exponential(g, W_pos - u**2 / 2.0)
    T = _angle_integral_T(p**2 * u**2 / 2.0)
    num = jnp.trapezoid(u**4 * E * T, u)
    den = jnp.trapezoid(u**2 * E * T, u)
    return jnp.where(W > 0.0, num / jnp.maximum(den, 1e-300), 0.0)


def solve_multicomponent_limepy(
    alpha_j: Float[Array, "n_comp"],
    rescale_j: Float[Array, "n_comp"],
    W0: float,
    g: float,
    xi_max: float = 300.0,
    n_points: int = 2000,
    ra_hat_j: Float[Array, "n_comp"] | None = None,
) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"], Float[Array, "n_comp n_points"]]:
    """Solve the GENERAL multi-component coupled LIMEPY Poisson equation (Engine A core).

    Given central density fractions alpha_j (sum to 1) and a DIRECT per-component
    potential-depth rescaling rescale_j, solve

        (1/xi^2) d/dxi(xi^2 dpsi/dxi) = -9 sum_j alpha_j rho_hat_j(rescale_j psi),

    with the shared King-radius (-9) nondimensionalization. Each component rides the one
    shared dimensionless potential psi(xi) at its own rescaled depth rescale_j*psi, so it
    has its own concentration. rescale_j = (s/s_j)^2 is the squared ratio of the reference
    velocity scale s to the component's velocity scale s_j -- the SINGLE free per-component
    scale (a smaller s_j i.e. larger rescale_j is colder and more concentrated). All
    rescale_j = 1 reproduces the single-component LIMEPY profile exactly.

    Mass segregation is the convenience rescale_j = mu_j^(2 delta) (see
    solve_multimass_limepy); two equal-mass populations of different concentration
    (GC 1G/2G, halo+core) just take different rescale_j directly.

    Optional per-component radial anisotropy: ra_hat_j[j] = hat_r_{a,j} = r_{a,j}/r_c is
    the component anisotropy radius (None = isotropic, fast path). The component source is
    then the anisotropic density at p_j(xi) = xi/ra_hat_j and rescaled potential
    rescale_j*psi.

    JIT/grad-safe in (alpha_j, rescale_j, W0, g, ra_hat_j); n_points, xi_max static.

    Returns:
        (xi_grid, psi_grid, rho_j_grid): dimensionless radius, shared potential W(xi)>=0,
        and per-component normalized densities rho_hat_j(xi), shape (n_comp, n_points).
    """
    alpha_j = jnp.asarray(alpha_j)
    rescale = jnp.asarray(rescale_j)
    isotropic = ra_hat_j is None

    if isotropic:
        def density_components(xi, psi):  # (n_comp,)
            return _multimass_density_sources(psi, rescale, W0, g)
    else:
        ra_j = jnp.asarray(ra_hat_j)  # per-component anisotropy radius (n_comp,)
        rho0_j = jax.vmap(lambda res: _aniso_density_scalar(res * W0, jnp.asarray(0.0), g))(rescale)

        def density_components(xi, psi):  # (n_comp,) normalized to central
            p_j = xi / ra_j
            rho = jax.vmap(lambda res, pj: _aniso_density_scalar(res * psi, pj, g))(rescale, p_j)
            return jnp.where(rho0_j > 1e-300, rho / rho0_j, 0.0)

    def rhs(xi, y, args):
        psi, dpsi = y[0], y[1]
        rho_tot = jnp.sum(alpha_j * density_components(xi, psi))
        d2 = jnp.where(xi > 1e-6, -9.0 * rho_tot - (2.0 / xi) * dpsi, -9.0 * rho_tot)
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
    psi_end = solution.ys[-1, 0]
    psi_grid = jnp.maximum(solution.ys[:, 0], 0.0)

    rho_j_grid = jax.vmap(density_components, out_axes=1)(xi_grid, psi_grid)
    rho_j_grid = jnp.where(psi_grid[None, :] > 0.0, rho_j_grid, 0.0)

    # Non-truncation guard for anisotropic models (concrete inputs only; mirrors
    # solve_michie / solve_limepy: too-small ra -> radial-orbit 1/r^2 tail -> no finite
    # tidal radius). Skipped under tracing -- psi_end is a tracer when ANY input is
    # differentiated, even if ra_hat_j itself is concrete.
    if not isotropic:
        try:
            psi_end_val = float(psi_end)
            W0_val = float(W0)
        except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError,
                TypeError):
            psi_end_val = None
        if psi_end_val is not None and psi_end_val > 1e-3 * W0_val:
            raise ValueError(
                f"Anisotropic multi-component LIMEPY (W0={W0_val}, r_a/r_c={ra_hat_j}) "
                f"does not truncate within xi_max={xi_max} (W(xi_max)={psi_end_val:.3f}): "
                f"the anisotropy is too strong (no finite tidal radius). Increase ra / xi_max."
            )
    return xi_grid, psi_grid, rho_j_grid


def solve_multimass_limepy(
    alpha_j: Float[Array, "n_comp"],
    m_j: Float[Array, "n_comp"],
    W0: float,
    g: float,
    delta: float,
    xi_max: float = 300.0,
    n_points: int = 2000,
    ra_hat: float | None = None,
    eta: float = 0.0,
) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"], Float[Array, "n_comp n_points"]]:
    """Mass-segregation convenience over solve_multicomponent_limepy (Engine A).

    The Gieles & Zocchi (2015) equipartition parametrization: per-component rescaling
    rescale_j = mu_j^(2 delta), mu_j = m_j / bar_m, bar_m = sum_j m_j alpha_j (central
    density-weighted mean mass, Eq. 26), and per-component anisotropy radius
    hat_r_{a,j} = ra_hat * mu_j^eta (eta=0 = mass-independent, the paper default). At
    delta=0 every rescaling is 1 -- identical to solve_limepy_profile. ra_hat=None is the
    (fast) isotropic case.

    JIT/grad-safe in (alpha_j, m_j, W0, g, delta, ra_hat, eta); n_points, xi_max static.

    Returns (xi_grid, psi_grid, rho_j_grid) as solve_multicomponent_limepy.
    """
    alpha_j = jnp.asarray(alpha_j)
    m_j = jnp.asarray(m_j)
    bar_m = jnp.sum(m_j * alpha_j)
    mu_j = m_j / bar_m
    rescale_j = mu_j ** (2.0 * delta)  # mu_j^(2 delta) per component
    ra_hat_j = None if ra_hat is None else ra_hat * mu_j ** eta
    return solve_multicomponent_limepy(
        alpha_j, rescale_j, W0, g, xi_max=xi_max, n_points=n_points, ra_hat_j=ra_hat_j
    )


def _realized_fractions(
    alpha_j: Float[Array, "n_comp"],
    m_j: Float[Array, "n_comp"],
    W0: float,
    g: float,
    delta: float,
    xi_max: float,
    n_points: int,
    ra_hat=None,
    eta: float = 0.0,
) -> Float[Array, "n_comp"]:
    """Realized mass fractions f_j' = alpha_j nu_j / sum_k alpha_k nu_k from a coupled
    solve, nu_j = int rho_hat_j xi^2 dxi (the per-component dimensionless mass)."""
    xi, _, rho_j = solve_multimass_limepy(alpha_j, m_j, W0, g, delta, xi_max, n_points,
                                          ra_hat=ra_hat, eta=eta)
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
    ra_hat=None,
    eta: float = 0.0,
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
        f_real = _realized_fractions(alpha, m_j, W0, g, delta, xi_max, n_points, ra_hat, eta)
        alpha_new = alpha * jnp.sqrt(f_target / (f_real + 1e-300))
        alpha_new = alpha_new / jnp.sum(alpha_new)
        return alpha_new, None

    alpha_final, _ = jax.lax.scan(step, f_target, None, length=n_iter)
    f_real = _realized_fractions(alpha_final, m_j, W0, g, delta, xi_max, n_points, ra_hat, eta)
    residual = jnp.max(jnp.abs(f_real - f_target))
    return alpha_final, residual


# ==============================================================================
# Layer C helpers (IMF binning + sampling resolutions), used by
# progenax.cluster.multicomponent.MultiComponentCluster
# ==============================================================================

_N_SPEED = 256  # per-particle speed inverse-CDF resolution
_N_C = 128      # per-particle cos(theta) inverse-CDF resolution (anisotropic conditional)


def _isotropic_dirs(key: PRNGKeyArray, n: int) -> Float[Array, "n 3"]:
    """n random unit vectors (isotropic)."""
    v = jax.random.normal(key, (n, 3))
    return v / (jnp.linalg.norm(v, axis=1, keepdims=True) + 1e-30)


def _bin_imf(imf, n_comp: int, m_range):
    """Bin an IMF into n_comp log-spaced mass components -> (m_j, M_j).

    Per bin: number N_j = int xi dm (from the IMF number CDF), mass
    M_j = int m xi dm (trapezoid of m * pdf over a sub-grid), representative mass
    m_j = M_j / N_j (number-weighted mean). Test/convenience path (concrete inputs).
    """
    edges = jnp.asarray(__import__("numpy").geomspace(m_range[0], m_range[1], n_comp + 1))
    N_j, M_j = [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        n_sub = 64
        m_sub = jnp.linspace(e0, e1, n_sub)
        pdf = jnp.exp(imf.logpdf(m_sub))
        N = jnp.trapezoid(pdf, m_sub)
        M = jnp.trapezoid(m_sub * pdf, m_sub)
        N_j.append(N)
        M_j.append(M)
    N_j = jnp.array(N_j)
    M_j = jnp.array(M_j)
    m_j = M_j / N_j
    return m_j, M_j


__all__ = [
    "solve_multicomponent_limepy",
    "solve_multimass_limepy",
    "find_alpha_for_masses",
]
