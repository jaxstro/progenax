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
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax import defaults
from progenax.profiles.king import _find_tidal_radius
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


def _grid_density_components(psi_arr, xi_arr, rescale, mu_j, W0, g, ra_hat, eta, is_aniso):
    """Per-component normalized densities on a grid, shape (n_comp, len(psi_arr)).

    Isotropic: rho_hat_j = limepy_density_hat(mu_j^(2 delta) psi)/norm. Anisotropic
    (finite ra_hat): the anisotropic density at p_j = xi/(ra_hat mu_j^eta), normalized to
    the central (p=0) value. xi_arr is the dimensionless radius r/r_c.
    """
    if is_aniso:
        ra_j = ra_hat * mu_j ** eta
        rho0_j = jax.vmap(lambda res: _aniso_density_scalar(res * W0, jnp.asarray(0.0), g))(rescale)

        def at(xi, psi):
            p_j = xi / ra_j
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
    """Solve the multi-mass coupled LIMEPY Poisson equation (Layer A).

    Given central density fractions alpha_j (sum to 1) and component masses m_j, solve

        (1/xi^2) d/dxi(xi^2 dpsi/dxi) = -9 sum_j alpha_j rho_hat_j(psi),

    with the shared King-radius (-9) nondimensionalization. The per-component rescaling
    is mu_j^(2 delta), mu_j = m_j / bar_m, bar_m = sum_j m_j alpha_j (central
    density-weighted mean mass, Eq. 26). At delta=0 every rescaling is 1 and the source
    is (sum alpha_j) rho_hat = rho_hat -- identical to solve_limepy_profile.

    Optional per-component radial anisotropy (Gieles & Zocchi Eq. 25): with a finite
    ra_hat = r_a/r_c, component j has anisotropy radius hat_r_{a,j} = ra_hat mu_j^eta,
    so its source is the anisotropic density at p_j(xi) = xi/(ra_hat mu_j^eta) and
    rescaled potential mu_j^(2 delta) psi. ra_hat=None is the (fast) isotropic case.
    eta=0 is mass-independent anisotropy (the paper default).

    JIT/grad-safe in (alpha_j, m_j, W0, g, delta, ra_hat, eta): n_points, xi_max static.

    Returns:
        (xi_grid, psi_grid, rho_j_grid): dimensionless radius, shared potential W(xi)>=0,
        and per-component normalized densities rho_hat_j(xi), shape (n_comp, n_points).
    """
    alpha_j = jnp.asarray(alpha_j)
    m_j = jnp.asarray(m_j)
    bar_m = jnp.sum(m_j * alpha_j)
    mu_j = m_j / bar_m
    rescale = mu_j ** (2.0 * delta)  # mu_j^(2 delta) per component
    isotropic = ra_hat is None

    if isotropic:
        def density_components(xi, psi):  # (n_comp,)
            return _multimass_density_sources(psi, rescale, W0, g)
    else:
        ra_j = ra_hat * mu_j ** eta  # per-component anisotropy radius (n_comp,)
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
    # solve_michie / solve_limepy: too-small ra_hat -> radial-orbit 1/r^2 tail -> no
    # finite tidal radius). Skipped under tracing -- psi_end is a tracer when ANY input
    # (incl. eta/delta) is differentiated, even if ra_hat itself is concrete.
    if not isotropic:
        try:
            psi_end_val = float(psi_end)
            W0_val = float(W0)
        except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError,
                TypeError):
            psi_end_val = None
        if psi_end_val is not None and psi_end_val > 1e-3 * W0_val:
            raise ValueError(
                f"Anisotropic multi-mass LIMEPY (W0={W0_val}, r_a/r_c={ra_hat}, eta={eta}) "
                f"does not truncate within xi_max={xi_max} (W(xi_max)={psi_end_val:.3f}): "
                f"the anisotropy is too strong (no finite tidal radius). Increase ra_hat / xi_max."
            )
    return xi_grid, psi_grid, rho_j_grid


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
# Layer C: MultiMassLIMEPY model (construction, IMF binning, segregated-IC sampling)
# ==============================================================================

_N_SPEED = 256  # per-particle speed inverse-CDF resolution


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


class MultiMassLIMEPY(eqx.Module):
    """Multi-mass LIMEPY coupled equilibrium model (isotropic; Layer C).

    A mass-segregated star cluster in which segregation is a TRUE equilibrium: one
    shared potential, per-component velocity scales s_j = s mu_j^(-delta). Each mass
    component is individually virial (Q_j ~ 0.5). Construct with `from_alpha` (direct
    central density fractions) or `from_imf` (bin an IMF, solve for alpha_j). Isotropic;
    per-component anisotropy (eta) is a planned extension (Phase 2b).

    Attributes:
        W0, g, delta, r_c, r_t: model parameters / scales.
        m_j, alpha_j, mu_j, rescale_j (= mu_j^(2 delta)), nu_j, N_frac_j: per component.
        mu_tot: total dimensionless mass integral (sets the velocity scale).
        residual: eigenvalue-solve residual (0 for from_alpha).
        xi_grid, psi_grid: shared ODE solution. _r_grid, _cdf_j: per-component mass CDFs.
    """

    W0: Float[Array, ""]
    g: Float[Array, ""]
    delta: Float[Array, ""]
    r_c: Float[Array, ""]
    r_t: Float[Array, ""]
    r_a: Float[Array, ""]
    eta: Float[Array, ""]
    m_j: Float[Array, "n_comp"]
    alpha_j: Float[Array, "n_comp"]
    mu_j: Float[Array, "n_comp"]
    rescale_j: Float[Array, "n_comp"]
    nu_j: Float[Array, "n_comp"]
    N_frac_j: Float[Array, "n_comp"]
    mu_tot: Float[Array, ""]
    residual: Float[Array, ""]
    xi_grid: Float[Array, "n_ode"]
    psi_grid: Float[Array, "n_ode"]
    _r_grid: Float[Array, "n_grid"]
    _cdf_j: Float[Array, "n_comp n_grid"]
    is_aniso: bool = eqx.field(static=True)

    def __init__(self, alpha_j, m_j, W0, g, delta, r_c, xi_grid, psi_grid,
                 r_a=None, eta=0.0, residual=0.0, n_grid: int = 1000):
        is_aniso = r_a is not None
        alpha_j = jnp.asarray(alpha_j, dtype=jnp.float64)
        m_j = jnp.asarray(m_j, dtype=jnp.float64)
        W0 = jnp.asarray(W0, dtype=jnp.float64)
        g = jnp.asarray(g, dtype=jnp.float64)
        delta = jnp.asarray(delta, dtype=jnp.float64)
        r_c = jnp.asarray(r_c, dtype=jnp.float64)
        r_a_arr = jnp.asarray(jnp.inf if r_a is None else r_a, dtype=jnp.float64)
        eta = jnp.asarray(eta, dtype=jnp.float64)
        xi_grid = jnp.asarray(xi_grid, dtype=jnp.float64)
        psi_grid = jnp.asarray(psi_grid, dtype=jnp.float64)

        bar_m = jnp.sum(m_j * alpha_j)
        mu_j = m_j / bar_m
        rescale = mu_j ** (2.0 * delta)
        ra_hat = r_a_arr / r_c  # inf for isotropic

        def dens(psi_arr, xi_arr):
            return _grid_density_components(psi_arr, xi_arr, rescale, mu_j, W0, g,
                                            ra_hat, eta, is_aniso)

        rho_on_xi = dens(psi_grid, xi_grid)
        rho_on_xi = jnp.where(psi_grid[None, :] > 0.0, rho_on_xi, 0.0)
        nu_j = jnp.trapezoid(rho_on_xi * xi_grid**2, xi_grid, axis=1)
        mu_tot = jnp.sum(alpha_j * nu_j)
        M_real = alpha_j * nu_j
        N_frac = (M_real / m_j) / jnp.sum(M_real / m_j)

        r_t = r_c * _find_tidal_radius(xi_grid, psi_grid)
        r_grid = jnp.linspace(0.0, r_t, n_grid)
        psi_r = jnp.interp(r_grid / r_c, xi_grid, psi_grid, left=W0, right=0.0)
        rho_j_r = dens(psi_r, r_grid / r_c)
        rho_j_r = jnp.where(r_grid[None, :] <= r_t, rho_j_r, 0.0)

        integrand = 4.0 * jnp.pi * r_grid[None, :] ** 2 * rho_j_r
        dr = r_grid[1] - r_grid[0]
        M_cum = jnp.concatenate([
            jnp.zeros((rho_j_r.shape[0], 1)),
            jnp.cumsum(0.5 * (integrand[:, 1:] + integrand[:, :-1]), axis=1) * dr,
        ], axis=1)
        cdf_j = M_cum / (M_cum[:, -1:] + 1e-30)

        for name, val in dict(
            W0=W0, g=g, delta=delta, r_c=r_c, r_t=r_t, r_a=r_a_arr, eta=eta,
            m_j=m_j, alpha_j=alpha_j, mu_j=mu_j, rescale_j=rescale, nu_j=nu_j,
            N_frac_j=N_frac, mu_tot=mu_tot,
            residual=jnp.asarray(residual, dtype=jnp.float64),
            xi_grid=xi_grid, psi_grid=psi_grid, _r_grid=r_grid, _cdf_j=cdf_j,
        ).items():
            object.__setattr__(self, name, val)
        object.__setattr__(self, "is_aniso", is_aniso)

    @classmethod
    def from_alpha(cls, alpha_j, m_j, W0, g, delta, r_c=1.0, r_a=None, eta=0.0,
                   xi_max: float = 300.0, n_ode_points: int = 2000, n_grid: int = 1000):
        """Direct constructor: solve the coupled equilibrium for given alpha_j (Layer A).

        r_a=None is the isotropic model; a finite r_a adds per-component anisotropy
        r_{a,j} = r_a mu_j^eta (eta=0 = mass-independent; pass a larger xi_max, e.g. 800).
        """
        ra_hat = None if r_a is None else r_a / r_c
        xi, psi, _ = solve_multimass_limepy(alpha_j, m_j, W0, g, delta, xi_max,
                                            n_ode_points, ra_hat=ra_hat, eta=eta)
        return cls(alpha_j, m_j, W0, g, delta, r_c, xi, psi, r_a=r_a, eta=eta,
                   residual=0.0, n_grid=n_grid)

    @classmethod
    def from_imf(cls, imf, n_comp, W0, g, delta, m_range=(0.1, 100.0), r_c=1.0,
                 r_a=None, eta=0.0, n_iter: int = 30, xi_max: float = 300.0,
                 n_ode_points: int = 2000, n_grid: int = 1000):
        """Bin an IMF into n_comp log-spaced components and solve for alpha_j (Layer B).

        Anisotropy (r_a, eta) is supported but the eigenvalue iteration with the
        per-component anisotropic density quadrature is expensive; reduce n_iter or use
        from_alpha for exploratory anisotropic work.
        """
        m_j, M_j = _bin_imf(imf, n_comp, m_range)
        ra_hat = None if r_a is None else r_a / r_c
        alpha_j, residual = find_alpha_for_masses(
            m_j, M_j, W0, g, delta, n_iter=n_iter, xi_max=xi_max, n_points=n_ode_points,
            ra_hat=ra_hat, eta=eta,
        )
        xi, psi, _ = solve_multimass_limepy(alpha_j, m_j, W0, g, delta, xi_max,
                                            n_ode_points, ra_hat=ra_hat, eta=eta)
        return cls(alpha_j, m_j, W0, g, delta, r_c, xi, psi, r_a=r_a, eta=eta,
                   residual=residual, n_grid=n_grid)

    def component_virial_ratios(self, n: int = 4000) -> Float[Array, "n_comp"]:
        """Theoretical per-component virial ratio Q_j = T_j/|W_j| from the model itself.

        The bias-free equilibrium proof (no sampling, no softening, no finite-N): for a
        component in steady state in the shared potential, 2 T_j + W_j = 0, so Q_j = 0.5.
        Computed in the smooth mean field, so it returns exactly 0.5 for every component
        of a self-consistent model -- the rigorous statement that the sampled per-group
        Q_j is a finite-N estimator of. Dimensionless (independent of G, M, r_c).

            T_j = int 0.5 rho_j <v^2>_j 4 pi r^2 dr,  <v^2>_j = s_j^2 * 3 Eg(g+5/2,W_j)/Eg(g+3/2,W_j)
            W_j = - int rho_j r (dphi/dr) 4 pi r^2 dr, dphi/dr = G M_enc(r)/r^2,
        with W_j(r) = mu_j^(2 delta) psi(r), s_j^2 = mu_j^(-2 delta)/(9 mu_tot) (G=M=r_c=1).
        """
        r = jnp.linspace(1e-3, self.r_t, n)
        psi = jnp.interp(r / self.r_c, self.xi_grid, self.psi_grid, left=self.W0, right=0.0)
        ra_hat = self.r_a / self.r_c
        # per-component mass density rho_j = alpha_j * rho_hat_j (normalization cancels in Q_j)
        rho_j = _grid_density_components(psi, r / self.r_c, self.rescale_j, self.mu_j,
                                         self.W0, self.g, ra_hat, self.eta, self.is_aniso)
        rho_j = jnp.where(r[None, :] <= self.r_t, rho_j, 0.0) * self.alpha_j[:, None]
        rho_tot = jnp.sum(rho_j, axis=0)

        integ = 4.0 * jnp.pi * r**2 * rho_tot
        dr = r[1] - r[0]
        M_enc = jnp.concatenate([jnp.zeros(1),
                                 jnp.cumsum(0.5 * (integ[1:] + integ[:-1])) * dr])
        # Normalize to total mass M=1 so the velocity scale s^2 = 1/(9 mu_tot) is
        # consistent with the potential (G = M = r_c = 1). W_j carries two powers of this
        # normalization (rho_j and M_enc), T_j one (rho_j) -- so it must NOT be dropped.
        norm = 1.0 / M_enc[-1]
        rho_j = rho_j * norm
        dphi_dr = (M_enc * norm) / jnp.maximum(r, 1e-6) ** 2  # G=1

        s2 = 1.0 / (9.0 * self.mu_tot)  # G=M=r_c=1
        Qs = []
        for j in range(self.m_j.shape[0]):
            W_j = self.rescale_j[j] * psi
            s_j2 = s2 * self.mu_j[j] ** (-2.0 * self.delta)
            if self.is_aniso:
                p_j = (r / self.r_c) / (ra_hat * self.mu_j[j] ** self.eta)
                v2hat = jax.vmap(lambda w, pp: _aniso_v2hat_scalar(w, pp, self.g))(W_j, p_j)
            else:
                v2hat = 3.0 * lowered_exponential(self.g + 2.5, W_j) / \
                    jnp.maximum(lowered_exponential(self.g + 1.5, W_j), 1e-300)
            v2 = s_j2 * jnp.where(W_j > 0.0, v2hat, 0.0)
            T = jnp.trapezoid(0.5 * rho_j[j] * v2 * 4.0 * jnp.pi * r**2, r)
            W = jnp.trapezoid(-rho_j[j] * r * dphi_dr * 4.0 * jnp.pi * r**2, r)
            Qs.append(T / jnp.abs(W))
        return jnp.stack(Qs)

    def total_density(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """Total (mass-weighted) volume density sum_j alpha_j rho_hat_j(r), 0 outside r_t."""
        r1 = jnp.atleast_1d(jnp.asarray(r))
        psi_r = jnp.interp(r1 / self.r_c, self.xi_grid, self.psi_grid, left=self.W0, right=0.0)
        rho_j = _grid_density_components(psi_r, r1 / self.r_c, self.rescale_j, self.mu_j,
                                         self.W0, self.g, self.r_a / self.r_c, self.eta,
                                         self.is_aniso)
        tot = jnp.sum(self.alpha_j[:, None] * rho_j, axis=0).reshape(jnp.shape(r))
        return jnp.where(r <= self.r_t, tot, 0.0)

    def sample_cluster(self, key, n_stars: int, G=None):
        """Sample a mass-segregated equilibrium IC -> (positions, velocities, masses).

        Each star is assigned a component by a categorical draw (probabilities N_frac_j),
        then its position is drawn from that component's mass CDF and its speed from
        u^2 E_gamma(g, W_j(r) - u^2/2) at scale s_j = s mu_j^(-delta), s^2 = G M /
        (9 r_c mu_tot). The total cluster mass M = sum_i m_i is the sum of the sampled
        stellar masses (as in the single-mass DF) -- so kinetic and potential energies
        use a consistent M and the cluster is virial (Q=0.5). Differentiable in
        (W0, g, delta) through the per-star scales.
        """
        from progenax.kinematics.limepy_df import _sample_unit_speed

        if self.is_aniso:
            raise NotImplementedError(
                "Anisotropic multi-mass IC sampling is not yet implemented (Phase 2b "
                "delivered the anisotropic equilibrium model + theoretical virial; the "
                "per-component anisotropic velocity sampler is a follow-up). Use the "
                "isotropic model (r_a=None) to sample, or component_virial_ratios() / "
                "total_density() on the anisotropic model."
            )
        if G is None:
            G = defaults.DEFAULT_UNITS.G
        k_assign, k_pos, k_pdir, k_speed, k_vdir = jax.random.split(key, 5)

        c = jax.random.categorical(k_assign, jnp.log(self.N_frac_j + 1e-30), shape=(n_stars,))
        m_i = self.m_j[c]
        rescale_i = self.rescale_j[c]
        M_total = jnp.sum(m_i)  # the cluster mass IS the sum of its stars
        s = jnp.sqrt(G * M_total / (9.0 * self.r_c * self.mu_tot))
        s_i = s * self.mu_j[c] ** (-self.delta)

        # Positions: per-star inverse-CDF on its component's mass CDF + isotropic dirs.
        u = jax.random.uniform(k_pos, (n_stars,))
        radii = jax.vmap(lambda uu, cc: jnp.interp(uu, self._cdf_j[cc], self._r_grid))(u, c)
        pos = radii[:, None] * _isotropic_dirs(k_pdir, n_stars)

        # Velocities: per-star speed at the component's rescaled potential + scale s_i.
        W_i = rescale_i * jnp.maximum(
            jnp.interp(radii / self.r_c, self.xi_grid, self.psi_grid,
                       left=self.W0, right=0.0), 0.0
        )
        speed_keys = jax.random.split(k_speed, n_stars)
        u_speed = jax.vmap(lambda kk, w: _sample_unit_speed(kk, w, self.g, _N_SPEED))(
            speed_keys, W_i)
        vel = (s_i * u_speed)[:, None] * _isotropic_dirs(k_vdir, n_stars)
        return pos, vel, m_i


__all__ = ["solve_multimass_limepy", "find_alpha_for_masses", "MultiMassLIMEPY"]
