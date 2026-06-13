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
  B. find_alpha_for_masses(m_j, M_j, ...) -- the GZ15 Section 4 mass-function
     iteration ("eigenvalue problem ... solved by iteration") for the alpha_j that
     reproduce target masses, using the GZ15 stabilized sqrt-update
     alpha_j <- alpha_j sqrt(M_j/M_j') -- NOT Gunn & Griffin's linear M_j/M_j'
     update, which diverges for wide mass functions (this module).
  C. The user-facing model is progenax.cluster.multicomponent.MultiComponentCluster;
     this module keeps its grid/density/sampling helpers.

Isotropic and per-component anisotropic (ra_hat_j; mass path: r_{a,j} = r_a mu_j^eta).

References:
    Gieles, M. & Zocchi, A. (2015), MNRAS, 454, 576 (Eqs. 24-29, Section 4.1).
"""

import functools
from typing import Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax.profiles.limepy import (
    _aniso_density_scalar,
    _angle_integral_T,
    limepy_density_hat,
    lowered_exponential,
)
from progenax.profiles.limepy_tables import AnisoDensityTable


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


# Solver grid for the shared anisotropic density table, sized to the SOLVE-level
# budget (|psi_table - psi_quad| <= 1e-4 W0, |rho_j| <= 2e-4 absolute; asserted in
# tests/unit/profiles/test_limepy_tables.py::TestTableBackedSolver). Measured
# 2026-06 (W0=7, rescale=[1,1.6], ra_hat=10, xi_max=800, cubic Lagrange):
# max|dpsi| 8.9e-5, max|drho_j| 8.8e-5, warm solve 0.15 s vs 0.78 s quadrature.
_TAB_N_W = 160
_TAB_N_P = 40

# Jitted build: the tabulation is one big batched quadrature; jit caches its
# compilation across solves (n_W/n_p static, W_max/p_max/g dynamic -> grad-safe).
_build_density_table = jax.jit(AnisoDensityTable.build, static_argnames=("n_W", "n_p"))


def _solver_table(rescale, ra_j, W0, g, xi_max):
    """The ONE shared AnisoDensityTable covering the whole coupled-solve box:
    W <= max_j(rescale_j) W0, p <= xi_max / min_j(ra_hat_j).

    THE single source of the box formula: `MultiComponentCluster` builds this
    table once per construction and passes it to BOTH the Poisson solve
    (`aniso_table=`) and the mass-CDF density source, so the table quadrature
    (~0.4 s) runs once, not twice. All-inf ra_j (degenerate all-isotropic
    input; the caller's iso convention is ra_hat_j=None) would give
    p_max = 0 -> zero-width q grid -> NaN; floor it.
    """
    p_max = jnp.maximum(xi_max / jnp.min(ra_j), 1e-3)
    return _build_density_table(jnp.max(rescale) * W0, p_max, g,
                                n_W=_TAB_N_W, n_p=_TAB_N_P)


class _TableRHS(eqx.Module):
    """Table-backed anisotropic Poisson RHS for the coupled solve.

    An eqx.Module (not a closure) so diffrax's internal jit treats the table and
    parameters as DYNAMIC pytree leaves: repeated solves hit the compile cache.
    A fresh-closure RHS is re-traced and re-compiled on every diffeqsolve call
    (~0.4 s, measured 2026-06), which would defeat the table speedup.
    """

    alpha_j: Float[Array, "n_comp"]
    rescale: Float[Array, "n_comp"]
    ra_j: Float[Array, "n_comp"]
    rho0_j: Float[Array, "n_comp"]
    tab: AnisoDensityTable

    def density_components(self, xi, psi):  # (n_comp,) normalized to central
        p_j = xi / self.ra_j
        rho = jax.vmap(lambda res, pj: self.tab.evaluate(res * psi, pj))(self.rescale, p_j)
        return jnp.where(self.rho0_j > 1e-300, rho / self.rho0_j, 0.0)

    def __call__(self, xi, y, args):
        psi, dpsi = y[0], y[1]
        rho_tot = jnp.sum(self.alpha_j * self.density_components(xi, psi))
        d2 = jnp.where(xi > 1e-6, -9.0 * rho_tot - (2.0 / xi) * dpsi, -9.0 * rho_tot)
        return jnp.array([dpsi, d2])


def _aniso_density_fn(alpha_j, rescale, ra_j, W0, g, xi_max, aniso_method,
                      table=None):
    """Anisotropic per-component density source for the coupled RHS.

    Returns (density_components, rhs_fn_or_None). aniso_method is a STATIC
    Python string selecting a code path at trace time:

    "table" (default): ONE shared AnisoDensityTable covering the whole solve box
    (W <= max_j(rescale_j) W0, p <= xi_max / min_j(ra_hat_j); see
    `_solver_table`) replaces the pointwise quadrature -- the 86% hotspot.
    `table` reuses a prebuilt solve-box table (the MultiComponentCluster
    constructors build it ONCE and share it between the solve and the mass-CDF
    density source); None builds it here. Normalized by the TABLE's own
    central values rho0_j = tab(rescale_j W0, 0) so the interpolation error
    cancels at the centre exactly as the quadrature normalization does. Also
    returns the full RHS as an eqx.Module (see _TableRHS).

    "quadrature": the exact oracle path, verbatim; rhs_fn is None (the caller
    builds its closure RHS as before) and `table` is ignored. Both paths are
    differentiable in (rescale, ra_j); W0/g/alpha_j differentiability is
    inherited from the table build and solver grad tests.
    """
    if aniso_method == "table":
        tab = _solver_table(rescale, ra_j, W0, g, xi_max) if table is None else table
        rho0_j = jax.vmap(lambda res: tab.evaluate(res * W0, jnp.asarray(0.0)))(rescale)
        rhs_fn = _TableRHS(alpha_j, rescale, ra_j, rho0_j, tab)
        return rhs_fn.density_components, rhs_fn
    if aniso_method != "quadrature":
        raise ValueError(
            f"aniso_method must be 'table' or 'quadrature', got {aniso_method!r}")
    rho0_j = jax.vmap(lambda res: _aniso_density_scalar(res * W0, jnp.asarray(0.0), g))(rescale)

    def density_components(xi, psi):  # (n_comp,) normalized to central
        p_j = xi / ra_j
        rho = jax.vmap(lambda res, pj: _aniso_density_scalar(res * psi, pj, g))(rescale, p_j)
        return jnp.where(rho0_j > 1e-300, rho / rho0_j, 0.0)

    return density_components, None


def solve_multicomponent_limepy(
    alpha_j: Float[Array, "n_comp"],
    rescale_j: Float[Array, "n_comp"],
    W0: float,
    g: float,
    xi_max: float = 300.0,
    n_points: int = 2000,
    ra_hat_j: Float[Array, "n_comp"] | None = None,
    aniso_method: str = "table",
    aniso_table: AnisoDensityTable | None = None,
) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"], Float[Array, "n_points"], Float[Array, "n_comp n_points"]]:
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
    rescale_j*psi. aniso_method selects how that source is evaluated: "table" (default)
    uses one shared AnisoDensityTable over the solve box (|psi - psi_quad| <= 1e-4 W0,
    |rho_j| <= 2e-4, asserted in tests); "quadrature" is the exact oracle path. It is a
    STATIC Python string (path selection at trace time); ignored when ra_hat_j is None.
    aniso_table (private-ish) reuses a prebuilt `_solver_table(...)` for the "table"
    path instead of building one here -- the MultiComponentCluster constructors build
    the table ONCE and share it with the mass-CDF grid; it MUST cover the solve box
    (same formula) and is ignored unless aniso_method="table" with finite ra_hat_j.

    JIT/grad-safe in (alpha_j, rescale_j, W0, g, ra_hat_j); n_points, xi_max, aniso_method static.

    Returns:
        (xi_grid, psi_grid, psi_raw, rho_j_grid): dimensionless radius, shared
        CLAMPED potential W(xi)>=0 (for density/CDF/virial), the UNCLAMPED psi_raw
        (negative past the zero-crossing; feed to `_find_tidal_radius` so d(r_t)/dW0
        flows -- the clamp would zero the crossing-node gradient, as in
        solve_king_profile), and per-component normalized densities rho_hat_j(xi),
        shape (n_comp, n_points). The forward r_t value is interpolation-identical
        to the clamped path; only the gradient differs.
    """
    alpha_j = jnp.asarray(alpha_j)
    rescale = jnp.asarray(rescale_j)
    isotropic = ra_hat_j is None

    rhs_fn = None  # the table path supplies an eqx.Module RHS (jit-cache stable)
    if isotropic:
        def density_components(xi, psi):  # (n_comp,)
            return _multimass_density_sources(psi, rescale, W0, g)
    else:
        density_components, rhs_fn = _aniso_density_fn(
            alpha_j, rescale, jnp.asarray(ra_hat_j), W0, g, xi_max, aniso_method,
            table=aniso_table)

    if rhs_fn is None:
        def rhs(xi, y, args):
            psi, dpsi = y[0], y[1]
            rho_tot = jnp.sum(alpha_j * density_components(xi, psi))
            d2 = jnp.where(xi > 1e-6, -9.0 * rho_tot - (2.0 / xi) * dpsi, -9.0 * rho_tot)
            return jnp.array([dpsi, d2])
        rhs_fn = rhs

    y0 = jnp.array([W0, 0.0])
    xi_span = (1e-6, xi_max)
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs_fn),
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
    psi_raw = solution.ys[:, 0]  # UNCLAMPED W(xi) (negative past r_t)
    psi_grid = jnp.maximum(psi_raw, 0.0)

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
    return xi_grid, psi_grid, psi_raw, rho_j_grid


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
    aniso_method: str = "table",
) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"], Float[Array, "n_points"], Float[Array, "n_comp n_points"]]:
    """Mass-segregation convenience over solve_multicomponent_limepy (Engine A).

    The Gieles & Zocchi (2015) equipartition parametrization: per-component rescaling
    rescale_j = mu_j^(2 delta), mu_j = m_j / bar_m, bar_m = sum_j m_j alpha_j (central
    density-weighted mean mass, Eq. 26), and per-component anisotropy radius
    hat_r_{a,j} = ra_hat * mu_j^eta (eta=0 = mass-independent, the paper default). At
    delta=0 every rescaling is 1 -- identical to solve_limepy_profile. ra_hat=None is the
    (fast) isotropic case. aniso_method ("table" default, "quadrature" oracle) is passed
    through to solve_multicomponent_limepy; ignored when ra_hat is None.

    JIT/grad-safe in (alpha_j, m_j, W0, g, delta, ra_hat, eta); n_points, xi_max,
    aniso_method static.

    Returns (xi_grid, psi_grid, psi_raw, rho_j_grid) as solve_multicomponent_limepy.
    """
    alpha_j = jnp.asarray(alpha_j)
    m_j = jnp.asarray(m_j)
    bar_m = jnp.sum(m_j * alpha_j)
    mu_j = m_j / bar_m
    rescale_j = mu_j ** (2.0 * delta)  # mu_j^(2 delta) per component
    ra_hat_j = None if ra_hat is None else ra_hat * mu_j ** eta
    return solve_multicomponent_limepy(
        alpha_j, rescale_j, W0, g, xi_max=xi_max, n_points=n_points, ra_hat_j=ra_hat_j,
        aniso_method=aniso_method,
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
    aniso_method: str = "table",
) -> Float[Array, "n_comp"]:
    """Realized mass fractions f_j' = alpha_j nu_j / sum_k alpha_k nu_k from a coupled
    solve, nu_j = int rho_hat_j xi^2 dxi (the per-component dimensionless mass)."""
    xi, _, _, rho_j = solve_multimass_limepy(alpha_j, m_j, W0, g, delta, xi_max, n_points,
                                             ra_hat=ra_hat, eta=eta,
                                             aniso_method=aniso_method)
    nu_j = jnp.trapezoid(rho_j * xi**2, xi, axis=1)
    M_real = alpha_j * nu_j
    return M_real / (jnp.sum(M_real) + 1e-300)


# ------------------------------------------------------------------------------
# Layer B fixed-point map + residual (shared by the two custom_vjp solvers below).
# Explicit args (no closure) so jax.vjp is clean for the implicit backward.
# ------------------------------------------------------------------------------
def _alpha_map(alpha, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta,
               aniso_method):
    """One Gieles & Zocchi sqrt-update: alpha <- normalize(alpha sqrt(f_target/f_real))."""
    f_real = _realized_fractions(alpha, m_j, W0, g, delta, xi_max, n_points, ra_hat, eta,
                                 aniso_method)
    a = alpha * jnp.sqrt(f_target / (f_real + 1e-300))
    return a / jnp.sum(a)


def _alpha_residual(alpha, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta,
                    aniso_method):
    """Fixed-point residual R(alpha, theta) = alpha - sqrt-map(alpha); zero at alpha*.

    The implicit VJP differentiates THIS residual (cond ~2.6 with a benign Sigma=0
    simplex null direction handled by lstsq), NOT f_real - f_target (cond ~1e16).
    """
    return alpha - _alpha_map(alpha, m_j, f_target, W0, g, delta, xi_max, n_points,
                              ra_hat, eta, aniso_method)


# ------------------------------------------------------------------------------
# ISOTROPIC solver (ra_hat is None): ra_hat, eta are NONDIFF (closed over), so the
# backward never tries to emit a cotangent for the Python None ra_hat. Diff set is
# (m_j, M_j, W0, g, delta) -> bwd returns a 5-tuple. This is the demo + released
# grad-test path. ra_hat MUST stay a Python None so _realized_fractions hits its
# `is None` branch (the fast isotropic table path); a traced sentinel would route
# the isotropic case through the slow/wrong anisotropic path.
# nondiff_argnums = (5,6,7,8,9,10,11) = (ra_hat, eta, xi_max, n_points,
#                                        aniso_method, tol, max_iter)
# ------------------------------------------------------------------------------
@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11))
def _solve_alpha_iso(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points,
                     aniso_method, tol, max_iter):
    f_target = M_j / jnp.sum(M_j)

    def cond(s):
        _, it, r = s
        return jnp.logical_and(it < max_iter, r > tol)

    def body(s):
        a, it, _ = s
        a_new = _alpha_map(a, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta,
                           aniso_method)
        f_real = _realized_fractions(a_new, m_j, W0, g, delta, xi_max, n_points, ra_hat,
                                     eta, aniso_method)
        return a_new, it + 1, jnp.max(jnp.abs(f_real - f_target))

    a_star, _, _ = jax.lax.while_loop(
        cond, body, (f_target, jnp.array(0), jnp.array(jnp.inf)))
    return a_star


def _solve_alpha_iso_fwd(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points,
                         aniso_method, tol, max_iter):
    a_star = _solve_alpha_iso(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points,
                              aniso_method, tol, max_iter)
    return a_star, (a_star, m_j, M_j, W0, g, delta)


def _solve_alpha_iso_bwd(ra_hat, eta, xi_max, n_points, aniso_method, tol, max_iter,
                         res, a_bar):
    a_star, m_j, M_j, W0, g, delta = res
    f_target = M_j / jnp.sum(M_j)
    R_a = lambda a: _alpha_residual(a, m_j, f_target, W0, g, delta, xi_max, n_points,
                                    ra_hat, eta, aniso_method)
    _, vjp_a = jax.vjp(R_a, a_star)
    J = jax.vmap(lambda e: vjp_a(e)[0])(jnp.eye(a_star.shape[0]))  # n x n, reverse-mode
    w = jnp.linalg.lstsq(J.T, a_bar, rcond=None)[0]
    R_th = lambda mj, Mj, W, gg, d: _alpha_residual(
        a_star, mj, Mj / jnp.sum(Mj), W, gg, d, xi_max, n_points, ra_hat, eta,
        aniso_method)
    _, vjp_th = jax.vjp(R_th, m_j, M_j, W0, g, delta)
    gm, gM, gW, gg, gd = vjp_th(w)
    return (-gm, -gM, -gW, -gg, -gd)


_solve_alpha_iso.defvjp(_solve_alpha_iso_fwd, _solve_alpha_iso_bwd)


# ------------------------------------------------------------------------------
# ANISOTROPIC solver (ra_hat finite): differentiate (m_j, M_j, W0, g, delta,
# ra_hat, eta) -> bwd returns a 7-tuple. Statics via nondiff_argnums = (7,8,9,10,11).
# ------------------------------------------------------------------------------
@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11))
def _solve_alpha_aniso(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points,
                       aniso_method, tol, max_iter):
    f_target = M_j / jnp.sum(M_j)

    def cond(s):
        _, it, r = s
        return jnp.logical_and(it < max_iter, r > tol)

    def body(s):
        a, it, _ = s
        a_new = _alpha_map(a, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta,
                           aniso_method)
        f_real = _realized_fractions(a_new, m_j, W0, g, delta, xi_max, n_points, ra_hat,
                                     eta, aniso_method)
        return a_new, it + 1, jnp.max(jnp.abs(f_real - f_target))

    a_star, _, _ = jax.lax.while_loop(
        cond, body, (f_target, jnp.array(0), jnp.array(jnp.inf)))
    return a_star


def _solve_alpha_aniso_fwd(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points,
                           aniso_method, tol, max_iter):
    a_star = _solve_alpha_aniso(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points,
                                aniso_method, tol, max_iter)
    return a_star, (a_star, m_j, M_j, W0, g, delta, ra_hat, eta)


def _solve_alpha_aniso_bwd(xi_max, n_points, aniso_method, tol, max_iter, res, a_bar):
    a_star, m_j, M_j, W0, g, delta, ra_hat, eta = res
    f_target = M_j / jnp.sum(M_j)
    R_a = lambda a: _alpha_residual(a, m_j, f_target, W0, g, delta, xi_max, n_points,
                                    ra_hat, eta, aniso_method)
    _, vjp_a = jax.vjp(R_a, a_star)
    J = jax.vmap(lambda e: vjp_a(e)[0])(jnp.eye(a_star.shape[0]))  # n x n, reverse-mode
    w = jnp.linalg.lstsq(J.T, a_bar, rcond=None)[0]
    R_th = lambda mj, Mj, W, gg, d, rah, et: _alpha_residual(
        a_star, mj, Mj / jnp.sum(Mj), W, gg, d, xi_max, n_points, rah, et, aniso_method)
    _, vjp_th = jax.vjp(R_th, m_j, M_j, W0, g, delta, ra_hat, eta)
    gm, gM, gW, gg, gd, gra, get = vjp_th(w)
    return (-gm, -gM, -gW, -gg, -gd, -gra, -get)


_solve_alpha_aniso.defvjp(_solve_alpha_aniso_fwd, _solve_alpha_aniso_bwd)


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
    aniso_method: str = "table",
    tol: float = 1e-6,
) -> Tuple[Float[Array, "n_comp"], Float[Array, ""]]:
    """Find the central density fractions alpha_j that reproduce target masses M_j (Layer B).

    Fixed-point iteration on the realized mass fractions, with the stabilized update of
    Gieles & Zocchi (2015, Section 4.1) -- NOT Gunn & Griffin's linear M_j/M_j' (which
    diverges for wide mass functions):

        alpha_j <- alpha_j sqrt(f_j / f_j'),   renormalize sum_j alpha_j = 1,

    f_j = M_j / sum M (target), f_j' = realized fraction. Starts from alpha_j = f_j.

    Solved by a hand-rolled jax.custom_vjp: the forward is an adaptive
    jax.lax.while_loop that iterates the sqrt-update until the residual
    max_j |f_j' - f_j| < tol (or the n_iter safety cap), and the backward is the
    EXACT fixed-point gradient via a reverse-mode implicit VJP of the sqrt-map
    residual R(alpha, theta) = alpha - sqrt-map(alpha) (n x n Jacobian by vmapped
    vjp, lstsq solve, -vjp_theta). This is flat-in-n_iter and ~3x faster per
    value_and_grad than the old unrolled lax.scan, with gradients matching central
    finite differences to <1e-5. Two solvers are dispatched on `ra_hat is None`:
    the isotropic solver keeps (ra_hat, eta) out of the differentiated set so it can
    take the fast isotropic density path; the anisotropic solver also
    differentiates (ra_hat, eta).

    The iteration deliberately uses the SAME aniso_method as the final solve
    (default "table") so the converged alpha_j are self-consistent with the model
    actually built; the residual remains a reported diagnostic. Pass
    aniso_method="quadrature" for the exact oracle path.

    Args:
        m_j: component representative masses. M_j: target mass per component.
        W0, g, delta: model parameters. n_iter: forward iteration safety cap.
        xi_max, n_points: ODE grid (static). aniso_method: density-source path
        ("table" default, "quadrature" oracle; static, ignored when ra_hat is None).
        tol: forward residual tolerance for the adaptive while_loop.

    Returns:
        (alpha_j, residual): converged central density fractions (sum to 1, positive)
        and the final fractional residual max_j |f_j' - f_j| (reported, never branched on).
    """
    m_j = jnp.asarray(m_j)
    M_j = jnp.asarray(M_j)
    W0 = jnp.asarray(W0)
    g = jnp.asarray(g)
    delta = jnp.asarray(delta)
    if ra_hat is None:
        # iso: eta passed bare -- it is nondiff here; jnp.asarray(eta) would become a
        # tracer under jit in a nondiff_argnums slot and crash jit(grad).
        alpha = _solve_alpha_iso(m_j, M_j, W0, g, delta, None, eta,
                                 xi_max, n_points, aniso_method, tol, n_iter)
    else:
        alpha = _solve_alpha_aniso(m_j, M_j, W0, g, delta, jnp.asarray(ra_hat),
                                   jnp.asarray(eta), xi_max, n_points, aniso_method,
                                   tol, n_iter)
    f_real = _realized_fractions(alpha, m_j, W0, g, delta, xi_max, n_points, ra_hat,
                                 eta, aniso_method)
    residual = jnp.max(jnp.abs(f_real - M_j / jnp.sum(M_j)))
    return alpha, residual


# ==============================================================================
# Layer C helpers (IMF binning + sampling resolutions), used by
# progenax.cluster.multicomponent.MultiComponentCluster
# ==============================================================================

_N_SPEED = 256  # per-particle speed inverse-CDF resolution
# (_N_C, the cos(theta) resolution of the anisotropic angular conditional,
#  moved to progenax.kinematics._speed_kernels with its kernel.)


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
