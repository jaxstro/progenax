"""
Unified multi-component cluster equilibrium model (Engine A: lowered isothermal).

ONE differentiable model for any spherical multi-population cluster as a TRUE
shared-potential equilibrium: mass segregation, GC multiple populations (1G/2G),
binaries-vs-singles, halo+core. Each component j is a lowered-isothermal DF
(Woolley/King/Wilson via g; Gieles & Zocchi 2015) riding the one self-consistent
potential, and is defined by DIRECT per-component scales:

    w_j      = s_j / s            velocity-scale ratio -- THE free per-component scale
    rescale_j = w_j^-2            potential-depth rescaling in the coupled Poisson solve
    ra_hat_j = r_{a,j} / r_c      anisotropy radius (None = isotropic)

The representative stellar mass m_j is DECOUPLED from the dynamics: it labels the
stars (and sets number fractions), but the structure depends only on (alpha_j, w_j,
ra_hat_j). Mass segregation is the equipartition convenience w_j = mu_j^(-delta)
(`from_mass_segregation`); equal-mass populations of different concentration set
w_j directly (`from_components`). Every component is individually virial
(Q_j = 0.5) and the sampled cluster is globally virial without rescaling -- for
ANY mass spectrum (the property the legacy two-population superposition lacked).

Engine B (density-defined Eddington/Osipkov-Merritt for Plummer/EFF/King) is the
Phase-2 extension; this module is the Engine-A (DF-defined) member of the family.

References:
    Gieles, M. & Zocchi, A. (2015), MNRAS, 454, 576 (Eqs. 24-29, Section 4.1).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from progenax import defaults
from progenax.builders import ICResult, compute_stellar_radii
from progenax.profiles.king import _find_tidal_radius
from progenax.profiles.limepy import lowered_exponential
from progenax.profiles.limepy_multimass import (
    _N_C,
    _aniso_density_fn,
    _aniso_v2hat_scalar,
    _bin_imf,
    _grid_density_components,
    _isotropic_dirs,
    _solver_table,
    find_alpha_for_masses,
    solve_multicomponent_limepy,
)
from progenax.profiles.limepy_tables import AnisoSpeedCDFTable, SpeedCDFTable


def _shared_table_and_dens_fn(alpha_j, rescale, ra_hat_j, W0, g, xi_max,
                              aniso_method):
    """Build the solve-box AnisoDensityTable ONCE per construction.

    Returns (table, dens_fn): the constructors pass `table` to the coupled
    Poisson solve (`aniso_table=`) AND wrap it as the CDF r-grid density
    source `dens_fn`, so the table quadrature runs once -- the same table the
    solve uses, hence the r-grid mass CDF in `MultiComponentCluster.__init__`
    uses the SAME method (and values) as the solve. (None, None) for isotropic
    models or aniso_method="quadrature" (the `__init__` quadrature fallback is
    then method-consistent by itself); any other method raises -- a future
    third method must never silently fall back to quadrature. The r-grid stays
    inside the table box: r_t <= xi_max * r_c (the solver's truncation guard)
    and W_j = rescale_j * psi <= max(rescale) * W0.
    """
    if ra_hat_j is None or aniso_method == "quadrature":
        return None, None
    if aniso_method != "table":
        raise ValueError(
            f"aniso_method must be 'table' or 'quadrature', got {aniso_method!r}")
    ra_arr = jnp.asarray(ra_hat_j, dtype=jnp.float64)
    table = _solver_table(rescale, ra_arr, W0, g, xi_max)
    dens_fn, _ = _aniso_density_fn(
        jnp.asarray(alpha_j, dtype=jnp.float64), rescale, ra_arr, W0, g,
        xi_max, "table", table=table)
    return table, dens_fn


class MultiComponentCluster(eqx.Module):
    """Multi-component lowered-isothermal cluster in shared-potential equilibrium.

    Construct with `from_components` (direct per-component velocity-scale ratios
    w_j -- the general case), `from_mass_segregation` (the equipartition law
    w_j = mu_j^(-delta)), or `from_imf` (bin an IMF, eigenvalue-solve for alpha_j).
    `sample_cluster` returns an :class:`~progenax.builders.ICResult` whose
    `component_id` labels each star's generating component.

    Differentiable in every per-component parameter (alpha_j, w_j, ra_hat_j, W0, g)
    through construction AND sampling.

    Attributes:
        W0, g, r_c, r_t: structural parameters / scales.
        m_j: representative stellar mass per component [M_sun] (labels only).
        alpha_j: central density fractions (sum to 1).
        w_j: per-component velocity-scale ratios s_j/s (rescale_j = w_j^-2).
        ra_hat_j: per-component anisotropy radii r_{a,j}/r_c (inf = isotropic).
        N_frac_j: number fraction of stars per component.
        mu_tot: total dimensionless mass integral (sets the velocity scale).
        residual: eigenvalue-solve residual (0 for direct constructors).
        xi_grid, psi_grid: shared coupled-Poisson solution W(xi).
    """

    W0: Float[Array, ""]
    g: Float[Array, ""]
    r_c: Float[Array, ""]
    r_t: Float[Array, ""]
    m_j: Float[Array, "n_comp"]
    alpha_j: Float[Array, "n_comp"]
    w_j: Float[Array, "n_comp"]
    ra_hat_j: Float[Array, "n_comp"]
    N_frac_j: Float[Array, "n_comp"]
    mu_tot: Float[Array, ""]
    residual: Float[Array, ""]
    xi_grid: Float[Array, "n_ode"]
    psi_grid: Float[Array, "n_ode"]
    _r_grid: Float[Array, "n_grid"]
    _cdf_j: Float[Array, "n_comp n_grid"]
    is_aniso: bool = eqx.field(static=True)

    def __init__(self, alpha_j, w_j, m_j, W0, g, r_c, xi_grid, psi_grid,
                 ra_hat_j=None, residual=0.0, n_grid: int = 1000,
                 rho_on_xi=None, dens_fn=None):
        """Assemble model state from a finished coupled solve; constructors
        forward `rho_on_xi` (the solver's (n_comp, n_ode) density grid, reused
        verbatim for nu_j/mu_tot) and `dens_fn` (the solve's pointwise
        (xi, psi) -> (n_comp,) density source, e.g. the shared-table source,
        used for the r-grid mass CDF) -- either may be None to recompute /
        fall back to exact quadrature."""
        is_aniso = ra_hat_j is not None
        alpha_j = jnp.asarray(alpha_j, dtype=jnp.float64)
        w_j = jnp.asarray(w_j, dtype=jnp.float64)
        m_j = jnp.asarray(m_j, dtype=jnp.float64)
        W0 = jnp.asarray(W0, dtype=jnp.float64)
        g = jnp.asarray(g, dtype=jnp.float64)
        r_c = jnp.asarray(r_c, dtype=jnp.float64)
        ra_arr = (jnp.full(w_j.shape, jnp.inf, dtype=jnp.float64) if ra_hat_j is None
                  else jnp.asarray(ra_hat_j, dtype=jnp.float64))
        xi_grid = jnp.asarray(xi_grid, dtype=jnp.float64)
        psi_grid = jnp.asarray(psi_grid, dtype=jnp.float64)
        rescale = w_j ** -2.0

        # Density evaluator for the r-grid mass CDF. Constructors forward the
        # solve's density source (`dens_fn`, pointwise (xi, psi) -> (n_comp,);
        # the shared-table source when aniso_method="table") so the CDF is
        # built with the SAME method as the coupled Poisson solve. dens_fn=None
        # (direct construction, isotropic, or aniso_method="quadrature") falls
        # back to the exact quadrature. `component_virial_ratios` stays on the
        # quadrature path unconditionally -- it is the equilibrium oracle.
        if dens_fn is None:
            def dens(psi_arr, xi_arr):
                return _grid_density_components(psi_arr, xi_arr, rescale, W0, g,
                                                ra_arr, is_aniso)
        else:
            def dens(psi_arr, xi_arr):
                return jax.vmap(dens_fn, out_axes=1)(xi_arr, psi_arr)

        # Reuse the solver's per-component density grid when provided (exact: the
        # solver evaluates the same normalized densities on the same (xi, psi));
        # recomputing it doubles the anisotropic quadrature cost for nothing.
        if rho_on_xi is None:
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
            W0=W0, g=g, r_c=r_c, r_t=r_t, m_j=m_j, alpha_j=alpha_j, w_j=w_j,
            ra_hat_j=ra_arr, N_frac_j=N_frac, mu_tot=mu_tot,
            residual=jnp.asarray(residual, dtype=jnp.float64),
            xi_grid=xi_grid, psi_grid=psi_grid, _r_grid=r_grid, _cdf_j=cdf_j,
        ).items():
            object.__setattr__(self, name, val)
        object.__setattr__(self, "is_aniso", is_aniso)

    @property
    def rescale_j(self) -> Float[Array, "n_comp"]:
        """Per-component potential-depth rescaling rescale_j = w_j^-2."""
        return self.w_j ** -2.0

    @classmethod
    def from_components(cls, alpha_j, w_j, m_j, W0, g, r_c=1.0, ra_hat_j=None,
                        xi_max: float = 300.0, n_ode_points: int = 2000,
                        n_grid: int = 1000, aniso_method: str = "table"):
        """Direct constructor: components defined by their velocity-scale ratios w_j.

        The general Engine-A case -- GC 1G/2G, halo+core, binaries-vs-singles: any
        set of populations, equal-mass or not, with per-component concentration set
        by w_j (smaller w_j = colder = more concentrated). ra_hat_j (per-component
        r_{a,j}/r_c) is the optional radial anisotropy; pass a larger xi_max
        (e.g. 800) for anisotropic models. aniso_method ("table" default,
        "quadrature" oracle) selects the anisotropic density source for the solve
        AND the mass-CDF grid -- a construction choice, not model state; ignored
        when ra_hat_j is None. `component_virial_ratios` always uses quadrature.
        """
        w_arr = jnp.asarray(w_j, dtype=jnp.float64)
        tab, dens_fn = _shared_table_and_dens_fn(alpha_j, w_arr ** -2.0, ra_hat_j,
                                                 W0, g, xi_max, aniso_method)
        xi, psi, rho_j = solve_multicomponent_limepy(
            alpha_j, w_arr ** -2.0, W0, g, xi_max=xi_max, n_points=n_ode_points,
            ra_hat_j=ra_hat_j, aniso_method=aniso_method, aniso_table=tab)
        return cls(alpha_j, w_arr, m_j, W0, g, r_c, xi, psi, ra_hat_j=ra_hat_j,
                   residual=0.0, n_grid=n_grid, rho_on_xi=rho_j, dens_fn=dens_fn)

    @classmethod
    def from_mass_segregation(cls, alpha_j, m_j, W0, g, delta, r_a=None, eta=0.0,
                              r_c=1.0, xi_max: float = 300.0,
                              n_ode_points: int = 2000, n_grid: int = 1000,
                              aniso_method: str = "table"):
        """Equipartition constructor: w_j = mu_j^(-delta) (Gieles & Zocchi 2015).

        mu_j = m_j / bar_m with bar_m = sum_j m_j alpha_j (central density-weighted
        mean mass, Eq. 26); heavier components are colder and sink. Optional
        anisotropy r_{a,j} = r_a mu_j^eta (eta=0 = mass-independent, the paper
        default). delta=0 is the unsegregated single-mass corner.
        """
        alpha_arr = jnp.asarray(alpha_j, dtype=jnp.float64)
        m_arr = jnp.asarray(m_j, dtype=jnp.float64)
        mu_j = m_arr / jnp.sum(m_arr * alpha_arr)
        w_j = mu_j ** (-delta)
        ra_hat_j = None if r_a is None else (r_a / r_c) * mu_j ** eta
        return cls.from_components(alpha_arr, w_j, m_arr, W0, g, r_c=r_c,
                                   ra_hat_j=ra_hat_j, xi_max=xi_max,
                                   n_ode_points=n_ode_points, n_grid=n_grid,
                                   aniso_method=aniso_method)

    @classmethod
    def from_imf(cls, imf, n_comp, W0, g, delta, m_range=(0.1, 100.0), r_c=1.0,
                 r_a=None, eta=0.0, n_iter: int = 30, xi_max: float = 300.0,
                 n_ode_points: int = 2000, n_grid: int = 1000,
                 aniso_method: str = "table"):
        """IMF constructor: bin into n_comp log-spaced components, solve for alpha_j.

        The eigenvalue solve (`find_alpha_for_masses`) finds the central density
        fractions that reproduce the IMF's per-bin mass budget under the
        equipartition law w_j = mu_j^(-delta). Anisotropy (r_a, eta) is supported
        but expensive inside the iteration; prefer from_mass_segregation /
        from_components for exploratory anisotropic work. aniso_method ("table"
        default, "quadrature" oracle) is threaded through the eigenvalue
        iteration, the final solve, AND the mass-CDF grid (a construction
        choice, not model state; ignored when r_a is None).
        """
        m_j, M_j = _bin_imf(imf, n_comp, m_range)
        ra_hat = None if r_a is None else r_a / r_c
        alpha_j, residual = find_alpha_for_masses(
            m_j, M_j, W0, g, delta, n_iter=n_iter, xi_max=xi_max,
            n_points=n_ode_points, ra_hat=ra_hat, eta=eta,
            aniso_method=aniso_method)
        mu_j = m_j / jnp.sum(m_j * alpha_j)
        w_j = mu_j ** (-delta)
        ra_hat_j = None if r_a is None else ra_hat * mu_j ** eta
        tab, dens_fn = _shared_table_and_dens_fn(alpha_j, w_j ** -2.0, ra_hat_j,
                                                 W0, g, xi_max, aniso_method)
        xi, psi, rho_j = solve_multicomponent_limepy(
            alpha_j, w_j ** -2.0, W0, g, xi_max=xi_max, n_points=n_ode_points,
            ra_hat_j=ra_hat_j, aniso_method=aniso_method, aniso_table=tab)
        return cls(alpha_j, w_j, m_j, W0, g, r_c, xi, psi, ra_hat_j=ra_hat_j,
                   residual=residual, n_grid=n_grid, rho_on_xi=rho_j,
                   dens_fn=dens_fn)

    def component_virial_ratios(self, n: int = 4000) -> Float[Array, "n_comp"]:
        """Theoretical per-component virial ratio Q_j = T_j/|W_j| from the model.

        The bias-free equilibrium proof (no sampling, no softening, no finite-N):
        for a component in steady state in the shared potential, 2 T_j + W_j = 0,
        so Q_j = 0.5 exactly for every component of a self-consistent model. The
        sampled per-group Q_j is a finite-N estimator of this. Dimensionless
        (independent of G, M, r_c).

            T_j = int 0.5 rho_j <v^2>_j 4 pi r^2 dr,
            <v^2>_j = s_j^2 * 3 Eg(g+5/2, W_j)/Eg(g+3/2, W_j)   (isotropic)
                      s_j^2 * <u^2>(W_j, p_j)                    (anisotropic)
            W_j = - int rho_j r (dphi/dr) 4 pi r^2 dr,  dphi/dr = G M_enc(r)/r^2,

        with W_j(r) = rescale_j psi(r) = psi(r)/w_j^2 and s_j^2 = s^2 w_j^2,
        s^2 = 1/(9 mu_tot) in G = M = r_c = 1 units.
        """
        r = jnp.linspace(1e-3, self.r_t, n)
        psi = jnp.interp(r / self.r_c, self.xi_grid, self.psi_grid,
                         left=self.W0, right=0.0)
        rescale = self.rescale_j
        # per-component mass density rho_j = alpha_j rho_hat_j (norm cancels in Q_j)
        rho_j = _grid_density_components(psi, r / self.r_c, rescale, self.W0,
                                         self.g, self.ra_hat_j, self.is_aniso)
        rho_j = jnp.where(r[None, :] <= self.r_t, rho_j, 0.0) * self.alpha_j[:, None]
        rho_tot = jnp.sum(rho_j, axis=0)

        integ = 4.0 * jnp.pi * r**2 * rho_tot
        dr = r[1] - r[0]
        M_enc = jnp.concatenate([jnp.zeros(1),
                                 jnp.cumsum(0.5 * (integ[1:] + integ[:-1])) * dr])
        # Normalize to total mass M=1 so the velocity scale s^2 = 1/(9 mu_tot) is
        # consistent with the potential (G = M = r_c = 1). W_j carries two powers
        # of this normalization (rho_j and M_enc), T_j one (rho_j).
        norm = 1.0 / M_enc[-1]
        rho_j = rho_j * norm
        dphi_dr = (M_enc * norm) / jnp.maximum(r, 1e-6) ** 2  # G=1

        s2 = 1.0 / (9.0 * self.mu_tot)  # G=M=r_c=1
        Qs = []
        for j in range(self.m_j.shape[0]):
            W_j = rescale[j] * psi
            s_j2 = s2 * self.w_j[j] ** 2
            if self.is_aniso:
                p_j = (r / self.r_c) / self.ra_hat_j[j]
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
        psi_r = jnp.interp(r1 / self.r_c, self.xi_grid, self.psi_grid,
                           left=self.W0, right=0.0)
        rho_j = _grid_density_components(psi_r, r1 / self.r_c, self.rescale_j,
                                         self.W0, self.g, self.ra_hat_j,
                                         self.is_aniso)
        tot = jnp.sum(self.alpha_j[:, None] * rho_j, axis=0).reshape(jnp.shape(r))
        return jnp.where(r <= self.r_t, tot, 0.0)

    def sample_cluster(self, key: PRNGKeyArray, n_stars: int, G=None) -> ICResult:
        """Sample an equilibrium multi-component IC -> ICResult (with component_id).

        Each star is assigned a component by a categorical draw (probabilities
        N_frac_j), its position from that component's mass CDF, and its velocity
        from the component's lowered DF at the rescaled potential
        W_j(r) = psi(r)/w_j^2 and velocity scale s_j = s w_j, with
        s^2 = G M / (9 r_c mu_tot):

          * isotropic model: speed ~ u^2 E_gamma(g, W_j - u^2/2), isotropic direction;
          * anisotropic model (finite ra_hat_j): speed-angle (u_r, u_t) from the
            Michie/OM LIMEPY DF at per-star anisotropy p_i = (r/r_c)/ra_hat_j[c];
            v_r along r_hat, v_t in a random azimuth perpendicular to r_hat.

        The total cluster mass M = sum_i m_i is the sum of the sampled stellar
        masses, so kinetic and potential energies use a consistent M and the
        cluster is virial (Q=0.5) -- globally AND per component -- for ANY mass
        spectrum (the scalar virial theorem is anisotropy-blind). Differentiable
        in (alpha_j, w_j, ra_hat_j, W0, g) through the per-star scales and the
        angular draw.

        The numerical core is JIT-compiled (compiled once per (n_stars, model
        structure); repeated draws -- seed averaging, MC studies -- reuse the
        compiled kernel). ICResult is assembled outside the JIT boundary.
        """
        if G is None:
            G = defaults.DEFAULT_UNITS.G
        pos, vel, m_i, radii_stellar, c = _sample_cluster_arrays(self, key, n_stars, G)
        return ICResult(positions=pos, velocities=vel, masses=m_i,
                        stellar_radii=radii_stellar, component_id=c)


@eqx.filter_jit
def _sample_cluster_arrays(model: MultiComponentCluster, key: PRNGKeyArray,
                           n_stars: int, G: float):
    """JIT-compiled sampler core -> (pos, vel, m_i, stellar_radii, component_id).

    `model` enters as a PyTree (is_aniso is a static field, so the iso/aniso
    branch is resolved at trace time); n_stars and G are static arguments
    (one compilation per distinct value). Key splits match the original eager
    path; the isotropic speed VALUES come from the SpeedCDFTable inverse (one
    batched E_gamma build per call, distributionally identical to the exact
    per-star sampler -- statistical oracles in test_limepy_tables.py).
    """
    from progenax.kinematics.limepy_df import _sample_costheta_given_u

    k_assign, k_pos, k_pdir, k_speed, k_vdir = jax.random.split(key, 5)

    c = jax.random.categorical(k_assign, jnp.log(model.N_frac_j + 1e-30),
                               shape=(n_stars,))
    m_i = model.m_j[c]
    rescale_i = model.rescale_j[c]
    M_total = jnp.sum(m_i)  # the cluster mass IS the sum of its stars
    s = jnp.sqrt(G * M_total / (9.0 * model.r_c * model.mu_tot))
    s_i = s * model.w_j[c]

    # Positions: per-star inverse-CDF on its component's mass CDF + isotropic dirs.
    u = jax.random.uniform(k_pos, (n_stars,))
    radii = jax.vmap(lambda uu, cc: jnp.interp(uu, model._cdf_j[cc], model._r_grid))(u, c)
    pos = radii[:, None] * _isotropic_dirs(k_pdir, n_stars)

    # Per-star rescaled potential W_j(r) = psi(r)/w_j^2 (shared by both paths).
    W_i = rescale_i * jnp.maximum(
        jnp.interp(radii / model.r_c, model.xi_grid, model.psi_grid,
                   left=model.W0, right=0.0), 0.0)
    speed_keys = jax.random.split(k_speed, n_stars)

    if model.is_aniso:
        # Anisotropic: the speed u comes from ONE precomputed 3-D speed-MARGINAL
        # CDF table (Task 6) replacing the per-star 256-point quadrature of
        # _sample_speed_angle; the angular conditional cos(theta)|u stays EXACT
        # (_sample_costheta_given_u -- the same code _sample_speed_angle calls).
        # The box covers every star exactly: W_i = rescale_i*psi <=
        # max(rescale)*W0 and p_i = (r/r_c)/ra_hat_j[c] <= (r_t/r_c)/min(ra_hat)
        # (radii never exceed r_t by construction of the mass-CDF draw); the
        # 1e-3 p floor guards the degenerate all-isotropic corner like
        # _solver_table. Built per call, differentiable in (W0, g, w_j, ra_hat_j).
        p_i = (radii / model.r_c) / model.ra_hat_j[c]
        p_box = jnp.maximum((model.r_t / model.r_c) / jnp.min(model.ra_hat_j), 1e-3)
        table = AnisoSpeedCDFTable.build(jnp.max(model.rescale_j) * model.W0,
                                         p_box, model.g)
        ku_kc = jax.vmap(jax.random.split)(speed_keys)
        unif = jax.vmap(lambda kk: jax.random.uniform(kk))(ku_kc[:, 0])
        u_sp = jax.vmap(table.inverse)(W_i, p_i, unif)
        cos_t = jax.vmap(
            lambda kk, uu, pp: _sample_costheta_given_u(kk, uu, pp, _N_C)
        )(ku_kc[:, 1], u_sp, p_i)
        u_r = u_sp * cos_t
        u_t = u_sp * jnp.sqrt(jnp.maximum(1.0 - cos_t**2, 0.0))
        v_r, v_t = s_i * u_r, s_i * u_t
        # v_r along r_hat (signed); v_t in a random azimuth perpendicular to r_hat.
        r_hat = pos / (radii[:, None] + 1e-30)
        rand = jax.random.normal(k_vdir, (n_stars, 3))
        rand = rand - jnp.sum(rand * r_hat, axis=1, keepdims=True) * r_hat
        t_hat = rand / (jnp.linalg.norm(rand, axis=1, keepdims=True) + 1e-30)
        vel = v_r[:, None] * r_hat + v_t[:, None] * t_hat
    else:
        # Isotropic: ONE precomputed speed-CDF table (Task 5) replaces the
        # per-star 256-point E_gamma quadrature: a star's draw is a sqrt(W)-row
        # lookup + inverse-CDF interp. The box W_max = max(rescale)*W0 covers
        # every star exactly (W_i = rescale_i * psi <= max(rescale) * W0); the
        # build is traced per call (65k E_gamma points, amortized over n_stars)
        # and stays differentiable in (W0, g, w_j) -- the model is unchanged.
        table = SpeedCDFTable.build(jnp.max(model.rescale_j) * model.W0, model.g)
        unif = jax.vmap(lambda kk: jax.random.uniform(kk))(speed_keys)
        u_speed = jax.vmap(table.inverse)(W_i, unif)
        vel = (s_i * u_speed)[:, None] * _isotropic_dirs(k_vdir, n_stars)

    return pos, vel, m_i, compute_stellar_radii(m_i), c


__all__ = ["MultiComponentCluster"]
