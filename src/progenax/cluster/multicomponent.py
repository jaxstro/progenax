"""
Unified multi-component cluster equilibrium model (Engines A and B).

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
w_j directly (`from_components`). The continuum model satisfies component and
global virial balance. Finite samples fluctuate around that balance; sampling
does not force their virial ratios to 0.5.

This module hosts BOTH engines: the Engine-A (DF-defined, lowered-isothermal)
machinery lives here, while Engine-B (density-defined Eddington/Osipkov-Merritt for
Plummer/EFF/King, entered via `from_density_profiles`) builds its state in
`cluster/eddington_engine.py`.

References:
    Gieles, M. & Zocchi, A. (2015), MNRAS, 454, 576 (Eqs. 24-29, Section 4.1).
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxstro.quad import cumulative_trapezoid
from jaxtyping import Array, Float, PRNGKeyArray

from progenax.builders import ICResult
from progenax.cluster._radial_cdf import potential_interp, radial_grid
from progenax.cluster.eddington_engine import (
    _EngineBState,
    assemble_engine_b_fields,
    engine_b_component_virials,
)
from progenax.cluster.sampling import (
    _estimate_expectation,
    _estimate_mass_squared_and_kinetic,
    _estimate_population_moments,
    _sample_cluster_arrays,
)
from progenax.profiles.king import _find_tidal_radius, _require_tidal_crossing
from progenax.profiles.limepy import lowered_exponential
from progenax.profiles.limepy_multimass import (
    _aniso_density_fn,
    _aniso_v2hat_scalar,
    _bin_imf,
    _grid_density_components,
    _solver_table,
    find_alpha_for_masses,
    solve_multicomponent_limepy,
)


def _validate_component_shapes(**components):
    """Require one nonempty vector shape for every component quantity.

    Shapes are static under JIT, so invalid inputs fail before any solve in
    eager, compiled and differentiated constructors.
    """
    shapes = {name: jnp.asarray(value).shape for name, value in components.items()}
    if (
        any(len(shape) != 1 or shape[0] == 0 for shape in shapes.values())
        or len(set(shapes.values())) != 1
    ):
        raise ValueError(
            "Engine A per-component inputs must share one shape: nonempty "
            "one-dimensional vectors; got "
            + ", ".join(f"{name}={shape}" for name, shape in shapes.items())
            + "."
        )


def _shared_table_and_dens_fn(alpha_j, rescale, ra_hat_j, W0, g, xi_max, aniso_method):
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
            f"aniso_method must be 'table' or 'quadrature', got {aniso_method!r}"
        )
    ra_arr = jnp.asarray(ra_hat_j, dtype=jnp.float64)
    table = _solver_table(rescale, ra_arr, W0, g, xi_max)
    dens_fn, _ = _aniso_density_fn(
        jnp.asarray(alpha_j, dtype=jnp.float64),
        rescale,
        ra_arr,
        W0,
        g,
        xi_max,
        "table",
        table=table,
    )
    return table, dens_fn


class _EngineAState(eqx.Module):
    """Engine A (lowered-isothermal) leaves, grouped (one field on the model).

    The DF-defined machinery's state: structural parameters (W0, g, r_c), the
    per-component scales (alpha_j, w_j, ra_hat_j), the velocity-scale integral
    mu_tot, the eigenvalue-solve residual, and the coupled-Poisson solution
    (xi_grid, psi_grid). Engine B models carry `engine_a = None` -- A-only
    access there raises an informative AttributeError via the model's
    delegating properties (this REPLACED the NaN-sentinel tripwires, which
    poisoned results silently on accidental A-path use).
    """

    W0: Float[Array, ""]
    g: Float[Array, ""]
    r_c: Float[Array, ""]
    mu_tot: Float[Array, ""]
    alpha_j: Float[Array, "n_comp"]
    w_j: Float[Array, "n_comp"]
    ra_hat_j: Float[Array, "n_comp"]
    xi_grid: Float[Array, "n_ode"]
    psi_grid: Float[Array, "n_ode"]
    psi_raw: Float[Array, "n_ode"]
    residual: Float[Array, ""]
    radial_pdf_j: Float[Array, "n_comp n_grid"]


def _engine_a_property(name: str, doc: str) -> property:
    """A delegating property for an Engine-A-only quantity `name`.

    Reads through `self.engine_a`; on an Engine B model (engine_a is None) it
    raises an AttributeError NAMING the engine -- loud and immediate, never a
    silent NaN. Property access happens at Python/trace time (`engine` is a
    static field), so raising here is jit-safe: a traced Engine A build never
    takes the None branch.
    """

    def getter(self):
        if self.engine_a is None:
            raise AttributeError(
                f"{name} is an Engine A (lowered-isothermal) quantity; this "
                f"model was built by from_density_profiles (Engine B), which "
                f"has no {name}. Use the profile parameters / the Engine B "
                f"state on self.engine_b instead."
            )
        return getattr(self.engine_a, name)

    getter.__name__ = name
    getter.__doc__ = doc
    return property(getter)


class MultiComponentCluster(eqx.Module):
    """Multi-component cluster in shared-potential equilibrium (two engines).

    Engine A (DF-defined, lowered-isothermal/LIMEPY family): construct with
    `from_components` (direct per-component velocity-scale ratios w_j -- the
    general case), `from_mass_segregation` (the equipartition law
    w_j = mu_j^(-delta)), or `from_imf` (bin an IMF, eigenvalue-solve for
    alpha_j).

    Engine B (density-defined, Eddington inversion): construct with
    `from_density_profiles` (Plummer/EFF/King density components in ONE
    shared self-consistent potential, per-component DFs via Eddington
    inversion, optional per-component Osipkov-Merritt `r_a_j`).

    Either way, `sample_cluster` returns an
    :class:`~progenax.builders.ICResult` whose `component_id` labels each
    star's generating component.

    AD through ``sample_cluster`` follows continuous parameters with component
    labels fixed; it omits changes in categorical assignment probabilities.
    ``estimate_expectation(observable, key, n_stars, G, n_realizations=...)``
    adds that contribution for scalar population observables using a first-order
    score-plus-pathwise estimator. The observable must be differentiable within
    fixed assignments, and at least two independent realizations are required.
    Statistical precision is observable-dependent; full population-gradient
    qualification and higher derivatives remain open.

    Engine A constructors accept ``adjoint=None`` for checkpointed reverse AD,
    ``diffrax.ForwardMode()`` for forward AD, or ``diffrax.DirectAdjoint()``
    for both. Keep this choice static under JIT. ``from_imf`` differentiates
    its central-fraction constraint implicitly in either mode and requires
    ``residual <= fit_tol`` at construction. This value-convergence check does
    not certify derivative precision; small FD steps can require tighter fits.

    For Engine A, ``n_grid`` is the minimum number of positive-width radial
    intervals. The cumulative distribution grid also retains every saved
    potential boundary and 16 subdivisions per potential interval. Array
    lengths can exceed ``n_grid + 1``; repeated tidal-boundary nodes have zero
    width and mass. Increasing ``n_grid`` refines this grid without removing
    the potential boundaries.

    Attributes:
        r_t: tidal/truncation radius (shared by both engines).
        m_j: representative stellar mass per component [M_sun] (labels only).
        N_frac_j: number fraction of stars per component.
        engine: "A" or "B" (static; resolves dispatch at trace time).
        engine_a: grouped Engine A state (`_EngineAState`; None on B models).
        engine_b: grouped Engine B state (`_EngineBState`; None on A models).

    Engine-A-only quantities (W0, g, r_c, mu_tot, alpha_j, w_j, ra_hat_j,
    xi_grid, psi_grid, residual) are exposed as delegating properties reading
    through `engine_a`:
        W0, g, r_c: structural parameters / scales.
        alpha_j: central density fractions (sum to 1).
        w_j: per-component velocity-scale ratios s_j/s (rescale_j = w_j^-2).
        ra_hat_j: per-component anisotropy radii r_{a,j}/r_c (inf = isotropic).
        mu_tot: total dimensionless mass integral (sets the velocity scale).
        residual: eigenvalue-solve residual (0 for direct constructors).
        xi_grid, psi_grid: shared coupled-Poisson solution W(xi).
    On an Engine B model each raises an informative AttributeError naming the
    engine (this replaced the NaN-sentinel tripwires).
    """

    r_t: Float[Array, ""]
    m_j: Float[Array, "n_comp"]
    N_frac_j: Float[Array, "n_comp"]
    _r_grid: Float[Array, "n_grid"]
    _cdf_j: Float[Array, "n_comp n_grid"]
    is_aniso: bool = eqx.field(static=True)
    engine: str = eqx.field(static=True)
    engine_a: Optional[_EngineAState] = None
    engine_b: Optional[_EngineBState] = None

    W0 = _engine_a_property("W0", "Central dimensionless potential depth (Engine A).")
    g = _engine_a_property(
        "g", "Truncation parameter of the lowered isothermal (Engine A)."
    )
    r_c = _engine_a_property("r_c", "King core radius scale (Engine A).")
    mu_tot = _engine_a_property(
        "mu_tot",
        "Total dimensionless mass integral; sets the velocity scale (Engine A).",
    )
    alpha_j = _engine_a_property(
        "alpha_j", "Per-component central density fractions, sum to 1 (Engine A)."
    )
    w_j = _engine_a_property(
        "w_j", "Per-component velocity-scale ratios s_j/s (Engine A)."
    )
    ra_hat_j = _engine_a_property(
        "ra_hat_j",
        "Per-component anisotropy radii r_{a,j}/r_c, inf = isotropic (Engine A).",
    )
    xi_grid = _engine_a_property(
        "xi_grid", "Coupled-Poisson radial grid xi = r/r_c (Engine A)."
    )
    psi_grid = _engine_a_property(
        "psi_grid", "Coupled-Poisson solution W(xi) on xi_grid (Engine A)."
    )
    residual = _engine_a_property(
        "residual", "Eigenvalue-solve residual; 0 for direct constructors (Engine A)."
    )

    def __init__(
        self,
        alpha_j=None,
        w_j=None,
        m_j=None,
        W0=None,
        g=None,
        r_c=None,
        xi_grid=None,
        psi_grid=None,
        ra_hat_j=None,
        residual=0.0,
        n_grid: int = 1000,
        rho_on_xi=None,
        dens_fn=None,
        psi_raw=None,
        *,
        _fields=None,
    ):
        """Assemble model state from a finished coupled solve; constructors
        forward `rho_on_xi` (the solver's (n_comp, n_ode) density grid, reused
        verbatim for nu_j/mu_tot) and `dens_fn` (the solve's pointwise
        (xi, psi) -> (n_comp,) density source, e.g. the shared-table source,
        used for the r-grid mass CDF) -- either may be None to recompute /
        fall back to exact quadrature.

        `psi_raw` (the solver's UNCLAMPED psi, negative past the zero-crossing)
        is fed to `_find_tidal_radius` so the forward r_t is the interpolated
        crossing and d(r_t)/dW0 flows (the clamped psi_grid zeros the
        crossing-node gradient, as in solve_king_profile). None falls back to
        psi_grid (e.g. legacy callers) -- the forward value is unchanged.

        `_fields` (internal, keyword-only): a complete name -> value dict
        assembled by a non-A constructor (`from_density_profiles`, Engine B);
        when given, fields are set verbatim (keys validated against the
        declared field names) and the Engine-A assembly is skipped."""
        if _fields is not None:
            unknown = sorted(set(_fields) - set(type(self).__dataclass_fields__))
            if unknown:
                raise ValueError(
                    f"_fields contains unknown field name(s) {unknown}; "
                    f"declared fields are "
                    f"{sorted(type(self).__dataclass_fields__)}."
                )
            for name, val in _fields.items():
                object.__setattr__(self, name, val)
            return
        is_aniso = ra_hat_j is not None
        alpha_j = jnp.asarray(alpha_j, dtype=jnp.float64)
        w_j = jnp.asarray(w_j, dtype=jnp.float64)
        m_j = jnp.asarray(m_j, dtype=jnp.float64)
        W0 = jnp.asarray(W0, dtype=jnp.float64)
        g = jnp.asarray(g, dtype=jnp.float64)
        r_c = jnp.asarray(r_c, dtype=jnp.float64)
        ra_arr = (
            jnp.full(w_j.shape, jnp.inf, dtype=jnp.float64)
            if ra_hat_j is None
            else jnp.asarray(ra_hat_j, dtype=jnp.float64)
        )
        _validate_component_shapes(alpha_j=alpha_j, w_j=w_j, m_j=m_j, ra_hat_j=ra_arr)
        xi_grid = jnp.asarray(xi_grid, dtype=jnp.float64)
        psi_grid = jnp.asarray(psi_grid, dtype=jnp.float64)
        rescale = w_j**-2.0

        # Density evaluator for the r-grid mass CDF. Constructors forward the
        # solve's density source (`dens_fn`, pointwise (xi, psi) -> (n_comp,);
        # the shared-table source when aniso_method="table") so the CDF is
        # built with the SAME method as the coupled Poisson solve. dens_fn=None
        # (direct construction, isotropic, or aniso_method="quadrature") falls
        # back to the exact quadrature. `component_virial_ratios` stays on the
        # quadrature path unconditionally -- it is the equilibrium oracle.
        if dens_fn is None:

            def dens(psi_arr, xi_arr):
                return _grid_density_components(
                    psi_arr, xi_arr, rescale, W0, g, ra_arr, is_aniso
                )
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

        # Unclamped values retain tidal-boundary sensitivity. The legacy
        # clamped fallback locates a zero plateau without that information.
        psi_for_rt = psi_grid if psi_raw is None else psi_raw
        r_t = r_c * _find_tidal_radius(xi_grid, psi_for_rt)
        # Preserve potential interpolation boundaries in the radial integral.
        r_grid = radial_grid(r_c * xi_grid, r_t, n_grid)
        psi_r = potential_interp(r_grid / r_c, xi_grid, psi_for_rt, left=W0, right=0.0)
        rho_j_r = dens(psi_r, r_grid / r_c)
        rho_j_r = jnp.where(r_grid[None, :] <= r_t, rho_j_r, 0.0)

        integrand = 4.0 * jnp.pi * r_grid[None, :] ** 2 * rho_j_r
        # Non-uniform trapezoid: pass the stretched grid as x (a scalar dr would be wrong).
        M_cum = cumulative_trapezoid(integrand, x=r_grid, axis=1)
        cdf_j = M_cum / (M_cum[:, -1:] + 1e-30)

        state_a = _EngineAState(
            W0=W0,
            g=g,
            r_c=r_c,
            mu_tot=mu_tot,
            alpha_j=alpha_j,
            w_j=w_j,
            ra_hat_j=ra_arr,
            xi_grid=xi_grid,
            psi_grid=psi_grid,
            psi_raw=psi_for_rt,
            residual=jnp.asarray(residual, dtype=jnp.float64),
            radial_pdf_j=integrand / (M_cum[:, -1:] + 1e-30),
        )
        # An ordered check survives derivative-only readers of constant model
        # fields, whose zero tangents can discard value-only array guards.
        _require_tidal_crossing(psi_for_rt)
        for name, val in dict(
            r_t=r_t,
            m_j=m_j,
            N_frac_j=N_frac,
            _r_grid=r_grid,
            _cdf_j=cdf_j,
        ).items():
            object.__setattr__(self, name, val)
        object.__setattr__(self, "is_aniso", is_aniso)
        object.__setattr__(self, "engine", "A")
        object.__setattr__(self, "engine_a", state_a)
        object.__setattr__(self, "engine_b", None)

    @property
    def rescale_j(self) -> Float[Array, "n_comp"]:
        """Per-component potential-depth rescaling rescale_j = w_j^-2 (Engine A).

        Engine B models have no w_j -- their per-component DFs are Eddington
        inversions of prescribed densities, not rescaled lowered isothermals --
        so the concept is physically meaningless there: raises ValueError
        (kept distinct from the delegating properties' AttributeError for the
        stored A-only leaves; rescale_j is a DERIVED quantity).
        """
        if self.engine == "B":
            raise ValueError(
                "rescale_j (= w_j^-2) is an Engine A (lowered-isothermal) "
                "quantity; this model is Engine B (from_density_profiles), "
                "which has no per-component velocity-scale ratio w_j. The "
                "Engine B per-component state lives on self.engine_b "
                "(f_j_grid, mass_fractions, r_a_j, ...)."
            )
        return self.w_j**-2.0

    @classmethod
    def from_components(
        cls,
        alpha_j,
        w_j,
        m_j,
        W0,
        g,
        r_c=1.0,
        ra_hat_j=None,
        xi_max: float = 300.0,
        n_ode_points: int = 2000,
        n_grid: int = 1000,
        aniso_method: str = "table",
        *,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        adjoint=None,
    ):
        """Direct constructor: components defined by their velocity-scale ratios w_j.

        The general Engine-A case -- GC 1G/2G, halo+core, binaries-vs-singles: any
        set of populations, equal-mass or not, with per-component concentration set
        by w_j (smaller w_j = colder = more concentrated). ra_hat_j (per-component
        r_{a,j}/r_c) is the optional radial anisotropy; pass a larger xi_max
        (e.g. 800) for anisotropic models. aniso_method ("table" default,
        "quadrature" oracle) selects the anisotropic density source for the solve
        AND the mass-CDF grid -- a construction choice, not model state; ignored
        when ra_hat_j is None. `component_virial_ratios` always uses quadrature.
        ``rtol`` and ``atol`` set the Poisson solve accuracy. Tighten them for
        derivative qualification; they do not change table resolution.
        ``n_grid`` requests a minimum number of positive-width radial intervals.
        The CDF grid also includes 16 subdivisions per saved potential interval;
        its fixed-shape arrays can include repeated tidal-boundary nodes.

        Use adjoint=None for checkpointed reverse AD (default), diffrax.ForwardMode()
        for forward AD, or diffrax.DirectAdjoint() for both. Keep this choice
        static under JIT; it does not change the physical equations or tolerances.
        """
        _validate_component_shapes(
            alpha_j=alpha_j,
            w_j=w_j,
            m_j=m_j,
            **({} if ra_hat_j is None else {"ra_hat_j": ra_hat_j}),
        )
        w_arr = jnp.asarray(w_j, dtype=jnp.float64)
        tab, dens_fn = _shared_table_and_dens_fn(
            alpha_j, w_arr**-2.0, ra_hat_j, W0, g, xi_max, aniso_method
        )
        xi, psi, psi_raw, rho_j = solve_multicomponent_limepy(
            alpha_j,
            w_arr**-2.0,
            W0,
            g,
            xi_max=xi_max,
            n_points=n_ode_points,
            ra_hat_j=ra_hat_j,
            aniso_method=aniso_method,
            aniso_table=tab,
            rtol=rtol,
            atol=atol,
            adjoint=adjoint,
        )
        return cls(
            alpha_j,
            w_arr,
            m_j,
            W0,
            g,
            r_c,
            xi,
            psi,
            ra_hat_j=ra_hat_j,
            residual=0.0,
            n_grid=n_grid,
            rho_on_xi=rho_j,
            dens_fn=dens_fn,
            psi_raw=psi_raw,
        )

    @classmethod
    def from_mass_segregation(
        cls,
        alpha_j,
        m_j,
        W0,
        g,
        delta,
        r_a=None,
        eta=0.0,
        r_c=1.0,
        xi_max: float = 300.0,
        n_ode_points: int = 2000,
        n_grid: int = 1000,
        aniso_method: str = "table",
        *,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        adjoint=None,
    ):
        """Equipartition constructor: w_j = mu_j^(-delta) (Gieles & Zocchi 2015).

        mu_j = m_j / bar_m with bar_m = sum_j m_j alpha_j (central density-weighted
        mean mass, Eq. 26); heavier components are colder and sink. Optional
        anisotropy r_{a,j} = r_a mu_j^eta (eta=0 = mass-independent, the paper
        default). delta=0 is the unsegregated single-mass corner.
        ``rtol`` and ``atol`` set the Poisson solve accuracy. Tighten them for
        derivative qualification; they do not change table resolution.

        Use adjoint=None for checkpointed reverse AD (default), diffrax.ForwardMode()
        for forward AD, or diffrax.DirectAdjoint() for both. Keep this choice
        static under JIT; it does not change the physical equations or tolerances.
        """
        _validate_component_shapes(alpha_j=alpha_j, m_j=m_j)
        alpha_arr = jnp.asarray(alpha_j, dtype=jnp.float64)
        m_arr = jnp.asarray(m_j, dtype=jnp.float64)
        mu_j = m_arr / jnp.sum(m_arr * alpha_arr)
        w_j = mu_j ** (-delta)
        ra_hat_j = None if r_a is None else (r_a / r_c) * mu_j**eta
        return cls.from_components(
            alpha_arr,
            w_j,
            m_arr,
            W0,
            g,
            r_c=r_c,
            ra_hat_j=ra_hat_j,
            xi_max=xi_max,
            n_ode_points=n_ode_points,
            n_grid=n_grid,
            aniso_method=aniso_method,
            rtol=rtol,
            atol=atol,
            adjoint=adjoint,
        )

    @classmethod
    def from_imf(
        cls,
        imf,
        n_comp,
        W0,
        g,
        delta,
        m_range=(0.1, 100.0),
        r_c=1.0,
        r_a=None,
        eta=0.0,
        n_iter: int = 30,
        xi_max: float = 300.0,
        n_ode_points: int = 2000,
        n_grid: int = 1000,
        aniso_method: str = "table",
        *,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        adjoint=None,
        fit_tol: float = 1e-6,
    ):
        """IMF constructor: bin into n_comp log-spaced components, solve for alpha_j.

        The eigenvalue solve (`find_alpha_for_masses`) finds the central density
        fractions that reproduce the IMF's per-bin mass budget under the
        equipartition law w_j = mu_j^(-delta). Anisotropy (r_a, eta) is supported
        but expensive inside the iteration; prefer from_mass_segregation /
        from_components for exploratory anisotropic work. aniso_method ("table"
        default, "quadrature" oracle) is threaded through the eigenvalue
        iteration, the final solve, AND the mass-CDF grid (a construction
        choice, not model state; ignored when r_a is None).

        The central-fraction fit uses an implicit JVP supporting forward and reverse
        AD at convergence. Construction raises if the final mass-fraction residual
        is nonfinite or exceeds fit_tol, including under JIT and AD. The residual
        is the largest absolute component mass-fraction error (dimensionless).
        The keyword-only fit_tol must be finite and nonnegative; n_iter caps
        iterations. Failed-fit diagnostics remain available from find_alpha_for_masses.
        Small finite-difference steps can require tighter fits than the default.
        The keyword-only rtol and atol control Poisson integration accuracy in
        the fit, its implicit derivative, residual check and final solve. Keep
        them static under JIT. They do not change table or mass-CDF resolution
        and do not replace fit_tol, which controls the mass-fraction residual.
        The adjoint keyword selects the ODE mode: None for checkpointed reverse (default),
        diffrax.ForwardMode() for forward, or diffrax.DirectAdjoint() for both.
        """
        fit_tol_array = jnp.asarray(fit_tol)
        fit_tol_array = eqx.error_if(
            fit_tol_array,
            ~jnp.isfinite(fit_tol_array) | (fit_tol_array < 0.0),
            "fit_tol must be finite and nonnegative.",
            on_error="raise",
        )
        m_j, M_j = _bin_imf(imf, n_comp, m_range)
        ra_hat = None if r_a is None else r_a / r_c
        alpha_j, residual = find_alpha_for_masses(
            m_j,
            M_j,
            W0,
            g,
            delta,
            n_iter=n_iter,
            tol=fit_tol_array,
            xi_max=xi_max,
            n_points=n_ode_points,
            ra_hat=ra_hat,
            eta=eta,
            aniso_method=aniso_method,
            rtol=rtol,
            atol=atol,
            adjoint=adjoint,
        )
        mu_j = m_j / jnp.sum(m_j * alpha_j)
        w_j = mu_j ** (-delta)
        ra_hat_j = None if r_a is None else ra_hat * mu_j**eta
        tab, dens_fn = _shared_table_and_dens_fn(
            alpha_j, w_j**-2.0, ra_hat_j, W0, g, xi_max, aniso_method
        )
        xi, psi, psi_raw, rho_j = solve_multicomponent_limepy(
            alpha_j,
            w_j**-2.0,
            W0,
            g,
            xi_max=xi_max,
            n_points=n_ode_points,
            ra_hat_j=ra_hat_j,
            aniso_method=aniso_method,
            rtol=rtol,
            atol=atol,
            adjoint=adjoint,
            aniso_table=tab,
        )
        model = cls(
            alpha_j,
            w_j,
            m_j,
            W0,
            g,
            r_c,
            xi,
            psi,
            ra_hat_j=ra_hat_j,
            residual=residual,
            n_grid=n_grid,
            rho_on_xi=rho_j,
            dens_fn=dens_fn,
            psi_raw=psi_raw,
        )
        # Check the whole result so reading any array field retains the guard
        # under JIT dead-code elimination, including fields independent of alpha_j.
        return eqx.error_if(
            model,
            ~jnp.isfinite(residual) | (residual > fit_tol_array),
            "IMF central-fraction fit did not converge: residual must be finite "
            "and <= fit_tol. Increase n_iter or inspect find_alpha_for_masses "
            "diagnostics before using the model.",
            on_error="raise",
        )

    @classmethod
    def from_density_profiles(
        cls,
        profiles,
        mass_fractions,
        m_j,
        r_a_j=None,
        r_t=None,
        f_enc: float = 0.995,
        n_r: int = 6000,
        n_e: int = 1000,
        n_grid: int = 1000,
    ):
        """Engine B constructor: prescribed-density components in ONE shared Psi.

        Each component j is a prescribed density shape (PlummerProfile,
        EFFProfile, or KingProfile) with amplitude `mass_fractions[j]` = M_j /
        M_total (must sum to 1); the shared self-consistent potential is one
        quadrature pass (no ODE) and each component's DF is its Eddington
        inversion in that shared Psi -- optionally Osipkov-Merritt anisotropic
        via `r_a_j` (per-component anisotropy radii; None/inf = isotropic).
        The domain r_t is DERIVED from the component extents (max finite
        extent, else the f_enc summed-mass radius; explicit `r_t=` override
        available -- see `derive_r_t`); provenance + realizability diagnostics
        live on `self.engine_b`. A genuinely unrealizable mix (negative DF)
        raises naming the component, including under JIT and AD. Each raw DF
        must be finite with min(f_j)/max(abs(f_j)) > -1e-3. Concrete
        low-level checks raise ValueError; runtime checks raise a JAX/Equinox
        error. Raw traced diagnostics remain available from build_engine_b_state.

        `m_j` are decoupled stellar-mass labels exactly as in Engine A:
        N_frac_j proportional to mass_fractions_j / m_j. Engine-A-only
        quantities (W0, g, r_c, mu_tot, alpha_j, w_j, ra_hat_j, xi_grid,
        psi_grid, residual) do not exist here (`engine_a` is None) -- accessing
        them raises an informative AttributeError naming the engine.
        """
        return cls(
            _fields=assemble_engine_b_fields(
                profiles,
                mass_fractions,
                m_j,
                r_a_j=r_a_j,
                r_t=r_t,
                f_enc=f_enc,
                n_r=n_r,
                n_e=n_e,
                n_grid=n_grid,
            )
        )

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

        Engine B models dispatch to `engine_b_component_virials`: the same
        Clausius statement with <v^2>_j from the per-component Eddington DF
        speed moments on the full Poisson grid (`n` is unused there -- the
        stored grid resolution is the quadrature).
        """
        if self.engine == "B":
            # engine == "B" guarantees engine_b was populated at construction.
            assert self.engine_b is not None
            return engine_b_component_virials(self.engine_b)
        r = jnp.linspace(1e-3, self.r_t, n)
        psi = jnp.interp(
            r / self.r_c, self.xi_grid, self.psi_grid, left=self.W0, right=0.0
        )
        rescale = self.rescale_j
        # per-component mass density rho_j = alpha_j rho_hat_j (norm cancels in Q_j)
        rho_j = _grid_density_components(
            psi, r / self.r_c, rescale, self.W0, self.g, self.ra_hat_j, self.is_aniso
        )
        rho_j = jnp.where(r[None, :] <= self.r_t, rho_j, 0.0) * self.alpha_j[:, None]
        rho_tot = jnp.sum(rho_j, axis=0)

        integ = 4.0 * jnp.pi * r**2 * rho_tot
        dr = r[1] - r[0]
        M_enc = cumulative_trapezoid(integ, dx=dr)
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
                v2hat = jax.vmap(lambda w, pp: _aniso_v2hat_scalar(w, pp, self.g))(
                    W_j, p_j
                )
            else:
                v2hat = (
                    3.0
                    * lowered_exponential(self.g + 2.5, W_j)
                    / jnp.maximum(lowered_exponential(self.g + 1.5, W_j), 1e-300)
                )
            v2 = s_j2 * jnp.where(W_j > 0.0, v2hat, 0.0)
            T = jnp.trapezoid(0.5 * rho_j[j] * v2 * 4.0 * jnp.pi * r**2, r)
            W = jnp.trapezoid(-rho_j[j] * r * dphi_dr * 4.0 * jnp.pi * r**2, r)
            Qs.append(T / jnp.abs(W))
        return jnp.stack(Qs)

    def total_density(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """Total dimensionless volume density at r (engine-dispatched), 0 outside r_t.

        Engine A: sum_j alpha_j rho_hat_j(r) -- the model's dimensionless
        density in the LIMEPY convention (each rho_hat_j normalized to its
        central value, alpha_j the central density fractions), so the total is
        1 at the center. Engine B (previously a silent NaN: the A formula on
        the NaN tripwire fields): the PRESCRIBED total density interpolated
        from the stored Poisson-grid rows sum_j engine_b.rho_j_poisson, each
        component mass-normalized to its mass fraction (G = 1, total
        dimensionless mass 1: 4 pi int_0^{r_t} rho r^2 dr = 1). The two
        engines' amplitudes are therefore DIFFERENT dimensionless conventions
        (A: unit central density; B: unit total mass) -- shapes are comparable,
        amplitudes are not.
        """
        if self.engine == "B":
            # engine == "B" guarantees engine_b was populated at construction.
            assert self.engine_b is not None
            rho_tot = jnp.sum(self.engine_b.rho_j_poisson, axis=0)
            r1 = jnp.atleast_1d(jnp.asarray(r))
            tot = jnp.interp(
                r1, self.engine_b.r_poisson, rho_tot, left=rho_tot[0], right=0.0
            ).reshape(jnp.shape(r))
            return jnp.where(r <= self.r_t, tot, 0.0)
        r1 = jnp.atleast_1d(jnp.asarray(r))
        psi_r = jnp.interp(
            r1 / self.r_c, self.xi_grid, self.psi_grid, left=self.W0, right=0.0
        )
        rho_j = _grid_density_components(
            psi_r,
            r1 / self.r_c,
            self.rescale_j,
            self.W0,
            self.g,
            self.ra_hat_j,
            self.is_aniso,
        )
        tot = jnp.sum(self.alpha_j[:, None] * rho_j, axis=0).reshape(jnp.shape(r))
        return jnp.where(r <= self.r_t, tot, 0.0)

    def sample_cluster(self, key: PRNGKeyArray, n_stars: int, G: float) -> ICResult:
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

        The total cluster mass M = sum_i m_i sets the velocity normalization
        consistently with the sampled stellar masses. The underlying equilibrium
        distribution satisfies virial balance; a finite realization has sampling
        fluctuations. Neither its global nor its component virial ratios are
        forced to Q=0.5, where Q is kinetic energy divided by the magnitude of
        the corresponding gravitational virial term. Differentiable
        in (alpha_j, w_j, ra_hat_j, W0, g) through the per-star scales and the
        angular draw.

        Engine A positions invert the integrated piecewise-linear radial mass
        density. Engine B models (`from_density_profiles`) linearly invert their
        tabulated mass CDF, then draw speeds from the per-component shared-Psi
        Eddington tables with velocity scale sqrt(G M_sampled / (4 pi mu)) and
        directions via the OM stretched split at per-star r_a_j[c]
        (`engine_b_star_velocities`).

        The numerical core is JIT-compiled (compiled once per (n_stars, model
        structure); repeated draws -- seed averaging, MC studies -- reuse the
        compiled kernel). ICResult is assembled outside the JIT boundary.

        G is REQUIRED (explicit-units policy; e.g. ``STELLAR.G``).
        """
        pos, vel, m_i, radii_stellar, c = _sample_cluster_arrays(self, key, n_stars, G)
        return ICResult(
            positions=pos,
            velocities=vel,
            masses=m_i,
            stellar_radii=radii_stellar,
            component_id=c,
        )

    def estimate_expectation(
        self,
        observable,
        key: PRNGKeyArray,
        n_stars: int,
        G: float,
        *,
        n_realizations: int,
    ):
        """Estimate a scalar population observable with first-order AD.

        ``observable(ICResult)`` must return a scalar and have valid pathwise
        derivatives within each fixed component assignment. The returned value
        is its mean over ``n_realizations`` independent clusters of ``n_stars``
        stars. ``G`` is explicit, as in :meth:`sample_cluster`.

        Forward or reverse AD combines the continuous sampling derivative with
        the categorical probability score, including the effect of all labels
        on the sampled total mass and velocity normalization. A leave-one-out
        baseline reduces score variance without changing its expectation.
        Unlike differentiating :meth:`sample_cluster`, this includes changes in
        component assignment probabilities. Random keys and sample counts are
        held fixed during differentiation; estimates still have sampling error.

        Requires at least two realizations. Higher derivatives are unsupported.
        Discontinuous observables (for example hard spatial cuts) need their own
        boundary treatment and are not made differentiable by this estimator.
        Constructor AD mode and fit-convergence requirements still apply.
        """
        if n_realizations < 2:
            raise ValueError("n_realizations must be at least 2")
        if n_stars < 1:
            raise ValueError("n_stars must be positive")
        return _estimate_expectation(self, observable, key, n_stars, G, n_realizations)

    def estimate_population_moments(
        self,
        key: PRNGKeyArray,
        n_stars: int,
        G: float,
        *,
        n_realizations: int,
        draws_per_component: int,
    ):
        """Estimate three population moments for Engine A or Engine B.

        Return a dictionary with ``mean_mass``, ``mean_radius_squared`` and
        ``mean_speed_squared``: expected catalogue means of mass, squared radius
        about the model centre, and squared speed in the model rest frame.
        Their units are mass, length squared, and velocity squared in the model's
        unit system; supply a consistent gravitational constant ``G`` explicitly.
        There is no recentering, selection cut, or extra virial rescaling.

        Mass and radius moments are integrated analytically over the discrete
        components and the sampler's piecewise-linear radial density. Velocity
        for Engine A uses ``draws_per_component`` conditional draws for every component in each
        of ``n_realizations`` independent integration realizations. These are
        integration samples, not cluster catalogues. ``n_stars`` independently
        sets the target population size. The velocity expectation includes the
        exact correlation between a star's component and sampled total mass.
        Engine B instead integrates the clipped interpolated DF speed moments
        at its energy knots, averages OM directions analytically, and applies
        four-point quadrature within each radial-CDF interval. Its result is
        deterministic: ``key``, ``n_realizations`` and ``draws_per_component``
        do not change it. This integrates the continuous interpolated DF;
        catalogue inverse-CDF speed discretization has a separate convergence
        requirement.

        First derivatives include mixture-weight and conditional contributions;
        constructor AD-mode and convergence requirements still apply. Higher
        derivatives are unqualified. Statistical velocity precision depends on
        the model and exposure. Current numerical qualification is limited to
        seven tested isotropic configurations and one four-component anisotropic
        configuration, not the full Engine A domain. The anisotropic check uses
        W0=5, g=1, delta=0.4, r_a/r_c=10, eta=0 and a Maschberger IMF
        high-mass slope of 2.3 over 0.1–20 solar masses, with fit_tol=1e-10
        and n_iter=60.
        Engine B's deterministic moment checks cover the retained Plummer/EFF
        mixture in isotropic and anisotropic configurations (halo anisotropy
        radius 3 pc, isotropic core), including first derivatives in the mass
        fraction, both component scales and finite halo anisotropy radius. Broader population-moment coverage remains
        under qualification.
        """
        for name, value in (
            ("n_stars", n_stars),
            ("n_realizations", n_realizations),
            ("draws_per_component", draws_per_component),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive static integer")
        return _estimate_population_moments(
            self, key, n_stars, G, n_realizations, draws_per_component
        )

    def estimate_mass_squared_and_kinetic(
        self,
        key: PRNGKeyArray,
        n_stars: int,
        G: float,
        *,
        n_realizations: int,
        draws_per_component: int,
    ):
        """Estimate squared total mass and twice kinetic energy per star.

        Return ``total_mass_squared`` = E[(sum_i m_i)^2] and
        ``twice_kinetic_energy_per_star`` = E[sum_i m_i |v_i|^2 / n_stars].
        Here m_i and v_i are the catalogue masses and model-rest-frame
        velocities before recentering or virial rescaling. Units are mass
        squared and mass times velocity squared; supply consistent ``G``.
        The kinetic observable is 2T/n_stars, not divided by catalogue mass.

        For Engine A, sum over component labels analytically. Mass squared is
        deterministic. Kinetic energy uses ``draws_per_component`` conditional
        samples per component in each of ``n_realizations`` independent
        integration realizations. These draws do not form a physical catalogue;
        ``n_stars`` sets the separate target population. Exact finite-population
        weights retain component/total-mass correlation in velocity scaling.

        First-order forward/reverse qualification covers the isotropic
        four-component Maschberger model over 0.1-20 solar masses with alpha=2.3,
        delta=0.4, W0=5, g=1, fit_tol=1e-10, n_iter=60 and a 100000-star target.
        The qualified exposure uses eight batches of four realizations and
        25000 draws/component. Broader Engine A domains, higher derivatives and
        Engine B use of this method are unqualified. Existing constructor
        AD-mode and fit-convergence requirements apply. The general scalar
        ``estimate_expectation`` API retains its observable-dependent variance.
        """
        if self.engine != "A":
            raise ValueError("estimate_mass_squared_and_kinetic supports Engine A only")
        for name, value in (
            ("n_stars", n_stars),
            ("n_realizations", n_realizations),
            ("draws_per_component", draws_per_component),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive static integer")
        return _estimate_mass_squared_and_kinetic(
            self, key, n_stars, G, n_realizations, draws_per_component
        )


__all__ = ["MultiComponentCluster"]
