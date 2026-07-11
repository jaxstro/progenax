# progenax/src/progenax/profiles/density_poisson.py
"""Prescribed-density shared potential for multi-component equilibria (Engine B).

The Engine B total density is *prescribed*: rho_tot(r) = sum_j mass_fraction_j *
rho_hat_j(r) / m_hat_j (each component's truncated mass normalized to 1, then scaled
by its mass fraction), so the shared self-consistent potential needs NO ODE -- one
cumulative-trapezoid pass gives M(<r), Phi, Psi = Phi(r_t) - Phi, and dPsi/dr =
-M(<r)/r^2 on a fixed grid (the same inner/tail/outer pattern as the validated EFF
Eddington table in kinematics/eff_df.py).

Domain (design decision 2, "derived"): a component's extent is part of the
prescribed model. Plummer is infinite (extent None); EFF and King end at their own
r_t. The cluster edge r_t is the max over finite component extents; if ALL
components are infinite, the radius enclosing ``f_enc`` of the summed analytic
mass. An explicit ``r_t=`` override wins, but a King component whose natural r_t
exceeds the override raises (no silent re-truncation of a lowered-Maxwellian edge).

Components whose extent is smaller than the cluster r_t contribute ZERO density
beyond their own extent (where-masked, differentiable).

All quantities are dimensionless (G = 1, total truncated mass = 1).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from progenax.numerics import cumulative_trapz
from progenax.profiles.eff import EFFProfile
from progenax.profiles.king import KingProfile, king_lowered_maxwellian_density
from progenax.profiles.plummer import PlummerProfile

_SUPPORTED = "PlummerProfile, EFFProfile, KingProfile"


def component_extent(profile) -> Float[Array, ""] | None:
    """Natural radial extent of a prescribed-density component.

    Plummer -> None (infinite); EFF/King -> their own truncation radius r_t.
    Raises TypeError for profiles outside the Engine B v1 set.
    """
    if isinstance(profile, PlummerProfile):
        return None
    if isinstance(profile, (EFFProfile, KingProfile)):
        return profile.r_t
    raise TypeError(
        f"Engine B does not support {type(profile).__name__}; supported density "
        f"components are: {_SUPPORTED}."
    )


def _analytic_enclosed_fraction(profile, r: Float[Array, ""]) -> Float[Array, ""]:
    """Analytic M(<r)/M(inf) for an infinite-extent component (Plummer only)."""
    if isinstance(profile, PlummerProfile):
        x = r / profile.a
        return x**3 / (1.0 + x**2) ** 1.5
    raise TypeError(  # pragma: no cover - guarded by component_extent dispatch
        f"No analytic enclosed-mass fraction for {type(profile).__name__}."
    )


def derive_r_t(profiles, mass_fractions, r_t=None, f_enc: float = 0.995):
    """Resolve the cluster truncation radius r_t (design decision 2, derived).

    Returns ``(r_t, provenance)`` where provenance is a static string naming the
    rule that set the domain: explicit "override", the profile class whose finite
    extent wins, or the ``f_enc`` summed-mass radius for an all-infinite mix.

    An explicit override wins, BUT a King component whose natural r_t exceeds the
    override raises ValueError: a King model's lowered-Maxwellian edge is part of
    the prescribed physics and must not be silently re-truncated. The
    override/conflict branch requires a CONCRETE r_t -- domain resolution is a
    construction-time decision, not a traced quantity. The finite-extent branch
    is likewise concrete-only: the max over component extents picks its winner
    via float() (concretizes), so grad w.r.t. a component's r_t is unsupported
    BY DESIGN -- the domain is decided at construction time, not differentiated
    through.
    """
    extents = [component_extent(p) for p in profiles]

    if r_t is not None:
        rt_arr = jnp.asarray(r_t)
        for j, (p, ext) in enumerate(zip(profiles, extents)):
            if not isinstance(p, KingProfile):
                continue
            # component_extent returns r_t (never None) for a KingProfile.
            assert ext is not None
            if float(ext) > float(rt_arr) * (1.0 + 1e-12):
                raise ValueError(
                    f"r_t override {float(rt_arr):.6g} would re-truncate the King "
                    f"component {j} (natural r_t = {float(ext):.6g}). A King model's "
                    f"edge is part of the prescribed physics -- raise the override "
                    f"to >= the King r_t or drop it."
                )
        # Provenance carries the RULE only, never the float value: the string
        # is a STATIC PyTree field, so embedding the number would give models
        # differing only in r_t distinct treedefs (per-value recompiles,
        # vmap/stack breakage). The value itself is dynamic data (the r_t field).
        return rt_arr, "override (explicit r_t)"

    finite = [
        (j, p, ext)
        for j, (p, ext) in enumerate(zip(profiles, extents))
        if ext is not None
    ]
    if finite:
        rt_arr = jnp.max(jnp.stack([jnp.asarray(ext) for _, _, ext in finite]))
        j_max, p_max, _ = max(finite, key=lambda t: float(t[2]))
        prov = f"{type(p_max).__name__} component {j_max} extent"
        return rt_arr, prov

    # All components infinite (pure Plummer mix): radius enclosing f_enc of the
    # summed analytic mass, via a fixed 80-step bisection (jax.lax.scan -- a
    # fixed-iteration, differentiable-enough construction-time scalar).
    mass_fractions = jnp.asarray(mass_fractions)
    hi0 = 1e4 * jnp.max(jnp.stack([p.a for p in profiles]))

    def summed_enclosed(r):
        return sum(
            mass_fractions[j] * _analytic_enclosed_fraction(p, r)
            for j, p in enumerate(profiles)
        ) / jnp.sum(mass_fractions)

    def step(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        below = summed_enclosed(mid) < f_enc
        return (jnp.where(below, mid, lo), jnp.where(below, hi, mid)), None

    (lo, hi), _ = jax.lax.scan(step, (jnp.zeros(()), hi0), None, length=80)
    rt_arr = 0.5 * (lo + hi)
    # Rule-only provenance (static field; f_enc is a constructor argument, not
    # re-embedded here -- see the override branch's treedef rationale).
    return rt_arr, "f_enc summed-mass radius (all components infinite)"


def _king_drho_dW(W: Float[Array, "..."]) -> Float[Array, "..."]:
    """d rho_hat/dW of the King lowered-Maxwellian volume density (closed form).

    rho_hat(W) = e^W erf(sqrt(W)) - (2/sqrt(pi)) sqrt(W) (1 + 2W/3); the erf' and
    boundary 1/sqrt(W) terms cancel exactly, leaving
        d rho_hat/dW = e^W erf(sqrt(W)) - (2/sqrt(pi)) sqrt(W).
    Gradient-safe at W <= 0 (clamped before sqrt, masked to 0).
    """
    W_pos = jnp.where(W > 0.0, W, 1.0)
    sqrt_W = jnp.sqrt(W_pos)
    val = (
        jnp.exp(W_pos) * jax.scipy.special.erf(sqrt_W)
        - (2.0 / jnp.sqrt(jnp.pi)) * sqrt_W
    )
    return jnp.where(W > 0.0, val, 0.0)


def _king_density_and_dW(profile, r: Float[Array, "n_r"]):
    """King (rho_hat, d rho_hat/dr) via King's own Poisson identity (extent-masked).

    Both outputs come from the solved W-grid WITHOUT differentiating interpolated
    data:

    * **rho** is the closed-form King density integrand evaluated at the
      interpolated potential, rho_hat(psi)/rho_hat(W0) with
      rho_hat(W) = e^W erf(sqrt(W)) - (2/sqrt(pi)) sqrt(W) (1 + 2W/3).
    * **dW/dr** comes from King's Poisson identity (king.py ODE convention,
      (1/xi^2) d/dxi(xi^2 dpsi/dxi) = -9 rho_hat(psi)/rho_hat(W0)), i.e. the
      cumulative mass integral
      dpsi/dxi = -9 xi^-2 int_0^xi (rho_hat(psi(s))/rho_hat(W0)) s^2 ds --
      one cumulative trapezoid of the already-computed CLOSED-FORM density.
    * **d rho_hat/dr** is then the chain rule (d rho_hat/dW) * dW/dr with the
      closed-form ``_king_drho_dW``.

    Why differentiating the interpolated psi is FORBIDDEN: jnp.gradient of the
    piecewise-LINEAR interpolated psi is a staircase whose ringing the Eddington
    d^2 rho/dPsi^2 + Abel 1/sqrt(E - Psi) weight focuses into f(E -> Psi0). The
    historical single-King read had min f/max|f| = -0.68 that way, while the
    true King ergodic DF is strictly positive (fixed in commit dccedbe).

    Grid contract: ``r`` must be a dense, ASCENDING grid anchored at ~0; the
    cumulative integral uses the non-uniform trapezoid (x=xi), so it is correct
    on both the historical uniform linspace and the sqrt-stretched Poisson grid
    (audit S2) that both call sites (shared_potential, build_engine_b_state)
    now pass. Density still needs the grid DENSE near the core for accuracy.
    """
    xi = r / profile.r_c
    psi = jnp.interp(xi, profile.xi_grid, profile.psi_grid, left=profile.W0, right=0.0)
    rho0 = king_lowered_maxwellian_density(profile.W0)
    rho = king_lowered_maxwellian_density(psi) / rho0
    # dW/dr from King's own Poisson identity (king.py ODE convention,
    # (1/xi^2) d/dxi(xi^2 dpsi/dxi) = -9 rho_hat(psi)/rho_hat(W0)):
    #     dpsi/dxi = -9 xi^-2 int_0^xi (rho_hat(psi(s))/rho_hat(W0)) s^2 ds,
    # one cumulative trapezoid of the already-computed CLOSED-FORM density
    # -- no differentiation of interpolated data (see function docstring).
    # xi -> 0 limit: dpsi/dxi -> -3 rho_tilde xi -> 0 (double-where guard,
    # gradient-safe: no 0/0 enters the graph). xi = r/r_c, so dW/dr =
    # (dpsi/dxi)/r_c.
    integ = rho * xi**2
    # Non-uniform trapezoid (x=xi): the Poisson grid is sqrt-stretched (audit
    # S2), so a scalar dxi would be silently wrong here.
    cum = cumulative_trapz(integ, x=xi)
    small = xi <= 1e-4
    xi_safe = jnp.where(small, 1.0, xi)
    dpsi_dxi = jnp.where(small, -3.0 * rho * xi, -9.0 * cum / xi_safe**2)
    dW_dr = dpsi_dxi / profile.r_c
    drho = (_king_drho_dW(psi) / rho0) * dW_dr
    inside = r <= profile.r_t
    return jnp.where(inside, rho, 0.0), jnp.where(inside, drho, 0.0)


def _density_and_derivative(profile, r: Float[Array, "n_r"]):
    """Unnormalized rho_hat_j(r) and d rho_hat_j/dr on the grid (extent-masked).

    Plummer/EFF use ANALYTIC closed-form derivatives; King delegates to
    ``_king_density_and_dW`` (rho + dW/dr from King's own Poisson identity --
    deliberately NOT differentiating interpolated data; see that helper's
    docstring for the staircase-bug rationale and the uniform-grid contract).
    """
    if isinstance(profile, PlummerProfile):
        a = profile.a
        base = 1.0 + (r / a) ** 2
        return base**-2.5, -5.0 * (r / a**2) * base**-3.5
    if isinstance(profile, EFFProfile):
        a, gamma = profile.a, profile.gamma
        base = 1.0 + (r / a) ** 2
        rho = base ** (-gamma / 2.0)
        drho = -gamma * (r / a**2) * base ** (-gamma / 2.0 - 1.0)
        inside = r <= profile.r_t
        return jnp.where(inside, rho, 0.0), jnp.where(inside, drho, 0.0)
    if isinstance(profile, KingProfile):
        return _king_density_and_dW(profile, r)
    raise TypeError(
        f"Engine B does not support {type(profile).__name__}; supported density "
        f"components are: {_SUPPORTED}."
    )


def _untruncated_mass(profile, m_trunc):
    """Analytic total mass M_j(inf) of the unnormalized rho_hat_j (diagnostic).

    Plummer: 4 pi a^3 / 3 (closed form). EFF gamma > 3: 4 pi a^3 *
    (sqrt(pi)/4) Gamma((gamma-3)/2)/Gamma(gamma/2); gamma <= 3 has DIVERGENT mass
    -> returns inf so trunc_frac_j stores 0.0 (documented on the field). King ends
    naturally at its own r_t, so its "infinite" mass IS its truncated mass.
    """
    if isinstance(profile, PlummerProfile):
        return 4.0 * jnp.pi * profile.a**3 / 3.0
    if isinstance(profile, EFFProfile):
        gamma = profile.gamma
        g_safe = jnp.where(gamma > 3.0, gamma, 4.0)
        integral = (jnp.sqrt(jnp.pi) / 4.0) * jnp.exp(
            jax.scipy.special.gammaln((g_safe - 3.0) / 2.0)
            - jax.scipy.special.gammaln(g_safe / 2.0)
        )
        return jnp.where(gamma > 3.0, 4.0 * jnp.pi * profile.a**3 * integral, jnp.inf)
    # King: the lowered-Maxwellian density ends smoothly at its own r_t; the model
    # has no untruncated counterpart, so M(inf) = M(<r_t) = the truncated mass.
    return m_trunc


class SharedPotential(eqx.Module):
    """Shared self-consistent potential of a prescribed multi-component density.

    All quantities dimensionless (G = 1; total truncated mass = sum_j
    mass_fractions_j = 1).

    Attributes:
        r_grid: radial grid, sqrt-stretched 1e-5 + (r_t - 1e-5) u^2 with
            u = linspace(0, 1, n_r) (audit S2): nodes concentrate in the core;
            the 1e-5 floor guards the 1/r in Phi and dPsi/dr.
        Psi_grid: relative potential Psi = Phi(r_t) - Phi, Psi(r_t) = 0.
        dPsi_dr_grid: -M(<r)/r^2 (analytic from the enclosed mass).
        rho_j_grid: (n_comp, n_r) per-component densities, each normalized so its
            TRUNCATED mass is its mass fraction; zero beyond the component extent.
        M_cum_j: (n_comp, n_r) per-component enclosed mass M_j(<r);
            M_cum_j[j, -1] = mass_fractions[j].
        mu: sum_j int_0^{r_t} rho_j r^2 dr (velocity-scale integral; the EFF
            kappa = G M_total / (4 pi mu) pattern).
        trunc_frac_j: M_j(<r_t)/M_j(inf) per component, with ANALYTIC M_j(inf)
            (Plummer closed form; EFF gamma > 3 closed form). DIAGNOSTIC ONLY:
            EFF gamma <= 3 has logarithmically divergent untruncated mass, so
            0.0 is stored there (EFF always carries a finite r_t in practice, so
            this never feeds a domain decision). King = 1.0 by construction
            (its natural edge is the model edge and derive_r_t guarantees
            r_t >= King r_t).
        r_t: cluster truncation radius (the grid edge).
        r_t_provenance: static string naming what set r_t (see derive_r_t).
    """

    r_grid: Float[Array, "n_r"]
    Psi_grid: Float[Array, "n_r"]
    dPsi_dr_grid: Float[Array, "n_r"]
    rho_j_grid: Float[Array, "n_comp n_r"]
    M_cum_j: Float[Array, "n_comp n_r"]
    mu: Float[Array, ""]
    trunc_frac_j: Float[Array, "n_comp"]
    r_t: Float[Array, ""]
    r_t_provenance: str = eqx.field(static=True)


def shared_potential(
    profiles,
    mass_fractions,
    r_t,
    n_r: int = 6000,
    r_t_provenance: str = "explicit",
) -> SharedPotential:
    """One cumulative-trapezoid pass: prescribed densities -> shared Psi.

    Normalizes each component's truncated mass to its mass fraction, sums, and
    integrates Poisson directly (no ODE): M(<r), Phi = -4 pi (inner/r + outer),
    Psi = Phi(r_t) - Phi, dPsi/dr = -M(<r)/r^2. The inner/tail/outer pattern is
    copied from the validated _eff_eddington_table.

    Raises ValueError if |sum(mass_fractions) - 1| > 1e-8 (concrete inputs; a
    wrong sum is a user bug, not a convention).
    """
    mass_fractions = jnp.asarray(mass_fractions)
    total = jnp.sum(mass_fractions)
    try:
        total_f = float(total)
    except (
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerArrayConversionError,
        TypeError,
    ):
        total_f = None  # traced build: the sum cannot be checked at trace time
    if total_f is not None and abs(total_f - 1.0) > 1e-8:
        raise ValueError(
            f"mass_fractions must sum to 1 (got {total_f:.10g}); they are "
            f"M_j/M_total amplitudes of the prescribed component densities."
        )

    r_t = jnp.asarray(r_t)
    # Sqrt-stretched grid r = floor + (r_t - floor) u^2 (audit S2, third sibling
    # after LIMEPYProfile and Engine-A): a linear grid under-resolves the core
    # when r_t >> r_c (King W0=12: r_t ~ 548 r_c -> +2.5% on M(<0.5 r_c) at the
    # default n_r). The 1e-5 floor is kept — it guards the 1/r in Phi (inner/r)
    # and dPsi/dr = -M/r^2 below. Smooth in r_t, so the build stays
    # differentiable. All integrals on this grid MUST use the non-uniform
    # trapezoid (x=r), never a scalar dx.
    r = 1e-5 + (r_t - 1e-5) * jnp.linspace(0.0, 1.0, n_r) ** 2

    rho_rows, M_rows, trunc_rows = [], [], []
    for j, p in enumerate(profiles):
        rho_hat, _ = _density_and_derivative(p, r)
        m_hat = (
            4.0 * jnp.pi * cumulative_trapz(rho_hat * r**2, x=r)[-1]
        )  # truncated mass of rho_hat
        rho_j = mass_fractions[j] * rho_hat / m_hat
        rho_rows.append(rho_j)
        M_rows.append(4.0 * jnp.pi * cumulative_trapz(rho_j * r**2, x=r))
        trunc_rows.append(m_hat / _untruncated_mass(p, m_hat))

    rho_j_grid = jnp.stack(rho_rows)
    M_cum_j = jnp.stack(M_rows)
    trunc_frac_j = jnp.stack(trunc_rows)
    rho_tot = jnp.sum(rho_j_grid, axis=0)

    inner = cumulative_trapz(rho_tot * r**2, x=r)  # int_0^r rho s^2 ds
    tail = cumulative_trapz(rho_tot * r, x=r)  # int_0^r rho s ds
    outer = tail[-1] - tail  # int_r^{r_t} rho s ds
    Phi = -4.0 * jnp.pi * (inner / r + outer)
    Psi = Phi[-1] - Phi  # Psi(r_t) = 0, increases inward
    M_cum = 4.0 * jnp.pi * inner
    dPsi_dr = -M_cum / r**2  # r starts at 1e-5; M_cum[0] = 0 -> 0
    mu = inner[-1]  # sum_j int rho_j r^2 dr

    return SharedPotential(
        r_grid=r,
        Psi_grid=Psi,
        dPsi_dr_grid=dPsi_dr,
        rho_j_grid=rho_j_grid,
        M_cum_j=M_cum_j,
        mu=mu,
        trunc_frac_j=trunc_frac_j,
        r_t=r_t,
        r_t_provenance=r_t_provenance,
    )


__all__ = [
    "SharedPotential",
    "component_extent",
    "derive_r_t",
    "shared_potential",
]
