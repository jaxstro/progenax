"""W0-OED concentration demo core (scripts-only; consumer of project_dispersion).

Headlines W0 = concentration: "where to spend a fixed star budget (radial bins x
{RV, PM_R, PM_T}) to best constrain a cluster's concentration W0", on the Stage-1
additive-Fisher backbone (scripts/_demo_oed.py), for King (headline OM-King) and
Michie. See docs/plans/2026-06-18-oed-concentration-w0-{plan,design}.md.

Scripts-only: no src/progenax/ surface, informax-bound, held OUT of v0.1.0 (the OED
hold-out decision). This module is a CONSUMER of progenax.project_dispersion (the
Osipkov-Merritt Jeans + Binney&Mamon82/Strigari+07 projection forward model) and of
progenax.KingProfile / MichieProfile.

The OM particle sampler (Task 1) -- the load-bearing piece -- draws each star from
the EXACT model project_dispersion projects, so the calibration's sampler IS the
Fisher model:

  * project_dispersion is Osipkov-Merritt ONLY: it imposes beta(r)=r^2/(r^2+r_a^2)
    on the PROFILE'S DENSITY (not the profile's own intrinsic anisotropy). So the
    matching sampler is "that density under OM", NOT MichieVelocityDF's native beta.
  * King: drawn via Engine B `MultiComponentCluster.from_density_profiles` -- a
    single-component King density in its own shared-Psi potential, Eddington/OM DF
    (maximal reuse; Engine B's realizability gate guarantees f >= 0).
  * Michie: Engine B does NOT ingest MichieProfile, so the OM-on-Michie-density DF
    is assembled here directly from the generic eddington_invert + sample_speed +
    assign_om_directions primitives (the EFFVelocityDF pattern), with the potential
    built from MichieProfile.density EXACTLY as project_dispersion builds it (the
    Poisson integral of the profile density), and a CLEAN analytic d rho/dr from
    Michie's own Poisson identity (NOT a finite-difference of interpolated psi --
    that staircase poisons the Eddington d^2 rho/dPsi^2; cf. density_poisson.py).

Velocity units: pc/Myr (STELLAR), matching project_dispersion's sqrt(G M / length).
"""
import os
import sys

import jax
import jax.numpy as jnp

import progenax  # noqa: F401  enables float64 at import
from progenax import KingProfile, MichieProfile, MultiComponentCluster
from progenax.profiles.michie import michie_density
from progenax.numerics import cumulative_trapz
from progenax.kinematics.eddington import (
    assign_om_directions,
    eddington_invert,
    sample_speed_from_f_table,
)
from jaxstro.units import STELLAR

# Reuse the Stage-1 sky projection (line of sight = +z).
sys.path.insert(0, os.path.dirname(__file__))
from _demo_oed import project_to_sky  # noqa: E402

# Resolution of the hand-rolled Michie Eddington table (mirrors eff_df defaults).
_N_R = 6000      # radial grid for the potential / density tabulation
_N_E = 1000      # energy grid for f(E)
_N_SPEED = 256   # per-particle speed inverse-CDF resolution


def build_profile(W0, r_a, model):
    """Spatial profile for (W0, r_a) in the chosen model (r_c == 1 length unit).

    King is isotropic in density (anisotropy is layered on by OM in the velocity
    DF / project_dispersion); Michie's r_a sets BOTH the intrinsic profile shape
    (its native anisotropy) AND the OM anisotropy radius used downstream. Both are
    the SAME r_a -- one source of truth (design decision 2).
    """
    if model == "king":
        return KingProfile.from_W0_rc(W0, 1.0)
    if model == "michie":
        return MichieProfile.from_W0_rc(W0, 1.0, r_a)
    raise ValueError(f"model must be 'king' or 'michie', got {model!r}")


# ---------------------------------------------------------------------------
# King OM sampler: Engine B single-component (maximal reuse).
# ---------------------------------------------------------------------------


def _sample_king_om(W0, r_a, M, n_stars, key):
    """King density under OM via Engine B (from_density_profiles), total mass M.

    A SINGLE King component (mass_fraction 1) builds its own shared-Psi potential
    and an OM Eddington DF with anisotropy radius r_a. Per-star stellar mass label
    m_j = M / n_stars makes the sampled total mass sum_i m_i == M exactly (the
    Engine B speed scale sqrt(G M_sampled / (4 pi mu)) is then at total mass M,
    matching project_dispersion's M). Engine B's realizability gate raises if the
    OM DF is genuinely negative.
    """
    king = build_profile(W0, r_a, "king")
    model = MultiComponentCluster.from_density_profiles(
        [king], jnp.array([1.0]), m_j=jnp.array([M / n_stars]),
        r_a_j=jnp.array([r_a]),
    )
    ic = model.sample_cluster(key, n_stars=n_stars, G=STELLAR.G)
    return ic.positions, ic.velocities


# ---------------------------------------------------------------------------
# Michie OM sampler: hand-rolled eddington_invert on the Michie density.
# ---------------------------------------------------------------------------


def _michie_dimensionless_table(prof, r_a, n_r=_N_R, n_e=_N_E):
    """Dimensionless (G=1, rho_0=1) Michie potential + OM Eddington f(E).

    Builds (r, Psi, E_grid, f_grid, mu) for the MICHIE DENSITY under Osipkov-Merritt
    anisotropy radius r_a, mirroring eff_df._eff_eddington_table but with:

      * rho(r) = prof.density(r)  -- the self-consistent Michie volume density,
        already normalized to rho_0 = 1 at the centre (the profile convention);
      * Psi(r) the Poisson integral of that density (the SAME potential construction
        project_dispersion uses: it integrates profile.density for M_enc), so the
        sampler's binding potential matches the Fisher forward model;
      * a CLEAN analytic d rho/dr = (d rho/dW) * (dW/dr): d rho/dW from jax.grad of
        the closed-form michie_density at the (smooth) interpolated potential, and
        dW/dr from Michie's own Poisson identity dpsi/dxi = -9 xi^-2 int_0^xi
        rho_tilde(s) s^2 ds (one cumulative trapezoid of the closed-form density --
        NOT a finite difference of the interpolated psi, whose staircase would ring
        through the Eddington d^2 rho/dPsi^2; cf. density_poisson._king_density_and_dW).

    The OM augmentation (rho_Q = (1 + r^2/r_a^2) rho) and the singular Eddington
    integral live in the generic eddington_invert.
    """
    r_t = prof.r_t
    r = jnp.linspace(1e-5, r_t, n_r)
    dr = r[1] - r[0]
    xi = r / prof.r_c

    # rho(r): the self-consistent Michie volume density (rho_0 = 1 at centre).
    rho = prof.density(r)

    # Smooth interpolated potential W = psi(r) (clamped >= 0 by the profile grid).
    psi = jnp.interp(xi, prof.xi_grid, prof.psi_grid, left=prof.W0, right=0.0)
    s_arg = r / prof.r_a  # dimensionless radius r/r_a entering the Michie density

    # d rho_dimless/dW from the CLOSED-FORM michie_density (smooth in W; no
    # interp differentiation). rho_norm = michie_density(psi, s)/michie_density(W0, 0):
    # the central normalization michie_density(W0, 0) is a constant in W, so
    # d(rho_norm)/dW = (d michie_density/dW)(psi, s) / michie_density(W0, 0).
    rho0 = michie_density(prof.W0, 0.0)
    drho_dW_raw = jax.vmap(lambda W, s: jax.grad(michie_density, argnums=0)(W, s))(
        jnp.maximum(psi, 0.0), s_arg
    )
    drho_dW = jnp.where(psi > 0.0, drho_dW_raw / rho0, 0.0)

    # dW/dr from Michie's Poisson identity (King-convention factor of 9), one
    # cumulative trapezoid of the closed-form normalized density. xi -> 0 limit
    # dpsi/dxi -> -3 rho xi (double-where guard); dW/dr = (dpsi/dxi)/r_c.
    integ = rho * xi**2
    cum = cumulative_trapz(integ, dx=(xi[1] - xi[0]))
    small = xi <= 1e-4
    xi_safe = jnp.where(small, 1.0, xi)
    dpsi_dxi = jnp.where(small, -3.0 * rho * xi, -9.0 * cum / xi_safe**2)
    dW_dr = dpsi_dxi / prof.r_c
    drho_dr = drho_dW * dW_dr
    inside = r <= r_t
    rho = jnp.where(inside, rho, 0.0)
    drho_dr = jnp.where(inside, drho_dr, 0.0)

    # Shared self-consistent potential from the SAME density (Poisson integral).
    #
    # NORMALIZATION NOTE (load-bearing -- do NOT "harmonize" dW_dr and dPsi_dr).
    # This Poisson Psi and the dimensionless `psi` above are NOT the same object:
    # `psi`/`dW_dr` carry the profile's central normalization rho_hat(W0,0), while
    # this Psi/`dPsi_dr` are the Poisson integral of `rho = density/rho_0` (centre
    # value 1). They differ by the CONSTANT C = michie_density(W0, 0) (~1.4):
    #   Psi_poisson(r) = C * psi_dimless(r).
    # Correctness survives this mismatch ONLY because eddington_invert +
    # sample_speed_from_f_table are exactly equivariant under Psi -> lambda*Psi
    # (eddington.py:118-120) and kappa is tied to the SAME Poisson `mu`, so C cancels
    # in the physical dispersion (verified: sampled sigma_r vs jeans_dispersion 1-2%).
    # A future cleanup that forces dW_dr and dPsi_dr to be numerically equal would
    # silently reintroduce a real bias (the 5% gate could still pass). Keep them
    # built from their own normalizations.
    inner = cumulative_trapz(rho * r**2, dx=dr)   # int_0^r rho s^2 ds
    tail = cumulative_trapz(rho * r, dx=dr)       # int_0^r rho s ds
    outer = tail[-1] - tail                        # int_r^{r_t} rho s ds
    Phi = -4.0 * jnp.pi * (inner / r + outer)
    Psi = Phi[-1] - Phi                            # Psi(r_t) = 0, increases inward
    mu = inner[-1]
    Mr = 4.0 * jnp.pi * inner
    dPsi_dr = -Mr / r**2

    E_grid, f_grid = eddington_invert(r, rho, drho_dr, Psi, dPsi_dr, r_a, n_e=n_e)
    return r, Psi, E_grid, f_grid, mu


def michie_om_table_diagnostics(W0, r_a):
    """(f_min, f_max) of the hand-rolled OM-Michie Eddington table (positivity check).

    Returns the raw min and max|f| so a caller can assert f_min > -tol * f_max
    (Merritt's r_a lower bound / DF realizability). Concrete-input diagnostic.
    """
    prof = build_profile(W0, r_a, "michie")
    _, _, _, f_grid, _ = _michie_dimensionless_table(prof, r_a)
    return float(jnp.min(f_grid)), float(jnp.max(jnp.abs(f_grid)))


def _sample_michie_om(W0, r_a, M, n_stars, key):
    """Michie density under OM via the hand-rolled Eddington table, total mass M.

    Positions from MichieProfile.sample_positions (the Michie density CDF), speeds
    from the OM Eddington table at each star's binding potential, OM stretched
    directions at anisotropy radius r_a. The physical velocity scale is
    sqrt(kappa), kappa = G M / (4 pi mu) (the EFF convention; M is the total mass
    project_dispersion uses), so the sampled dispersion is in pc/Myr (STELLAR).
    """
    prof = build_profile(W0, r_a, "michie")
    r_grid, Psi_grid, E_grid, f_grid, mu = _michie_dimensionless_table(prof, r_a)

    k_pos, k_speed, k_dir = jax.random.split(key, 3)
    masses = jnp.ones(n_stars)
    pos = prof.sample_positions(masses, k_pos)
    radii = jnp.linalg.norm(pos, axis=1)

    Psi_r = jnp.interp(radii, r_grid, Psi_grid, left=Psi_grid[0], right=0.0)
    kappa = STELLAR.G * M / (4.0 * jnp.pi * mu)

    speed_keys = jax.random.split(k_speed, n_stars)
    s = jax.vmap(
        lambda kk, pp: sample_speed_from_f_table(kk, pp, E_grid, f_grid, _N_SPEED)
    )(speed_keys, Psi_r)
    speeds = jnp.sqrt(kappa) * s
    vel = assign_om_directions(k_dir, pos, speeds, r_a)
    return pos, vel


def sample_om_cluster(model, W0, r_a, M, n_stars, key):
    """Sample an OM cluster and project to sky -> (R, v_los, v_pm_r, v_pm_t).

    The sampler IS the Fisher model (project_dispersion's OM-on-that-density), so
    its binned dispersions match project_dispersion (the Task-1 calibration anchor).
    Returns per-star projected quantities in pc/Myr (STELLAR), line of sight = +z.

    model: "king" (Engine B from_density_profiles) or "michie" (hand-rolled OM
    Eddington on the Michie density; Engine B does not ingest MichieProfile).
    """
    if model == "king":
        pos, vel = _sample_king_om(W0, r_a, M, n_stars, key)
    elif model == "michie":
        pos, vel = _sample_michie_om(W0, r_a, M, n_stars, key)
    else:
        raise ValueError(f"model must be 'king' or 'michie', got {model!r}")
    return project_to_sky(pos, vel)
