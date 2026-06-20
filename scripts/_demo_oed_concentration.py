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

import functools
import os
import pathlib
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp

# opt#3 -- persistent XLA compilation cache. The expensive thing the W0-OED demo
# compiles is the King/Michie ODE-jacrev GN-fit scan (and the batched calibration
# pipeline); these are stable across runs (figure regen, the informax port, repeat
# opt-in calibration), so caching the compiled executables to disk lets a SECOND run
# skip the multi-second XLA compile entirely. Repo-local + gitignored (.jax_cache/),
# set BEFORE any JAX array is created (the env may override via JAX_COMPILATION_CACHE_DIR).
_CACHE_DIR = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    str(pathlib.Path(__file__).resolve().parents[1] / ".jax_cache"),
)
jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
# Cache even fast-to-compile executables (default min is high) so the demo's modest
# kernels are persisted; this is a scripts-only demo cache, not a CI hot path.
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

from jaxstro.units import STELLAR

import progenax  # noqa: F401  enables float64 at import
from progenax import (
    KingProfile,
    MichieProfile,
    MultiComponentCluster,
    project_dispersion,
)
from progenax.kinematics.eddington import (
    assign_om_directions,
    eddington_invert,
    sample_speed_from_f_table,
)
from progenax.numerics import cumulative_trapz
from progenax.profiles.michie import michie_density

# Reuse the Stage-1 sky projection (line of sight = +z) + unit conversions, and the
# MODEL-AGNOSTIC Fisher/criteria backbone (consume per-star blocks Mb + a design
# vector; nothing below differentiates the forward model). Re-exported so callers use
# ONE namespace (oedc.c_criterion, oedc.fisher, ...). The Stage-1 optimize_design /
# _optimize_one are NOT reused: they hardcode Stage-1's PRIOR_DIAG (a module global),
# so this module reimplements them threading OUR W0-OED PRIOR_DIAG (see below).
#
# NOTE on the imported c/D/A docstrings: they were written for Stage-1's parameter
# vector theta=(r_a, M, r_h) and so name `r_h`. The functions are pure 3x3 linear
# algebra on the Fisher F and are parameter-agnostic; here they map index-for-index
# onto THIS arc's theta=(W0, r_a, M) (so `help(oedc.c_criterion)`'s `r_h` reads as our
# index-2 nuisance M). The arc's own index map is fixed by theta_truth()/MOCK below.
sys.path.insert(0, os.path.dirname(__file__))
import optax  # noqa: E402
from _demo_oed import (  # noqa: E402
    _MIN_CELL,
    DesignResult,
    blocks_from_eps,
    design_counts,
    fisher,
    kms_to_pcMyr,
    pm_masyr_to_kms,
    project_to_sky,
)
from _demo_oed import (
    c_criterion as c_criterion,  # re-export: callers use oedc.c_criterion (see header note)
)

# Resolution of the hand-rolled Michie Eddington table (mirrors eff_df defaults).
_N_R = 6000  # radial grid for the potential / density tabulation
_N_E = 1000  # energy grid for f(E)
_N_SPEED = 256  # per-particle speed inverse-CDF resolution

# Levenberg-Marquardt damping for the calibration's MAP fit (see _fit_theta_W0_gn). LAM0
# is the initial damping; the per-draw fit adapts it (down on a cost decrease, up on an
# increase). The lam*I floor bounds the curvature of weakly-constrained directions -- here
# the M-only-prior r_a anisotropy nuisance -- so the step stays bounded instead of
# overshooting/oscillating; well-constrained directions (W0, M) recover near-Newton steps.
_GN_LM_LAM0 = 1e-2
# Fixed LM iterations (jax.lax.scan length). LM settles the weakly-constrained r_a that the
# old fixed step-cap left oscillating (28/48 King draws), so a few more iters than the old
# GN=12 converge all three params; the per-draw fit is cheap (King calibration ~minutes).
_GN_N_ITER = 25


# ---------------------------------------------------------------------------
# Mock truth + observing setup (analog of Stage-1's MOCK). theta = (W0, r_a, M);
# index map W0=0 (TARGET concentration), r_a=1, M=2. r_c == 1 is the length unit.
# ---------------------------------------------------------------------------
MOCK = dict(
    W0=6.0,
    r_a=6.0,
    M=1e5,
    r_c=1.0,
    d_kpc=4.0,
    eps_RV_kms=1.0,
    eps_PM_masyr=0.05,
)

# K=12 log-spaced on-sky bin-centre radii in r_c units. At (W0=6, r_a=6) King r_t
# ~ 18 and Michie r_t ~ 24 (Task 1), both > 12, so every bin is dynamically bound
# (asserted by test_predict_sigma_shape_and_bins_bound).
R_BINS = jnp.logspace(jnp.log10(0.3), jnp.log10(12.0), 12)

# Per-channel per-star measurement error eps_c = (eps_RV, eps_PM, eps_PM) [pc/Myr],
# via the Stage-1 unit conversions. Both PM axes (pm_r, pm_t) share the single
# astrometric error.
_eps_RV = kms_to_pcMyr(MOCK["eps_RV_kms"])
_eps_PM = kms_to_pcMyr(pm_masyr_to_kms(MOCK["eps_PM_masyr"], MOCK["d_kpc"]))
EPS = jnp.array([_eps_RV, _eps_PM, _eps_PM])  # (3,) [pc/Myr]


def theta_truth():
    """Truth parameter vector theta = (W0, r_a, M) -- index 0 = W0 (TARGET)."""
    return jnp.array([MOCK["W0"], MOCK["r_a"], MOCK["M"]])


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


def _build_king_om_sampler(W0, r_a, M, n_stars):
    """Build the EXPENSIVE King Engine B model ONCE (the per-draw-invariant structure).

    A SINGLE King component (mass_fraction 1) builds its own shared-Psi potential
    and an OM Eddington DF with anisotropy radius r_a. Per-star stellar mass label
    m_j = M / n_stars makes the sampled total mass sum_i m_i == M exactly (the
    Engine B speed scale sqrt(G M_sampled / (4 pi mu)) is then at total mass M,
    matching project_dispersion's M). Engine B's realizability gate raises if the
    OM DF is genuinely negative.

    The returned MultiComponentCluster depends only on (W0, r_a, M, n_stars) -- the
    truth the calibration holds fixed -- so it is built ONCE per calibration and
    reused across all draws (the cheap per-draw work is just model.sample_cluster).
    """
    king = build_profile(W0, r_a, "king")
    return MultiComponentCluster.from_density_profiles(
        [king],
        jnp.array([1.0]),
        m_j=jnp.array([M / n_stars]),
        r_a_j=jnp.array([r_a]),
    )


def _draw_king_om(model, n_stars, key):
    """Cheap per-key draw from a pre-built King Engine B model -> (positions, velocities)."""
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
    inner = cumulative_trapz(rho * r**2, dx=dr)  # int_0^r rho s^2 ds
    tail = cumulative_trapz(rho * r, dx=dr)  # int_0^r rho s ds
    outer = tail[-1] - tail  # int_r^{r_t} rho s ds
    Phi = -4.0 * jnp.pi * (inner / r + outer)
    Psi = Phi[-1] - Phi  # Psi(r_t) = 0, increases inward
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


class _MichieOMSampler(NamedTuple):
    """Pre-built per-draw-invariant Michie OM sampler structure (the EXPENSIVE part).

    Holds the profile + the hand-rolled OM Eddington table (r_grid, Psi_grid, E_grid,
    f_grid, mu) + the velocity scale kappa = G M / (4 pi mu). All depend only on the
    fixed truth (W0, r_a, M), so the table is built ONCE and reused across draws.
    """

    prof: object
    r_grid: object
    Psi_grid: object
    E_grid: object
    f_grid: object
    r_a: object
    kappa: object


def _build_michie_om_sampler(W0, r_a, M):
    """Build the EXPENSIVE Michie OM Eddington table ONCE -> a reusable _MichieOMSampler.

    The table (_michie_dimensionless_table) and the velocity scale kappa depend only on
    the fixed truth (W0, r_a, M), so this is hoisted out of the per-draw loop.
    """
    prof = build_profile(W0, r_a, "michie")
    r_grid, Psi_grid, E_grid, f_grid, mu = _michie_dimensionless_table(prof, r_a)
    kappa = STELLAR.G * M / (4.0 * jnp.pi * mu)
    return _MichieOMSampler(prof, r_grid, Psi_grid, E_grid, f_grid, r_a, kappa)


def _draw_michie_om(sampler, n_stars, key):
    """Cheap per-key draw from a pre-built _MichieOMSampler -> (positions, velocities).

    Positions from MichieProfile.sample_positions (the Michie density CDF), speeds
    from the OM Eddington table at each star's binding potential, OM stretched
    directions at anisotropy radius r_a. The physical velocity scale is
    sqrt(kappa), kappa = G M / (4 pi mu) (the EFF convention; M is the total mass
    project_dispersion uses), so the sampled dispersion is in pc/Myr (STELLAR).
    """
    k_pos, k_speed, k_dir = jax.random.split(key, 3)
    masses = jnp.ones(n_stars)
    pos = sampler.prof.sample_positions(masses, k_pos)
    radii = jnp.linalg.norm(pos, axis=1)

    Psi_r = jnp.interp(
        radii, sampler.r_grid, sampler.Psi_grid, left=sampler.Psi_grid[0], right=0.0
    )

    speed_keys = jax.random.split(k_speed, n_stars)
    s = jax.vmap(
        lambda kk, pp: sample_speed_from_f_table(
            kk, pp, sampler.E_grid, sampler.f_grid, _N_SPEED
        )
    )(speed_keys, Psi_r)
    speeds = jnp.sqrt(sampler.kappa) * s
    vel = assign_om_directions(k_dir, pos, speeds, sampler.r_a)
    return pos, vel


def sample_om_cluster(model, W0, r_a, M, n_stars, key):
    """Sample an OM cluster and project to sky -> (R, v_los, v_pm_r, v_pm_t).

    The sampler IS the Fisher model (project_dispersion's OM-on-that-density), so
    its binned dispersions match project_dispersion (the Task-1 calibration anchor).
    Returns per-star projected quantities in pc/Myr (STELLAR), line of sight = +z.

    model: "king" (Engine B from_density_profiles) or "michie" (hand-rolled OM
    Eddington on the Michie density; Engine B does not ingest MichieProfile).

    This convenience wrapper is build+draw in ONE call (used by Task-1's
    test_om_sampler_matches_project_dispersion). The calibration (calibrate_fisher_W0)
    instead builds the per-draw-invariant sampler ONCE and calls the cheap _draw_*_om
    per key -- see _build_om_sampler / _draw_om below.
    """
    sampler = _build_om_sampler(model, W0, r_a, M, n_stars)
    pos, vel = _draw_om(model, sampler, n_stars, key)
    return project_to_sky(pos, vel)


def _build_om_sampler(model, W0, r_a, M, n_stars):
    """Build the EXPENSIVE per-draw-invariant OM sampler structure ONCE (King or Michie).

    King: the Engine B MultiComponentCluster (its shared-Psi potential + OM Eddington DF
    build is the cost). Michie: the hand-rolled OM Eddington table (_MichieOMSampler).
    Both depend only on the fixed truth (W0, r_a, M, n_stars), so calibrate_fisher_W0
    builds this once and reuses it across every draw.
    """
    if model == "king":
        return _build_king_om_sampler(W0, r_a, M, n_stars)
    if model == "michie":
        return _build_michie_om_sampler(W0, r_a, M)
    raise ValueError(f"model must be 'king' or 'michie', got {model!r}")


def _draw_om(model, sampler, n_stars, key):
    """Cheap per-key draw from a pre-built OM sampler -> (positions, velocities)."""
    if model == "king":
        return _draw_king_om(sampler, n_stars, key)
    if model == "michie":
        return _draw_michie_om(sampler, n_stars, key)
    raise ValueError(f"model must be 'king' or 'michie', got {model!r}")


# ===========================================================================
# Task 2: forward observable g(theta) + dimensionless ln-theta Jacobian.
# ===========================================================================
#
# Mirrors the Stage-1 _demo_oed.{predict_sigma, jacobian_and_sigma}, but the
# parameter vector is theta = (W0, r_a, M) with index map W0=0 (TARGET), r_a=1,
# M=2 -- a DIFFERENT order from Stage-1 (r_a, M, r_h). The forward model is the
# Osipkov-Merritt Jeans + Binney&Mamon82 projection of the King / Michie profile
# built from (W0, r_a), so r_a is BOTH theta[1] AND the OM anisotropy argument of
# project_dispersion (the same value, by design decision 2 -- one source of truth).


def predict_sigma(theta, R_bins, G, model):
    """Predicted observable g(theta): (3, K) dispersions, rows = (los, pm_r, pm_t).

    Channels in pc/Myr at the K on-sky bin-centre radii R_bins, via the
    Osipkov-Merritt Jeans + Binney & Mamon (1982) projection of the King
    (model="king") or Michie (model="michie") profile built from (W0, r_a).
    theta = (W0, r_a, M); r_a (theta[1]) is both a theta-component and the OM
    anisotropy radius passed to project_dispersion (same value, by design).
    """
    prof = build_profile(theta[0], theta[1], model)
    pd = project_dispersion(prof, theta[1], R_bins, theta[2], G)
    return jnp.stack([pd.sigma_los, pd.sigma_pm_r, pd.sigma_pm_t])  # (3, K)


def jacobian_and_sigma(theta, R_bins, G, model):
    """Return (J, sigma): J = d sigma_pred / d ln theta (3, K, 3), sigma (3, K). ONE jacrev.

    theta = (W0, r_a, M) spans ~5 orders of magnitude (W0~6, r_a~6, M~1e5), so the
    raw Fisher is ill-conditioned. Differentiating wrt ln theta (J -> J * diag(theta))
    makes the Fisher dimensionless and every (F^-1) entry a FRACTIONAL variance
    (ADR 0011) -- the natural metric for "fractional precision on the concentration
    W0". The ln-theta scaling is the single multiply J * theta[None, None, :], by the
    chain rule d sigma / d ln theta_i = (d sigma / d theta_i) * theta_i.

    Reverse-mode jacrev BY POLICY: project_dispersion's King/Michie equilibrium
    solvers hit custom_vjp ODEs with no jvp rule, so forward-mode (jacfwd/hessian)
    would crash. jacrev is the supported/tested gradient path for both profiles.
    """
    sig = predict_sigma(theta, R_bins, G, model)  # (3, K)
    J = jax.jacrev(predict_sigma, argnums=0)(
        theta, R_bins, G, model
    )  # (3, K, 3) -- d sigma / d theta
    return J * theta[
        None, None, :
    ], sig  # -> d sigma / d ln theta (DIMENSIONLESS, ADR 0011)


# ===========================================================================
# Task 4: per-star Fisher blocks, additive design Fisher prior, c/D/A optimizer.
# ===========================================================================
#
# Almost everything here is REUSED from Stage-1 (imported above): the per-star
# block builder blocks_from_eps (Mb = 2 J J^T / (sigma^2 + eps^2)), the additive
# design Fisher fisher(z, Mb, cb, N_total, prior) = Sum n_eff*Mb (+ prior diag), the
# design allocation design_counts (softmax budget x completeness), and the c/D/A
# criteria -- all model-agnostic once (J, sigma) are built. This module adds only
# what is SPECIFIC to the W0-OED arc: a prior tied to OUR theta=(W0, r_a, M) order, a
# completeness curve tied to OUR length scale, the per_star_blocks wrapper, and a
# faithful copy of the Stage-1 multi-start optimizer that threads OUR prior.


# Prior precision on the NUISANCE M ONLY -- diag in ln theta = ln (W0, r_a, M).
#
# Rationale (W0-OED arc; index map W0=0, r_a=1, M=2):
#   * W0 (index 0) is the TARGET concentration -> ZERO prior (the design must
#     constrain it from the kinematics alone).
#   * r_a (index 1) is the OM anisotropy radius -> ZERO prior: anisotropy is
#     constrained by the kinematic dataset (the RV vs PM split) alone; there is no
#     independent external r_a measurement, so it carries no prior.
#   * M (index 2) is the total mass -> a 30% fractional prior (precision 1/0.3**2):
#     M has an external observational constraint (integrated light x M/L) OUTSIDE
#     the kinematic dataset, so we encode it as a weak Gaussian prior. In the
#     dimensionless (d ln theta) metric (ADR 0011) this is a FRACTIONAL precision.
#
# SPD finding (measured, not assumed; test_blocks_shape_symmetry_and_fisher_spd +
# test_fisher_spd_over_random_designs): with this M-only prior the additive design
# Fisher F = Sum n_eff*Mb + diag(PRIOR_DIAG) is SPD for BOTH King and Michie at the
# uniform design AND across random design vectors z -- i.e. the (W0, r_a) 2-block is
# constrained by the DATA alone (the three velocity channels x 12 radial bins break
# the W0<->r_a degeneracy). So the M-only prior is LOCKED (the most honest choice; no
# conditioning regularizer on W0 or r_a is needed). The escalation path -- a WEAK r_a
# conditioning regularizer PRIOR_DIAG[1] = 1/0.5**2 (NOT an external r_a constraint) --
# is documented but UNUSED, since SPD holds without it.
_FRAC_PRIOR_M = 0.3  # 30% fractional prior on M (the only externally-constrained param)
PRIOR_DIAG = jnp.array(
    [0.0, 0.0, 1.0 / _FRAC_PRIOR_M**2]
)  # [W0, r_a, M] fractional precision


def completeness(R_bins, R_turn=6.0, width=1.5):
    """Smooth faint-end roll-off (logistic in R): ~1 in the core -> <1 outskirts.

    An ILLUSTRATIVE selection function, NOT a real survey curve: a logistic that is
    ~1 where crowding/depth let nearly every star be measured and rolls smoothly
    below 1 in the outskirts. Tied to OUR length scale (r_c == 1; R_BINS run to 12
    r_c): the turnover R_turn = 6.0 r_c sits near the half-extent of R_BINS and the
    scale width = 1.5 r_c rolls it off over the outer few bins. (This is the
    concentration arc's OWN curve -- NOT Stage-1's completeness, whose defaults
    R_turn = 2*r_h, width = 0.5*r_h reference Stage-1's r_h = 3, absent from our
    r_c-scaled mock.)
    """
    return 1.0 / (1.0 + jnp.exp((R_bins - R_turn) / width))


def per_star_blocks(theta, R_bins, eps, G, model):
    """Design-INDEPENDENT per-star Fisher blocks Mb (3, K, 3, 3) + sigma (3, K).

    Thin wrapper: ONE reverse-mode jacrev through the OM-King/Michie -> project_dispersion
    forward model (jacobian_and_sigma) gives the dimensionless ln-theta Jacobian J and
    the predicted dispersions sigma; the Stage-1 blocks_from_eps then forms the rank-1
    3x3 blocks Mb_{c,b} = 2 J_{c,b} J_{c,b}^T / (sigma_{c,b}^2 + eps_c^2) (model-agnostic).
    The full design Fisher is the linear sum F = Sum_{c,b} n_eff,{c,b} Mb_{c,b}, so this
    jacrev is computed ONCE and the optimization is pure 3x3 linear algebra.
    """
    J, sig = jacobian_and_sigma(theta, R_bins, G, model)
    return blocks_from_eps(J, sig, eps), sig


# --- Multi-start optax optimizer (faithful copy of Stage-1, threading OUR prior) ---
#
# The Stage-1 _demo_oed.optimize_design / _optimize_one hardcode the Stage-1 module
# global PRIOR_DIAG ([0, 1/0.3**2, 1/0.3**2] on r_a, M, r_h). This arc needs OUR
# PRIOR_DIAG ([0, 0, 1/0.3**2] on W0, r_a, M), so we reimplement the optimizer with
# the SAME structure (multi-start Adam over the softmax design vector z, jit+scan
# step, keep the lowest-criterion start, return a DesignResult) and swap in our prior.


def _optimize_one(criterion_fn, z0, Mb, cb, N_total, n_steps, lr):
    """One Adam trajectory: returns (z_final, trace) where trace is the per-step
    criterion value. The step is jit-compiled and unrolled via jax.lax.scan."""
    opt = optax.adam(lr)
    state = opt.init(z0)
    loss = lambda z: criterion_fn(fisher(z, Mb, cb, N_total, PRIOR_DIAG))

    @jax.jit
    def step(carry, _):
        z, st = carry
        l, g = jax.value_and_grad(loss)(z)
        upd, st = opt.update(g, st)
        return (optax.apply_updates(z, upd), st), l

    (z, _), trace = jax.lax.scan(step, (z0, state), None, length=n_steps)
    return z, trace


def optimize_design(
    criterion_fn, Mb, cb, N_total, key, n_starts=8, n_steps=500, lr=0.05
):
    """Multi-start Adam over the design vector z; keep the lowest-criterion result.

    Returns a DesignResult (z, trace, criterion) for the best start. Faithful copy of
    the Stage-1 optimizer threading THIS arc's PRIOR_DIAG (M-only).

    SPD invariant (load-bearing -- do NOT silently break it on a refactor): the
    Fisher F = fisher(z, Mb, cb, N_total, PRIOR_DIAG) stays symmetric positive-definite
    throughout the optimization, so c/D/A's inv/slogdet never hit a singular F. Here
    that holds from TWO facts: (1) jax.nn.softmax(z) is strictly positive for every
    finite z, so every n_eff,{c,b} > 0 and the additive F = Sum n_eff*Mb is at least
    PSD (each Mb is rank-1 PSD); and (2) for the W0-OED arc the DATA alone makes the
    (W0, r_a) 2-block PD (measured: test_fisher_spd_over_random_designs) and PRIOR_DIAG
    adds strictly positive precision on M, so the full F is strictly PD without any
    prior on W0 or r_a. A future change that allocates with a hard top-k
    (softmax -> argmax, exact zeros), or a model/scale change that made the (W0, r_a)
    data-block rank-deficient at some design, could reintroduce a singular F here -- in
    which case escalate to the documented weak r_a conditioning regularizer.
    """
    K = cb.shape[0]
    best = None
    for s in range(n_starts):
        z0 = jax.random.normal(jax.random.fold_in(key, s), (3 * K,)) * 0.5
        z, trace = _optimize_one(criterion_fn, z0, Mb, cb, N_total, n_steps, lr)
        crit = float(criterion_fn(fisher(z, Mb, cb, N_total, PRIOR_DIAG)))
        if best is None or crit < best.criterion:
            best = DesignResult(z=z, trace=trace, criterion=crit)
    return best


# ===========================================================================
# Task 5: real-star @slow calibration gate (the headline validation).
# ===========================================================================
#
# The END-TO-END gate on the whole W0-OED demo: it confirms that the additive,
# dimensionless DESIGN Fisher (Tasks 1-4) actually predicts the REALIZED fractional
# scatter of W0_hat across independent mock catalogs sampled and fit forward. If the
# design Fisher were wrong (wrong Jacobian, wrong SE, wrong ln-theta metric, or a
# sampler that did not equal the Fisher forward model), this number would not close.
#
# Mirrors _demo_oed.calibrate_fisher with THREE arc-specific changes:
#   (a) mocks drawn with sample_om_cluster(model, ...) -- the Task-1 OM-King /
#       OM-on-Michie-density sampler that IS project_dispersion's forward model
#       (NOT Stage-1's _draw_mock, which is OM-Plummer);
#   (b) theta = (W0, r_a, M) fit in the ln-theta GAUSS-NEWTON metric started at the
#       truth (a faithful local reimplementation of _demo_oed_depth._fit_theta_gn,
#       wired to OUR predict_sigma(., model) / theta_truth / R_BINS / PRIOR_DIAG --
#       Stage-2's fitter is hardwired to Stage-1's theta=(r_a, M, r_h) forward model
#       and takes no model arg, so it cannot be reused directly). Physical-Adam is
#       NOT used: it pins the large-scale M~1e5 (Stage-2 lesson);
#   (c) collect ln(W0_hat) and compare Var(ln W0_hat) to (inv F_design)_{W0, W0}
#       (already a fractional/ln variance in the ln-theta metric, ADR 0011).
#
# Binning-helper COUPLING TRAP (handled): _demo_oed._r_bin_edges / _binned_sigma_hat
# read Stage-1's MODULE-GLOBAL R_BINS / EPS (different VALUES from ours -- Stage-1 is
# logspace(0.9, 9, 12) about r_h=3, ours is logspace(0.3, 12, 12) in r_c). Importing
# them would silently bin against the WRONG radii. So both are reimplemented LOCALLY
# here against OUR R_BINS / EPS (faithful copies, globals swapped). Only the design-
# and-radius-AGNOSTIC scalar _MIN_CELL is reused by import.


def _r_bin_edges():
    """K+1 geometric-mean bin edges bracketing the K log-spaced R_BINS centres.

    Local reimplementation of _demo_oed._r_bin_edges against THIS arc's R_BINS (the
    coupling trap: Stage-1's helper reads Stage-1's R_BINS, whose values differ).
    R_BINS is log-uniform with constant log step dlog, so the edges are the centres
    shifted by +-dlog/2 in log space. Used only to bin the parent mock catalog.
    """
    lc = jnp.log(R_BINS)
    dlog = lc[1] - lc[0]
    return jnp.exp(jnp.concatenate([lc[:1] - dlog / 2.0, lc + dlog / 2.0]))


def _binned_sigma_hat(key, channels, R, n_eff, edges):
    """One mock's binned dispersions sigma_hat (3, K) + SEs se (3, K) -- HOST reference.

    Local reimplementation of _demo_oed._binned_sigma_hat against THIS arc's R_BINS /
    EPS (the coupling trap). For each radial bin b and channel c: take that channel's
    parent-star velocities in bin b, subsample n_use = max(round(n_eff[c,b]), _MIN_CELL)
    WITHOUT replacement, add per-star Gaussian measurement error ~ Normal(0, EPS[c]) (so
    sigma_hat^2 ~ sigma_true^2 + EPS[c]^2, matching the design Fisher denom), and take the
    ddof=1 sample std. The SE is sigma_hat / sqrt(2 n_eff) -- the Gaussian delta-method SE
    of a 1-D dispersion -- evaluated at the DESIGN count so it matches the Fisher weight.

    This host-side reference is KEPT (still used by sample_om_cluster's lone-draw callers
    and as the correctness oracle for the vectorised _binned_sigma_hat_jax below); the
    calibration loop (calibrate_fisher_W0) uses the vmapped JAX path so the whole
    draw->bin->fit pipeline batches across draws (opt#4). All randomness stays in jax.random.
    """
    import numpy as np  # host-side bookkeeping only; never numpy.random

    K = R_BINS.shape[0]
    edges_np = np.asarray(edges)
    bin_of = np.digitize(np.asarray(R), edges_np) - 1  # 0..K-1; -1/K out of range
    sigma_hat = np.zeros((3, K))
    se = np.zeros((3, K))
    for b in range(K):
        members = np.flatnonzero(bin_of == b)
        n_member = members.shape[0]
        for c in range(3):
            n_need = int(round(float(n_eff[c, b])))
            n_use = max(n_need, _MIN_CELL)
            if n_use > n_member:
                raise ValueError(
                    f"calibration parent catalog too small: bin {b} channel {c} "
                    f"needs {n_use} stars but only {n_member} fell in the bin; "
                    f"increase n_parent."
                )
            key, ksub, knoise = jax.random.split(key, 3)
            pick = jax.random.choice(
                ksub, jnp.asarray(members), shape=(n_use,), replace=False
            )
            v = channels[c][pick]
            v_obs = v + EPS[c] * jax.random.normal(knoise, (n_use,))
            sig = float(jnp.std(v_obs, ddof=1))
            sigma_hat[c, b] = sig
            # SE at the TRUE design count n_eff (NOT floored at _MIN_CELL): the fit's
            # per-cell weight 1/se^2 must equal the design Fisher's per-cell weight
            # 2 n_eff / (sigma^2 + eps^2) for the calibration to be self-consistent.
            # _MIN_CELL floors only n_use (how many stars we DRAW for a numerically
            # stable ddof=1 sigma_hat), NOT the information weight. (Stage-1 floored
            # n_se too, but at its N_total few cells were sub-floor; here 15/36 cells
            # fall below _MIN_CELL=10, and flooring n_se gave those cells a too-tight
            # SE -> the realized fit over-weighted them vs the Fisher -> realized
            # variance ~0.4x too small. Using the true n_eff closes the gate at ~1.0x.)
            n_se = max(
                float(n_eff[c, b]), 1e-6
            )  # true design count; guard div-by-0 only
            se[c, b] = sig / jnp.sqrt(2.0 * n_se)
    return jnp.asarray(sigma_hat), jnp.asarray(se)


# ---------------------------------------------------------------------------
# Vectorised (vmappable) binning -- opt#4. The host-side _binned_sigma_hat above
# dominated the calibration wall-clock (~3.3 s/draw of host-side jax.random.choice +
# jnp.std dispatch over 36 cells; STEP-0 profile). This JAX-native version is a pure
# array function of (R, channels, per-star noise) with a STATIC n_use (3, K), so the
# whole draw->bin->fit pipeline can be jax.vmapped over draws (ONE compiled batched
# pass instead of n_draws sequential host loops).
# ---------------------------------------------------------------------------


def _bin_of(R, edges):
    """Per-star bin index (n,) in 0..K-1; out-of-range stars get K (a never-selected
    sentinel bin). searchsorted on the K+1 edges == np.digitize(R, edges) - 1."""
    K = R_BINS.shape[0]
    b = jnp.searchsorted(edges, R, side="right") - 1  # -1 (below) .. K (above)
    return jnp.where((b < 0) | (b >= K), K, b)  # sentinel K for out-of-range


def _within_bin_rank(bin_of, priority, n, K):
    """Rank of each star WITHIN its bin under `priority` (0 = lowest priority in bin).

    Vectorised without-replacement selection key: sorting stars by the composite key
    (bin_of, priority) puts each bin's members in a contiguous, priority-sorted run, so
    a star's within-bin rank is its global sorted position minus the count of stars in
    strictly-earlier bins. Selecting the n_use lowest-priority members of a bin <=>
    keeping stars with within-bin rank < n_use (a uniform-priority draw == uniform
    without-replacement subsample). bin sentinel K sorts last and is never selected.
    """
    composite = (
        bin_of.astype(jnp.float64) + priority
    )  # priority in [0,1) keeps bins separate
    order = jnp.argsort(composite)  # (n,) ascending
    sorted_bin = bin_of[order]
    # bin_start[k] = number of stars in bins < k = first global sorted index of bin k.
    counts = jnp.bincount(bin_of, length=K + 1)  # (K+1,) incl sentinel
    bin_start = jnp.concatenate([jnp.zeros(1, counts.dtype), jnp.cumsum(counts)[:-1]])
    rank_sorted = (
        jnp.arange(n) - bin_start[sorted_bin]
    )  # within-bin rank in sorted order
    rank = jnp.zeros(n, dtype=rank_sorted.dtype).at[order].set(rank_sorted)
    return rank  # (n,) within-bin rank, original order


def _binned_sigma_hat_jax(key, channels, R, n_use, n_se, edges):
    """vmappable JAX twin of _binned_sigma_hat: binned sigma_hat (3, K) + se (3, K).

    PURE array function (no host control flow), so jax.vmap batches it across draws.
    n_use (3, K) int and n_se (3, K) float are STATIC across draws (functions of the
    design count n_eff and _MIN_CELL), so they are precomputed ONCE on the host (where
    the parent-catalog-too-small guard also lives) and threaded in as constants.

    Per channel c: draw a uniform priority per parent star, rank stars WITHIN their bin
    (a uniform-priority rank == a uniform without-replacement subsample, matching the
    host reference's jax.random.choice(replace=False)), select the n_use[c,b] lowest-rank
    members of each bin, add per-star Gaussian error ~Normal(0, EPS[c]), and take the
    masked ddof=1 sample std. The SE is sigma_hat / sqrt(2 n_se) at the TRUE design count
    (n_se, NOT floored at _MIN_CELL) -- identical contract to the host reference.
    """
    K = R_BINS.shape[0]
    n = R.shape[0]
    bin_of = _bin_of(R, edges)  # (n,) 0..K-1, K = sentinel
    cell = jnp.arange(K)  # (K,)

    def per_channel(c, kc):
        kp, kn = jax.random.split(kc)
        priority = jax.random.uniform(kp, (n,))  # (n,) in [0,1)
        rank = _within_bin_rank(bin_of, priority, n, K)  # (n,) within-bin rank
        # star i selected for its cell c iff rank < n_use[c, bin_of[i]] (sentinel bin
        # K has n_use 0 by construction -> never selected).
        nuse_star = jnp.where(bin_of < K, n_use[c][jnp.minimum(bin_of, K - 1)], 0)
        selected = rank < nuse_star  # (n,) bool
        v_obs = channels[c] + EPS[c] * jax.random.normal(kn, (n,))  # (n,)
        # Masked per-cell ddof=1 std via grouped moments. sel_cb[b, i] = star i in cell (c,b).
        in_bin = bin_of[None, :] == cell[:, None]  # (K, n)
        sel_cb = in_bin & selected[None, :]  # (K, n)
        w = sel_cb.astype(jnp.float64)  # (K, n) 0/1 weights
        cnt = jnp.sum(w, axis=1)  # (K,) selected count == n_use[c]
        mean = jnp.sum(w * v_obs[None, :], axis=1) / jnp.maximum(cnt, 1.0)
        var = jnp.sum(w * (v_obs[None, :] - mean[:, None]) ** 2, axis=1) / jnp.maximum(
            cnt - 1.0, 1.0
        )
        return jnp.sqrt(var)  # (K,) ddof=1 sigma_hat for channel c

    kc0, kc1, kc2 = jax.random.split(key, 3)
    sigma_hat = jnp.stack(
        [per_channel(0, kc0), per_channel(1, kc1), per_channel(2, kc2)]
    )
    se = sigma_hat / jnp.sqrt(2.0 * n_se)  # (3, K); n_se = true design count
    return sigma_hat, se


def _static_cell_sizes(n_eff, R, edges):
    """Host-side STATIC per-cell sizes for the JAX binning: (n_use (3,K) int, n_se (3,K)).

    n_use = max(round(n_eff), _MIN_CELL) (how many stars to DRAW); n_se = max(n_eff, 1e-6)
    (the TRUE design count for the SE weight). Also runs the parent-catalog-too-small guard
    ON THE HOST (the JAX binning cannot raise): if any cell needs more stars than fell in
    its bin, raise the SAME clear error as the host reference _binned_sigma_hat.
    """
    import numpy as np  # host-side bookkeeping only

    K = R_BINS.shape[0]
    bin_of = np.digitize(np.asarray(R), np.asarray(edges)) - 1
    members_per_bin = np.array([int(np.sum(bin_of == b)) for b in range(K)])
    n_eff_np = np.asarray(n_eff)
    n_use = np.maximum(np.round(n_eff_np).astype(int), _MIN_CELL)  # (3, K)
    for b in range(K):
        for c in range(3):
            if n_use[c, b] > members_per_bin[b]:
                raise ValueError(
                    f"calibration parent catalog too small: bin {b} channel {c} "
                    f"needs {n_use[c, b]} stars but only {members_per_bin[b]} fell in "
                    f"the bin; increase n_parent."
                )
    n_se = np.maximum(n_eff_np, 1e-6)
    return jnp.asarray(n_use), jnp.asarray(n_se)


@functools.partial(jax.jit, static_argnames=("model", "n_iter"))
def _fit_theta_W0_gn(sigma_hat, se, G, model, n_iter=_GN_N_ITER):
    """Gauss-Newton MAP fit of theta=(W0, r_a, M) in the DIMENSIONLESS ln-theta metric.

    A faithful reimplementation of _demo_oed_depth._fit_theta_gn wired to OUR forward
    model: predict_sigma(., R_BINS, G, model) (a model arg Stage-2's fitter lacks),
    theta_truth() = (W0, r_a, M), and PRIOR_DIAG (M-only). We fit u = ln(theta) -
    ln(theta_fid), theta = theta_fid * exp(u), so every direction is O(1) (W0~6, r_a~6,
    M~1e5 span ~5 orders of magnitude; a single-LR physical optimiser cannot move the
    large-scale M -- the Stage-2 lesson). Started at the truth (u=0). Each Gauss-Newton
    iteration solves  (Jr^T Jr + diag(PRIOR_DIAG)) du = Jr^T r + PRIOR_DIAG * u  with
    r = (model - data)/se the whitened residual and Jr = d r / d u (reverse-mode jacrev
    through the King/Michie custom_vjp ODE; never forward-mode), then takes a STEP-CAPPED
    update.

    Levenberg-Marquardt damping (replaces the old fixed step-cap). Each iteration solves the
    DAMPED system  (Jr^T Jr + diag(PRIOR_DIAG) + lam I) du = -(Jr^T r + PRIOR_DIAG u), ACCEPTS
    the step only if it lowers the MAP cost 0.5(||r||^2 + sum PRIOR_DIAG u^2), and adapts lam
    (x0.3 on accept, x3 on reject). The lam I floor bounds the step in weakly-constrained
    directions -- the M-only-prior r_a anisotropy nuisance, whose tiny curvature made the
    plain GN step overshoot and OSCILLATE at the old fixed step-cap (28/48 King draws never
    settled) -- while well-constrained directions (W0, M) keep near-Newton steps. The
    accept/reject ALSO subsumes the step-cap's Michie-ODE-basin guard: a step that pushes
    (W0, r_a) past the Michie truncation makes the ODE return NaN -> cost NaN -> rejected ->
    lam up -> smaller step, so the over-anisotropic region is auto-avoided. Returns
    (theta_hat (3,), w0_witness) -- w0_witness is the MAX |W0-component step| over the LAST 5
    iterations (review I1): it witnesses convergence of the TARGET W0 (the only param that
    enters the realized variance), NOT the weakly-constrained r_a nuisance (which is EXPECTED
    not to settle and would dominate an all-param witness), and the last-5 WINDOW avoids the
    false-zero a single rejected final step would report. ~0 means W0_hat has settled.

    JIT (static_argnames model/n_iter): sigma_hat, se, G are TRACED args, so XLA compiles
    the King/Michie ODE-jacrev scan ONCE and reuses it across every calibration draw (the
    un-jitted function recompiled the whole ODE-jacrev scan each draw because it closed over
    each draw's sigma_hat/se). The per-iteration ODE re-solve at RUNTIME is unavoidable and
    stays; only the COMPILATION is hoisted. jit is value-preserving -- theta_hat is unchanged.
    """
    theta_fid = theta_truth()
    sf = sigma_hat.flatten()
    ef = se.flatten()

    def resid(u):  # whitened residual (model - data)/se
        theta = theta_fid * jnp.exp(u)
        return (predict_sigma(theta, R_BINS, G, model).flatten() - sf) / ef

    def cost_of(u, r):  # MAP negative-log-posterior (whitened)
        return 0.5 * (r @ r + jnp.sum(PRIOR_DIAG * u**2))

    def lm_step(carry, _):
        u, lam = carry
        r = resid(u)
        c = cost_of(u, r)
        Jr = jax.jacrev(resid)(u)  # (n_obs, 3) = d r / d u
        grad = Jr.T @ r + PRIOR_DIAG * u
        hess = Jr.T @ Jr + jnp.diag(PRIOR_DIAG)  # Gauss-Newton Hessian (PSD)
        du = -jnp.linalg.solve(hess + lam * jnp.eye(3), grad)  # Levenberg-damped step
        u_try = u + du
        c_try = cost_of(u_try, resid(u_try))
        improved = c_try < c  # NaN (Michie ODE blow-up) -> False -> reject
        u_next = jnp.where(improved, u_try, u)
        lam_next = jnp.clip(jnp.where(improved, lam * 0.3, lam * 3.0), 1e-9, 1e9)
        # Emit the W0-COMPONENT (index 0) of the accepted step, not the all-param ||du||_inf.
        # B2 must witness the TARGET we extract (W0); the weakly-constrained r_a nuisance
        # (M-only prior) is EXPECTED not to reach a tight optimum and would dominate an
        # all-param witness without affecting W0_hat (verified: stuck draws are r_a-stuck,
        # W0_hat well-behaved). (un - u)[0] is 0 on a reject.
        w0_moved = jnp.abs((u_next - u)[0])
        return (u_next, lam_next), w0_moved

    (u_hat, _), w0_steps = jax.lax.scan(
        lm_step, (jnp.zeros(3), _GN_LM_LAM0), None, length=n_iter
    )
    # B2 convergence witness: the MAX |W0-step| over the LAST 5 iterations (not just the final
    # iter). Taking the last-5 max -- not step_norms[-1] -- is load-bearing: a single rejected
    # final step reports 0 and would FALSELY read as converged (review I1); the window catches
    # a draw still moving or oscillating W0 in its endgame. ~0 means the EXTRACTED W0_hat has
    # settled (the only thing that enters the realized variance).
    return theta_fid * jnp.exp(u_hat), jnp.max(w0_steps[-5:])


class CalibResultW0(NamedTuple):
    """Result of calibrate_fisher_W0 (variances are FRACTIONAL/ln variances, ADR 0011):
    * realized_var_W0  : Var(ln W0_hat over draws, ddof=1),
    * fisher_var_W0    : (inv F_design)_{W0, W0} at the same (z, N_total),
    * max_W0_step      : max over draws of each LM fit's W0-target witness (the max
                         |W0-step| over its last 5 iterations; ~0 means W0_hat settled),
    * n_unconverged    : # draws whose W0 witness exceeds _GN_CONVERGED_STEP (the B2 count
                         gated on; LM settles W0 for all but the occasional hard draw)."""

    realized_var_W0: float
    fisher_var_W0: float
    max_W0_step: float
    n_unconverged: int


# B2 convergence threshold on each LM fit's W0-target witness (max |W0-step| over the last 5
# iterations, ln-theta). After _GN_N_ITER LM iterations a settled fit moves W0 < this; a draw
# above it has a still-moving (underdispersed / non-converged) W0_hat and is SURFACED, not
# swallowed. Measured: King 48 draws -> 46/48 below this, median ~6e-5 (the r_a NUISANCE does
# NOT converge -- expected, M-only prior -- but it does not enter the W0 witness).
_GN_CONVERGED_STEP = 1e-3


def calibrate_fisher_W0(z, N_total, n_draws, key, model, n_iter=_GN_N_ITER):
    """Calibrate the design Fisher against the realized scatter of ln(W0_hat).

    Returns a CalibResultW0(realized_var_W0, fisher_var_W0, max_W0_step, n_unconverged). The
    Fisher prediction is (inv F_design)_{W0, W0} at (z, N_total) with the per-star blocks at the
    truth; the realized quantity is Var(ln W0_hat over n_draws independent OM mocks, ddof=1)
    (both fractional/ln variances, ADR 0011). The gate
    (test_W0_fisher_calibration_matches_realized_scatter) asserts they agree to
    2 sqrt(2/n_draws) -- the MC error on a variance from n_draws draws.

    PERFORMANCE (opt#4, MEMORY-BOUNDED): the whole draw->bin->fit pipeline runs over the
    n_draws independent draw keys via jax.lax.map (a SEQUENTIAL scan), so `one_draw`
    compiles ONCE (compile-once win) yet executes draw-by-draw -- peak memory is a SINGLE
    draw's reverse-mode-through-ODE backward tape, not n_draws of them. (A full jax.vmap
    here batched all n_draws Michie ODE-jacrev solves into one fused computation, ~48x the
    live tape memory, and OOM-killed the host; reverse-mode through an equilibrium ODE is
    exactly where batching the memory is the wrong trade.) The binning runs JAX-native
    (_binned_sigma_hat_jax) so the pipeline is a pure traceable body (no host control flow),
    which is what lets lax.map compile it once. The per-draw-invariant OM sampler structure
    is still built ONCE (identical every draw -- same truth). n_use/n_se (static per-cell
    sizes + the parent-catalog-too-small guard) are computed ONCE on the host from the
    draw-independent design count n_eff (n_use does not depend on the draw's R; the guard is
    validated on a representative draw). Value-preserving in the statistical sense (a
    different but equivalent RNG stream): the realized variance agrees within the
    calibration band (King ratio ~0.976 with the Levenberg-Marquardt fit).
    """
    G = STELLAR.G
    theta = theta_truth()
    Mb, _ = per_star_blocks(theta, R_BINS, EPS, G, model)
    cb = completeness(R_BINS)
    n_eff = design_counts(z, cb, N_total)  # (3, K)
    fisher_var_W0 = float(jnp.linalg.inv(fisher(z, Mb, cb, N_total, PRIOR_DIAG))[0, 0])

    edges = _r_bin_edges()
    # Parent catalog large enough that every R-bin holds >> the largest design cell
    # count (the thin outer King/Michie bins are the binding constraint; _static_cell_sizes
    # raises a clear error if any cell underflows). Mirror Stage-1's sizing.
    n_parent = int(max(8000, 4 * N_total))

    # Build the per-draw-invariant OM sampler structure ONCE: all n_draws mocks are
    # sampled at the SAME truth (W0, r_a, M, n_parent), so the expensive build (King
    # Engine B model / Michie Eddington table) is identical every draw. Only the cheap
    # per-key _draw_om is batched across draws below.
    sampler = _build_om_sampler(model, MOCK["W0"], MOCK["r_a"], MOCK["M"], n_parent)

    # Static per-cell sizes (draw-independent) + the host-side parent-catalog guard,
    # validated on a representative draw (draw 0). n_use does NOT depend on the draw's R.
    k0cat = jax.random.split(jax.random.fold_in(key, 0))[0]
    R0, *_ = project_to_sky(*_draw_om(model, sampler, n_parent, k0cat))
    n_use, n_se = _static_cell_sizes(n_eff, R0, edges)

    def one_draw(kdraw):
        """One mock draw -> ln(W0_hat) + the GN fit's final-iter step norm (B2 witness)."""
        kcat, kbin = jax.random.split(kdraw)
        pos, vel = _draw_om(model, sampler, n_parent, kcat)
        R, v_los, v_pm_r, v_pm_t = project_to_sky(pos, vel)
        channels = jnp.stack([v_los, v_pm_r, v_pm_t])
        sigma_hat, se = _binned_sigma_hat_jax(kbin, channels, R, n_use, n_se, edges)
        theta_hat, step_norm = _fit_theta_W0_gn(sigma_hat, se, G, model, n_iter=n_iter)
        return jnp.log(theta_hat[0]), step_norm

    # Run the pipeline over the n_draws draw keys with jax.lax.map (SEQUENTIAL scan),
    # NOT jax.vmap. lax.map compiles `one_draw` ONCE (same compile-once win) but executes
    # it draw-by-draw, so peak memory is ONE draw's ODE-jacrev backward tape, not n_draws
    # of them. A full jax.vmap here batches all n_draws Michie reverse-mode solves into one
    # fused computation (~48x the live tape memory) and OOM-killed the host -- the memory/
    # speed trade is wrong for a reverse-mode-through-ODE inner loop, so we keep it serial.
    draw_keys = jax.vmap(lambda d: jax.random.fold_in(key, d))(
        jnp.arange(n_draws)
    )  # cheap (no ODE)
    ln_W0_hats, w0_witnesses = jax.lax.map(one_draw, draw_keys)

    # B2: surface (do NOT swallow) any draw whose W0_hat had not settled (W0-target witness:
    # max |W0-step| over the fit's last 5 iters; the r_a nuisance is expected not to converge
    # and is deliberately NOT in this witness).
    max_W0_step = float(jnp.max(w0_witnesses))
    n_unconverged = int(jnp.sum(w0_witnesses > _GN_CONVERGED_STEP))
    if n_unconverged > 0:
        print(
            f"[calibrate_fisher_W0/{model}] WARNING: {n_unconverged}/{n_draws} draws had a "
            f"W0-target witness > {_GN_CONVERGED_STEP:g} (max {max_W0_step:.2e}); the W0_hat for "
            f"those draws may be underdispersed -- investigate before trusting the realized "
            f"variance. (The r_a nuisance not settling is expected and excluded from this witness.)"
        )

    realized_var_W0 = float(jnp.var(ln_W0_hats, ddof=1))
    return CalibResultW0(
        realized_var_W0=realized_var_W0,
        fisher_var_W0=fisher_var_W0,
        max_W0_step=max_W0_step,
        n_unconverged=n_unconverged,
    )
