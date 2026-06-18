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
from progenax import KingProfile, MichieProfile, MultiComponentCluster, project_dispersion
from progenax.profiles.michie import michie_density
from progenax.numerics import cumulative_trapz
from progenax.kinematics.eddington import (
    assign_om_directions,
    eddington_invert,
    sample_speed_from_f_table,
)
from jaxstro.units import STELLAR

# Reuse the Stage-1 sky projection (line of sight = +z) + unit conversions, and the
# MODEL-AGNOSTIC Fisher/criteria backbone (consume per-star blocks Mb + a design
# vector; nothing below differentiates the forward model). Re-exported so callers use
# ONE namespace (oedc.c_criterion, oedc.fisher, ...). The Stage-1 optimize_design /
# _optimize_one are NOT reused: they hardcode Stage-1's PRIOR_DIAG (a module global),
# so this module reimplements them threading OUR W0-OED PRIOR_DIAG (see below).
sys.path.insert(0, os.path.dirname(__file__))
import optax  # noqa: E402
from _demo_oed import (  # noqa: E402
    DesignResult,
    a_criterion,
    blocks_from_eps,
    c_criterion,
    d_criterion,
    design_counts,
    fisher,
    kms_to_pcMyr,
    pm_masyr_to_kms,
    project_to_sky,
)

# Resolution of the hand-rolled Michie Eddington table (mirrors eff_df defaults).
_N_R = 6000      # radial grid for the potential / density tabulation
_N_E = 1000      # energy grid for f(E)
_N_SPEED = 256   # per-particle speed inverse-CDF resolution


# ---------------------------------------------------------------------------
# Mock truth + observing setup (analog of Stage-1's MOCK). theta = (W0, r_a, M);
# index map W0=0 (TARGET concentration), r_a=1, M=2. r_c == 1 is the length unit.
# ---------------------------------------------------------------------------
MOCK = dict(
    W0=6.0, r_a=6.0, M=1e5, r_c=1.0, d_kpc=4.0,
    eps_RV_kms=1.0, eps_PM_masyr=0.05,
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
EPS = jnp.array([_eps_RV, _eps_PM, _eps_PM])      # (3,) [pc/Myr]


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
    return jnp.stack([pd.sigma_los, pd.sigma_pm_r, pd.sigma_pm_t])   # (3, K)


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
    sig = predict_sigma(theta, R_bins, G, model)                          # (3, K)
    J = jax.jacrev(predict_sigma, argnums=0)(theta, R_bins, G, model)     # (3, K, 3) -- d sigma / d theta
    return J * theta[None, None, :], sig    # -> d sigma / d ln theta (DIMENSIONLESS, ADR 0011)


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
_FRAC_PRIOR_M = 0.3   # 30% fractional prior on M (the only externally-constrained param)
PRIOR_DIAG = jnp.array([0.0, 0.0, 1.0 / _FRAC_PRIOR_M**2])   # [W0, r_a, M] fractional precision


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


def optimize_design(criterion_fn, Mb, cb, N_total, key, n_starts=8, n_steps=500, lr=0.05):
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
