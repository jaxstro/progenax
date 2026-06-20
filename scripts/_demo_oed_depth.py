"""OED Stage-2 depth layer: the magnitude-limit ``m_lim`` as an optimisable design knob.

This module sits ON TOP of the Stage-1 OED core (``_demo_oed``) and the shared selection /
photometry physics (``_demo_selection``). It promotes the limiting (apparent) magnitude ``m_lim``
from Stage-1's fixed completeness to a continuous, differentiable design variable that headlines the
dynamical mass ``M`` (theta index 1).

Three couplings depth introduces, all differentiable in ``m_lim`` (see the design doc, section B):

1. ``eps_eff(m_lim)`` -- the IMF-weighted RMS per-star error over the DETECTABLE mass range.
   Deeper ``m_lim`` admits fainter (photon-noisier) stars, so the per-channel effective error RISES.
   This is the direct tie-back to Stage 1: ``oed.EPS`` are the per-channel errors at a bright
   reference magnitude ``M_REF``; ``sel.photon_noise_error`` scales them with apparent magnitude.

2. ``avail_bins(m_lim)`` -- the per-bin pool of stars depth unlocks: an intrinsic radial star pool
   ``N_FIELD_BINS`` (projected Plummer surface density x annular area, DECREASING with radius)
   times the global IMF-detectable fraction ``sel.detectable_fraction(m_lim)``.

3. ``depth_fisher(z, m_lim, N_total)`` -- the additive design Fisher rebuilt at ``eps_eff(m_lim)``
   with the allocation smoothly capped by availability (``n_eff = avail * tanh(n_design / avail)``).
   The Stage-1 backbone survives: ``sigma_pred`` is ``m_lim``-INDEPENDENT (single-population,
   mass-follows-light), so the per-star Jacobian ``J`` is still computed ONCE via
   ``oed.jacobian_and_sigma``; ``m_lim`` enters only through the cheap scalars above.

Differentiability is load-bearing: every ``m_lim`` path here is smooth jnp (geometric grids whose
ENDPOINTS move with ``m_lim`` -- no boolean-mask cuts), so the AD ``m_lim`` gradient that Task 4's
AD-vs-FD gate depends on flows cleanly. Reverse-mode only (``jacrev``/``grad``); forward-mode is
banned through ``project_dispersion``'s ``custom_vjp`` (inherited from the Stage-1 core).
"""

import math
from typing import NamedTuple

import _demo_oed as oed
import _demo_selection as sel
import jax
import jax.numpy as jnp
import optax
from jaxstro.units import STELLAR

from progenax import ChabrierIMF

# --- Selection constants (single cluster distance; bolometric magnitudes, no band/BC/extinction) ---
D_PC = oed.MOCK["d_kpc"] * 1000.0  # cluster distance: 4 kpc -> 4000 pc
M_MAX = 100.0  # IMF upper mass [M_sun]
_IMF = ChabrierIMF(m_min=0.08, m_max=M_MAX)

# Per-channel Stage-1 errors (pc/Myr): [RV, PM_R, PM_T]. These are the errors AT the reference
# apparent magnitude M_REF; fainter stars are noisier via sel.photon_noise_error (Stage-1 tie-back).
EPS0 = oed.EPS  # (3,) [pc/Myr]

# Reference apparent magnitude: a BRIGHT reference star near the top of the present-day mass function
# -- a 5 M_sun star at the cluster distance. Rationale: anchor the per-star error at a well-measured
# bright star; everything fainter degrades via photon noise. For DEEP limits the bottom-heavy IMF bulk
# sits fainter than this reference, so the IMF-weighted eps_eff rises ABOVE EPS0; for SHALLOW limits
# (m_lim below the EPS0 crossover, ~12.5 here) only the brightest stars are detected and eps_eff dips
# BELOW EPS0 -- physically correct (a shallow survey sees only well-measured stars). The load-bearing
# property is that eps_eff rises MONOTONICALLY with depth, which holds across the full range.
# (Task 5 will tune the normalisations; this is a defensible start.)
M_REF = sel.apparent_mag(jnp.array(5.0), D_PC)

# Total intrinsic field population (the dominant Task-5 tunable): sets the per-bin star pool scale.
N_FIELD = 2.0e4

# Number of mass-quadrature nodes for the eps_eff IMF integral (fixed count so the geometric grid's
# moving endpoints keep eps_eff differentiable in m_lim; no boolean masking).
_NGRID = 200

# Stage-2 prior: theta = (r_a, M, r_h). The headline TARGET is M (dynamical mass), so M is left FREE
# (no prior) -- you do not constrain the very quantity you are trying to measure. The NUISANCES r_a and
# r_h carry the same fractional-0.3 ln-theta prior the nuisances had in Stage 1. (Stage 1's oed.PRIOR_DIAG
# = [0, 1/0.3^2, 1/0.3^2] left r_a free because r_a was THAT stage's target; this is the M-target analogue,
# obtained by swapping which entry is zero.) Used by depth_fisher AND the calibration MAP fit so the two
# stay consistent.
PRIOR_DIAG_M = jnp.array([1.0 / 0.3**2, 0.0, 1.0 / 0.3**2])


def _n_field_bins():
    """Intrinsic per-bin star pool N_FIELD_BINS (K,), DECREASING with radius.

    Derived from the projected Plummer surface density Sigma(R) = M a^2 / [pi (a^2 + R^2)^2]
    (Plummer 1911 projection) evaluated at each R_BINS centre, times the bin's annular area
    (R_out^2 - R_in^2). Constant factors (M, a^2, pi) cancel in the per-bin FRACTIONS, so only the
    shape matters; the result is normalised to the total field population N_FIELD. The Plummer scale
    radius a follows from the mock half-mass radius (a = r_h * sqrt(2^(2/3) - 1)). The surface density
    Sigma(R) is strictly decreasing, but the per-bin POOL (Sigma * annular area) peaks at intermediate
    radii because annular area ~ R^2 beats the central falloff: both the innermost AND the outermost
    bins are relatively star-poor, with N_FIELD_BINS[0] > N_FIELD_BINS[-1] (outskirts starved -- the
    physics that makes depth a real trade for M_dyn's radial leverage in the outer bins).
    """
    R = oed.R_BINS  # (K,) on-sky bin centres
    a = oed.MOCK["r_h"] * jnp.sqrt(2.0 ** (2.0 / 3.0) - 1.0)  # Plummer scale radius
    edges = oed._r_bin_edges()  # (K+1,) geometric-mean edges
    area = edges[1:] ** 2 - edges[:-1] ** 2  # (K,) annular area (drop the 2pi)
    sigma = a**2 / (a**2 + R**2) ** 2  # projected Plummer surface density (shape)
    pool = sigma * area  # per-bin intrinsic count (shape)
    return N_FIELD * pool / jnp.sum(pool)  # normalise to N_FIELD


N_FIELD_BINS = _n_field_bins()  # (K,) module constant

# The per-star Jacobian J = d sigma_pred / d ln theta (3,K,3) and sigma (3,K), evaluated at the FIXED
# fiducial theta. These are the single expensive computation in the whole Stage-2 layer: jacrev through
# project_dispersion's diffrax ODE. They are INDEPENDENT of the design variables (z, m_lim) -- sigma_pred
# is m_lim-independent (single-population) and theta is the fixed fiducial -- so they are computed ONCE
# here at import, NOT inside depth_fisher. (Computing them per depth_fisher call made the optimiser, which
# evaluates depth_fisher thousands of times, ~32 min instead of seconds; XLA cannot hoist the opaque
# custom_vjp ODE jacrev out of the Adam scan.) The design optimisation differentiates only wrt (z, m_lim),
# never wrt theta, so caching J/SIG as constants leaves every design gradient bit-identical.
_J, _SIG = oed.jacobian_and_sigma(
    oed.theta_truth(), oed.R_BINS, STELLAR.G
)  # ODE jacrev, ONCE at import


def eps_eff(m_lim):
    """IMF-weighted RMS per-star measurement error over the detectable masses, per channel (3,).

    Builds a FIXED-COUNT geometric mass grid from the differentiable lower edge
    ``m_lo = sel.m_min(m_lim, D_PC)`` up to ``M_MAX``; the endpoints move SMOOTHLY with ``m_lim``
    (no boolean-mask cut on a fixed grid -- that would kill the m_lim gradient). IMF weights are the
    probability mass per cell ``dP_i = cdf(m_grid[1:]) - cdf(m_grid[:-1])`` (differentiable; no pdf
    method needed); the error is evaluated at the cell-midpoint masses. For each channel c,
    ``eps_eff_c = sqrt(sum dP_i * eps_c(m_i)^2 / sum dP_i)`` with
    ``eps_c(m) = sel.photon_noise_error(sel.apparent_mag(m, D_PC), EPS0[c], M_REF)``.

    Monotone: deeper m_lim -> smaller m_lo -> admits fainter (noisier) stars -> eps_eff RISES.
    """
    # Belt-and-braces NaN guard (review item I1): m_min(m_lim) can exceed M_MAX at
    # unphysically bright m_lim (~m_lim<4), which would INVERT the geometric grid and
    # NaN-poison the gradient. Clip the lower edge below M_MAX and floor the normaliser.
    # These guards activate ONLY in the unphysical m_lim<4 regime -- far outside the
    # [M_LIM_LO, M_LIM_HI] operating range -- so they do NOT alter eps_eff or its
    # gradient anywhere in range (verified byte-identical at m_lim=12; see Task 4 report).
    m_lo = jnp.minimum(
        sel.m_min(m_lim, D_PC), M_MAX * (1.0 - 1e-3)
    )  # differentiable, clipped edge
    frac = jnp.linspace(0.0, 1.0, _NGRID)
    m_grid = m_lo * (M_MAX / m_lo) ** frac  # geometric grid, moving lower endpoint
    dP = _IMF.cdf(m_grid[1:]) - _IMF.cdf(
        m_grid[:-1]
    )  # (_NGRID-1,) IMF probability mass/cell
    m_mid = jnp.sqrt(m_grid[1:] * m_grid[:-1])  # geometric cell midpoints
    m_app = sel.apparent_mag(m_mid, D_PC)  # (_NGRID-1,) apparent magnitudes
    norm = jnp.maximum(jnp.sum(dP), 1e-30)  # guard the normaliser (never 0)

    def rms(eps0_c):
        eps_c = sel.photon_noise_error(
            m_app, eps0_c, M_REF
        )  # per-star error in channel c
        return jnp.sqrt(jnp.sum(dP * eps_c**2) / norm)

    return jnp.array([rms(EPS0[0]), rms(EPS0[1]), rms(EPS0[2])])


def avail_bins(m_lim):
    """Per-bin available star pool (K,): intrinsic radial pool x global IMF-detectable fraction.

    ``N_FIELD_BINS * sel.detectable_fraction(m_lim, D_PC, _IMF)``. The detectable fraction is global
    (single cluster distance -> one apparent-magnitude map), so it multiplies the radial pool
    uniformly; the radial structure comes entirely from N_FIELD_BINS (core-rich, outskirt-poor).
    Monotone in m_lim (deeper -> larger detectable fraction -> more per bin).
    """
    return N_FIELD_BINS * sel.detectable_fraction(m_lim, D_PC, _IMF)


def depth_fisher(z, m_lim, N_total, prior_diag=PRIOR_DIAG_M):
    """Additive design Fisher at limiting magnitude m_lim. Symmetric (3, 3), SPD (with prior_diag).

    The Stage-1 backbone: J computed ONCE (sigma_pred is m_lim-independent), the per-star blocks
    rebuilt at eps_eff(m_lim), and the softmax allocation n_design SMOOTHLY capped by availability
    via ``n_eff = avail * tanh(n_design / avail)`` (~n when n << avail, saturates at avail when
    n >> avail -- a differentiable finite-supply constraint). eps_eff (3,) broadcasts per channel;
    avail (K,) is shared across channels at a single distance. Differentiable in both z and m_lim.
    """
    Mb = oed.blocks_from_eps(_J, _SIG, eps_eff(m_lim))  # (3,K,3,3), J/SIG cached once

    K = oed.R_BINS.shape[0]
    n_design = N_total * jax.nn.softmax(z).reshape(3, K)  # (3,K) budget allocation
    avail = avail_bins(m_lim)[None, :]  # (1,K) per-channel pool
    n_eff = avail * jnp.tanh(n_design / avail)  # smooth availability cap
    return jnp.einsum("ck,ckpq->pq", n_eff, Mb) + jnp.diag(prior_diag)


# ===========================================================================
# Task 4: joint [z, m_lim] optimiser + bounded m_lim reparametrisation
# ===========================================================================
#
# m_lim is a PHYSICALLY bounded knob: too bright and m_min(m_lim) exceeds the IMF
# ceiling (the eps_eff geometric grid would invert -> NaN, review item I1); too
# faint is unphysical for any survey. We optimise an UNCONSTRAINED scalar u and map
# it through a sigmoid into [M_LIM_LO, M_LIM_HI], so every finite u is a valid,
# differentiable design (the bounds keep the optimiser away from the NaN crossover,
# and the I1 belt-and-braces guard in eps_eff covers any excursion). The joint design
# vector is [z (3K logits), u (1 scalar)] and Adam runs over the whole thing.

M_LIM_LO = 9.0  # expit lower bound. m_min(9, 4kpc) ~ 8.4 M_sun, FAR above the
# m_min=M_MAX NaN crossover (~m_lim<4); confirmed m_min(M_LIM_LO) << M_MAX.
M_LIM_HI = 18.0  # m_min(18, 4kpc) ~ 1.03 M_sun (deep into the IMF bulk).


def u_to_mlim(u):
    """Map the unconstrained design scalar u -> m_lim in [M_LIM_LO, M_LIM_HI] (smooth sigmoid)."""
    return M_LIM_LO + (M_LIM_HI - M_LIM_LO) * jax.nn.sigmoid(u)


def depth_fisher_u(z, u, N_total, prior_diag=PRIOR_DIAG_M):
    """depth_fisher in the bounded reparametrisation: m_lim = u_to_mlim(u). Differentiable in z, u."""
    return depth_fisher(z, u_to_mlim(u), N_total, prior_diag)


class DepthDesignResult(NamedTuple):
    """Result of optimize_depth_design (best multi-start trajectory):
    * criterion : best scalar criterion value (minimised),
    * m_lim     : optimal limiting magnitude = u_to_mlim(best u),
    * z         : best allocation logits (3*K,),
    * u         : best unconstrained depth scalar,
    * n_design  : realised per-cell counts N_total * softmax(z) (3, K), pre-cap,
    * n_eff     : per-cell counts after the availability soft-cap (3, K)."""

    criterion: float
    m_lim: float
    z: jnp.ndarray
    u: jnp.ndarray
    n_design: jnp.ndarray
    n_eff: jnp.ndarray


def _n_design_eff(z, m_lim, N_total):
    """Realised (n_design, n_eff) (3, K) each: softmax budget and its availability soft-cap."""
    K = oed.R_BINS.shape[0]
    n_design = N_total * jax.nn.softmax(z).reshape(3, K)
    avail = avail_bins(m_lim)[None, :]
    n_eff = avail * jnp.tanh(n_design / avail)
    return n_design, n_eff


def _optimize_joint(loss, params0, n_steps, lr):
    """One Adam trajectory over an arbitrary pytree of params; returns (params_final, trace).

    Fixed-iteration jax.lax.scan (NOT while_loop) so the trajectory stays differentiable-safe
    and JIT-compilable. ``loss`` is a scalar function of the params pytree."""
    opt = optax.adam(lr)
    state = opt.init(params0)

    @jax.jit
    def step(carry, _):
        p, st = carry
        l, g = jax.value_and_grad(loss)(p)
        upd, st = opt.update(g, st)
        return (optax.apply_updates(p, upd), st), l

    (params, _), trace = jax.lax.scan(step, (params0, state), None, length=n_steps)
    return params, trace


def optimize_depth_design(
    target, N_total, key, n_starts=8, n_steps=500, lr=0.05, prior_diag=PRIOR_DIAG_M
):
    """Multi-start Adam over the JOINT design [z (3K logits), u (1 scalar)]; keep the best.

    Minimises ``c_criterion(depth_fisher_u(z, u, N_total), target)`` -- the marginal fractional
    variance of theta[target] (Stage 2: target=1 = M_dyn). Reuses the Stage-1 optax + lax.scan
    pattern (fixed-iteration scan, no while_loop); each start draws an independent z0 (and u0),
    runs one Adam trajectory, and the lowest-criterion start wins. The sigmoid reparametrisation
    keeps m_lim in [M_LIM_LO, M_LIM_HI] for every finite u (no constraints needed).

    SPD invariant (as in oed.optimize_design): softmax(z) > 0 and PRIOR_DIAG adds positive
    nuisance precision, so depth_fisher_u stays SPD throughout and c_criterion's inverse never
    hits a singular F. Returns a DepthDesignResult for the best start.
    """
    K = oed.R_BINS.shape[0]
    loss = lambda p: oed.c_criterion(
        depth_fisher_u(p["z"], p["u"], N_total, prior_diag), target=target
    )
    best = None
    for s in range(n_starts):
        ks = jax.random.fold_in(key, s)
        kz, ku = jax.random.split(ks)
        p0 = {
            "z": jax.random.normal(kz, (3 * K,)) * 0.5,
            "u": jax.random.normal(ku, ()) * 0.5,
        }
        p, _ = _optimize_joint(loss, p0, n_steps, lr)
        crit = float(loss(p))
        if math.isfinite(crit) and (
            best is None or crit < best.criterion
        ):  # skip NaN/inf starts (M1)
            m_lim = u_to_mlim(p["u"])
            n_design, n_eff = _n_design_eff(p["z"], m_lim, N_total)
            best = DepthDesignResult(
                criterion=crit,
                m_lim=float(m_lim),
                z=p["z"],
                u=p["u"],
                n_design=n_design,
                n_eff=n_eff,
            )
    return best


def crit_at_fixed_depth(
    m_lim,
    target,
    N_total,
    key=jax.random.PRNGKey(0),
    n_starts=6,
    n_steps=400,
    lr=0.05,
    prior_diag=PRIOR_DIAG_M,
):
    """Best criterion optimising the ALLOCATION z ONLY at a FROZEN m_lim (depth held fixed).

    Multi-start Adam over z alone (same fixed-iteration scan pattern), m_lim a constant. Used by
    Task 5's depth sweep and the beats-fixed-depth test: the joint optimum must beat the best
    achievable allocation at any single fixed depth. Returns the best scalar criterion.
    """
    K = oed.R_BINS.shape[0]
    loss = lambda z: oed.c_criterion(
        depth_fisher(z, m_lim, N_total, prior_diag), target=target
    )
    best = None
    for s in range(n_starts):
        z0 = jax.random.normal(jax.random.fold_in(key, s), (3 * K,)) * 0.5
        z, _ = _optimize_joint(loss, z0, n_steps, lr)
        crit = float(loss(z))
        if math.isfinite(crit):  # skip NaN/inf starts (M1)
            best = crit if best is None else min(best, crit)
    return best


# ===========================================================================
# Task 5: the interior-optimum-in-depth result (the Stage-2 headline)
# ===========================================================================


def sigma_M_vs_depth(
    m_grid, target, N_total, key=jax.random.PRNGKey(0), n_starts=4, n_steps=300
):
    """sigma(theta[target])/theta[target] (fractional, ln-metric) vs limiting magnitude.

    For each m_lim in m_grid, the best achievable fractional precision at that FROZEN depth
    (optimal allocation z). The headline sweep: a too-shallow survey is supply-starved (few bright
    stars, esp. in the outskirts), a too-deep one is photon-noise-limited, so sigma(M)/M has an
    INTERIOR minimum -- the optimal depth to weigh the cluster. Returns an array shaped like m_grid.
    """
    return jnp.array(
        [
            jnp.sqrt(
                crit_at_fixed_depth(
                    float(m),
                    target,
                    N_total,
                    key=key,
                    n_starts=n_starts,
                    n_steps=n_steps,
                )
            )
            for m in m_grid
        ]
    )


# ===========================================================================
# Task 6: magnitude-selected calibration of the depth Fisher (the Stage-2 gate)
# ===========================================================================
#
# This is the Stage-2 analogue of Stage-1's oed.calibrate_fisher: it confirms that
# the depth Fisher's predicted sigma(M_dyn) at a GIVEN design (z, m_lim, N_total)
# matches the realised scatter of M_hat when the mock is drawn with the ACTUAL
# magnitude selection + per-star photon-noise errors. The whole point of the
# eps_eff approximation is that a heterogeneous, magnitude-limited sample carries
# the same per-cell information as a homogeneous sample with one effective error
# eps_eff -- this gate proves it forward.
#
# The consistency algebra (mock <-> Fisher), per (channel c, bin b) cell:
#   * The Fisher's per-cell denominator is sigma_pred^2 + eps_eff_c^2, where
#       eps_eff_c^2 = E_{m~IMF|detect}[ eps_c(m)^2 ]
#     is the IMF-weighted MEAN squared per-star error over the DETECTABLE mass
#     range [m_lo, M_MAX] with m_lo = sel.m_min(m_lim) (exactly the distribution
#     eps_eff() integrates), and the per-cell weight is n_eff (the availability-
#     capped observed count from _n_design_eff).
#   * The mock draws n_eff masses from that SAME truncated-IMF detection
#     distribution, gives each star a heterogeneous error
#       eps_i = sel.photon_noise_error(sel.apparent_mag(mass_i), EPS0[c], M_REF),
#     and a velocity v_i ~ Normal(0, sqrt(sigma_pred^2 + eps_i^2)). The sample
#     variance of n_eff such draws has expectation
#       mean_i(sigma_pred^2 + eps_i^2) = sigma_pred^2 + mean_i(eps_i^2)
#     and mean_i(eps_i^2) -> E_{IMF|detect}[eps^2] = eps_eff_c^2 as n_eff grows.
#   So E[sigma_hat^2] -> sigma_pred^2 + eps_eff_c^2 and its sampling SE matches the
#   Fisher's (sigma^2 + eps_eff^2)/(2 n_eff) EXACTLY -- mock and Fisher consistent
#   on the eps_eff definition. THAT consistency is what this gate verifies.


def _truncated_imf_masses(key, n, m_lo):
    """Sample n masses from _IMF conditioned on detection: mass in [m_lo, M_MAX].

    Inverse-CDF on the truncated distribution: with u ~ U(0,1),
    u' = cdf(m_lo) + u * (cdf(M_MAX) - cdf(m_lo)) = cdf(m_lo) + u * (1 - cdf(m_lo))
    (since M_MAX = _IMF.m_max so cdf(M_MAX) = 1), then mass = ppf(u'). This is the
    EXACT detection-conditional IMF that eps_eff() integrates over with weights dP.

    Note: the float() cast makes this a non-differentiable SAMPLING path -- correct here (it draws
    a mock), but do NOT reuse it on a gradient path. The m_lim differentiability lives in eps_eff /
    depth_fisher (smooth geometric grid), not in this sampler.
    """
    m_lo = float(jnp.minimum(m_lo, M_MAX * (1.0 - 1e-3)))
    c_lo = _IMF.cdf(jnp.array(m_lo))
    u = jax.random.uniform(key, (n,))
    return _IMF.ppf(c_lo + u * (1.0 - c_lo))


def _binned_sigma_hat_selected(key, m_lim, n_eff):
    """One magnitude-selected mock's binned dispersions sigma_hat (3, K) + SEs se (3, K).

    Stage-2 analogue of oed._binned_sigma_hat: instead of subsampling a parent catalog
    and adding a SINGLE per-channel error, it draws, per (channel c, bin b) cell,
    n_use = max(round(n_eff[c,b]), _MIN_CELL) masses from the detection-conditional IMF
    (_truncated_imf_masses), maps each to its HETEROGENEOUS per-star error
    eps_i = sel.photon_noise_error(sel.apparent_mag(mass_i), EPS0[c], M_REF), draws each
    star's velocity component v_i ~ Normal(0, sqrt(sigma_pred^2 + eps_i^2)) at the truth
    sigma_pred (_SIG, the module-cached (3,K) truth dispersions), and takes the ddof=1 std. By the consistency
    algebra above, E[sigma_hat^2] -> sigma_pred^2 + eps_eff_c^2, matching depth_fisher's
    denominator. The SE is sigma_hat / sqrt(2 n_se) at n_se = max(n_eff, _MIN_CELL) -- the
    same Gaussian-delta SE Stage-1 uses, so the fit weights match the Fisher's per-cell
    weight exactly.

    Host-side control flow over (bin, channel) is fine here: this is the @slow gate path,
    not a jitted hot loop. All randomness stays in jax.random.
    """
    import numpy as np  # host-side bookkeeping only; never numpy.random

    K = oed.R_BINS.shape[0]
    m_lo = sel.m_min(
        jnp.array(float(m_lim)), D_PC
    )  # detection floor (differentiable; here scalar)
    sigma_hat = np.zeros((3, K))
    se = np.zeros((3, K))
    for c in range(3):
        for b in range(K):
            n_need = int(round(float(n_eff[c, b])))
            n_use = max(n_need, oed._MIN_CELL)
            key, kmass, kvel = jax.random.split(key, 3)
            masses = _truncated_imf_masses(
                kmass, n_use, m_lo
            )  # (n_use,) detected masses
            m_app = sel.apparent_mag(masses, D_PC)  # (n_use,) apparent mags
            eps_i = sel.photon_noise_error(
                m_app, EPS0[c], M_REF
            )  # (n_use,) heterogeneous errors
            sd = jnp.sqrt(
                _SIG[c, b] ** 2 + eps_i**2
            )  # per-star total spread (truth sigma_pred)
            v_obs = sd * jax.random.normal(
                kvel, (n_use,)
            )  # zero-mean velocity component
            sig = float(jnp.std(v_obs, ddof=1))
            sigma_hat[c, b] = sig
            n_se = max(
                float(n_eff[c, b]), float(oed._MIN_CELL)
            )  # Fisher per-cell weight, floored
            se[c, b] = sig / jnp.sqrt(2.0 * n_se)
    return jnp.asarray(sigma_hat), jnp.asarray(se)


def _fit_theta_gn(sigma_hat, se, prior_diag, n_iter=8):
    """Gauss-Newton MAP fit of theta=(r_a, M, r_h) in the DIMENSIONLESS ln-theta metric (ADR 0011),
    for the Stage-2 (M-target) calibration.

    Stage 2 targets the dynamical mass M~1e5, which a single-learning-rate optimiser over physical
    theta cannot move (oed.fit_map_theta pins it); so we fit the dimensionless u = ln(theta) -
    ln(theta_fid), theta = theta_fid * exp(u), where every direction is O(1). Gauss-Newton converges
    in ~8 iterations for this mildly-nonlinear, WELL-CONSTRAINED-in-M problem (GM sets the dispersion
    scale, so the M direction of Jr^T Jr is strong and the GN step is stable -- unlike Stage 1's
    weakly-constrained r_a, which is why Stage 1 keeps its own physical-Adam fit). Each iteration
    solves  (Jr^T Jr + diag(prior_diag)) du = Jr^T r + prior_diag * u  with r = (model - data)/se the
    whitened residual and Jr = d r / d u (reverse-mode jacrev through the ODE; never forward-mode).
    Returns theta_hat (3,).
    """
    theta_fid = oed.theta_truth()
    sf = sigma_hat.flatten()
    ef = se.flatten()

    def resid(u):  # whitened residual (model - data)/se
        theta = theta_fid * jnp.exp(u)
        return (oed.predict_sigma(theta, oed.R_BINS, STELLAR.G).flatten() - sf) / ef

    def gn_step(u, _):
        r = resid(u)
        Jr = jax.jacrev(resid)(u)  # (n_obs, 3) = d r / d u
        grad = Jr.T @ r + prior_diag * u
        hess = Jr.T @ Jr + jnp.diag(prior_diag)  # Gauss-Newton Hessian (SPD)
        return u - jnp.linalg.solve(hess, grad), None

    u_hat, _ = jax.lax.scan(gn_step, jnp.zeros(3), None, length=n_iter)
    return theta_fid * jnp.exp(u_hat)


class DepthCalibResult(NamedTuple):
    """Result of calibrate_depth_fisher (both FRACTIONAL variances of M, ADR 0011):
    * realized : Var(M_hat over draws) / M_truth**2,
    * predicted : (inv depth_fisher(z, m_lim, N_total))_{M, M}."""

    realized: float
    predicted: float


def calibrate_depth_fisher(z, m_lim, N_total, n_draws, key):
    """Calibrate the depth Fisher's sigma(M_dyn) against the realised M_hat scatter.

    Stage-2 analogue of oed.calibrate_fisher, but each mock is drawn with the ACTUAL
    magnitude selection + heterogeneous per-star photon-noise errors (see the module
    block comment for the mock<->Fisher consistency algebra). At the GIVEN design
    (z, m_lim, N_total):
      predicted = (inv depth_fisher(z, m_lim, N_total))_{M, M}   [M = theta index 1],
      realized  = Var(M_hat over n_draws independent magnitude-selected mocks) / M_truth**2.
    Per draw: take the observed counts n_eff from _n_design_eff (availability-capped),
    form the magnitude-selected binned dispersions (_binned_sigma_hat_selected), and
    MAP-fit theta with the Gauss-Newton ln-theta fitter (_fit_theta_gn, ADR 0011), with the SAME
    M-free Stage-2 prior PRIOR_DIAG_M the depth Fisher uses, picking M = index 1. (Stage 2 needs its
    own GN fitter -- oed.fit_map_theta's physical-Adam pins the large-scale M target.)
    Returns DepthCalibResult(realized, predicted) -- a tuple so callers can unpack.
    """
    _M = 1  # M_dyn index in theta = (r_a, M, r_h)
    F = depth_fisher(z, m_lim, N_total)  # uses PRIOR_DIAG_M (M free) by default
    predicted = float(jnp.linalg.inv(F)[_M, _M])

    _, n_eff = _n_design_eff(
        z, m_lim, N_total
    )  # availability-capped observed counts (3, K)

    M_hats = []
    for d in range(n_draws):
        kdraw = jax.random.fold_in(key, d)
        sigma_hat, se = _binned_sigma_hat_selected(kdraw, m_lim, n_eff)
        M_hats.append(_fit_theta_gn(sigma_hat, se, PRIOR_DIAG_M)[_M])

    M_hats = jnp.asarray(M_hats)
    realized = float(jnp.var(M_hats, ddof=1) / oed.MOCK["M"] ** 2)
    return DepthCalibResult(realized=realized, predicted=predicted)
