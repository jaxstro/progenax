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
import jax
import jax.numpy as jnp

from jaxstro.units import STELLAR
from progenax import ChabrierIMF

import _demo_oed as oed
import _demo_selection as sel

# --- Selection constants (single cluster distance; bolometric magnitudes, no band/BC/extinction) ---
D_PC = oed.MOCK["d_kpc"] * 1000.0      # cluster distance: 4 kpc -> 4000 pc
M_MAX = 100.0                          # IMF upper mass [M_sun]
_IMF = ChabrierIMF(m_min=0.08, m_max=M_MAX)

# Per-channel Stage-1 errors (pc/Myr): [RV, PM_R, PM_T]. These are the errors AT the reference
# apparent magnitude M_REF; fainter stars are noisier via sel.photon_noise_error (Stage-1 tie-back).
EPS0 = oed.EPS                         # (3,) [pc/Myr]

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
    R = oed.R_BINS                                              # (K,) on-sky bin centres
    a = oed.MOCK["r_h"] * jnp.sqrt(2.0 ** (2.0 / 3.0) - 1.0)    # Plummer scale radius
    edges = oed._r_bin_edges()                                  # (K+1,) geometric-mean edges
    area = edges[1:] ** 2 - edges[:-1] ** 2                     # (K,) annular area (drop the 2pi)
    sigma = a ** 2 / (a ** 2 + R ** 2) ** 2                     # projected Plummer surface density (shape)
    pool = sigma * area                                         # per-bin intrinsic count (shape)
    return N_FIELD * pool / jnp.sum(pool)                       # normalise to N_FIELD


N_FIELD_BINS = _n_field_bins()                                 # (K,) module constant


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
    m_lo = sel.m_min(m_lim, D_PC)                               # differentiable lower edge [M_sun]
    frac = jnp.linspace(0.0, 1.0, _NGRID)
    m_grid = m_lo * (M_MAX / m_lo) ** frac                      # geometric grid, moving lower endpoint
    dP = _IMF.cdf(m_grid[1:]) - _IMF.cdf(m_grid[:-1])           # (_NGRID-1,) IMF probability mass/cell
    m_mid = jnp.sqrt(m_grid[1:] * m_grid[:-1])                  # geometric cell midpoints
    m_app = sel.apparent_mag(m_mid, D_PC)                       # (_NGRID-1,) apparent magnitudes
    norm = jnp.sum(dP)

    def rms(eps0_c):
        eps_c = sel.photon_noise_error(m_app, eps0_c, M_REF)    # per-star error in channel c
        return jnp.sqrt(jnp.sum(dP * eps_c ** 2) / norm)

    return jnp.array([rms(EPS0[0]), rms(EPS0[1]), rms(EPS0[2])])


def avail_bins(m_lim):
    """Per-bin available star pool (K,): intrinsic radial pool x global IMF-detectable fraction.

    ``N_FIELD_BINS * sel.detectable_fraction(m_lim, D_PC, _IMF)``. The detectable fraction is global
    (single cluster distance -> one apparent-magnitude map), so it multiplies the radial pool
    uniformly; the radial structure comes entirely from N_FIELD_BINS (core-rich, outskirt-poor).
    Monotone in m_lim (deeper -> larger detectable fraction -> more per bin).
    """
    return N_FIELD_BINS * sel.detectable_fraction(m_lim, D_PC, _IMF)


def depth_fisher(z, m_lim, N_total, prior_diag=oed.PRIOR_DIAG):
    """Additive design Fisher at limiting magnitude m_lim. Symmetric (3, 3), SPD (with prior_diag).

    The Stage-1 backbone: J computed ONCE (sigma_pred is m_lim-independent), the per-star blocks
    rebuilt at eps_eff(m_lim), and the softmax allocation n_design SMOOTHLY capped by availability
    via ``n_eff = avail * tanh(n_design / avail)`` (~n when n << avail, saturates at avail when
    n >> avail -- a differentiable finite-supply constraint). eps_eff (3,) broadcasts per channel;
    avail (K,) is shared across channels at a single distance. Differentiable in both z and m_lim.
    """
    J, sig = oed.jacobian_and_sigma(oed.theta_truth(), oed.R_BINS, STELLAR.G)   # J ONCE
    Mb = oed.blocks_from_eps(J, sig, eps_eff(m_lim))                            # (3,K,3,3), eps (3,)
    K = oed.R_BINS.shape[0]
    n_design = N_total * jax.nn.softmax(z).reshape(3, K)                        # (3,K) budget allocation
    avail = avail_bins(m_lim)[None, :]                                          # (1,K) per-channel pool
    n_eff = avail * jnp.tanh(n_design / avail)                                  # smooth availability cap
    return jnp.einsum("ck,ckpq->pq", n_eff, Mb) + jnp.diag(prior_diag)
