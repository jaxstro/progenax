r"""Core forward-model pieces for the binary-misspecification robustness OED arc.

Scripts-local (NOT a packaged API) helper module for the binary-misspecification
optimal-design demo (``scripts/demo_oed_binary.py``). It rides the Stage-1 additive-Fisher
backbone (``scripts/_demo_oed.py``) and the B12 binary blend kernel (``scripts/_demo_binaries.py``).

Forward model (RV-only, single-epoch). The observed line-of-sight velocity *variance*
in each on-sky radial bin ``R`` of a young massive cluster (YMC) with an unresolved
binary fraction ``f_bin`` is::

    sigma_obs^2(R) = sigma_cluster^2(R; M, r_a, gamma, a) + f_bin * V_bin + eps_RV^2

where:

* ``sigma_cluster(R)`` is the EFF-OM (Elson-Fall-Freeman density + Osipkov-Merritt
  anisotropy) projected line-of-sight dispersion, the RV channel of
  ``progenax.project_dispersion`` (Binney & Mamon 1982), converted pc/Myr -> km/s; and
* ``V_bin = Var(K_orb)`` is a build-once population scalar: the flux-weighted binary
  blend-velocity variance of the OBSERVED massive-primary population (Moe & Di Stefano
  2017 P-q-e orbits + Tout+1996 ZAMS flux weights), a ~flat radial pedestal.

``M`` scales the cluster amplitude, ``(r_a, gamma, a)`` set the radial shape, and
binaries add a flat offset -- radial leverage (core-vs-outskirts contrast) separates them.

This module is Phase 0 (Task 0.2): it PINS the two scales (``V_BIN``,
``sigma_cluster_ref``) and the fiducial EFF-OM YMC, and exposes the ``eff_profile`` helper
and theta accessors the later phases (the binary-inflated observable, the jacrev, the
f_bin Fisher block, the marginalize / maximin criteria) will build on.

JAX-native (``jax.numpy``); ``import progenax`` enables float64 before this module is used.

See docs/plans/2026-06-19-oed-binary-misspecification-{plan,design}.md.
"""
import functools
import math
import os
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from jaxstro.units import STELLAR
from progenax import EFFProfile, EFFVelocityDF, project_dispersion
from progenax.imf import Maschberger

# Scripts-local siblings (NOT a packaged API): the Stage-1 OED backbone (reused for the
# unit conversion + Fisher machinery in later phases) and the B12 binary blend kernel
# (the V_bin population scalar). Ensure scripts/ is importable regardless of how this
# module is imported (the test inserts it too; this is belt-and-braces).
sys.path.insert(0, os.path.dirname(__file__))
import _demo_binaries as binaries  # noqa: E402
import _demo_oed as oed  # noqa: E402  -- the Stage-1 c-criterion (matrix-generic) is reused

# pc/Myr -> km/s. project_dispersion returns velocities in sqrt(G M / length) units,
# which under STELLAR (Msun, pc, Myr) is pc/Myr. STELLAR.velocity_scale_km_s == 0.977792
# matches the Stage-1 demo constant exactly (KMS_PER_PC_PER_MYR in _demo_oed.py).
VEL_KMS_PER_PC_MYR = STELLAR.velocity_scale_km_s


def kms(v_pcMyr):
    """pc/Myr -> km/s (the native velocity unit of project_dispersion under STELLAR)."""
    return v_pcMyr * VEL_KMS_PER_PC_MYR


# ===========================================================================
# Fiducial young-massive cluster (PINNED in Phase 0 -- Task 0.2)
# ===========================================================================
#
# A compact young massive cluster (super-star-cluster / R136-class). EFF density
# (analytic -> ODE-free, OOM-safe for the cross-model MC) + Osipkov-Merritt anisotropy.
#
# Modeling judgment calls (Task 0.2):
#  * theta uses the EFF SCALE radius `a`, NOT a derived half-mass radius r_h. EFFProfile
#    is parameterized by (a, gamma, r_t) with no closed-form r_h(a, gamma, r_t), and the
#    design doc's `a(r_h, gamma)` mapping is not pinned. `a` IS the cluster's
#    concentration scale, so theta = (M, r_a, gamma, a) carries the same physics with no
#    spurious inversion. (The MyST page will state this; r_h is recoverable post-hoc.)
#  * gamma = 2.7: a young-cluster 3-D density slope (gamma=3 typical young clusters;
#    gamma=5 -> Plummer). The bonus concentration-bias parameter.
#  * a = 1.0 pc: a compact YMC core scale.
#  * r_t = 18 pc (= 18 a): EFFProfile needs a finite truncation for project_dispersion
#    (the EFF branch uses r_t as the outer quadrature limit). 18 a comfortably contains
#    the cluster (rho(r_t)/rho(0) ~ (r_t/a)^-gamma ~ 5e-4).
#  * r_a = 3.0 pc (= 3 a): mild Osipkov-Merritt radial anisotropy (a few x a).
#  * M = 4e5 Msun: TUNED so the CENTRAL (peak) sigma_los ~ 9 km/s, in the physical
#    8-12 km/s YMC band. sigma scales as sqrt(G M / length), so sigma_central ~ sqrt(M):
#    M=1e5 -> 4.49 km/s, M=4e5 -> ~8.98 km/s (verified). This is a genuine SSC mass.
#  * eps_RV = 1.0 km/s: a representative single-epoch RV precision for bright cluster
#    members (carried for the Fisher denominator at the H1 gate; not used in Phase 0).
#  * f_bin_truth = 0.5: the OBSERVED massive-primary binary fraction is high (Moe &
#    Di Stefano: ~70% companion frequency for O/B; 0.5 is a conservative truth).
#  * MASSIVE_M_MIN = 2.0 Msun: the observable RV-tracer floor -- bright B/O stars are
#    the spectroscopic tracers in an RV survey of a YMC. They follow Moe's
#    massive-primary statistics (high companion frequency, short periods -> large sigma_bin).
GAMMA_FID = 2.7        # EFF 3-D density slope (concentration knob)
A_FID = 1.0            # EFF scale radius [pc]
R_T_FID = 18.0         # EFF truncation radius [pc]
R_A_FID = 3.0          # Osipkov-Merritt anisotropy radius [pc]
M_FID = 4.0e5          # cluster mass [Msun] -> central sigma_los ~ 9 km/s
EPS_RV_KMS = 1.0       # single-epoch RV precision [km/s]
F_BIN_TRUTH = 0.5      # observed massive-primary binary fraction

MASSIVE_M_MIN = 2.0    # observable RV-tracer primary-mass floor [Msun]
MASSIVE_M_MAX = 100.0  # massive-primary upper mass [Msun]
MASSIVE_IMF_ALPHA = 2.3  # Maschberger high-mass slope (Salpeter-like)

# K=12 on-sky bin-centre radii, log-spaced from 0.2 a (deep core) to ~0.95 r_t (outskirts),
# mirroring _demo_oed.R_BINS' log-spaced style. Spans the full radial leverage (high-sigma
# core where M info lives -> low-sigma outskirts where the f_bin pedestal dominates).
R_BINS = jnp.logspace(jnp.log10(0.2 * A_FID), jnp.log10(0.95 * R_T_FID), 12)

# Build-once V_bin pool: the variance estimator's sample size. 40000 massive-primary Moe
# binaries give a kernel std stable to ~2% across seeds (seed sweep verified in Task 0.2).
V_BIN_N_POOL = 40000
V_BIN_SEED = 0         # fixed PRNG seed -> V_BIN is a deterministic build-once constant
V_BIN_Z = 1e-3         # metallicity for the ZAMS flux weighting


def massive_primary_imf():
    """The observable RV-tracer IMF: a Maschberger high-mass tail truncated to the
    bright B/O primaries (M1 >= MASSIVE_M_MIN) that dominate a YMC RV survey."""
    return Maschberger(
        alpha=MASSIVE_IMF_ALPHA, m_min=MASSIVE_M_MIN, m_max=MASSIVE_M_MAX
    )


# ---------------------------------------------------------------------------
# V_BIN: built ONCE at import (fixed key) -- the binary blend-velocity variance
# Var(K_orb) [(km/s)^2] of the OBSERVED massive-primary population. Never recomputed
# per call (enforce-jax-performance: build-once). f_bin scales it linearly downstream.
# ---------------------------------------------------------------------------
V_BIN = binaries.population_blend_variance(
    jax.random.PRNGKey(V_BIN_SEED),
    n_pool=V_BIN_N_POOL,
    imf=massive_primary_imf(),
    Z=V_BIN_Z,
)


def eff_profile(gamma=GAMMA_FID, a=A_FID, r_t=R_T_FID):
    """The fiducial EFF density profile (analytic -> ODE-free; finite r_t for projection).

    EFFProfile(a, gamma, r_t): rho(r) = (1 + r^2/a^2)^(-gamma/2) for r <= r_t. The
    cluster mass M is NOT a profile attribute -- it enters project_dispersion as the
    enclosed-mass normalization (the dispersion amplitude). Defaults are the pinned
    fiducials; later phases pass theta-driven (gamma, a).
    """
    return EFFProfile(a=a, gamma=gamma, r_t=r_t)


def cluster_sigma_los(theta_clusteronly, R, G):
    r"""EFF-OM RV-only projected line-of-sight dispersion per on-sky radial bin [km/s].

    The cluster forward model's RV channel: ``sigma_los(R)`` of the EFF density +
    Osipkov-Merritt anisotropy model at the cluster-only parameter vector
    ``theta = (M, r_a, gamma, a)``, via the Binney & Mamon (1982) projection of the
    Jeans model (``progenax.project_dispersion``), converted pc/Myr -> km/s.

    ``M`` scales the dispersion amplitude (``sigma ~ sqrt(G M / length)``), while
    ``(r_a, gamma, a)`` set the radial SHAPE. This is the single source of truth for
    the cluster term; the binary-inflated observable (Task 1.2) adds the flat
    ``f_bin * V_bin`` pedestal on top of ``cluster_sigma_los**2``.

    Parameters
    ----------
    theta_clusteronly : (4,) array  (M, r_a, gamma, a)
        The binary-free cluster parameter vector. The EFF truncation radius ``r_t``
        is held at the pinned fiducial ``R_T_FID`` (an outer quadrature limit, not a
        free design/inference parameter).
    R : (K,) array of on-sky bin-centre radii [pc].
    G : gravitational constant in the caller's unit system (use ``STELLAR.G``).

    Returns
    -------
    (K,) array of sigma_los [km/s].
    """
    prof = eff_profile(gamma=th_gamma(theta_clusteronly),
                       a=th_a(theta_clusteronly), r_t=R_T_FID)
    return kms(project_dispersion(prof, th_ra(theta_clusteronly), R,
                                  th_M(theta_clusteronly), G).sigma_los)


def sigma_cluster_ref(theta=None, R=None):
    r"""Central (peak) EFF-OM line-of-sight dispersion of the fiducial YMC [km/s].

    The CONSERVATIVE reference scale for the H1 sigma_bin/sigma_cluster ratio: the
    LARGEST sigma_los over the radial bins (the core). Binaries hurt fractionally MORE
    in the low-sigma outskirts, so if binaries rival the cluster at the central peak
    they rival it everywhere. A thin wrapper over ``cluster_sigma_los`` (the single
    source of truth for the cluster term) taking the per-bin maximum.

    Parameters
    ----------
    theta : optional (M, r_a, gamma, a)
        Override the fiducial theta (used in later-phase sweeps). Default -> fiducials.
    R : optional array of on-sky radii. Default -> R_BINS.
    """
    theta = theta_truth_clusteronly() if theta is None else theta
    R = R_BINS if R is None else R
    return jnp.max(cluster_sigma_los(theta, R, STELLAR.G))


# ===========================================================================
# theta accessors  (theta = (M, r_a, gamma, a, f_bin); cluster-only drops f_bin)
# ===========================================================================
#
# Parameter ordering with M first (the headline TARGET / dynamical mass); the
# kinematic nuisances (r_a, f_bin) and the photometrically-pinned shape params
# (gamma, a) follow. Index constants are exposed for the later-phase Fisher columns.
IDX_M = 0
IDX_RA = 1
IDX_GAMMA = 2
IDX_A = 3
IDX_FBIN = 4


def theta_truth_clusteronly():
    """Cluster-only truth theta = (M, r_a, gamma, a) -- the binary-free parameter vector."""
    return jnp.array([M_FID, R_A_FID, GAMMA_FID, A_FID])


def theta_truth():
    """Full truth theta = (M, r_a, gamma, a, f_bin) -- includes the binary fraction."""
    return jnp.array([M_FID, R_A_FID, GAMMA_FID, A_FID, F_BIN_TRUTH])


def th_M(theta):
    """Cluster mass M [Msun] from a theta vector (TARGET, index 0)."""
    return theta[IDX_M]


def th_ra(theta):
    """Osipkov-Merritt anisotropy radius r_a [pc] from a theta vector (index 1)."""
    return theta[IDX_RA]


def th_gamma(theta):
    """EFF density slope gamma from a theta vector (index 2)."""
    return theta[IDX_GAMMA]


def th_a(theta):
    """EFF scale radius a [pc] from a theta vector (index 3)."""
    return theta[IDX_A]


def th_fbin(theta):
    """Binary fraction f_bin from a full theta vector (index 4)."""
    return theta[IDX_FBIN]


# ===========================================================================
# Binary-inflated observable + the ONE reverse-mode jacrev (ln-theta metric)
# ===========================================================================
#
# The single-epoch RV observable is the binary-INFLATED line-of-sight dispersion:
# sigma_obs^2(R) = sigma_cluster^2(R; M, r_a, gamma, a) + f_bin * V_bin, where the
# cluster term carries the radial SHAPE and the binaries add a flat PEDESTAL. The
# eps_RV^2 measurement term is added at the Fisher denominator (Task 1.3), NOT here
# (this is the model prediction, not the noise model).
#
# The Fisher backbone (Stage-1, _demo_oed) needs J = d sigma_obs / d ln theta, ONE
# reverse-mode jacrev (ADR 0011: differentiating wrt ln theta makes F dimensionless
# and every F^-1 entry a FRACTIONAL variance; reverse-mode by policy -- King/Michie
# use custom_vjp ODEs with no jvp rule, so forward-mode would crash, and EFF is kept
# reverse-mode for consistency). jacrev of g(ln theta) = sigma_obs(exp(ln theta))
# returns d sigma_obs / d ln theta directly (the chain rule's exp(ln theta) = theta
# factor IS the ln-theta scaling), so no separate "* theta" step is needed.


def predict_sigma_obs(theta_full, R, G):
    r"""Binary-inflated observable sigma_obs(R) [km/s], the single-epoch RV model.

    ``sigma_obs^2(R) = cluster_sigma_los(M, r_a, gamma, a)^2 + f_bin * V_bin``. The
    cluster term sets the radial shape and amplitude; ``f_bin * V_bin`` is the flat
    binary-blend pedestal (``V_bin`` a build-once population variance, (km/s)^2). The
    measurement error ``eps_RV`` is added at the Fisher denominator (Task 1.3), not
    in the model prediction. Returns sigma_obs (sqrt of the variance), (K,) km/s.
    """
    sig2_cluster = cluster_sigma_los(theta_full[:4], R, G) ** 2
    return jnp.sqrt(sig2_cluster + th_fbin(theta_full) * V_BIN)


def jacobian_lntheta(theta_full, R, G):
    r"""ONE reverse-mode jacrev: J = d sigma_obs / d ln theta, shape (K, 5).

    The full (5-parameter, theta = (M, r_a, gamma, a, f_bin)) sensitivity matrix in
    the dimensionless ln-theta metric (ADR 0011). Implemented as
    ``jax.jacrev`` of ``lambda lnth: predict_sigma_obs(exp(lnth), R, G)`` evaluated at
    ``log(theta_full)`` -- the ``d/d ln theta = theta * d/d theta`` scaling is built in
    by the ``exp`` reparametrisation, so no separate ``* theta`` step. Reverse-mode by
    policy. The ``f_bin`` column is ``f_bin * V_bin / (2 sigma_obs)``: it GROWS toward
    the outskirts (where ``sigma_obs`` falls), concentrating binary-fraction
    information in the cold outer bins.
    """
    def g(lnth):
        return predict_sigma_obs(jnp.exp(lnth), R, G)         # (K,)
    return jax.jacrev(g)(jnp.log(theta_full))                 # (K, 5)


def jacobian_lntheta_clusteronly(theta_clusteronly, R, G):
    r"""ONE reverse-mode jacrev for the BINARY-FREE model: J = d sigma_los / d ln theta,
    shape (K, 4), theta = (M, r_a, gamma, a).

    The binary-free analogue of ``jacobian_lntheta`` (no f_bin column): jacrev of
    ``lambda lnth: cluster_sigma_los(exp(lnth), R, G)`` at ``log(theta_clusteronly)``.
    Equals the first four columns of the full jacrev at the truth (the f_bin pedestal
    is a pure additive offset that does not couple into the cluster sensitivities).
    This is the matrix Task 1.3 caches ONCE at the truth for the binary-free Fisher.
    """
    def g(lnth):
        return cluster_sigma_los(jnp.exp(lnth), R, G)         # (K,)
    return jax.jacrev(g)(jnp.log(theta_clusteronly))          # (K, 4)


# ===========================================================================
# Task 1.3: additive Fisher (RV-only, single channel) + binary-free
#           c-optimal-for-M design
# ===========================================================================
#
# The binary-free model drops f_bin: theta = (M, r_a, gamma, a). The design is a
# per-radial-bin allocation of N_total RV measurements over the K=12 bins (a single
# RV channel -- no PM channel in this RV-only arc). The Fisher is the Stage-1
# additive form, restricted to one channel:
#
#   F = Sum_b n_b * M_b + diag(PRIOR_DIAG_BF),   M_b = 2 * outer(J_b, J_b) / (sigma_b^2 + eps_RV^2)
#
# with n_b = N_total * softmax(z) the per-bin counts, J_b = d sigma_cluster / d ln theta
# (row b of the cached binary-free jacrev), sigma_b the cached truth sigma_los. In the
# dimensionless ln-theta metric (ADR 0011) every (F^-1) entry is a FRACTIONAL variance,
# so c_criterion(F)[M, M] = [sigma(M)/M]^2 directly.

# Total RV-measurement budget allocated across the K bins. A round number on the scale
# of a deep YMC RV survey (thousands of resolved bright members); the OED conclusions
# are scale-free in N_total (F is linear in it), so this is just a reporting anchor.
N_TOTAL = 5000.0

# Priors on the binary-free theta = (M, r_a, gamma, a), as ln-theta FRACTIONAL
# precisions 1/sigma_frac^2 on the diagonal (ADR 0011). Choices (design doc
# "Priors / degeneracy structure"):
#  * M (idx 0)  = 0      -- the TARGET dynamical mass: NO prior. You do not constrain
#                           the very quantity you are designing to measure; M must be
#                           pinned by the kinematic RV design alone.
#  * r_a (idx 1) = 1/0.5^2 -- a WEAK prior (50% fractional). The OM anisotropy radius is
#                           a kinematic nuisance with only loose external constraints;
#                           weak regularization keeps F conditioned without dictating r_a.
#  * gamma (idx 2) = 1/0.1^2, a (idx 3) = 1/0.1^2 -- TIGHT photometric priors (10%
#                           fractional). The EFF density slope and scale radius are
#                           measured precisely from the surface-brightness profile, so
#                           they enter the kinematic fit as well-pinned shape parameters.
_FRAC_RA = 0.5    # weak prior on the anisotropy nuisance
_FRAC_PHOT = 0.1  # tight photometric prior on the shape params (gamma, a)
PRIOR_DIAG_BF = jnp.array([0.0, 1.0 / _FRAC_RA**2, 1.0 / _FRAC_PHOT**2, 1.0 / _FRAC_PHOT**2])

# eps_RV in the native pc/Myr unit of the cached sigma (so the Fisher denominator
# sigma^2 + eps^2 is dimensionally consistent before the km/s conversion). Equivalently
# EPS_RV_KMS / VEL_KMS_PER_PC_MYR; we keep everything in km/s instead (sigma cached in
# km/s), so eps enters as EPS_RV_KMS directly. See _SIG_BF below.

# ---------------------------------------------------------------------------
# Build-once caches at the TRUTH (design-INDEPENDENT -- enforce-jax-performance).
# The binary-free jacrev (the single expensive jacrev through project_dispersion) and
# the truth sigma_los are computed ONCE here at import, NOT inside the optimizer loop.
# The optimizer evaluates the Fisher thousands of times over the design weights z; both
# _J_BF and _SIG_BF are constant wrt z (theta is the fixed truth), so caching them as
# module constants leaves every design gradient bit-identical while avoiding a re-jacrev
# per step (mirrors _demo_oed_depth._J/_SIG). Both in km/s (sigma) and dimensionless
# (J = d sigma / d ln theta) -- consistent with EPS_RV_KMS.
_J_BF = jacobian_lntheta_clusteronly(theta_truth_clusteronly(), R_BINS, STELLAR.G)  # (K, 4)
_SIG_BF = cluster_sigma_los(theta_truth_clusteronly(), R_BINS, STELLAR.G)           # (K,) km/s


def uniform_design():
    """The uniform (equal-weight) design logits: a length-K zero vector, so
    softmax gives an equal 1/K allocation to every radial bin (the no-OED baseline)."""
    return jnp.zeros(R_BINS.shape[0])


def c_criterion_M(F):
    """c-optimality for the dynamical mass M (binary-free theta index IDX_M = 0):
    the (M, M) entry of F^-1 in the ln-theta metric = [sigma(M)/M]^2. Thin wrapper
    over the Stage-1 matrix-generic oed.c_criterion(F, target=IDX_M)."""
    return oed.c_criterion(F, target=IDX_M)


def fisher_binary_free(design_weights, N_total, J=None, sig=None, prior_diag=None):
    r"""Additive single-channel RV Fisher for the binary-free theta = (M, r_a, gamma, a).

    ``F = Sum_b n_b * M_b + diag(prior_diag)`` with per-bin counts
    ``n_b = N_total * softmax(design_weights)`` and the rank-1 per-bin block
    ``M_b = 2 * outer(J_b, J_b) / (sigma_b^2 + eps_RV^2)`` (the Gaussian-dispersion
    Fisher: a dispersion from n stars has variance (sigma^2+eps^2)/(2n)). ``J`` and
    ``sig`` default to the build-once truth caches ``_J_BF`` / ``_SIG_BF`` (km/s);
    ``prior_diag`` defaults to ``PRIOR_DIAG_BF``. Symmetric (4, 4); SPD with the prior.
    Differentiable in ``design_weights`` (pure softmax + linear algebra -- no re-jacrev).
    """
    J = _J_BF if J is None else J
    sig = _SIG_BF if sig is None else sig
    prior_diag = PRIOR_DIAG_BF if prior_diag is None else prior_diag
    n_b = N_total * jax.nn.softmax(design_weights)                  # (K,) per-bin counts
    denom = sig**2 + EPS_RV_KMS**2                                  # (K,) Gaussian-dispersion denom
    M_b = 2.0 * jnp.einsum("kp,kq->kpq", J, J) / denom[:, None, None]  # (K, 4, 4) per-bin blocks
    F = jnp.einsum("k,kpq->pq", n_b, M_b)                           # additive design Fisher
    return F + jnp.diag(prior_diag)


class DesignResultM(NamedTuple):
    """Result of optimize_design_M (the binary-free c-optimal-for-M design):
      * n_eff           : optimal per-bin RV counts (K,), summing to N_total,
      * sigma_M_over_M  : the c-optimal fractional precision sqrt((F^-1)[M, M]) (ln metric),
      * z               : the optimal design logits (K,),
      * trace           : the per-step criterion trace of the winning Adam start."""
    n_eff: jnp.ndarray
    sigma_M_over_M: float
    z: jnp.ndarray
    trace: jnp.ndarray


def _optimize_one_M(z0, N_total, n_steps, lr):
    """One Adam trajectory minimizing c_criterion_M(fisher_binary_free(z, N_total)).

    Fixed-iteration jax.lax.scan (NOT while_loop) over the length-K design logits z;
    the step is jit-compiled. The Fisher is pure linear algebra over the cached
    _J_BF/_SIG_BF (no re-jacrev), so each step is cheap. Returns (z_final, trace)."""
    opt = optax.adam(lr)
    state = opt.init(z0)
    loss = lambda z: c_criterion_M(fisher_binary_free(z, N_total))

    @jax.jit
    def step(carry, _):
        z, st = carry
        l, g = jax.value_and_grad(loss)(z)
        upd, st = opt.update(g, st)
        return (optax.apply_updates(z, upd), st), l

    (z, _), trace = jax.lax.scan(step, (z0, state), None, length=n_steps)
    return z, trace


def optimize_design_M(N_total, key, n_starts=8, n_steps=500, lr=0.05):
    """Multi-start Adam for the binary-free c-optimal-for-M radial RV design.

    Minimizes ``c_criterion_M(fisher_binary_free(z, N_total))`` -- the marginal
    fractional variance [sigma(M)/M]^2 -- over the length-K design logits z, running
    ``n_starts`` independent Adam trajectories (the simplex landscape can be multimodal)
    and keeping the lowest-criterion result. Mirrors the Stage-1 multi-start pattern
    (oed.optimize_design) for the single-channel 4-parameter case.

    SPD invariant: softmax(z) > 0 for every finite z (every bin populated) and
    PRIOR_DIAG_BF adds strictly positive precision on the (r_a, gamma, a) nuisance
    subspace, so the regularized F stays SPD throughout and c_criterion_M's inverse
    never hits a singular F. Returns a DesignResultM.
    """
    K = R_BINS.shape[0]
    best = None
    for s in range(n_starts):
        z0 = jax.random.normal(jax.random.fold_in(key, s), (K,)) * 0.5
        z, trace = _optimize_one_M(z0, N_total, n_steps, lr)
        crit = float(c_criterion_M(fisher_binary_free(z, N_total)))
        if math.isfinite(crit) and (best is None or crit < best[0]):
            best = (crit, z, trace)
    crit, z, trace = best
    n_eff = N_total * jax.nn.softmax(z)
    return DesignResultM(n_eff=n_eff, sigma_M_over_M=float(jnp.sqrt(crit)), z=z, trace=trace)


# ===========================================================================
# Task 1.4: cross-model bias harness (the H1 headline machinery)
# ===========================================================================
#
# The discriminating MC (pre-registration LOCKED 2026-06-19): GENERATE mocks UNDER
# the binary model (EFF-OM cluster + a Moe-binary RV pedestal), then FIT the
# BINARY-FREE model (f_bin == 0 -- the misspecification) and collect M_hat. If the
# binary-free design walks into a biased M_hat -- larger than its own forecast sigma --
# H1 bites.
#
# PERFORMANCE (enforce-jax-performance), mirroring _demo_oed_concentration:
#  * BUILD-ONCE (per truth, before the draw loop): the EFF profile + EFFVelocityDF
#    Eddington table (the expensive sampler structure -- identical every draw, same
#    truth), and a large K_orb blend-velocity POOL (Moe massive-primary Delta samples
#    drawn ONCE; per-draw injections are a cheap uniform resample of this pool). Neither
#    is rebuilt inside the loop.
#  * MC via jax.lax.map (SEQUENTIAL scan) over the draw keys -- one_draw compiles ONCE
#    yet runs draw-by-draw, so peak memory is a SINGLE draw's GN-fit reverse-mode tape,
#    NOT n_draws of them. (EFF density is analytic -> no ODE tape, but the binary-free
#    GN fit's per-iter jacrev through project_dispersion's quadrature is the heavy part;
#    lax.map keeps it memory-bounded.)
#  * the per-draw fn is jit (static n_iter); the GN fit's jacrev is reverse-mode.

# Parent-catalog size for the cross-model MC. The c-optimal-for-M design concentrates
# counts into a FEW radial bins -- including the sparsely-populated outer EFF tail (R near
# 0.95 r_t), where the steeply-declining EFF density puts only ~6% of the catalog. The
# parent catalog must hold MORE stars in every bin than that bin's design cell needs (a
# real survey cannot observe more stars than exist), so n_parent is sized from BOTH the
# design (its largest per-bin count) and the EFF bin-occupancy profile (the sparsest bin
# that carries a non-trivial cell), with a safety factor. _static_cell_sizes_1d still runs
# the hard parent-too-small guard as a backstop. Mirrors the concentration demo's "size the
# parent so no real cell underflows" sizing, made design-adaptive because our outer bins
# are deep in the EFF tail.
_N_PARENT_FLOOR = 8000      # floor: enough for a stable ddof=1 sigma_hat in every bin
_N_PARENT_SAFETY = 1.6      # >1: hold comfortably more parent stars than any design cell needs


def _bin_fractions(prof, edges, n_probe=80_000, seed=7):
    """Empirical per-bin occupancy FRACTIONS of the EFF parent density (build-time probe).

    Samples a large probe catalog ONCE to estimate what fraction of stars fall in each
    R bin (the EFF tail bins are sparse: ~6% in the outermost). Used to size n_parent so
    the design's per-bin counts fit the (sparsely-populated) outer bins. Host-side
    bookkeeping (numpy digitize over a JAX-sampled catalog).
    """
    import numpy as np

    masses = jnp.full((n_probe,), M_FID / n_probe)
    pos = prof.sample_positions(masses, jax.random.PRNGKey(seed))
    R = jnp.hypot(pos[:, 0], pos[:, 1])
    K = R_BINS.shape[0]
    binidx = np.digitize(np.asarray(R), np.asarray(edges)) - 1
    frac = np.array([np.mean(binidx == b) for b in range(K)])
    return frac


def _n_parent_for_design(design_n_eff, prof, edges):
    """Parent-catalog size so every bin holds >= its design cell (+safety), given the EFF
    occupancy profile. n_parent = max(floor, safety * max_b n_eff[b] / frac[b]); the worst
    case is the sparse-but-heavily-allocated outer tail bin. Guards frac>0."""
    import numpy as np

    frac = _bin_fractions(prof, edges)
    n_eff = np.asarray(design_n_eff)
    frac_safe = np.maximum(frac, 1e-3)               # never divide by an empty bin estimate
    need = _N_PARENT_SAFETY * np.max(n_eff / frac_safe)
    return int(max(_N_PARENT_FLOOR, math.ceil(need)))

# K_orb injection pool size: a fixed pool of Moe massive-primary blend velocities Delta
# [km/s]. Per draw, each binary-contaminated star draws one Delta uniformly (with
# replacement) from this pool, so the realized injection variance -> Var(pool) -> V_BIN
# as the pool grows (the same Moe massive-primary population that defines V_BIN).
#
# SIZE TRADE (perf, measured): the per-star gather korb_pool[idx] of n_parent indices,
# fused in the lax.map body, has an XLA-CPU compile workspace that scales with the POOL
# size: pool 2e5 -> 9 GB peak RSS, 6.5e4 -> 4.6 GB, 1.6e4 -> 2.9 GB. 16384 keeps the @slow
# MC comfortably under the 4 GB OOM gate while still sampling Var(K_orb) to <~1% (16k
# Moe draws; verified Var(pool) vs V_BIN). The bias is statistically unchanged from the
# 2e5-pool value (the injected pedestal variance is set by the pool VARIANCE, not its size).
_KORB_POOL_N = 16384

# Levenberg-Marquardt MAP-fit settings for the binary-free GN fit (mirrors the
# concentration demo's _GN_LM_LAM0 / _GN_N_ITER). The fit has P=4 params (M, r_a, gamma,
# a); M and the photometrically-pinned (gamma, a) are well-constrained, r_a is the weak
# nuisance the lam*I floor keeps bounded. 40 iters settles the TARGET M (the M-witness)
# for all draws; the fit is cheap (it does NOT drive the peak RSS -- the EFF sampler +
# K_orb gather do) so a generous iteration count is free.
_BF_LM_LAM0 = 1e-2
_BF_N_ITER = 40


def _r_bin_edges_1d():
    """K+1 geometric-mean bin edges bracketing the K log-spaced R_BINS centres.

    Local reimplementation against THIS module's R_BINS (the coupling trap documented in
    _demo_oed_concentration: the Stage-1 / concentration helpers read their OWN module-global
    R_BINS, whose VALUES differ from ours -- importing them would bin against the wrong radii).
    R_BINS is log-uniform, so the edges are the centres shifted by +-dlog/2 in log space.
    """
    lc = jnp.log(R_BINS)
    dlog = lc[1] - lc[0]
    return jnp.exp(jnp.concatenate([lc[:1] - dlog / 2.0, lc + dlog / 2.0]))


def _bin_of_1d(R, edges):
    """Per-star bin index (n,) in 0..K-1; out-of-range stars get the sentinel bin K."""
    K = R_BINS.shape[0]
    b = jnp.searchsorted(edges, R, side="right") - 1
    return jnp.where((b < 0) | (b >= K), K, b)


def _static_cell_sizes_1d(n_eff, R, edges):
    """Host-side STATIC per-cell sizes for the single-channel JAX binning.

    Returns (p_keep (K,), n_se (K,), members (K,)):

      * ``p_keep[b] = min(1, n_use[b] / members[b])`` -- the per-bin Bernoulli inclusion
        probability that thins the parent catalog to ~n_use[b] = max(round(n_eff[b]),
        _MIN_CELL) selected stars in bin b (the survey selection function; see
        _binned_sigma_hat_1d for why Bernoulli, not an exact-count sort);
      * ``n_se[b] = max(n_eff[b], 1e-6)`` -- the TRUE design count for the SE weight, NOT
        floored (the fit's per-cell weight 1/se^2 must equal the design Fisher's
        2 n_eff/(sigma^2+eps^2), the concentration-demo lesson);
      * ``members[b]`` -- the parent-catalog occupancy of bin b (reported for the guard).

    Runs the parent-catalog-too-small guard on the host (the JAX binning cannot raise):
    if any bin needs more selected stars than fell in it, raise a clear error. n_eff is
    (K,) for the single RV channel.
    """
    import numpy as np  # host-side bookkeeping only; never numpy.random

    K = R_BINS.shape[0]
    bin_of = np.digitize(np.asarray(R), np.asarray(edges)) - 1
    members = np.array([int(np.sum(bin_of == b)) for b in range(K)])
    n_eff_np = np.asarray(n_eff)
    n_use = np.maximum(np.round(n_eff_np).astype(int), oed._MIN_CELL)   # (K,)
    for b in range(K):
        if n_use[b] > members[b]:
            raise ValueError(
                f"cross-model parent catalog too small: bin {b} needs {n_use[b]} stars "
                f"but only {members[b]} fell in the bin; increase n_parent."
            )
    members_safe = np.maximum(members, 1)
    p_keep = np.minimum(n_use / members_safe, 1.0)
    n_se = np.maximum(n_eff_np, 1e-6)
    return jnp.asarray(p_keep), jnp.asarray(n_se), jnp.asarray(members)


def _binned_sigma_hat_1d(key, v_los, R, p_keep, n_se, edges):
    """vmappable single-channel binned sigma_hat (K,) + se (K,) from the LOS velocities.

    PURE array function (no host control flow), so lax.map batches it across draws.

    Subsampling = per-bin BERNOULLI THINNING (NOT an exact-count without-replacement
    sort). Each parent star in bin b is independently kept with probability ``p_keep[b]``
    (precomputed on the host so E[selected] = n_use[b]), then we add per-star Gaussian RV
    error ~Normal(0, EPS_RV_KMS) and take the masked ddof=1 sample std of the kept stars.
    The SE is sigma_hat / sqrt(2 n_se) at the TRUE design count (matches the Fisher weight).

    WHY BERNOULLI, NOT THE EXACT-COUNT SORT (a deliberate perf+honesty choice): the
    concentration demo's _within_bin_rank does a GLOBAL jnp.argsort over the parent
    catalog to take exactly n_use lowest-priority members per bin. Fused into the lax.map
    body alongside the EFF sampler + the GN-fit jacrev, that sort balloons the XLA-CPU
    workspace to ~8 GB (measured: argsort fused = 7.8 GB; Bernoulli = 2.4 GB -- a >3x cut
    that brings the @slow MC under the OOM gate). Bernoulli thinning is sort-free and is
    ALSO the more faithful survey-selection model (a real RV survey draws each tracer
    independently, not an exact quota). It preserves the science exactly for the bias: the
    ddof=1 sample std is an UNBIASED dispersion estimate on whatever subset is kept (the
    central sigma_hat does not depend on the exact count), and the SE weight uses the
    design count n_se regardless -- so the realized bias and its forecast stay
    apples-to-apples. The only difference is the per-bin count is ~Binomial(members,
    p_keep) rather than fixed; with hundreds-to-thousands of stars per heavy bin the
    +-sqrt(n_use) jitter is negligible.

    NOTE: v_los is the ALREADY-binary-CONTAMINATED LOS velocity (the cluster COM velocity
    plus, for the binary subset, the injected K_orb blend Delta); eps is the measurement
    noise added on top here.
    """
    K = R_BINS.shape[0]
    n = R.shape[0]
    bin_of = _bin_of_1d(R, edges)                            # (n,) 0..K-1, K = sentinel
    cell = jnp.arange(K)

    kp, kn = jax.random.split(key)
    # Per-star keep probability = its bin's p_keep (sentinel bin K -> 0 -> never kept).
    p_star = jnp.where(bin_of < K, p_keep[jnp.minimum(bin_of, K - 1)], 0.0)
    selected = jax.random.uniform(kp, (n,)) < p_star        # (n,) Bernoulli mask
    v_obs = v_los + EPS_RV_KMS * jax.random.normal(kn, (n,))  # (n,) + measurement noise

    in_bin = bin_of[None, :] == cell[:, None]               # (K, n)
    w = (in_bin & selected[None, :]).astype(jnp.float64)    # (K, n) 0/1 weights
    cnt = jnp.sum(w, axis=1)                                 # (K,) ~ Binomial(members, p_keep)
    mean = jnp.sum(w * v_obs[None, :], axis=1) / jnp.maximum(cnt, 1.0)
    var = jnp.sum(w * (v_obs[None, :] - mean[:, None]) ** 2, axis=1) / jnp.maximum(cnt - 1.0, 1.0)
    sigma_hat = jnp.sqrt(var)                                # (K,) ddof=1
    se = sigma_hat / jnp.sqrt(2.0 * n_se)                    # (K,) at the true design count
    return sigma_hat, se


@functools.partial(jax.jit, static_argnames=("n_iter",))
def _fit_theta_bf_gn(sigma_hat, se, G, n_iter=_BF_N_ITER):
    r"""Levenberg-Marquardt MAP fit of the BINARY-FREE theta = (M, r_a, gamma, a) in the
    dimensionless ln-theta metric, started at the cluster-only truth.

    The misspecified fit: the model is cluster_sigma_los (f_bin == 0, NO binary pedestal),
    fit to the binary-CONTAMINATED binned dispersions. We fit u = ln(theta) - ln(theta_fid),
    theta = theta_fid * exp(u) (every direction O(1); M~4e5, r_a/gamma/a are O(1), so a
    single-LR physical step cannot move M -- the Stage-2 lesson). Each iteration solves the
    DAMPED Gauss-Newton system
      (Jr^T Jr + diag(PRIOR_DIAG_BF) + lam I) du = -(Jr^T r + PRIOR_DIAG_BF u)
    with r = (model - data)/se the whitened residual and Jr = d r / d u (REVERSE-mode jacrev
    through project_dispersion's quadrature; by policy), ACCEPTS the step only if it lowers
    the MAP cost 0.5(||r||^2 + sum PRIOR_DIAG_BF u^2), and adapts lam (x0.3 accept, x3 reject).
    The prior is PRIOR_DIAG_BF (M free, r_a weak, gamma/a tight -- the SAME prior the
    forecast Fisher uses, so the cross-model bias is apples-to-apples with sigma_forecast).

    Returns (theta_hat (4,), m_witness) where m_witness = max |M-component step| over the
    LAST 5 iterations (the TARGET-M convergence witness; ~0 means M_hat has settled).
    jit (static n_iter): sigma_hat/se/G are traced, so the project_dispersion-jacrev scan
    compiles ONCE and is reused across every draw. Value-preserving.
    """
    theta_fid = theta_truth_clusteronly()                   # (M, r_a, gamma, a)

    def resid(u):                                           # whitened residual (model - data)/se
        theta = theta_fid * jnp.exp(u)
        return (cluster_sigma_los(theta, R_BINS, G) - sigma_hat) / se

    def cost_of(u, r):
        return 0.5 * (r @ r + jnp.sum(PRIOR_DIAG_BF * u**2))

    def lm_step(carry, _):
        u, lam = carry
        r = resid(u)
        c = cost_of(u, r)
        Jr = jax.jacrev(resid)(u)                           # (K, 4) = d r / d u (reverse-mode)
        grad = Jr.T @ r + PRIOR_DIAG_BF * u
        hess = Jr.T @ Jr + jnp.diag(PRIOR_DIAG_BF)
        du = -jnp.linalg.solve(hess + lam * jnp.eye(4), grad)
        u_try = u + du
        c_try = cost_of(u_try, resid(u_try))
        improved = c_try < c                               # NaN -> False -> reject
        u_next = jnp.where(improved, u_try, u)
        lam_next = jnp.clip(jnp.where(improved, lam * 0.3, lam * 3.0), 1e-9, 1e9)
        m_moved = jnp.abs((u_next - u)[IDX_M])             # TARGET-M step witness
        return (u_next, lam_next), m_moved

    (u_hat, _), m_steps = jax.lax.scan(lm_step, (jnp.zeros(4), _BF_LM_LAM0), None, length=n_iter)
    return theta_fid * jnp.exp(u_hat), jnp.max(m_steps[-5:])


# B2-style convergence threshold on each fit's M-target witness (max |M-step| over the
# last 5 LM iters, ln-theta). A settled fit moves M below this; a draw above it has a
# still-moving M_hat and is SURFACED (not swallowed).
_BF_CONVERGED_STEP = 1e-3


class CrossModelResult(NamedTuple):
    """Result of cross_model_bias (the H1 headline statistic; bias/std are FRACTIONAL,
    i.e. relative to the truth M, in the ln-theta spirit of ADR 0011):

      * bias_M_frac   : mean over draws of (M_hat - M_true) / M_true,
      * std_M_frac    : std (ddof=1) over draws of (M_hat - M_true) / M_true,
      * sem_M_frac    : standard error of the mean bias = std_M_frac / sqrt(n_draws),
      * mhat_mean     : mean M_hat [Msun] (so M_FID*(1+bias_M_frac)),
      * bias_other    : mean fractional bias of (r_a, gamma, a) -- is the pedestal absorbed
                        into M, or partly into the nuisances?  (3,) array in theta order
                        (r_a, gamma, a),
      * n_unconverged : # draws whose M-target witness exceeded _BF_CONVERGED_STEP,
      * max_M_step    : max over draws of the M-target witness."""
    bias_M_frac: float
    std_M_frac: float
    sem_M_frac: float
    mhat_mean: float
    bias_other: jnp.ndarray
    n_unconverged: int
    max_M_step: float


def cross_model_bias(design_n_eff, n_draws, key, n_iter=_BF_N_ITER):
    r"""Cross-model MC: generate UNDER the binary model, fit the BINARY-FREE model.

    The headline H1 machinery. For each of n_draws independent mocks (via SEQUENTIAL
    jax.lax.map):

      1. sample n_parent EFF particles at the fiducial cluster truth (positions +
         OM velocities; total mass M_FID so the speed scale matches project_dispersion);
      2. project to the sky (LOS = +z): R = hypot(x, y), v_los = v_z [km/s];
      3. BINARY CONTAMINATION: each star is a binary-contaminated tracer with probability
         f_bin = F_BIN_TRUTH (per-star Bernoulli); each contaminated star has one blend
         velocity Delta (drawn uniformly with replacement from the build-once K_orb pool)
         ADDED to its v_los. So E[Var added] = f_bin * V_BIN -- the SAME flat pedestal the
         predict_sigma_obs model encodes;
      4. bin into R_BINS, subsample the design counts `design_n_eff` per bin (without
         replacement), add per-star eps_RV measurement noise -> per-bin sigma_hat + SE;
      5. FIT the BINARY-FREE theta = (M, r_a, gamma, a) (NO binary pedestal -- the
         misspecification) by LM-damped GN MAP in ln-theta with PRIOR_DIAG_BF;
      6. collect M_hat (and the nuisance estimates).

    Returns a CrossModelResult: mean fractional bias (M_hat - M)/M, its std + SEM, the
    nuisance biases, and the M-target convergence diagnostics.

    Build-once (enforce-jax-performance): the EFF profile + EFFVelocityDF Eddington table
    (sampler, identical every draw -- same truth) and the K_orb blend-velocity pool are
    built ONCE here, before the draw loop, and threaded into the jit'd per-draw body. The
    velocities of stars (sampler) and the binary Delta pool are NEVER recomputed per draw.

    Parameters
    ----------
    design_n_eff : (K,) array
        Per-bin RV counts (the design under test; from optimize_design_M(...).n_eff).
    n_draws : int
        Number of independent cross-model mock draws (the MC sample size).
    key : PRNGKey
    n_iter : int
        LM iterations of the binary-free MAP fit (static; default _BF_N_ITER).
    """
    G = STELLAR.G
    K = R_BINS.shape[0]
    edges = _r_bin_edges_1d()

    # --- BUILD-ONCE (per truth, before the draw loop) ---------------------------------
    # EFF profile + OM Eddington-table sampler at the fiducial truth (the expensive,
    # draw-invariant sampler structure). n_parent is sized from the design + the EFF
    # bin-occupancy profile so the sparse outer tail bins still hold their design cells.
    # Per-star mass label M_FID/n_parent so the sampled total mass == M_FID (the EFF speed
    # scale sqrt(G M / (4 pi mu)) is then at total mass M_FID, matching project_dispersion's
    # M in cluster_sigma_los).
    prof = eff_profile(gamma=GAMMA_FID, a=A_FID, r_t=R_T_FID)
    df = EFFVelocityDF(a=A_FID, gamma=GAMMA_FID, r_t=R_T_FID, anisotropy_radius=R_A_FID)
    n_parent = _n_parent_for_design(design_n_eff, prof, edges)
    masses = jnp.full((n_parent,), M_FID / n_parent)

    # K_orb blend-velocity POOL (Moe massive-primary Delta [km/s]) -- the SAME population
    # that defines V_BIN (massive_primary_imf, V_BIN_Z), drawn ONCE. Per-draw injections
    # resample this pool, so the injected pedestal variance -> Var(pool) (no per-draw Moe
    # orbit integration -- enforce-jax-performance). Delta is ~zero-mean (random phase +
    # isotropic LOS), so we recentre then RESCALE the pool to EXACTLY Var = V_BIN: the
    # K_orb distribution is heavy-tailed (short-period binaries -> large Delta), so a
    # finite pool's sample variance scatters ~10% from V_BIN; matching the second moment
    # makes the INJECTED pedestal exactly the V_BIN the forecast Fisher uses (apples-to-
    # apples for the H1 bias-vs-forecast test). This is principled: the forward model is a
    # SECOND-MOMENT model (design caveat 2), so fixing Var(injection) = V_BIN is the
    # faithful injection, not a fudge.
    korb_raw = jnp.asarray(
        binaries.sample_blend_velocities(
            jax.random.PRNGKey(V_BIN_SEED + 1), _KORB_POOL_N,
            imf=massive_primary_imf(), Z=V_BIN_Z,
        )
    )
    korb_centered = korb_raw - jnp.mean(korb_raw)
    korb_pool = korb_centered * jnp.sqrt(V_BIN / jnp.var(korb_centered, ddof=1))

    # Static per-cell sizes (Bernoulli keep-probabilities + SE weights) + the host-side
    # parent-catalog-too-small guard, computed ONCE from the draw-independent design count
    # on a representative draw's R (p_keep/n_se do not depend on the draw's specific R).
    k0 = jax.random.fold_in(key, 0)
    pos0 = prof.sample_positions(masses, jax.random.fold_in(k0, 7))
    R0 = jnp.hypot(pos0[:, 0], pos0[:, 1])
    p_keep, n_se, _members = _static_cell_sizes_1d(design_n_eff, R0, edges)

    def one_draw(kdraw):
        """One cross-model mock -> ((M_hat - M)/M, nuisance fractional biases, M-witness)."""
        k_pos, k_vel, k_inj, k_bin = jax.random.split(kdraw, 4)
        # 1-2. EFF particles + sky projection (LOS = +z).
        pos = prof.sample_positions(masses, k_pos)
        vel = df.sample_velocities(pos, masses, k_vel, G=G)
        R = jnp.hypot(pos[:, 0], pos[:, 1])
        v_los = kms(vel[:, 2])                              # pc/Myr -> km/s
        # 3. Binary contamination: per-star Bernoulli(f_bin), add a pooled Delta to the hits.
        k_mask, k_pick = jax.random.split(k_inj)
        is_binary = jax.random.uniform(k_mask, (n_parent,)) < F_BIN_TRUTH
        idx = jax.random.randint(k_pick, (n_parent,), 0, korb_pool.shape[0])
        delta = jnp.where(is_binary, korb_pool[idx], 0.0)
        v_los = v_los + delta
        # 4. Bin + Bernoulli-thin to design counts + measurement noise -> sigma_hat, se.
        sigma_hat, se = _binned_sigma_hat_1d(k_bin, v_los, R, p_keep, n_se, edges)
        # 5. Misspecified binary-free MAP fit.
        theta_hat, m_witness = _fit_theta_bf_gn(sigma_hat, se, G, n_iter=n_iter)
        theta_true = theta_truth_clusteronly()
        frac = (theta_hat - theta_true) / theta_true       # (4,) fractional bias per param
        return frac, m_witness

    # SEQUENTIAL lax.map (memory-bounded): one_draw compiles ONCE, runs draw-by-draw.
    draw_keys = jax.vmap(lambda d: jax.random.fold_in(key, d + 1))(jnp.arange(n_draws))
    fracs, witnesses = jax.lax.map(one_draw, draw_keys)     # (n_draws, 4), (n_draws,)

    m_frac = fracs[:, IDX_M]
    bias_M_frac = float(jnp.mean(m_frac))
    std_M_frac = float(jnp.std(m_frac, ddof=1)) if n_draws > 1 else 0.0
    sem_M_frac = std_M_frac / math.sqrt(n_draws) if n_draws > 1 else 0.0
    bias_other = jnp.mean(fracs[:, 1:], axis=0)            # (3,) r_a, gamma, a
    max_M_step = float(jnp.max(witnesses))
    n_unconverged = int(jnp.sum(witnesses > _BF_CONVERGED_STEP))
    if n_unconverged > 0:
        print(
            f"[cross_model_bias] WARNING: {n_unconverged}/{n_draws} draws had an "
            f"M-target witness > {_BF_CONVERGED_STEP:g} (max {max_M_step:.2e}); M_hat for "
            f"those draws may not have settled -- investigate before trusting the bias."
        )
    return CrossModelResult(
        bias_M_frac=bias_M_frac,
        std_M_frac=std_M_frac,
        sem_M_frac=sem_M_frac,
        mhat_mean=float(M_FID * (1.0 + bias_M_frac)),
        bias_other=bias_other,
        n_unconverged=n_unconverged,
        max_M_step=max_M_step,
    )
