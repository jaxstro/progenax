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
from progenax import EFFProfile, project_dispersion
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
# Task 2.1: marginalized (5-param) Fisher + f_bin prior (the binary-AWARE model)
# ===========================================================================
#
# The binary-AWARE model carries f_bin as a FREE nuisance: theta = (M, r_a, gamma, a,
# f_bin). The marginalized c-optimal-for-M design (Task 2.2) tightens M while
# marginalizing over the unknown binary fraction, so the reported sigma(M) reflects
# HONEST binary-fraction uncertainty (it is LARGER than the binary-free 4.5% -- you pay
# for not knowing f_bin). The mechanism that makes M recoverable at all is the radial
# leverage: the f_bin column of J (= f_bin*V_bin/(2 sigma_obs)) grows toward the cold
# outskirts, while M information is everywhere, so the core<->outskirts contrast breaks
# the M<->f_bin degeneracy without any external f_bin prior.
#
# The Fisher is the SAME additive single-RV-channel form as fisher_binary_free, lifted
# to 5 params:
#
#   F = Sum_b n_b * M_b + diag(PRIOR_DIAG_MARG),  M_b = 2 * outer(J_b, J_b) / (sigma_b^2 + eps_RV^2)
#
# with n_b = N_total * softmax(z), J_b the 5-vector row of the full jacrev _J_MARG, and
# CRUCIALLY sigma_b = the binary-INFLATED observed dispersion _SIG_MARG (= predict_sigma_obs
# at the full truth). The Fisher denominator (sigma^2 + eps^2) is the variance of the
# OBSERVED dispersion estimator, so it must use the observed (binary-inflated) sigma_obs,
# NOT the bare cluster sigma -- the binaries inflate the per-bin scatter the analyst sees.

# Priors on the marginalized theta = (M, r_a, gamma, a, f_bin), as ln-theta FRACTIONAL
# precisions 1/sigma_frac^2 (ADR 0011). The first four match PRIOR_DIAG_BF exactly (so
# the binary-aware vs binary-free comparison is apples-to-apples on the cluster subspace);
# the new f_bin entry is the modeling choice this task documents:
#  * f_bin (idx 4) = 1/0.5^2 -- a WEAK prior (50% fractional), the SAME weak structure as
#    the r_a kinematic nuisance. f_bin is a DATA-DRIVEN nuisance: it is constrained by the
#    radial RV leverage (the flat f_bin*V_bin pedestal vs the cluster's sigma(R) profile),
#    NOT by an external measurement. A weak (not tight, not zero) prior is the honest
#    choice: it (i) keeps the marginalized sigma(M) reflecting genuine binary-fraction
#    uncertainty (a tight f_bin prior would fake away the cost of not knowing f_bin, and a
#    zero/no prior would leave f_bin under-conditioned in bins with little leverage), while
#    (ii) deferring identification to the design -- the whole point of H2/H3 is that the
#    binary-AWARE design earns the f_bin constraint from radial leverage, not from a prior.
_FRAC_FBIN = 0.5  # weak prior on the binary-fraction nuisance (data-driven via radial leverage)
PRIOR_DIAG_MARG = jnp.array(
    [0.0, 1.0 / _FRAC_RA**2, 1.0 / _FRAC_PHOT**2, 1.0 / _FRAC_PHOT**2, 1.0 / _FRAC_FBIN**2]
)

# ---------------------------------------------------------------------------
# Build-once caches at the FULL truth (design-INDEPENDENT -- enforce-jax-performance),
# mirroring the binary-free _J_BF/_SIG_BF. The full 5-column jacrev (the single expensive
# jacrev through project_dispersion) and the binary-INFLATED observed dispersion are
# computed ONCE here at import, NEVER re-jacrev'd inside the optimizer loop. Both constant
# wrt the design weights z (theta is the fixed truth).
#  * _J_MARG = d sigma_obs / d ln theta at the full truth, (K, 5) -- includes the f_bin
#    column (verified AD-vs-FD in Task 2.1's gate).
#  * _SIG_MARG = predict_sigma_obs(full truth) -- the binary-inflated observed sigma_los
#    (km/s); the Fisher denominator (sigma^2 + eps^2) is the OBSERVED-dispersion variance.
_J_MARG = jacobian_lntheta(theta_truth(), R_BINS, STELLAR.G)    # (K, 5)
_SIG_MARG = predict_sigma_obs(theta_truth(), R_BINS, STELLAR.G)  # (K,) km/s (binary-inflated)


def fisher_marginalized(design_weights, N_total, J=None, sig=None, prior_diag=None):
    r"""Additive single-channel RV Fisher for the marginalized theta = (M, r_a, gamma, a, f_bin).

    The binary-AWARE analogue of ``fisher_binary_free``, 5-param:
    ``F = Sum_b n_b * M_b + diag(prior_diag)`` with per-bin counts
    ``n_b = N_total * softmax(design_weights)`` and the rank-1 per-bin block
    ``M_b = 2 * outer(J_b, J_b) / (sigma_b^2 + eps_RV^2)``. ``J`` defaults to the
    build-once 5-column truth jacrev ``_J_MARG`` (includes the f_bin column) and ``sig`` to
    the build-once binary-INFLATED observed dispersion ``_SIG_MARG`` (km/s) -- the Fisher
    denominator is the OBSERVED-dispersion variance, so it uses the binary-inflated sigma,
    not the bare cluster sigma. ``prior_diag`` defaults to ``PRIOR_DIAG_MARG``. Symmetric
    (5, 5); SPD with the prior (the weak f_bin + r_a priors regularize the nuisance
    subspace). Differentiable in ``design_weights`` (pure softmax + linear algebra -- no
    re-jacrev: ``_J_MARG`` / ``_SIG_MARG`` are constants).
    """
    J = _J_MARG if J is None else J
    sig = _SIG_MARG if sig is None else sig
    prior_diag = PRIOR_DIAG_MARG if prior_diag is None else prior_diag
    n_b = N_total * jax.nn.softmax(design_weights)                  # (K,) per-bin counts
    denom = sig**2 + EPS_RV_KMS**2                                  # (K,) observed-dispersion denom
    M_b = 2.0 * jnp.einsum("kp,kq->kpq", J, J) / denom[:, None, None]  # (K, 5, 5) per-bin blocks
    F = jnp.einsum("k,kpq->pq", n_b, M_b)                           # additive design Fisher
    return F + jnp.diag(prior_diag)


# ===========================================================================
# Task 2.2: binary-aware c-optimal-for-M design + H2 (OED payoff) + H3 (allocation)
# ===========================================================================
#
# The binary-AWARE design is c-optimal-for-M over the 5-param marginalized Fisher (M is
# index IDX_M = 0; f_bin is marginalized as a free nuisance). It tightens M while paying
# the honest cost of not knowing f_bin -- so its sigma(M)/M is LARGER than the binary-free
# 4.5% (which is over-confident: the binary-free Fisher pretends f_bin is known exactly).
#
# H2 (OED payoff, pre-registered ACCEPT iff >= 1.3x): scored under the SAME binary-aware
# (marginalized) Fisher, how much does the binary-aware design tighten M vs the binary-free
# design? precision_gain = sigmaM_under_marg(binary_free_z) / sigmaM_under_marg(binary_aware_z).
# The binary-free design did not optimize for the marginalized problem, so its marginalized
# sigma(M) is inflated -- the binary-aware design recovers the gain.
#
# H3 (non-obvious allocation, pre-registered): the binary-aware allocation is NOT a monotone
# rescaling of the binary-free one. A monotone rescaling preserves the per-bin weight RANK
# order and gives a near-1 cosine similarity; the binary-aware design REORDERS the bins
# (it pulls budget toward the f_bin-constraining radii), so the ranks differ and the cosine
# similarity is far below 1.


def _optimize_one_M_marg(z0, N_total, n_steps, lr):
    """One Adam trajectory minimizing c-optimality for M over the marginalized Fisher.

    The binary-aware analogue of ``_optimize_one_M``: fixed-iteration ``jax.lax.scan``
    over the length-K design logits z, jit step. The marginalized Fisher is pure linear
    algebra over the cached _J_MARG/_SIG_MARG (no re-jacrev), so each step is cheap.
    Returns (z_final, trace).
    """
    opt = optax.adam(lr)
    state = opt.init(z0)
    loss = lambda z: oed.c_criterion(fisher_marginalized(z, N_total), target=IDX_M)

    @jax.jit
    def step(carry, _):
        z, st = carry
        l, g = jax.value_and_grad(loss)(z)
        upd, st = opt.update(g, st)
        return (optax.apply_updates(z, upd), st), l

    (z, _), trace = jax.lax.scan(step, (z0, state), None, length=n_steps)
    return z, trace


def optimize_design_M_marg(N_total, key, n_starts=8, n_steps=500, lr=0.05):
    """Multi-start Adam for the binary-AWARE c-optimal-for-M radial RV design.

    Minimizes ``c_criterion(fisher_marginalized(z, N_total), target=IDX_M)`` -- the
    MARGINALIZED fractional variance [sigma(M)/M]^2 with f_bin a free nuisance -- over the
    length-K design logits z, running ``n_starts`` independent Adam trajectories and keeping
    the lowest-criterion result. Mirrors ``optimize_design_M`` exactly, swapping the
    binary-free 4-param Fisher for the 5-param marginalized one.

    The reported sigma(M)/M is LARGER than the binary-free design's (you pay for
    marginalizing the unknown f_bin) -- the HONEST binary-aware precision. The SPD invariant
    holds for the same reason as the binary-free case: softmax(z) > 0 for every finite z and
    PRIOR_DIAG_MARG adds strictly positive precision on the (r_a, gamma, a, f_bin) nuisance
    subspace, so the regularized F stays SPD throughout. Returns a DesignResultM.
    """
    K = R_BINS.shape[0]
    best = None
    for s in range(n_starts):
        z0 = jax.random.normal(jax.random.fold_in(key, s), (K,)) * 0.5
        z, trace = _optimize_one_M_marg(z0, N_total, n_steps, lr)
        crit = float(oed.c_criterion(fisher_marginalized(z, N_total), target=IDX_M))
        if math.isfinite(crit) and (best is None or crit < best[0]):
            best = (crit, z, trace)
    crit, z, trace = best
    n_eff = N_total * jax.nn.softmax(z)
    return DesignResultM(n_eff=n_eff, sigma_M_over_M=float(jnp.sqrt(crit)), z=z, trace=trace)


def sigmaM_under_marg(z, N_total=N_TOTAL):
    """Marginalized fractional precision sigma(M)/M of a design z under the binary-AWARE
    (5-param marginalized) Fisher: sqrt((F^-1)[M, M]) in the ln-theta metric. The common
    yard-stick for H2: both the binary-free and the binary-aware design are SCORED here,
    on the same marginalized Fisher, so the comparison is apples-to-apples."""
    return float(jnp.sqrt(oed.c_criterion(fisher_marginalized(z, N_total), target=IDX_M)))


def h2_precision_gain(N_total=N_TOTAL, key=None, **opt_kwargs):
    r"""H2 OED payoff: precision_gain = sigmaM_under_marg(binary_free_z) /
    sigmaM_under_marg(binary_aware_z) (pre-registered ACCEPT iff >= 1.3x).

    Both designs are evaluated under the SAME binary-aware marginalized Fisher (the honest
    inference model). ``binary_free_z`` = the Phase-1 c-optimal-for-M design under the
    binary-free Fisher (the design a binary-UNAWARE analyst adopts); ``binary_aware_z`` =
    optimize_design_M_marg (c-optimal-for-M under the marginalized Fisher). The binary-free
    design is over-confident -- it tightened M under a Fisher that pretends f_bin is known,
    so under the marginalized Fisher its sigma(M) is inflated; the binary-aware design
    recovers the precision_gain. A gain < 1.3 is a reportable NULL finding (the binary-free
    design was accidentally near-optimal for the marginalized problem) -- do NOT weaken.
    """
    key = jax.random.PRNGKey(0) if key is None else key
    binary_free_z = optimize_design_M(N_total, key=key, **opt_kwargs).z
    binary_aware_z = optimize_design_M_marg(N_total, key=key, **opt_kwargs).z
    return sigmaM_under_marg(binary_free_z, N_total) / sigmaM_under_marg(binary_aware_z, N_total)


class AllocationComparison(NamedTuple):
    """H3 allocation comparison (binary-aware vs binary-free per-bin design weights):

      * ranks_differ      : True iff the per-bin weight RANK ORDER differs (a monotone
                            rescaling would preserve it),
      * cosine_similarity : cosine similarity of the two normalized weight vectors (a
                            monotone rescaling -> ~1; reordering -> far below 1),
      * w_binary_free     : binary-free per-bin weights (softmax, sums to 1) (K,),
      * w_binary_aware    : binary-aware per-bin weights (softmax, sums to 1) (K,),
      * rank_binary_free  : per-bin ascending rank of w_binary_free (K,) ints,
      * rank_binary_aware : per-bin ascending rank of w_binary_aware (K,) ints."""
    ranks_differ: bool
    cosine_similarity: float
    w_binary_free: jnp.ndarray
    w_binary_aware: jnp.ndarray
    rank_binary_free: jnp.ndarray
    rank_binary_aware: jnp.ndarray


def h3_allocation_comparison(N_total=N_TOTAL, key=None, **opt_kwargs):
    r"""H3 non-obvious allocation: compare the binary-aware vs binary-free per-bin
    allocations and test whether the binary-aware one is a MONOTONE RESCALING of the
    binary-free one (pre-registered: ACCEPT non-monotone).

    A monotone rescaling w_aware[b] = c * w_free[b] preserves the per-bin weight RANK
    ORDER and gives cosine_similarity ~ 1. The binary-aware design instead REORDERS the
    bins -- it pulls budget toward the f_bin-constraining radii (the cold outskirts where
    the binary pedestal has leverage) and away from the bins the binary-free design favored
    -- so ``ranks_differ`` is True and ``cosine_similarity`` is far below 1. Both quantified
    metrics are reported (per the design doc "per-bin weight rank change / KL / cosine").
    """
    key = jax.random.PRNGKey(0) if key is None else key
    z_bf = optimize_design_M(N_total, key=key, **opt_kwargs).z
    z_ba = optimize_design_M_marg(N_total, key=key, **opt_kwargs).z
    w_bf = jax.nn.softmax(z_bf)                       # (K,) sums to 1
    w_ba = jax.nn.softmax(z_ba)                       # (K,)
    rank_bf = jnp.argsort(jnp.argsort(w_bf))          # ascending rank per bin
    rank_ba = jnp.argsort(jnp.argsort(w_ba))
    ranks_differ = bool(jnp.any(rank_bf != rank_ba))
    cos = float(jnp.dot(w_bf, w_ba) / (jnp.linalg.norm(w_bf) * jnp.linalg.norm(w_ba)))
    return AllocationComparison(
        ranks_differ=ranks_differ,
        cosine_similarity=cos,
        w_binary_free=w_bf,
        w_binary_aware=w_ba,
        rank_binary_free=rank_bf,
        rank_binary_aware=rank_ba,
    )


# ===========================================================================
# Task 1.4: cross-model bias harness (the H1 headline machinery)
# ===========================================================================
#
# The discriminating MC (pre-registration LOCKED 2026-06-19): GENERATE mocks UNDER
# the binary model (cluster RV + a Moe-binary RV pedestal), then FIT the BINARY-FREE
# model (f_bin == 0 -- the misspecification) and collect M_hat. If the binary-free
# design walks into a biased M_hat -- larger than its own forecast sigma -- H1 bites.
#
# FORWARD-MODEL-CONSISTENT MOCK:
# The cluster line-of-sight velocities are drawn from the SAME model the fit uses --
# per bin b, n_b = round(design_n_eff[b]) velocities from Normal(0, sig_model[b]^2),
# where sig_model = the per-bin truth sigma_los of cluster_sigma_los (project_dispersion
# / Jeans). Drawing the cluster mock straight from this Gaussian (no spatial particle
# sampling, no R-binning) makes the f_bin=0 baseline UNBIASED, so H1's bias is purely the
# binary effect (test_H0_no_binary_baseline_is_unbiased is the proof). [History: an
# earlier mock sampled EFF particles whose Eddington-sampled sigma_los disagrees with the
# Jeans projection ~6% core -> ~25% outskirts, biasing M_hat even at ZERO binaries; that
# confound is removed by drawing directly from sig_model.]
#
# Binary contamination is PER-STAR Bernoulli(f_bin_truth): each of the n_b velocities, with
# probability f_bin_truth, gets a blend Delta added (NOT a Bernoulli down-sampling/thinning
# of the population -- every drawn star is kept; some are contaminated). The per-star eps_RV
# measurement noise is added to every star, then sigma_hat[b] = std(v, ddof=1).
#
# PERFORMANCE (enforce-jax-performance):
#  * BUILD-ONCE (per truth, before the draw loop): sig_model = cluster_sigma_los(truth)
#    (the single project_dispersion call -- the only quadrature, never repeated per draw)
#    and the K_orb blend-velocity POOL (Moe massive-primary Delta, drawn once; per-draw
#    injections are a cheap uniform resample). Only these two scalars/arrays are precomputed.
#  * MC via jax.lax.map (SEQUENTIAL) over the draw keys -- one_draw compiles ONCE yet runs
#    draw-by-draw, so peak memory is a SINGLE draw's GN-fit reverse-mode tape, NOT n_draws
#    of them. The heavy part is the binary-free GN fit's per-iter jacrev through
#    project_dispersion's quadrature; lax.map keeps it memory-bounded.
#  * the per-draw fn is jit (static n_iter, n_max); the GN fit's jacrev is reverse-mode.

# K_orb injection pool size: a fixed pool of Moe massive-primary blend velocities Delta
# [km/s]. Per draw, each binary-contaminated star draws one Delta uniformly (with
# replacement) from this pool, so the realized injection variance -> Var(pool) = V_BIN
# (the pool is rescaled to exactly Var = V_BIN below). 16384 Moe draws sample the
# heavy-tailed K_orb distribution to <~1% and keep the @slow MC's gather workspace small.
_KORB_POOL_N = 16384

# HONEST-ANALYST empty-bin threshold (refinement 2026-06-19). The c-optimal-for-M design
# concentrates N_total on a FEW radial bins (at the YMC operating point: ~44% @ 7.6 pc,
# ~31% @ 17 pc, ~24% @ 1.5 pc; the rest round to ~0). A real analyst FITS ONLY the bins
# the design actually populates -- they do not insert 2-star filler bins. A bin enters the
# mock + fit iff n_b = round(design_n_eff[b]) >= N_MIN_FIT; the rest are MASKED OUT of the
# GN residual entirely (not down-weighted, not floored). 10 stars makes the per-bin ddof=1
# sample std a meaningful scatter estimate (the realized-SE weight below divides by it) and
# is below the ~26-star count of the smallest design-populated bin, so every genuinely-used
# bin survives. (10 stars > 0.5% of N_total=5000 would be 25 stars; 10 is the looser,
# more-inclusive choice -- it keeps the marginal ~0.5% bin a fit would still informatively use.)
N_MIN_FIT = 10

# Levenberg-Marquardt MAP-fit settings for the binary-free GN fit (mirrors the
# concentration demo's _GN_LM_LAM0 / _GN_N_ITER). The fit has P=4 params (M, r_a, gamma,
# a); M and the photometrically-pinned (gamma, a) are well-constrained, r_a is the weak
# nuisance the lam*I floor keeps bounded.
#
# 200 iters: the f_bin=0.5 fit must drive M FAR from the truth start (the binary pedestal
# 0.5*V_bin ~ 47 (km/s)^2 is huge vs sig_cluster^2, so the binary-free model strains to
# soak it up -> M_hat ~ 9x the truth). The VALUE settles in ~40 iters, but the LM lambda
# decay is slow when the misfit is large, so the M-step WITNESS needs ~200 iters to certify
# convergence (< _BF_CONVERGED_STEP for all draws; the baseline f_bin=0 converges to ~0 in
# a handful of iters). The fit is cheap (the per-draw mock is a few Gaussian draws + a
# gather; the GN jacrev dominates) so 200 iters is affordable.
_BF_LM_LAM0 = 1e-2
_BF_N_ITER = 200


def _per_bin_star_counts(design_n_eff):
    """Host-side STATIC per-bin sampling counts + array width + the KEEP mask (drop-empty).

    Returns (counts (K,) ints, n_max int, keep (K,) bool):

      * ``counts[b] = round(design_n_eff[b])`` -- the number of cluster stars Route-1
        SAMPLES in bin b (the mock's draw size). NO floor: the cold near-empty bins the
        c-optimal design barely uses round to ~0 and are DROPPED (see keep), not floored to
        2-star filler. counts[b] is forced to >= 1 for the DROPPED bins only so the (K,
        n_max) sampling block has a valid (masked-out, never-read) width -- the keep mask is
        what removes them from the fit.
      * ``n_max = max_b counts[b] over the KEPT bins`` -- the per-bin array width. The mock
        draws an (K, n_max) velocity block and masks the first counts[b] entries per bin, so
        sigma_hat is a fixed-shape masked ddof=1 std (lax.map-friendly).
      * ``keep[b] = round(design_n_eff[b]) >= N_MIN_FIT`` -- the bins a real analyst FITS.
        The c-optimal design populates only a few bins; ``keep`` selects exactly those, and
        the GN residual is MASKED to zero on the dropped bins (they contribute nothing to
        the fit -- not down-weighted, removed). This is the honest-analyst choice: no
        truth-based weighting, no 2-star filler bins.

    Host-side (the design is fixed across draws), so they fix the jit shapes ONCE; never
    recomputed per draw.
    """
    import numpy as np  # host-side bookkeeping only; never numpy.random

    n_eff_np = np.asarray(design_n_eff)
    counts_raw = np.round(n_eff_np).astype(int)
    keep = counts_raw >= N_MIN_FIT
    # Dropped bins are masked out of the fit; give them a valid (>=1) sampling width so the
    # (K, n_max) block + ddof=1 reduction never divide by zero (their sigma_hat is unused).
    counts = np.maximum(counts_raw, 1)
    n_max = int(counts[keep].max()) if keep.any() else int(counts.max())
    return jnp.asarray(counts), n_max, jnp.asarray(keep)


def _draw_binned_sigma_hat(key, sig_model, counts, n_max, keep, korb_pool, f_bin_truth):
    r"""PURE forward-model-consistent per-bin sigma_hat (K,) + se (K,) -- the Route-1 mock.

    For each radial bin b, draw an (K, n_max) cluster-velocity block from the SAME model
    the fit uses (Normal(0, sig_model[b]^2), sig_model = the truth cluster_sigma_los), then
    for the per-star Bernoulli(f_bin_truth) binary subset ADD a blend Delta drawn uniformly
    (with replacement) from the build-once K_orb pool (Var(pool) == V_BIN), and add per-star
    Normal(0, EPS_RV_KMS^2) measurement noise to EVERY star. Mask the first counts[b]
    entries per bin and take the masked ddof=1 sample std: sigma_hat[b] (the DATA).

    The SE is the HONEST-ANALYST realized scatter (refinement 2026-06-19):
    ``se[b] = sigma_hat[b] / sqrt(2 n_b)``, n_b = counts[b] the ACTUAL kept star count. An
    analyst does NOT know the truth sig_model; they weight each fitted bin by its OWN
    measured per-bin std -- the Gaussian-dispersion SE of a realized scatter. The
    contaminated cold outskirts then have a LARGE sigma_hat -> large se -> are appropriately
    DOWN-weighted (instead of being up-weighted by a truth-based se, which drove M_hat ~9x
    truth and r_a unphysical). This is the bias a real analyst (without truth) would incur.

    No spatial particle sampling and no R-binning: the design counts DIRECTLY set the
    per-bin sample size, so the cluster mock is consistent with the fit by construction.
    The binary effect is per-star Bernoulli CONTAMINATION (a Delta added to the hits), not a
    down-sampling of the population. ``keep`` flags the design-populated bins; the dropped
    bins' sigma_hat/se
    are never read (the GN residual masks them), but se is finite-floored here anyway so the
    masked GN linear algebra stays NaN-free.

    PURE array function (no host control flow), so lax.map batches it across draws.
    """
    K = sig_model.shape[0]
    idx_in_bin = jnp.arange(n_max)
    valid = idx_in_bin[None, :] < counts[:, None]            # (K, n_max) per-bin mask

    k_clu, k_mask, k_pick, k_eps = jax.random.split(key, 4)
    # 1. cluster velocities from the FIT model (Normal(0, sig_model^2)) per bin.
    v = sig_model[:, None] * jax.random.normal(k_clu, (K, n_max))      # (K, n_max)
    # 2. binary contamination: per-star Bernoulli(f_bin), add a pooled Delta to the hits.
    is_binary = jax.random.uniform(k_mask, (K, n_max)) < f_bin_truth   # (K, n_max)
    pick = jax.random.randint(k_pick, (K, n_max), 0, korb_pool.shape[0])
    delta = jnp.where(is_binary, korb_pool[pick], 0.0)                 # (K, n_max)
    # 3. per-star measurement noise on EVERY star.
    v = v + delta + EPS_RV_KMS * jax.random.normal(k_eps, (K, n_max))  # (K, n_max)

    w = valid.astype(jnp.float64)                            # (K, n_max) 0/1 weights
    cnt = jnp.sum(w, axis=1)                                 # (K,) == counts (float)
    mean = jnp.sum(w * v, axis=1) / jnp.maximum(cnt, 1.0)
    var = jnp.sum(w * (v - mean[:, None]) ** 2, axis=1) / jnp.maximum(cnt - 1.0, 1.0)
    sigma_hat = jnp.sqrt(var)                                # (K,) ddof=1 (the data)
    # SE = the HONEST-ANALYST realized scatter: each fitted bin weighted by its OWN measured
    # std / sqrt(2 n_b) (n_b the actual count) -- NOT the truth sig_model. Floor the
    # denominator so the masked-out (dropped) bins' se stays finite (they never enter the
    # residual; keep just guards the linear algebra).
    n_b = jnp.maximum(cnt, 1.0)                              # (K,) actual kept star count
    se = sigma_hat / jnp.sqrt(2.0 * n_b)                     # (K,) realized-scatter SE
    se = jnp.where(keep, se, 1.0)                            # dropped bins: finite placeholder
    return sigma_hat, se


@functools.partial(jax.jit, static_argnames=("n_iter",))
def _fit_theta_bf_gn(sigma_hat, se, keep, G, n_iter=_BF_N_ITER):
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

    DROP-EMPTY-BINS (refinement 2026-06-19): ``keep`` (K,) bool flags the bins the c-optimal
    design actually populates (n_b >= N_MIN_FIT). The residual is MASKED to zero on the
    dropped bins -- they contribute NOTHING to ||r||^2, Jr^T r, or Jr^T Jr (the masked
    residual is independent of u, so its jacrev row is zero). A real analyst fits only the
    bins they populate; no 2-star filler bins. Combined with the realized-scatter se
    (se[b] = sigma_hat[b]/sqrt(2 n_b)) this is HONEST-ANALYST weighting -- no truth-based
    up-weighting of the cold contaminated outskirts.

    eps-CONSISTENCY (review issue I1, CRITICAL): the mock adds Normal(0, EPS_RV_KMS^2) to
    EVERY star, so sigma_hat[b]^2 estimates sigma_cluster[b]^2 + f_bin*V_bin + EPS_RV^2. The
    binary-free PREDICTION must therefore include EPS_RV in quadrature too -- it compares
    sigma_hat to sqrt(cluster_sigma_los(theta, R_b)^2 + EPS_RV_KMS^2), NOT to the bare
    cluster sigma. Otherwise the eps^2 pedestal ALONE biases M (badly: the design weights
    the cold outskirts where sigma_cluster ~ 0.4 km/s << eps = 1 km/s, so a fit that omits
    eps would inflate M to soak up the missing eps^2). With eps consistent and f_bin = 0 the
    fit's predicted observable equals the mock's expectation exactly -> M_hat unbiased.

    Returns (theta_hat (4,), m_witness) where m_witness = max |M-component step| over the
    LAST 5 iterations (the TARGET-M convergence witness; ~0 means M_hat has settled).
    jit (static n_iter): sigma_hat/se/keep/G are traced, so the project_dispersion-jacrev
    scan compiles ONCE and is reused across every draw. Value-preserving.
    """
    theta_fid = theta_truth_clusteronly()                   # (M, r_a, gamma, a)
    keep_w = keep.astype(jnp.float64)                       # (K,) 0/1 residual mask

    def predict(theta):                                    # eps-consistent observable [km/s]
        # sqrt(sigma_cluster^2 + eps^2): the mock noises every star by eps, so the
        # binary-free model's expected sigma_hat carries eps in quadrature (NOT the bare
        # cluster sigma). This is the misspecified prediction (no f_bin*V_bin pedestal).
        return jnp.sqrt(cluster_sigma_los(theta, R_BINS, G) ** 2 + EPS_RV_KMS**2)

    def resid(u):                                           # whitened residual (model - data)/se
        theta = theta_fid * jnp.exp(u)
        r = (predict(theta) - sigma_hat) / se
        return r * keep_w                                  # dropped bins contribute nothing

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


def cross_model_bias(design_n_eff, n_draws, key, f_bin_truth=F_BIN_TRUTH, n_iter=_BF_N_ITER):
    r"""Cross-model MC: generate UNDER the binary model, fit the BINARY-FREE model.

    The headline H1 machinery, with a forward-model-consistent mock. For each of n_draws
    independent mocks (via SEQUENTIAL jax.lax.map):

      1. DROP-EMPTY-BINS: fit only the bins the c-optimal design populates, n_b =
         round(design_n_eff[b]) >= N_MIN_FIT (the rest round to ~0 and are masked out of the
         fit entirely -- no 2-star filler). For each KEPT bin b, draw n_b cluster
         line-of-sight velocities from the SAME model the fit uses: Normal(0,
         sig_model[b]^2), sig_model = the truth cluster_sigma_los (project_dispersion /
         Jeans). The mock is the Gaussian whose variance the fit predicts, so the cluster
         term is consistent with the fit by construction;
      2. BINARY CONTAMINATION: each star is a binary-contaminated tracer with probability
         f_bin = f_bin_truth (per-star Bernoulli); each contaminated star has one blend
         velocity Delta (drawn uniformly with replacement from the build-once K_orb pool,
         Var(pool) == V_BIN) ADDED to its v_los, so E[Var added] = f_bin * V_BIN;
      3. add per-star Normal(0, eps_RV^2) measurement noise to EVERY star;
      4. sigma_hat[b] = std(v_b, ddof=1), SE[b] = sigma_hat[b] / sqrt(2 n_b);
      5. FIT the BINARY-FREE theta = (M, r_a, gamma, a) (NO binary pedestal -- the
         misspecification; the prediction DOES carry eps_RV in quadrature) by
         LM-damped GN MAP in ln-theta with PRIOR_DIAG_BF;
      6. collect M_hat (and the nuisance estimates).

    With f_bin_truth = 0 the mock's per-bin EXPECTATION is exactly the fit's predicted
    observable sqrt(sigma_cluster^2 + eps^2), so M_hat is unbiased to ~0.5% -- a small
    residual from the small-sample realized-sigma_hat-weighted fit (the SE uses each bin's
    OWN measured scatter, not the truth), far below the forecast sigma(M)/M. This is the
    no-binary baseline that isolates the pure binary effect
    (test_H0_no_binary_baseline_is_unbiased). At f_bin_truth = 0.5 the binary pedestal the
    fit cannot model biases M_hat HIGH (H1).

    Returns a CrossModelResult: mean fractional bias (M_hat - M)/M, its std + SEM, the
    nuisance biases, and the M-target convergence diagnostics.

    Build-once (enforce-jax-performance): the per-bin truth sigma_los sig_model (the ONLY
    project_dispersion call -- the single quadrature, never repeated per draw) and the
    K_orb blend-velocity pool are built ONCE here, before the draw loop, and threaded into
    the jit'd per-draw body. These two are the only precomputed quantities.

    Parameters
    ----------
    design_n_eff : (K,) array
        Per-bin RV counts (the design under test; from optimize_design_M(...).n_eff).
    n_draws : int
        Number of independent cross-model mock draws (the MC sample size).
    key : PRNGKey
    f_bin_truth : float
        The TRUE binary fraction of the mock (default F_BIN_TRUTH; pass 0.0 for the
        no-binary baseline that proves Route 1 isolates binaries).
    n_iter : int
        LM iterations of the binary-free MAP fit (static; default _BF_N_ITER).
    """
    G = STELLAR.G

    # --- BUILD-ONCE (per truth, before the draw loop) ---------------------------------
    # sig_model = the per-bin TRUTH cluster sigma_los (km/s) from cluster_sigma_los
    # (project_dispersion / Jeans -- the SAME model the fit uses). The single quadrature;
    # never repeated per draw. The Route-1 mock draws Normal(0, sig_model^2) per bin, so
    # the cluster mock is consistent with the fit by construction (review issue I1).
    sig_model = cluster_sigma_los(theta_truth_clusteronly(), R_BINS, G)   # (K,) km/s

    # Static per-bin SAMPLING counts (= round(design_n_eff), NO floor) + array width n_max
    # + the KEEP mask (bins with >= N_MIN_FIT stars, the ones a real analyst fits). Host-side
    # (the design is fixed across draws), so they fix the jit shapes ONCE. The cold near-empty
    # bins are DROPPED (masked out of the GN residual), not floored to 2-star filler.
    counts, n_max, keep = _per_bin_star_counts(design_n_eff)

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

    def one_draw(kdraw):
        """One cross-model mock -> ((M_hat - M)/M, nuisance fractional biases, M-witness)."""
        k_mock, _ = jax.random.split(kdraw)
        # 1-4. Route-1 forward-model-consistent per-bin sigma_hat + realized-scatter SE
        #      (cluster from the fit model + binary pedestal + eps noise; design counts set
        #      n_b directly; se[b] = sigma_hat[b]/sqrt(2 n_b) -- honest-analyst weighting).
        sigma_hat, se = _draw_binned_sigma_hat(
            k_mock, sig_model, counts, n_max, keep, korb_pool, f_bin_truth
        )
        # 5. Misspecified binary-free MAP fit (eps-consistent prediction; dropped bins masked).
        theta_hat, m_witness = _fit_theta_bf_gn(sigma_hat, se, keep, G, n_iter=n_iter)
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


# ===========================================================================
# Task 1.5: the H1 gate -- @slow calibration MC at the YMC operating point
# ===========================================================================
#
# H1 (pre-registration LOCKED 2026-06-19): the NAIVE binary-free c-optimal-for-M design,
# fit with the BINARY-FREE model on binary-contaminated mocks, biases M_hat HIGH by more
# than its OWN forecast sigma(M) -- false confidence. ACCEPT iff
#   bias_M_frac > 2 * forecast_sigma_M_frac  AND  bias_M_frac > 0.
# The forecast sigma(M)/M is the binary-free Fisher's c-optimal precision at that SAME
# design and SAME prior (PRIOR_DIAG_BF) -- apples-to-apples with the cross-model fit.

# n_draws for the H1 gate. Pre-registration: "start 48-64; raise only if 2*SEM straddles
# the threshold." At the YMC operating point the realized bias (~1.84 fractional, M_hat/M
# ~ 2.84) DWARFS both the forecast (~0.045) and 2*SEM (~0.12 at n=48), so 48 leaves the
# accept margin un-straddled. The whole @slow MC is ~80 s and ~2.3 GB peak, so 48 is cheap.
N_DRAWS_H1 = 48


class H1Result(NamedTuple):
    """Result of run_H1 (the pre-registered H1 bias-beyond-forecast gate):

      * bias_M_frac           : realized mean fractional bias (M_hat - M)/M (cross-model MC),
      * sem                   : standard error of that mean bias over the draws,
      * std_M_frac            : draw-to-draw std of (M_hat - M)/M,
      * forecast_sigma_M_frac : the naive design's OWN binary-free Fisher forecast
                                sigma(M)/M (c-optimal, same PRIOR_DIAG_BF -- apples-to-apples),
      * ratio                 : bias_M_frac / forecast_sigma_M_frac (the headline number),
      * accept                : True iff bias_M_frac > 2*forecast_sigma_M_frac AND > 0
                                (the LOCKED pre-registration rule),
      * design_n_eff          : the naive c-optimal-for-M per-bin counts evaluated,
      * bias_other            : mean fractional bias of (r_a, gamma, a) (the absorption map),
      * n_unconverged         : # MC draws whose M_hat did not settle (surfaced, not swallowed)."""
    bias_M_frac: float
    sem: float
    std_M_frac: float
    forecast_sigma_M_frac: float
    ratio: float
    accept: bool
    design_n_eff: jnp.ndarray
    bias_other: jnp.ndarray
    n_unconverged: int


def run_H1(n_draws=N_DRAWS_H1, key=None, N_total=N_TOTAL, f_bin_truth=F_BIN_TRUTH):
    r"""Run the H1 bias-beyond-forecast gate at the YMC operating point.

    1. compute the NAIVE binary-free c-optimal-for-M design (optimize_design_M) -- the
       design a binary-UNAWARE analyst would adopt, and read its OWN forecast sigma(M)/M
       (the binary-free Fisher's c-optimal precision, same PRIOR_DIAG_BF);
    2. run the cross-model MC (cross_model_bias) at that design: generate mocks WITH Moe
       binaries, fit the binary-free model WITHOUT them, measure the realized bias(M_hat)/M;
    3. ACCEPT H1 iff bias_M_frac > 2 * forecast_sigma_M_frac AND bias_M_frac > 0 (the
       LOCKED pre-registration rule). A reject is a reportable finding (do NOT weaken).

    N_total is a reporting anchor: the Fisher is linear in N_total, so the forecast
    sigma(M)/M scales as 1/sqrt(N_total) but the design SHAPE (and hence the realized
    bias, which is N_total-independent at fixed per-bin shape) does not. Default 5000
    (N_TOTAL) -- the scale of a deep YMC RV survey of bright members.

    f_bin_truth is the TRUE binary fraction of the mock (default F_BIN_TRUTH). Passing
    f_bin_truth = 0.0 runs the NO-BINARY BASELINE: the mock then equals the fit model
    (sqrt(sigma_cluster^2 + eps^2)), so M_hat is unbiased -- the proof Route 1 isolates the
    binary effect (test_H0_no_binary_baseline_is_unbiased).
    """
    key = jax.random.PRNGKey(0) if key is None else key
    k_design, k_mc = jax.random.split(key)

    # 1. naive binary-free c-optimal-for-M design + its own forecast sigma(M)/M.
    design = optimize_design_M(N_total, key=k_design)
    forecast_sigma_M_frac = float(design.sigma_M_over_M)

    # 2. cross-model MC at that design (generate WITH binaries, fit WITHOUT).
    cm = cross_model_bias(design.n_eff, n_draws=n_draws, key=k_mc, f_bin_truth=f_bin_truth)

    # 3. LOCKED pre-registration accept rule.
    ratio = cm.bias_M_frac / forecast_sigma_M_frac if forecast_sigma_M_frac > 0 else float("inf")
    accept = bool(cm.bias_M_frac > 2.0 * forecast_sigma_M_frac and cm.bias_M_frac > 0.0)

    return H1Result(
        bias_M_frac=cm.bias_M_frac,
        sem=cm.sem_M_frac,
        std_M_frac=cm.std_M_frac,
        forecast_sigma_M_frac=forecast_sigma_M_frac,
        ratio=ratio,
        accept=accept,
        design_n_eff=design.n_eff,
        bias_other=cm.bias_other,
        n_unconverged=cm.n_unconverged,
    )


# ===========================================================================
# Task 2.5: the FIX -- the binary-AWARE fit removes the M bias (@slow calibration)
# ===========================================================================
#
# The H1 gate showed the binary-FREE fit (f_bin == 0) on binary-contaminated mocks
# biases M_hat ~ 2.85x truth (+185%). The FIX: fit the BINARY-AWARE model -- the SAME
# 5-param theta = (M, r_a, gamma, a, f_bin) with f_bin FREE -- so the analyst no longer
# misattributes the f_bin*V_bin pedestal to the cluster mass. The fit's prediction now
# carries the binary pedestal AND eps in quadrature:
#   sigma_obs(theta) = sqrt(cluster_sigma_los(M, r_a, gamma, a)^2 + f_bin*V_bin + eps^2),
# which equals the mock's per-bin EXPECTATION exactly, so M_hat is recovered unbiased and
# f_bin_hat is pulled to the truth by the radial leverage (the f_bin column of J grows
# toward the cold outskirts; the core<->outskirts contrast breaks the M<->f_bin degeneracy).
#
# This is the same honest-analyst machinery as the binary-free fit (_fit_theta_bf_gn):
# realized-scatter SE (se[b] = sigma_hat[b]/sqrt(2 n_b), NOT the truth), DROP-EMPTY-BINS
# (fit only the bins the c-optimal design populates), LM-damped Gauss-Newton MAP in ln-theta
# with PRIOR_DIAG_MARG -- just +f_bin (5 params instead of 4).

# LM-MAP settings for the 5-param binary-AWARE GN fit. Same lambda schedule as the
# binary-free fit; the f_bin free parameter is started AT the truth (theta_fid carries
# f_bin = F_BIN_TRUTH) and the well-specified model means M_hat settles fast -- but the
# 5-param fit on only ~3 populated bins needs enough iters for the M<->f_bin direction to
# converge through the radial leverage, so we keep the same generous 200-iter budget as the
# binary-free fit (the M-step witness certifies convergence; the fit is cheap).
_MARG_LM_LAM0 = 1e-2
_MARG_N_ITER = 200


@functools.partial(jax.jit, static_argnames=("n_iter",))
def _fit_theta_marg_gn(sigma_hat, se, keep, G, n_iter=_MARG_N_ITER):
    r"""Levenberg-Marquardt MAP fit of the BINARY-AWARE theta = (M, r_a, gamma, a, f_bin)
    in the dimensionless ln-theta metric, started at the full truth (f_bin FREE).

    The CORRECTLY-SPECIFIED fit (the FIX): the model is the binary-inflated, eps-consistent
    observable sqrt(cluster_sigma_los^2 + f_bin*V_bin + eps^2) with f_bin a FREE parameter,
    fit to the binary-contaminated binned dispersions. We fit u = ln(theta) - ln(theta_fid),
    theta = theta_fid * exp(u) (every direction O(1)), theta_fid = theta_truth() (the full
    5-vector, so f_bin starts at F_BIN_TRUTH). Each iteration solves the damped Gauss-Newton
    system
      (Jr^T Jr + diag(PRIOR_DIAG_MARG) + lam I) du = -(Jr^T r + PRIOR_DIAG_MARG u)
    with r = (model - data)/se the whitened residual and Jr = d r / d u (REVERSE-mode jacrev
    through project_dispersion's quadrature; by policy), ACCEPTS the step only if it lowers
    the MAP cost 0.5(||r||^2 + sum PRIOR_DIAG_MARG u^2), and adapts lam (x0.3 accept,
    x3 reject). The prior is PRIOR_DIAG_MARG (M free; r_a, f_bin weak; gamma/a tight -- the
    SAME prior the marginalized forecast Fisher uses, so the recovered bias is
    apples-to-apples with sigma_M_marg).

    DROP-EMPTY-BINS + realized-scatter SE: identical honest-analyst machinery to
    ``_fit_theta_bf_gn`` -- ``keep`` (K,) bool masks the GN residual to the design-populated
    bins (n_b >= N_MIN_FIT), and se[b] = sigma_hat[b]/sqrt(2 n_b) is the analyst's OWN
    realized scatter (NOT the truth sig_model). The only difference from the binary-free fit
    is the +f_bin free parameter and the binary-aware (f_bin*V_bin) prediction term.

    eps-CONSISTENCY: the mock adds Normal(0, eps^2) to EVERY star, so sigma_hat[b]^2 estimates
    sigma_cluster[b]^2 + f_bin*V_bin + eps^2. The prediction carries eps in quadrature
    (predict_sigma_obs is the f_bin pedestal ONLY; this fit adds eps^2 on top). With the
    binary pedestal modeled AND eps consistent, the fit's predicted observable equals the
    mock's expectation exactly -> M_hat unbiased and f_bin_hat -> truth.

    Returns (theta_hat (5,), m_witness) where m_witness = max |M-component step| over the
    LAST 5 iterations (the TARGET-M convergence witness; ~0 means M_hat has settled).
    jit (static n_iter): sigma_hat/se/keep/G are traced, so the project_dispersion-jacrev
    scan compiles ONCE and is reused across every draw. Value-preserving.
    """
    theta_fid = theta_truth()                               # (M, r_a, gamma, a, f_bin)
    keep_w = keep.astype(jnp.float64)                       # (K,) 0/1 residual mask

    def predict(theta):                                    # binary-aware eps-consistent [km/s]
        # sqrt(cluster^2 + f_bin*V_bin + eps^2): the binary pedestal (predict_sigma_obs is
        # cluster^2 + f_bin*V_bin) PLUS eps in quadrature (the mock noises every star). This
        # is the CORRECTLY-specified observable -- f_bin is fit, not assumed zero.
        return jnp.sqrt(predict_sigma_obs(theta, R_BINS, G) ** 2 + EPS_RV_KMS**2)

    def resid(u):                                           # whitened residual (model - data)/se
        theta = theta_fid * jnp.exp(u)
        r = (predict(theta) - sigma_hat) / se
        return r * keep_w                                  # dropped bins contribute nothing

    def cost_of(u, r):
        return 0.5 * (r @ r + jnp.sum(PRIOR_DIAG_MARG * u**2))

    def lm_step(carry, _):
        u, lam = carry
        r = resid(u)
        c = cost_of(u, r)
        Jr = jax.jacrev(resid)(u)                           # (K, 5) = d r / d u (reverse-mode)
        grad = Jr.T @ r + PRIOR_DIAG_MARG * u
        hess = Jr.T @ Jr + jnp.diag(PRIOR_DIAG_MARG)
        du = -jnp.linalg.solve(hess + lam * jnp.eye(5), grad)
        u_try = u + du
        c_try = cost_of(u_try, resid(u_try))
        improved = c_try < c                               # NaN -> False -> reject
        u_next = jnp.where(improved, u_try, u)
        lam_next = jnp.clip(jnp.where(improved, lam * 0.3, lam * 3.0), 1e-9, 1e9)
        m_moved = jnp.abs((u_next - u)[IDX_M])             # TARGET-M step witness
        return (u_next, lam_next), m_moved

    (u_hat, _), m_steps = jax.lax.scan(lm_step, (jnp.zeros(5), _MARG_LM_LAM0), None, length=n_iter)
    return theta_fid * jnp.exp(u_hat), jnp.max(m_steps[-5:])


# Convergence threshold on the binary-aware fit's M-target witness (max |M-step| over the
# last 5 LM iters, ln-theta), reusing the binary-free _BF_CONVERGED_STEP scale -- a settled
# fit moves M below this; a draw above it has a still-moving M_hat and is SURFACED.
_MARG_CONVERGED_STEP = _BF_CONVERGED_STEP


class FixResult(NamedTuple):
    """Result of run_fix (the Task 2.5 headline: the binary-AWARE fit removes the M bias):

      * bias_M_frac   : mean over draws of (M_hat - M_true)/M_true -- should be ~0 (vs the
                        binary-FREE fit's +1.84 at the same operating point),
      * sem           : standard error of the mean bias over the draws,
      * std_M_frac    : draw-to-draw std of (M_hat - M_true)/M_true,
      * sigma_M_marg  : the binary-AWARE forecast sigma(M)/M (optimize_design_M_marg's
                        c-optimal precision under the 5-param marginalized Fisher; ~0.069),
      * fbin_hat_mean : mean recovered f_bin_hat over draws -- does the radial leverage
                        RECOVER the binary fraction (~F_BIN_TRUTH = 0.5)?  (the Item-3b bonus),
      * fbin_hat_std  : draw-to-draw std of f_bin_hat,
      * n_unconverged : # draws whose M-target witness exceeded _MARG_CONVERGED_STEP,
      * max_M_step    : max over draws of the M-target witness,
      * design_n_eff  : the binary-aware c-optimal-for-M per-bin counts evaluated."""
    bias_M_frac: float
    sem: float
    std_M_frac: float
    sigma_M_marg: float
    fbin_hat_mean: float
    fbin_hat_std: float
    n_unconverged: int
    max_M_step: float
    design_n_eff: jnp.ndarray


def run_fix(n_draws=N_DRAWS_H1, key=None, N_total=N_TOTAL, f_bin_truth=F_BIN_TRUTH,
            n_iter=_MARG_N_ITER):
    r"""Run the Task 2.5 FIX MC: the binary-AWARE fit removes the M bias.

    Cross-model MC on the BINARY-AWARE design: generate mocks WITH Moe binaries (the SAME
    Route-1 forward-model-consistent mock as H1, at f_bin = f_bin_truth) and fit the
    BINARY-AWARE model (5-param, f_bin FREE) -- so the analyst MODELS the binary pedestal
    instead of misattributing it to M. For each of n_draws independent mocks (SEQUENTIAL
    jax.lax.map):

      1. binary-AWARE design (optimize_design_M_marg) sets the per-bin counts; DROP-EMPTY-BINS
         keeps only the bins it populates (n_b = round(design_n_eff[b]) >= N_MIN_FIT);
      2. draw cluster velocities Normal(0, sig_model^2) (sig_model = truth cluster_sigma_los),
         per-star Bernoulli(f_bin) blend Delta from the build-once K_orb pool, per-star eps;
      3. sigma_hat[b] = std(v_b, ddof=1), se[b] = sigma_hat[b]/sqrt(2 n_b) (honest-analyst);
      4. FIT the BINARY-AWARE theta = (M, r_a, gamma, a, f_bin) by LM-damped GN MAP in
         ln-theta with PRIOR_DIAG_MARG (the prediction carries f_bin*V_bin AND eps in
         quadrature -- the CORRECTLY-specified model);
      5. collect M_hat and f_bin_hat.

    The reported ``sigma_M_marg`` is the binary-aware FORECAST (optimize_design_M_marg's
    c-optimal sigma(M)/M under the 5-param marginalized Fisher, ~0.069) -- the honest
    precision the design promises. The pre-registered ACCEPT (Phase 2) is
    |bias_M_frac| < 2 * sigma_M_marg: the binary-aware fit recovers M unbiased within its
    own (honest, marginalized) forecast.

    Build-once (enforce-jax-performance): the per-bin truth sigma_los (the ONLY
    project_dispersion quadrature) and the K_orb blend-velocity pool are built ONCE, exactly
    as in cross_model_bias; the per-draw body is jit'd and run draw-by-draw via lax.map (peak
    memory = a SINGLE draw's GN reverse-mode tape). Returns a FixResult.

    Parameters
    ----------
    n_draws : int       -- number of independent cross-model mock draws (the MC sample size).
    key : PRNGKey
    N_total : float     -- RV budget (reporting anchor; Fisher linear in it).
    f_bin_truth : float -- the TRUE binary fraction of the mock (default F_BIN_TRUTH).
    n_iter : int        -- LM iterations of the binary-aware MAP fit (static).
    """
    key = jax.random.PRNGKey(0) if key is None else key
    k_design, k_mc = jax.random.split(key)
    G = STELLAR.G

    # 1. binary-AWARE c-optimal-for-M design + its honest marginalized forecast sigma(M)/M.
    design = optimize_design_M_marg(N_total, key=k_design)
    sigma_M_marg = float(design.sigma_M_over_M)

    # --- BUILD-ONCE (per truth, before the draw loop) -- mirrors cross_model_bias exactly.
    sig_model = cluster_sigma_los(theta_truth_clusteronly(), R_BINS, G)   # (K,) km/s (the quadrature)
    counts, n_max, keep = _per_bin_star_counts(design.n_eff)              # static shapes + keep mask
    korb_raw = jnp.asarray(
        binaries.sample_blend_velocities(
            jax.random.PRNGKey(V_BIN_SEED + 1), _KORB_POOL_N,
            imf=massive_primary_imf(), Z=V_BIN_Z,
        )
    )
    korb_centered = korb_raw - jnp.mean(korb_raw)
    korb_pool = korb_centered * jnp.sqrt(V_BIN / jnp.var(korb_centered, ddof=1))  # Var == V_BIN

    theta_true = theta_truth()                                            # (5,) full truth

    def one_draw(kdraw):
        """One cross-model mock -> ((M_hat - M)/M, f_bin_hat, M-witness) for the binary-aware fit."""
        k_mock, _ = jax.random.split(kdraw)
        sigma_hat, se = _draw_binned_sigma_hat(
            k_mock, sig_model, counts, n_max, keep, korb_pool, f_bin_truth
        )
        theta_hat, m_witness = _fit_theta_marg_gn(sigma_hat, se, keep, G, n_iter=n_iter)
        m_frac = (th_M(theta_hat) - th_M(theta_true)) / th_M(theta_true)
        return m_frac, th_fbin(theta_hat), m_witness

    # SEQUENTIAL lax.map (memory-bounded): one_draw compiles ONCE, runs draw-by-draw.
    draw_keys = jax.vmap(lambda d: jax.random.fold_in(k_mc, d + 1))(jnp.arange(n_draws))
    m_frac, fbin_hat, witnesses = jax.lax.map(one_draw, draw_keys)        # (n_draws,) x3

    bias_M_frac = float(jnp.mean(m_frac))
    std_M_frac = float(jnp.std(m_frac, ddof=1)) if n_draws > 1 else 0.0
    sem = std_M_frac / math.sqrt(n_draws) if n_draws > 1 else 0.0
    fbin_hat_mean = float(jnp.mean(fbin_hat))
    fbin_hat_std = float(jnp.std(fbin_hat, ddof=1)) if n_draws > 1 else 0.0
    max_M_step = float(jnp.max(witnesses))
    n_unconverged = int(jnp.sum(witnesses > _MARG_CONVERGED_STEP))
    if n_unconverged > 0:
        print(
            f"[run_fix] WARNING: {n_unconverged}/{n_draws} draws had an M-target witness "
            f"> {_MARG_CONVERGED_STEP:g} (max {max_M_step:.2e}); M_hat for those draws may "
            f"not have settled -- investigate before trusting the bias."
        )
    return FixResult(
        bias_M_frac=bias_M_frac,
        sem=sem,
        std_M_frac=std_M_frac,
        sigma_M_marg=sigma_M_marg,
        fbin_hat_mean=fbin_hat_mean,
        fbin_hat_std=fbin_hat_std,
        n_unconverged=n_unconverged,
        max_M_step=max_M_step,
        design_n_eff=design.n_eff,
    )
