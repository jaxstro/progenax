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
