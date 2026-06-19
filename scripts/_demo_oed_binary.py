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
import os
import sys

import jax
import jax.numpy as jnp

from jaxstro.units import STELLAR
from progenax import EFFProfile, project_dispersion
from progenax.imf import Maschberger

# Scripts-local siblings (NOT a packaged API): the Stage-1 OED backbone (reused for the
# unit conversion + Fisher machinery in later phases) and the B12 binary blend kernel
# (the V_bin population scalar). Ensure scripts/ is importable regardless of how this
# module is imported (the test inserts it too; this is belt-and-braces).
sys.path.insert(0, os.path.dirname(__file__))
import _demo_binaries as binaries  # noqa: E402

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


def sigma_cluster_ref(theta=None, R=None):
    r"""Central (peak) EFF-OM line-of-sight dispersion of the fiducial YMC [km/s].

    The CONSERVATIVE reference scale for the H1 sigma_bin/sigma_cluster ratio: the
    LARGEST sigma_los over the radial bins (the core). Binaries hurt fractionally MORE
    in the low-sigma outskirts, so if binaries rival the cluster at the central peak
    they rival it everywhere. Reads the RV channel of project_dispersion (the EFF-OM
    Jeans + Binney & Mamon 1982 projection) and converts pc/Myr -> km/s.

    Parameters
    ----------
    theta : optional (M, r_a, gamma, a)
        Override the fiducial theta (used in later-phase sweeps). Default -> fiducials.
    R : optional array of on-sky radii. Default -> R_BINS.
    """
    M, r_a, gamma, a = theta_truth_clusteronly() if theta is None else theta
    R = R_BINS if R is None else R
    prof = eff_profile(gamma=gamma, a=a, r_t=R_T_FID)
    sig_los_kms = kms(project_dispersion(prof, r_a, R, M, STELLAR.G).sigma_los)
    return jnp.max(sig_los_kms)


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
