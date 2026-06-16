"""Stage-1 OED demo core: additive Fisher over (radius x channel), c/D/A criteria,
optax optimizer, sky projection + calibration. Consumer of progenax.project_dispersion.
See docs/plans/2026-06-16-oed-demo-stage1-design.md.

Task 1 (this commit): the predicted observable g(theta) and the design-INDEPENDENT
per-star Fisher blocks M_{c,b} = 2 J J^T / (sigma^2 + eps_c^2), computed via ONE
reverse-mode jacrev through project_dispersion (the only place the forward model is
differentiated). We use jacrev (reverse-mode) by policy: it is the supported/tested
gradient path for project_dispersion across ALL profiles, and it keeps the demo robust
to a future King/Michie swap, where the equilibrium-solver profiles hit custom_vjp ODE
solvers with no jvp rule so forward-mode (jacfwd/hessian) would genuinely crash. For the
Plummer profile used here the quadratures are plain jnp (no ODE / custom_vjp), so
forward-mode also happens to work -- but reverse-mode is the canonical choice. See the
src/progenax/kinematics/dispersion.py module docstring for the per-profile forward-mode
support matrix.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from progenax import PlummerProfile, project_dispersion
from jaxstro.units import STELLAR  # noqa: F401  -- re-exported for the demo's callers

# --- Unit conversions (STELLAR: M_sun, pc, Myr; project_dispersion returns pc/Myr) ---
# 1 km/s = 1 / 0.977792 pc/Myr (1 pc/Myr = 0.977792 km/s).
KMS_PER_PC_PER_MYR = 0.977792


def kms_to_pcMyr(v_kms):
    """km/s -> pc/Myr (the native velocity unit of project_dispersion under STELLAR)."""
    return v_kms / KMS_PER_PC_PER_MYR


def pm_masyr_to_kms(mu, d_kpc):
    """Proper motion [mas/yr] at distance d [kpc] -> transverse velocity [km/s]."""
    return 4.74047 * mu * d_kpc


# --- Mock cluster (generic GC-scale, unnamed -- no overclaim). theta = (r_a, M, r_h). ---
MOCK = dict(M=1e5, r_h=3.0, r_a=6.0, d_kpc=4.0, eps_RV_kms=1.0, eps_PM_masyr=0.05)

# K=12 log-spaced on-sky bin-centre radii out to ~3 r_h.
R_BINS = jnp.logspace(jnp.log10(0.3 * MOCK["r_h"]), jnp.log10(3.0 * MOCK["r_h"]), 12)

# Per-channel per-star measurement error eps_c = (eps_RV, eps_PM, eps_PM) [pc/Myr].
# Both PM axes (pm_r, pm_t) share the single astrometric error.
_eps_RV = kms_to_pcMyr(MOCK["eps_RV_kms"])
_eps_PM = kms_to_pcMyr(pm_masyr_to_kms(MOCK["eps_PM_masyr"], MOCK["d_kpc"]))
EPS = jnp.array([_eps_RV, _eps_PM, _eps_PM])      # (3,) [pc/Myr]


def theta_truth():
    """Truth parameter vector theta = (r_a, M, r_h) -- index 0 = r_a (TARGET)."""
    return jnp.array([MOCK["r_a"], MOCK["M"], MOCK["r_h"]])


def predict_sigma(theta, R_bins, G):
    """Predicted observable g(theta): (3, K) dispersions, rows = (los, pm_r, pm_t).

    Channels in pc/Myr at the K on-sky bin-centre radii R_bins, via the
    Binney & Mamon (1982) projection of the OM-Plummer Jeans model.
    """
    r_a, M, r_h = theta[0], theta[1], theta[2]
    prof = PlummerProfile(r_h=r_h)
    pd = project_dispersion(prof, r_a, R_bins, M, G)
    return jnp.stack([pd.sigma_los, pd.sigma_pm_r, pd.sigma_pm_t])   # (3, K)


def per_star_blocks(theta, R_bins, eps, G):
    """Design-INDEPENDENT per-star Fisher blocks M_{c,b} = 2 J J^T / (sigma^2 + eps_c^2),
    in the DIMENSIONLESS fractional (d ln theta) metric (ADR 0011).

    A dispersion measured from n stars (per-star error eps, predicted dispersion
    sigma) has Gaussian error delta_sigma^2 = (sigma^2 + eps^2) / (2 n), so the
    per-star Fisher contribution of channel c, bin b is the rank-1 3x3 block
    M_{c,b} = 2 J_{c,b} J_{c,b}^T / (sigma_{c,b}^2 + eps_c^2), with
    J_{c,b} = d sigma_pred,{c,b} / d ln theta. The full design Fisher is then the linear
    sum F = sum_{c,b} n_eff,{c,b} M_{c,b} (Task 2), so this jacrev is computed ONCE
    and the optimization is pure 3x3 linear algebra.

    The raw-unit theta = (r_a, M, r_h) spans ~5 orders of magnitude, so the raw Fisher
    is ill-conditioned (cond ~ 1.7e9). Differentiating wrt ln theta (J -> J * diag(theta))
    makes F dimensionless (cond ~ 45) and every (F^-1) entry a FRACTIONAL variance; the
    c-headline is then the fractional precision sigma(r_a)/r_a.

    Reverse-mode jacrev by policy (the supported/tested grad path for all profiles;
    forward-mode also works for the analytic-density Plummer path but would crash through
    the King/Michie equilibrium-solver custom_vjp ODEs -- see the module docstring).

    Returns (Mb (3, K, 3, 3): channel x bin x P x P, sigma (3, K)).
    """
    sig = predict_sigma(theta, R_bins, G)                       # (3, K)
    J = jax.jacrev(predict_sigma, argnums=0)(theta, R_bins, G)  # (3, K, 3) -- d sigma / d theta
    J = J * theta[None, None, :]                                # -> d sigma / d ln theta (DIMENSIONLESS, ADR 0011)
    denom = sig**2 + (eps[:, None])**2                          # (3, K)
    Mb = 2.0 * jnp.einsum("ckp,ckq->ckpq", J, J) / denom[..., None, None]
    return Mb, sig


# ===========================================================================
# Task 2: design allocation, completeness, and the additive Fisher F = Sum n*c*M
# ===========================================================================
#
# The Fisher information is ADDITIVE and LINEAR in the design (ADR 0004): each
# per-star block M_{c,b} (Task 1) is design-independent, so the full design
# Fisher is just the weighted sum F = sum_{c,b} n_eff,{c,b} M_{c,b}, where the
# effective per-(channel,bin) star count is the budget allocated via a softmax
# over the free design vector z, then attenuated by an illustrative completeness
# (selection) curve. Optimization (Tasks 3-4) is therefore pure 3x3 linear
# algebra; nothing below differentiates through the forward model.


def completeness(R_bins, R_turn=None, width=None):
    """Smooth faint-end roll-off (logistic in R): ~1 in the core -> <1 outskirts.

    An ILLUSTRATIVE selection function, NOT a real survey curve: a logistic
    that is ~1 where crowding/depth let nearly every star be measured and rolls
    smoothly below 1 in the outskirts. Defaults: turnover R_turn = 2*r_h, scale
    width = 0.5*r_h (both from the mock half-mass radius).
    """
    R_turn = 2.0 * MOCK["r_h"] if R_turn is None else R_turn
    width = 0.5 * MOCK["r_h"] if width is None else width
    return 1.0 / (1.0 + jnp.exp((R_bins - R_turn) / width))


def design_counts(z, completeness_b, N_total):
    """Effective per-(channel, bin) star counts n_eff (3, K).

    The free design vector z (length 3*K) is mapped to non-negative allocation
    fractions via a softmax (budget-conserving, unconstrained-optimizable),
    scaled by the total star budget N_total and attenuated per bin by the
    completeness curve. Differentiable in z (pure softmax + multiply).
    """
    K = completeness_b.shape[0]
    n = N_total * jax.nn.softmax(z).reshape(3, K)
    return n * completeness_b[None, :]


# Weak prior precision on the NUISANCES only -- diag in ln theta = ln (r_a, M, r_h).
# Rationale: M and r_h have independent observational constraints outside the
# kinematic dataset (M from integrated light / total luminosity x M/L; r_h from
# the photometric surface-brightness profile), so we encode them as a weak
# Gaussian prior. This keeps the design Fisher F well-conditioned (the
# nuisances cannot run away) WITHOUT placing any prior on the TARGET r_a -- the
# anisotropy radius must be constrained by the kinematic design alone, so its
# prior precision is exactly 0. In the dimensionless (d ln theta) metric (ADR
# 0011) the prior is a FRACTIONAL precision: a 30% fractional prior on each
# nuisance is precision 1/0.3**2; deliberately weak.
_FRAC_PRIOR = 0.3   # 30% fractional prior on each nuisance (M, r_h); none on the target r_a
PRIOR_DIAG = jnp.array([0.0, 1.0 / _FRAC_PRIOR**2, 1.0 / _FRAC_PRIOR**2])   # fractional precision


def fisher(z, Mb, completeness_b, N_total, prior_diag=None):
    """Additive design Fisher F = sum_{c,b} n_eff,{c,b} M_{c,b}  (+ optional prior).

    Linear in the effective counts n_eff (and hence in N_total at fixed design
    fractions). With prior_diag (e.g. PRIOR_DIAG) the nuisance prior precision
    is added on the diagonal. Returns a symmetric (3, 3) Fisher in theta.
    """
    n_eff = design_counts(z, completeness_b, N_total)            # (3, K)
    F = jnp.einsum("ck,ckpq->pq", n_eff, Mb)
    if prior_diag is not None:
        F = F + jnp.diag(prior_diag)
    return F


# ===========================================================================
# Task 3: c / D / A optimality criteria (all minimized)
# ===========================================================================
#
# F is the dimensionless ln-theta Fisher (Task 2.5), so every F^-1 entry is a
# FRACTIONAL variance (ADR 0011). The three classical alphabet-optimality
# criteria below are therefore all in the fractional metric and are each cast
# as a quantity to MINIMIZE, so the same optax loop (Task 4) drives all three:
#   c-optimality: minimize the marginal fractional variance of the TARGET r_a
#                 (the (r_a, r_a) entry of F^-1) -- the headline science goal.
#   D-optimality: maximize the information volume det(F) <=> minimize -logdet F
#                 (slogdet for numerical stability; sign is +1 for SPD F).
#   A-optimality: minimize the TOTAL fractional variance tr(F^-1) (the trace of
#                 the covariance) -- the average over all three parameters.
# Each is a smooth function of the SPD F, so jax.grad flows cleanly through the
# 3x3 inverse / slogdet (AD-vs-FD gate, rtol 1e-4, at cond(F) ~ 45).

_TARGET = 0   # index of r_a in theta = (r_a, M, r_h)


def c_criterion(F):
    """c-optimality (MINIMIZE): marginal FRACTIONAL variance of the target r_a.

    The (r_a, r_a) entry of F^-1 in the dimensionless ln-theta metric (ADR
    0011), i.e. [sigma(r_a)/r_a]^2 -- the squared fractional precision on the
    anisotropy radius. This is the Stage-1 headline objective.
    """
    return jnp.linalg.inv(F)[_TARGET, _TARGET]


def d_criterion(F):
    """D-optimality (MINIMIZE): -logdet F == maximize the information volume det F.

    Uses slogdet (returns (sign, logabsdet)) and returns -logabsdet. The sign is
    +1 only when the caller passes a Fisher that is SPD -- in this demo that is
    guaranteed by the prior-regularized fisher(..., PRIOR_DIAG) (see the
    SPD-invariant note on optimize_design); a bare, prior-free F is not
    guaranteed SPD and this term then ignores the slogdet sign by construction.
    """
    return -jnp.linalg.slogdet(F)[1]


def a_criterion(F):
    """A-optimality (MINIMIZE): total FRACTIONAL variance tr(F^-1).

    The trace of the covariance in the ln-theta metric -- the sum of the
    fractional variances of (r_a, M, r_h).
    """
    return jnp.trace(jnp.linalg.inv(F))


# ===========================================================================
# Task 4: optax multi-start design optimizer
# ===========================================================================
#
# Optimize the unconstrained design vector z (length 3*K) by Adam over the
# chosen criterion (c/D/A). Because the per-star blocks Mb are design-
# independent (Task 1), each gradient step is pure 3x3 linear algebra -- no
# differentiation through the forward model. The softmax in design_counts keeps
# the allocation a budget-conserving probability simplex for every (finite) z,
# so the optimization is genuinely unconstrained. We run n_starts independent
# Adam trajectories (the criterion landscape over the simplex can be multimodal)
# and keep the lowest-criterion result.


class DesignResult(NamedTuple):
    """Result of optimize_design: best design vector z (3*K,), the per-step
    criterion trace of that start, and the final scalar criterion value."""
    z: jnp.ndarray
    trace: jnp.ndarray
    criterion: float


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

    Returns a DesignResult (z, trace, criterion) for the best start.

    SPD invariant (load-bearing -- do NOT silently break it on a refactor): the
    Fisher F = fisher(z, Mb, cb, N_total, PRIOR_DIAG) stays symmetric
    positive-definite throughout the optimization, so c/D/A's inv/slogdet never
    hit a singular F. Two facts guarantee this jointly:
      (1) jax.nn.softmax(z) is strictly positive for every finite z, so every
          n_eff,{c,b} > 0 and the additive sum F = sum n_eff*Mb is at least PSD
          (each Mb is rank-1 PSD); and
      (2) PRIOR_DIAG adds strictly positive precision on the nuisance subspace
          (M, r_h), covering the directions the data may not constrain, so the
          regularized F is strictly PD.
    A future change that allocates with a hard top-k (softmax -> argmax, exact
    zeros) or zeroes the nuisance prior could reintroduce a singular F here --
    keep both (1) and (2) intact.
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
