r"""B2 science demo (Task 3): truth dataset + jitted joint (alpha, delta, W0)
likelihood for the self-consistent IMF + equipartition recovery.

This script builds ONLY the truth mock dataset and the jitted joint negative
log-likelihood (the MLE / Fisher / figures are Task 4+). It ends with a
mandatory RUNTIME BUDGET GATE on one warm ``jit(value_and_grad(negloglike))``
call: if it exceeds 5 s the script STOPs (it does NOT degrade the solve) and
reports for escalation.

Physics
-------
A 4-component equipartition LIMEPY cluster (Gieles & Zocchi 2015) is sampled at
the truth ``(alpha=2.3, delta=0.4, W0=5.0)``. Two observable channels constrain
the joint fit:

* **Kinematics** -- per-component binned 1-D velocity dispersion sigma_1d,j(r).
  ALL stars out to the cluster edge ``R_CUT`` (the outermost sampled radius ~ r_t)
  are used, split into 16 equal-count (quantile) radial bins on [0, R_CUT]
  (``R_EDGES``, frozen after one draw). No truncation: the binned-EXPECTATION
  comparison is unbiased at any width, so the sparse halo cells are simply
  down-weighted by their large SE rather than discarded.

  The model is compared to the data via the binned EXPECTATION of the estimator,
  NOT via the oracle evaluated at a single representative radius. The binned
  estimator's expectation is the number-weighted average of sigma^2 across the
  bin:

      E[sigma_hat^2_{jk}] = (int_bin n_j(r) sigma_j^2(r) dr) / (int_bin n_j(r) dr),

  with n_j(r) = 4 pi r^2 rho_j(r) the model's OWN per-component number density
  (encoded in the model's stored normalized cumulative number fraction
  ``m._cdf_j`` on ``m._r_grid``, so the number weight between two radii is just
  the cdf increment) and sigma_j^2(r) = s_j^2 * g(W_j(r)), W_j(r) = rescale_j *
  psi(r), g(W) = I4(W)/I2(W)/3 with I_p(W) = int_0^{sqrt(2 W)} u^p
  lowered_exponential(g, W - u^2/2) du. g(W) is a 1-D function of the local
  potential only, so it is precomputed ONCE per loss eval on a 256-point
  W-table and interpolated -- this keeps the runtime budget (a 256-point
  I4/I2 quadrature per group/bin/subpoint would be ~64x too expensive).

  This is a like-with-like comparison (binned DATA vs binned EXPECTATION) with
  NO evaluation-point approximation; it is theta-dependent (cdf_j, psi,
  rescale_j, w_j, mu_tot all rebuilt from theta) and fully differentiable. The
  global velocity scale s = sqrt(G * M_FIXED / (9 r_c mu_tot)) uses the measured
  total cluster mass M_FIXED (a fixed observed scalar), so the dispersion scale
  is data-anchored rather than re-derived from theta inside the loss.

* **Masses** (Option A, plan amendment 1) -- a SEPARATE global IMF sample
  ``m_obs`` drawn from the SAME truth ``Maschberger(alpha)`` over the full
  ``M_RANGE``, independent of the kinematic group. The mass channel is
  ``sum(Maschberger(alpha).logpdf(m_obs))`` with NO truncation correction (the
  class bounds equal the draw bounds, so the normalization matches the sample).
  Per-star mass<->group correlation is NOT modeled -- a clean-mock choice that
  keeps alpha self-consistent at the population level (one alpha drives both the
  mass histogram and the equipartition groups).

The unconstrained parameter z in R^3 maps to theta = (alpha, delta, W0) via
expit boxes (di.expit), so the fit is unconstrained and gradient-safe.

Runtime budget gate + MLE config (measured 2026-06-11, CPU/float64)
-------------------------------------------------------------------
    R_CUT (cluster edge = max radius) ~7.0 (model units; all stars used)
    R_EDGES                          16 quantile bins on [0, R_CUT]
    M_FIXED                          57382.116 (measured total mass; a constant)
    per-group occupancy (truncated)  j=0: 61896, j=1: 25972, j=2: 5989,
                                     j=3 (top): 1143 (>= 300 gate)
    find_alpha_for_masses            IFT-accelerated (hand-rolled jax.custom_vjp:
                                     adaptive while_loop forward to tol=1e-6 +
                                     reverse-mode implicit-VJP backward, flat in
                                     n_iter) -- ~3x faster per value_and_grad than
                                     the prior unrolled 30-iter eigenvalue scan.
    MLE optimizer                    3 dispersed inits (z0=0 + 2 draws) x 300 Adam
                                     steps (lr=3e-2). Right-sized from the prior
                                     4 inits x 600 steps now that each loss eval is
                                     ~3x cheaper: the full demo runs in ~5.7 min
                                     (measured 5:40 wall; vs the prior ~50 min,
                                     ~9x faster) and still recovers the reference
                                     theta_hat (alpha=2.2931, delta=0.3972,
                                     W0=4.9900) bit-for-bit.
    compile  8.84 s   (cold: traces + differentiates the adaptive
                       find_alpha_for_masses while_loop + ODE)
    warm     0.302 s  (second call at a different z) -- PASS (<= 5 s budget)
    grad(z1) = [1762.96, -233.77, 1406.92]  -- finite AND nonzero in all 3 of
               (alpha, delta, W0). Independently grad-checked vs central finite
               differences: rel-err ~1e-9..1e-11 per component (correct, not
               just finite). The g(W)-moment oracle uses the double-``where``
               safe pattern at W=0 (clamp before sqrt, clamp the I4/I2
               denominator before the divide) so the W-table's boundary node
               does not poison the gradient.
Self-consistency at truth: predict_binned(truth) vs sig_hat over ALL populated
(truncated, 16-bin) cells gives max |dev/se| = 1.93 -- no wide-outer-bin
artifact (the binned-expectation comparison is like-with-like).

Per-(group,bin) cell counts span ~17 (top group, outer bins) to ~6000 (light
group, inner bins). Both data sig_hat and model prediction are sqrt of a
number-weighted sigma^2, so the only inexactness is the O(1/n) Jensen bias of
sqrt(.); at the sparsest cell (n~17) that is ~1/(12 n) ~ 0.5%, a factor ~20
below the 1-SE width (sigma/sqrt(6 n) ~ 10%), hence negligible everywhere.
"""

import os
import sys
import time

import jax
import jax.numpy as jnp

import progenax  # noqa: F401  -- enables float64 at import
from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.imf.smooth import Maschberger
from progenax.profiles.limepy import lowered_exponential

sys.path.insert(0, os.path.dirname(__file__))
import _demo_inference as di  # noqa: E402

# --------------------------------------------------------------------------- #
# Truth configuration (module constants)
# --------------------------------------------------------------------------- #
ALPHA_TRUE = 2.3
DELTA_TRUE = 0.4
W0_TRUE = 5.0
G_MODEL = 1.0  # model units; same G used for sampling AND the oracle
N_COMP = 4
M_RANGE = (0.1, 20.0)
N_STARS = 100_000

# Unconstrained-reparam boxes (expit):
ALPHA_BOX = (1.5, 3.2)
# DELTA_BOX: delta >~ 0.9 is the Spitzer-unstable equipartition regime where
# from_imf's eigenvalue/ODE solve hard-crashes (an uncatchable diffrax max_steps
# error); delta <= 0.6 is the documented physical limit, so 0.7 gives margin
# while keeping the box inside the realizable manifold.
DELTA_BOX = (0.0, 0.7)
W0_BOX = (3.0, 8.0)

# Quadrature resolution for the I_p moment integrals (fixed-length -> grad-safe).
N_QUAD = 256
# Resolution of the precomputed g(W) = I4(W)/I2(W)/3 table (built once per eval).
N_WTAB = 256
_TINY = 1e-30


def _truth_imf():
    """The truth Maschberger over M_RANGE (class bounds == draw bounds)."""
    return Maschberger(alpha=ALPHA_TRUE, m_min=M_RANGE[0], m_max=M_RANGE[1])


# --------------------------------------------------------------------------- #
# g(W) = I4(W)/I2(W)/3 table (Engine A 1-D moment ratio; built once per eval)
# --------------------------------------------------------------------------- #
def _moment_ratio(W, g):
    r"""g(W) = I4(W) / I2(W) / 3 for one potential value W (= sigma^2 / s_j^2).

    I_p(W) = int_0^{sqrt(2 W)} u^p lowered_exponential(g, W - u^2/2) du, on a
    shared normalized grid t in [0, 1] scaled per W by u_max = sqrt(2 W). The
    Jacobian u_max cancels in the I4/I2 ratio but is carried explicitly. This is
    the VERBATIM Engine A oracle of
    ``tests/validation/test_multimass_equilibrium_physics.py:70-86`` (same
    lowered_exponential, same I4 / I2 / 3).

    ``W <= 0`` returns 0 with a FINITE gradient via the double-``where`` safe
    pattern: ``W`` is clamped to a safe positive value BEFORE ``sqrt`` (whose
    derivative is +inf at 0), and the ``I4/I2`` denominator is clamped BEFORE
    the divide (0/0 -> NaN at W=0 would otherwise poison the cotangent even in
    the dead branch). A single ``where(I2>0, ..., 0)`` is NOT enough -- the
    ratio is still differentiated at W=0. (This boundary node only ever feeds
    ``jnp.interp`` as an endpoint of the g-table; psi>0 strictly inside r_t, so
    physical on-grid W values are positive.)
    """
    W_safe = jnp.where(W > 0.0, W, 1.0)         # clamp BEFORE sqrt (deriv +inf at 0)
    u_max = jnp.sqrt(2.0 * W_safe)
    t = jnp.linspace(0.0, 1.0, N_QUAD)
    u = u_max * t
    E = lowered_exponential(g, W_safe - u * u / 2.0)
    I2 = jnp.trapezoid(u**2 * E, u)
    I4 = jnp.trapezoid(u**4 * E, u)
    I2_safe = jnp.where(I2 > 0.0, I2, 1.0)      # clamp BEFORE divide (0/0 -> NaN)
    ratio = I4 / (I2_safe * 3.0)
    return jnp.where(W > 0.0, ratio, 0.0)


def _build_g_table(W_max, g):
    """Tabulate g(W) on a fixed [0, W_max] grid (vmap over the W-table)."""
    W_tab = jnp.linspace(0.0, W_max, N_WTAB)
    g_tab = jax.vmap(lambda w: _moment_ratio(w, g))(W_tab)
    return W_tab, g_tab


# --------------------------------------------------------------------------- #
# Binned-expectation predictor: E[sigma_hat^2_{jk}] under the model's n_j(r)
# --------------------------------------------------------------------------- #
def predict_binned(theta, r_edges, m_fixed):
    r"""Per-(group, bin) sqrt(E[sigma_hat^2_{jk}]) for theta=(alpha,delta,W0).

    Rebuilds the Engine A model from theta inside the traced function (the
    find_alpha_for_masses eigenvalue solve is differentiable in alpha, delta,
    W0), then returns the number-weighted bin-average dispersion: the binned
    estimator's EXPECTATION under the model's own number density n_j(r) (from
    the stored cumulative number fraction ``m._cdf_j`` on ``m._r_grid``). No
    evaluation-point approximation -- a like-with-like comparison to the binned
    data. The velocity scale uses the FIXED observed total mass ``m_fixed`` (a
    constant), so sigma is data-anchored. Returns shape (N_COMP, K).
    """
    alpha, delta, W0 = theta
    imf = Maschberger(alpha=alpha, m_min=M_RANGE[0], m_max=M_RANGE[1])
    m = MultiComponentCluster.from_imf(imf, N_COMP, W0, g=1.0, delta=delta,
                                       m_range=M_RANGE)
    s = jnp.sqrt(G_MODEL * m_fixed / (9.0 * m.r_c * m.mu_tot))  # m_fixed constant

    # g(W) table built ONCE per eval (W spans 0 .. max(rescale_j) * W0).
    W_max = jnp.max(m.rescale_j) * W0
    W_tab, g_tab = _build_g_table(W_max, m.g)

    # psi on the model's own r-grid: psi(r/r_c), left=W0 (r->0), right=0 (r>r_t).
    psi_grid = jnp.interp(m._r_grid / m.r_c, m.xi_grid, m.psi_grid,
                          left=W0, right=0.0)  # (n_grid,)
    r_grid = m._r_grid
    edges_lo = r_edges[:-1]
    edges_hi = r_edges[1:]

    def _per_group(j):
        W_grid_j = m.rescale_j[j] * jnp.maximum(psi_grid, 0.0)  # (n_grid,)
        g_grid_j = jnp.interp(W_grid_j, W_tab, g_tab)           # sigma^2/s_j^2 on grid
        cdf_j = m._cdf_j[j]                                     # cumulative number 0..1
        # cumulative int_0^r g dN along the grid (trapezoid in dN = d cdf):
        dN = jnp.diff(cdf_j)                                    # >= 0
        gmid = 0.5 * (g_grid_j[1:] + g_grid_j[:-1])
        C2 = jnp.concatenate([jnp.zeros(1), jnp.cumsum(gmid * dN)])  # (n_grid,)
        # number-weighted <g> per bin via the cumulative ratio at the bin edges:
        C2_lo = jnp.interp(edges_lo, r_grid, C2)
        C2_hi = jnp.interp(edges_hi, r_grid, C2)
        N_lo = jnp.interp(edges_lo, r_grid, cdf_j)
        N_hi = jnp.interp(edges_hi, r_grid, cdf_j)
        gbar_jk = (C2_hi - C2_lo) / ((N_hi - N_lo) + _TINY)    # <g> per bin
        s_j = s * m.w_j[j]
        # double-where: clamp BEFORE sqrt (deriv +inf at 0) so empty/zero-g bins
        # (which carry data weight=0) keep a finite gradient through the model.
        gbar_safe = jnp.where(gbar_jk > 0.0, gbar_jk, 1.0)
        sig = s_j * jnp.sqrt(gbar_safe)
        return jnp.where(gbar_jk > 0.0, sig, 0.0)              # sqrt(E[sigma_hat^2])

    return jax.vmap(_per_group)(jnp.arange(N_COMP))            # (N_COMP, K)


def sigma_oracle(theta, r_eval, m_fixed):
    r"""Per-group 1-D dispersion sigma_j(r) at theta on radii r_eval (FOR PLOTTING).

    Evaluates the SAME Engine A oracle predict_binned integrates, but pointwise:
    sigma_j(r) = s * w_j * sqrt(g(W_j(r))), W_j(r) = rescale_j * psi(r). Returns
    shape (N_COMP, len(r_eval)). This is a smooth-curve helper for the fit figure,
    NOT used in the likelihood (the likelihood uses the unbiased binned
    EXPECTATION predict_binned).
    """
    alpha, delta, W0 = theta
    imf = Maschberger(alpha=alpha, m_min=M_RANGE[0], m_max=M_RANGE[1])
    m = MultiComponentCluster.from_imf(imf, N_COMP, W0, g=1.0, delta=delta,
                                       m_range=M_RANGE)
    s = jnp.sqrt(G_MODEL * m_fixed / (9.0 * m.r_c * m.mu_tot))
    W_max = jnp.max(m.rescale_j) * W0
    W_tab, g_tab = _build_g_table(W_max, m.g)
    psi_eval = jnp.interp(r_eval / m.r_c, m.xi_grid, m.psi_grid, left=W0, right=0.0)

    def _per_group(j):
        W_j = m.rescale_j[j] * jnp.maximum(psi_eval, 0.0)
        g_j = jnp.interp(W_j, W_tab, g_tab)
        s_j = s * m.w_j[j]
        g_safe = jnp.where(g_j > 0.0, g_j, 1.0)
        sig = s_j * jnp.sqrt(g_safe)
        return jnp.where(g_j > 0.0, sig, 0.0)

    return jax.vmap(_per_group)(jnp.arange(N_COMP))


# --------------------------------------------------------------------------- #
# Mock data construction (run once; R_CUT, R_EDGES then frozen)
# --------------------------------------------------------------------------- #
def build_truth_data():
    """Sample the truth cluster, truncate at R_CUT, bin sigma_1d,j(r), draw masses.

    Returns a dict of constants the loss closes over: R_EDGES (16 quantile bins
    on [0, R_CUT]), sig_hat, se, weight, n (kinematics), m_obs (Option A mass
    channel), M_FIXED, R_CUT.
    """
    imf = _truth_imf()
    key = jax.random.PRNGKey(0)
    k_kin, k_mass = jax.random.split(key)

    model_true = MultiComponentCluster.from_imf(
        imf, N_COMP, W0_TRUE, g=1.0, delta=DELTA_TRUE, m_range=M_RANGE)
    ic = model_true.sample_cluster(k_kin, n_stars=N_STARS, G=G_MODEL)

    # Mass-weighted COM subtraction (like the validation tests' _com_arrays).
    pos = ic.positions - jnp.average(ic.positions, axis=0, weights=ic.masses)
    vel = ic.velocities - jnp.average(ic.velocities, axis=0, weights=ic.masses)
    cid = ic.component_id  # int in [0, N_COMP)

    r = jnp.linalg.norm(pos, axis=1)

    # Use ALL stars out to the cluster edge (the outermost sampled radius ~ r_t).
    # The binned-EXPECTATION comparison is unbiased at any bin width, so no
    # truncation is needed: the sparse outer cells are down-weighted by their
    # large SE (and masked per-cell if n < n_min). Keeping the halo recovers the
    # light group's outer-sigma information, which constrains W0/concentration.
    R_CUT = float(r.max())  # cluster edge (the outermost sampled star)
    # 16 equal-count (quantile) radial bins on [0, R_CUT], FROZEN after one draw.
    # 16 is near-optimal: the heavy (segregated) group has only ~1150 stars, so
    # finer binning would push its outer cells below n_min and drop the very
    # cells carrying the equipartition (delta) signal; sigma(r) is smooth so the
    # Fisher information saturates past ~16 bins anyway.
    R_EDGES = jnp.quantile(r, jnp.linspace(0.0, 1.0, 17))  # 16 bins, FROZEN

    sig_hat, se, weight, n = di.binned_sigma1d(pos, vel, cid, N_COMP, R_EDGES,
                                               n_min=30)

    # Per-(group, bin) mean radius (for placing the figure's data points).
    bin_ids = di._bin_index(r, R_EDGES)
    r_sum, r_cnt = di._grouped_bin_sums(r, cid, bin_ids, N_COMP, R_EDGES.shape[0] - 1)
    r_mean = jnp.where(r_cnt > 0, r_sum / jnp.where(r_cnt > 0, r_cnt, 1.0), 0.0)

    # Observed-mass sample (Option A): global, independent of kinematic group.
    m_obs = imf.ppf(jax.random.uniform(k_mass, (N_STARS,)))
    M_fixed = float(jnp.sum(ic.masses))  # measured total mass (a CONSTANT)

    return dict(r_edges=R_EDGES, r_cut=R_CUT, sig_hat=sig_hat, se=se,
                weight=weight, n=n, r_mean=r_mean, m_obs=m_obs, M_fixed=M_fixed)


# --------------------------------------------------------------------------- #
# Joint negative log-likelihood (one jit(value_and_grad)-able function)
# --------------------------------------------------------------------------- #
def _theta_of_z(z):
    """Box-reparam z in R^3 -> theta = (alpha, delta, W0) in their open boxes."""
    return (di.expit(z[0], *ALPHA_BOX),
            di.expit(z[1], *DELTA_BOX),
            di.expit(z[2], *W0_BOX))


def _dtheta_dz(z):
    """Per-component derivative dtheta_i/dz_i of the expit boxes at z (shape (3,))."""
    return jnp.array([
        jax.grad(lambda zi: di.expit(zi, *ALPHA_BOX))(z[0]),
        jax.grad(lambda zi: di.expit(zi, *DELTA_BOX))(z[1]),
        jax.grad(lambda zi: di.expit(zi, *W0_BOX))(z[2]),
    ])


def make_residual_fn(data):
    r"""Build the STANDARDIZED kinematic residual vector r(z) (flattened cells).

    ``r_i = sqrt(weight_i) * (sig_hat_i - sig_model_i(z)) / safe_se_i`` so that
    ``-0.5 sum r_i^2`` equals the kinematic log-likelihood ``ll_kin`` exactly
    (weight in {0, 1}, so sqrt(weight) = weight; masked cells -> 0, contributing
    nothing to ``J^T J``). This is the Gauss-Newton residual whose ``jacrev``
    gives the kinematic Fisher information.
    """
    r_edges = data["r_edges"]
    sig_hat, se, weight = data["sig_hat"], data["se"], data["weight"]
    m_fixed = data["M_fixed"]
    safe_se = jnp.where(se > 0, se, 1.0)
    sqrt_w = jnp.sqrt(weight)

    def residual_fn(z):
        sig_model = predict_binned(_theta_of_z(z), r_edges, m_fixed)
        return (sqrt_w * (sig_hat - sig_model) / safe_se).ravel()

    return residual_fn


def make_mass_negloglike(data):
    r"""Build the ODE-free mass-channel negloglike(z) = -sum logpdf(m_obs).

    Contains NO diffrax ODE, so ``jax.hessian`` is safe on it (only the alpha-alpha
    entry of its 3x3 Hessian is nonzero). Used as ``extra_negloglike`` for the
    Gauss-Newton Fisher.
    """
    m_obs = data["m_obs"]

    def mass_negloglike(z):
        alpha = di.expit(z[0], *ALPHA_BOX)
        return -jnp.sum(
            Maschberger(alpha=alpha, m_min=M_RANGE[0], m_max=M_RANGE[1]).logpdf(m_obs))

    return mass_negloglike


def make_negloglike(data):
    """Build the joint negloglike(z) = kinematic + mass channel over the data.

    Equivalent to ``-0.5 sum r(z)^2 + mass_negloglike(z)`` using the shared
    residual / mass builders, so the loss and the Gauss-Newton Fisher are
    guaranteed consistent.
    """
    residual_fn = make_residual_fn(data)
    mass_negloglike = make_mass_negloglike(data)

    def negloglike(z):
        r = residual_fn(z)
        return 0.5 * jnp.sum(r * r) + mass_negloglike(z)

    return negloglike


# --------------------------------------------------------------------------- #
# Checks + budget gate
# --------------------------------------------------------------------------- #
def self_consistency_check(data):
    """predict_binned(truth) vs binned sig_hat in populated bins (units of se)."""
    sig_model = predict_binned((ALPHA_TRUE, DELTA_TRUE, W0_TRUE),
                               data["r_edges"], data["M_fixed"])
    sig_hat, se, weight = data["sig_hat"], data["se"], data["weight"]
    dev = jnp.where(weight > 0, (sig_hat - sig_model) / jnp.where(se > 0, se, 1.0), 0.0)

    print("\nSelf-consistency: predict_binned(truth) vs sig_hat (deviation in SE)")
    print(f"{'grp':>3} {'bin':>3} {'sig_hat':>9} {'sig_pred':>9} "
          f"{'se':>8} {'dev/se':>8}")
    max_dev = 0.0
    for j in range(sig_hat.shape[0]):
        for k in range(sig_hat.shape[1]):
            if float(weight[j, k]) > 0:
                d = float(dev[j, k])
                max_dev = max(max_dev, abs(d))
                print(f"{j:>3} {k:>3} "
                      f"{float(sig_hat[j, k]):>9.4f} {float(sig_model[j, k]):>9.4f} "
                      f"{float(se[j, k]):>8.4f} {d:>8.2f}")
    print(f"max |dev/se| over populated bins = {max_dev:.2f}")
    return max_dev


# --------------------------------------------------------------------------- #
# MLE recovery from dispersed inits + Gauss-Newton Fisher errors
# --------------------------------------------------------------------------- #
N_INITS = 3
INIT_KEY = 7
INIT_SCALE = 1.5
N_ADAM_STEPS = 300
ADAM_LR = 3e-2


def dispersed_inits():
    """3 unconstrained inits: z0=0 plus 2 draws from N(0, INIT_SCALE^2 I_3)."""
    key = jax.random.PRNGKey(INIT_KEY)
    draws = jax.random.normal(key, (N_INITS - 1, 3)) * INIT_SCALE
    return jnp.concatenate([jnp.zeros((1, 3)), draws], axis=0)


def plateau_ok(trace, frac=0.1, rel_tol=0.01):
    r"""Convergence check: the improvement over the LAST ``frac`` of steps is a
    small fraction of the TOTAL loss decrease.

    ``trace[k]`` is the loss BEFORE update k, so ``trace[0]`` is the initial loss
    and ``trace[-1]`` is the loss one step before the returned z_hat. Total
    decrease = trace[0] - trace[-1]; tail decrease = trace[k_tail] - trace[-1].
    Plateau iff tail_decrease < rel_tol * total_decrease (or total_decrease tiny).
    """
    n = trace.shape[0]
    k_tail = int(n * (1.0 - frac))
    total = float(trace[0] - trace[-1])
    tail = float(trace[k_tail] - trace[-1])
    if total <= 0:
        return False, total, tail
    return (tail < rel_tol * total), total, tail


def run_mle(negloglike, data):
    """Run Adam MLE from N_INITS dispersed inits; return the lowest-loss result.

    Returns (z_hat, best_trace, finals, i_best) where ``finals`` is the final
    negloglike per init (computed at the returned z_hat of each run).
    """
    loss_jit = jax.jit(negloglike)
    z0s = dispersed_inits()
    z_hats, traces, finals = [], [], []
    for i in range(N_INITS):
        z_hat, trace = di.mle_adam(negloglike, z0s[i],
                                   n_steps=N_ADAM_STEPS, lr=ADAM_LR)
        final = float(loss_jit(z_hat))
        z_hats.append(z_hat)
        traces.append(trace)
        finals.append(final)
    i_best = int(jnp.argmin(jnp.array(finals)))
    return z_hats[i_best], traces[i_best], finals, i_best, z0s


def recovery_table(theta_hat, sigma_theta):
    """Print param | truth | theta_hat | sigma_hat | (theta_hat-truth)/sigma_hat."""
    truths = (ALPHA_TRUE, DELTA_TRUE, W0_TRUE)
    names = ("alpha", "delta", "W0")
    print(f"\n{'param':>6} {'truth':>8} {'theta_hat':>10} {'sigma_hat':>10} "
          f"{'(hat-truth)/sigma':>18}")
    pulls = []
    for nm, tr, th, sg in zip(names, truths, theta_hat, sigma_theta):
        pull = (float(th) - tr) / float(sg)
        pulls.append(pull)
        print(f"{nm:>6} {tr:>8.4f} {float(th):>10.4f} {float(sg):>10.4f} "
              f"{pull:>18.3f}")
    return pulls


# --------------------------------------------------------------------------- #
# Fit figure (panel a: sigma_j(r) data + fit curves; panel b: mass histogram)
# --------------------------------------------------------------------------- #
def make_fit_figure(data, theta_hat, sigma_theta, out_dir):
    """Two-panel fit figure: kinematic sigma_j(r) + observed-mass IMF fit."""
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, os.path.dirname(__file__))
    import _plotstyle as ps  # noqa: E402

    ps.apply_pub_style()
    colors = [ps.OI["blue"], ps.OI["green"], ps.OI["orange"], ps.OI["vermilion"]]

    sig_hat = np.asarray(data["sig_hat"])
    se = np.asarray(data["se"])
    weight = np.asarray(data["weight"])
    r_mean = np.asarray(data["r_mean"])
    m_fixed = data["M_fixed"]
    a_hat, d_hat, w0_hat = (float(x) for x in theta_hat)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.5, 3.6))

    # Panel (a): per-group sigma(r) data (SE bars) + best-fit smooth curves.
    r_grid = jnp.linspace(1e-3, float(data["r_cut"]), 400)
    sig_fit = np.asarray(sigma_oracle((a_hat, d_hat, w0_hat), r_grid, m_fixed))
    sig_true = np.asarray(
        sigma_oracle((ALPHA_TRUE, DELTA_TRUE, W0_TRUE), r_grid, m_fixed))
    rg = np.asarray(r_grid)
    for j in range(sig_hat.shape[0]):
        mask = weight[j] > 0
        axa.errorbar(r_mean[j][mask], sig_hat[j][mask], yerr=se[j][mask],
                     fmt="o", ms=3.5, color=colors[j], capsize=1.5,
                     elinewidth=0.8, mew=0.0, zorder=3,
                     label=rf"group $j={j}$")
        axa.plot(rg, sig_fit[j], "-", color=colors[j], lw=1.6, zorder=2)
        axa.plot(rg, sig_true[j], ":", color=colors[j], lw=0.8, alpha=0.7,
                 zorder=1)
    axa.set_xlabel(r"$r$ (model units)")
    axa.set_ylabel(r"$\sigma_{1\mathrm{D}, j}(r)$")
    axa.set_xlim(0.0, float(data["r_cut"]))
    axa.legend(loc="upper right", ncol=1)
    ps.panel_label(axa, "(a)")

    # Panel (b): observed-mass histogram + fitted Maschberger(alpha_hat).
    m_obs = np.asarray(data["m_obs"])
    bins = np.logspace(np.log10(M_RANGE[0]), np.log10(M_RANGE[1]), 40)
    axb.hist(m_obs, bins=bins, density=True, histtype="stepfilled",
             color=ps.OI["sky"], alpha=0.45, edgecolor=ps.OI["blue"],
             lw=0.8, label=r"observed $m_{\rm obs}$")
    m_fine = jnp.logspace(jnp.log10(M_RANGE[0]), jnp.log10(M_RANGE[1]), 300)
    pdf_hat = jnp.exp(
        Maschberger(alpha=a_hat, m_min=M_RANGE[0], m_max=M_RANGE[1]).logpdf(m_fine))
    pdf_true = jnp.exp(
        Maschberger(alpha=ALPHA_TRUE, m_min=M_RANGE[0],
                    m_max=M_RANGE[1]).logpdf(m_fine))
    axb.plot(np.asarray(m_fine), np.asarray(pdf_hat), "-", color=ps.OI["vermilion"],
             lw=1.6, label=rf"fit $\alpha={a_hat:.3f}$")
    axb.plot(np.asarray(m_fine), np.asarray(pdf_true), ":", color=ps.OI["black"],
             lw=0.8, label=rf"truth $\alpha={ALPHA_TRUE}$")
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlabel(r"$m\ (M_\odot)$")
    axb.set_ylabel(r"$p(m)$")
    axb.legend(loc="upper right")
    ps.panel_label(axb, "(b)")

    cap = (rf"$\hat\alpha={a_hat:.3f}\pm{float(sigma_theta[0]):.3f},\ "
           rf"\hat\delta={d_hat:.3f}\pm{float(sigma_theta[1]):.3f},\ "
           rf"\hat W_0={w0_hat:.3f}\pm{float(sigma_theta[2]):.3f}$")
    fig.suptitle("")  # no in-figure title (paper caption carries it)
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, out_dir, "demo_delta_recovery_fit")
    print(f"\nfit figure -> {out_dir}/demo_delta_recovery_fit.png (+ .pdf)")


def main():
    print("=" * 72)
    print("B2 demo (Task 4): joint (alpha, delta, W0) MLE recovery + GN Fisher")
    print("=" * 72)
    print(f"truth: alpha={ALPHA_TRUE}, delta={DELTA_TRUE}, W0={W0_TRUE}; "
          f"N_COMP={N_COMP}, N_STARS={N_STARS}, M_RANGE={M_RANGE}, G={G_MODEL}")

    data = build_truth_data()
    n = data["n"]
    J = n.shape[0]
    print(f"\nM_FIXED (measured total mass) = {data['M_fixed']:.3f}")
    print(f"R_CUT (cluster edge = max radius) = {data['r_cut']:.4f}")
    print("R_EDGES (16 quantile bins on [0, R_CUT]) =",
          [round(float(x), 4) for x in data["r_edges"]])
    print("per-group counts (group x bin):")
    for j in range(J):
        print(f"  group {j}: {[int(x) for x in n[j]]}  (total {int(n[j].sum())})")
    top_occ = int(n[J - 1].sum())
    print(f"top-group (j={J - 1}) occupancy = {top_occ}"
          + ("  [WARN < 300 -- Task 4 has a >=300 gate]" if top_occ < 300 else ""))

    # Self-consistency BEFORE the budget gate.
    max_dev = self_consistency_check(data)
    sc_ok = max_dev < 4.0  # populated-bin deviations must be a few x se
    print(f"self-consistency {'OK' if sc_ok else 'FAIL'} "
          f"(max |dev/se| = {max_dev:.2f}, threshold 4.0)")
    if not sc_ok:
        print("\nSTOP: predict_binned(truth) does not match sig_hat -- "
              "miscalibrated oracle scale or group alignment. Not proceeding.")
        sys.exit(1)

    # Budget gate.
    negloglike = make_negloglike(data)
    loss_and_grad = jax.jit(jax.value_and_grad(negloglike))
    z0 = jnp.zeros(3)

    t0 = time.perf_counter()
    v, g = loss_and_grad(z0)
    v.block_until_ready(); g.block_until_ready()
    t_compile = time.perf_counter() - t0

    z1 = jnp.array([0.1, -0.2, 0.3])
    t0 = time.perf_counter()
    v2, g2 = loss_and_grad(z1)
    v2.block_until_ready(); g2.block_until_ready()
    t_warm = time.perf_counter() - t0

    grad_finite = bool(jnp.all(jnp.isfinite(g2)))
    grad_nonzero = bool(jnp.all(jnp.abs(g2) > 0))
    print("\n" + "=" * 72)
    print("RUNTIME BUDGET GATE")
    print("=" * 72)
    print(f"compile {t_compile:.2f}s, warm {t_warm:.3f}s")
    print(f"loss(z0) = {float(v):.4e}, loss(z1) = {float(v2):.4e}")
    print(f"grad(z1) = {[float(x) for x in g2]}")
    print(f"grad finite: {grad_finite}; grad nonzero (all 3): {grad_nonzero}")

    budget_ok = t_warm <= 5.0
    print(f"\nBUDGET {'PASS' if budget_ok else 'STOP'} "
          f"(warm {t_warm:.3f}s {'<=' if budget_ok else '>'} 5.0s)")
    if not budget_ok:
        print("STOP: warm loss+grad exceeds 5 s. NOT degrading the solve. "
              "Report measured numbers + plan options to the orchestrator.")
        sys.exit(2)

    # --------------------------------------------------------------------- #
    # MLE recovery from N_INITS dispersed inits.
    # --------------------------------------------------------------------- #
    print("\n" + "=" * 72)
    print("MLE RECOVERY (Adam, %d dispersed inits)" % N_INITS)
    print("=" * 72)
    z_hat, best_trace, finals, i_best, z0s = run_mle(negloglike, data)
    for i in range(N_INITS):
        tag = "  <-- WON" if i == i_best else ""
        print(f"  init {i} (z0={[round(float(x), 2) for x in z0s[i]]}): "
              f"final negloglike = {finals[i]:.6e}{tag}")
    plat_ok, total_dec, tail_dec = plateau_ok(best_trace)
    print(f"\nplateau check (winning trace): total decrease = {total_dec:.4e}, "
          f"last-10% decrease = {tail_dec:.4e}")
    print(f"plateau {'PASS' if plat_ok else 'FAIL'} "
          f"(tail {tail_dec:.3e} {'<' if plat_ok else '>='} "
          f"1% of total {0.01 * total_dec:.3e})")

    # Interior-optimum sanity: gradient norm small at z_hat.
    _, g_hat = loss_and_grad(z_hat)
    grad_norm = float(jnp.linalg.norm(g_hat))
    print(f"grad norm at z_hat = {grad_norm:.4e} (interior-optimum check)")

    # --------------------------------------------------------------------- #
    # theta_hat + Gauss-Newton Fisher errors (reverse-mode only).
    # --------------------------------------------------------------------- #
    theta_hat = _theta_of_z(z_hat)
    residual_fn = make_residual_fn(data)
    mass_negloglike = make_mass_negloglike(data)
    F_z = di.fisher_information_gn(residual_fn, z_hat,
                                   extra_negloglike=mass_negloglike)
    eig = jnp.linalg.eigvalsh(F_z)
    F_pd = bool(jnp.all(eig > 0))
    cond = float(eig[-1] / eig[0]) if F_pd else float("inf")
    print(f"\nGauss-Newton Fisher F_z PD: {F_pd}; eigenvalues = "
          f"{[float(x) for x in eig]}; cond = {cond:.3e}")
    if not F_pd:
        print("STOP: F_z is not positive definite -- degenerate / saddle fit. "
              "Reporting rather than masking.")
        sys.exit(3)

    cov_theta = di.constrained_cov(F_z, _dtheta_dz(z_hat))
    sigma_theta = jnp.sqrt(jnp.diag(cov_theta))
    pulls = recovery_table(theta_hat, sigma_theta)

    # --------------------------------------------------------------------- #
    # GATES (real -- never weaken).
    # --------------------------------------------------------------------- #
    print("\n" + "=" * 72)
    print("GATES")
    print("=" * 72)
    recovery_ok = all(abs(p) < 3.0 for p in pulls)
    names = ("alpha", "delta", "W0")
    for nm, p in zip(names, pulls):
        print(f"  3-sigma {nm}: |pull| = {abs(p):.3f} "
              f"{'<' if abs(p) < 3.0 else '>='} 3  "
              f"({'PASS' if abs(p) < 3.0 else 'FAIL'})")
    top_occ = int(n[J - 1].sum())
    occ_ok = top_occ >= 300
    print(f"  occupancy: top-group (j={J - 1}) total = {top_occ} "
          f"{'>=' if occ_ok else '<'} 300  ({'PASS' if occ_ok else 'FAIL'})")
    print(f"  plateau: {'PASS' if plat_ok else 'FAIL'}")
    print(f"  recovery (3-sigma, all params): {'PASS' if recovery_ok else 'FAIL'}")

    # --------------------------------------------------------------------- #
    # Fit figure (regenerated; promoted in Task 8 -- gitignored).
    # --------------------------------------------------------------------- #
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "validation", "plots")
    os.makedirs(out_dir, exist_ok=True)
    make_fit_figure(data, theta_hat, sigma_theta, out_dir)

    all_ok = (sc_ok and grad_finite and grad_nonzero and budget_ok
              and plat_ok and occ_ok and recovery_ok and F_pd)
    print("\n" + "=" * 72)
    print(f"OVERALL {'ALL PASS' if all_ok else 'FAIL'}")
    print("=" * 72)
    if not all_ok and recovery_ok is False:
        print("\nNOTE: the 3-sigma recovery gate is REAL. A >3-sigma miss is a "
              "PHYSICS finding -- do NOT widen the gate. Report the table above.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
