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

* **Kinematics** -- per-component binned 1-D velocity dispersion sigma_1d,j(r),
  predicted by the Engine A analytic moment oracle (verbatim from
  ``tests/validation/test_multimass_equilibrium_physics.py:70-86``):
  sigma_jk = s * w_j * sqrt(I4 / I2 / 3) with I_p = int_0^{sqrt(2 W_jk)}
  u^p E_gamma(g, W_jk - u^2/2) du, W_jk = rescale_j * psi(r_bar_jk), and the
  global velocity scale s = sqrt(G * M_FIXED / (9 r_c mu_tot)). M_FIXED is the
  measured total cluster mass (a fixed observed scalar), so the dispersion scale
  is data-anchored rather than re-derived from theta inside the loss.

  The model is evaluated at the per-(group, bin) MEAN stellar radius r_bar_jk
  (a frozen data constant from the truth draw), NOT the geometric bin center.
  This matters in the wide outermost bin: the steeply-falling density toward
  r_t front-loads the population, so the mean radius (~4.5) sits well inside the
  bin center (~5.4) where psi -- hence sigma -- is much lower. Evaluating at the
  mean radius is the physically correct comparison point for the binned summary;
  at bin centers the outer bin mispredicts by ~60 SE (an artifact of the
  comparison point, not the oracle, which matches interior bins to ~1 SE).

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

Runtime budget gate (measured 2026-06-10, CPU/float64; recorded by ``main()``)
-----------------------------------------------------------------------------
    compile  5.18 s   (cold: traces + differentiates the 30-iter
                       find_alpha_for_masses eigenvalue solve + ODE)
    warm     0.568 s  (second call at a different z) -- PASS (<= 5 s budget)
    grad(z1) = [1291.5, 191.5, 1888.7]  -- finite AND nonzero in all 3 of
               (alpha, delta, W0): gradients flow cleanly through from_imf's
               eigenvalue solve. No degradation applied; n_ode_points/n_iter
               left at defaults (2000 / 30).
Self-consistency at truth: max |dev/se| = 2.74 over populated bins (interior
bins < ~2 SE; the model is evaluated at the per-bin MEAN stellar radius, see
``predict_sigma`` -- evaluating at geometric bin centers mispredicts the wide
outer bin by ~60 SE, a comparison-point artifact, not the oracle).
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
DELTA_BOX = (0.0, 1.0)
W0_BOX = (3.0, 8.0)

# Quadrature resolution for the I_p moment integrals (fixed-length -> grad-safe).
N_QUAD = 256


def _truth_imf():
    """The truth Maschberger over M_RANGE (class bounds == draw bounds)."""
    return Maschberger(alpha=ALPHA_TRUE, m_min=M_RANGE[0], m_max=M_RANGE[1])


# --------------------------------------------------------------------------- #
# Engine A per-group sigma_1d oracle (verbatim recipe; vectorized over j x k)
# --------------------------------------------------------------------------- #
def _sigma_moment(W_jk, g):
    r"""sqrt(I4 / I2 / 3) for one cell, I_p = int_0^{sqrt(2 W)} u^p E_g(g, W - u^2/2) du.

    Fixed-length quadrature on a shared normalized grid t in [0, 1] scaled per
    cell by u_max = sqrt(2 W). The Jacobian u_max cancels in the I4/I2 ratio but
    is carried explicitly. ``W_jk <= 0`` (escaped) returns 0 with no NaN.
    """
    W_pos = jnp.maximum(W_jk, 0.0)
    u_max = jnp.sqrt(2.0 * W_pos)
    t = jnp.linspace(0.0, 1.0, N_QUAD)
    u = u_max * t  # (N_QUAD,)
    E = lowered_exponential(g, W_pos - u * u / 2.0)
    I2 = jnp.trapezoid(u**2 * E, u)
    I4 = jnp.trapezoid(u**4 * E, u)
    ratio = jnp.where(I2 > 0.0, I4 / (I2 * 3.0), 0.0)
    return jnp.sqrt(jnp.maximum(ratio, 0.0))


def predict_sigma(theta, r_bar, m_fixed):
    r"""Per-group sigma_1d,j(r_bar_jk) for theta = (alpha, delta, W0), shape (N_COMP, K).

    Rebuilds the Engine A model from theta inside the traced function (the
    find_alpha_for_masses eigenvalue solve is differentiable in alpha, delta,
    W0), then evaluates the analytic moment oracle at the per-(group, bin) MEAN
    stellar radii ``r_bar`` (frozen data constants). The velocity scale uses the
    FIXED observed total mass ``m_fixed`` (a constant), so sigma is data-anchored.
    """
    alpha, delta, W0 = theta
    imf = Maschberger(alpha=alpha, m_min=M_RANGE[0], m_max=M_RANGE[1])
    m = MultiComponentCluster.from_imf(imf, N_COMP, W0, g=1.0, delta=delta,
                                       m_range=M_RANGE)
    s = jnp.sqrt(G_MODEL * m_fixed / (9.0 * m.r_c * m.mu_tot))  # m_fixed constant

    # psi at the per-(group, bin) mean radius: psi(r/r_c), left=W0, right=0.
    psi_jk = jnp.interp((r_bar / m.r_c).ravel(), m.xi_grid, m.psi_grid,
                        left=W0, right=0.0).reshape(r_bar.shape)  # (N_COMP, K)
    W_jk = m.rescale_j[:, None] * psi_jk  # (N_COMP, K)
    moment = jax.vmap(jax.vmap(lambda w: _sigma_moment(w, m.g)))(W_jk)
    return s * m.w_j[:, None] * moment  # (N_COMP, K)


# --------------------------------------------------------------------------- #
# Mock data construction (run once; R_EDGES then frozen)
# --------------------------------------------------------------------------- #
def build_truth_data():
    """Sample the truth cluster, bin sigma_1d,j(r), draw the global mass sample.

    Returns a dict of constants the loss closes over: R_CENTERS, sig_hat, se,
    weight, n (kinematics), m_obs (Option A mass channel), M_FIXED, R_EDGES.
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
    r_edges = jnp.quantile(r, jnp.linspace(0.0, 1.0, 9))  # 8 bins, FROZEN
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    sig_hat, se, weight, n = di.binned_sigma1d(pos, vel, cid, N_COMP, r_edges,
                                               n_min=30)

    # Per-(group, bin) MEAN stellar radius (frozen data constant; the physically
    # correct evaluation point for predict_sigma, not the geometric bin center).
    n_bins = r_edges.shape[0] - 1
    bin_ids = di._bin_index(r, r_edges)
    sum_r, _counts = di._grouped_bin_sums(r, cid, bin_ids, N_COMP, n_bins)
    r_bar = jnp.where(n > 0, sum_r / jnp.where(n > 0, n, 1.0), r_centers[None, :])

    # Observed-mass sample (Option A): global, independent of kinematic group.
    m_obs = imf.ppf(jax.random.uniform(k_mass, (N_STARS,)))
    M_fixed = float(jnp.sum(ic.masses))  # measured total mass (a CONSTANT)

    return dict(r_edges=r_edges, r_centers=r_centers, r_bar=r_bar,
                sig_hat=sig_hat, se=se, weight=weight, n=n, m_obs=m_obs,
                M_fixed=M_fixed)


# --------------------------------------------------------------------------- #
# Joint negative log-likelihood (one jit(value_and_grad)-able function)
# --------------------------------------------------------------------------- #
def make_negloglike(data):
    """Build the joint negloglike(z) closure over the frozen truth data."""
    r_bar = data["r_bar"]
    sig_hat, se, weight = data["sig_hat"], data["se"], data["weight"]
    m_obs = data["m_obs"]
    m_fixed = data["M_fixed"]
    safe_se = jnp.where(se > 0, se, 1.0)

    def negloglike(z):
        alpha = di.expit(z[0], *ALPHA_BOX)
        delta = di.expit(z[1], *DELTA_BOX)
        W0 = di.expit(z[2], *W0_BOX)
        sig_model = predict_sigma((alpha, delta, W0), r_bar, m_fixed)
        resid = (sig_hat - sig_model) / safe_se
        ll_kin = -0.5 * jnp.sum(weight * resid * resid)
        ll_mass = jnp.sum(
            Maschberger(alpha=alpha, m_min=M_RANGE[0], m_max=M_RANGE[1]).logpdf(m_obs))
        return -(ll_kin + ll_mass)

    return negloglike


# --------------------------------------------------------------------------- #
# Checks + budget gate
# --------------------------------------------------------------------------- #
def self_consistency_check(data):
    """predict_sigma(truth) vs binned sig_hat in populated bins (units of se)."""
    sig_model = predict_sigma((ALPHA_TRUE, DELTA_TRUE, W0_TRUE),
                              data["r_bar"], data["M_fixed"])
    sig_hat, se, weight = data["sig_hat"], data["se"], data["weight"]
    r_bar = data["r_bar"]
    dev = jnp.where(weight > 0, (sig_hat - sig_model) / jnp.where(se > 0, se, 1.0), 0.0)

    print("\nSelf-consistency: predict_sigma(truth) vs sig_hat (deviation in SE)")
    print(f"{'grp':>3} {'bin':>3} {'r_bar':>7} {'sig_hat':>9} {'sig_pred':>9} "
          f"{'se':>8} {'dev/se':>8}")
    max_dev = 0.0
    for j in range(sig_hat.shape[0]):
        for k in range(sig_hat.shape[1]):
            if float(weight[j, k]) > 0:
                d = float(dev[j, k])
                max_dev = max(max_dev, abs(d))
                print(f"{j:>3} {k:>3} {float(r_bar[j, k]):>7.3f} "
                      f"{float(sig_hat[j, k]):>9.4f} {float(sig_model[j, k]):>9.4f} "
                      f"{float(se[j, k]):>8.4f} {d:>8.2f}")
    print(f"max |dev/se| over populated bins = {max_dev:.2f}")
    return max_dev


def main():
    print("=" * 72)
    print("B2 demo (Task 3): truth data + jitted joint (alpha, delta, W0) loss")
    print("=" * 72)
    print(f"truth: alpha={ALPHA_TRUE}, delta={DELTA_TRUE}, W0={W0_TRUE}; "
          f"N_COMP={N_COMP}, N_STARS={N_STARS}, M_RANGE={M_RANGE}, G={G_MODEL}")

    data = build_truth_data()
    n = data["n"]
    J = n.shape[0]
    print(f"\nM_FIXED (measured total mass) = {data['M_fixed']:.3f}")
    print("R_EDGES =", [round(float(x), 4) for x in data["r_edges"]])
    print("per-group counts (group x bin):")
    for j in range(J):
        print(f"  group {j}: {[int(x) for x in n[j]]}  (total {int(n[j].sum())})")
    top_occ = int(n[J - 1].sum())
    print(f"top-group (j={J - 1}) occupancy = {top_occ}"
          + ("  [WARN < 500 -- Task 4 has a >=300 gate]" if top_occ < 500 else ""))

    # Self-consistency BEFORE the budget gate.
    max_dev = self_consistency_check(data)
    sc_ok = max_dev < 4.0  # populated-bin deviations must be a few x se
    print(f"self-consistency {'OK' if sc_ok else 'FAIL'} "
          f"(max |dev/se| = {max_dev:.2f}, threshold 4.0)")
    if not sc_ok:
        print("\nSTOP: predict_sigma(truth) does not match sig_hat -- "
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

    ok = sc_ok and grad_finite and grad_nonzero and budget_ok
    print(f"\nOVERALL {'ALL PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
