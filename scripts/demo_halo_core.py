r"""B3 science demo (Task 7): halo+core Engine-B recovery of (t, r_a, r_h).

A two-family prescribed-density cluster -- a Plummer HALO (Osipkov-Merritt
anisotropic) plus an EFF(gamma=5) CORE (isotropic) -- is sampled in detailed
shared-potential equilibrium (Engine B: prescribed rho_j -> one quadrature Psi
-> per-component Eddington inversion; NO ODE). Three parameters are recovered by
joint MLE + Gauss-Newton Fisher from two binned observables:

  theta = (t, r_a, r_h)
    t   -- halo (Plummer) mass fraction (core = 1 - t),         truth 0.6
    r_a -- halo Osipkov-Merritt anisotropy radius,              truth 3.0
    r_h -- Plummer half-mass radius,                            truth 2.0

  fixed:  EFF core (a=0.8, gamma=5, r_t=9), m_j=[0.5, 1.0], core isotropic
          (r_a=inf), domain r_t=9 (the EFF extent -- concrete by construction).

Observables (clean-mock; no LOS projection, no errors, no incompleteness):

  * sigma channel -- per-component binned 1-D dispersion sigma_1d,j(r) for BOTH
    components (di.binned_sigma_beta with component_id, n_min=50).
  * beta  channel -- the HALO (component 0) Binney anisotropy beta_hat(r). The
    core is isotropic (truth beta ~ 0), gated on the halo only (the plan's
    "halo beta_hat(r)").

  Bins: 16 equal-count radial bins over [0, R_CUT] with R_CUT = max sampled
  radius (~ the EFF edge r_t=9). R_EDGES is FROZEN after one truth draw.

predict_fn (model side; plan amendment 4 -- BINNED EXPECTATION, no bin-center
oracle). Inside the traced loss the Engine-B state is rebuilt from theta:

    m = from_density_profiles([Plummer(r_h), EFF(a=0.8,gamma=5,r_t=9)],
                              [t, 1-t], m_j=[0.5,1.0], r_a_j=[r_a, inf], r_t=9)
    st = m.engine_b   # r_poisson, Psi_poisson, dPsi_dr_poisson, E_grid,
                      # f_j_grid [J, n_e], rho_j_poisson [J, n_r], r_a_j, f_min_j

  r_t=9 is passed EXPLICITLY: the domain extent is the (fixed) EFF r_t, NOT a
  function of the recovered theta, and derive_r_t concretizes its max-extent
  pick (float()), so a traced r_h alone would raise ConcretizationTypeError. The
  explicit override gives the IDENTICAL model the truth draw uses (derived r_t
  there is also 9.0) while keeping the build fully traceable/differentiable.

  sigma recipe per component j (VERBATIM from _predicted_component_Q,
  tests/validation/test_engine_b_physics.py:146-161, replicated as JAX):

      moments(Psi_r, f_row): w = linspace(0, sqrt(2 Psi_r), n_w);
        f_at = max(interp(Psi_r - w^2/2, E_grid, f_row), 0);
        m0 = trapz(w^2 f_at, w), m2 = trapz(w^4 f_at, w)
      v2(r)      = (m2/(m0+1e-300)) * (1/3 + (2/3) * inv_st2(r)),
                   inv_st2 = 1/(1 + (r/r_a)^2)  (=1 for the isotropic core)
      sigma_j^2(r) = v2(r) / 3                    # 1-D dispersion squared

  BINNED EXPECTATION (number-weighted by the prescribed density rho_j_poisson[j],
  the 1/m_j cancels in the per-bin ratio):

      E[sigma_hat^2_jk] = (int_bin rho_j sigma_j^2 dr) / (int_bin rho_j dr)

  via a cumulative-trapezoid-at-bin-edges trick (cumsum of sigma^2 dN ratioed to
  cumsum of dN at the bin edges, jnp.interp at edges) -- the same construction as
  demo_delta_recovery.predict_binned. predict returns sqrt of that, shape (J, K),
  with the double-`where` safe-sqrt (clamp BEFORE sqrt so empty bins keep a
  finite gradient).

  beta channel predict -- beta(r) = r^2/(r^2 + r_a^2) closed-form in r_a (the OM
  definition, amendment 3; NOT from the f-moments), number-weighted the SAME way
  per bin for a like-with-like comparison to the binned beta_hat (small
  correction; beta is smooth).

Loss: residual_fn(z) stacks the weighted sigma residuals (both components) and
the weighted halo-beta residuals; negloglike = 0.5 * sum residual^2. The Fisher
information is the Gauss-Newton J^T J (di.fisher_information_gn, jacrev) mapped to
theta-space by di.constrained_cov with the dtheta/dz box Jacobian. MLE is
di.mle_adam from dispersed inits with a plateau check.

Gates (REAL -- exit nonzero on any failure; never weaken):
  1. 3-sigma recovery: |theta_hat - theta_true| < 3 sigma_hat for t, r_a, r_h.
  2. Realizability: rebuild Engine B at theta_hat; min(f_min_j) >= -1e-3.
  3. Plateau: MLE last-10% improvement < 1% of the total decrease.
  4. Occupancy: each component has enough resolved sigma bins (>= MIN_BINS_PER_CMP)
     to constrain the fit (STOP, don't weaken, if a component is starved).

Runtime (measured 2026-06-11, CPU/float64; XLA single-thread; one full run)
--------------------------------------------------------------------------
    R_CUT (cluster edge = max radius)  8.998 (the EFF r_t; all stars used)
    R_EDGES                            16 quantile bins on [0, R_CUT], FROZEN
    per-component sigma-bin occupancy  halo 16/16, core 16/16 (n_min=50)
    Engine B is QUADRATURE (no ODE), but value_and_grad through a rebuild is
                                     not free (Eddington inversion + OM moments).
    compile (cold value_and_grad)      1.44 s
    warm    jit(value_and_grad)        0.137 s  -- PASS (<= 5 s budget)
    grad(z1) finite AND nonzero in all 3 of (t, r_a, r_h).
    end-to-end wall-time               ~3.7 min -- DOMINATED by the MLE stage:
        three dispersed-init Adam runs, each a jit-compiled 400-step lax.scan
        whose body rebuilds Engine B and backprops; the three scan COMPILES are
        the bottleneck (the warm per-eval cost above is tiny by comparison).
Self-consistency at truth: predict(truth) vs sig_hat/beta_hat over all populated
cells -- max |dev/se| = 2.34 (binned-expectation is like-with-like).
Recovery (this run): t=0.602+/-0.008, r_a=3.05+/-0.05, r_h=2.02+/-0.03 (all <1sigma).
"""

import os
import sys
import time

import jax
import jax.numpy as jnp

import progenax  # noqa: F401  -- enables float64 at import
from progenax import EFFProfile, MultiComponentCluster, PlummerProfile
from jaxstro.units import STELLAR

sys.path.insert(0, os.path.dirname(__file__))
import _demo_inference as di  # noqa: E402

# --------------------------------------------------------------------------- #
# Truth configuration (module constants) -- verbatim from
# tests/validation/test_engine_b_physics.py:_headline_model + :376.
# --------------------------------------------------------------------------- #
G = STELLAR.G                              # match test_engine_b_physics.py:41
T_TRUE = 0.6                               # halo (Plummer) mass fraction
R_A_TRUE = 3.0                             # halo OM anisotropy radius
R_H_TRUE = 2.0                             # Plummer half-mass radius
M_J = [0.5, 1.0]                           # stellar-mass labels, FIXED
N_STARS = 30_000

# Fixed EFF core shape + domain (the EFF extent is the model's r_t).
EFF_A, EFF_GAMMA, EFF_RT = 0.8, 5.0, 9.0

# Unconstrained-reparam: t via expit box; r_a, r_h via exp (log-parametrized).
T_BOX = (0.3, 0.9)                         # truth 0.6 well interior

# Quadrature resolution of the f_j speed-moment integrals (fixed -> grad-safe).
N_W = 400                                  # matches _predicted_component_Q n_w
N_BINS = 16
N_MIN = 50                                 # binned_sigma_beta occupancy floor
MIN_BINS_PER_CMP = 8                       # occupancy gate: resolved sigma bins
_TINY = 1e-30


# --------------------------------------------------------------------------- #
# Box reparametrization theta = (t, r_a, r_h) <- z in R^3
# --------------------------------------------------------------------------- #
def _theta_of_z(z):
    """z in R^3 -> theta = (t in T_BOX via expit, r_a = exp(z1), r_h = exp(z2))."""
    return (di.expit(z[0], *T_BOX), jnp.exp(z[1]), jnp.exp(z[2]))


def _dtheta_dz(z):
    """Per-component dtheta_i/dz_i at z (shape (3,)): expit' on t, exp' on r_a, r_h."""
    return jnp.array([
        jax.grad(lambda zi: di.expit(zi, *T_BOX))(z[0]),
        jnp.exp(z[1]),     # d(exp z)/dz = exp z = r_a
        jnp.exp(z[2]),     # = r_h
    ])


# --------------------------------------------------------------------------- #
# Engine-B rebuild (traced) + per-component sigma_1d(r) on the Poisson grid
# --------------------------------------------------------------------------- #
def _engine_b(theta):
    """Rebuild the Engine-B state from theta=(t, r_a, r_h) (traceable).

    r_t=EFF_RT passed EXPLICITLY so the (fixed) EFF extent sets the domain
    without concretizing a traced r_h (derive_r_t's max-extent float() pick).
    """
    t, r_a, r_h = theta
    m = MultiComponentCluster.from_density_profiles(
        [PlummerProfile(r_h=r_h),
         EFFProfile(a=EFF_A, gamma=EFF_GAMMA, r_t=EFF_RT)],
        jnp.stack([t, 1.0 - t]),
        m_j=jnp.array(M_J),
        r_a_j=jnp.stack([r_a, jnp.array(jnp.inf)]),
        r_t=EFF_RT,
    )
    return m.engine_b


def _sigma_sq_on_grid(st, m_fixed):
    r"""sigma_j^2(r) = <v^2>_j(r) / 3 on st.r_poisson for every component j.

    VERBATIM the _predicted_component_Q speed-moment recipe
    (test_engine_b_physics.py:146-157): the f_j-row speed moments m0, m2 give
    <v^2>_DF,j = (m2/m0)(1/3 + (2/3)/(1 + r^2/r_a_j^2)); sigma_1d^2 = <v^2>/3.
    Psi clamped to 1e-12 before the sqrt(2 Psi) integration limit (Psi_safe).

    PHYSICAL velocity scale (the EFF kappa = G M / (4 pi mu) pattern,
    eddington_engine.py:257-261): the f_j tables are built on mass-normalized
    densities (Psi_dimless, M=1), so the sampled speeds carry a factor
    v = sqrt(G * M_sampled / (4 pi mu)) -> sigma^2_phys = (G m_fixed / (4 pi mu))
    * sigma^2_dimless, where 4 pi mu == 1 identically (computed explicitly from
    rho_j_poisson for honesty; theta-independent). ``m_fixed`` is the measured
    total cluster mass (a fixed observed scalar, like B2's M_FIXED), so sigma is
    DATA-ANCHORED rather than re-derived from theta inside the loss.
    """
    r = st.r_poisson
    Psi_safe = jnp.maximum(st.Psi_poisson, 1e-12)
    # 4 pi mu = 4 pi sum_j int rho_j r^2 dr == 1 for mass-normalized rho_j; carry
    # it explicitly (theta-independent) rather than hard-coding the identity.
    mu = jnp.trapezoid(jnp.sum(st.rho_j_poisson, axis=0) * r**2, r)
    vel_scale_sq = G * m_fixed / (4.0 * jnp.pi * mu)         # = sigma^2_phys / sigma^2_dimless

    def moments(Psi_r, f_row):
        w = jnp.linspace(0.0, jnp.sqrt(2.0 * Psi_r), N_W)
        f_at = jnp.maximum(jnp.interp(Psi_r - 0.5 * w**2, st.E_grid, f_row), 0.0)
        return jnp.trapezoid(w**2 * f_at, w), jnp.trapezoid(w**4 * f_at, w)

    def per_comp(j):
        m0, m2 = jax.vmap(lambda P: moments(P, st.f_j_grid[j]))(Psi_safe)
        finite = jnp.isfinite(st.r_a_j[j])
        ra_safe = jnp.where(finite, st.r_a_j[j], 1.0)
        inv_st2 = jnp.where(finite, 1.0 / (1.0 + (r / ra_safe) ** 2), 1.0)
        v2 = (m2 / (m0 + 1e-300)) * (1.0 / 3.0 + (2.0 / 3.0) * inv_st2)
        return vel_scale_sq * v2 / 3.0                       # sigma_1d^2(r) PHYSICAL

    return jax.vmap(per_comp)(jnp.arange(st.f_j_grid.shape[0]))   # (J, n_r)


def _bin_average(value_grid, weight_grid, r_grid, r_edges):
    r"""Number-weighted bin average of ``value_grid`` over ``r_edges``.

    <value>_k = (int_bin w value dr) / (int_bin w dr), computed by the same
    cumulative-trapezoid-at-edges trick as predict_binned: build the cumulative
    integrals C_val = int_0^r w value dr and C_w = int_0^r w dr on the grid, then
    take the edge differences via jnp.interp. ``value_grid``, ``weight_grid``,
    ``r_grid`` are 1-D over the Poisson grid; returns (K,) bin averages.
    """
    dr = jnp.diff(r_grid)
    wval = weight_grid * value_grid
    wval_mid = 0.5 * (wval[1:] + wval[:-1])
    w_mid = 0.5 * (weight_grid[1:] + weight_grid[:-1])
    C_val = jnp.concatenate([jnp.zeros(1), jnp.cumsum(wval_mid * dr)])
    C_w = jnp.concatenate([jnp.zeros(1), jnp.cumsum(w_mid * dr)])
    lo, hi = r_edges[:-1], r_edges[1:]
    num = jnp.interp(hi, r_grid, C_val) - jnp.interp(lo, r_grid, C_val)
    den = jnp.interp(hi, r_grid, C_w) - jnp.interp(lo, r_grid, C_w)
    return num / (den + _TINY)


# --------------------------------------------------------------------------- #
# Binned-expectation predictors (sigma both components; beta halo only)
# --------------------------------------------------------------------------- #
def predict_sigma_binned(theta, r_edges, m_fixed):
    r"""Per-(component, bin) sqrt(E[sigma_hat^2_jk]) under the model's rho_j(r).

    Number-weighted bin average of sigma_j^2(r) with weight rho_j_poisson[j]
    (the prescribed density; 1/m_j cancels in the ratio), then sqrt with the
    double-`where` safe pattern (clamp BEFORE sqrt -> empty/zero bins keep a
    finite gradient). Returns shape (J, K). ``m_fixed`` is the measured total
    mass that sets the physical velocity scale (see _sigma_sq_on_grid).
    """
    st = _engine_b(theta)
    r = st.r_poisson
    sig2 = _sigma_sq_on_grid(st, m_fixed)                    # (J, n_r)
    # NUMBER weight n_j(r) dr = 4 pi r^2 rho_j(r) dr (the data sig_hat pools over
    # STARS, so the bin average is number-weighted; the r^2 shell-volume factor
    # is essential in the wide outer truncation cell -- amendment 4). The 4 pi
    # and 1/m_j constants cancel in the per-bin ratio, so r^2 rho_j suffices.
    n_weight = r**2 * st.rho_j_poisson                       # (J, n_r)

    def per_comp(j):
        sig2_bar = _bin_average(sig2[j], n_weight[j], r, r_edges)  # (K,)
        safe = jnp.where(sig2_bar > 0.0, sig2_bar, 1.0)
        sig = jnp.sqrt(safe)
        return jnp.where(sig2_bar > 0.0, sig, 0.0)
    return jax.vmap(per_comp)(jnp.arange(sig2.shape[0]))      # (J, K)


def predict_beta_halo_binned(theta, r_edges):
    r"""Per-bin number-weighted OM beta(r)=r^2/(r^2+r_a^2) for the HALO (j=0).

    Closed-form in r_a (amendment 3), number-weighted by rho_0(r) the same way
    the sigma channel is, for a like-with-like comparison to the binned
    beta_hat. Returns shape (K,).
    """
    st = _engine_b(theta)
    r = st.r_poisson
    r_a = st.r_a_j[0]
    beta = r**2 / (r**2 + r_a**2)                             # OM, halo finite r_a
    n_weight = r**2 * st.rho_j_poisson[0]                     # 4 pi const cancels
    return _bin_average(beta, n_weight, r, r_edges)          # (K,)


def sigma_curve(theta, r_eval, m_fixed):
    """Pointwise sigma_1d,j(r) at theta on r_eval (FOR PLOTTING only, not loss).

    Interpolates the on-grid sigma_j(r) = sqrt(sigma_j^2(r)) onto r_eval.
    Returns (J, len(r_eval)).
    """
    st = _engine_b(theta)
    sig2 = _sigma_sq_on_grid(st, m_fixed)
    sig = jnp.sqrt(jnp.maximum(sig2, 0.0))

    def per_comp(j):
        return jnp.interp(r_eval, st.r_poisson, sig[j])
    return jax.vmap(per_comp)(jnp.arange(sig.shape[0]))


# --------------------------------------------------------------------------- #
# Mock data construction (run once; R_CUT, R_EDGES then frozen)
# --------------------------------------------------------------------------- #
def build_truth_data(key=None):
    r"""Sample the truth halo+core cluster, bin sigma_1d,j(r) + halo beta_hat(r).

    Returns a dict of constants the loss closes over: R_EDGES (16 quantile bins
    on [0, R_CUT]), per-component sig_hat/se/weight/n, the halo beta_hat/se, the
    per-(comp, bin) mean radius (for the figure), and R_CUT.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    model = MultiComponentCluster.from_density_profiles(
        [PlummerProfile(r_h=R_H_TRUE),
         EFFProfile(a=EFF_A, gamma=EFF_GAMMA, r_t=EFF_RT)],
        jnp.array([T_TRUE, 1.0 - T_TRUE]),
        m_j=jnp.array(M_J),
        r_a_j=jnp.array([R_A_TRUE, jnp.inf]),
    )
    ic = model.sample_cluster(key, n_stars=N_STARS, G=G)

    # Mass-weighted COM subtraction (the validation tests' _com_arrays).
    pos = ic.positions - jnp.average(ic.positions, axis=0, weights=ic.masses)
    vel = ic.velocities - jnp.average(ic.velocities, axis=0, weights=ic.masses)
    cid = ic.component_id                                     # int in [0, J)

    r = jnp.linalg.norm(pos, axis=1)
    R_CUT = float(r.max())                                    # cluster edge ~ r_t
    # 16 equal-count radial bins on [0, R_CUT], FROZEN after this one draw.
    R_EDGES = jnp.quantile(r, jnp.linspace(0.0, 1.0, N_BINS + 1))

    res = di.binned_sigma_beta(pos, vel, R_EDGES, component_id=cid, n_min=N_MIN)
    sig_hat, se, beta_hat, weight, n = res

    # beta SE: same 3-pooled-component effective sample size as sig_hat. With
    # sigma_r, sigma_t each ~ stable, beta_hat = 1 - sigma_t^2/(2 sigma_r^2)
    # has Var ~ (1-beta)^2 (1/N_t + ... ) ~ O(1/n); use a delta-method estimate
    # beta_se ~ (1 + |beta_hat|) / sqrt(n) -- a conservative honest SE (used only
    # as the beta-channel weight; no gate is loosened by it).
    beta_se = jnp.where(weight > 0, (1.0 + jnp.abs(beta_hat))
                        / jnp.sqrt(jnp.maximum(n, 1.0)), 0.0)

    # Per-(comp, bin) mean radius (for placing the figure's data points).
    bin_ids = di._bin_index(r, R_EDGES)
    r_sum, r_cnt = di._grouped_bin_sums(r, cid, bin_ids, sig_hat.shape[0],
                                        R_EDGES.shape[0] - 1)
    r_mean = jnp.where(r_cnt > 0, r_sum / jnp.where(r_cnt > 0, r_cnt, 1.0), 0.0)

    M_fixed = float(jnp.sum(ic.masses))  # measured total mass (a CONSTANT)

    return dict(r_edges=R_EDGES, r_cut=R_CUT, sig_hat=sig_hat, se=se,
                weight=weight, n=n, beta_hat=beta_hat, beta_se=beta_se,
                r_mean=r_mean, M_fixed=M_fixed)


# --------------------------------------------------------------------------- #
# Residual + joint negloglike (sigma both components + halo beta)
# --------------------------------------------------------------------------- #
def make_residual_fn(data):
    r"""Standardized residual vector r(z): sigma (both comps) + halo beta cells.

    r = [ sqrt(w_sig) (sig_hat - sig_model)/se ; sqrt(w_b) (beta_hat - beta_model)/beta_se ]
    so -0.5 sum r^2 is the joint Gaussian log-likelihood; jacrev(r) gives the
    Gauss-Newton Fisher. Masked cells (weight 0) contribute 0.
    """
    r_edges = data["r_edges"]
    m_fixed = data["M_fixed"]
    sig_hat, se, weight = data["sig_hat"], data["se"], data["weight"]
    beta_hat, beta_se = data["beta_hat"], data["beta_se"]
    w_halo = weight[0]                                        # halo beta weight
    safe_se = jnp.where(se > 0, se, 1.0)
    safe_bse = jnp.where(beta_se[0] > 0, beta_se[0], 1.0)
    sqrt_w = jnp.sqrt(weight)
    sqrt_wb = jnp.sqrt(w_halo)

    def residual_fn(z):
        theta = _theta_of_z(z)
        sig_model = predict_sigma_binned(theta, r_edges, m_fixed)     # (J, K)
        beta_model = predict_beta_halo_binned(theta, r_edges)        # (K,)
        r_sig = (sqrt_w * (sig_hat - sig_model) / safe_se).ravel()
        r_beta = sqrt_wb * (beta_hat[0] - beta_model) / safe_bse
        return jnp.concatenate([r_sig, r_beta])

    return residual_fn


def make_negloglike(data):
    """Joint negloglike(z) = 0.5 sum residual(z)^2 (sigma + halo beta)."""
    residual_fn = make_residual_fn(data)

    def negloglike(z):
        r = residual_fn(z)
        return 0.5 * jnp.sum(r * r)

    return negloglike


# --------------------------------------------------------------------------- #
# Self-consistency check (predict(truth) vs data, in units of SE)
# --------------------------------------------------------------------------- #
def self_consistency_check(data):
    """predict(truth) vs sig_hat/beta_hat in populated cells (units of se)."""
    truth = (T_TRUE, R_A_TRUE, R_H_TRUE)
    r_edges = data["r_edges"]
    m_fixed = data["M_fixed"]
    sig_model = predict_sigma_binned(truth, r_edges, m_fixed)
    beta_model = predict_beta_halo_binned(truth, r_edges)
    sig_hat, se, weight = data["sig_hat"], data["se"], data["weight"]
    beta_hat, beta_se = data["beta_hat"], data["beta_se"]
    safe_se = jnp.where(se > 0, se, 1.0)
    safe_bse = jnp.where(beta_se[0] > 0, beta_se[0], 1.0)            # (K,) halo
    dev_sig = jnp.where(weight > 0, (sig_hat - sig_model) / safe_se, 0.0)
    dev_beta = jnp.where(weight[0] > 0, (beta_hat[0] - beta_model) / safe_bse, 0.0)

    print("\nSelf-consistency: predict(truth) vs data (deviation in SE)")
    print(f"{'chan':>6} {'cmp':>3} {'bin':>3} {'data':>9} {'pred':>9} "
          f"{'se':>8} {'dev/se':>8}")
    max_dev = 0.0
    for j in range(sig_hat.shape[0]):
        for k in range(sig_hat.shape[1]):
            if float(weight[j, k]) > 0:
                d = float(dev_sig[j, k])
                max_dev = max(max_dev, abs(d))
                print(f"{'sigma':>6} {j:>3} {k:>3} "
                      f"{float(sig_hat[j, k]):>9.4f} {float(sig_model[j, k]):>9.4f} "
                      f"{float(se[j, k]):>8.4f} {d:>8.2f}")
    for k in range(beta_hat.shape[1]):
        if float(weight[0, k]) > 0:
            d = float(dev_beta[k])
            max_dev = max(max_dev, abs(d))
            print(f"{'beta':>6} {0:>3} {k:>3} "
                  f"{float(beta_hat[0, k]):>9.4f} {float(beta_model[k]):>9.4f} "
                  f"{float(beta_se[0, k]):>8.4f} {d:>8.2f}")
    print(f"max |dev/se| over populated cells = {max_dev:.2f}")
    return max_dev


# --------------------------------------------------------------------------- #
# MLE recovery from dispersed inits + Gauss-Newton Fisher errors
# --------------------------------------------------------------------------- #
N_INITS = 3
INIT_KEY = 7
INIT_SCALE = 0.5
N_ADAM_STEPS = 400
ADAM_LR = 3e-2


def _z_truth():
    """Unconstrained z at the truth (the z0=truth init starting point)."""
    return jnp.array([di.logit(T_TRUE, *T_BOX), jnp.log(R_A_TRUE), jnp.log(R_H_TRUE)])


def dispersed_inits():
    """3 unconstrained inits: z at truth plus 2 draws from N(z_truth, scale^2 I)."""
    key = jax.random.PRNGKey(INIT_KEY)
    z_t = _z_truth()
    draws = z_t[None, :] + jax.random.normal(key, (N_INITS - 1, 3)) * INIT_SCALE
    return jnp.concatenate([z_t[None, :], draws], axis=0)


def plateau_ok(trace, frac=0.1, rel_tol=0.01):
    """Tail-decrease < rel_tol * total-decrease over the loss trace (lagged trace)."""
    n = trace.shape[0]
    k_tail = int(n * (1.0 - frac))
    total = float(trace[0] - trace[-1])
    tail = float(trace[k_tail] - trace[-1])
    if total <= 0:
        return False, total, tail
    return (tail < rel_tol * total), total, tail


def run_mle(negloglike):
    """Adam MLE from N_INITS dispersed inits; return the lowest-loss result."""
    loss_jit = jax.jit(negloglike)
    z0s = dispersed_inits()
    z_hats, traces, finals = [], [], []
    for i in range(N_INITS):
        z_hat, trace = di.mle_adam(negloglike, z0s[i],
                                   n_steps=N_ADAM_STEPS, lr=ADAM_LR)
        finals.append(float(loss_jit(z_hat)))
        z_hats.append(z_hat)
        traces.append(trace)
    i_best = int(jnp.argmin(jnp.array(finals)))
    return z_hats[i_best], traces[i_best], finals, i_best, z0s


def recovery_table(theta_hat, sigma_theta):
    """Print param | truth | theta_hat | sigma_hat | pull = (hat-truth)/sigma."""
    truths = (T_TRUE, R_A_TRUE, R_H_TRUE)
    names = ("t", "r_a", "r_h")
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
# Figures: data-vs-fit sigma panels + halo beta panel + Fisher ellipses
# --------------------------------------------------------------------------- #
def _ellipse_xy(mean2, cov2, n_sigma=2.0, n_pts=200):
    """(x, y) of the n_sigma covariance ellipse for a 2-D Gaussian."""
    t = jnp.linspace(0.0, 2.0 * jnp.pi, n_pts)
    circle = jnp.stack([jnp.cos(t), jnp.sin(t)], axis=0)
    L = jnp.linalg.cholesky(cov2)
    pts = mean2[:, None] + n_sigma * (L @ circle)
    return pts[0], pts[1]


def make_figure(data, theta_hat, sigma_theta, cov_theta, out_dir):
    """4-panel figure: sigma_j(r) (2 comps) + halo beta(r) + 2 Fisher ellipses."""
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, os.path.dirname(__file__))
    import _plotstyle as ps  # noqa: E402

    ps.apply_pub_style()
    colors = [ps.OI["blue"], ps.OI["vermilion"]]
    cmp_names = ["halo (Plummer)", "core (EFF)"]

    sig_hat = np.asarray(data["sig_hat"])
    se = np.asarray(data["se"])
    weight = np.asarray(data["weight"])
    r_mean = np.asarray(data["r_mean"])
    beta_hat = np.asarray(data["beta_hat"])
    beta_se = np.asarray(data["beta_se"])
    m_fixed = data["M_fixed"]
    t_hat, ra_hat, rh_hat = (float(x) for x in theta_hat)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0))
    ax_h, ax_c, ax_b, ax_e = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panels (a),(b): per-component sigma(r) data (SE bars) + best-fit curves.
    r_grid = jnp.linspace(1e-3, float(data["r_cut"]), 400)
    sig_fit = np.asarray(sigma_curve((t_hat, ra_hat, rh_hat), r_grid, m_fixed))
    sig_true = np.asarray(sigma_curve((T_TRUE, R_A_TRUE, R_H_TRUE), r_grid, m_fixed))
    rg = np.asarray(r_grid)
    for j, (ax, lab) in enumerate([(ax_h, "(a)"), (ax_c, "(b)")]):
        mask = weight[j] > 0
        ax.errorbar(r_mean[j][mask], sig_hat[j][mask], yerr=se[j][mask],
                    fmt="o", ms=3.5, color=colors[j], capsize=1.5,
                    elinewidth=0.8, mew=0.0, zorder=3, label="mock data")
        ax.plot(rg, sig_fit[j], "-", color=colors[j], lw=1.6, zorder=2,
                label="best fit")
        ax.plot(rg, sig_true[j], ":", color=ps.OI["black"], lw=0.9, alpha=0.7,
                zorder=1, label="truth")
        ax.set_xlabel(r"$r$ (model units)")
        ax.set_ylabel(rf"$\sigma_{{1\mathrm{{D}}}}(r)$ -- {cmp_names[j]}")
        ax.set_xlim(0.0, float(data["r_cut"]))
        ax.legend(loc="upper right")
        ps.panel_label(ax, lab)

    # Panel (c): halo beta_hat(r) + the closed-form OM beta(r; r_a_hat).
    mask = weight[0] > 0
    ax_b.errorbar(r_mean[0][mask], beta_hat[0][mask], yerr=beta_se[0][mask],
                  fmt="o", ms=3.5, color=ps.OI["blue"], capsize=1.5,
                  elinewidth=0.8, mew=0.0, zorder=3, label=r"halo $\hat\beta(r)$")
    beta_fit = rg**2 / (rg**2 + ra_hat**2)
    beta_true = rg**2 / (rg**2 + R_A_TRUE**2)
    ax_b.plot(rg, beta_fit, "-", color=ps.OI["blue"], lw=1.6, zorder=2,
              label=rf"OM fit $r_a={ra_hat:.2f}$")
    ax_b.plot(rg, beta_true, ":", color=ps.OI["black"], lw=0.9, alpha=0.7,
              zorder=1, label=rf"truth $r_a={R_A_TRUE}$")
    ax_b.axhline(0.0, color=ps.OI["green"], lw=0.8, ls="--", alpha=0.6,
                 label="core (isotropic)")
    ax_b.set_xlabel(r"$r$ (model units)")
    ax_b.set_ylabel(r"$\beta(r)$ (Binney anisotropy)")
    ax_b.set_xlim(0.0, float(data["r_cut"]))
    ax_b.legend(loc="lower right")
    ps.panel_label(ax_b, "(c)")

    # Panel (d): 2-sigma Fisher ellipses for (r_a, r_h) and (t, r_a).
    cov = np.asarray(cov_theta)
    # (r_a, r_h) block = indices (1, 2); (t, r_a) block = indices (0, 1).
    mean_rarh = jnp.array([ra_hat, rh_hat])
    cov_rarh = jnp.asarray(cov[1:, 1:])
    mean_tra = jnp.array([t_hat, ra_hat])
    cov_tra = jnp.asarray(cov[jnp.ix_(jnp.array([0, 1]), jnp.array([0, 1]))])

    # Primary: (r_a, r_h) 2-sigma ellipse + MLE + truth.
    x1, y1 = _ellipse_xy(mean_rarh, cov_rarh, n_sigma=2.0)
    ax_e.plot(np.asarray(x1), np.asarray(y1), "-", color=ps.OI["purple"], lw=1.8,
              label=r"$2\sigma$ $(r_a, r_h)$")
    ax_e.fill(np.asarray(x1), np.asarray(y1), color=ps.OI["purple"], alpha=0.10)
    ax_e.plot(ra_hat, rh_hat, "o", color=ps.OI["black"], ms=5, zorder=5,
              label=r"MLE $\hat\theta$")
    ax_e.plot(R_A_TRUE, R_H_TRUE, "*", color=ps.OI["orange"], ms=13, zorder=6,
              mec=ps.OI["black"], mew=0.5, label="truth")
    ax_e.set_xlabel(r"$r_a$ (halo OM radius)")
    ax_e.set_ylabel(r"$r_h$ (Plummer half-mass)")
    rho_rarh = float(cov[1, 2] / np.sqrt(cov[1, 1] * cov[2, 2]))
    rho_tra = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
    ax_e.legend(loc="best")
    ps.panel_label(ax_e, "(d)")

    # Inset: the second requested pair (t, r_a) 2-sigma ellipse.
    axin = ax_e.inset_axes([0.62, 0.10, 0.35, 0.35])
    x2, y2 = _ellipse_xy(mean_tra, cov_tra, n_sigma=2.0)
    axin.plot(np.asarray(x2), np.asarray(y2), "-", color=ps.OI["green"], lw=1.4)
    axin.fill(np.asarray(x2), np.asarray(y2), color=ps.OI["green"], alpha=0.10)
    axin.plot(t_hat, ra_hat, "o", color=ps.OI["black"], ms=3, zorder=5)
    axin.plot(T_TRUE, R_A_TRUE, "*", color=ps.OI["orange"], ms=8, zorder=6,
              mec=ps.OI["black"], mew=0.4)
    axin.set_xlabel(r"$t$", fontsize=7, labelpad=1)
    axin.set_ylabel(r"$r_a$", fontsize=7, labelpad=1)
    axin.tick_params(labelsize=6)
    axin.set_title(r"$2\sigma\,(t, r_a)$", fontsize=7, pad=2)

    cap = (rf"$\hat t={t_hat:.3f}\pm{float(sigma_theta[0]):.3f},\ "
           rf"\hat r_a={ra_hat:.3f}\pm{float(sigma_theta[1]):.3f},\ "
           rf"\hat r_h={rh_hat:.3f}\pm{float(sigma_theta[2]):.3f}$; "
           rf"$\rho(r_a,r_h)={rho_rarh:+.2f},\ \rho(t,r_a)={rho_tra:+.2f}$")
    fig.text(0.5, -0.01, cap, ha="center", va="top", fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, out_dir, "demo_halo_core")
    print(f"\nfigure -> {out_dir}/demo_halo_core.png (+ .pdf)")
    return rho_rarh, rho_tra


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def main():
    print("=" * 72)
    print("B3 demo (Task 7): halo+core Engine-B (t, r_a, r_h) MLE + Fisher")
    print("=" * 72)
    print(f"truth: t={T_TRUE}, r_a={R_A_TRUE}, r_h={R_H_TRUE}; "
          f"core EFF(a={EFF_A}, gamma={EFF_GAMMA}, r_t={EFF_RT}), "
          f"m_j={M_J}, N_STARS={N_STARS}, G={G:.5g}")

    data = build_truth_data()
    sig_hat = data["sig_hat"]
    J = sig_hat.shape[0]
    weight = data["weight"]
    print(f"\nR_CUT (cluster edge = max radius) = {data['r_cut']:.4f}")
    print("R_EDGES (16 quantile bins on [0, R_CUT]) =",
          [round(float(x), 4) for x in data["r_edges"]])
    print("per-component sigma-bin counts (component x bin):")
    n = data["n"]
    resolved = []
    for j in range(J):
        nb = int(jnp.sum(weight[j] > 0))
        resolved.append(nb)
        print(f"  component {j}: {[int(x) for x in n[j]]}  "
              f"({nb}/{sig_hat.shape[1]} resolved)")

    # Self-consistency BEFORE the budget gate.
    max_dev = self_consistency_check(data)
    sc_ok = max_dev < 4.0
    print(f"self-consistency {'OK' if sc_ok else 'FAIL'} "
          f"(max |dev/se| = {max_dev:.2f}, threshold 4.0)")
    if not sc_ok:
        print("\nSTOP: predict(truth) does not match the data -- miscalibrated "
              "oracle or binning. Not proceeding.")
        sys.exit(1)

    # Budget gate: one warm jit(value_and_grad).
    negloglike = make_negloglike(data)
    loss_and_grad = jax.jit(jax.value_and_grad(negloglike))
    z0 = _z_truth()

    t0 = time.perf_counter()
    v, g = loss_and_grad(z0)
    v.block_until_ready(); g.block_until_ready()
    t_compile = time.perf_counter() - t0

    z1 = z0 + jnp.array([0.2, 0.15, -0.1])
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
              "Report measured numbers + options to the orchestrator.")
        sys.exit(2)

    # MLE recovery.
    print("\n" + "=" * 72)
    print("MLE RECOVERY (Adam, %d dispersed inits)" % N_INITS)
    print("=" * 72)
    z_hat, best_trace, finals, i_best, z0s = run_mle(negloglike)
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

    _, g_hat = loss_and_grad(z_hat)
    print(f"grad norm at z_hat = {float(jnp.linalg.norm(g_hat)):.4e} "
          "(interior-optimum check)")

    # theta_hat + Gauss-Newton Fisher errors.
    theta_hat = _theta_of_z(z_hat)
    residual_fn = make_residual_fn(data)
    F_z = di.fisher_information_gn(residual_fn, z_hat)
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

    # Realizability at theta_hat: rebuild Engine B, report f_min_j.
    st_hat = _engine_b(theta_hat)
    f_min_hat = st_hat.f_min_j
    min_fmin = float(jnp.min(f_min_hat))
    print(f"\nRealizability at theta_hat: f_min_j = "
          f"{[float(x) for x in f_min_hat]}, min = {min_fmin:.4e}")

    # GATES.
    print("\n" + "=" * 72)
    print("GATES")
    print("=" * 72)
    recovery_ok = all(abs(p) < 3.0 for p in pulls)
    names = ("t", "r_a", "r_h")
    for nm, p in zip(names, pulls):
        print(f"  3-sigma {nm}: |pull| = {abs(p):.3f} "
              f"{'<' if abs(p) < 3.0 else '>='} 3  "
              f"({'PASS' if abs(p) < 3.0 else 'FAIL'})")
    realiz_ok = min_fmin >= -1e-3
    print(f"  realizability: min(f_min_j) = {min_fmin:.4e} "
          f"{'>=' if realiz_ok else '<'} -1e-3  "
          f"({'PASS' if realiz_ok else 'FAIL'})")
    occ_ok = all(nb >= MIN_BINS_PER_CMP for nb in resolved)
    for j, nb in enumerate(resolved):
        print(f"  occupancy comp {j}: {nb} resolved sigma bins "
              f"{'>=' if nb >= MIN_BINS_PER_CMP else '<'} {MIN_BINS_PER_CMP}  "
              f"({'PASS' if nb >= MIN_BINS_PER_CMP else 'FAIL'})")
    print(f"  plateau: {'PASS' if plat_ok else 'FAIL'}")
    print(f"  recovery (3-sigma, all params): {'PASS' if recovery_ok else 'FAIL'}")

    if not occ_ok:
        print("\nSTOP: a component has too few resolved sigma bins to constrain "
              "the fit. NOT weakening n_min or the bin count -- report.")
        sys.exit(4)   # stop BEFORE the figure (match the sc/budget early exits)

    # Figure.
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "validation", "plots")
    os.makedirs(out_dir, exist_ok=True)
    rho_rarh, rho_tra = make_figure(data, theta_hat, sigma_theta, cov_theta, out_dir)
    print(f"Fisher correlations: rho(r_a, r_h) = {rho_rarh:+.3f}, "
          f"rho(t, r_a) = {rho_tra:+.3f}")

    all_ok = (sc_ok and grad_finite and grad_nonzero and budget_ok
              and plat_ok and occ_ok and recovery_ok and F_pd and realiz_ok)
    print("\n" + "=" * 72)
    print(f"OVERALL {'ALL PASS' if all_ok else 'FAIL'}")
    print("=" * 72)
    if not all_ok and not recovery_ok:
        print("\nNOTE: the 3-sigma recovery gate is REAL. A >3-sigma miss is a "
              "PHYSICS finding -- do NOT widen the gate. Report the table above.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
