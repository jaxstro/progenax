r"""B2d science demo (Task 6): wrong-IMF bias curve + alpha_true robustness grid.

Sibling of ``demo_delta_recovery.py`` (Tasks 3-5): that script's default run is
the budgeted single-dataset MLE headline, so the Task 6 ENSEMBLES live in this
separate gated CLI (the plan's "sibling script" branch). It reuses the B2
machinery verbatim -- truth constants, ``build_truth_data``, ``predict_binned``,
the expit boxes, and the ``_demo_inference`` Adam/Fisher helpers -- nothing is
re-derived here.

Two ensembles
-------------
1. **Wrong-IMF bias curve** (REPORTED, not gated). For each
   ``alpha_assumed in {1.9, 2.1, 2.3, 2.5, 2.7}``: a KINEMATICS-ONLY refit of
   ``(delta, W0)`` -- the mass channel DROPPED, alpha FROZEN at
   ``alpha_assumed`` -- on mock data generated at the truth
   ``(alpha=2.3, delta=0.4, W0=5)``. ``N_SEEDS`` independent truth datasets
   (different sampling keys, SAME dataset reused across the 5 alpha_assumed
   within a seed, so the per-seed slope is a paired measurement). The quoted
   slope ``d delta_hat / d alpha_assumed`` is a per-seed linear fit, reported
   as mean +/- SE over the seed ensemble. It is REPORTED rather than gated
   because it is a sensitivity MEASUREMENT of this mock configuration -- there
   is no published reference value for the equipartition-vs-IMF-slope bias of
   a 4-component LIMEPY fit to assert against; the demo's claim is the number
   itself, with an honest seed-scatter uncertainty.
2. **Robustness grid** (GATED). ``alpha_true in {1.9, 2.3, 2.7}`` with a FRESH
   truth dataset at each ``alpha_true`` (same ``delta=0.4, W0=5``), then the
   FULL joint ``(alpha, delta, W0)`` refit WITH the mass term (3 dispersed
   inits, exactly the Task 4 recipe). Gate: each recovery within 3 sigma_hat
   componentwise (Gauss-Newton Fisher errors via ``fisher_information_gn`` +
   ``constrained_cov``). The script exits nonzero if any component misses.

Runtime budget (STOP checkpoint, executed BEFORE the ensembles run)
-------------------------------------------------------------------
``N_SEEDS`` is scaled to the MEASURED warm eval cost so the whole bias panel
stays under ``BIAS_BUDGET_MIN = 30`` min. The script measures (i) one truth
dataset build and (ii) one warm ``jit(value_and_grad)`` of the 2-parameter
kinematics-only loss, then projects

    T_bias(N_SEEDS) ~ N_SEEDS * [ t_data + N_ALPHA * N_ADAM_STEPS * t_warm ]

and picks ``N_SEEDS = clamp(floor(budget / per-seed), 3, 5)``. The arithmetic
is PRINTED before any ensemble fit runs; if even 3 seeds do not fit the
budget the script STOPs (exit 2) and reports options instead of degrading.

Captured headline run (2026-06-11, CPU/float64; authoritative run record
/tmp/b2d_run.log): t_data = 6.1 s; one warm 2-param kin-only FIT (300 fused
Adam steps) = 64.3 s (the projection uses the measured warm-FIT cost, not
300 x a lone warm value_and_grad, which understates the per-step cost).
Per-seed ~5.46 min -> N_SEEDS = 5, PROJECTED 27.3 min < 30 min budget;
bias-ensemble ACTUAL wall 30.3 min (post-hoc machine-load variance -- the
projection BEFORE the run is the budget gate, and it passed). Robustness
grid: measured 13.6 min. NOTE that ``N_SEEDS`` is LOAD-ADAPTIVE: the budget
arithmetic picks 3-5 seeds from the warm-fit cost measured at run time, so
a rerun on a loaded machine may legitimately produce a 4-seed figure (the
captured headline run measured 64.3 s -> 5 seeds).

Peaked-shape caveat (captured run): the measured ensemble-mean
``delta_hat(alpha_assumed)`` = (0.138, 0.393, 0.420, 0.285, 0.187) is
NON-MONOTONIC and peaked AT the truth alpha=2.3, so the linear slope is not
an adequate summary by itself -- wrong alpha in EITHER direction biases
delta_hat LOW (by ~0.2-0.3 at |Delta alpha| = 0.4). The script therefore
also prints the peak-to-end drops alongside the slope and annotates them on
figure panel (a).

CLI ``--diag-21`` (valley diagnostic; runs NO ensemble and NO gate)
-------------------------------------------------------------------
The captured bias curve has its largest seed scatter at alpha_assumed=2.1
(seed std 0.084 vs <=0.016 elsewhere; seeds 0 and 1 split delta_hat ~0.47
vs ~0.27). ``--diag-21`` probes whether that split is an optimizer artifact
or a real likelihood feature: (i) it profiles the kinematics-only
negloglike along delta (25 points over ``DELTA_BOX``) at alpha_assumed=2.1
with W0 FIXED at each seed's fitted value, for seeds 0 and 1, printing the
profile values so the valley shape (single-well vs flat/double-well) is in
the output; (ii) it refits alpha_assumed=2.1 for all 5 headline seed
datasets from 3 dispersed inits (``b2.dispersed_inits`` adapted to 2
params) with 600 Adam steps, printing per-seed per-init
(delta_hat, W0_hat, final loss). Interpretation rule: same-dataset
init-dependence => optimizer artifact; init-independence with
seed-dependence => genuinely flat/bistable likelihood direction (a
reportable finding). The headline ensemble numbers and gates are untouched.

Bias-curve refit design notes
-----------------------------
* alpha is frozen by building ``theta = (alpha_assumed, expit(z[0], DELTA_BOX),
  expit(z[1], W0_BOX))`` inside the loss -- the optimizer sees only the
  2-vector ``z``; no gradient flows to alpha.
* Single init ``z = 0`` (delta=0.35, W0=5.5): Task 4 showed the joint 3-param
  landscape is init-insensitive (3 dispersed inits, identical optimum); the
  kinematics-only 2-param problem is a slice of it. Each fit's Adam trace is
  plateau-checked (reported; a non-plateau run is flagged in the table).
* The fit driver is ONE jitted function over (alpha_assumed, data arrays), so
  XLA compiles once and all 5 x N_SEEDS fits reuse the executable; the
  robustness grid likewise compiles its joint driver once for 3 x 3 fits.
"""

import os
import sys
import time

import jax
import jax.numpy as jnp

import progenax  # noqa: F401  -- enables float64 at import
from progenax.imf.smooth import Maschberger

sys.path.insert(0, os.path.dirname(__file__))
import _demo_inference as di  # noqa: E402
import demo_delta_recovery as b2  # noqa: E402  (Tasks 3-5 machinery, reused)

# --------------------------------------------------------------------------- #
# Task 6 configuration
# --------------------------------------------------------------------------- #
ALPHAS_ASSUMED = (1.9, 2.1, 2.3, 2.5, 2.7)   # bias curve (truth alpha = 2.3)
ALPHAS_TRUE_GRID = (1.9, 2.3, 2.7)           # robustness grid truths
SEED_BASE_BIAS = 100                          # PRNGKey(100 + s), s = 0..N_SEEDS-1
SEED_BASE_GRID = 200                          # PRNGKey(200 + i) per alpha_true
N_SEEDS_MIN, N_SEEDS_MAX = 3, 5
BIAS_BUDGET_MIN = 30.0                        # whole-bias-panel budget (STOP gate)


# --------------------------------------------------------------------------- #
# Kinematics-only (delta, W0) refit with alpha FROZEN (bias curve)
# --------------------------------------------------------------------------- #
def _kin_negloglike_2p(z2, alpha_assumed, sig_hat, se, weight, r_edges, m_fixed):
    """Kinematics-only negloglike in z2 = (z_delta, z_W0); alpha frozen, mass
    term DROPPED. Same standardized chi^2 as the joint loss's kinematic part."""
    theta = (alpha_assumed,
             di.expit(z2[0], *b2.DELTA_BOX),
             di.expit(z2[1], *b2.W0_BOX))
    sig_model = b2.predict_binned(theta, r_edges, m_fixed)
    safe_se = jnp.where(se > 0, se, 1.0)
    resid = jnp.sqrt(weight) * (sig_hat - sig_model) / safe_se
    return 0.5 * jnp.sum(resid * resid)


def _fit_kin_only(alpha_assumed, sig_hat, se, weight, r_edges, m_fixed):
    """One kinematics-only (delta, W0) Adam fit (Task 4 step count / lr).
    Jitted ONCE at module scope (below); all 5 x N_SEEDS fits reuse it."""
    def negll(z2):
        return _kin_negloglike_2p(z2, alpha_assumed, sig_hat, se, weight,
                                  r_edges, m_fixed)

    return di.mle_adam(negll, jnp.zeros(2),
                       n_steps=b2.N_ADAM_STEPS, lr=b2.ADAM_LR)


fit_kin_only = jax.jit(_fit_kin_only)
kin_negll_2p = jax.jit(_kin_negloglike_2p)


# --------------------------------------------------------------------------- #
# --diag-21: valley diagnostic at alpha_assumed = 2.1 (see module docstring)
# --------------------------------------------------------------------------- #
DIAG_ALPHA = 2.1
DIAG_N_DELTA = 25
DIAG_PROFILE_SEEDS = (0, 1)   # the seeds that split: delta_hat ~0.47 vs ~0.27
DIAG_N_SEEDS = 5              # the captured headline run's seeds (keys 100-104)
DIAG_N_STEPS = 600            # 2x the ensemble's 300 (rules out slow convergence)


def dispersed_inits_2p():
    """``b2.dispersed_inits`` adapted to the 2-param (delta, W0) refit:
    z0 = 0 plus ``N_INITS - 1`` draws from N(0, INIT_SCALE^2 I_2), same
    INIT_KEY / INIT_SCALE as the joint-refit recipe."""
    key = jax.random.PRNGKey(b2.INIT_KEY)
    draws = jax.random.normal(key, (b2.N_INITS - 1, 2)) * b2.INIT_SCALE
    return jnp.concatenate([jnp.zeros((1, 2)), draws], axis=0)


def _kin_negloglike_theta(delta, w0, alpha_assumed, sig_hat, se, weight,
                          r_edges, m_fixed):
    """The SAME kinematics-only negloglike as ``_kin_negloglike_2p`` but in
    PHYSICAL (delta, W0) -- used to profile the likelihood on a delta grid."""
    sig_model = b2.predict_binned((alpha_assumed, delta, w0), r_edges, m_fixed)
    safe_se = jnp.where(se > 0, se, 1.0)
    resid = jnp.sqrt(weight) * (sig_hat - sig_model) / safe_se
    return 0.5 * jnp.sum(resid * resid)


profile_negll = jax.jit(jax.vmap(
    _kin_negloglike_theta,
    in_axes=(0, None, None, None, None, None, None, None)))


def _fit_kin_only_diag(z0, alpha_assumed, sig_hat, se, weight, r_edges,
                       m_fixed):
    """Kinematics-only fit from an EXPLICIT init with DIAG_N_STEPS Adam steps
    (the ensemble fit is z0=0 with N_ADAM_STEPS)."""
    def negll(z2):
        return _kin_negloglike_2p(z2, alpha_assumed, sig_hat, se, weight,
                                  r_edges, m_fixed)

    return di.mle_adam(negll, z0, n_steps=DIAG_N_STEPS, lr=b2.ADAM_LR)


fit_kin_only_diag = jax.jit(_fit_kin_only_diag)


def diag_alpha21():
    """--diag-21 driver: (i) delta profile at fixed W0_hat for seeds 0 and 1;
    (ii) 3-dispersed-init x 600-step refits for all 5 headline seed datasets.
    Runs NO ensemble and NO gate; headline numbers untouched."""
    t_all0 = time.perf_counter()
    print("=" * 72)
    print(f"DIAG-21: alpha_assumed = {DIAG_ALPHA} valley diagnostic "
          "(kinematics-only refit)")
    print("=" * 72)
    print(f"context: the captured bias curve has its largest seed scatter at "
          f"alpha_assumed={DIAG_ALPHA} (seed std 0.084 vs <=0.016 elsewhere; "
          f"seeds 0/3 delta_hat~0.47 vs seeds 1/2/4 ~0.27-0.37).")
    print("question: optimizer artifact or genuine likelihood feature?")

    datasets = {}

    def get_data(s):
        if s not in datasets:
            datasets[s] = b2.build_truth_data(
                key=jax.random.PRNGKey(SEED_BASE_BIAS + s))
        return datasets[s]

    def args_of(data):
        return (data["sig_hat"], data["se"], data["weight"],
                data["r_edges"], data["M_fixed"])

    # ------------------------------------------------------------------ #
    # (i) negloglike profile along delta, W0 FIXED at each seed's fit
    # ------------------------------------------------------------------ #
    print("\n" + "-" * 72)
    print(f"(i) negloglike profile along delta ({DIAG_N_DELTA} points over "
          f"DELTA_BOX={b2.DELTA_BOX}), W0 FIXED at each seed's fitted value")
    print("-" * 72)
    deltas = jnp.linspace(b2.DELTA_BOX[0], b2.DELTA_BOX[1], DIAG_N_DELTA)
    for s in DIAG_PROFILE_SEEDS:
        data = get_data(s)
        args = args_of(data)
        # The ensemble fit itself (300 steps, z0 = 0) -> this seed's W0_hat.
        z2_hat, _ = fit_kin_only(jnp.asarray(DIAG_ALPHA), *args)
        d_hat = float(di.expit(z2_hat[0], *b2.DELTA_BOX))
        w_hat = float(di.expit(z2_hat[1], *b2.W0_BOX))
        vals = profile_negll(deltas, jnp.asarray(w_hat),
                             jnp.asarray(DIAG_ALPHA), *args)
        v = [float(x) for x in vals]
        print(f"\n  seed {s} (key {SEED_BASE_BIAS + s}): ensemble fit "
              f"delta_hat={d_hat:.4f}, W0_hat={w_hat:.4f} (W0 frozen below)")
        print(f"  {'delta':>8} {'negloglike':>14}")
        for k in range(DIAG_N_DELTA):
            mark = ""
            if 0 < k < DIAG_N_DELTA - 1 and v[k] < v[k - 1] and v[k] < v[k + 1]:
                mark = "  <- local min"
            print(f"  {float(deltas[k]):>8.4f} {v[k]:>14.4f}{mark}")
        i_min = int(jnp.argmin(vals))
        print(f"  grid minimum at delta = {float(deltas[i_min]):.4f} "
              f"(negloglike {v[i_min]:.4f}); profile span max-min = "
              f"{max(v) - min(v):.4f}")

    # ------------------------------------------------------------------ #
    # (ii) dispersed-init refits, all headline seeds
    # ------------------------------------------------------------------ #
    print("\n" + "-" * 72)
    print(f"(ii) refit alpha_assumed={DIAG_ALPHA} for all {DIAG_N_SEEDS} seed "
          f"datasets: {b2.N_INITS} dispersed inits x {DIAG_N_STEPS} Adam steps")
    print("-" * 72)
    z0s = dispersed_inits_2p()
    print(f"  inits (unconstrained z = (z_delta, z_W0)): "
          f"{[[round(float(x), 3) for x in z0] for z0 in z0s]}")
    spreads = []
    best_d = []
    for s in range(DIAG_N_SEEDS):
        data = get_data(s)
        args = args_of(data)
        d_hats, finals = [], []
        print(f"\n  seed {s} (key {SEED_BASE_BIAS + s}):")
        for i in range(b2.N_INITS):
            z2_hat, trace = fit_kin_only_diag(z0s[i], jnp.asarray(DIAG_ALPHA),
                                              *args)
            d = float(di.expit(z2_hat[0], *b2.DELTA_BOX))
            w = float(di.expit(z2_hat[1], *b2.W0_BOX))
            final = float(kin_negll_2p(z2_hat, jnp.asarray(DIAG_ALPHA), *args))
            plat, _, _ = b2.plateau_ok(trace)
            d_hats.append(d)
            finals.append(final)
            print(f"    init {i}: delta_hat={d:.4f}, W0_hat={w:.4f}, "
                  f"final loss={final:.4f}, plateau "
                  f"{'PASS' if plat else 'FAIL'}")
        spread = max(d_hats) - min(d_hats)
        spreads.append(spread)
        best_d.append(d_hats[int(jnp.argmin(jnp.array(finals)))])
        print(f"    across-init delta_hat spread = {spread:.4f} "
              f"(best-loss init's delta_hat = {best_d[-1]:.4f})")

    print(f"\n  across-INIT spread per seed (max) = {max(spreads):.4f}")
    print(f"  across-SEED spread of best-loss delta_hat = "
          f"{max(best_d) - min(best_d):.4f}")
    print("\n" + "=" * 72)
    print("INTERPRETATION RULE")
    print("=" * 72)
    print("  same-dataset INIT-dependence of delta_hat => optimizer artifact")
    print("  init-INDEPENDENCE with seed-dependence    => genuinely flat / "
          "bistable likelihood direction at alpha_assumed=2.1 "
          "(a reportable finding, not an optimizer bug)")
    print(f"\n  diag wall-time = {(time.perf_counter() - t_all0) / 60.0:.1f} "
          "min")
    print("  (headline ensemble numbers and gates untouched -- this mode runs "
          "NO ensemble and NO gate)")


# --------------------------------------------------------------------------- #
# Full joint (alpha, delta, W0) refit (robustness grid) -- Task 4 recipe
# --------------------------------------------------------------------------- #
def _joint_negloglike(z, sig_hat, se, weight, r_edges, m_fixed, m_obs):
    """The Task 3 joint negloglike (kinematics + Option A mass channel), with
    the data arrays as arguments so the jitted fit compiles once."""
    theta = b2._theta_of_z(z)
    sig_model = b2.predict_binned(theta, r_edges, m_fixed)
    safe_se = jnp.where(se > 0, se, 1.0)
    resid = jnp.sqrt(weight) * (sig_hat - sig_model) / safe_se
    alpha = theta[0]
    mass_nll = -jnp.sum(
        Maschberger(alpha=alpha, m_min=b2.M_RANGE[0],
                    m_max=b2.M_RANGE[1]).logpdf(m_obs))
    return 0.5 * jnp.sum(resid * resid) + mass_nll


def _fit_joint(z0, sig_hat, se, weight, r_edges, m_fixed, m_obs):
    def negll(z):
        return _joint_negloglike(z, sig_hat, se, weight, r_edges, m_fixed, m_obs)

    return di.mle_adam(negll, z0, n_steps=b2.N_ADAM_STEPS, lr=b2.ADAM_LR)


fit_joint = jax.jit(_fit_joint)


def joint_refit(data):
    """Full joint refit on one dataset: 3 dispersed inits -> best z_hat,
    theta_hat, Gauss-Newton Fisher sigma_theta, plateau flag. Task 4 recipe via
    the b2 builders (make_residual_fn / make_mass_negloglike / fisher_*)."""
    args = (data["sig_hat"], data["se"], data["weight"],
            data["r_edges"], data["M_fixed"], data["m_obs"])
    z0s = b2.dispersed_inits()
    z_hats, traces, finals = [], [], []
    for i in range(b2.N_INITS):
        z_hat, trace = fit_joint(z0s[i], *args)
        z_hats.append(z_hat)
        traces.append(trace)
        finals.append(float(_joint_negloglike(z_hat, *args)))
    i_best = int(jnp.argmin(jnp.array(finals)))
    z_hat, trace = z_hats[i_best], traces[i_best]
    plat_ok, _, _ = b2.plateau_ok(trace)

    residual_fn = b2.make_residual_fn(data)
    mass_negloglike = b2.make_mass_negloglike(data)
    F_z = di.fisher_information_gn(residual_fn, z_hat,
                                   extra_negloglike=mass_negloglike)
    cov_theta = di.constrained_cov(F_z, b2._dtheta_dz(z_hat))  # raises if not PD
    sigma_theta = jnp.sqrt(jnp.diag(cov_theta))
    theta_hat = b2._theta_of_z(z_hat)
    return theta_hat, sigma_theta, plat_ok, finals, i_best


# --------------------------------------------------------------------------- #
# Figure: (a) bias curve  (b) robustness-grid pulls
# --------------------------------------------------------------------------- #
def make_bias_figure(bias, grid, out_dir):
    """Two-panel Task 6 figure via _plotstyle.

    Panel (a): delta_hat(alpha_assumed) per seed (light points) + ensemble
    mean +/- seed std, truth lines, quoted slope. Panel (b): robustness-grid
    pulls (theta_hat - truth)/sigma_hat per parameter vs alpha_true, with the
    +/- 3 sigma gate band.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    import _plotstyle as ps  # noqa: E402  (path already inserted)

    ps.apply_pub_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.5, 3.6))

    # ---- panel (a): bias curve --------------------------------------------- #
    alphas = np.asarray(bias["alphas"])                   # (A,)
    d_hat = np.asarray(bias["delta_hat"])                 # (S, A)
    d_mean = d_hat.mean(axis=0)
    d_std = d_hat.std(axis=0, ddof=1)
    for s in range(d_hat.shape[0]):
        axa.plot(alphas, d_hat[s], "o", ms=3, color=ps.OI["sky"], alpha=0.6,
                 mec="none", zorder=2,
                 label="per-seed fits" if s == 0 else None)
    axa.errorbar(alphas, d_mean, yerr=d_std, fmt="s-", ms=4.5,
                 color=ps.OI["blue"], capsize=2.0, elinewidth=1.0, lw=1.4,
                 zorder=3, label=r"seed mean $\pm$ std")
    axa.axhline(b2.DELTA_TRUE, color=ps.OI["orange"], lw=1.0, ls=":",
                label=rf"truth $\delta={b2.DELTA_TRUE}$")
    axa.axvline(b2.ALPHA_TRUE, color=ps.OI["orange"], lw=1.0, ls=":")
    axa.set_xlabel(r"$\alpha_{\rm assumed}$ (frozen IMF slope)")
    axa.set_ylabel(r"$\hat\delta$ (kinematics-only refit)")
    axa.legend(loc="best")
    axa.text(0.97, 0.05,
             rf"$\mathrm{{d}}\hat\delta/\mathrm{{d}}\alpha = "
             rf"{bias['slope_mean']:+.4f} \pm {bias['slope_se']:.4f}$",
             transform=axa.transAxes, ha="right", va="bottom", fontsize=9)
    axa.text(0.97, 0.13,
             rf"peaked at truth: wrong $\alpha$ either way biases "
             rf"$\hat\delta$ low" "\n"
             rf"($-{bias['drop_lo']:.2f}$ at $\alpha={alphas[0]:.1f}$, "
             rf"$-{bias['drop_hi']:.2f}$ at $\alpha={alphas[-1]:.1f}$)",
             transform=axa.transAxes, ha="right", va="bottom", fontsize=7.5)
    ps.panel_label(axa, "(a)")

    # ---- panel (b): robustness-grid pulls ---------------------------------- #
    a_true = np.asarray(grid["alphas_true"])              # (G,)
    pulls = np.asarray(grid["pulls"])                     # (G, 3)
    names = (r"$\alpha$", r"$\delta$", r"$W_0$")
    colors = (ps.OI["blue"], ps.OI["green"], ps.OI["vermilion"])
    markers = ("o", "s", "^")
    off = (-0.02, 0.0, 0.02)
    axb.axhspan(-3.0, 3.0, color=ps.OI["sky"], alpha=0.15, zorder=0)
    axb.axhline(0.0, color=ps.OI["black"], lw=0.7, ls="-", zorder=1)
    for p in range(3):
        axb.plot(a_true + off[p], pulls[:, p], markers[p], ms=5,
                 color=colors[p], mec=ps.OI["black"], mew=0.4, zorder=3,
                 label=names[p])
    for y in (-3.0, 3.0):
        axb.axhline(y, color=ps.OI["vermilion"], lw=0.8, ls="--", zorder=1)
    axb.set_xlabel(r"$\alpha_{\rm true}$ (fresh truth dataset)")
    axb.set_ylabel(r"pull $(\hat\theta - \theta_{\rm true})/\hat\sigma$")
    axb.set_xticks(a_true)
    axb.set_ylim(-4.0, 4.0)
    axb.legend(loc="best", ncol=3)
    ps.panel_label(axb, "(b)")

    cap = (rf"(a) wrong-IMF bias of the kinematics-only $(\delta, W_0)$ refit, "
           rf"{d_hat.shape[0]} seeds; (b) joint refit pulls vs $\alpha_{{\rm true}}$ "
           rf"($\pm3\hat\sigma$ gate band)")
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, out_dir, "demo_delta_recovery_bias")
    print(f"\nbias figure -> {out_dir}/demo_delta_recovery_bias.png (+ .pdf)")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    print("=" * 72)
    print("B2d demo (Task 6): wrong-IMF bias curve + alpha_true robustness grid")
    print("=" * 72)
    print(f"truth: alpha={b2.ALPHA_TRUE}, delta={b2.DELTA_TRUE}, W0={b2.W0_TRUE}; "
          f"N_COMP={b2.N_COMP}, N_STARS={b2.N_STARS}, M_RANGE={b2.M_RANGE}")
    print(f"bias curve: alpha_assumed in {ALPHAS_ASSUMED} (kinematics-only, "
          f"alpha frozen, mass term dropped)")
    print(f"robustness grid: alpha_true in {ALPHAS_TRUE_GRID} (fresh datasets, "
          f"full joint refit; GATED at 3 sigma)")

    # ------------------------------------------------------------------ #
    # RUNTIME BUDGET CHECKPOINT (before any ensemble fit runs)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print(f"RUNTIME BUDGET CHECKPOINT (bias panel < {BIAS_BUDGET_MIN:.0f} min)")
    print("=" * 72)
    t0 = time.perf_counter()
    data0 = b2.build_truth_data(key=jax.random.PRNGKey(SEED_BASE_BIAS))
    t_data = time.perf_counter() - t0
    print(f"  one truth-dataset build (sample + bin)  t_data = {t_data:.1f} s")

    args0 = (data0["sig_hat"], data0["se"], data0["weight"],
             data0["r_edges"], data0["M_fixed"])
    # Probe with TWO REAL fits (reused below as the first two seed-0 ensemble
    # members, so no probe time is wasted). The projection uses the measured
    # warm FIT cost rather than N_ADAM_STEPS x (one warm eval): measured here,
    # the fused Adam scan costs slightly MORE per step than a lone
    # value_and_grad (0.28 vs 0.23 s on this machine), so the per-eval
    # arithmetic would UNDERSTATE the panel.
    t0 = time.perf_counter()
    z2_p0, trace_p0 = fit_kin_only(jnp.asarray(ALPHAS_ASSUMED[0]), *args0)
    z2_p0.block_until_ready()
    t_fit_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    z2_p1, trace_p1 = fit_kin_only(jnp.asarray(ALPHAS_ASSUMED[1]), *args0)
    z2_p1.block_until_ready()
    t_fit_warm = time.perf_counter() - t0
    probe_fits = {(0, 0): (z2_p0, trace_p0), (0, 1): (z2_p1, trace_p1)}
    print(f"  2-param kin-only fit ({b2.N_ADAM_STEPS} Adam steps): "
          f"cold (incl. compile) {t_fit_cold:.1f} s, warm {t_fit_warm:.1f} s")

    n_alpha = len(ALPHAS_ASSUMED)
    per_seed = t_data + n_alpha * t_fit_warm
    print(f"  per-seed ~ t_data + {n_alpha} x warm-fit = {t_data:.1f} + "
          f"{n_alpha} x {t_fit_warm:.1f} = {per_seed:.1f} s "
          f"= {per_seed / 60.0:.2f} min")
    n_seeds_fit = int((BIAS_BUDGET_MIN * 60.0) // per_seed)
    n_seeds = max(N_SEEDS_MIN, min(N_SEEDS_MAX, n_seeds_fit))
    proj_min = n_seeds * per_seed / 60.0
    print(f"  budget arithmetic: floor({BIAS_BUDGET_MIN:.0f} min / "
          f"{per_seed / 60.0:.2f} min) = {n_seeds_fit} seeds -> "
          f"N_SEEDS = clamp(.., {N_SEEDS_MIN}, {N_SEEDS_MAX}) = {n_seeds}")
    print(f"  PROJECTED bias-panel wall-time = {n_seeds} x {per_seed / 60.0:.2f} "
          f"= {proj_min:.1f} min (budget {BIAS_BUDGET_MIN:.0f} min; "
          f"data build + 2 probe fits already banked)")
    if n_seeds_fit < N_SEEDS_MIN:
        print(f"\n  STOP: even {N_SEEDS_MIN} seeds project to "
              f"{N_SEEDS_MIN * per_seed / 60.0:.1f} min > "
              f"{BIAS_BUDGET_MIN:.0f} min. NOT degrading the solve. Options: "
              "reduce N_ADAM_STEPS for the 2-param refit (with a plateau "
              "re-check), reduce the alpha_assumed grid to 3 points, or raise "
              "the budget with Anna's ok.")
        sys.exit(2)
    # Informational projection for the (un-budgeted) robustness grid: the
    # joint loss adds the (cheap) mass channel + a 3rd parameter; ~1.3x the
    # kin-only fit cost is a generous per-fit allowance.
    grid_proj = (len(ALPHAS_TRUE_GRID)
                 * (t_data + b2.N_INITS * 1.3 * t_fit_warm)) / 60.0
    print(f"  robustness-grid projection (informational, ~1.3x kin fit cost "
          f"x {b2.N_INITS} inits x {len(ALPHAS_TRUE_GRID)} datasets): "
          f"~{grid_proj:.0f} min + one compile + 3 Fisher evals")

    # ------------------------------------------------------------------ #
    # Ensemble 1: wrong-IMF bias curve (REPORTED, not gated)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print(f"BIAS CURVE: kinematics-only (delta, W0) refit, {n_seeds} seeds")
    print("=" * 72)
    t_bias0 = time.perf_counter()
    delta_hat = []   # (S, A)
    w0_hat = []      # (S, A)
    n_no_plateau = 0
    for s in range(n_seeds):
        data = data0 if s == 0 else b2.build_truth_data(
            key=jax.random.PRNGKey(SEED_BASE_BIAS + s))
        args = (data["sig_hat"], data["se"], data["weight"],
                data["r_edges"], data["M_fixed"])
        d_row, w_row = [], []
        for ia, a in enumerate(ALPHAS_ASSUMED):
            if (s, ia) in probe_fits:           # banked during the budget probe
                z2_hat, trace = probe_fits[(s, ia)]
            else:
                z2_hat, trace = fit_kin_only(jnp.asarray(a), *args)
            plat, _, _ = b2.plateau_ok(trace)
            if not plat:
                n_no_plateau += 1
            d_row.append(float(di.expit(z2_hat[0], *b2.DELTA_BOX)))
            w_row.append(float(di.expit(z2_hat[1], *b2.W0_BOX)))
            print(f"  seed {s} (key {SEED_BASE_BIAS + s})  "
                  f"alpha_assumed={a:.1f}: delta_hat={d_row[-1]:.4f}, "
                  f"W0_hat={w_row[-1]:.4f}"
                  + ("" if plat else "  [NO PLATEAU]"))
        delta_hat.append(d_row)
        w0_hat.append(w_row)
    t_bias = time.perf_counter() - t_bias0
    delta_hat = jnp.array(delta_hat)
    w0_hat = jnp.array(w0_hat)

    alphas = jnp.asarray(ALPHAS_ASSUMED)
    print(f"\n{'alpha_assumed':>13} {'delta_hat mean':>14} {'seed std':>9} "
          f"{'W0_hat mean':>11} {'seed std':>9}")
    for i, a in enumerate(ALPHAS_ASSUMED):
        print(f"{a:>13.1f} {float(delta_hat[:, i].mean()):>14.4f} "
              f"{float(delta_hat[:, i].std(ddof=1)):>9.4f} "
              f"{float(w0_hat[:, i].mean()):>11.4f} "
              f"{float(w0_hat[:, i].std(ddof=1)):>9.4f}")

    # Per-seed linear slope d delta_hat / d alpha (paired across alpha_assumed).
    a_c = alphas - alphas.mean()
    slopes = (delta_hat @ a_c) / jnp.sum(a_c * a_c)       # (S,) LSQ slopes
    slope_mean = float(slopes.mean())
    slope_std = float(slopes.std(ddof=1))
    slope_se = slope_std / float(jnp.sqrt(n_seeds))
    print(f"\n  per-seed slopes d delta_hat/d alpha = "
          f"{[round(float(x), 4) for x in slopes]}")
    print(f"  SLOPE d delta_hat/d alpha = {slope_mean:+.4f} +/- {slope_se:.4f} "
          f"(SE of mean over {n_seeds} seeds; seed scatter {slope_std:.4f})")
    print("  [REPORTED, not gated: a sensitivity measurement of this mock "
          "configuration -- no published reference value to assert against.]")
    # Peaked-shape summary: the response delta_hat(alpha_assumed) peaks AT the
    # truth, so a linear slope alone is not an adequate summary.
    i_truth = ALPHAS_ASSUMED.index(b2.ALPHA_TRUE)
    d_mean_a = jnp.mean(delta_hat, axis=0)
    drop_lo = float(d_mean_a[i_truth] - d_mean_a[0])    # vs alpha_assumed = 1.9
    drop_hi = float(d_mean_a[i_truth] - d_mean_a[-1])   # vs alpha_assumed = 2.7
    print(f"  peak-to-end drops (ensemble mean): "
          f"delta_hat({b2.ALPHA_TRUE}) - delta_hat({ALPHAS_ASSUMED[0]}) = "
          f"{drop_lo:+.4f}; delta_hat({b2.ALPHA_TRUE}) - "
          f"delta_hat({ALPHAS_ASSUMED[-1]}) = {drop_hi:+.4f} "
          f"(max |Delta delta_hat| = {max(abs(drop_lo), abs(drop_hi)):.4f})")
    print("  the measured response is PEAKED at the truth alpha, so a linear "
          "slope is NOT an adequate summary -- wrong alpha in EITHER "
          "direction biases delta_hat LOW (by ~0.2-0.3 at |Delta alpha| = 0.4).")
    print(f"  plateau: {n_seeds * n_alpha - n_no_plateau}/{n_seeds * n_alpha} "
          f"fits plateaued")
    print(f"  bias-ensemble wall-time = {t_bias / 60.0:.1f} min "
          f"(projected {proj_min:.1f} min)")
    if t_bias / 60.0 > BIAS_BUDGET_MIN:
        print(f"  NOTE: actual wall-time exceeded the {BIAS_BUDGET_MIN:.0f}-min "
              f"target by {t_bias / 60.0 - BIAS_BUDGET_MIN:.1f} min -- "
              "post-hoc machine-load variance; the PROJECTION before the run "
              "is the budget gate (and it passed).")

    # ------------------------------------------------------------------ #
    # Ensemble 2: robustness grid (GATED at 3 sigma componentwise)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("ROBUSTNESS GRID: full joint (alpha, delta, W0) refit per alpha_true")
    print("=" * 72)
    t_grid0 = time.perf_counter()
    grid_pulls, grid_rows = [], []
    grid_ok = True
    for i, a_true in enumerate(ALPHAS_TRUE_GRID):
        print(f"\nalpha_true = {a_true} (key {SEED_BASE_GRID + i}; "
              f"delta={b2.DELTA_TRUE}, W0={b2.W0_TRUE})")
        data = b2.build_truth_data(key=jax.random.PRNGKey(SEED_BASE_GRID + i),
                                   alpha_true=a_true)
        n = data["n"]
        print(f"  top-group occupancy = {int(n[-1].sum())}; "
              f"M_FIXED = {data['M_fixed']:.1f}; R_CUT = {data['r_cut']:.3f}")
        theta_hat, sigma_theta, plat, finals, i_best = joint_refit(data)
        print(f"  init finals = {[round(f, 4) for f in finals]} "
              f"(best: init {i_best}); plateau {'PASS' if plat else 'FAIL'}")
        truths = (a_true, b2.DELTA_TRUE, b2.W0_TRUE)
        pulls = b2.recovery_table(theta_hat, sigma_theta, truths=truths)
        ok = all(abs(p) < 3.0 for p in pulls)
        grid_ok = grid_ok and ok
        print(f"  3-sigma componentwise: {'PASS' if ok else 'FAIL'} "
              f"(max |pull| = {max(abs(p) for p in pulls):.3f})")
        grid_pulls.append(pulls)
        grid_rows.append((a_true, theta_hat, sigma_theta, pulls, ok))
    t_grid = time.perf_counter() - t_grid0
    print(f"\n  robustness-grid wall-time = {t_grid / 60.0:.1f} min")

    print(f"\n{'alpha_true':>10} {'alpha_hat':>16} {'delta_hat':>16} "
          f"{'W0_hat':>16} {'max|pull|':>9} {'gate':>5}")
    for a_true, th, sg, pulls, ok in grid_rows:
        print(f"{a_true:>10.1f} "
              f"{float(th[0]):>9.4f}+/-{float(sg[0]):.4f} "
              f"{float(th[1]):>9.4f}+/-{float(sg[1]):.4f} "
              f"{float(th[2]):>9.4f}+/-{float(sg[2]):.4f} "
              f"{max(abs(p) for p in pulls):>9.3f} "
              f"{'PASS' if ok else 'FAIL':>5}")

    # ------------------------------------------------------------------ #
    # Figure + verdict
    # ------------------------------------------------------------------ #
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "validation", "plots")
    os.makedirs(out_dir, exist_ok=True)
    bias = dict(alphas=alphas, delta_hat=delta_hat, slope_mean=slope_mean,
                slope_se=slope_se, drop_lo=drop_lo, drop_hi=drop_hi)
    grid = dict(alphas_true=jnp.asarray(ALPHAS_TRUE_GRID),
                pulls=jnp.array(grid_pulls))
    make_bias_figure(bias, grid, out_dir)

    print("\n" + "=" * 72)
    print("GATES")
    print("=" * 72)
    print(f"  robustness grid 3-sigma (all alpha_true, componentwise): "
          f"{'PASS' if grid_ok else 'FAIL'}")
    print("  bias curve: REPORTED (slope quoted with seed-ensemble "
          "uncertainty; not gated by design)")
    print("\n" + "=" * 72)
    print(f"OVERALL {'ALL PASS' if grid_ok else 'FAIL'}")
    print("=" * 72)
    if not grid_ok:
        print("\nNOTE: the 3-sigma robustness gate is REAL. A >3-sigma miss at "
              "some alpha_true is a PHYSICS finding -- do NOT widen the gate; "
              "report the table above.")
    sys.exit(0 if grid_ok else 1)


if __name__ == "__main__":
    if "--diag-21" in sys.argv[1:]:
        diag_alpha21()
    else:
        main()
