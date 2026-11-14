"""Single-catalogue IMF/equipartition recovery with BlackJAX.

The count term is the sampler's categorical likelihood. The conditional
kinematic term retains the Gaussian binned-dispersion approximation; its error
calibration and mass-bin convergence require qualification. Individual exact
representative masses are deliberately not used as continuous IMF observations.
"""

import os
import sys
import time

import jax
import jax.numpy as jnp

import progenax  # noqa: F401  -- enables float64 at import
from progenax.cluster._radial_cdf import potential_interp
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
# Retained demo inference bounds; not a universal stability criterion.
DELTA_BOX = (0.0, 0.7)
W0_BOX = (3.0, 8.0)

# Quadrature resolution for the I_p moment integrals (fixed-length -> grad-safe).
N_QUAD = 256
# Resolution of the precomputed g(W) = I4(W)/I2(W)/3 table (built once per eval).
N_WTAB = 256
_TINY = 1e-30


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
    W_safe = jnp.where(W > 0.0, W, 1.0)  # clamp BEFORE sqrt (deriv +inf at 0)
    u_max = jnp.sqrt(2.0 * W_safe)
    t = jnp.linspace(0.0, 1.0, N_QUAD)
    u = u_max * t
    E = lowered_exponential(g, W_safe - u * u / 2.0)
    I2 = jnp.trapezoid(u**2 * E, u)
    I4 = jnp.trapezoid(u**4 * E, u)
    I2_safe = jnp.where(I2 > 0.0, I2, 1.0)  # clamp BEFORE divide (0/0 -> NaN)
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
def build_model(theta):
    """Rebuild the same equilibrium used by catalogue generation."""
    alpha, delta, W0 = theta
    imf = Maschberger(alpha=alpha, m_min=M_RANGE[0], m_max=M_RANGE[1])
    return MultiComponentCluster.from_imf(
        imf, N_COMP, W0, g=1.0, delta=delta, m_range=M_RANGE
    )


def component_log_probabilities(theta):
    model = build_model(theta)
    return jax.nn.log_softmax(jnp.log(model.N_frac_j + 1e-30))


def count_negloglike(theta, counts):
    """Multinomial NLL up to a parameter-independent combinatorial constant."""
    return -jnp.dot(counts, component_log_probabilities(theta))


def joint_information(data, z):
    """Conditional Gaussian information plus expected multinomial information.

    Uses only first derivatives through the equilibrium solver. This local
    approximation is not a calibrated posterior uncertainty estimate.
    """
    kin = di.fisher_information_gn(make_residual_fn(data), z)
    logp = lambda x: component_log_probabilities(_theta_of_z(x))
    scores = jax.jacrev(logp)(z)
    probabilities = jnp.exp(logp(z))
    return kin + jnp.sum(data["component_counts"]) * (
        scores.T @ (probabilities[:, None] * scores)
    )


def predict_binned(theta, r_edges, component_counts):
    """Predict conditional bin dispersions with M(theta)=sum_j n_j m_j(theta).

    Counts are observed and fixed; representative masses change with the IMF.
    The Gaussian summary likelihood remains an approximation.
    """
    alpha, delta, W0 = theta
    m = build_model(theta)
    s = jnp.sqrt(
        G_MODEL * jnp.dot(component_counts, m.m_j) / (9.0 * m.r_c * m.mu_tot)
    )  # component_counts constant

    # g(W) table built ONCE per eval (W spans 0 .. max(rescale_j) * W0).
    W_max = jnp.max(m.rescale_j) * W0
    W_tab, g_tab = _build_g_table(W_max, m.g)

    # psi on the model's own r-grid: psi(r/r_c), left=W0 (r->0), right=0 (r>r_t).
    psi_grid = potential_interp(
        m._r_grid / m.r_c, m.xi_grid, m.engine_a.psi_raw, left=W0, right=0.0
    )  # (n_grid,)
    r_grid = m._r_grid
    edges_lo = r_edges[:-1]
    edges_hi = r_edges[1:]

    def _per_group(j):
        W_grid_j = m.rescale_j[j] * jnp.maximum(psi_grid, 0.0)  # (n_grid,)
        g_grid_j = jnp.interp(W_grid_j, W_tab, g_tab)  # sigma^2/s_j^2 on grid
        cdf_j = m._cdf_j[j]  # cumulative number 0..1
        # cumulative int_0^r g dN along the grid (trapezoid in dN = d cdf):
        dN = jnp.diff(cdf_j)  # >= 0
        gmid = 0.5 * (g_grid_j[1:] + g_grid_j[:-1])
        C2 = jnp.concatenate([jnp.zeros(1), jnp.cumsum(gmid * dN)])  # (n_grid,)
        # number-weighted <g> per bin via the cumulative ratio at the bin edges:
        C2_lo = jnp.interp(edges_lo, r_grid, C2)
        C2_hi = jnp.interp(edges_hi, r_grid, C2)
        N_lo = jnp.interp(edges_lo, r_grid, cdf_j)
        N_hi = jnp.interp(edges_hi, r_grid, cdf_j)
        gbar_jk = (C2_hi - C2_lo) / ((N_hi - N_lo) + _TINY)  # <g> per bin
        s_j = s * m.w_j[j]
        # double-where: clamp BEFORE sqrt (deriv +inf at 0) so empty/zero-g bins
        # (which carry data weight=0) keep a finite gradient through the model.
        gbar_safe = jnp.where(gbar_jk > 0.0, gbar_jk, 1.0)
        sig = s_j * jnp.sqrt(gbar_safe)
        return jnp.where(gbar_jk > 0.0, sig, 0.0)  # sqrt(E[sigma_hat^2])

    return jax.vmap(_per_group)(jnp.arange(N_COMP))  # (N_COMP, K)


def sigma_oracle(theta, r_eval, component_counts):
    r"""Per-group 1-D dispersion sigma_j(r) at theta on radii r_eval (FOR PLOTTING).

    Evaluates the SAME Engine A oracle predict_binned integrates, but pointwise:
    sigma_j(r) = s * w_j * sqrt(g(W_j(r))), W_j(r) = rescale_j * psi(r). Returns
    shape (N_COMP, len(r_eval)). This is a smooth-curve helper for the fit figure,
    NOT used in the likelihood (the likelihood uses the bin-averaged
    EXPECTATION predict_binned).
    """
    alpha, delta, W0 = theta
    m = build_model(theta)
    s = jnp.sqrt(G_MODEL * jnp.dot(component_counts, m.m_j) / (9.0 * m.r_c * m.mu_tot))
    W_max = jnp.max(m.rescale_j) * W0
    W_tab, g_tab = _build_g_table(W_max, m.g)
    psi_eval = potential_interp(
        r_eval / m.r_c, m.xi_grid, m.engine_a.psi_raw, left=W0, right=0.0
    )

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
def build_truth_data(key=None, alpha_true=ALPHA_TRUE):
    """Draw one cluster and retain its counts and binned kinematic summaries.

    Coordinates use the known generating centre and rest frame. Exact
    representative masses are not treated as continuous IMF measurements.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    k_kin = jax.random.split(key)[0]

    model_true = build_model((alpha_true, DELTA_TRUE, W0_TRUE))
    ic = model_true.sample_cluster(k_kin, n_stars=N_STARS, G=G_MODEL)

    # Use the known generating centre and rest frame.
    pos = ic.positions
    vel = ic.velocities
    cid = ic.component_id  # int in [0, N_COMP)

    r = jnp.linalg.norm(pos, axis=1)

    # Retain the historical 16 quantile bins and occupancy floor. Their
    # finite-sample approximation is checked, not assumed unbiased.
    R_CUT = float(r.max())
    R_EDGES = jnp.quantile(r, jnp.linspace(0.0, 1.0, 17))  # 16 bins, FROZEN

    sig_hat, se, weight, n = di.binned_sigma1d(pos, vel, cid, N_COMP, R_EDGES, n_min=30)

    # Per-(group, bin) mean radius (for placing the figure's data points).
    bin_ids = di._bin_index(r, R_EDGES)
    r_sum, r_cnt = di._grouped_bin_sums(r, cid, bin_ids, N_COMP, R_EDGES.shape[0] - 1)
    r_mean = jnp.where(r_cnt > 0, r_sum / jnp.where(r_cnt > 0, r_cnt, 1.0), 0.0)

    component_counts = jnp.bincount(cid, length=N_COMP)
    M_fixed = float(jnp.sum(ic.masses))  # measured total mass (a CONSTANT)

    return dict(
        r_edges=R_EDGES,
        r_cut=R_CUT,
        sig_hat=sig_hat,
        se=se,
        weight=weight,
        n=n,
        r_mean=r_mean,
        component_counts=component_counts,
        positions=pos,
        component_id=cid,
        representative_masses=model_true.m_j,
        core_radius=model_true.r_c,
        M_fixed=M_fixed,
    )


# --------------------------------------------------------------------------- #
# Joint negative log-likelihood (one jit(value_and_grad)-able function)
# --------------------------------------------------------------------------- #
def _theta_of_z(z):
    """Box-reparam z in R^3 -> theta = (alpha, delta, W0) in their open boxes."""
    return (
        di.expit(z[0], *ALPHA_BOX),
        di.expit(z[1], *DELTA_BOX),
        di.expit(z[2], *W0_BOX),
    )


def _dtheta_dz(z):
    """Per-component derivative dtheta_i/dz_i of the expit boxes at z (shape (3,))."""
    return jnp.array(
        [
            jax.grad(lambda zi: di.expit(zi, *ALPHA_BOX))(z[0]),
            jax.grad(lambda zi: di.expit(zi, *DELTA_BOX))(z[1]),
            jax.grad(lambda zi: di.expit(zi, *W0_BOX))(z[2]),
        ]
    )


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
    component_counts = data["component_counts"]
    safe_se = jnp.where(se > 0, se, 1.0)
    sqrt_w = jnp.sqrt(weight)

    def residual_fn(z):
        sig_model = predict_binned(_theta_of_z(z), r_edges, component_counts)
        return (sqrt_w * (sig_hat - sig_model) / safe_se).ravel()

    return residual_fn


def make_count_negloglike(data):
    """Component-count likelihood from the same catalogue as the kinematics."""
    counts = data["component_counts"]
    return lambda z: count_negloglike(_theta_of_z(z), counts)


def make_negloglike(data):
    """Build the joint negloglike(z) = kinematic + mass channel over the data.

    Equivalent to ``0.5 sum r(z)^2 + count_negloglike_fn(z)`` using the shared
    residual / mass builders, so the loss and the Gauss-Newton Fisher are
    guaranteed consistent.
    """
    residual_fn = make_residual_fn(data)
    count_negloglike_fn = make_count_negloglike(data)

    def negloglike(z):
        r = residual_fn(z)
        return 0.5 * jnp.sum(r * r) + count_negloglike_fn(z)

    return negloglike


# --------------------------------------------------------------------------- #
# Checks + budget gate
# --------------------------------------------------------------------------- #
def self_consistency_check(data):
    """predict_binned(truth) vs binned sig_hat in populated bins (units of se)."""
    sig_model = predict_binned(
        (ALPHA_TRUE, DELTA_TRUE, W0_TRUE), data["r_edges"], data["component_counts"]
    )
    sig_hat, se, weight = data["sig_hat"], data["se"], data["weight"]
    dev = jnp.where(weight > 0, (sig_hat - sig_model) / jnp.where(se > 0, se, 1.0), 0.0)

    print("\nSelf-consistency: predict_binned(truth) vs sig_hat (deviation in SE)")
    print(
        f"{'grp':>3} {'bin':>3} {'sig_hat':>9} {'sig_pred':>9} {'se':>8} {'dev/se':>8}"
    )
    max_dev = 0.0
    for j in range(sig_hat.shape[0]):
        for k in range(sig_hat.shape[1]):
            if float(weight[j, k]) > 0:
                d = float(dev[j, k])
                max_dev = max(max_dev, abs(d))
                print(
                    f"{j:>3} {k:>3} "
                    f"{float(sig_hat[j, k]):>9.4f} {float(sig_model[j, k]):>9.4f} "
                    f"{float(se[j, k]):>8.4f} {d:>8.2f}"
                )
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
    fit = jax.jit(
        lambda start: di.mle_adam(negloglike, start, n_steps=N_ADAM_STEPS, lr=ADAM_LR)
    )
    for i in range(N_INITS):
        z_hat, trace = fit(z0s[i])
        final = float(loss_jit(z_hat))
        z_hats.append(z_hat)
        traces.append(trace)
        finals.append(final)
        print(f"  start {i + 1}/{N_INITS}: final objective {final:.8g}", flush=True)
    i_best = int(jnp.argmin(jnp.array(finals)))
    return z_hats[i_best], traces[i_best], finals, i_best, z0s


def recovery_table(theta_hat, sigma_theta, truths=None):
    """Print param | truth | theta_hat | sigma_hat | (theta_hat-truth)/sigma_hat.

    ``truths`` defaults to the module truth ``(ALPHA_TRUE, DELTA_TRUE, W0_TRUE)``;
    the Task 6 robustness grid passes its per-dataset ``alpha_true``.
    """
    if truths is None:
        truths = (ALPHA_TRUE, DELTA_TRUE, W0_TRUE)
    names = ("alpha", "delta", "W0")
    print(
        f"\n{'param':>6} {'truth':>8} {'theta_hat':>10} {'sigma_hat':>10} "
        f"{'(hat-truth)/sigma':>18}"
    )
    pulls = []
    for nm, tr, th, sg in zip(names, truths, theta_hat, sigma_theta):
        pull = (float(th) - tr) / float(sg)
        pulls.append(pull)
        print(f"{nm:>6} {tr:>8.4f} {float(th):>10.4f} {float(sg):>10.4f} {pull:>18.3f}")
    return pulls


# --------------------------------------------------------------------------- #
# Fit figure (panel a: sigma_j(r) data + fit curves; panel b: component counts)
# --------------------------------------------------------------------------- #
def make_cluster_figure(data, out_dir):
    """Project the actual catalogue and compare empirical component radii."""
    import _plotstyle as ps
    import matplotlib.pyplot as plt
    import numpy as np

    ps.apply_pub_style()
    xy = np.asarray(data["positions"])[:, :2] / float(data["core_radius"])
    component = np.asarray(data["component_id"])
    radius = np.linalg.norm(xy, axis=1)
    colors = [ps.OI[c] for c in ("blue", "green", "orange", "vermilion")]
    styles = ["-", "--", "-.", ":"]
    fig, (cluster, cumulative) = plt.subplots(
        1, 2, figsize=(8.5, 3.6), layout="constrained"
    )
    extent = float(np.abs(xy).max())
    cells = cluster.hexbin(
        xy[:, 0],
        xy[:, 1],
        gridsize=100,
        bins="log",
        mincnt=1,
        cmap="viridis",
        rasterized=True,
    )
    cluster.set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        xlabel=r"$x/r_c$",
        ylabel=r"$y/r_c$",
        aspect="equal",
    )
    fig.colorbar(cells, ax=cluster, label="Stars per hexagon", shrink=0.85)
    edges = np.geomspace(*M_RANGE, N_COMP + 1)
    for j in range(N_COMP):
        r = np.sort(radius[component == j])
        fraction = np.arange(1, r.size + 1) / r.size
        cumulative.step(
            r,
            fraction,
            where="post",
            color=colors[j % len(colors)],
            linestyle=styles[j % len(styles)],
            label=rf"{edges[j]:.3g}–{edges[j + 1]:.3g} $M_\odot$",
        )
    cumulative.set(
        xscale="log",
        ylim=(0, 1.02),
        xlabel=r"Projected radius $R/r_c$",
        ylabel="Fraction of component inside R",
    )
    cumulative.legend(loc="lower right", fontsize=8)
    ps.panel_label(cluster, "(a)")
    ps.panel_label(cumulative, "(b)")
    ps.save_fig(fig, out_dir, "imf_equip_cluster")
    plt.close(fig)


def make_fit_figure(data, theta_hat, sigma_theta, out_dir):
    """Two-panel fit figure: kinematic sigma_j(r) + component counts."""
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
    component_counts = data["component_counts"]
    a_hat, d_hat, w0_hat = (float(x) for x in theta_hat)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.5, 3.6))

    # Panel (a): per-group sigma(r) data (SE bars) + best-fit smooth curves.
    r_grid = jnp.linspace(1e-3, float(data["r_cut"]), 400)
    sig_fit = np.asarray(sigma_oracle((a_hat, d_hat, w0_hat), r_grid, component_counts))
    sig_true = np.asarray(
        sigma_oracle((ALPHA_TRUE, DELTA_TRUE, W0_TRUE), r_grid, component_counts)
    )
    rg = np.asarray(r_grid)
    for j in range(sig_hat.shape[0]):
        mask = weight[j] > 0
        axa.errorbar(
            r_mean[j][mask],
            sig_hat[j][mask],
            yerr=se[j][mask],
            fmt="o",
            ms=3.5,
            color=colors[j % len(colors)],
            capsize=1.5,
            elinewidth=0.8,
            mew=0.0,
            zorder=3,
            label=rf"group $j={j}$",
        )
        axa.plot(rg, sig_fit[j], "-", color=colors[j % len(colors)], lw=1.6, zorder=2)
        axa.plot(
            rg,
            sig_true[j],
            ":",
            color=colors[j % len(colors)],
            lw=0.8,
            alpha=0.7,
            zorder=1,
        )
    axa.set_xlabel(r"$r$ (model units)")
    axa.set_ylabel(r"$\sigma_{1\mathrm{D}, j}(r)$")
    axa.set_xlim(0.0, float(data["r_cut"]))
    axa.legend(loc="upper right", ncol=1)
    ps.panel_label(axa, "(a)")

    # The observed groups have fixed labels, not continuously measured masses.
    groups = np.arange(N_COMP)
    counts = np.asarray(data["component_counts"])
    predicted = counts.sum() * np.exp(
        np.asarray(component_log_probabilities(theta_hat))
    )
    axb.bar(groups, counts, alpha=0.45, color=ps.OI["sky"], label="catalogue counts")
    axb.plot(
        groups, predicted, "o", color=ps.OI["vermilion"], label="fitted expected counts"
    )
    axb.set_yscale("log")
    axb.set_xticks(groups)
    axb.set_xlabel("Mass-component index")
    axb.set_ylabel("Number of stars")
    axb.legend()
    ps.panel_label(axb, "(b)")

    cap = (
        rf"$\hat\alpha={a_hat:.3f}\pm{float(sigma_theta[0]):.3f},\ "
        rf"\hat\delta={d_hat:.3f}\pm{float(sigma_theta[1]):.3f},\ "
        rf"\hat W_0={w0_hat:.3f}\pm{float(sigma_theta[2]):.3f}$"
    )
    fig.suptitle("")  # no in-figure title (paper caption carries it)
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, out_dir, "demo_delta_recovery_fit")
    print(f"\nfit figure -> {out_dir}/demo_delta_recovery_fit.png (+ .pdf)")


# --------------------------------------------------------------------------- #
# Task 5: Fisher information panel (alpha, delta) -- kinematics-only vs joint
# (compare measured local information with and without component counts)
# --------------------------------------------------------------------------- #
def _ad_block_cov(cov_theta3):
    """The (alpha, delta) 2x2 sub-block of a 3x3 constrained covariance.

    Slicing the COVARIANCE (not the Fisher) gives the MARGINAL (alpha, delta)
    distribution -- W0 already integrated out. (Slicing the Fisher would give the
    conditional, which hides the (alpha, delta) correlation.)
    """
    return cov_theta3[jnp.ix_(jnp.array([0, 1]), jnp.array([0, 1]))]


def _ellipse_xy(mean2, cov2, n_sigma=2.0, n_pts=200):
    """(x, y) of the ``n_sigma`` covariance ellipse for a 2-D Gaussian.

    The ellipse is ``mean + n_sigma * L @ unit_circle`` with ``L`` the Cholesky
    factor of ``cov2`` (so radius = n_sigma in Mahalanobis units)."""
    t = jnp.linspace(0.0, 2.0 * jnp.pi, n_pts)
    circle = jnp.stack([jnp.cos(t), jnp.sin(t)], axis=0)  # (2, n_pts)
    L = jnp.linalg.cholesky(cov2)
    pts = mean2[:, None] + n_sigma * (L @ circle)
    return pts[0], pts[1]


def fisher_degeneracy(data, z_hat):
    """Compare local kinematic and joint information, marginalized over W0."""
    residual_fn = make_residual_fn(data)
    dtheta_dz = _dtheta_dz(z_hat)

    F_kin = di.fisher_information_gn(residual_fn, z_hat)  # kinematics-only J^T J
    F_joint = joint_information(data, z_hat)

    cov_kin3 = di.constrained_cov(F_kin, dtheta_dz)
    cov_joint3 = di.constrained_cov(F_joint, dtheta_dz)
    cov_kin = _ad_block_cov(cov_kin3)
    cov_joint = _ad_block_cov(cov_joint3)

    rho_kin = float(cov_kin[0, 1] / jnp.sqrt(cov_kin[0, 0] * cov_kin[1, 1]))
    rho_joint = float(cov_joint[0, 1] / jnp.sqrt(cov_joint[0, 0] * cov_joint[1, 1]))
    # Dchi2=4 ('2sigma' per parameter; 86.5% in 2D) ellipse area =
    # pi * (radius=2)^2 * sqrt(det(cov)). The kin/joint AREA RATIO is
    # independent of the radius convention.
    area_kin = float(jnp.pi * 4.0 * jnp.sqrt(jnp.linalg.det(cov_kin)))
    area_joint = float(jnp.pi * 4.0 * jnp.sqrt(jnp.linalg.det(cov_joint)))
    return dict(
        cov_kin=cov_kin,
        cov_joint=cov_joint,
        rho_kin=rho_kin,
        rho_joint=rho_joint,
        area_kin=area_kin,
        area_joint=area_joint,
    )


def make_fisher_figure(deg, theta_hat, out_dir):
    """Dchi2=4 ('2sigma' per param; 86.5% in 2D) (alpha, delta) ellipses:
    kinematics-only vs joint, MLE + truth."""
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, os.path.dirname(__file__))
    import _plotstyle as ps  # noqa: E402

    ps.apply_pub_style()
    a_hat, d_hat = float(theta_hat[0]), float(theta_hat[1])
    mean2 = jnp.array([a_hat, d_hat])

    fig, ax = plt.subplots(figsize=(4.6, 4.0))

    xk, yk = _ellipse_xy(mean2, deg["cov_kin"], n_sigma=2.0)
    xj, yj = _ellipse_xy(mean2, deg["cov_joint"], n_sigma=2.0)
    ax.plot(
        np.asarray(xk),
        np.asarray(yk),
        "-",
        color=ps.OI["vermilion"],
        lw=1.8,
        label=rf"kinematics only ($\rho={deg['rho_kin']:+.3f}$)",
    )
    ax.fill(np.asarray(xk), np.asarray(yk), color=ps.OI["vermilion"], alpha=0.08)
    ax.plot(
        np.asarray(xj),
        np.asarray(yj),
        "-",
        color=ps.OI["blue"],
        lw=1.8,
        label=rf"joint (+ counts, $\rho={deg['rho_joint']:+.3f}$)",
    )
    ax.fill(np.asarray(xj), np.asarray(yj), color=ps.OI["blue"], alpha=0.12)

    ax.plot(
        a_hat,
        d_hat,
        "o",
        color=ps.OI["black"],
        ms=5,
        zorder=5,
        label=r"MLE $\hat\theta$",
    )
    ax.plot(
        ALPHA_TRUE,
        DELTA_TRUE,
        "*",
        color=ps.OI["orange"],
        ms=13,
        zorder=6,
        mec=ps.OI["black"],
        mew=0.5,
        label="truth",
    )

    ax.set_xlabel(r"$\alpha$ (IMF high-mass slope)")
    ax.set_ylabel(r"$\delta$ (equipartition)")
    ax.legend(loc="best")
    cap = (
        rf"$\Delta\chi^2=4$ ('2$\sigma$' per param; 86.5\% in 2D) marginal "
        rf"ellipses; kinematics-only $\rho(\alpha,\delta)"
        rf"={deg['rho_kin']:+.3f}$, area ratio "
        rf"kin/joint $={deg['area_kin'] / deg['area_joint']:.1f}$"
    )
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, out_dir, "demo_delta_recovery_fisher")
    print(f"\nfisher panel -> {out_dir}/demo_delta_recovery_fisher.png (+ .pdf)")


# --------------------------------------------------------------------------- #
# Task 5: NUTS corner -- budget projection (STOP gate) + sampler + figure
# --------------------------------------------------------------------------- #
N_WARMUP = 300
N_SAMPLES = 600
NUTS_KEY = 11


def run_nuts_corner(data, z_hat, theta_hat, sigma_theta, out_dir):
    """Sample the joint posterior with NUTS and draw the 3-param corner.

    logdensity(z) = -negloglike(z) + sum log(dtheta_i/dz_i) -- the SAME negloglike
    the MLE minimizes PLUS the box-reparam Jacobian, so the target is the
    flat-in-theta likelihood posterior (mode at the MLE) rather than the
    sigmoid-peaked pushforward of a flat-in-z prior. Draws are in unconstrained z,
    transformed back to theta = (alpha, delta, W0) = expit(z).
    Returns (theta_samples (n, 3), n_divergent, post_mean (3,)).
    """
    negloglike = make_negloglike(data)

    def logdensity_fn(z):
        # Flat-in-theta likelihood posterior: pi(z) = L(theta(z)) * |dtheta/dz|
        # -> logpi(z) = -negloglike(z) + sum log(dtheta_i/dz_i). See the probe's
        # logdensity_fn for the derivation; the SAME target is projected + sampled.
        return -negloglike(z) + jnp.sum(jnp.log(_dtheta_dz(z)))

    import json

    import arviz as az
    import numpy as np

    keys = jax.random.split(jax.random.PRNGKey(NUTS_KEY), 4)
    starts = jnp.concatenate((dispersed_inits(), z_hat[None, :]), axis=0)
    chains, divergences = [], []
    for key, start in zip(keys, starts):
        out = di.run_nuts(
            logdensity_fn, start, key, n_warmup=N_WARMUP, n_samples=N_SAMPLES
        )
        chains.append(jax.vmap(lambda z: jnp.stack(_theta_of_z(z)))(out.samples))
        divergences.append(int(out.n_divergent))
    draws = np.asarray(jnp.stack(chains))
    names = ("alpha", "delta", "W0")
    inference = az.from_dict(
        posterior={name: draws[:, :, i] for i, name in enumerate(names)}
    )
    diagnostics = {}
    for method, result in (
        ("rhat", az.rhat(inference, method="rank")),
        ("ess_bulk", az.ess(inference, method="bulk")),
        ("ess_tail", az.ess(inference, method="tail")),
    ):
        diagnostics[method] = {name: float(result[name]) for name in names}
    diagnostics["divergences_per_chain"] = divergences
    diagnostics["scope"] = (
        "Approximate summary posterior; coverage and predictive checks remain open."
    )
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "demo_delta_recovery_chains.npz"), samples=draws)
    with open(
        os.path.join(out_dir, "demo_delta_recovery_diagnostics.json"), "w"
    ) as stream:
        json.dump(diagnostics, stream, indent=2)
    print(json.dumps(diagnostics, indent=2))
    diagnostics_ok = (
        sum(divergences) == 0
        and all(np.isfinite(v) and v < 1.01 for v in diagnostics["rhat"].values())
        and all(np.isfinite(v) and v >= 400 for v in diagnostics["ess_bulk"].values())
        and all(np.isfinite(v) and v >= 400 for v in diagnostics["ess_tail"].values())
    )
    if not diagnostics_ok:
        raise RuntimeError(
            "BlackJAX convergence checks failed; draws and diagnostics retained."
        )
    theta_samp = jnp.asarray(draws.reshape(-1, 3))
    post_mean = jnp.mean(theta_samp, axis=0)
    n_div = sum(divergences)
    make_corner_figure(theta_samp, theta_hat, sigma_theta, out_dir)
    return theta_samp, n_div, post_mean


def make_corner_figure(theta_samp, theta_hat, sigma_theta, out_dir):
    """3-param (alpha, delta, W0) corner of the NUTS posterior; MLE + truth."""
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, os.path.dirname(__file__))
    import _plotstyle as ps  # noqa: E402

    ps.apply_pub_style()
    samp = np.asarray(theta_samp)
    names = (r"$\alpha$", r"$\delta$", r"$W_0$")
    truths = (ALPHA_TRUE, DELTA_TRUE, W0_TRUE)
    hats = [float(x) for x in theta_hat]
    P = 3

    fig, axes = plt.subplots(P, P, figsize=(6.0, 6.0))
    for i in range(P):
        for k in range(P):
            ax = axes[i, k]
            if k > i:
                ax.axis("off")
                continue
            if i == k:
                ax.hist(
                    samp[:, i],
                    bins=40,
                    histtype="stepfilled",
                    color=ps.OI["sky"],
                    alpha=0.5,
                    edgecolor=ps.OI["blue"],
                    lw=0.7,
                    density=True,
                )
                ax.axvline(hats[i], color=ps.OI["black"], lw=1.2, ls="-")
                ax.axvline(truths[i], color=ps.OI["orange"], lw=1.2, ls=":")
                ax.set_yticks([])
            else:
                ax.scatter(
                    samp[:, k],
                    samp[:, i],
                    s=2,
                    color=ps.OI["blue"],
                    alpha=0.12,
                    ec="none",
                    rasterized=True,
                )
                ax.plot(hats[k], hats[i], "o", color=ps.OI["black"], ms=4, zorder=5)
                ax.plot(
                    truths[k],
                    truths[i],
                    "*",
                    color=ps.OI["orange"],
                    ms=11,
                    zorder=6,
                    mec=ps.OI["black"],
                    mew=0.4,
                )
            if i == P - 1:
                ax.set_xlabel(names[k])
            else:
                ax.set_xticklabels([])
            if k == 0 and i > 0:
                ax.set_ylabel(names[i])
            elif k > 0:
                ax.set_yticklabels([])

    fig.tight_layout()
    ps.save_fig(fig, out_dir, "demo_delta_recovery_corner")
    print(f"corner -> {out_dir}/demo_delta_recovery_corner.png (+ .pdf)")


def main():
    import argparse

    global N_COMP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--illustrate-only",
        action="store_true",
        help="Draw the cluster illustration without fitting",
    )
    parser.add_argument("--seed", type=int, default=0, help="Synthetic catalogue seed")
    parser.add_argument(
        "--n-comp", type=int, default=N_COMP, help="IMF component resolution"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run catalogue and derivative checks without fitting",
    )
    parser.add_argument(
        "--run-nuts", action="store_true", help="Run four BlackJAX chains after fitting"
    )
    args = parser.parse_args()
    if args.n_comp < 2:
        parser.error("--n-comp must be at least two for this multimass experiment")
    N_COMP = args.n_comp
    print("=" * 72)
    print("B2 single-catalogue recovery: MLE + local information + BlackJAX")
    print("Scope: discrete component counts plus approximate conditional kinematics.")
    print("=" * 72)
    print(
        f"truth: alpha={ALPHA_TRUE}, delta={DELTA_TRUE}, W0={W0_TRUE}; "
        f"N_COMP={N_COMP}, N_STARS={N_STARS}, M_RANGE={M_RANGE}, G={G_MODEL}"
    )

    data = build_truth_data(jax.random.PRNGKey(args.seed))
    if args.illustrate_only:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "validation", "plots"
        )
        os.makedirs(out_dir, exist_ok=True)
        make_cluster_figure(data, out_dir)
        print("Catalogue counts:", data["component_counts"].tolist())
        print("Representative masses:", data["representative_masses"].tolist())
        print("Cluster illustration saved to", out_dir)
        return
    n = data["n"]
    J = n.shape[0]
    print(f"\nCatalogue total mass (diagnostic only) = {data['M_fixed']:.3f}")
    print(f"R_CUT (cluster edge = max radius) = {data['r_cut']:.4f}")
    print(
        "R_EDGES (16 quantile bins on [0, R_CUT]) =",
        [round(float(x), 4) for x in data["r_edges"]],
    )
    print("per-group counts (group x bin):")
    for j in range(J):
        print(f"  group {j}: {[int(x) for x in n[j]]}  (total {int(n[j].sum())})")
    top_occ = int(n[J - 1].sum())
    print(
        f"top-group (j={J - 1}) occupancy = {top_occ}"
        + ("  [WARN < 300 -- Task 4 has a >=300 gate]" if top_occ < 300 else "")
    )

    if args.check_only:
        objective = jax.jit(make_negloglike(data))
        z = jnp.array(
            [
                di.logit(ALPHA_TRUE, *ALPHA_BOX),
                di.logit(DELTA_TRUE, *DELTA_BOX),
                di.logit(W0_TRUE, *W0_BOX),
            ]
        )
        ad = jax.jit(jax.grad(objective))(z)
        print("float64:", jax.config.jax_enable_x64)
        print("counts:", data["component_counts"].tolist())
        print("AD gradient:", ad.tolist())
        checks = []
        for h in (1e-4, 1e-5):
            fd = jnp.array(
                [
                    (
                        objective(z + h * jnp.eye(3)[i])
                        - objective(z - h * jnp.eye(3)[i])
                    )
                    / (2 * h)
                    for i in range(3)
                ]
            )
            error = jnp.abs(ad - fd) / jnp.maximum(
                jnp.maximum(jnp.abs(ad), jnp.abs(fd)), 1e-12
            )
            print(
                "FD step", h, "gradient", fd.tolist(), "relative error", error.tolist()
            )
            checks.append(bool(jnp.all(error <= 0.003)))
        max_dev = self_consistency_check(data)
        checks.append(max_dev < 4.0)
        checks.append(bool(jnp.sum(data["component_counts"]) == N_STARS))
        print("CHECK-ONLY", "PASS" if all(checks) else "FAIL")
        raise SystemExit(0 if all(checks) else 1)

    # Self-consistency BEFORE the budget gate.
    max_dev = self_consistency_check(data)
    sc_ok = max_dev < 4.0  # populated-bin deviations must be a few x se
    print(
        f"self-consistency {'OK' if sc_ok else 'FAIL'} "
        f"(max |dev/se| = {max_dev:.2f}, threshold 4.0)"
    )
    if not sc_ok:
        print(
            "\nSTOP: predict_binned(truth) does not match sig_hat -- "
            "miscalibrated oracle scale or group alignment. Not proceeding."
        )
        sys.exit(1)

    # Budget gate.
    negloglike = make_negloglike(data)
    loss_and_grad = jax.jit(jax.value_and_grad(negloglike))
    z0 = jnp.zeros(3)

    t0 = time.perf_counter()
    v, g = loss_and_grad(z0)
    v.block_until_ready()
    g.block_until_ready()
    t_compile = time.perf_counter() - t0

    z1 = jnp.array([0.1, -0.2, 0.3])
    t0 = time.perf_counter()
    v2, g2 = loss_and_grad(z1)
    v2.block_until_ready()
    g2.block_until_ready()
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
    print(
        f"\nBUDGET {'PASS' if budget_ok else 'STOP'} "
        f"(warm {t_warm:.3f}s {'<=' if budget_ok else '>'} 5.0s)"
    )
    if not budget_ok:
        print(
            "STOP: warm loss+grad exceeds 5 s. NOT degrading the solve. "
            "Report measured numbers + plan options to the orchestrator."
        )
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
        print(
            f"  init {i} (z0={[round(float(x), 2) for x in z0s[i]]}): "
            f"final negloglike = {finals[i]:.6e}{tag}"
        )
    plat_ok, total_dec, tail_dec = plateau_ok(best_trace)
    print(
        f"\nplateau check (winning trace): total decrease = {total_dec:.4e}, "
        f"last-10% decrease = {tail_dec:.4e}"
    )
    print(
        f"plateau {'PASS' if plat_ok else 'FAIL'} "
        f"(tail {tail_dec:.3e} {'<' if plat_ok else '>='} "
        f"1% of total {0.01 * total_dec:.3e})"
    )

    # Interior-optimum sanity: gradient norm small at z_hat.
    _, g_hat = loss_and_grad(z_hat)
    grad_norm = float(jnp.linalg.norm(g_hat))
    print(f"grad norm at z_hat = {grad_norm:.4e} (interior-optimum check)")

    # --------------------------------------------------------------------- #
    # theta_hat + Gauss-Newton Fisher errors (reverse-mode only).
    # --------------------------------------------------------------------- #
    theta_hat = _theta_of_z(z_hat)
    F_z = joint_information(data, z_hat)
    eig = jnp.linalg.eigvalsh(F_z)
    F_pd = bool(jnp.all(eig > 0))
    cond = float(eig[-1] / eig[0]) if F_pd else float("inf")
    print(
        f"\nGauss-Newton Fisher F_z PD: {F_pd}; eigenvalues = "
        f"{[float(x) for x in eig]}; cond = {cond:.3e}"
    )
    if not F_pd:
        print(
            "STOP: F_z is not positive definite -- degenerate / saddle fit. "
            "Reporting rather than masking."
        )
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
        print(
            f"  3-sigma {nm}: |pull| = {abs(p):.3f} "
            f"{'<' if abs(p) < 3.0 else '>='} 3  "
            f"({'PASS' if abs(p) < 3.0 else 'FAIL'})"
        )
    top_occ = int(n[J - 1].sum())
    occ_ok = top_occ >= 300
    print(
        f"  occupancy: top-group (j={J - 1}) total = {top_occ} "
        f"{'>=' if occ_ok else '<'} 300  ({'PASS' if occ_ok else 'FAIL'})"
    )
    print(f"  plateau: {'PASS' if plat_ok else 'FAIL'}")
    print(f"  recovery (3-sigma, all params): {'PASS' if recovery_ok else 'FAIL'}")

    # --------------------------------------------------------------------- #
    # Fit figure (regenerated; promoted in Task 8 -- gitignored).
    # --------------------------------------------------------------------- #
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "validation", "plots"
    )
    os.makedirs(out_dir, exist_ok=True)
    make_fit_figure(data, theta_hat, sigma_theta, out_dir)

    # --------------------------------------------------------------------- #
    # Task 5a: Fisher information panel (alpha, delta) -- kinematics-only vs joint.
    # --------------------------------------------------------------------- #
    print("\n" + "=" * 72)
    print("FISHER INFORMATION (alpha, delta marginal): kinematics-only vs joint")
    print("=" * 72)
    deg = fisher_degeneracy(data, z_hat)
    print(
        f"  kinematics-only rho(alpha, delta) = {deg['rho_kin']:+.4f}  "
        "(measured for this catalogue)"
    )
    print(f"  joint          rho(alpha, delta) = {deg['rho_joint']:+.4f}")
    sa_k = float(jnp.sqrt(deg["cov_kin"][0, 0]))
    sd_k = float(jnp.sqrt(deg["cov_kin"][1, 1]))
    sa_j = float(jnp.sqrt(deg["cov_joint"][0, 0]))
    sd_j = float(jnp.sqrt(deg["cov_joint"][1, 1]))
    print(
        f"  kinematics-only widths: sigma_alpha = {sa_k:.4f}, sigma_delta = {sd_k:.4f}"
    )
    print(
        f"  joint          widths: sigma_alpha = {sa_j:.4f}, sigma_delta = {sd_j:.4f}"
    )
    print(
        f"  Dchi2=4 ('2sigma'/param; 86.5% in 2D) ellipse area  "
        f"kin = {deg['area_kin']:.4e}, "
        f"joint = {deg['area_joint']:.4e}, ratio kin/joint = "
        f"{deg['area_kin'] / deg['area_joint']:.2f}"
    )
    make_fisher_figure(deg, theta_hat, out_dir)

    run_nuts_here = args.run_nuts
    nuts_done = False
    if run_nuts_here:
        run_nuts_corner(data, z_hat, theta_hat, sigma_theta, out_dir)
        nuts_done = True  # run_nuts_corner raises after saving failed diagnostics
    else:
        print("BlackJAX posterior not run; use --run-nuts for four chains.")

    all_ok = (
        sc_ok
        and grad_finite
        and grad_nonzero
        and budget_ok
        and plat_ok
        and occ_ok
        and recovery_ok
        and F_pd
    )
    # The Fisher panel always contributes; the NUTS corner is budget-gated, so a
    # deferred/stopped NUTS does NOT fail the demo (it is reported, not gated away).
    if run_nuts_here and not nuts_done:
        all_ok = False  # if we explicitly ran NUTS, its gates are real
    print("\n" + "=" * 72)
    print(f"RECORDED DEMO CHECKS {'PASS' if all_ok else 'FAIL'}")
    print("These checks do not establish posterior calibration or release readiness.")
    print("=" * 72)
    if not all_ok and recovery_ok is False:
        print(
            "\nNOTE: the 3-sigma recovery gate is REAL. A >3-sigma miss is a "
            "failed recovery check; inspect numerical and statistical causes. Do not widen the gate."
        )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
