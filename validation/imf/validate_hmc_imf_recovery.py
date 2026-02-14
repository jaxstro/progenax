"""
IMF HMC Recovery: Bayesian inference of α from environment-dependent IMFs.

Compares two IMF families:
  - Piecewise 4-segment (Kroupa/Marks+2012): α₃ controls m > 1 M☉ only
  - Maschberger (2013) smooth: α controls full mass range

Tests 4 physically motivated cluster environments spanning the Marks+2012 /
Jerabkova+2018 parameter space, from canonical Kroupa to extreme top-heavy.

Environments (via BirthEnvironment → env_to_imf_params):
  1. Solar neighborhood    (10³ M☉, [Fe/H]=0.0)    α₃ ≈ 2.30
  2. Young Massive Cluster (10⁵ M☉, [Fe/H]=-0.5)   α₃ ≈ 2.07
  3. Low-Z                 (10⁶ M☉, [Fe/H]=-1.5)   α₃ ≈ 1.77
  4. NGC 7078 / M15        (10⁶·⁵ M☉, [Fe/H]=-2.16) α₃ ≈ 1.60

SUCCESS CRITERIA:
  - All 4 α values recovered within 95% credible interval
  - ESS > 200 for every run
  - R-hat < 1.05 for every run

Output:
  validation/plots/hmc_recovery/piecewise_*.{png,pdf}
  validation/plots/hmc_recovery/maschberger_*.{png,pdf}

Run:
  cd /Users/anna/projects/jaxstro-dev/progenax
  python validation/imf/validate_hmc_imf_recovery.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Set host device count BEFORE any JAX computation for multi-chain HMC.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size, gelman_rubin
from numpyro.infer import MCMC, NUTS
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.stats import gaussian_kde

from progenax.imf import (
    IMFParams,
    BirthEnvironment,
    Maschberger,
    env_to_imf_params,
    log_prob_masses,
    sample_masses_from_params,
)

from jaxstroviz import savefig, to_numpy, set_paper, newfig, PALETTE

# ---------------------------------------------------------------------------
# Environment definitions
# ---------------------------------------------------------------------------
ENV_CONFIGS: list[dict] = [
    {
        "name": "Solar",
        "short": "Solar",
        "env": BirthEnvironment.solar(),
    },
    {
        "name": "YMC",
        "short": "YMC",
        "env": BirthEnvironment.from_cluster_mass(M_ecl=1e5, FeH=-0.5),
    },
    {
        "name": "Low-Z",
        "short": "Low-Z",
        "env": BirthEnvironment.massive_gc(FeH=-1.5),
    },
    {
        "name": "Starburst",
        "short": "Starburst",
        "env": BirthEnvironment.ngc_7078(),
    },
]

# Derive α₃ from each environment via Jerabkova+2018 generalized model
for cfg in ENV_CONFIGS:
    params = env_to_imf_params(cfg["env"], model="jerabkova_generalized")
    cfg["alpha3"] = float(params.alpha3)

# Reference-figure palette (Set2-inspired)
ENV_COLORS = [
    "#66c2a5",  # teal-green  — Solar
    "#fc8d62",  # coral       — YMC
    "#e78ac3",  # pink        — Low-Z
    "#7570b3",  # violet      — Starburst
]

# ---------------------------------------------------------------------------
# HMC/NUTS configuration
# ---------------------------------------------------------------------------
N_MASSES = 10_000
N_WARMUP = 500
N_SAMPLES = 1000
N_CHAINS = 2
TARGET_ACCEPT = 0.8
BASE_SEED = 100

PLOT_DIR = Path(__file__).parent / "plots"

LINE_STYLES = ["-", "--", "-.", ":"]


@dataclass
class RecoveryResult:
    """Results from a single NUTS recovery run."""

    env_name: str
    env_short: str
    family: str            # "piecewise" or "maschberger"
    alpha_true: float
    samples: jnp.ndarray   # (n_total_samples,) flattened posterior
    chains: jnp.ndarray    # (n_chains, n_samples) per-chain
    median: float
    mean: float
    std: float
    q16: float
    q84: float
    q025: float
    q975: float
    ess: float
    rhat: float
    in_68ci: bool
    in_95ci: bool
    color: str


# ═══════════════════════════════════════════════════════════════════════════
# Piecewise IMF (4-segment Kroupa): α₃ controls m > 1 M☉
# ═══════════════════════════════════════════════════════════════════════════

def generate_mock_piecewise(alpha3: float, n: int, seed: int) -> jnp.ndarray:
    """Sample n masses from 4-segment piecewise IMF with given α₃."""
    params = IMFParams(
        alpha0=jnp.array(0.3),
        alpha1=jnp.array(1.3),
        alpha2=jnp.array(2.3),
        alpha3=jnp.array(alpha3),
    )
    key = jax.random.PRNGKey(seed)
    u = jax.random.uniform(key, (n,))
    return sample_masses_from_params(params, u)


def numpyro_model_piecewise(observed_masses: jnp.ndarray) -> None:
    """NumPyro model: Uniform prior on α₃, fixed α₀/α₁/α₂."""
    alpha3 = numpyro.sample("alpha", dist.Uniform(0.5, 4.0))

    params = IMFParams(
        alpha0=jnp.array(0.3),
        alpha1=jnp.array(1.3),
        alpha2=jnp.array(2.3),
        alpha3=alpha3,
    )

    log_probs = log_prob_masses(observed_masses, params)

    # NaN guard: NUTS leapfrog can step through α≈1 where 1/(1-α) diverges.
    safe_ll = jnp.sum(jnp.where(jnp.isfinite(log_probs), log_probs, -1e10))
    numpyro.factor("likelihood", safe_ll)


# ═══════════════════════════════════════════════════════════════════════════
# Maschberger (2013) smooth IMF: α controls full mass range
# ═══════════════════════════════════════════════════════════════════════════

def generate_mock_maschberger(alpha: float, n: int, seed: int) -> jnp.ndarray:
    """Sample n masses from Maschberger IMF with given α."""
    imf = Maschberger(alpha=alpha, m_min=0.01, m_max=150.0)
    key = jax.random.PRNGKey(seed)
    return imf.sample(key, n)


def numpyro_model_maschberger(observed_masses: jnp.ndarray) -> None:
    """NumPyro model: Uniform prior on α (Maschberger high-mass slope)."""
    alpha = numpyro.sample("alpha", dist.Uniform(0.5, 4.0))

    imf = Maschberger(alpha=alpha, m_min=0.01, m_max=150.0)
    log_probs = imf.logpdf(observed_masses)

    safe_ll = jnp.sum(jnp.where(jnp.isfinite(log_probs), log_probs, -1e10))
    numpyro.factor("likelihood", safe_ll)


# ═══════════════════════════════════════════════════════════════════════════
# HMC/NUTS runner (family-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

FAMILY_CONFIG = {
    "piecewise": {
        "generate": generate_mock_piecewise,
        "model": numpyro_model_piecewise,
        "label": "Piecewise (4-segment)",
        "title": "Piecewise — Kroupa/Marks+2012",
        "param_label": r"\alpha_3",
    },
    "maschberger": {
        "generate": generate_mock_maschberger,
        "model": numpyro_model_maschberger,
        "label": "Maschberger (2013)",
        "title": "Smooth — Maschberger 2013",
        "param_label": r"\alpha",
    },
}


def run_single_recovery(
    cfg: dict,
    color: str,
    family: str = "piecewise",
    n_masses: int = N_MASSES,
    seed: int = BASE_SEED,
) -> RecoveryResult:
    """Run NUTS for one environment + IMF family."""
    fc = FAMILY_CONFIG[family]
    alpha_true = cfg["alpha3"]
    name = cfg["name"]

    feh = float(cfg["env"].metallicity)
    log_mecl = float(cfg["env"].log_mecl)

    print(f"\n{'─'*55}")
    print(f"  {name}  ({fc['label']})")
    print(f"  [Fe/H]={feh:.2f}, log M_ecl={log_mecl:.1f}")
    print(f"  True α = {alpha_true:.3f}  (Jerabkova+2018)")
    print(f"{'─'*55}")

    masses = fc["generate"](alpha_true, n_masses, seed)
    print(f"  Masses: N={n_masses}, range=[{float(masses.min()):.3f}, "
          f"{float(masses.max()):.1f}] M☉")

    kernel = NUTS(fc["model"], target_accept_prob=TARGET_ACCEPT)
    mcmc = MCMC(
        kernel,
        num_warmup=N_WARMUP,
        num_samples=N_SAMPLES,
        num_chains=N_CHAINS,
        chain_method="sequential",
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(seed + 1000), masses)

    samples_by_chain = mcmc.get_samples(group_by_chain=True)["alpha"]
    chains = jnp.array(samples_by_chain)
    flat = chains.flatten()

    ess = float(effective_sample_size(chains))
    rhat = float(gelman_rubin(chains))

    median = float(jnp.median(flat))
    mean = float(jnp.mean(flat))
    std = float(jnp.std(flat))
    q16, q84 = float(jnp.percentile(flat, 16)), float(jnp.percentile(flat, 84))
    q025, q975 = float(jnp.percentile(flat, 2.5)), float(jnp.percentile(flat, 97.5))

    in_68ci = q16 <= alpha_true <= q84
    in_95ci = q025 <= alpha_true <= q975

    result = RecoveryResult(
        env_name=name,
        env_short=cfg["short"],
        family=family,
        alpha_true=alpha_true,
        samples=flat,
        chains=chains,
        median=median, mean=mean, std=std,
        q16=q16, q84=q84, q025=q025, q975=q975,
        ess=ess, rhat=rhat,
        in_68ci=in_68ci, in_95ci=in_95ci,
        color=color,
    )

    print(f"  Recovered: {median:.3f}  68% CI: [{q16:.3f}, {q84:.3f}]")
    print(f"  Bias: {median - alpha_true:+.4f}")
    print(f"  ESS: {ess:.0f}  R-hat: {rhat:.4f}  In 95% CI: {in_95ci}")

    return result


def run_all_recoveries(family: str) -> list[RecoveryResult]:
    """Run NUTS recovery for all 4 environments with one IMF family."""
    print(f"\n{'='*60}")
    print(f"  Running: {FAMILY_CONFIG[family]['label']}")
    print(f"{'='*60}")

    results = []
    for i, (cfg, color) in enumerate(zip(ENV_CONFIGS, ENV_COLORS)):
        result = run_single_recovery(cfg, color, family=family, seed=BASE_SEED + i)
        results.append(result)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plotting: Panel (a) — IMF shapes m·ξ(m)
# ═══════════════════════════════════════════════════════════════════════════

def plot_imf_shapes(
    ax: plt.Axes, results: list[RecoveryResult], family: str,
) -> None:
    """Panel (a): Mass-weighted IMF m·ξ(m) for each environment."""
    m_grid = np.logspace(-2, np.log10(150.0), 600)
    m_jax = jnp.array(m_grid)
    plabel = FAMILY_CONFIG[family]["param_label"]

    for i, r in enumerate(results):
        if family == "piecewise":
            params = IMFParams(
                alpha0=jnp.array(0.3),
                alpha1=jnp.array(1.3),
                alpha2=jnp.array(2.3),
                alpha3=jnp.array(r.alpha_true),
            )
            log_p = to_numpy(log_prob_masses(m_jax, params))
        else:
            imf = Maschberger(alpha=r.alpha_true, m_min=0.01, m_max=150.0)
            log_p = to_numpy(imf.logpdf(m_jax))

        m_xi = m_grid * np.exp(log_p)

        ax.plot(
            m_grid, m_xi,
            color=r.color, lw=1.6, ls=LINE_STYLES[i % len(LINE_STYLES)],
            label=rf"{r.env_short} (${plabel}={r.alpha_true:.2f}$)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 2e2)
    ax.set_xlabel(r"Stellar Mass $[M_\odot]$")
    ax.set_ylabel(r"$m \cdot \xi(m)$")

    short_title = "Piecewise IMF" if family == "piecewise" else "Smooth IMF"
    ax.set_title(f"(a) {short_title}")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=6,
              edgecolor=PALETTE["light"])
    ax.tick_params(which="both", direction="out")
    ax.tick_params(which="minor", length=2)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting: Panel (b) — Recovery scatter
# ═══════════════════════════════════════════════════════════════════════════

def plot_recovery(ax: plt.Axes, results: list[RecoveryResult]) -> None:
    """Panel (b): True vs recovered α with error bars."""
    plabel = FAMILY_CONFIG[results[0].family]["param_label"]
    alphas = [r.alpha_true for r in results]
    span = [min(alphas) - 0.15, max(alphas) + 0.15]

    ax.plot(span, span, ls="-", color=PALETTE["light"], lw=1.2, zorder=1)

    for r in results:
        c = r.color

        # 95% CI — thin whiskers
        ax.errorbar(
            r.alpha_true, r.median,
            yerr=[[r.median - r.q025], [r.q975 - r.median]],
            fmt="none", ecolor=c, elinewidth=0.8,
            capsize=3.0, capthick=0.8, alpha=0.45, zorder=2,
        )
        # 68% CI — thick bars
        ax.errorbar(
            r.alpha_true, r.median,
            yerr=[[r.median - r.q16], [r.q84 - r.median]],
            fmt="none", ecolor=c, elinewidth=2.5,
            capsize=0, zorder=3,
        )
        # Marker
        ax.plot(
            r.alpha_true, r.median, "o",
            color=c, markeredgecolor="white", markeredgewidth=1.0,
            markersize=6, zorder=4,
        )
        # Environment label
        ax.annotate(
            r.env_short,
            (r.alpha_true, r.median),
            textcoords="offset points",
            xytext=(7, -2),
            fontsize=6,
            color=c,
            fontweight="bold",
        )

    ax.set_xlabel(rf"True ${plabel}$")
    ax.set_ylabel(rf"Recovered ${plabel}$")
    ax.set_title(r"(b) Parameter recovery")
    ax.set_xlim(span)
    ax.set_ylim(span)

    ax.text(
        0.05, 0.95,
        f"$N = {N_MASSES:,}$",
        transform=ax.transAxes, fontsize=6, va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec=PALETTE["light"], alpha=0.9),
    )
    ax.minorticks_on()
    ax.tick_params(which="both", direction="out")
    ax.tick_params(which="minor", length=2)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))


# ═══════════════════════════════════════════════════════════════════════════
# Plotting: Panel (c) — Residual posteriors
# ═══════════════════════════════════════════════════════════════════════════

def plot_residuals(ax: plt.Axes, results: list[RecoveryResult]) -> None:
    """Panel (c): Residual posteriors Δα = recovered - true."""
    plabel = FAMILY_CONFIG[results[0].family]["param_label"]

    # Fixed x-range so both families share comparable axes
    x_lo, x_hi = -0.10, 0.04
    x_grid = np.linspace(x_lo, x_hi, 400)

    for r in results:
        delta = to_numpy(r.samples) - r.alpha_true
        kde = gaussian_kde(delta)
        density = kde(x_grid)

        ax.fill_between(x_grid, density, alpha=0.20, color=r.color)
        ax.plot(x_grid, density, color=r.color, lw=1.3,
                label=rf"{r.env_short}")

    ax.axvline(0, color=PALETTE["dark"], ls="--", lw=0.7, alpha=0.5, zorder=0)

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(rf"$\Delta {plabel}$  (recovered $-$ true)")
    ax.set_ylabel("Density")
    ax.set_title(r"(c) Residual posteriors")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=6,
              edgecolor=PALETTE["light"])
    ax.minorticks_on()
    ax.tick_params(which="both", direction="out")
    ax.tick_params(which="minor", length=2)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="x", labelsize=6, rotation=45)


# ═══════════════════════════════════════════════════════════════════════════
# Figure assembly
# ═══════════════════════════════════════════════════════════════════════════

def create_main_figure(results: list[RecoveryResult], family: str) -> None:
    """Create the 1×3 proposal figure for one IMF family."""
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.5))

    plot_imf_shapes(axes[0], results, family)
    plot_recovery(axes[1], results)
    plot_residuals(axes[2], results)

    fig.tight_layout(w_pad=2.0)

    stem = f"{family}_recovery"
    savefig(fig, PLOT_DIR / f"{stem}.png")
    savefig(fig, PLOT_DIR / f"{stem}.pdf")
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR / stem}.{{png,pdf}}")


# ═══════════════════════════════════════════════════════════════════════════
# Pass/fail evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_pass_fail(results: list[RecoveryResult], family: str) -> bool:
    """Print summary and return True if all criteria pass."""
    label = FAMILY_CONFIG[family]["label"]
    n = len(results)
    print(f"\n{'='*60}")
    print(f"  PASS / FAIL: {label}")
    print(f"{'='*60}")

    all_pass = True

    for r in results:
        bias = r.median - r.alpha_true
        status = "PASS" if r.in_95ci else "FAIL"
        if not r.in_95ci:
            all_pass = False
        print(
            f"  {r.env_short:>10s} (α={r.alpha_true:.3f}): "
            f"median={r.median:.3f}  bias={bias:+.4f}  "
            f"95%CI=[{r.q025:.3f},{r.q975:.3f}]  "
            f"ESS={r.ess:.0f}  R̂={r.rhat:.4f}  [{status}]"
        )

    ess_ok = all(r.ess > 200 for r in results)
    rhat_ok = all(r.rhat < 1.05 for r in results)

    if not ess_ok:
        all_pass = False
    if not rhat_ok:
        all_pass = False

    print(f"\n  95% CI coverage: {sum(r.in_95ci for r in results)}/{n}")
    print(f"  68% CI coverage: {sum(r.in_68ci for r in results)}/{n}")
    print(f"  ESS > 200:       {'PASS' if ess_ok else 'FAIL'}")
    print(f"  R̂ < 1.05:        {'PASS' if rhat_ok else 'FAIL'}")

    max_bias = max(abs(r.median - r.alpha_true) for r in results)
    print(f"  Max |bias|:      {max_bias:.4f}")

    print(f"  {'─'*40}")
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"  {'─'*40}\n")
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    n_envs = len(ENV_CONFIGS)
    print("=" * 60)
    print("  IMF HMC Recovery: Piecewise vs Maschberger")
    print(f"  {n_envs} environments × 2 IMF families")
    print("=" * 60)

    print(f"\n  {'Environment':<20s} {'[Fe/H]':>8s} {'log M_ecl':>10s} {'α₃':>8s}")
    print(f"  {'─'*50}")
    for cfg in ENV_CONFIGS:
        feh = float(cfg["env"].metallicity)
        lm = float(cfg["env"].log_mecl)
        print(f"  {cfg['name']:<20s} {feh:>8.2f} {lm:>10.2f} {cfg['alpha3']:>8.3f}")

    # Run both families
    pw_results = run_all_recoveries("piecewise")
    ma_results = run_all_recoveries("maschberger")

    # Figures
    print("\nCreating figures...")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    set_paper(light=True)
    create_main_figure(pw_results, "piecewise")
    create_main_figure(ma_results, "maschberger")

    # Evaluate
    pw_pass = evaluate_pass_fail(pw_results, "piecewise")
    ma_pass = evaluate_pass_fail(ma_results, "maschberger")

    # Comparison summary
    print("=" * 60)
    print("  BIAS COMPARISON")
    print("=" * 60)
    print(f"  {'Env':>10s}  {'Piecewise':>12s}  {'Maschberger':>12s}")
    print(f"  {'─'*40}")
    for pw, ma in zip(pw_results, ma_results):
        pw_bias = pw.median - pw.alpha_true
        ma_bias = ma.median - ma.alpha_true
        print(f"  {pw.env_short:>10s}  {pw_bias:>+12.4f}  {ma_bias:>+12.4f}")
    print(f"  {'─'*40}")

    if not (pw_pass and ma_pass):
        raise AssertionError("IMF HMC recovery FAILED — see summary above")


if __name__ == "__main__":
    main()
