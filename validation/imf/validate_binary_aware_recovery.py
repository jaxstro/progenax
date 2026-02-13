"""
Binary-aware IMF recovery: naive vs Moe+17-aware NUTS inference.

Generates system masses (singles + unresolved binaries) from Maschberger IMF
with Moe & Di Stefano (2017) binary population, then compares:
  - Naive: fit single-star IMF to system masses (biased)
  - Binary-aware: mixture likelihood marginalizing over binary status (unbiased)

Environments (via BirthEnvironment -> env_to_imf_params):
  1. Solar neighborhood    (10^3 M_sun, [Fe/H]=0.0)    alpha ~ 2.30
  2. Young Massive Cluster (10^5 M_sun, [Fe/H]=-0.5)   alpha ~ 2.07
  3. Low-Z                 (10^6 M_sun, [Fe/H]=-1.5)   alpha ~ 1.77
  4. NGC 7078 / M15        (10^6.5 M_sun, [Fe/H]=-2.16) alpha ~ 1.60

SUCCESS CRITERIA:
  - All 4 binary-aware alpha values recovered within 95% credible interval
  - ESS > 200 for binary-aware runs
  - R-hat < 1.05 for binary-aware runs
  - Naive runs show systematic negative bias (expected)

Output:
  validation/imf/plots/binary_aware_recovery.{png,pdf}

Run:
  cd /Users/anna/projects/jaxstro-dev/progenax
  python validation/imf/validate_binary_aware_recovery.py
"""

from __future__ import annotations

import json
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
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from numpyro.diagnostics import effective_sample_size, gelman_rubin
from numpyro.infer import MCMC, NUTS
from scipy.special import roots_legendre
from scipy.stats import gaussian_kde

from progenax.imf import (
    BinaryIMF,
    BirthEnvironment,
    Maschberger,
    MassDependentBinaryFraction,
    MoeDiStefano2017,
    env_to_imf_params,
)

from jaxstroviz import PALETTE, savefig, set_paper, to_numpy

# ---------------------------------------------------------------------------
# Environment definitions (same as IMF-only script)
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

for cfg in ENV_CONFIGS:
    params = env_to_imf_params(cfg["env"], model="jerabkova_generalized")
    cfg["alpha3"] = float(params.alpha3)

# Reference-figure palette (Set2-inspired, same as IMF-only)
ENV_COLORS = [
    "#66c2a5",  # teal-green  -- Solar
    "#fc8d62",  # coral       -- YMC
    "#e78ac3",  # pink        -- Low-Z
    "#7570b3",  # violet      -- Starburst
]

# ---------------------------------------------------------------------------
# HMC/NUTS configuration
# ---------------------------------------------------------------------------
N_MASSES = 10_000
N_WARMUP = 500
N_SAMPLES = 1000
N_CHAINS = 2
TARGET_ACCEPT = 0.8
BASE_SEED = 200

PLOT_DIR = Path(__file__).parent / "plots"
Q_MIN = 0.1
M_MIN = 0.01
M_MAX = 150.0

LINE_STYLES = ["-", "--", "-.", ":"]


RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class RecoveryResult:
    """Results from a single HMC/NUTS recovery run."""

    env_name: str
    env_short: str
    method: str            # "naive" or "binary_aware"
    alpha_true: float
    samples: jnp.ndarray
    chains: jnp.ndarray
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

    def to_dict(self) -> dict:
        """Serialize to dict (samples as lists for JSON)."""
        return {
            "env_name": self.env_name,
            "env_short": self.env_short,
            "method": self.method,
            "alpha_true": self.alpha_true,
            "samples": self.samples.tolist(),
            "chains": self.chains.tolist(),
            "median": self.median,
            "mean": self.mean,
            "std": self.std,
            "q16": self.q16,
            "q84": self.q84,
            "q025": self.q025,
            "q975": self.q975,
            "ess": self.ess,
            "rhat": self.rhat,
            "in_68ci": self.in_68ci,
            "in_95ci": self.in_95ci,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RecoveryResult":
        """Deserialize from dict."""
        d = dict(d)
        d["samples"] = jnp.array(d["samples"])
        d["chains"] = jnp.array(d["chains"])
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════
# Data generation: system masses from Maschberger + BinaryIMF(Moe+17)
# ═══════════════════════════════════════════════════════════════════════════

def generate_system_masses(
    alpha: float, n: int, seed: int,
) -> tuple[jnp.ndarray, float]:
    """Generate observed system masses with Moe+17 binary contamination.

    Returns:
        m_sys: System masses (n,). Singles have m_sys = m1,
               binaries have m_sys = m1 + m2.
        f_bin_actual: Realized binary fraction in the sample.
    """
    key = jax.random.PRNGKey(seed)
    imf = Maschberger(alpha=alpha, m_min=M_MIN, m_max=M_MAX)

    binary_imf = BinaryIMF(
        primary_imf=imf,
        q_distribution=MoeDiStefano2017(q_min=Q_MIN),
        binary_fraction=MassDependentBinaryFraction(),
    )
    m1, m2, is_binary = binary_imf.sample_systems(key, n)

    m_sys = jnp.where(is_binary, m1 + m2, m1)
    f_bin_actual = float(jnp.mean(is_binary.astype(float)))

    return m_sys, f_bin_actual


# ═══════════════════════════════════════════════════════════════════════════
# Gauss-Legendre quadrature for binary mixture likelihood
# ═══════════════════════════════════════════════════════════════════════════

# Precompute nodes and weights (done once at module load, outside JIT)
_GL_NODES, _GL_WEIGHTS = roots_legendre(128)
GL_NODES = jnp.array(_GL_NODES)
GL_WEIGHTS = jnp.array(_GL_WEIGHTS)


def log_system_mass_likelihood(
    m_sys: jnp.ndarray, alpha: float,
) -> float:
    """Log-likelihood of observed system masses under binary mixture model.

    p(m_sys | alpha) = (1 - f_b(m_sys)) * xi(m_sys; alpha)      [singles]
                     + integral over m1 of binary contribution    [binaries]

    The binary integral uses 128-point Gauss-Legendre quadrature over m1.
    Moe+17 binary fraction and mass-ratio distribution are fixed.
    """
    imf = Maschberger(alpha=alpha, m_min=M_MIN, m_max=M_MAX)
    bf = MassDependentBinaryFraction()
    qd = MoeDiStefano2017(q_min=Q_MIN)

    # Singles: p_single(m_sys) = (1 - f_b(m_sys)) * xi(m_sys)
    log_xi_sys = imf.logpdf(m_sys)
    f_b_sys = bf(m_sys)
    p_single = (1.0 - f_b_sys) * jnp.exp(log_xi_sys)

    # Binaries: integrate f_b(m1) * xi(m1) * g(q|m1) / m1 dm1
    # where q = m_sys/m1 - 1, m1 in [m_sys/2, m_sys/(1+q_min)]
    def _binary_integral_one(m_sys_i):
        m1_lo = jnp.maximum(m_sys_i / 2.0, M_MIN)
        m1_hi = jnp.minimum(m_sys_i / (1.0 + Q_MIN), M_MAX)
        # Guard: if m1_lo >= m1_hi, integral is zero
        valid = m1_hi > m1_lo

        # Map GL nodes from [-1, 1] to [m1_lo, m1_hi]
        half_width = 0.5 * (m1_hi - m1_lo)
        midpoint = 0.5 * (m1_hi + m1_lo)
        m1 = half_width * GL_NODES + midpoint

        q = m_sys_i / m1 - 1.0
        # Clamp q to valid range
        q = jnp.clip(q, Q_MIN, 1.0)

        f_b_m1 = bf(m1)
        xi_m1 = jnp.exp(imf.logpdf(m1))
        # vmap over GL nodes: pdf_given_primary uses jax.lax.cond
        # which requires scalar m1
        g_q = jax.vmap(lambda qi, mi: qd.pdf_given_primary(qi, mi))(q, m1)
        integrand = f_b_m1 * xi_m1 * g_q / m1

        result = half_width * jnp.dot(GL_WEIGHTS, integrand)
        return jnp.where(valid, result, 0.0)

    p_binary = jax.vmap(_binary_integral_one)(m_sys)

    p_total = p_single + p_binary
    log_likes = jnp.log(jnp.maximum(p_total, 1e-30))

    # NaN guard for NUTS leapfrog
    safe_ll = jnp.where(jnp.isfinite(log_likes), log_likes, -1e10)
    return jnp.sum(safe_ll)


# ═══════════════════════════════════════════════════════════════════════════
# NumPyro models
# ═══════════════════════════════════════════════════════════════════════════

def numpyro_model_naive(m_sys: jnp.ndarray) -> None:
    """Naive model: single-star IMF ignoring binary contamination."""
    alpha = numpyro.sample("alpha", dist.Uniform(0.5, 4.0))
    imf = Maschberger(alpha=alpha, m_min=M_MIN, m_max=M_MAX)
    log_probs = imf.logpdf(m_sys)
    safe_ll = jnp.sum(jnp.where(jnp.isfinite(log_probs), log_probs, -1e10))
    numpyro.factor("likelihood", safe_ll)


def numpyro_model_binary_aware(m_sys: jnp.ndarray) -> None:
    """Binary-aware model: mixture likelihood with Moe+17 fixed."""
    alpha = numpyro.sample("alpha", dist.Uniform(0.5, 4.0))
    ll = log_system_mass_likelihood(m_sys, alpha)
    numpyro.factor("likelihood", ll)


# ═══════════════════════════════════════════════════════════════════════════
# HMC/NUTS runner
# ═══════════════════════════════════════════════════════════════════════════

METHOD_CONFIG = {
    "naive": {
        "model": numpyro_model_naive,
        "label": "Naive (no binaries)",
    },
    "binary_aware": {
        "model": numpyro_model_binary_aware,
        "label": "Binary-aware (Moe+17)",
    },
}


def run_single_recovery(
    cfg: dict,
    color: str,
    m_sys: jnp.ndarray,
    method: str,
    seed: int,
) -> RecoveryResult:
    """Run HMC/NUTS for one environment + method (naive or binary-aware)."""
    mc = METHOD_CONFIG[method]
    alpha_true = cfg["alpha3"]
    name = cfg["name"]

    print(f"\n  {name} -- {mc['label']}")
    print(f"  True alpha = {alpha_true:.3f}")

    kernel = NUTS(mc["model"], target_accept_prob=TARGET_ACCEPT)
    mcmc = MCMC(
        kernel,
        num_warmup=N_WARMUP,
        num_samples=N_SAMPLES,
        num_chains=N_CHAINS,
        chain_method="sequential",
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(seed), m_sys)

    chains = jnp.array(mcmc.get_samples(group_by_chain=True)["alpha"])
    flat = chains.flatten()

    ess = float(effective_sample_size(chains))
    rhat = float(gelman_rubin(chains))

    median = float(jnp.median(flat))
    mean = float(jnp.mean(flat))
    std = float(jnp.std(flat))
    q16, q84 = float(jnp.percentile(flat, 16)), float(jnp.percentile(flat, 84))
    q025, q975 = (
        float(jnp.percentile(flat, 2.5)),
        float(jnp.percentile(flat, 97.5)),
    )

    in_68ci = q16 <= alpha_true <= q84
    in_95ci = q025 <= alpha_true <= q975

    result = RecoveryResult(
        env_name=name,
        env_short=cfg["short"],
        method=method,
        alpha_true=alpha_true,
        samples=flat,
        chains=chains,
        median=median, mean=mean, std=std,
        q16=q16, q84=q84, q025=q025, q975=q975,
        ess=ess, rhat=rhat,
        in_68ci=in_68ci, in_95ci=in_95ci,
        color=color,
    )

    bias = median - alpha_true
    print(f"  Recovered: {median:.3f}  bias={bias:+.4f}  "
          f"95%CI=[{q025:.3f},{q975:.3f}]  ESS={ess:.0f}  R-hat={rhat:.4f}")

    return result


def save_results(
    naive: list[RecoveryResult],
    aware: list[RecoveryResult],
) -> Path:
    """Save HMC results to JSON for fast re-plotting."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "binary_aware_results.json"
    payload = {
        "naive": [r.to_dict() for r in naive],
        "aware": [r.to_dict() for r in aware],
    }
    path.write_text(json.dumps(payload))
    print(f"  Results saved: {path}")
    return path


def load_results() -> tuple[list[RecoveryResult], list[RecoveryResult]]:
    """Load saved HMC results."""
    path = RESULTS_DIR / "binary_aware_results.json"
    if not path.exists():
        raise FileNotFoundError(f"No saved results at {path}. Run without --plot-only first.")
    payload = json.loads(path.read_text())
    naive = [RecoveryResult.from_dict(d) for d in payload["naive"]]
    aware = [RecoveryResult.from_dict(d) for d in payload["aware"]]
    print(f"  Loaded results: {path}")
    return naive, aware


def save_scaling_results(points: list[ScalingPoint]) -> Path:
    """Save scaling experiment results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "scaling_results.json"
    path.write_text(json.dumps([p.to_dict() for p in points]))
    print(f"  Scaling results saved: {path}")
    return path


def load_scaling_results() -> list[ScalingPoint]:
    """Load saved scaling results."""
    path = RESULTS_DIR / "scaling_results.json"
    if not path.exists():
        raise FileNotFoundError(f"No scaling results at {path}. Run with --run-scaling first.")
    data = json.loads(path.read_text())
    print(f"  Loaded scaling results: {path}")
    return [ScalingPoint.from_dict(d) for d in data]


def run_all_recoveries() -> tuple[list[RecoveryResult], list[RecoveryResult]]:
    """Run naive + binary-aware for all 4 environments."""
    naive_results = []
    aware_results = []

    for i, (cfg, color) in enumerate(zip(ENV_CONFIGS, ENV_COLORS)):
        seed = BASE_SEED + i
        alpha_true = cfg["alpha3"]

        print(f"\n{'='*55}")
        print(f"  {cfg['name']}  (alpha_true = {alpha_true:.3f})")
        print(f"{'='*55}")

        # Generate data once, run both models on same data
        m_sys, f_bin = generate_system_masses(alpha_true, N_MASSES, seed)
        print(f"  Generated: N={N_MASSES}, f_bin={f_bin:.2f}, "
              f"mass range=[{float(m_sys.min()):.3f}, {float(m_sys.max()):.1f}]")

        naive_r = run_single_recovery(cfg, color, m_sys, "naive", seed + 1000)
        aware_r = run_single_recovery(cfg, color, m_sys, "binary_aware",
                                      seed + 2000)

        naive_results.append(naive_r)
        aware_results.append(aware_r)

    return naive_results, aware_results


# ═══════════════════════════════════════════════════════════════════════════
# Scaling experiment: σ(α) vs N for Solar environment
# ═══════════════════════════════════════════════════════════════════════════

SCALING_N_VALUES = [500, 1000, 3000, 10_000, 30_000]
SCALING_ENV_IDX = 0  # Solar


@dataclass
class ScalingPoint:
    """Results from one (N, method) HMC run."""

    n_masses: int
    method: str
    alpha_true: float
    median: float
    std: float
    bias: float
    q025: float
    q975: float
    ci_width: float
    ess: float
    rhat: float

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "ScalingPoint":
        return cls(**d)


def run_scaling_experiment() -> list[ScalingPoint]:
    """Run naive + binary-aware at multiple N for Solar environment."""
    cfg = ENV_CONFIGS[SCALING_ENV_IDX]
    color = ENV_COLORS[SCALING_ENV_IDX]
    alpha_true = cfg["alpha3"]
    results = []

    for n in SCALING_N_VALUES:
        print(f"\n  Scaling: N={n}")
        seed = BASE_SEED + 100 + n
        m_sys, f_bin = generate_system_masses(alpha_true, n, seed)
        print(f"    f_bin={f_bin:.2f}")

        for method in ["naive", "binary_aware"]:
            r = run_single_recovery(
                cfg, color, m_sys, method, seed + (1000 if method == "naive" else 2000),
            )
            pt = ScalingPoint(
                n_masses=n,
                method=method,
                alpha_true=alpha_true,
                median=r.median,
                std=r.std,
                bias=r.median - r.alpha_true,
                q025=r.q025,
                q975=r.q975,
                ci_width=r.q975 - r.q025,
                ess=r.ess,
                rhat=r.rhat,
            )
            results.append(pt)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Panel (a): System mass function vs single-star IMF
# ═══════════════════════════════════════════════════════════════════════════

def plot_system_mf(
    ax: plt.Axes,
    naive_results: list[RecoveryResult],
    aware_results: list[RecoveryResult],
) -> None:
    """Panel (a): true single-star IMF (solid) vs observed system MF (dashed)."""
    m_grid = np.logspace(np.log10(M_MIN), np.log10(M_MAX * 2), 600)
    m_jax = jnp.array(m_grid)

    for i, (nr, ar) in enumerate(zip(naive_results, aware_results)):
        alpha = nr.alpha_true
        ls = LINE_STYLES[i % len(LINE_STYLES)]

        # Single-star IMF: solid, strong
        imf = Maschberger(alpha=alpha, m_min=M_MIN, m_max=M_MAX)
        log_p = to_numpy(imf.logpdf(m_jax))
        m_xi = m_grid * np.exp(log_p)
        ax.plot(
            m_grid, m_xi,
            color=nr.color, lw=1.3, ls=ls,
            label=rf"{nr.env_short} ($\alpha = {alpha:.2f}$)",
        )

        # System MF via KDE on log-masses
        m_sys, _ = generate_system_masses(alpha, 100_000, BASE_SEED + i + 500)
        log_m_sys = to_numpy(jnp.log10(m_sys))
        kde = gaussian_kde(log_m_sys, bw_method=0.04)
        log_m_eval = np.log10(m_grid)
        p_m = kde(log_m_eval) / (m_grid * np.log(10))
        m_xi_sys = m_grid * p_m
        # Shaded band + dashed line: all system MFs use dashed to
        # separate from the solid single-star IMFs
        ax.fill_between(
            m_grid, m_xi, m_xi_sys,
            color=nr.color, alpha=0.15, lw=0,
        )
        ax.plot(
            m_grid, m_xi_sys,
            color=nr.color, lw=1.3, ls="--", alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(M_MIN, M_MAX * 2)
    ax.set_xlabel(r"Stellar Mass $[M_\odot]$")
    ax.set_ylabel(r"$m \cdot \xi(m)$")
    ax.set_title("(a) IMF + binary distortion", fontsize=9)

    ax.legend(loc="lower left", framealpha=0.9, fontsize=6,
              edgecolor=PALETTE["light"])
    ax.text(
        0.03, 0.34,
        "solid = single-star IMF"
        "\n"
        "dashed = system MF (with binaries)",
        transform=ax.transAxes, fontsize=5.5, va="bottom",
        color="0.35", linespacing=1.4,
    )
    ax.tick_params(which="both", direction="out")
    ax.tick_params(which="minor", length=2)


# ═══════════════════════════════════════════════════════════════════════════
# Panel (b): Recovery scatter -- naive vs binary-aware
# ═══════════════════════════════════════════════════════════════════════════

def plot_recovery(
    ax: plt.Axes,
    naive_results: list[RecoveryResult],
    aware_results: list[RecoveryResult],
) -> None:
    """Panel (b): true vs recovered alpha, two series."""
    alphas = [r.alpha_true for r in naive_results]
    span = [min(alphas) - 0.15, max(alphas) + 0.15]

    # 1:1 reference line — visible but not dominant
    ax.plot(span, span, ls="-", color="0.78", lw=1.5, zorder=1)

    dodge = 0.016  # horizontal dodge between naive/aware markers

    # Per-environment label offsets (x-pts, y-pts from annotated point)
    label_offsets = {
        "Solar": (7, 7),
        "YMC": (7, -12),
        "Low-Z": (7, 6),
        "Starburst": (7, -12),
    }

    for nr, ar in zip(naive_results, aware_results):
        c = nr.color
        x_true = nr.alpha_true

        # --- Naive: diamond markers, semi-transparent fill ---
        ax.errorbar(
            x_true - dodge, nr.median,
            yerr=[[nr.median - nr.q025], [nr.q975 - nr.median]],
            fmt="none", ecolor=c, elinewidth=1.0,
            capsize=3, capthick=0.9, alpha=0.55, zorder=2,
        )
        ax.plot(
            x_true - dodge, nr.median, "D",
            color=c, alpha=0.30,
            markeredgecolor=c, markeredgewidth=1.0,
            markersize=5, zorder=4,
        )

        # --- Binary-aware: filled circles, strong ---
        ax.errorbar(
            x_true + dodge, ar.median,
            yerr=[[ar.median - ar.q025], [ar.q975 - ar.median]],
            fmt="none", ecolor=c, elinewidth=1.2,
            capsize=3, capthick=1.0, alpha=0.7, zorder=3,
        )
        ax.plot(
            x_true + dodge, ar.median, "o",
            color=c, markeredgecolor="white", markeredgewidth=0.8,
            markersize=6, zorder=5,
        )

        # Environment label
        ofs = label_offsets.get(nr.env_short, (7, 5))
        ax.annotate(
            nr.env_short,
            (x_true + dodge, ar.median),
            textcoords="offset points", xytext=ofs,
            fontsize=6, color=c, fontweight="bold",
        )

    ax.set_xlabel(r"True $\alpha$")
    ax.set_ylabel(r"Recovered $\alpha$")
    ax.set_title("(b) Parameter recovery", fontsize=9)
    ax.set_xlim(span)
    ax.set_ylim(span)

    # Legend: method markers only (env colors shown in panel a)
    ax.plot([], [], "o", color="0.45", markeredgecolor="white",
            markeredgewidth=0.8, markersize=5.5, label="Binary-aware")
    ax.plot([], [], "D", color="0.45", alpha=0.4,
            markeredgecolor="0.45", markeredgewidth=1.0,
            markersize=4.5, label="Naive")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=6.5,
              edgecolor=PALETTE["light"], handletextpad=0.4)

    ax.text(
        0.96, 0.04, rf"$N = {N_MASSES:,}$",
        transform=ax.transAxes, fontsize=6.5, va="bottom", ha="right",
        color="0.45",
    )

    ax.minorticks_on()
    ax.tick_params(which="both", direction="out")
    ax.tick_params(which="minor", length=2)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))


# ═══════════════════════════════════════════════════════════════════════════
# Panel (c): Residual posteriors -- solid (aware) vs dashed (naive)
# ═══════════════════════════════════════════════════════════════════════════

def plot_residuals(
    ax: plt.Axes,
    naive_results: list[RecoveryResult],
    aware_results: list[RecoveryResult],
) -> None:
    """Panel (c): residual posteriors, solid=binary-aware, dashed=naive."""
    x_lo, x_hi = -0.22, 0.08
    x_grid = np.linspace(x_lo, x_hi, 500)

    for nr, ar in zip(naive_results, aware_results):
        c = nr.color

        # Binary-aware: solid + fill
        delta_a = to_numpy(ar.samples) - ar.alpha_true
        kde_a = gaussian_kde(delta_a)
        dens_a = kde_a(x_grid)
        ax.fill_between(x_grid, dens_a, alpha=0.18, color=c, lw=0)
        ax.plot(x_grid, dens_a, color=c, lw=1.5)

        # Naive: dashed, no fill
        delta_n = to_numpy(nr.samples) - nr.alpha_true
        kde_n = gaussian_kde(delta_n)
        dens_n = kde_n(x_grid)
        ax.plot(x_grid, dens_n, color=c, lw=1.1, ls="--", alpha=0.75)

    ax.axvline(0, color="0.55", ls=":", lw=0.8, zorder=0)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\Delta\alpha$  (recovered $-$ true)")
    ax.set_ylabel("Density")
    ax.set_title("(c) Residual posteriors", fontsize=9)

    # Legend: 1-column, top-left (narrow, avoids KDE peaks)
    for nr in naive_results:
        ax.plot([], [], color=nr.color, lw=1.5, label=nr.env_short)
    ax.plot([], [], color="0.40", lw=1.5, label="Binary-aware")
    ax.plot([], [], color="0.40", lw=1.1, ls="--", alpha=0.75,
            label="Naive")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=5,
              edgecolor=PALETTE["light"], ncol=1,
              handletextpad=0.4, labelspacing=0.3)

    ax.minorticks_on()
    ax.tick_params(which="both", direction="out")
    ax.tick_params(which="minor", length=2)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))


# ═══════════════════════════════════════════════════════════════════════════
# Panel (d): Precision scaling -- σ(α) vs N
# ═══════════════════════════════════════════════════════════════════════════

def plot_scaling(
    ax: plt.Axes,
    scaling_points: list[ScalingPoint],
) -> None:
    """Panel (d): posterior width and bias vs sample size."""
    naive_pts = [p for p in scaling_points if p.method == "naive"]
    aware_pts = [p for p in scaling_points if p.method == "binary_aware"]

    n_naive = [p.n_masses for p in naive_pts]
    n_aware = [p.n_masses for p in aware_pts]
    sigma_naive = [p.ci_width for p in naive_pts]
    sigma_aware = [p.ci_width for p in aware_pts]
    bias_naive = [abs(p.bias) for p in naive_pts]

    c_aware = ENV_COLORS[SCALING_ENV_IDX]
    c_naive = ENV_COLORS[SCALING_ENV_IDX]

    # 95% CI width (σ proxy)
    ax.plot(n_aware, sigma_aware, "o-", color=c_aware, lw=1.3,
            markersize=5, markeredgecolor="white", markeredgewidth=0.6,
            label=r"Binary-aware 95% CI", zorder=3)
    ax.plot(n_naive, sigma_naive, "D--", color=c_naive, lw=1.0,
            markersize=4, alpha=0.5,
            markeredgecolor=c_naive, markeredgewidth=0.8,
            label=r"Naive 95% CI", zorder=2)

    # Naive |bias| — stays constant while CI shrinks
    ax.plot(n_naive, bias_naive, "s:", color="#D55E00", lw=1.0,
            markersize=4, alpha=0.8,
            label=r"Naive $|\mathrm{bias}|$", zorder=2)

    # √N reference line (anchored to aware at N=1000)
    n_ref = np.array([400, 40000])
    ref_pt = aware_pts[1] if len(aware_pts) > 1 else aware_pts[0]
    sigma_ref = ref_pt.ci_width * np.sqrt(ref_pt.n_masses / n_ref)
    ax.plot(n_ref, sigma_ref, ls="-", color="0.75", lw=0.8, zorder=0)
    ax.text(35000, sigma_ref[-1] * 1.3, r"$\propto 1/\sqrt{N}$",
            fontsize=6, color="0.55", ha="right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Sample size $N$")
    ax.set_ylabel(r"95% CI width or $|\mathrm{bias}|$")
    ax.set_title(r"(d) Precision scaling (Solar, $\alpha=2.30$)", fontsize=9)

    ax.legend(loc="upper right", framealpha=0.9, fontsize=5.5,
              edgecolor=PALETTE["light"], handletextpad=0.4)

    ax.minorticks_on()
    ax.tick_params(which="both", direction="out")
    ax.tick_params(which="minor", length=2)


# ═══════════════════════════════════════════════════════════════════════════
# Figure assembly
# ═══════════════════════════════════════════════════════════════════════════

def create_main_figure(
    naive_results: list[RecoveryResult],
    aware_results: list[RecoveryResult],
    scaling_points: list[ScalingPoint] | None = None,
) -> None:
    """Create the 2x2 proposal figure (or 1x3 if no scaling data)."""
    if scaling_points is not None:
        fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0))
        plot_system_mf(axes[0, 0], naive_results, aware_results)
        plot_recovery(axes[0, 1], naive_results, aware_results)
        plot_residuals(axes[1, 0], naive_results, aware_results)
        plot_scaling(axes[1, 1], scaling_points)
        fig.tight_layout(w_pad=1.4, h_pad=1.8)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.75))
        plot_system_mf(axes[0], naive_results, aware_results)
        plot_recovery(axes[1], naive_results, aware_results)
        plot_residuals(axes[2], naive_results, aware_results)
        fig.tight_layout(w_pad=1.2)

    savefig(fig, PLOT_DIR / "binary_aware_recovery.png")
    savefig(fig, PLOT_DIR / "binary_aware_recovery.pdf")
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR / 'binary_aware_recovery'}.{{png,pdf}}")


# ═══════════════════════════════════════════════════════════════════════════
# Pass/fail evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(
    naive_results: list[RecoveryResult],
    aware_results: list[RecoveryResult],
) -> bool:
    """Print summary and return True if all criteria pass."""
    print(f"\n{'='*65}")
    print("  RESULTS: Naive vs Binary-Aware")
    print(f"{'='*65}")
    print(f"  {'Env':>10s}  {'Method':>15s}  {'Bias':>8s}  "
          f"{'ESS':>6s}  {'R-hat':>6s}  {'95%CI':>6s}")
    print(f"  {chr(0x2500)*58}")

    all_pass = True
    for nr, ar in zip(naive_results, aware_results):
        for r in [nr, ar]:
            bias = r.median - r.alpha_true
            tag = "PASS" if r.in_95ci else "FAIL"
            label = "naive" if r.method == "naive" else "aware"
            print(f"  {r.env_short:>10s}  {label:>15s}  {bias:>+8.4f}  "
                  f"{r.ess:>6.0f}  {r.rhat:>6.4f}  [{tag}]")

    # Binary-aware pass criteria:
    #   1. At least 3/4 environments within 95% CI
    #   2. Mean |bias| < 0.05
    #   3. All ESS > 200, all R-hat < 1.05
    ci_count = sum(r.in_95ci for r in aware_results)
    ess_ok = all(r.ess > 200 for r in aware_results)
    rhat_ok = all(r.rhat < 1.05 for r in aware_results)
    aware_biases = [abs(r.median - r.alpha_true) for r in aware_results]
    mean_abs_bias = sum(aware_biases) / len(aware_biases)

    if ci_count < 3:
        all_pass = False
    if mean_abs_bias > 0.05:
        all_pass = False
    if not ess_ok or not rhat_ok:
        all_pass = False

    print(f"\n  Binary-aware 95% CI coverage: {ci_count}/4 (need >= 3)")
    print(f"  Binary-aware mean |bias|: {mean_abs_bias:.4f} (need < 0.05)")
    print(f"  ESS > 200:  {'PASS' if ess_ok else 'FAIL'}")
    print(f"  R-hat < 1.05: {'PASS' if rhat_ok else 'FAIL'}")
    print(f"  Naive bias (should be negative):")
    for nr in naive_results:
        bias = nr.median - nr.alpha_true
        print(f"    {nr.env_short}: {bias:+.4f}")

    print(f"\n  {chr(0x2500)*40}")
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"  {chr(0x2500)*40}\n")
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip HMC, re-plot from saved results",
    )
    parser.add_argument(
        "--run-scaling", action="store_true",
        help="Run scaling experiment (N vs precision) for panel (d)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  Binary-Aware IMF Recovery: Naive vs Moe+17")
    print(f"  {len(ENV_CONFIGS)} environments x 2 methods")
    print("=" * 65)

    # --- Main 4-environment recovery ---
    if args.plot_only:
        naive_results, aware_results = load_results()
    else:
        naive_results, aware_results = run_all_recoveries()
        save_results(naive_results, aware_results)

    # --- Scaling experiment (panel d) ---
    scaling_points = None
    if args.run_scaling:
        print("\n" + "=" * 65)
        print("  Scaling experiment: σ(α) vs N  (Solar environment)")
        print("=" * 65)
        scaling_points = run_scaling_experiment()
        save_scaling_results(scaling_points)
    elif (RESULTS_DIR / "scaling_results.json").exists():
        scaling_points = load_scaling_results()

    # --- Figure ---
    print("\nCreating figure...")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    set_paper(light=True)
    create_main_figure(naive_results, aware_results, scaling_points)

    passed = evaluate(naive_results, aware_results)
    if not passed:
        raise AssertionError("Binary-aware recovery FAILED -- see summary above")


if __name__ == "__main__":
    main()
