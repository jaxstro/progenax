# Binary-Aware IMF Recovery — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a validation script that demonstrates binary-aware IMF slope recovery via NUTS, comparing naive (ignoring binaries) vs Moe+17-aware mixture likelihood.

**Architecture:** A single validation script (`validation/imf/validate_binary_aware_recovery.py`) generates system masses from Maschberger + BinaryIMF, runs NUTS twice per environment (naive + binary-aware), and produces a 1x3 proposal figure. The core new code is a Gauss-Legendre mixture likelihood that marginalizes over binary status.

**Tech Stack:** JAX, NumPyro (NUTS), progenax (Maschberger, BinaryIMF, MoeDiStefano2017, MassDependentBinaryFraction), matplotlib, scipy (gaussian_kde, roots_legendre), jaxstroviz.

---

### Task 1: Script scaffold with data generation

**Files:**
- Create: `validation/imf/validate_binary_aware_recovery.py`

**Step 1: Write the scaffold with imports, constants, and data generation**

```python
"""
Binary-aware IMF recovery: naive vs Moe+17-aware NUTS inference.

Generates system masses (singles + unresolved binaries) from Maschberger IMF
with Moe & Di Stefano (2017) binary population, then compares:
  - Naive: fit single-star IMF to system masses (biased)
  - Binary-aware: mixture likelihood marginalizing over binary status (unbiased)

Run:
  cd /Users/anna/projects/jaxstro-dev/progenax
  python validation/imf/validate_binary_aware_recovery.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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

ENV_COLORS = ["#66c2a5", "#fc8d62", "#e78ac3", "#7570b3"]

# ---------------------------------------------------------------------------
# MCMC configuration
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


@dataclass
class RecoveryResult:
    """Results from a single NUTS recovery run."""
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
```

**Step 2: Run a smoke test**

Run: `cd /Users/anna/projects/jaxstro-dev/progenax && python -c "
from validation.imf.validate_binary_aware_recovery import generate_system_masses
m_sys, f_bin = generate_system_masses(2.3, 1000, 42)
print(f'Generated {len(m_sys)} systems, f_bin={f_bin:.2f}, mass range=[{float(m_sys.min()):.3f}, {float(m_sys.max()):.1f}]')
"`

Expected: Something like `Generated 1000 systems, f_bin=0.40, mass range=[0.010, 200.0]`

**Step 3: Commit**

```bash
git add validation/imf/validate_binary_aware_recovery.py
git commit -m "feat(validation): scaffold binary-aware IMF recovery script"
```

---

### Task 2: Gauss-Legendre mixture likelihood

**Files:**
- Modify: `validation/imf/validate_binary_aware_recovery.py`

**Step 1: Add the quadrature likelihood function**

Add after the `generate_system_masses` function:

```python
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
        g_q = qd.pdf_given_primary(q, m1)
        integrand = f_b_m1 * xi_m1 * g_q / m1

        result = half_width * jnp.dot(GL_WEIGHTS, integrand)
        return jnp.where(valid, result, 0.0)

    p_binary = jax.vmap(_binary_integral_one)(m_sys)

    p_total = p_single + p_binary
    log_likes = jnp.log(jnp.maximum(p_total, 1e-30))

    # NaN guard for NUTS leapfrog
    safe_ll = jnp.where(jnp.isfinite(log_likes), log_likes, -1e10)
    return jnp.sum(safe_ll)
```

**Step 2: Add a numerical test**

Run: `cd /Users/anna/projects/jaxstro-dev/progenax && python -c "
import jax.numpy as jnp
from validation.imf.validate_binary_aware_recovery import log_system_mass_likelihood

# Test with a small batch of masses
m_sys = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
ll = log_system_mass_likelihood(m_sys, 2.3)
print(f'Log-likelihood: {float(ll):.4f}')
assert jnp.isfinite(ll), 'Log-likelihood is not finite!'

# Test gradient flows
grad_fn = jax.grad(lambda a: log_system_mass_likelihood(m_sys, a))
g = grad_fn(2.3)
print(f'Gradient: {float(g):.4f}')
assert jnp.isfinite(g), 'Gradient is not finite!'
print('Likelihood + gradient: OK')
"
`

Expected: Finite log-likelihood and finite gradient.

**Step 3: Commit**

```bash
git add validation/imf/validate_binary_aware_recovery.py
git commit -m "feat(validation): add Gauss-Legendre binary mixture likelihood"
```

---

### Task 3: NumPyro models and MCMC runner

**Files:**
- Modify: `validation/imf/validate_binary_aware_recovery.py`

**Step 1: Add both NumPyro models and the MCMC runner**

Add after the likelihood function:

```python
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
# MCMC runner
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
    """Run NUTS for one environment + method (naive or binary-aware)."""
    mc = METHOD_CONFIG[method]
    alpha_true = cfg["alpha3"]
    name = cfg["name"]

    print(f"\n  {name} — {mc['label']}")
    print(f"  True α = {alpha_true:.3f}")

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
    q025, q975 = float(jnp.percentile(flat, 2.5)), float(jnp.percentile(flat, 97.5))

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
          f"95%CI=[{q025:.3f},{q975:.3f}]  ESS={ess:.0f}  R̂={rhat:.4f}")

    return result


def run_all_recoveries() -> tuple[list[RecoveryResult], list[RecoveryResult]]:
    """Run naive + binary-aware for all 4 environments."""
    naive_results = []
    aware_results = []

    for i, (cfg, color) in enumerate(zip(ENV_CONFIGS, ENV_COLORS)):
        seed = BASE_SEED + i
        alpha_true = cfg["alpha3"]

        print(f"\n{'='*55}")
        print(f"  {cfg['name']}  (α_true = {alpha_true:.3f})")
        print(f"{'='*55}")

        # Generate data once, run both models on same data
        m_sys, f_bin = generate_system_masses(alpha_true, N_MASSES, seed)
        print(f"  Generated: N={N_MASSES}, f_bin={f_bin:.2f}, "
              f"mass range=[{float(m_sys.min()):.3f}, {float(m_sys.max()):.1f}]")

        naive_r = run_single_recovery(cfg, color, m_sys, "naive", seed + 1000)
        aware_r = run_single_recovery(cfg, color, m_sys, "binary_aware", seed + 2000)

        naive_results.append(naive_r)
        aware_results.append(aware_r)

    return naive_results, aware_results
```

**Step 2: Smoke test the MCMC runner with one environment**

Run: `cd /Users/anna/projects/jaxstro-dev/progenax && python -c "
from validation.imf.validate_binary_aware_recovery import *
import jax.numpy as jnp

cfg = ENV_CONFIGS[0]  # Solar
alpha_true = cfg['alpha3']
m_sys, f_bin = generate_system_masses(alpha_true, 1000, 42)
print(f'Data: N=1000, f_bin={f_bin:.2f}')

# Quick naive run (fewer samples for smoke test)
import numpyro
from numpyro.infer import MCMC, NUTS
import jax
kernel = NUTS(numpyro_model_naive, target_accept_prob=0.8)
mcmc = MCMC(kernel, num_warmup=100, num_samples=200, num_chains=1, progress_bar=True)
mcmc.run(jax.random.PRNGKey(0), m_sys)
samples = mcmc.get_samples()['alpha']
print(f'Naive median: {float(jnp.median(samples)):.3f} (true: {alpha_true:.3f})')
"`

Expected: Runs without error. Naive median likely biased negative of true.

**Step 3: Commit**

```bash
git add validation/imf/validate_binary_aware_recovery.py
git commit -m "feat(validation): add MCMC runner for naive + binary-aware models"
```

---

### Task 4: Plotting functions

**Files:**
- Modify: `validation/imf/validate_binary_aware_recovery.py`

**Step 1: Add all three panel plotting functions and figure assembly**

Add after `run_all_recoveries`:

```python
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

        # Single-star IMF
        imf = Maschberger(alpha=alpha, m_min=M_MIN, m_max=M_MAX)
        log_p = to_numpy(imf.logpdf(m_jax))
        m_xi = m_grid * np.exp(log_p)
        ax.plot(
            m_grid, m_xi,
            color=nr.color, lw=1.6, ls=LINE_STYLES[i % len(LINE_STYLES)],
            label=rf"{nr.env_short} ($\alpha={alpha:.2f}$)",
        )

        # System MF via histogram of generated data
        m_sys, _ = generate_system_masses(alpha, 50_000, BASE_SEED + i + 500)
        m_sys_np = to_numpy(m_sys)
        bins = np.logspace(np.log10(M_MIN), np.log10(m_sys_np.max()), 80)
        counts, edges = np.histogram(m_sys_np, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        m_xi_sys = centers * counts
        ax.plot(
            centers, m_xi_sys,
            color=nr.color, lw=1.0, ls=LINE_STYLES[i % len(LINE_STYLES)],
            alpha=0.4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(M_MIN, M_MAX * 2)
    ax.set_xlabel(r"Mass $[M_\odot]$")
    ax.set_ylabel(r"$m \cdot p(m)$")
    ax.set_title("(a) IMF vs system MF")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=6,
              edgecolor=PALETTE["light"])
    ax.tick_params(which="both", direction="out")
    ax.tick_params(which="minor", length=2)


# ═══════════════════════════════════════════════════════════════════════════
# Panel (b): Recovery scatter — naive vs binary-aware
# ═══════════════════════════════════════════════════════════════════════════

def plot_recovery(
    ax: plt.Axes,
    naive_results: list[RecoveryResult],
    aware_results: list[RecoveryResult],
) -> None:
    """Panel (b): true vs recovered alpha, two series."""
    alphas = [r.alpha_true for r in naive_results]
    span = [min(alphas) - 0.15, max(alphas) + 0.15]
    ax.plot(span, span, ls="-", color=PALETTE["light"], lw=1.2, zorder=1)

    for nr, ar in zip(naive_results, aware_results):
        c = nr.color
        x_true = nr.alpha_true

        # Naive: open markers
        ax.errorbar(
            x_true - 0.008, nr.median,
            yerr=[[nr.median - nr.q025], [nr.q975 - nr.median]],
            fmt="none", ecolor=c, elinewidth=0.8,
            capsize=2.5, capthick=0.8, alpha=0.35, zorder=2,
        )
        ax.plot(
            x_true - 0.008, nr.median, "o",
            color="white", markeredgecolor=c, markeredgewidth=1.2,
            markersize=5, zorder=4,
        )

        # Binary-aware: filled markers
        ax.errorbar(
            x_true + 0.008, ar.median,
            yerr=[[ar.median - ar.q025], [ar.q975 - ar.median]],
            fmt="none", ecolor=c, elinewidth=0.8,
            capsize=2.5, capthick=0.8, alpha=0.35, zorder=2,
        )
        ax.plot(
            x_true + 0.008, ar.median, "o",
            color=c, markeredgecolor="white", markeredgewidth=1.0,
            markersize=5, zorder=4,
        )

        # Label
        ax.annotate(
            nr.env_short,
            (x_true, max(nr.median, ar.median)),
            textcoords="offset points", xytext=(7, 4),
            fontsize=5.5, color=c, fontweight="bold",
        )

    ax.set_xlabel(r"True $\alpha$")
    ax.set_ylabel(r"Recovered $\alpha$")
    ax.set_title("(b) Parameter recovery")
    ax.set_xlim(span)
    ax.set_ylim(span)

    # Legend for method markers
    ax.plot([], [], "o", color="grey", markeredgecolor="white",
            markersize=5, label="Binary-aware")
    ax.plot([], [], "o", color="white", markeredgecolor="grey",
            markeredgewidth=1.2, markersize=5, label="Naive")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=6,
              edgecolor=PALETTE["light"])

    ax.text(
        0.95, 0.05, f"$N = {N_MASSES:,}$",
        transform=ax.transAxes, fontsize=6, va="bottom", ha="right",
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
# Panel (c): Residual posteriors — solid (aware) vs dashed (naive)
# ═══════════════════════════════════════════════════════════════════════════

def plot_residuals(
    ax: plt.Axes,
    naive_results: list[RecoveryResult],
    aware_results: list[RecoveryResult],
) -> None:
    """Panel (c): residual posteriors, solid=binary-aware, dashed=naive."""
    x_lo, x_hi = -0.20, 0.06
    x_grid = np.linspace(x_lo, x_hi, 400)

    for nr, ar in zip(naive_results, aware_results):
        c = nr.color

        # Binary-aware: solid + fill
        delta_a = to_numpy(ar.samples) - ar.alpha_true
        kde_a = gaussian_kde(delta_a)
        dens_a = kde_a(x_grid)
        ax.fill_between(x_grid, dens_a, alpha=0.20, color=c)
        ax.plot(x_grid, dens_a, color=c, lw=1.3, label=rf"{nr.env_short}")

        # Naive: dashed, no fill
        delta_n = to_numpy(nr.samples) - nr.alpha_true
        kde_n = gaussian_kde(delta_n)
        dens_n = kde_n(x_grid)
        ax.plot(x_grid, dens_n, color=c, lw=1.0, ls="--", alpha=0.7)

    ax.axvline(0, color=PALETTE["dark"], ls="--", lw=0.7, alpha=0.5, zorder=0)

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(r"$\Delta \alpha$  (recovered $-$ true)")
    ax.set_ylabel("Density")
    ax.set_title("(c) Residual posteriors")

    # Method legend
    ax.plot([], [], color="grey", lw=1.3, label="Binary-aware")
    ax.plot([], [], color="grey", lw=1.0, ls="--", alpha=0.7, label="Naive")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=6,
              edgecolor=PALETTE["light"])

    ax.minorticks_on()
    ax.tick_params(which="both", direction="out")
    ax.tick_params(which="minor", length=2)
    ax.xaxis.set_major_locator(MultipleLocator(0.04))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis="x", labelsize=6, rotation=45)


# ═══════════════════════════════════════════════════════════════════════════
# Figure assembly
# ═══════════════════════════════════════════════════════════════════════════

def create_main_figure(
    naive_results: list[RecoveryResult],
    aware_results: list[RecoveryResult],
) -> None:
    """Create the 1x3 proposal figure."""
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.5))

    plot_system_mf(axes[0], naive_results, aware_results)
    plot_recovery(axes[1], naive_results, aware_results)
    plot_residuals(axes[2], naive_results, aware_results)

    fig.tight_layout(w_pad=2.0)

    savefig(fig, PLOT_DIR / "binary_aware_recovery.png")
    savefig(fig, PLOT_DIR / "binary_aware_recovery.pdf")
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR / 'binary_aware_recovery'}.{{png,pdf}}")
```

**Step 2: Commit**

```bash
git add validation/imf/validate_binary_aware_recovery.py
git commit -m "feat(validation): add plotting for binary-aware recovery figure"
```

---

### Task 5: Main function and pass/fail evaluation

**Files:**
- Modify: `validation/imf/validate_binary_aware_recovery.py`

**Step 1: Add evaluation and main**

```python
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
          f"{'ESS':>6s}  {'R̂':>6s}  {'95%CI':>6s}")
    print(f"  {'─'*58}")

    all_pass = True
    for nr, ar in zip(naive_results, aware_results):
        for r in [nr, ar]:
            bias = r.median - r.alpha_true
            tag = "PASS" if r.in_95ci else "FAIL"
            label = "naive" if r.method == "naive" else "aware"
            print(f"  {r.env_short:>10s}  {label:>15s}  {bias:>+8.4f}  "
                  f"{r.ess:>6.0f}  {r.rhat:>6.4f}  [{tag}]")

        # Binary-aware must pass
        if not ar.in_95ci:
            all_pass = False

    # Check MCMC diagnostics for binary-aware runs only
    ess_ok = all(r.ess > 200 for r in aware_results)
    rhat_ok = all(r.rhat < 1.05 for r in aware_results)
    if not ess_ok or not rhat_ok:
        all_pass = False

    print(f"\n  Binary-aware 95% CI coverage: "
          f"{sum(r.in_95ci for r in aware_results)}/4")
    print(f"  Naive bias (should be negative):")
    for nr in naive_results:
        bias = nr.median - nr.alpha_true
        print(f"    {nr.env_short}: {bias:+.4f}")

    print(f"\n  {'─'*40}")
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"  {'─'*40}\n")
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("  Binary-Aware IMF Recovery: Naive vs Moe+17")
    print(f"  {len(ENV_CONFIGS)} environments × 2 methods")
    print("=" * 65)

    naive_results, aware_results = run_all_recoveries()

    print("\nCreating figure...")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    set_paper(light=True)
    create_main_figure(naive_results, aware_results)

    passed = evaluate(naive_results, aware_results)
    if not passed:
        raise AssertionError("Binary-aware recovery FAILED — see summary above")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add validation/imf/validate_binary_aware_recovery.py
git commit -m "feat(validation): add main + evaluation for binary-aware recovery"
```

---

### Task 6: Run full validation

**Step 1: Run the script**

Run: `cd /Users/anna/projects/jaxstro-dev/progenax && source /Users/anna/miniforge3/etc/profile.d/conda.sh && conda activate astro && python validation/imf/validate_binary_aware_recovery.py`

Expected runtime: ~5-10 minutes (8 NUTS runs, binary-aware ones slower due to quadrature).

Expected output:
- All 4 binary-aware runs: alpha within 95% CI
- All 4 naive runs: alpha biased negative (especially Solar)
- Figure saved to `validation/imf/plots/binary_aware_recovery.{png,pdf}`

**Step 2: If quadrature is too slow**, reduce `N_MASSES` to 5000 or `roots_legendre(128)` to `roots_legendre(64)` and re-run.

**Step 3: If binary-aware posteriors don't recover alpha**, check:
- Is `pdf_given_primary` returning correct normalization? Test with `qd = MoeDiStefano2017(); print(qd.pdf_given_primary(jnp.linspace(0.1, 1.0, 10), 1.0))`
- Is the integral bounds logic correct? Print `m1_lo, m1_hi` for a few `m_sys` values.
- Is `MassDependentBinaryFraction.__call__` the right method name? (It should be `__call__` per the source.)

**Step 4: Commit final passing version**

```bash
git add validation/imf/validate_binary_aware_recovery.py validation/imf/plots/
git commit -m "feat(validation): binary-aware IMF recovery passes all environments"
```

---

### Task 7: Figure polish (if needed)

Apply the same refinements as the IMF-only figure:
- Verify panel sizes are balanced (no panel too small)
- Check tick labels aren't overlapping
- Verify colorblind safety (same palette as IMF figure)
- Ensure consistent font sizes across all panels

Use the plot-publication skill checklist to validate.
