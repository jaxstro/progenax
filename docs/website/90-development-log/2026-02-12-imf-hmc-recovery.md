# IMF HMC Recovery — Proposal Figure

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prove that NumPyro NUTS recovers the high-mass IMF slope α₃ across the full astrophysical range (1.7–2.8), producing a publication-quality 1×3 proposal figure.

**Architecture:** One standalone validation script runs 4 independent NUTS inferences (one per true α₃ value), collects posterior samples and diagnostics, and renders two figure variants (table vs overlay posteriors in panel c). Uses progenax's differentiable `log_prob_masses` as the likelihood via `numpyro.factor`.

**Tech Stack:** JAX, NumPyro (NUTS), progenax.imf (IMFParams, log_prob_masses, sample_masses_from_params), jaxstroviz (newfig, savefig, plot_posterior_1d, PALETTE), matplotlib. No scipy. No bare numpy in computation — use `jax.numpy` everywhere, `to_numpy()` only at the matplotlib boundary.

---

## Task 0: Gitignore PNG files

**Files:**
- Modify: `.gitignore`

**Step 1: Add PNG/PDF exclusion**

Append to end of `progenax/.gitignore`:

```
# Validation plots (regeneratable artifacts)
*.png
*.pdf
```

**Step 2: Verify**

Run (from the repo root): `git diff .gitignore`

Expected: Shows the two new lines.

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore generated plots (*.png, *.pdf)"
```

---

## Task 1: Script skeleton — imports, constants, paths

**Files:**
- Create: `validation/validate_hmc_imf_recovery.py`

**Step 1: Write the file with imports and configuration block**

```python
"""
IMF HMC Recovery: Bayesian inference of α₃ via NumPyro NUTS.

Tests 4 astrophysically motivated α₃ values and produces a 1×3 proposal
figure showing parameter recovery across the full observed range.

SUCCESS CRITERIA:
  - All 4 α₃ values recovered within 68% credible interval
  - ESS > 200 for every run
  - R-hat < 1.05 for every run

Output:
  validation/plots/imf_hmc_recovery_table.{png,pdf}
  validation/plots/imf_hmc_recovery_posteriors.{png,pdf}

Run (from the repo root):
  env -u VIRTUAL_ENV uv run python validation/validate_hmc_imf_recovery.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Set host device count BEFORE any JAX computation for multi-chain MCMC.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size, gelman_rubin
from numpyro.infer import MCMC, NUTS

from progenax.imf import IMFParams, log_prob_masses, sample_masses_from_params

from jaxstroviz import newfig, savefig, to_numpy, set_paper, PALETTE
from jaxstroviz.plots.inference import plot_posterior_1d

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALPHA_VALUES = [2.8, 2.3, 2.0, 1.7]
ALPHA_LABELS = {
    2.8: "Bottom-heavy",
    2.3: "Kroupa",
    2.0: "Top-heavy",
    1.7: "Extreme top-heavy",
}

N_MASSES = 2000
N_WARMUP = 500
N_SAMPLES = 1000
N_CHAINS = 2
TARGET_ACCEPT = 0.8
BASE_SEED = 42

PLOT_DIR = Path(__file__).parent / "plots"


@dataclass
class RecoveryResult:
    """Results from a single NUTS recovery run."""

    alpha_true: float
    samples: jnp.ndarray       # (n_total_samples,) flattened posterior
    chains: jnp.ndarray        # (n_chains, n_samples) per-chain
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
```

**Step 2: Verify imports**

Run (from the repo root): `env -u VIRTUAL_ENV uv run python -c "exec(open('validation/validate_hmc_imf_recovery.py').read().split('ALPHA_VALUES')[0])"`

Expected: No import errors. If `jaxstroviz` not found, install it from the sibling package directory (`uv pip install -e ../jaxstroviz`).

**Step 3: Commit**

```bash
git add validation/validate_hmc_imf_recovery.py
git commit -m "feat(validation): add IMF HMC recovery script skeleton"
```

---

## Task 2: Data generation + NumPyro model

**Files:**
- Modify: `validation/validate_hmc_imf_recovery.py`

**Step 1: Add data generation and model functions**

Append after the `RecoveryResult` dataclass:

```python
# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_mock_masses(alpha3_true: float, n: int, seed: int) -> jnp.ndarray:
    """Generate n mock masses from 4-segment IMF with given α₃."""
    params = IMFParams(
        alpha0=jnp.array(0.3),
        alpha1=jnp.array(1.3),
        alpha2=jnp.array(2.3),
        alpha3=jnp.array(alpha3_true),
    )
    key = jax.random.PRNGKey(seed)
    u = jax.random.uniform(key, (n,))
    return sample_masses_from_params(params, u)


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------
def imf_model(observed_masses: jnp.ndarray) -> None:
    """NumPyro model: Uniform prior on α₃, fixed α₀/α₁/α₂."""
    alpha3 = numpyro.sample("alpha3", dist.Uniform(0.5, 4.0))

    params = IMFParams(
        alpha0=jnp.array(0.3),
        alpha1=jnp.array(1.3),
        alpha2=jnp.array(2.3),
        alpha3=alpha3,
    )

    # Per-star log probability under the 4-segment IMF
    log_probs = log_prob_masses(observed_masses, params)

    # NaN guard: NUTS leapfrog can step through α≈1 where the normalization
    # integral has a 1/(1-α) singularity. Clamp to reject those proposals.
    safe_log_lik = jnp.sum(jnp.where(jnp.isfinite(log_probs), log_probs, -1e10))
    numpyro.factor("likelihood", safe_log_lik)
```

**Step 2: Smoke-test model**

Run (from the repo root):
```bash
env -u VIRTUAL_ENV uv run python -c "
from validation.validate_hmc_imf_recovery import generate_mock_masses, imf_model
import jax.numpy as jnp
masses = generate_mock_masses(2.3, 100, 42)
print(f'Generated {len(masses)} masses, range [{float(masses.min()):.3f}, {float(masses.max()):.1f}]')
from numpyro.infer import NUTS
kernel = NUTS(imf_model)
print('NUTS kernel created OK')
"
```

Expected: Prints mass range and "NUTS kernel created OK" with no errors.

**Step 3: Commit**

```bash
git add validation/validate_hmc_imf_recovery.py
git commit -m "feat(validation): add mock data generation and NumPyro model"
```

---

## Task 3: MCMC runner + diagnostics extraction

**Files:**
- Modify: `validation/validate_hmc_imf_recovery.py`

**Step 1: Add the single-run recovery function**

Append after `imf_model`:

```python
# ---------------------------------------------------------------------------
# MCMC runner
# ---------------------------------------------------------------------------
def run_single_recovery(
    alpha3_true: float,
    n_masses: int = N_MASSES,
    seed: int = BASE_SEED,
) -> RecoveryResult:
    """Run NUTS for one α₃ value and return recovery diagnostics."""
    print(f"\n{'─'*50}")
    print(f"  α₃ = {alpha3_true:.2f}  ({ALPHA_LABELS.get(alpha3_true, '')})")
    print(f"{'─'*50}")

    # Generate mock data
    masses = generate_mock_masses(alpha3_true, n_masses, seed)
    print(f"  Masses: N={n_masses}, range=[{float(masses.min()):.3f}, "
          f"{float(masses.max()):.1f}] M☉")

    # Run NUTS
    kernel = NUTS(imf_model, target_accept_prob=TARGET_ACCEPT)
    mcmc = MCMC(
        kernel,
        num_warmup=N_WARMUP,
        num_samples=N_SAMPLES,
        num_chains=N_CHAINS,
        chain_method="sequential",
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(seed + 1000), masses)

    # Extract samples — keep as JAX arrays
    samples_by_chain = mcmc.get_samples(group_by_chain=True)["alpha3"]
    chains = jnp.array(samples_by_chain)          # (n_chains, n_samples)
    flat = chains.flatten()                        # (n_chains * n_samples,)

    # Diagnostics (all JAX-native)
    ess = float(effective_sample_size(chains))
    rhat = float(gelman_rubin(chains))

    median = float(jnp.median(flat))
    mean = float(jnp.mean(flat))
    std = float(jnp.std(flat))
    q16, q84 = float(jnp.percentile(flat, 16)), float(jnp.percentile(flat, 84))
    q025, q975 = float(jnp.percentile(flat, 2.5)), float(jnp.percentile(flat, 97.5))

    in_68ci = q16 <= alpha3_true <= q84
    in_95ci = q025 <= alpha3_true <= q975

    result = RecoveryResult(
        alpha_true=alpha3_true,
        samples=flat,
        chains=chains,
        median=median, mean=mean, std=std,
        q16=q16, q84=q84, q025=q025, q975=q975,
        ess=ess, rhat=rhat,
        in_68ci=in_68ci, in_95ci=in_95ci,
    )

    print(f"  Recovered: {median:.3f}  68% CI: [{q16:.3f}, {q84:.3f}]")
    print(f"  ESS: {ess:.0f}  R-hat: {rhat:.4f}  In 68% CI: {in_68ci}")

    return result


def run_all_recoveries() -> list[RecoveryResult]:
    """Run NUTS recovery for all 4 α₃ values."""
    results = []
    for i, alpha3 in enumerate(ALPHA_VALUES):
        result = run_single_recovery(alpha3, seed=BASE_SEED + i)
        results.append(result)
    return results
```

**Step 2: Test with ONE α value (quick sanity check, reduced N)**

Run (from the repo root):
```bash
env -u VIRTUAL_ENV uv run python -c "
from validation.validate_hmc_imf_recovery import run_single_recovery
r = run_single_recovery(2.3, n_masses=500, seed=42)
print(f'PASS: median={r.median:.3f}, ESS={r.ess:.0f}, in_68ci={r.in_68ci}')
"
```

Expected: NUTS runs (~20-30s with N=500), median near 2.3.

**Step 3: Commit**

```bash
git add validation/validate_hmc_imf_recovery.py
git commit -m "feat(validation): add MCMC runner and diagnostics extraction"
```

---

## Task 4: Panel (a) — Recovery plot

**Files:**
- Modify: `validation/validate_hmc_imf_recovery.py`

**Step 1: Add the recovery plot function**

Append after `run_all_recoveries`:

```python
# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _kde_1d(samples: jnp.ndarray, x_grid: jnp.ndarray) -> jnp.ndarray:
    """Simple Gaussian KDE in JAX (Silverman bandwidth)."""
    n = samples.shape[0]
    bw = 1.06 * jnp.std(samples) * n ** (-0.2)
    diffs = x_grid[:, None] - samples[None, :]
    kernel = jnp.exp(-0.5 * (diffs / bw) ** 2) / (bw * jnp.sqrt(2 * jnp.pi))
    return jnp.mean(kernel, axis=1)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def plot_recovery(ax: plt.Axes, results: list[RecoveryResult]) -> None:
    """Panel (a): True vs recovered α₃ with 68% and 95% CI error bars."""
    true_vals = [r.alpha_true for r in results]
    medians = [r.median for r in results]

    # Asymmetric error bars
    err_68_lo = [r.median - r.q16 for r in results]
    err_68_hi = [r.q84 - r.median for r in results]
    err_95_lo = [r.median - r.q025 for r in results]
    err_95_hi = [r.q975 - r.median for r in results]

    # 1:1 reference line
    span = [min(true_vals) - 0.2, max(true_vals) + 0.2]
    ax.plot(span, span, ls="--", color=PALETTE["medium"], lw=1, zorder=1)

    # 95% CI (thinner, lighter)
    ax.errorbar(
        true_vals, medians,
        yerr=[err_95_lo, err_95_hi],
        fmt="none", ecolor=PALETTE["primary"], elinewidth=1.0,
        capsize=3, capthick=1.0, alpha=0.4, zorder=2,
    )

    # 68% CI (thicker, bolder)
    ax.errorbar(
        true_vals, medians,
        yerr=[err_68_lo, err_68_hi],
        fmt="o", color=PALETTE["primary"], ecolor=PALETTE["primary"],
        elinewidth=2.0, capsize=4, capthick=2.0, markersize=5, zorder=3,
    )

    ax.set_xlabel(r"True $\alpha_3$")
    ax.set_ylabel(r"Recovered $\alpha_3$")
    ax.set_title("(a) Parameter recovery")
    ax.set_xlim(span)
    ax.set_ylim(span)

    ax.text(
        0.05, 0.95,
        f"$N = {N_MASSES}$ masses per trial\nNUTS: {N_WARMUP}+{N_SAMPLES}",
        transform=ax.transAxes, fontsize=7, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
    )
```

**Step 2: Verify syntax**

Run (from the repo root): `env -u VIRTUAL_ENV uv run python -c "from validation.validate_hmc_imf_recovery import plot_recovery; print('OK')"`

Expected: "OK"

**Step 3: Commit**

```bash
git add validation/validate_hmc_imf_recovery.py
git commit -m "feat(validation): add recovery plot and JAX-native KDE (panel a)"
```

---

## Task 5: Panel (b) — Example posterior + Panel (c) — Both versions

**Files:**
- Modify: `validation/validate_hmc_imf_recovery.py`

**Step 1: Add panels (b) and (c)**

Append after `plot_recovery`:

```python
def plot_example_posterior(ax: plt.Axes, result: RecoveryResult) -> None:
    """Panel (b): Posterior histogram for one α₃ case."""
    plot_posterior_1d(
        ax,
        to_numpy(result.samples),
        param_name=r"$\alpha_3$",
        truth=result.alpha_true,
        n_bins=40,
    )
    label = ALPHA_LABELS.get(result.alpha_true, "")
    ax.set_title(f"(b) Posterior: $\\alpha_3 = {result.alpha_true}$ ({label})")


def plot_diagnostics_table(ax: plt.Axes, results: list[RecoveryResult]) -> None:
    """Panel (c) Version A: Monospace diagnostics table."""
    ax.axis("off")

    header = f"{'α_true':>7s} {'α_rec':>7s} {'68% CI':>14s} {'ESS':>5s} {'R̂':>6s}"
    lines = [header, "─" * len(header)]

    for r in results:
        ci = f"[{r.q16:.2f}, {r.q84:.2f}]"
        lines.append(
            f"{r.alpha_true:7.2f} {r.median:7.3f} {ci:>14s} "
            f"{r.ess:5.0f} {r.rhat:6.3f}"
        )

    all_in_68 = all(r.in_68ci for r in results)
    all_ess = all(r.ess > 200 for r in results)
    all_rhat = all(r.rhat < 1.05 for r in results)
    ok = lambda b: "PASS" if b else "FAIL"

    lines.append("─" * len(header))
    lines.append(f"All in 68% CI:  {ok(all_in_68)}")
    lines.append(f"All ESS > 200:  {ok(all_ess)}")
    lines.append(f"All R̂ < 1.05:   {ok(all_rhat)}")

    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, fontsize=7, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#dee2e6"),
    )
    ax.set_title("(c) Diagnostics")


def plot_overlaid_posteriors(ax: plt.Axes, results: list[RecoveryResult]) -> None:
    """Panel (c) Version B: Overlaid posterior KDE curves, color-coded."""
    cmap = plt.cm.coolwarm
    norm_vals = jnp.linspace(0, 1, len(results))

    for i, r in enumerate(results):
        color = cmap(float(norm_vals[i]))

        # JAX-native KDE
        pad = 0.15
        x_grid = jnp.linspace(
            jnp.min(r.samples) - pad,
            jnp.max(r.samples) + pad,
            300,
        )
        density = _kde_1d(r.samples, x_grid)

        # Convert to numpy only at matplotlib boundary
        x_np = to_numpy(x_grid)
        d_np = to_numpy(density)

        ax.fill_between(x_np, d_np, alpha=0.25, color=color)
        ax.plot(x_np, d_np, color=color, lw=1.5,
                label=f"$\\alpha_3 = {r.alpha_true}$")
        ax.axvline(r.alpha_true, color=color, ls="--", lw=1.0, alpha=0.7)

    ax.set_xlabel(r"$\alpha_3$")
    ax.set_ylabel("Density")
    ax.set_title("(c) All posteriors")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
```

**Step 2: Commit**

```bash
git add validation/validate_hmc_imf_recovery.py
git commit -m "feat(validation): add panels b and c (posterior + table + overlay)"
```

---

## Task 6: Figure assembly + main + pass/fail

**Files:**
- Modify: `validation/validate_hmc_imf_recovery.py`

**Step 1: Add figure creation, pass/fail evaluation, and main**

Append at end of file:

```python
# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def create_figures(results: list[RecoveryResult]) -> None:
    """Create both versions of the 1×3 proposal figure."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    set_paper(light=True)

    # Pick the extreme top-heavy case for panel (b)
    example = next(r for r in results if r.alpha_true == 1.7)

    for version, panel_c_fn, suffix in [
        ("A (table)", plot_diagnostics_table, "table"),
        ("B (posteriors)", plot_overlaid_posteriors, "posteriors"),
    ]:
        fig, axes = newfig(width=7.0, aspect=0.38, nrows=1, ncols=3)

        plot_recovery(axes[0], results)
        plot_example_posterior(axes[1], example)
        panel_c_fn(axes[2], results)

        fig.tight_layout(w_pad=2.5)

        stem = f"imf_hmc_recovery_{suffix}"
        savefig(fig, PLOT_DIR / f"{stem}.png")
        savefig(fig, PLOT_DIR / f"{stem}.pdf")
        plt.close(fig)

        print(f"  Saved: {PLOT_DIR / stem}.{{png,pdf}}  (version {version})")


# ---------------------------------------------------------------------------
# Pass/fail evaluation
# ---------------------------------------------------------------------------
def evaluate_pass_fail(results: list[RecoveryResult]) -> bool:
    """Print summary and return True if all criteria pass."""
    n = len(results)
    print("\n" + "=" * 60)
    print("  PASS / FAIL SUMMARY")
    print("=" * 60)

    all_pass = True

    for r in results:
        status = "PASS" if r.in_68ci else "FAIL"
        if not r.in_68ci:
            all_pass = False
        print(
            f"  α₃={r.alpha_true:.2f}: median={r.median:.3f} "
            f"CI=[{r.q16:.3f},{r.q84:.3f}] ESS={r.ess:.0f} "
            f"R̂={r.rhat:.4f}  [{status}]"
        )

    ess_ok = all(r.ess > 200 for r in results)
    rhat_ok = all(r.rhat < 1.05 for r in results)
    ci_ok = all(r.in_68ci for r in results)

    if not ess_ok:
        all_pass = False
    if not rhat_ok:
        all_pass = False

    print(f"\n  68% CI coverage: {sum(r.in_68ci for r in results)}/{n}")
    print(f"  ESS > 200:       {'PASS' if ess_ok else 'FAIL'}")
    print(f"  R̂ < 1.05:        {'PASS' if rhat_ok else 'FAIL'}")
    print(f"\n  {'='*40}")

    if all_pass:
        print("  OVERALL: PASS")
    else:
        print("  OVERALL: FAIL")

    print(f"  {'='*40}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  IMF HMC Recovery: NumPyro NUTS × progenax")
    print(f"  {len(ALPHA_VALUES)} α₃ values across the astrophysical range")
    print("=" * 60)

    results = run_all_recoveries()

    print("\nCreating figures...")
    create_figures(results)

    passed = evaluate_pass_fail(results)

    if not passed:
        raise AssertionError("IMF HMC recovery FAILED — see summary above")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add validation/validate_hmc_imf_recovery.py
git commit -m "feat(validation): add figure assembly, pass/fail, and main"
```

---

## Task 7: Run full script and verify

**Step 1: Run the complete validation**

Run (from the repo root):
```bash
env -u VIRTUAL_ENV uv run python validation/validate_hmc_imf_recovery.py
```

Expected: ~3-5 min total. All 4 α₃ values recovered. "OVERALL: PASS".

**Step 2: Verify figure files exist**

Run: `ls -la validation/plots/imf_hmc_recovery_*.{png,pdf}`

Expected: 4 files — `_table.png`, `_table.pdf`, `_posteriors.png`, `_posteriors.pdf`.

**Step 3: Visual check**

Open both PNG files:
- Panel (a): 4 points on 1:1 line with error bars
- Panel (b): Clean posterior histogram for α₃=1.7
- Panel (c): Either diagnostics table OR overlaid KDE curves

**Step 4: Final commit**

```bash
git add validation/validate_hmc_imf_recovery.py
git commit -m "feat(validation): IMF HMC recovery complete — 4-α NUTS proposal figure"
```

---

## Task 8: Debug and iterate (if needed)

Common issues and fixes:

**If NUTS diverges or gets stuck:**
- Check NaN guard is active (the `jnp.where(jnp.isfinite(...))` line)
- Try `target_accept_prob=0.9` (more conservative step sizes)
- Reduce `N_MASSES=1000` for faster debugging

**If ESS is low (<200):**
- Increase `N_SAMPLES` to 2000
- Increase `N_WARMUP` to 1000

**If jaxstroviz import fails:**
- Install it from the sibling package directory: `uv pip install -e ../jaxstroviz`

**If XLA_FLAGS device count errors:**
- Set `N_CHAINS=1` and skip R-hat check (single chain still gives ESS + CI)

**If `plot_posterior_1d` signature doesn't match:**
- Fall back to manual: `ax.hist(to_numpy(samples), bins=40, density=True, ...)`
