# Precision Scaling Panel (d) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add panel (d) to the binary-aware recovery figure showing how posterior precision σ(α) scales with sample size N, demonstrating that naive inference becomes *confidently wrong* at LSST-scale sample sizes.

**Architecture:** Run binary-aware and naive HMC at N = [500, 1000, 3000, 10000, 30000] for the Solar environment (α=2.30, steepest slope, most affected by binaries). Plot σ(α) and |bias| vs N on log-log axes. Reuse existing `generate_system_masses`, `run_single_recovery`, and `RecoveryResult` infrastructure. Save scaling results to a separate JSON file. Change figure layout from 1×3 to 2×2.

**Tech Stack:** JAX, NumPyro NUTS, matplotlib, scipy (Gauss-Legendre), jaxstroviz

---

### Task 1: Add `run_scaling_experiment` function

**Files:**
- Modify: `validation/imf/validate_binary_aware_recovery.py` (after `run_all_recoveries`, ~line 428)

**Step 1: Write the scaling experiment runner**

Add this function after `run_all_recoveries()`:

```python
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
```

**Step 2: Add save/load for scaling results**

Add after `load_results()`:

```python
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
```

**Step 3: Commit**

```bash
git add validation/imf/validate_binary_aware_recovery.py
git commit -m "feat(validation): add scaling experiment runner + serialization"
```

---

### Task 2: Add `plot_scaling` function (panel d)

**Files:**
- Modify: `validation/imf/validate_binary_aware_recovery.py` (after `plot_residuals`, ~line 644)

**Step 1: Write the scaling panel plotter**

```python
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
```

**Step 2: Commit**

```bash
git add validation/imf/validate_binary_aware_recovery.py
git commit -m "feat(validation): add panel (d) scaling plotter"
```

---

### Task 3: Change figure layout from 1×3 to 2×2

**Files:**
- Modify: `validation/imf/validate_binary_aware_recovery.py` — `create_main_figure` (~line 650)

**Step 1: Update `create_main_figure` to accept scaling data and use 2×2 layout**

Replace the existing `create_main_figure`:

```python
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
```

**Step 2: Commit**

```bash
git add validation/imf/validate_binary_aware_recovery.py
git commit -m "feat(validation): switch to 2x2 layout when scaling data available"
```

---

### Task 4: Update `main()` with `--run-scaling` CLI flag

**Files:**
- Modify: `validation/imf/validate_binary_aware_recovery.py` — `main()` (~line 730)

**Step 1: Add argparse flag and integrate scaling into main**

Replace the existing `main()`:

```python
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
```

**Step 2: Commit**

```bash
git add validation/imf/validate_binary_aware_recovery.py
git commit -m "feat(validation): add --run-scaling flag for panel (d)"
```

---

### Task 5: Run the scaling experiment

**Step 1: Run with --run-scaling**

```bash
cd /Users/anna/projects/jaxstro-dev/progenax
python validation/imf/validate_binary_aware_recovery.py --plot-only --run-scaling
```

Note: `--plot-only` loads existing 4-env results (skip re-running those). `--run-scaling` runs the 5×2 = 10 new HMC runs for Solar at varying N. Expected wall-clock: ~30-40 min on CPU.

**Step 2: Verify results saved**

Check that `validation/imf/results/scaling_results.json` exists and contains 10 entries (5 N values × 2 methods).

**Step 3: Verify figure is 2×2**

Open `validation/imf/plots/binary_aware_recovery.png` and confirm all 4 panels render correctly.

---

### Task 6: Polish panel (d) and verify figure

**Step 1: Re-plot with `--plot-only` for fast iteration**

```bash
python validation/imf/validate_binary_aware_recovery.py --plot-only
```

(This loads both the 4-env results AND the scaling results from JSON, no HMC needed.)

**Step 2: Check panel (d) visually**

Verify:
- Log-log axes with √N reference line visible
- Binary-aware CI width decreases with √N
- Naive CI width also decreases but naive |bias| stays constant (~0.07)
- At N=30000, naive |bias| > naive CI width (confidently wrong)
- Legend readable, no overlap

**Step 3: Adjust if needed**

Common fixes:
- Axis limits (xlim, ylim) for clean log-log range
- Reference line positioning
- Legend placement
- Marker sizes for print readability

**Step 4: Commit final polish**

```bash
git add validation/imf/validate_binary_aware_recovery.py
git commit -m "feat(validation): polish panel (d) precision scaling"
```

---

## Expected Results

| N | Naive |bias| | Naive 95% CI | Aware |bias| | Aware 95% CI |
|------|-------------|-------------|-------------|-------------|
| 500 | ~0.07 | ~0.12 | ~0.01 | ~0.13 |
| 1,000 | ~0.07 | ~0.08 | ~0.01 | ~0.09 |
| 3,000 | ~0.07 | ~0.05 | ~0.01 | ~0.05 |
| 10,000 | ~0.07 | ~0.03 | ~0.01 | ~0.03 |
| 30,000 | ~0.07 | ~0.02 | ~0.01 | ~0.02 |

The critical observation: at N=30,000, the naive 95% CI width (~0.02) is **smaller than the bias (~0.07)**, meaning the naive posterior **excludes the true value**. The binary-aware posterior stays centered on truth with shrinking width.

## Files to Modify

| File | Change |
|------|--------|
| `validation/imf/validate_binary_aware_recovery.py` | Add `ScalingPoint`, `run_scaling_experiment`, `save/load_scaling_results`, `plot_scaling`, update `create_main_figure` to 2×2, update `main()` with `--run-scaling` |

## Runtime

- Tasks 1-4: ~15 min (code changes only)
- Task 5: ~30-40 min (HMC runs, can run in background)
- Task 6: ~5 min (plot iteration)
