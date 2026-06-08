"""SCRATCH (verify-first, NOT production): per-step power-spectrum slope transfer.

Measures where the turbulent density power-spectrum slope beta gets distorted along the
chain  GRF g -> s (log-density) -> e^s (density) -> 2D projection -> rank-Gaussianized
projection -> rank-Gaussianized projected COUNTS, using ONE identical slope estimator at
every step so cross-space differences are physical, not estimator artifacts.

Reconciles design-doc s14's rank-Gaussianized-angular-clustering claim
(dslope/dbeta ~ 1, slopes ~ 2.3/2.8/3.35 for beta=2.5/3.0/3.5, sigma(beta) ~ 0.3-0.5).

Run:
  PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
      python src/experimental/gravoturb_fdf/validation/_v1a_transfer_decomposition.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import erfinv

from gravoturb_fdf.field.field import (
    gaussian_random_field,
    mass_conserving_copula_field,
    rank_copula_field,
)
from gravoturb_fdf.field.sampling import sample_cic_counts
from gravoturb_fdf.validation.measure import project_counts_los

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SHAPE = (96, 96, 96)
N_SIDE = SHAPE[0]
BETAS = [2.0, 2.5, 3.0, 3.5]
MACH, B, ALPHA = 8.0, 0.4, 2.5
N_REAL = 30
N_STARS = 10000
BASE_KEY = jax.random.PRNGKey(20260607)

# k-range: k in [1, n/4], 10 log bins (stated choice).  Same edges used for 2D and 3D.
K_LO, K_HI, N_KBINS = 1.0, N_SIDE / 4.0, 10
K_EDGES = np.logspace(np.log10(K_LO), np.log10(K_HI), N_KBINS + 1)

SPACE_NAMES = [
    "1: GRF g (3D)",
    "2: s log-density (3D)",
    "3: e^s density (3D)",
    "4: projected density (2D)",
    "5: rank-G projected density (2D)",
    "6: rank-G projected COUNTS (2D)",
]


# ---------------------------------------------------------------------------
# Identical slope estimator for any-dim field
# ---------------------------------------------------------------------------
def _kmag(shape):
    axes = [np.fft.fftfreq(n) * n for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.sqrt(sum(g**2 for g in grids))


def slope_of(field_nd, k_edges):
    """Mean-subtract -> P(k)=|fftn|^2/size -> radial-bin |k| -> LSQ fit log P vs log k.

    Returns (slope_positive, n_bins_used).  slope_positive is the exponent magnitude:
    for P(k) ∝ k^{-beta} we report +beta.
    """
    f = np.asarray(field_nd, dtype=float)
    f = f - f.mean()
    pk = np.abs(np.fft.fftn(f)) ** 2 / f.size
    kmag = _kmag(f.shape)
    pk_f, k_f = pk.ravel(), kmag.ravel()
    keep = k_f > 0
    pk_f, k_f = pk_f[keep], k_f[keep]

    ke = np.asarray(k_edges)
    kc, pc = [], []
    for lo, hi in zip(ke[:-1], ke[1:]):
        m = (k_f >= lo) & (k_f < hi)
        if m.any():
            kc.append(np.exp(np.mean(np.log(k_f[m]))))  # log-mean k of bin
            pc.append(pk_f[m].mean())
    kc, pc = np.asarray(kc), np.asarray(pc)
    good = pc > 0
    kc, pc = kc[good], pc[good]
    if kc.size < 2:
        return np.nan, kc.size
    coeffs = np.polyfit(np.log(kc), np.log(pc), 1)
    return -coeffs[0], kc.size  # negate so P∝k^{-beta} -> +beta


def rank_gaussianize_2d(map2d):
    """Neyrinck+2011 Eq.1 rank->standard-normal map: G = sqrt(2) erfinv(2 (rank+0.5)/N - 1)."""
    flat = np.asarray(map2d, dtype=float).ravel()
    n = flat.size
    ranks = np.argsort(np.argsort(flat))
    u = (ranks + 0.5) / n
    g = np.sqrt(2.0) * erfinv(2.0 * u - 1.0)
    return g.reshape(map2d.shape)


# ---------------------------------------------------------------------------
# Measurement loop
# ---------------------------------------------------------------------------
def measure_all():
    # slopes[space][beta_idx] -> list over realizations
    slopes = {name: {b: [] for b in BETAS} for name in SPACE_NAMES}

    for bi, beta in enumerate(BETAS):
        for r in range(N_REAL):
            key = jax.random.fold_in(jax.random.fold_in(BASE_KEY, bi), r)
            k_field, k_count = jax.random.split(key)

            g = gaussian_random_field(SHAPE, beta, k_field)
            s = mass_conserving_copula_field(g, MACH, B, ALPHA)
            g_np = np.asarray(g)
            s_np = np.asarray(s)
            rho_np = np.exp(s_np)

            # 1 GRF, 2 log-density, 3 density (all 3D)
            slopes[SPACE_NAMES[0]][beta].append(slope_of(g_np, K_EDGES)[0])
            slopes[SPACE_NAMES[1]][beta].append(slope_of(s_np, K_EDGES)[0])
            slopes[SPACE_NAMES[2]][beta].append(slope_of(rho_np, K_EDGES)[0])

            # 4 projected density (full-depth column sum)
            proj = rho_np.sum(axis=2)
            slopes[SPACE_NAMES[3]][beta].append(slope_of(proj, K_EDGES)[0])

            # 5 rank-Gaussianized projected density
            proj_rg = rank_gaussianize_2d(proj)
            slopes[SPACE_NAMES[4]][beta].append(slope_of(proj_rg, K_EDGES)[0])

            # 6 rank-Gaussianized projected COUNTS (Poisson, full-depth)
            cnt3d = np.asarray(sample_cic_counts(s, float(N_STARS) / s.size, 1, k_count))
            cnt2d = project_counts_los(cnt3d, depth=SHAPE[2], los_axis=2)
            cnt_rg = rank_gaussianize_2d(cnt2d.astype(float))
            slopes[SPACE_NAMES[5]][beta].append(slope_of(cnt_rg, K_EDGES)[0])

        print(f"  done beta={beta}")
    return slopes


def summarize(slopes):
    """Return per-space dict: mean[beta], sig[beta], gain (dslope/dbeta), gain_intercept."""
    betas = np.asarray(BETAS)
    summary = {}
    for name in SPACE_NAMES:
        means = np.array([np.mean(slopes[name][b]) for b in BETAS])
        sigs = np.array([np.std(slopes[name][b], ddof=1) for b in BETAS])
        gain, intercept = np.polyfit(betas, means, 1)
        summary[name] = dict(means=means, sigs=sigs, gain=gain, intercept=intercept)
    return summary


def print_report(summary):
    print("\n" + "=" * 92)
    print("SLOPE-vs-BETA TABLE  (measured positive slope = exponent magnitude; ML=8, b=0.4, "
          "alpha=2.5)")
    print(f"  shape={SHAPE}  n_real={N_REAL}  k in [{K_LO:.2f}, {K_HI:.2f}] {N_KBINS} log bins")
    print("=" * 92)
    hdr = f"{'space':<34}" + "".join(f"  b={b:<6}" for b in BETAS) + "  dslope/dbeta"
    print(hdr)
    print("-" * 92)
    for name in SPACE_NAMES:
        s = summary[name]
        row = f"{name:<34}"
        for i in range(len(BETAS)):
            row += f"  {s['means'][i]:5.2f}±{s['sigs'][i]:.2f}"
        row += f"   {s['gain']:5.3f}"
        print(row)

    print("\nIMPLIED per-cluster sigma(beta) = sigma_slope / (dslope/dbeta)  "
          "[observable spaces 4/5/6]:")
    for name in [SPACE_NAMES[3], SPACE_NAMES[4], SPACE_NAMES[5]]:
        s = summary[name]
        gain = s["gain"]
        sig_mean = np.mean(s["sigs"])
        implied = sig_mean / gain if gain != 0 else np.inf
        print(f"  {name:<34}  gain={gain:5.3f}  <sigma_slope>={sig_mean:.3f}  "
              f"-> sigma(beta)~{implied:6.3f}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(summary, outpath):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    betas = np.asarray(BETAS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: measured slope vs beta_true, one line per space + y=x
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(SPACE_NAMES)))
    for name, c in zip(SPACE_NAMES, colors):
        s = summary[name]
        ax1.errorbar(betas, s["means"], yerr=s["sigs"], marker="o", capsize=3,
                     color=c, label=name)
    lims = [min(BETAS) - 0.3, max(BETAS) + 0.3]
    ax1.plot(lims, lims, "k--", lw=1, label="y = x (slope-preserving)")
    ax1.set_xlabel(r"$\beta_{\rm true}$")
    ax1.set_ylabel("measured positive slope")
    ax1.set_title("Power-spectrum slope transfer per step")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_xlim(lims)

    # Panel 2: bars of dslope/dbeta per space + implied sigma(beta) for 4/5/6
    gains = [summary[n]["gain"] for n in SPACE_NAMES]
    x = np.arange(len(SPACE_NAMES))
    bars = ax2.bar(x, gains, color=colors, alpha=0.85)
    ax2.axhline(1.0, color="k", ls="--", lw=1, label="gain = 1 (slope-preserving)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.split(":")[0] for n in SPACE_NAMES])
    ax2.set_xlabel("space #")
    ax2.set_ylabel(r"$d({\rm slope})/d\beta$  (transfer gain)")
    ax2.set_title("Transfer gain per space  +  implied $\\sigma(\\beta)$ for 4/5/6")
    for xi, g in zip(x, gains):
        ax2.text(xi, g + 0.02, f"{g:.2f}", ha="center", fontsize=8)
    # annotate implied sigma(beta) for observable spaces
    for idx in (3, 4, 5):
        s = summary[SPACE_NAMES[idx]]
        sig_b = np.mean(s["sigs"]) / s["gain"] if s["gain"] != 0 else np.inf
        ax2.text(idx, gains[idx] / 2, f"$\\sigma_\\beta$\n{sig_b:.2f}",
                 ha="center", va="center", fontsize=8, color="white", weight="bold")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=130)
    print(f"\nwrote {outpath}")


def reconcile_s14(summary):
    print("\n" + "=" * 92)
    print("RECONCILIATION vs design-doc s14 (rank-G angular-clustering: -2.30/-2.81/-3.35 "
          "for b=2.5/3.0/3.5,")
    print("  dslope/dbeta~1, sigma(beta)~0.3-0.5).  Closest analogs = spaces 5 and 6.")
    print("=" * 92)
    for idx in (4, 5):
        name = SPACE_NAMES[idx]
        s = summary[name]
        # slopes at beta 2.5/3.0/3.5 = indices 1,2,3
        sl = s["means"]
        sig_b = np.mean(s["sigs"]) / s["gain"] if s["gain"] != 0 else np.inf
        print(f"\n{name}")
        print(f"  slopes @ beta=2.5/3.0/3.5 : {sl[1]:.2f} / {sl[2]:.2f} / {sl[3]:.2f}")
        print(f"  dslope/dbeta = {s['gain']:.3f}   implied sigma(beta) ~ {sig_b:.3f}")


if __name__ == "__main__":
    print("Measuring slope transfer (this takes a few minutes)...")
    slopes = measure_all()
    summary = summarize(slopes)
    print_report(summary)
    reconcile_s14(summary)
    make_plot(
        summary,
        "src/experimental/gravoturb_fdf/validation/plots/v1a_transfer_decomposition.png",
    )

    # Diagnosis deltas
    print("\n" + "=" * 92)
    print("DIAGNOSIS: per-step Delta(slope) averaged over beta (positive = flattening/"
          "compression)")
    print("=" * 92)
    chain = SPACE_NAMES
    prev = None
    for name in chain:
        m = summary[name]["means"]
        if prev is not None:
            d = np.mean(summary[prev]["means"] - m)  # prev - this; positive = flattened
            print(f"  {prev.split(':')[0]} -> {name.split(':')[0]} : "
                  f"mean Delta(slope) = {d:+.3f}")
        prev = name
