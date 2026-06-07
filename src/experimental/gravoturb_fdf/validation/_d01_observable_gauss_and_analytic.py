r"""D1+D3a+D3b — the Phase-0 gate diagnostic: is an ANALYTIC forward model possible?

Shares ONE beta-grid field ensemble across three questions (Anna 2026-06-07 deep dive):

  D1  (transform-skew, truthfulness gate): per-|k|-bin skew / excess-kurtosis of the projected
      band-powers under {raw, log, rank-G} maps, at the fiducial theta. Reproduces the retrospective
      claim raw->~12 / log->~3 / rank-G->~0.5 *first-hand* before we build on it.

  D3a (analytic-likelihood-on-raw): does a Hamimeche & Lewis 2008 (g(x)=sign(x-1)sqrt(2(x-ln x-1)),
      applied to x = P_bin / <P_bin>) or a log (Carron-lognormal leading term) transform of the RAW
      band-powers Gaussianize the per-bin marginal? If residual skew stays >>1 at high k, the field
      non-Gaussianity (not the estimator chi^2) defeats analytic-on-raw band-powers.

  D3b (THE GATE — analytic for the Gaussianized observable): compare the log-log beta-slope of
      ANALYTIC predictions vs SIMULATOR observables across the beta prior:
        ANALYTIC  A_rho : angular_bandpowers_2d_limber           (projected DENSITY, exact Mehler d_n)
        ANALYTIC  A_s   : projected analytic LOG-density 2-pt     (Limber of xi_s via c_n)
        SIM       S_rho : measure bp of exp(s).sum(LOS)           (projected density oracle)
        SIM       S_s   : measure bp of s.sum(LOS)                (projected log-density oracle; not obs.)
        OBS       O_logp: measure bp of log_plus(project counts)  (Neyrinck Eq.2 observable)
        OBS       O_rg  : measure bp of rank_G(project counts)    (Neyrinck Eq.1 observable)
      If some ANALYTIC X has slope(beta) tracking some near-Gaussian OBS Y to within a consistent,
      ~few-% transfer across beta  ==>  analytic forward model + Gaussian likelihood is POSSIBLE.

Config mirrors v2h (64^3, depth=64, M=8, b=0.4, alpha=2.5, N_stars=1e5, k in [1,28], 10 bins).
EXPERIMENTAL scratch; no production edits, no commits.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_d01_observable_gauss_and_analytic.py
"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special, stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gravoturb_fdf.field.field import gaussian_random_field
from gravoturb_fdf.field.sampling import sample_cic_counts
from gravoturb_fdf.inference.covariance import (
    _angular_bandpowers_from_xi_rho_2d,
    angular_bandpowers_2d_limber,
)
from gravoturb_fdf.theory.gaussianization import (
    bm19_hermite_coefficients,
    gaussianized_xi,
)
from gravoturb_fdf.theory.projection import (
    gaussian_correlation_grid,
    limber_project_slab,
)
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    project_counts_los,
    smooth_copula_field,
)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
SHAPE, DEPTH = (64, 64, 64), 64
B_FIXED, ALPHA = 0.4, 2.5
K_EDGES = np.linspace(1.0, 28.0, 11)              # 10 bins, as v2h
N_STARS = 10**5
N_BAR_3D = N_STARS / (SHAPE[0] ** 3)
BETAS = np.array([2.0, 2.5, 3.0, 3.5, 11.0 / 3.0])
M_FID = 8.0
N_REAL = 24
N_MAX = 14


# ----------------------------- transforms -----------------------------
def log_plus(N, n_bar):
    """Neyrinck+2011 Eq.2 modified log on a count map: A = ln(1+d) for d>0 else d, d = N/nbar - 1."""
    d = np.asarray(N, float) / n_bar - 1.0
    return np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)


def rank_gaussianize_2d(map2d):
    """Neyrinck+2011 Eq.1: pixel -> Gaussian quantile of its rank -> exact N(0,1) marginal."""
    f = np.asarray(map2d, float).ravel()
    ranks = np.argsort(np.argsort(f))
    u = (ranks + 0.5) / f.size
    return (np.sqrt(2.0) * special.erfinv(2.0 * u - 1.0)).reshape(map2d.shape)


def hamimeche_lewis(x):
    """H&L 2008 transform g(x)=sign(x-1) sqrt(2(x - ln x - 1)); x = P/<P> (>0). g(1)=0, ~Gaussian."""
    x = np.asarray(x, float)
    xs = np.where(x > 1e-12, x, 1e-12)
    return np.sign(xs - 1.0) * np.sqrt(2.0 * (xs - np.log(xs) - 1.0))


# ----------------------------- analytic predictors -----------------------------
def analytic_density_bp(beta, M):
    """A_rho: projected-DENSITY band-powers (existing validated predictor; exact Mehler d_n)."""
    _kc, P, _nm = angular_bandpowers_2d_limber(
        SHAPE, beta, M, B_FIXED, ALPHA, DEPTH, K_EDGES, n_max=N_MAX
    )
    return np.asarray(P)


def analytic_logdensity_bp(beta, M):
    """A_s: projected analytic LOG-density band-powers = bin( fft2( Limber( xi_s ) ) ).

    xi_s(r) = sum_{n>=1} c_n^2/n! rho_g(r)^n with c_n the log-density Hermite coeffs. Same Limber
    + 2-D periodogram binning as angular_bandpowers_2d_limber, but on s (near-Gaussian) not e^s."""
    rho_g = gaussian_correlation_grid(SHAPE, beta)
    c = bm19_hermite_coefficients(M, B_FIXED, ALPHA, N_MAX)
    xi_s = gaussianized_xi(rho_g, c)
    xi_Sigma = limber_project_slab(xi_s, DEPTH, los_axis=2)
    _kc, P, _nm = _angular_bandpowers_from_xi_rho_2d(xi_Sigma, K_EDGES)
    return np.asarray(P)


# ----------------------------- one realization: all observables -----------------------------
def measure_all(beta, M, key):
    """Return dict of band-power vectors for every sim/obs channel for ONE field realization."""
    g = gaussian_random_field(SHAPE, beta, key)
    s = smooth_copula_field(g, M, B_FIXED, ALPHA)               # numpy log-density, <e^s>~1
    rho = np.exp(s)
    cnt = np.asarray(sample_cic_counts(jnp.asarray(s), N_BAR_3D, 1, jax.random.fold_in(key, 1)))
    proj_cnt = project_counts_los(cnt, DEPTH, los_axis=2)       # 2-D count map
    n_bar_sky = proj_cnt.mean()
    return {
        "S_rho": measure_angular_bandpowers_2d(rho.sum(axis=2), K_EDGES),
        "S_s": measure_angular_bandpowers_2d(s.sum(axis=2), K_EDGES),
        "O_raw": measure_angular_bandpowers_2d(proj_cnt.astype(float), K_EDGES),
        "O_logp": measure_angular_bandpowers_2d(log_plus(proj_cnt, n_bar_sky), K_EDGES),
        "O_rg": measure_angular_bandpowers_2d(rank_gaussianize_2d(proj_cnt), K_EDGES),
    }


def loglog_slope(k, P):
    """Log-log slope of band-power vs |k| (the beta carrier)."""
    P = np.asarray(P)
    good = (P > 0) & np.isfinite(P)
    if good.sum() < 2:
        return np.nan
    return np.polyfit(np.log(np.asarray(k)[good]), np.log(P[good]), 1)[0]


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    k_cent = 0.5 * (K_EDGES[:-1] + K_EDGES[1:])
    print(f"D01 gate diagnostic  shape={SHAPE} depth={DEPTH} N_real={N_REAL} "
          f"betas={BETAS.round(3)} M={M_FID}")
    t0 = time.time()

    # ensemble: for each beta, N_real realizations of every channel
    chans = ["S_rho", "S_s", "O_raw", "O_logp", "O_rg"]
    bp = {c: np.zeros((len(BETAS), N_REAL, len(k_cent))) for c in chans}
    for bi, beta in enumerate(BETAS):
        for r in range(N_REAL):
            key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026), bi), r)
            d = measure_all(float(beta), M_FID, key)
            for c in chans:
                bp[c][bi, r] = d[c]
        print(f"  beta={beta:.3f} done ({time.time()-t0:.0f}s)")

    # analytic predictions per beta
    A_rho = np.array([analytic_density_bp(float(b), M_FID) for b in BETAS])
    A_s = np.array([analytic_logdensity_bp(float(b), M_FID) for b in BETAS])

    # ---------------- D1 + D3a: per-bin skew/kurtosis of raw band-powers + transforms (fiducial beta=3) ----------------
    bfid = int(np.argmin(np.abs(BETAS - 3.0)))
    raw = bp["O_raw"][bfid]                       # (N_real, nbins)
    mu_raw = raw.mean(axis=0)
    print("\n[D1/D3a] per-|k|-bin marginal non-Gaussianity at beta=3 (N_real samples per bin):")
    print(f"  {'k':>6} {'raw_skew':>9} {'raw_exk':>8} | {'log_skew':>9} {'HL_skew':>8} {'rankG_skew':>11}")
    for j in range(len(k_cent)):
        col = raw[:, j]
        log_col = np.log(np.where(col > 0, col, np.nan))
        hl_col = hamimeche_lewis(col / mu_raw[j])
        rg_col = bp["O_rg"][bfid][:, j]           # rank-G map's band-power (already Gaussianized field)
        print(f"  {k_cent[j]:6.1f} {stats.skew(col):9.2f} {stats.kurtosis(col):8.1f} | "
              f"{stats.skew(log_col, nan_policy='omit'):9.2f} {stats.skew(hl_col):8.2f} "
              f"{stats.skew(rg_col):11.2f}")

    # ---------------- D3b: slope-vs-beta, analytic vs observable ----------------
    print("\n[D3b] log-log band-power slope vs beta (mean over realizations); "
          "analytic vs simulator/observable:")
    rows = {}
    rows["A_rho(analytic density)"] = [loglog_slope(k_cent, A_rho[bi]) for bi in range(len(BETAS))]
    rows["A_s(analytic logdens) "] = [loglog_slope(k_cent, A_s[bi]) for bi in range(len(BETAS))]
    for c, label in [("S_rho", "S_rho(sim density)    "), ("S_s", "S_s(sim logdens)      "),
                     ("O_raw", "O_raw(counts)         "), ("O_logp", "O_logp(log+ counts)   "),
                     ("O_rg", "O_rg(rankG counts)    ")]:
        rows[label] = [loglog_slope(k_cent, bp[c][bi].mean(axis=0)) for bi in range(len(BETAS))]

    hdr = "  " + " ".join(f"b={b:5.2f}" for b in BETAS) + "   d(slope)/d(beta)"
    print(hdr)
    for label, sl in rows.items():
        sl = np.array(sl)
        gain = np.polyfit(BETAS, sl, 1)[0]
        print(f"  {label} " + " ".join(f"{x:7.3f}" for x in sl) + f"   gain={gain:6.3f}")

    # transfer: does an analytic slope TRACK an observable slope across beta? (slope-of-slopes ratio)
    print("\n[D3b] transfer = d(slope_obs)/d(slope_analytic) across beta "
          "(==1 means analytic predicts the observable's beta-response):")
    def dgain(a):
        return np.polyfit(BETAS, np.array(a), 1)[0]
    g_As = dgain(rows["A_s(analytic logdens) "])
    g_Arho = dgain(rows["A_rho(analytic density)"])
    for label in ["O_logp(log+ counts)   ", "O_rg(rankG counts)    ", "S_s(sim logdens)      ",
                  "S_rho(sim density)    "]:
        g_obs = dgain(rows[label])
        print(f"  {label}: vs A_s {g_obs/g_As:6.3f}   vs A_rho {g_obs/g_Arho:6.3f}")

    # ---------------- figures ----------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for label, sl in rows.items():
        style = "--" if label.startswith("A_") else "-"
        ax[0].plot(BETAS, sl, style, marker="o", label=label.strip())
    ax[0].set_xlabel("beta"); ax[0].set_ylabel("log-log band-power slope")
    ax[0].set_title("D3b: slope vs beta (dashed=analytic, solid=sim/obs)")
    ax[0].legend(fontsize=7)
    # skew vs k for the three transforms
    skews = {"raw": [stats.skew(raw[:, j]) for j in range(len(k_cent))],
             "log": [stats.skew(np.log(np.where(raw[:, j] > 0, raw[:, j], np.nan)), nan_policy='omit')
                     for j in range(len(k_cent))],
             "H&L": [stats.skew(hamimeche_lewis(raw[:, j] / mu_raw[j])) for j in range(len(k_cent))],
             "rank-G": [stats.skew(bp["O_rg"][bfid][:, j]) for j in range(len(k_cent))]}
    for label, sk in skews.items():
        ax[1].plot(k_cent, sk, marker="s", label=label)
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xlabel("|k|"); ax[1].set_ylabel("band-power skew")
    ax[1].set_title("D1/D3a: per-bin skew vs |k| by transform (beta=3)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "d01_gate.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    print(f"\nfigure: {path}\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
