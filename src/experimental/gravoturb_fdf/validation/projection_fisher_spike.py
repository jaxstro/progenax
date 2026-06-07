r"""De-risking SPIKE (throwaway): 2D-projected vs 3D Fisher forecast for gravoturb_fdf.

Phase-0 gate for the 2D-projection-native re-scope (plan: gravoturb-fdf-re-scope-to-a-snappy-
lovelace). The science question: the real observable is a 2D projected star catalog (we never see
the 3D field). How much (mach, alpha, beta) information survives projection -- and in particular,
does alpha (a density-PDF *tail* slope) survive a tail-destroying line-of-sight integration, or
must it become depth-information-gated?

METHOD (deliberately robust, not the production estimator):
  * Generate ONE star catalog per realization (full-resolution Poisson counts from a rank-copula
    BM19 field), then view it TWO ways with the SAME stars:
        - 3D: block-sum into cubic cells of side c.
        - 2D: sum along the line-of-sight axis (the projection), then block-sum into square sky
              cells of side c.
    Same catalog, two views => the 2D/3D contrast is the *pure* cost of losing depth.
  * Data vector (identical estimator in both views): count-map band-powers (slope -> beta),
    tail-robust Var[log_plus(N)] at two cell scales (amplitude -> mach), and the linear count
    variance + skewness (tail-sensitive -> alpha).
  * Fisher by NUMERICAL derivatives of the mock-mean data vector with COMMON RANDOM NUMBERS
    (same GRF phases + Poisson key across theta-points) so J = d<d_hat>/d theta and C = Cov(d_hat)
    come from the SAME measurement estimator -> self-consistent, no analytic-vs-measured
    normalization mismatch. F = J^T C^{-1} J (Hartlap-corrected), sigma = sqrt(diag(F^{-1})).

The decisive gate metric is the sigma_2D / sigma_3D RATIO under this identical estimator on the
identical catalog -- robust to the estimator's absolute calibration. NOT a production forecast
(small box, proxy alpha features); a go/no-go on alpha's 2D reach + identifiability.

Run:
  PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
    python -m gravoturb_fdf.validation.projection_fisher_spike
"""

import jax
import numpy as np

from gravoturb_fdf.field.field import gaussian_random_field, rank_copula_field
from gravoturb_fdf.field.sampling import sample_cic_counts
from gravoturb_fdf.validation.measure import measure_log_count_variance


# ----------------------------- data-vector estimator -----------------------------

def _block_sum_3d(cnt, c):
    m = cnt.shape[0] // c
    return cnt[: m * c, : m * c, : m * c].reshape(m, c, m, c, m, c).sum(axis=(1, 3, 5))


def _block_sum_2d(col, c):
    m = col.shape[0] // c
    return col[: m * c, : m * c].reshape(m, c, m, c).sum(axis=(1, 3))


def _bandpowers(field, k_edges):
    r"""<|fft(field-<field>)|^2 / size> binned by |k| (numpy; n-d via fftfreq grid)."""
    f = np.asarray(field, dtype=float)
    f = f - f.mean()
    pk = np.abs(np.fft.fftn(f)) ** 2 / f.size
    axes = [np.fft.fftfreq(n) * n for n in f.shape]
    grids = np.meshgrid(*axes, indexing="ij")
    kmag = np.sqrt(sum(g**2 for g in grids))
    out = np.empty(len(k_edges) - 1)
    for i, (lo, hi) in enumerate(zip(k_edges[:-1], k_edges[1:])):
        mask = (kmag >= lo) & (kmag < hi)
        out[i] = pk[mask].mean() if mask.any() else 0.0
    return out


def _skew(cnt):
    x = np.asarray(cnt, dtype=float).ravel()
    mu, sd = x.mean(), x.std()
    return float(((x - mu) ** 3).mean() / sd**3) if sd > 0 else 0.0


def _dvec(cells, n_bar, k_edges):
    r"""Per-view data vector from a count grid `cells` (3D or 2D) with design mean `n_bar`."""
    bp = _bandpowers(cells, k_edges)                         # slope -> beta (+ amplitude)
    v_log = measure_log_count_variance(cells, n_bar)         # tail-robust amplitude -> mach
    delta = np.asarray(cells, float) / n_bar - 1.0
    v_lin = float(np.var(delta))                             # linear (tail-sensitive) -> alpha,mach
    sk = _skew(cells)                                        # tail skewness -> alpha
    return np.concatenate([bp, [v_log, v_lin, sk]])


def measure_views(cnt_full, c, n_bar_1, k3, k2):
    r"""Both views of ONE full-resolution catalog `cnt_full` (n,n,n).

    Returns (d_3d, d_2d). 3D: cubic cells side c. 2D: project along axis 2, square cells side c.
    Design means: N_bar_3d = c^3 n_bar_1; N_bar_2d = n c^2 n_bar_1 (projection deepens the column).
    """
    n = cnt_full.shape[0]
    c3 = _block_sum_3d(cnt_full, c)
    d3 = _dvec(c3, c**3 * n_bar_1, k3)
    col = np.asarray(cnt_full).sum(axis=2)
    c2 = _block_sum_2d(col, c)
    d2 = _dvec(c2, n * c**2 * n_bar_1, k2)
    return d3, d2


# ----------------------------- mock ensemble (CRN) -----------------------------

def _ensemble(theta, b, shape, n_bar_1, c, k_edges3, k_edges2, base_grf, base_samp, n_real):
    r"""n_real measured data vectors (3D and 2D) at theta, with COMMON RANDOM NUMBERS:
    realization r ALWAYS uses fold_in(base_grf, r) and fold_in(base_samp, r) -- identical across
    every theta-point, so finite differences isolate d/d theta (not realization scatter)."""
    mach, alpha, beta = theta
    rows3, rows2 = [], []
    for r in range(n_real):
        g = gaussian_random_field(shape, beta, jax.random.fold_in(base_grf, r))
        s = rank_copula_field(g, mach, b, alpha)
        cnt = np.asarray(sample_cic_counts(s, n_bar_1, 1, jax.random.fold_in(base_samp, r)))
        d3, d2 = measure_views(cnt, c, n_bar_1, k_edges3, k_edges2)
        rows3.append(d3)
        rows2.append(d2)
    return np.array(rows3), np.array(rows2)


def _hartlap(n_real, n_data):
    return (n_real - n_data - 2.0) / (n_real - 1.0)


def _fisher(rows_fid, J):
    r"""F = J^T Cinv J with Hartlap-corrected mock precision; returns (F, sigma, corr, cond)."""
    C = np.cov(rows_fid, rowvar=False, ddof=1)
    cinv = _hartlap(rows_fid.shape[0], rows_fid.shape[1]) * np.linalg.inv(C)
    F = J.T @ cinv @ J
    Finv = np.linalg.inv(F)
    sigma = np.sqrt(np.diag(Finv))
    d = np.sqrt(np.diag(Finv))
    corr = Finv / np.outer(d, d)
    cond = float(np.linalg.cond(F))
    return F, sigma, corr, cond


# ----------------------------- driver -----------------------------

def main(shape=(48, 48, 48), n_real=120, c=4, b=0.4,
         mach=8.0, alpha=2.5, beta=3.0, n_bar_1=0.4,
         d_mach=1.0, d_alpha=0.3, d_beta=0.3, seed=0):
    n = shape[0]
    m3 = n // c
    m2 = n // c
    # band-power edges scaled to each (coarse) cell grid; DC excluded.
    k_edges3 = np.linspace(1.0, m3 / 2.0, 4)   # 3 bins on the m3^3 cell grid
    k_edges2 = np.linspace(1.0, m2 / 2.0, 4)   # 3 bins on the m2^2 sky-cell grid
    nbar3 = c**3 * n_bar_1
    nbar2 = n * c**2 * n_bar_1
    n_stars = int(n**3 * n_bar_1)

    base_grf = jax.random.PRNGKey(seed)
    base_samp = jax.random.PRNGKey(seed + 9991)

    fid = np.array([mach, alpha, beta])
    deltas = np.array([d_mach, d_alpha, d_beta])
    names = ["mach", "alpha", "beta"]

    def ens(theta):
        return _ensemble(theta, b, shape, n_bar_1, c, k_edges3, k_edges2,
                         base_grf, base_samp, n_real)

    print("=" * 78)
    print("  2D-projected vs 3D Fisher SPIKE (same catalog, numerical-derivative, CRN)")
    print("=" * 78)
    print(f"  shape={shape} c={c} n_real={n_real}  n_stars~{n_stars}  b={b} (fixed)")
    print(f"  fiducial (mach,alpha,beta)=({mach},{alpha},{beta})  "
          f"steps d=({d_mach},{d_alpha},{d_beta})")
    print(f"  N_bar per cell: 3D={nbar3:.1f}  2D(projected column)={nbar2:.1f}")

    # --- fiducial ensemble: covariance + a contrast read-out ---
    r3_fid, r2_fid = ens(fid)
    nb = len(k_edges3) - 1
    # data-vector layout: [bp_0..bp_{nb-1}, Var_log+, Var_lin, skew]
    vlin3 = r3_fid[:, nb + 1].mean()
    vlin2 = r2_fid[:, nb + 1].mean()
    sk3 = r3_fid[:, nb + 2].mean()
    sk2 = r2_fid[:, nb + 2].mean()
    print(f"\n  CONTRAST (projection suppresses fluctuations, tames the tail):")
    print(f"    linear count variance Var(delta):  3D={vlin3:.4f}   2D={vlin2:.4f}   "
          f"(2D/3D={vlin2 / vlin3:.3f})")
    print(f"    count skewness (tail proxy):       3D={sk3:.4f}   2D={sk2:.4f}   "
          f"(2D/3D={sk2 / sk3:.3f})")

    # --- numerical Jacobians (two-sided, CRN) ---
    J3 = np.zeros((r3_fid.shape[1], 3))
    J2 = np.zeros((r2_fid.shape[1], 3))
    for p in range(3):
        tp = fid.copy(); tp[p] += deltas[p]
        tm = fid.copy(); tm[p] -= deltas[p]
        r3p, r2p = ens(tp)
        r3m, r2m = ens(tm)
        J3[:, p] = (r3p.mean(0) - r3m.mean(0)) / (2 * deltas[p])
        J2[:, p] = (r2p.mean(0) - r2m.mean(0)) / (2 * deltas[p])

    F3, s3, corr3, cond3 = _fisher(r3_fid, J3)
    F2, s2, corr2, cond2 = _fisher(r2_fid, J2)

    print(f"\n  MARGINAL 1-sigma (this small box; read the RATIO, not absolutes):")
    print(f"    {'param':<6} {'sigma_3D':>10} {'sigma_2D':>10} {'2D/3D':>8} "
          f"{'sig/fid_3D':>11} {'sig/fid_2D':>11}")
    for i, nm in enumerate(names):
        print(f"    {nm:<6} {s3[i]:>10.4f} {s2[i]:>10.4f} {s2[i] / s3[i]:>8.2f} "
              f"{100 * s3[i] / fid[i]:>10.1f}% {100 * s2[i] / fid[i]:>10.1f}%")

    print(f"\n  Fisher condition number (identifiability): 3D={cond3:.1e}  2D={cond2:.1e}")
    print(f"  correlation matrix 3D (mach,alpha,beta):")
    for row in corr3:
        print("      " + "  ".join(f"{x:+.2f}" for x in row))
    print(f"  correlation matrix 2D (mach,alpha,beta):")
    for row in corr2:
        print("      " + "  ".join(f"{x:+.2f}" for x in row))

    print(f"\n  GATE READ:")
    print(f"    beta  2D/3D sigma ratio = {s2[2] / s3[2]:.2f}   (want <~ a few -> beta survives)")
    print(f"    mach  2D/3D sigma ratio = {s2[0] / s3[0]:.2f}")
    print(f"    alpha 2D/3D sigma ratio = {s2[1] / s3[1]:.2f}   "
          f"(large -> alpha is the projection casualty -> depth-gate it)")
    print("=" * 78)
    return {"sigma_3d": s3.tolist(), "sigma_2d": s2.tolist(),
            "ratio": (s2 / s3).tolist(), "cond_3d": cond3, "cond_2d": cond2}


if __name__ == "__main__":
    main()
