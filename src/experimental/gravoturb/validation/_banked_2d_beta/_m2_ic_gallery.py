"""Visual sanity gallery for the experimental gravoturb_fdf IC generator (M2).

Scratch verification script (NOT production code, NOT committed). Produces a set of
PNGs so the PI can judge whether the FDF field/star output looks physically realistic
for supersonic turbulent star-forming gas.

Run:
    PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
        python src/experimental/gravoturb_fdf/validation/_m2_ic_gallery.py

Reuses the real JAX-native generator; numpy/matplotlib only on the plotting/measuring side.
"""

import os

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gravoturb_fdf.field.pipeline import build_fdf_field, cloud_to_stars
from gravoturb_fdf.field.sampling import sample_cic_counts
from gravoturb_fdf.theory.bm19 import sigma_s_squared, transition_density
from gravoturb_fdf.theory.pdf import bm19_volume_pdf
from gravoturb_fdf.validation.measure import project_counts_los

# ----------------------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

MACH, B, ALPHA, BETA = 8.0, 0.4, 2.5, 3.0
SHAPE = (96, 96, 96)
N = SHAPE[0]
BOX = 1.0
SEEDS = [0, 1, 2, 3]  # >=4 realizations
N_STARS = 3000
F_SUB = 0.5
N_BAR = 5.0
CELL_SIZE = 3  # 96 -> 32 count cells per axis


def make_field(seed, mach=MACH, b=B, alpha=ALPHA, beta=BETA, shape=SHAPE):
    return build_fdf_field(mach, b, alpha, beta, shape, jax.random.PRNGKey(seed))


# ------------------------------------------------------------------ power spectrum tool
def radial_power_spectrum(field3d, n_bins=24):
    """Isotropic P(k) of a 3D field via FFT periodogram, radially binned in integer |k|."""
    f = np.asarray(field3d, dtype=float)
    f = f - f.mean()
    pk = np.abs(np.fft.fftn(f)) ** 2 / f.size
    n = f.shape[0]
    k1 = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kflat, pflat = kmag.ravel(), pk.ravel()
    keep = (kflat > 0) & (kflat <= n // 2)
    kf, pf = kflat[keep], pflat[keep]
    edges = np.logspace(np.log10(1.0), np.log10(n // 2), n_bins + 1)
    idx = np.clip(np.digitize(kf, edges) - 1, 0, n_bins - 1)
    kc = np.full(n_bins, np.nan)
    pc = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = idx == i
        if m.sum() > 2:
            kc[i] = kf[m].mean()
            pc[i] = pf[m].mean()
    good = ~np.isnan(kc)
    return kc[good], pc[good]


def fit_slope(k, p):
    """Fit log10(P) = a - slope*log10(k); return slope (so P ~ k^-slope)."""
    lk, lp = np.log10(k), np.log10(p)
    A = np.vstack([lk, np.ones_like(lk)]).T
    coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    return -coef[0]  # slope of k^-slope


# ============================================================= build fiducial ensemble
print("=" * 78)
print(f"FDF IC GALLERY  |  M={MACH}, b={B}, alpha={ALPHA}, beta={BETA}, shape={SHAPE}")
print("=" * 78)

fields = []
for sd in SEEDS:
    fields.append(make_field(sd))
s_arrays = [np.asarray(f.s) for f in fields]  # ln(rho/rho0)

sigma2_pred = float(sigma_s_squared(MACH, B))
s_t = float(transition_density(ALPHA, sigma_s_squared(MACH, B)))

# ----------------------------------------------------------- 1. field slices
fig, axes = plt.subplots(1, len(SEEDS), figsize=(4 * len(SEEDS), 4.3))
vmin = min(s.min() for s in s_arrays)
vmax = max(s.max() for s in s_arrays)
for ax, sd, s in zip(axes, SEEDS, s_arrays):
    im = ax.imshow(s[:, :, N // 2], origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(f"seed {sd}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(r"Central slice of $s=\ln(\rho/\rho_0)$  (FDF log-density field)")
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("s")
fig.savefig(os.path.join(PLOTS, "m2_field_slices.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------- 2. column density (Sum exp(s))
fig, axes = plt.subplots(1, len(SEEDS), figsize=(4 * len(SEEDS), 4.3))
cols = [np.exp(s).sum(axis=2) for s in s_arrays]  # projected mass surface density
vmin = min(c.min() for c in cols)
vmax = max(np.percentile(c, 99.5) for c in cols)
for ax, sd, c in zip(axes, SEEDS, cols):
    im = ax.imshow(c, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title(f"seed {sd}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(r"Column density $\Sigma=\sum_z e^{s}$  (projected mass surface density)")
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label(r"$\Sigma$")
fig.savefig(os.path.join(PLOTS, "m2_column_density.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------- 3. PDF vs BM19
fig, ax = plt.subplots(figsize=(7.5, 5.5))
s_lo = min(s.min() for s in s_arrays)
s_hi = max(s.max() for s in s_arrays)
s_grid = np.linspace(s_lo, s_hi, 600)
p_theory = np.asarray(bm19_volume_pdf(jnp.asarray(s_grid), MACH, B, ALPHA))
bins = np.linspace(min(s.min() for s in s_arrays), max(s.max() for s in s_arrays), 80)
centers = 0.5 * (bins[:-1] + bins[1:])
hist_stack = []
for s in s_arrays:
    h, _ = np.histogram(s.ravel(), bins=bins, density=True)
    hist_stack.append(h)
    ax.plot(centers, h, color="steelblue", alpha=0.3, lw=1)
mean_hist = np.mean(hist_stack, axis=0)
ax.plot(centers, mean_hist, color="steelblue", lw=2.5, label="realized (mean of seeds)")
ax.plot(s_grid, p_theory, "k--", lw=2, label="BM19 analytic p(s)")
ax.axvline(s_t, color="red", ls=":", lw=1.5, label=rf"$s_t={s_t:.2f}$")
ax.set_yscale("log")
ax.set_ylim(1e-5, max(p_theory.max(), mean_hist.max()) * 2)
ax.set_xlabel("s = ln(rho/rho_0)")
ax.set_ylabel("p(s)  (volume PDF)")

sigma2_real = np.mean([np.var(s.ravel()) for s in s_arrays])
mean_s_real = np.mean([s.mean() for s in s_arrays])
mean_es_real = np.mean([np.exp(s).mean() for s in s_arrays])
ax.set_title(
    f"1-pt PDF vs BM19   |   sigma_s^2: realized {sigma2_real:.3f} vs pred {sigma2_pred:.3f}\n"
    f"<e^s> realized {mean_es_real:.3f} (target 1.0),  mean(s) {mean_s_real:.3f} "
    f"(target -sigma^2/2 = {-0.5*sigma2_pred:.3f})"
)
ax.legend()
fig.savefig(os.path.join(PLOTS, "m2_pdf_vs_bm19.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------- 4. power spectrum
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fit_lo, fit_hi = 2.0, N // 3
slopes_exp, slopes_s = [], []
last = {}  # remember the last spectrum on each panel for the reference anchor
for sd, s in zip(SEEDS, s_arrays):
    k, p = radial_power_spectrum(np.exp(s))      # P(k) of density exp(s)
    axes[0].loglog(k, p, alpha=0.5, label=f"seed {sd}")
    sel = (k > fit_lo) & (k < fit_hi)
    slopes_exp.append(fit_slope(k[sel], p[sel]))
    last["exp"] = (k, p)

    ks, ps = radial_power_spectrum(s)            # P(k) of log-density s
    axes[1].loglog(ks, ps, alpha=0.5, label=f"seed {sd}")
    sels = (ks > fit_lo) & (ks < fit_hi)
    slopes_s.append(fit_slope(ks[sels], ps[sels]))
    last["s"] = (ks, ps)

kref = np.array([fit_lo, fit_hi])
for ax, key_, title, slopes in (
    (axes[0], "exp", r"$P(k)$ of $e^{s}$ (density)", slopes_exp),
    (axes[1], "s", r"$P(k)$ of $s$ (log-density)", slopes_s),
):
    kk, pp = last[key_]
    # anchor a k^-beta reference line to the first fitted bin of the last spectrum
    i0 = np.argmax(kk > fit_lo)
    norm = pp[i0] / (kk[i0] ** (-BETA))
    ax.loglog(kref, norm * kref ** (-BETA), "k--", lw=1.5,
              label=rf"$k^{{-\beta}}$, $\beta={BETA}$")
    ax.set_xlabel("k")
    ax.set_ylabel("P(k)")
    ax.set_title(f"{title}\nfitted slope = {np.mean(slopes):.2f} +/- {np.std(slopes):.2f}")
    ax.legend()
fig.suptitle(rf"Power spectrum vs input $\beta={BETA}$ (s-field is the direct GRF-marginal test)")
fig.savefig(os.path.join(PLOTS, "m2_power_spectrum.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------- 5. star catalog + count map
fig, axes = plt.subplots(2, len(SEEDS), figsize=(4 * len(SEEDS), 8.5))
for j, (sd, fld, s) in enumerate(zip(SEEDS, fields, s_arrays)):
    key = jax.random.PRNGKey(1000 + sd)
    pos = np.asarray(cloud_to_stars(fld, F_SUB, N_STARS, key, box_size=BOX))
    ax = axes[0, j]
    # underlay the column density faintly
    col = np.exp(s).sum(axis=2)
    ax.imshow(col.T, origin="lower", cmap="Greys", extent=[0, BOX, 0, BOX],
              vmax=np.percentile(col, 99))
    ax.scatter(pos[:, 0], pos[:, 1], s=2, c="red", alpha=0.4)
    ax.set_title(f"stars over Sigma, seed {sd}")
    ax.set_xticks([])
    ax.set_yticks([])
    # count map
    ckey = jax.random.PRNGKey(2000 + sd)
    counts = np.asarray(sample_cic_counts(fld.s, N_BAR, CELL_SIZE, ckey))
    proj = project_counts_los(counts, depth=counts.shape[2], los_axis=2)
    axc = axes[1, j]
    imc = axc.imshow(proj.T, origin="lower", cmap="viridis")
    axc.set_title(f"CIC Poisson counts (LOS sum), seed {sd}")
    axc.set_xticks([])
    axc.set_yticks([])
fig.suptitle(f"Star catalog (f_sub={F_SUB}, N*={N_STARS}) and CIC count map")
fig.savefig(os.path.join(PLOTS, "m2_star_catalog.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------- 6. beta sweep
betas = [2.0, 2.5, 3.0, 3.5]
fig, axes = plt.subplots(1, len(betas), figsize=(4 * len(betas), 4.3))
for ax, bb in zip(axes, betas):
    fld = make_field(7, beta=bb)
    col = np.exp(np.asarray(fld.s)).sum(axis=2)
    ax.imshow(col, origin="lower", cmap="magma", vmax=np.percentile(col, 99.5))
    ax.set_title(rf"$\beta={bb}$")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(r"Column density vs spectral slope $\beta$ (one seed; higher $\beta$ = more large-scale power)")
fig.savefig(os.path.join(PLOTS, "m2_param_sweep_beta.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------- 7. mach sweep
machs = [4.0, 8.0, 16.0]
fig, axes = plt.subplots(2, len(machs), figsize=(4 * len(machs), 8.5))
for j, mm in enumerate(machs):
    fld = make_field(7, mach=mm)
    s = np.asarray(fld.s)
    col = np.exp(s).sum(axis=2)
    axes[0, j].imshow(col, origin="lower", cmap="magma", vmax=np.percentile(col, 99.5))
    axes[0, j].set_title(rf"$\mathcal{{M}}={mm}$  column density")
    axes[0, j].set_xticks([])
    axes[0, j].set_yticks([])
    # PDF overlay
    s2p = float(sigma_s_squared(mm, B))
    sg = np.linspace(s.min(), s.max(), 400)
    pth = np.asarray(bm19_volume_pdf(jnp.asarray(sg), mm, B, ALPHA))
    h, edg = np.histogram(s.ravel(), bins=70, density=True)
    cen = 0.5 * (edg[:-1] + edg[1:])
    axes[1, j].plot(cen, h, color="steelblue", lw=2, label="realized")
    axes[1, j].plot(sg, pth, "k--", lw=1.5, label="BM19")
    axes[1, j].set_yscale("log")
    axes[1, j].set_ylim(1e-5, max(pth.max(), h.max()) * 2)
    axes[1, j].set_title(rf"$\sigma_s^2$ pred {s2p:.2f}, realized {np.var(s):.2f}")
    axes[1, j].set_xlabel("s")
    axes[1, j].legend(fontsize=8)
fig.suptitle(r"Mach sweep ($\beta=3.0$): higher $\mathcal{M}$ = wider $\sigma_s^2$, stronger tail")
fig.savefig(os.path.join(PLOTS, "m2_param_sweep_mach.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

# ============================================================= realism checklist table
print("\nPHYSICAL-REALISM CHECKLIST  (fiducial, per seed)")
print("-" * 78)
hdr = (
    f"{'seed':>4} {'<e^s>':>8} {'sig2_real':>10} {'mean(s)':>9} "
    f"{'Pk_slope':>9} {'fd_real':>9} {'fd_bm19':>9} {'|bias|%':>8} "
    f"{'s_min':>7} {'s_max':>7} {'frac>s_t':>9} {'lowres':>7}"
)
print(hdr)
print(f"{'':>4} {'~1.0':>8} {sigma2_pred:>10.3f} {-0.5*sigma2_pred:>9.3f} "
      f"{'~'+str(BETA):>9} {'(match)':>9}")
print("-" * 78)

rows = []
for sd, fld, s in zip(SEEDS, fields, s_arrays):
    es = float(np.exp(s).mean())
    sig2 = float(np.var(s))
    ms = float(s.mean())
    k, p = radial_power_spectrum(s)  # slope of s-field (true GRF marginal)
    sel = (k > 2) & (k < N // 3)
    pk_slope = fit_slope(k[sel], p[sel])
    fdr = float(fld.f_dense_realized)
    fdb = float(fld.f_dense)
    bias = 100.0 * abs(fdr - fdb) / fdb
    frac_above = float((s > s_t).mean())
    rows.append((sd, es, sig2, ms, pk_slope, fdr, fdb, bias,
                 float(s.min()), float(s.max()), frac_above, bool(fld.low_resolution)))
    print(f"{sd:>4} {es:>8.4f} {sig2:>10.3f} {ms:>9.3f} {pk_slope:>9.2f} "
          f"{fdr:>9.4f} {fdb:>9.4f} {bias:>8.2f} {s.min():>7.2f} {s.max():>7.2f} "
          f"{frac_above:>9.5f} {str(bool(fld.low_resolution)):>7}")

print("-" * 78)
arr = np.array([(r[1], r[2], r[3], r[4], r[7]) for r in rows])
print(f"MEAN over {len(SEEDS)} seeds: <e^s>={arr[:,0].mean():.4f}  "
      f"sig2={arr[:,1].mean():.3f}  mean(s)={arr[:,2].mean():.3f}  "
      f"Pk_slope={arr[:,3].mean():.2f}  |fd bias|%={arr[:,4].mean():.2f}")
print("=" * 78)
print("PNGs written to:", PLOTS)
for fn in sorted(os.listdir(PLOTS)):
    if fn.endswith(".png"):
        print("  ", os.path.join(PLOTS, fn))
