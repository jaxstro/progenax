"""Oracle-measurement utilities for validating the analytic predicted statistics.

numpy/scipy are permitted here (validation path, non-differentiable). These functions
measure 2-point statistics from realization fields and build the theory-consistent
``smooth_copula_field``: the exact pointwise map ``s = bm19_icdf(Phi(g_hat)) - log<e^s>``
on an EXACTLY unit-variance Gaussian ``g_hat``. That is the map the Gaussianization
series assumes, so comparing the series to this oracle isolates the series-truncation
error from the empirical-CDF / finite-grid non-Gaussianity of the rank-copula simulator.
"""

import jax.numpy as jnp
import numpy as np

from gravoturb_fdf.theory.gaussianization import s_of_g
from gravoturb_fdf.theory.projection import _kmag_grid


def smooth_copula_field(g, mach, b, alpha):
    """Theory-consistent log-density field ``s = T(g_hat)``, ``g_hat = (g-<g>)/std(g)``.

    Normalizing ``g`` to exactly unit variance makes ``Phi(g_hat)`` exactly uniform, so
    ``s`` carries the BM19 marginal by construction (the pointwise map the Gaussianization
    series assumes). Returns a numpy array.
    """
    g = np.asarray(g)
    g_hat = (g - g.mean()) / g.std()
    return np.asarray(s_of_g(jnp.asarray(g_hat), mach, b, alpha))


def autocovariance_3d(field):
    """Periodic autocovariance ``xi(r) = <f(x) f(x+r)> - <f>^2`` via FFT (Wiener-Khinchin).

    Returns an array of the same shape as ``field`` with ``xi[0,0,0] = Var(field)``.
    """
    f = np.asarray(field, dtype=float)
    f = f - f.mean()
    axes = tuple(range(f.ndim))
    ft = np.fft.rfftn(f, axes=axes)
    xi = np.fft.irfftn(np.abs(ft) ** 2, s=f.shape, axes=axes) / f.size
    return xi


def _separation_radius(shape):
    """Minimum-image separation magnitude (grid-cell units) for each lag cell (any ndim)."""
    lags = [np.fft.fftfreq(n, d=1.0 / n) for n in shape]  # signed integer lags
    grids = np.meshgrid(*lags, indexing="ij")
    return np.sqrt(sum(g**2 for g in grids))


def radial_average(values3d, n_bins=24, r_max=None, exclude_zero=True):
    """Radially (minimum-image) bin a 3D grid quantity. Returns ``(r_centers, binned)``.

    ``r`` in grid-cell units. With ``exclude_zero`` the r=0 self-cell is dropped (so a
    binned autocovariance's first bin is genuine small-r correlation, not the variance).
    The binning geometry is fixed (independent of the values), so this is a linear average
    -- safe to apply to analytic predictions and measurements alike for apples-to-apples.
    """
    v = np.asarray(values3d)
    r = _separation_radius(v.shape)
    if r_max is None:
        r_max = min(v.shape) // 2
    rflat, vflat = r.ravel(), v.ravel()
    keep = rflat <= r_max
    if exclude_zero:
        keep = keep & (rflat > 0.0)
    rk, vk = rflat[keep], vflat[keep]
    edges = np.linspace(rk.min(), r_max, n_bins + 1)
    idx = np.clip(np.digitize(rk, edges) - 1, 0, n_bins - 1)
    r_centers = np.full(n_bins, np.nan)
    binned = np.zeros(n_bins)
    for i in range(n_bins):
        m = idx == i
        if m.any():
            r_centers[i] = rk[m].mean()
            binned[i] = vk[m].mean()
    valid = ~np.isnan(r_centers)
    return r_centers[valid], binned[valid]


def measured_2pt(field, n_bins=24, r_max=None):
    """Binned periodic 2-point ``xi(r)`` for r>0 plus the zero-lag variance.

    Returns ``(r_centers, xi_r, variance)``; ``r`` in grid-cell units (minimum image).
    """
    xi3d = autocovariance_3d(field)
    variance = float(xi3d[0, 0, 0])
    r, xi_r = radial_average(xi3d, n_bins=n_bins, r_max=r_max, exclude_zero=True)
    return r, xi_r, variance


def gaussian_correlation_measured(g, n_bins=24, r_max=None):
    """Normalized Gaussian correlation ``rho_g(r) = xi_g(r) / Var(g)`` for r>0."""
    r, xi, var = measured_2pt(np.asarray(g), n_bins, r_max)
    return r, xi / var


def field_2pt_measured(s, n_bins=24, r_max=None):
    """Measured 2-point ``xi_s(r)`` of a log-density field ``s`` (r>0)."""
    r, xi, _var = measured_2pt(np.asarray(s), n_bins, r_max)
    return r, xi


def measure_exceedances(s_field, s_thr, n_bins=20):
    r"""Reduce a gas log-density field to the threshold-exceedance histogram for the POT alpha block.

    Returns ``(exc_counts, exc_edges, s_max, n_tail)``: counts of cells with ``s > s_thr`` binned
    into ``n_bins`` equal s-bins spanning ``[s_thr, s_max]`` (``s_max`` = the realized field maximum,
    i.e. the finite-field truncation ceiling, so the top bin is CLOSED at ``s_max``); the s-space bin
    edges; the realized maximum; and the number of exceedances ``n_tail``. The output feeds
    :func:`gravoturb_fdf.inference.likelihood.tail_exceedance_loglike`. ``s_thr`` and ``s_max`` are
    measured here on the SAME field as the counts, which is what makes the POT block shift-immune.
    numpy path (non-differentiable, validation/oracle only).
    """
    s = np.asarray(s_field).ravel()
    exc = s[s > s_thr]
    s_max = float(s.max())
    exc_edges = np.linspace(s_thr, s_max, n_bins + 1)
    exc_counts, _ = np.histogram(exc, bins=exc_edges)  # last bin closed at s_max
    return exc_counts.astype(float), exc_edges, s_max, int(exc.size)


def project_counts_los(counts3d, depth, los_axis=2):
    r"""Project a 3D count grid to a 2D map by summing the FIRST ``depth`` slices along ``los_axis``.

    A non-periodic line-of-sight slab: ``sum_{i<depth} N[..., i, ...]`` (the integer ``depth`` is a
    number of slices, not a fraction). Returns a 2D numpy array (the data-side projected map fed to
    :func:`measure_angular_bandpowers_2d` and :func:`measure_log_count_variance`).
    """
    c = np.asarray(counts3d)
    return c.take(range(depth), axis=los_axis).sum(axis=los_axis)


def measure_angular_bandpowers_2d(map2d, k_edges):
    r"""Measured 2D periodogram band-powers ``<|fft2(f-<f>)|^2 / N>`` of a projected map ``map2d``,
    binned by 2D ``|k|`` into ``k_edges`` (numpy; the angular analog of
    :func:`gravoturb_fdf.inference.covariance.measured_bandpowers`).

    Uses the EXACT same periodogram normalization (``/ f.size``) and the same ``|k|`` convention
    (``kx = fftfreq(n)*n`` per axis) so a future predicted ``angular_bandpowers_2d`` matches.
    """
    f = np.asarray(map2d, dtype=float)
    f = f - f.mean()
    pk = np.abs(np.fft.fft2(f)) ** 2 / f.size
    ny, nx = f.shape
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
    kmag = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    out = np.zeros(len(k_edges) - 1)
    ke = np.asarray(k_edges)
    for i, (lo, hi) in enumerate(zip(ke[:-1], ke[1:])):
        mask = (kmag >= lo) & (kmag < hi)
        out[i] = pk[mask].mean()
    return out


def measure_log_count_variance(counts, n_bar):
    r"""Measured CIC log-count variance ``Var_cells[log_plus(N_cell)]`` (Neyrinck+2011 Eq 2).

    The data-side counterpart of :func:`gravoturb_fdf.theory.cic.predict_log_count_variance`;
    uses the identical ``log_plus`` transform so the statistic is consistent in generation and
    inference (SBC-valid). ``counts`` is an integer count grid; ``n_bar`` the mean count per cell.
    """
    counts = np.asarray(counts, dtype=float)
    d = counts / n_bar - 1.0
    A = np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)
    return float(np.var(A))


def estimate_log_count_variance_var(mach, b, alpha, beta, shape, cell_size, n_bar, n_real, key):
    r"""Fixed (fiducial) variance of the ``measure_log_count_variance`` estimator from an ``n_real``
    mock ensemble at theta_fid. Used as ``var_v`` in :func:`log_count_variance_loglike` (mock-precision
    pattern; computed ONCE per inference, not per NUTS step -> SBC-valid as a fixed constant)."""
    import jax

    from gravoturb_fdf.field.field import gaussian_random_field, rank_copula_field
    from gravoturb_fdf.field.sampling import sample_cic_counts

    vals = []
    for r in range(n_real):
        k = jax.random.fold_in(key, r)
        s = rank_copula_field(gaussian_random_field(shape, beta, k), mach, b, alpha)
        cnt = np.asarray(sample_cic_counts(s, n_bar, cell_size, jax.random.fold_in(k, 1)))
        vals.append(measure_log_count_variance(cnt, n_bar))
    return float(np.var(vals, ddof=1))


def smoothed_linear_variance(rho_tilde, R, window_fn):
    r"""Variance of the linear field ``rho_tilde`` after smoothing at scale ``R`` (cells):
    ``(1/N^2) sum_{k!=0} |FFT(rho_tilde - mean)|^2 W(kR)^2`` for ONE realization.

    This is the oracle for the CIC clustering term ``xi_bar_rho(R) = Var(rho_tilde_cell)``:
    smoothing the realized linear density with the SAME window the prediction uses, then
    taking its variance. The k=0 mode is dropped (empirical-mean subtracted) so it measures
    cell-to-cell fluctuations, matching :func:`cell_averaged_xi_rho` which also excludes DC.
    Average over realizations to beat down the tail-driven scatter.
    """
    f = np.asarray(rho_tilde, dtype=float)
    f = f - f.mean()
    pk = np.abs(np.fft.fftn(f)) ** 2
    kmag = np.asarray(_kmag_grid(f.shape))
    w2 = np.asarray(window_fn(jnp.asarray(kmag * R))) ** 2
    w2 = np.where(kmag > 0, w2, 0.0)  # exclude DC (empirical mean already removed)
    return float(np.sum(pk * w2) / f.size ** 2)
