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
    """Minimum-image separation magnitude (grid-cell units) for each lag cell."""
    lags = [np.fft.fftfreq(n, d=1.0 / n) for n in shape]  # signed integer lags
    dx, dy, dz = np.meshgrid(*lags, indexing="ij")
    return np.sqrt(dx**2 + dy**2 + dz**2)


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
