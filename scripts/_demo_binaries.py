r"""Reusable forward-model pieces for the B12 binary-inflated dynamical-mass demo.

A scripts-local (NOT a packaged API) helper module for
``scripts/demo_binary_dynamical_mass.py``. It provides the three reusable parts
of the demo's forward model:

* :func:`project_los_velocity` -- isotropic line-of-sight projection of a 3-velocity;
* ``build_korb_kernel`` -- the sigma-independent flux-weighted binary blend kernel
  ``K_orb`` (Moe & Di Stefano P-q-e orbits + Tout+1996 ZAMS luminosity weighting),
  built in ``BINARY`` units (Msun, AU, yr) and returned in km/s;
* ``predict_vlos_counts`` -- the differentiable binned single+binary mixture model.

JAX-native (``jax.numpy``); the kernel/predict pieces are jit/grad-safe so the
demo can differentiate the mixture model in ``(sigma, f_b)``. float64 is the
demo's responsibility (``import progenax`` enables it before this module is used).
"""
import numpy as np
import jax
import jax.numpy as jnp

from jaxstro.units import BINARY
from progenax.binaries import period_to_semimajor_axis
from progenax.binaries.kepler import KeplerElements
from progenax.imf import Maschberger
from progenax.imf.binary import MoeJointOrbit
from progenax.stellar import zams_luminosity

# Kernel works in BINARY units (Msun, AU, yr): periods (days) -> years, Kepler-III
# gives a in AU, to_binary_state returns AU/yr, converted ONCE to km/s. This is the
# verified-clean path for day-scale binary periods (vs STELLAR's awkward ~1e-5 pc).
_G_BIN = BINARY.G
_VEL_KMS = BINARY.velocity_scale_km_s   # AU/yr -> km/s (~4.74); NOT `velocity_scale`
_DAY_YR = 1.0 / 365.25                   # day -> year


def project_los_velocity(vel3, los_hat):
    r"""Line-of-sight component of a 3-velocity along a unit direction.

    ``v_los = vel3 . los_hat``. For ``los_hat`` drawn isotropically, the
    projection of a fixed velocity has zero mean and variance ``|vel3|^2 / 3``
    (the velocity's energy shared equally over three orthogonal axes).

    Parameters
    ----------
    vel3 : (3,) array
        Velocity vector [any consistent units; km/s in the B12 demo].
    los_hat : (3,) array
        Line-of-sight unit vector (caller normalizes).

    Returns
    -------
    v_los : scalar, the LOS velocity component (same units as ``vel3``).
    """
    return jnp.dot(vel3, los_hat)


def _component_los_velocities(a, e, i, Omega, omega, M0, m1, m2, los_hat):
    """km/s LOS velocities of both binary components for one system.

    Builds the Keplerian state in BINARY units (AU/yr) via ``to_binary_state``
    (the exact barycentric split ``m1 v1 + m2 v2 = 0``) and projects each
    component velocity onto ``los_hat``, converting AU/yr -> km/s.
    """
    el = KeplerElements(a=a, e=e, i=i, Omega=Omega, omega=omega, M0=M0)
    bs = el.to_binary_state(m1, m2, G=_G_BIN)
    v1_los = jnp.dot(bs.v1, los_hat) * _VEL_KMS
    v2_los = jnp.dot(bs.v2, los_hat) * _VEL_KMS
    return v1_los, v2_los


def build_korb_kernel(
    n_pool=20000,
    q_fixed=None,
    Z=1e-3,
    seed=0,
    grid_max=150.0,
    n_grid=601,
    imf=None,
    joint=None,
):
    r"""The sigma-independent flux-weighted binary blend kernel ``K_orb``.

    For a large Moe & Di Stefano (2017) pool of binaries, the unresolved
    flux-weighted centroid sits at ``v_obs = v_COM + Delta`` with the *internal*
    blend velocity

        ``Delta = (L1 v1_los + L2 v2_los) / (L1 + L2)``,

    where ``v1_los, v2_los`` are the barycentric component LOS velocities and
    ``L = zams_luminosity(m, Z)`` (Tout+1996). ``Delta`` depends only on the
    orbit (P, q, e), masses, phase and orientation -- NOT on the cluster
    dispersion ``sigma_true`` -- so ``K_orb`` is precomputed ONCE and the observed
    distribution is the convolution ``N(0, sigma^2) (*) K_orb``.

    High-q binaries self-cancel (``L1=L2``, ``v1=-v2`` -> ``Delta -> 0``); low-q
    are primary-reflex dominated (``Delta -> v1_los``). The orbital phase is a
    uniform mean anomaly ``M0 ~ U(0, 2pi)`` (uniform-in-time, correctly
    over-weighting slow apocenter); isotropy comes from a random LOS per system.

    Parameters
    ----------
    n_pool : int
        Number of binaries in the template pool.
    q_fixed : float or None
        If given, override the Moe mass ratio with this fixed ``q`` (test knob);
        otherwise draw the coupled ``(P, q, e)`` from ``MoeJointOrbit``.
    Z : float
        Metallicity for the ZAMS luminosity weighting.
    seed : int
        PRNG seed.
    grid_max : float
        Half-width of the symmetric velocity grid [km/s]; must span the wings.
    n_grid : int
        Number of grid points (bin centers) returned.
    imf, joint : optional
        Override the default ``Maschberger`` IMF / ``MoeJointOrbit`` sampler.

    Returns
    -------
    v_grid : (n_grid,) float ndarray
        Velocity-grid bin centers [km/s].
    density : (n_grid,) float ndarray
        Normalized kernel density (``sum(density) * dv ~ 1``).
    """
    imf = imf if imf is not None else Maschberger(alpha=2.3, m_min=0.08, m_max=100.0)
    joint = joint if joint is not None else MoeJointOrbit.default()

    key = jax.random.PRNGKey(seed)
    k_imf, k_joint, k_phase, k_los = jax.random.split(key, 4)

    m1 = imf.sample(k_imf, n_pool)
    P_days, q, e = joint.sample(k_joint, m1)
    if q_fixed is not None:
        q = jnp.full_like(q, q_fixed)
    m2 = q * m1
    M_total = m1 + m2

    a = period_to_semimajor_axis(P_days * _DAY_YR, M_total, G=_G_BIN)  # AU

    # Uniform mean anomaly = uniform-in-time orbital phase. Orbit angles are left
    # at zero (orbit in the xy-plane); the per-system random isotropic LOS then
    # samples all viewing geometries -- equivalent to randomizing (i, Omega, omega)
    # by rotational symmetry, but cheaper.
    M0 = jax.random.uniform(k_phase, (n_pool,), maxval=2.0 * jnp.pi)
    los = jax.random.normal(k_los, (n_pool, 3))
    los = los / jnp.linalg.norm(los, axis=1, keepdims=True)
    zeros = jnp.zeros(n_pool)

    v1_los, v2_los = jax.vmap(_component_los_velocities)(
        a, e, zeros, zeros, zeros, M0, m1, m2, los
    )

    L1 = zams_luminosity(m1, Z)
    L2 = zams_luminosity(m2, Z)
    delta = np.asarray((L1 * v1_los + L2 * v2_los) / (L1 + L2))  # km/s

    edges = np.linspace(-grid_max, grid_max, n_grid + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dv = centers[1] - centers[0]
    counts, _ = np.histogram(delta, bins=edges)
    density = counts / (n_pool * dv)
    return centers, density


def predict_vlos_counts(sigma, f_b, N, v_edges, korb_grid, korb, eps):
    r"""Differentiable binned LOS-velocity counts for the single+binary mixture.

    The observed density is

        ``p(v) = (1 - f_b) N(v; 0, sigma^2 + eps^2)
                  + f_b [ N(.; 0, sigma^2 + eps^2) (*) K_orb ](v)``,

    i.e. singles are a Gaussian broadened by the RV precision ``eps``, and
    binaries are that same Gaussian convolved with the sigma-independent blend
    kernel ``K_orb``. The expected counts are ``mu_k = N * integral_{bin k} p``.

    Differentiability: the SINGLE term is integrated analytically per bin via the
    Gaussian CDF (``erf``) -- exact and differentiable in ``sigma`` -- so the
    total count is conserved to ~1e-9 (a grid sum would carry ~1e-3 trapezoid
    error). The BINARY term is convolved on ``korb_grid`` (``jnp.convolve``,
    differentiable in ``sigma``) and integrated over each bin with a STATIC
    membership mask (gradient flows through the density, not the mask). The
    mixture weight ``f_b`` enters linearly.

    Parameters
    ----------
    sigma : scalar
        Cluster LOS velocity dispersion [km/s] (the parameter of interest).
    f_b : scalar
        Unresolved binary fraction in (0, 1).
    N : scalar
        Total number of stars (sets the Poisson normalization).
    v_edges : (K+1,) array
        Monotone bin edges [km/s] for the observed histogram.
    korb_grid : (G,) array
        Velocity-grid bin centers of ``K_orb`` [km/s] (from ``build_korb_kernel``).
    korb : (G,) array
        ``K_orb`` density on ``korb_grid``.
    eps : scalar
        Per-star RV measurement precision [km/s] (Gaussian, added in quadrature).

    Returns
    -------
    mu : (K,) array of expected counts per bin, differentiable in (sigma, f_b).
    """
    from jax.scipy.stats import norm

    v_edges = jnp.asarray(v_edges)
    korb_grid = jnp.asarray(korb_grid)
    korb = jnp.asarray(korb)
    sig_obs = jnp.sqrt(sigma ** 2 + eps ** 2)

    # Single component: exact analytic CDF differences per bin.
    cdf_edges = norm.cdf(v_edges, loc=0.0, scale=sig_obs)
    phi_single = cdf_edges[1:] - cdf_edges[:-1]                      # (K,)

    # Binary component: (Gaussian (*) K_orb) integrated per bin on korb_grid.
    dv = korb_grid[1] - korb_grid[0]
    g = norm.pdf(korb_grid, loc=0.0, scale=sig_obs)                  # (G,) density
    b_density = jnp.convolve(g, korb, mode="same") * dv             # (G,) density
    # Static membership mask: grid point in bin k iff edges[k] <= v < edges[k+1].
    in_bin = (korb_grid[None, :] >= v_edges[:-1, None]) & (
        korb_grid[None, :] < v_edges[1:, None]
    )                                                                # (K, G) bool
    phi_binary = jnp.sum(in_bin * b_density[None, :], axis=1) * dv  # (K,)

    phi = (1.0 - f_b) * phi_single + f_b * phi_binary
    return N * phi


def dyn_mass_ratio(sigma_obs, sigma_true):
    r"""Naive virial dynamical-mass bias factor ``(sigma_obs / sigma_true)^2``.

    The virial / dynamical mass scales as ``M ~ sigma^2 r_h / G`` at fixed ``r_h``,
    so a dispersion inflated by unresolved binaries (``sigma_obs > sigma_true``)
    biases the inferred mass high by exactly ``(sigma_obs / sigma_true)^2``. Equals
    1 when the measured dispersion is unbiased.
    """
    return (sigma_obs / sigma_true) ** 2


def _kernel_std(v_grid, k):
    """Standard deviation of a (grid, density) kernel via discrete moments."""
    v_grid = np.asarray(v_grid)
    k = np.asarray(k)
    dv = v_grid[1] - v_grid[0]
    mean = np.sum(k * v_grid) * dv
    var = np.sum(k * (v_grid - mean) ** 2) * dv
    return np.sqrt(max(var, 0.0))
