"""Differentiable mass-segregation observables (JAX-native).

Mass segregation is commonly *measured* with non-differentiable estimators
(`compute_lambda_msr`: Allison et al. 2009, SciPy MST). Those break two ways under
autodiff: a hard top-k selection of the "massive" stars (``argsort``) and a
combinatorial spatial statistic (MST). This module provides smooth surrogates so
segregation can enter gradient-based / HMC inference, mirroring the surrogate +
calibration pattern of :mod:`progenax.diagnostics.q_approx` (CW04 Q).

Design: docs/plans/2026-06-09-differentiable-segregation-observable-design.md.

Shared kernel
-------------
Every star carries a smooth weight from a sigmoid soft mass-cut::

    w_i = sigmoid((m_i - m_cut) / tau)

mirroring the observer's choice of a "massive bin" defined by a mass/luminosity cut
``m_cut``. As ``tau -> 0`` this recovers the hard indicator ``1[m_i > m_cut]`` and the
observable reduces to its exact non-differentiable counterpart -- the central
validation route (Oracle 1).

All observables default to ``project_to_2d=True`` (observer-faithful: projected sky
positions), with a 3D flag for theory checks. Inputs are true masses for now; the
noisy mass proxy is a deferred data-realism layer (milestone B).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    "soft_mass_weights",
    "radial_concentration_approx",
    "lambda_msr_approx",
    "sigma_m_approx",
    "calibrate_segregation_approx",
]


def _project(
    positions: Float[Array, "N D"], project_to_2d: bool
) -> Float[Array, "N d"]:
    """Project to the (x, y) sky plane when requested; pass through otherwise."""
    if positions.shape[1] == 2:
        return positions
    if positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 2) or (N, 3), got {positions.shape}")
    return positions[:, :2] if project_to_2d else positions


def soft_mass_weights(
    masses: Float[Array, "N"],
    m_cut: Float[Array, ""],
    tau: Float[Array, ""],
) -> Float[Array, "N"]:
    """Smooth soft mass-cut weights ``w_i = sigmoid((m_i - m_cut) / tau)``.

    The shared weighting kernel for every differentiable segregation observable. As
    ``tau -> 0`` the weights approach the hard indicator ``1[m_i > m_cut]``.

    Args:
        masses: Stellar masses ``(N,)``.
        m_cut: Mass cut defining the "massive" population (same units as ``masses``).
        tau: Softness scale; smaller is sharper. Must be > 0.

    Returns:
        Weights ``(N,)`` in the open interval ``(0, 1)``.
    """
    return jax.nn.sigmoid((masses - m_cut) / tau)


def radial_concentration_approx(
    positions: Float[Array, "N D"],
    masses: Float[Array, "N"],
    *,
    m_cut: Float[Array, ""],
    tau: Float[Array, ""],
    project_to_2d: bool = True,
    calibration: float = 1.0,
) -> Float[Array, ""]:
    """Mass-weighted radial-concentration segregation observable.

    Compares the (soft-mass-weighted) mean cluster-centric radius of the massive
    population to the unweighted mean radius of all stars::

        C = [ sum_i w_i r_i / sum_i w_i ]  /  [ mean_i r_i ]

    where ``r_i = |x_i - xbar_w|`` and ``xbar_w`` is the mass-weighted centroid.

    Interpretation:
        - ``C < 1``: massive stars more centrally concentrated (segregated).
        - ``C ~ 1``: no segregation.
        - ``C > 1``: inverse segregation.

    Smooth in positions and ``m_cut``; no graph, no ranking -- the cleanest-gradient
    member of the family. As ``tau -> 0`` it reduces to the exact mass-cut radial
    ratio.

    Args:
        positions: Positions ``(N, 3)`` or ``(N, 2)``.
        masses: Stellar masses ``(N,)``.
        m_cut: Mass cut for the massive population.
        tau: Soft mass-cut softness (> 0).
        project_to_2d: Use projected (x, y) positions (observer-faithful) if True.
        calibration: Multiplicative calibration factor (fit vs the exact oracle).

    Returns:
        Scalar concentration ``C``.
    """
    xy = _project(positions, project_to_2d)
    w = soft_mass_weights(masses, m_cut, tau)
    W = jnp.sum(w)

    # Mass-weighted centroid (segregation is measured about the massive population's
    # center; translation-equivariant).
    center = jnp.sum(w[:, None] * xy, axis=0) / W
    r = jnp.sqrt(jnp.sum((xy - center) ** 2, axis=1) + 1e-12)

    r_massive = jnp.sum(w * r) / W
    r_all = jnp.mean(r)
    return calibration * r_massive / (r_all + 1e-12)


def _softmin_nn_distance(
    xy: Float[Array, "N d"], beta: Float[Array, ""]
) -> Float[Array, "N"]:
    """Per-star smooth nearest-neighbour distance ``softmin_{j!=i} d_ij``.

    softmin with temperature ``beta``: ``sum_j softmax(-(d_ij/scale)/beta) * d_ij``. As
    ``beta -> 0`` this approaches the true 1-NN distance (the ``min``). Self-pairs are
    masked with a large value so their softmax weight underflows to 0.

    ``beta`` is **scale-relative**: distances are normalised by the median hard 1-NN
    distance before applying the temperature, so ``beta`` is dimensionless and the
    observable is invariant to the cluster's physical size -- essential for inference
    across clusters spanning orders of magnitude in extent. Both dimensionlessness and
    size-invariance are *forward* properties (the scale enters as a ratio with ``dist``),
    so they are unaffected by whether the scale's derivative flows.
    """
    N = xy.shape[0]
    diff = xy[:, None, :] - xy[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)  # (N, N)
    dist = dist + jnp.eye(N) * 1e10  # exclude self
    # Scale-relative temperature: normalise by the median hard 1-NN distance. The median
    # scale depends on the positions, so its derivative is a genuine part of d(softmin)/d(x):
    # a stop_gradient here omitted ~27% of the true gradient at the default beta=0.1
    # (audit finding, Anna-ratified). Let it flow -- the forward value is identical.
    scale = jnp.median(jnp.min(dist, axis=1)) + 1e-12
    s = jax.nn.softmax(-dist / (scale * beta), axis=1)  # (N, N) soft 1-NN selector
    return jnp.sum(s * dist, axis=1)  # (N,)


def lambda_msr_approx(
    positions: Float[Array, "N D"],
    masses: Float[Array, "N"],
    *,
    m_cut: Float[Array, ""],
    tau: Float[Array, ""],
    beta: Float[Array, ""] = 0.1,
    project_to_2d: bool = True,
    calibration: float = 1.0,
) -> Float[Array, ""]:
    """Soft Lambda_MSR segregation observable (MST-ratio surrogate).

    Differentiable analogue of the Allison et al. (2009) mass-segregation ratio
    ``Lambda_MSR = <L_random> / L_massive`` (cf. :func:`compute_lambda_msr`), softening
    both discrete operations:

    - The combinatorial MST length is replaced by a per-star **softmin nearest-neighbour
      distance** (the ``q_approx`` ``L_MST ~ (N-1) <d_1NN>`` estimator, with a smooth
      ``min``).
    - The Monte-Carlo "random subset" baseline is replaced by its **closed-form
      expectation**: the unweighted mean NN-length over all stars. No sampling::

          Lambda_soft = mean_i d_i  /  ( sum_i w_i d_i / sum_i w_i )

      where ``d_i`` is the softmin 1-NN distance and ``w_i`` the soft mass-cut weight.

    Interpretation:
        - ``Lambda > 1``: massive stars more clustered (segregated).
        - ``Lambda ~ 1``: no segregation.
        - ``Lambda < 1``: inverse segregation.

    As ``tau, beta -> 0`` this reduces (up to the multiplicative ``calibration``) to the
    exact Lambda_MSR -- the central validation route (Oracle 1).

    Args:
        positions: Positions ``(N, 3)`` or ``(N, 2)``.
        masses: Stellar masses ``(N,)``.
        m_cut: Mass cut for the massive population.
        tau: Soft mass-cut softness (> 0).
        beta: Softmin temperature for the nearest-neighbour distance (> 0).
        project_to_2d: Use projected (x, y) positions if True (observer-faithful).
        calibration: Multiplicative calibration vs the exact Lambda_MSR oracle.

    Returns:
        Scalar ``Lambda_soft``.
    """
    xy = _project(positions, project_to_2d)
    w = soft_mass_weights(masses, m_cut, tau)
    d = _softmin_nn_distance(xy, beta)

    baseline = jnp.mean(d)  # closed-form "random subset" expectation
    massive = jnp.sum(w * d) / jnp.sum(w)
    return calibration * baseline / (massive + 1e-12)


def _weighted_pearson(x: Float[Array, "N"], y: Float[Array, "N"]) -> Float[Array, ""]:
    """Smooth (unweighted) Pearson correlation coefficient of two vectors."""
    xc = x - jnp.mean(x)
    yc = y - jnp.mean(y)
    num = jnp.sum(xc * yc)
    den = jnp.sqrt(jnp.sum(xc**2) * jnp.sum(yc**2)) + 1e-12
    return num / den


def sigma_m_approx(
    positions: Float[Array, "N D"],
    masses: Float[Array, "N"],
    *,
    m_cut: Float[Array, ""],
    tau: Float[Array, ""],
    k: int = 6,
    project_to_2d: bool = True,
    calibration: float = 1.0,
) -> Float[Array, ""]:
    """Soft Sigma--m segregation observable (Maschberger & Clarke 2011).

    Measures whether massive stars sit in locally denser regions, via the correlation
    between the soft mass-cut weight and the local surface density::

        S = corr_i( w_i , log Sigma_i ),   Sigma_i = (k - 1) / (pi r_ik^2)

    where ``r_ik`` is the distance to the ``k``-th nearest neighbour. The local
    surface-density estimator ``Sigma = (k-1)/(pi r_k^2)`` and the ``k = 6`` choice
    follow von Hoerner (1963) / Casertano & Hut (1985), as adopted by Maschberger &
    Clarke (2011, Eq. 4). The k-NN radius is computed with
    ``jnp.sort`` -- the **exact** k-th order statistic, which (unlike ``argsort``) has a
    well-defined JVP, so the observable is differentiable in positions without any
    softening of the radius.

    Interpretation:
        - ``S > 0``: massive stars in denser regions (segregated).
        - ``S ~ 0``: no mass--density correlation.
        - ``S < 0``: massive stars in sparser regions (inverse).

    Args:
        positions: Positions ``(N, 3)`` or ``(N, 2)``.
        masses: Stellar masses ``(N,)``.
        m_cut: Mass cut for the massive population.
        tau: Soft mass-cut softness (> 0).
        k: Nearest-neighbour rank for the local-density estimator (``k = 6``;
            Casertano & Hut 1985, via Maschberger & Clarke 2011). Must satisfy ``2 <= k < N``.
        project_to_2d: Use projected (x, y) positions if True (observer-faithful;
            surface density is intrinsically a projected quantity).
        calibration: Multiplicative calibration vs the exact Sigma--m oracle.

    Returns:
        Scalar correlation ``S``.
    """
    xy = _project(positions, project_to_2d)
    N = xy.shape[0]
    w = soft_mass_weights(masses, m_cut, tau)

    diff = xy[:, None, :] - xy[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)
    dist = dist + jnp.eye(N) * 1e10  # exclude self before sorting
    dist_sorted = jnp.sort(dist, axis=1)  # ascending; differentiable order statistic
    r_k = dist_sorted[:, k - 1]  # k-th nearest real neighbour (self masked out)

    sigma = (k - 1) / (jnp.pi * r_k**2 + 1e-12)
    log_sigma = jnp.log(sigma + 1e-12)
    return calibration * _weighted_pearson(w, log_sigma)


# --------------------------------------------------------------------------
# Calibration against exact (non-differentiable) oracles.
# --------------------------------------------------------------------------
def _exact_radial_concentration(xy, massive, m_cut):
    """Hard-cut radial concentration (NumPy) -- the tau->0 limit of the surrogate."""
    import numpy as np

    sel = massive
    center = xy[sel].mean(axis=0)
    r = np.sqrt(((xy - center) ** 2).sum(axis=1))
    return float(r[sel].mean() / (r.mean() + 1e-12))


def _exact_sigma_m(xy, massive, k):
    """Exact k-NN Sigma--m correlation (SciPy cKDTree)."""
    import numpy as np
    from scipy.spatial import cKDTree

    tree = cKDTree(xy)
    dists, _ = tree.query(xy, k=k + 1)  # +1 includes self at distance 0
    r_k = dists[:, k]  # k-th nearest real neighbour
    sigma = (k - 1) / (np.pi * r_k**2 + 1e-12)
    log_sigma = np.log(sigma + 1e-12)
    return float(np.corrcoef(massive.astype(float), log_sigma)[0, 1])


def calibrate_segregation_approx(
    n_samples: int = 100,
    N_stars: int = 300,
    n_massive: int = 20,
    m_cut: float = 2.0,
    tau: float = 0.5,
    beta: float = 0.1,
    k: int = 6,
    seed: int = 42,
) -> dict:
    """Calibrate the soft observables against their exact non-differentiable oracles.

    Mirrors :func:`calibrate_q_approx`. Generates clusters spanning a range of
    segregation strengths (massive-star core scale swept from tight to diffuse) and,
    for each soft observable, reports ``mean(exact) / mean(soft)`` (the multiplicative
    calibration) and the Pearson correlation between soft and exact across the sample.

    Exact oracles: :func:`compute_lambda_msr` (SciPy MST), a hard-cut radial
    concentration, and a SciPy cKDTree k-NN Sigma--m correlation.

    Args:
        n_samples: Number of random clusters.
        N_stars: Stars per cluster.
        n_massive: Number of massive stars (placed in a core of varying tightness).
        m_cut, tau, beta, k: Observable hyperparameters.
        seed: Base PRNG seed.

    Returns:
        Dict of calibration factors, correlations, and ``n_samples``.
    """
    import numpy as np

    from progenax.diagnostics.mass_segregation import compute_lambda_msr

    lam_soft, lam_exact = [], []
    rad_soft, rad_exact = [], []
    sig_soft, sig_exact = [], []

    for i in range(n_samples):
        key = jax.random.PRNGKey(seed + i)
        k_halo, k_core, k_scale = jax.random.split(key, 3)
        # Sweep segregation strength: core scale in [0.05, 1.0] (1.0 ~ unsegregated).
        core_scale = 0.05 + 0.95 * jax.random.uniform(k_scale, ())
        halo = jax.random.normal(k_halo, (N_stars - n_massive, 3)) * 1.0
        core = jax.random.normal(k_core, (n_massive, 3)) * core_scale
        positions = jnp.concatenate([halo, core], axis=0)
        masses = jnp.concatenate(
            [jnp.full(N_stars - n_massive, 0.5), jnp.full(n_massive, 10.0)]
        )

        lam_soft.append(
            float(lambda_msr_approx(positions, masses, m_cut=m_cut, tau=tau, beta=beta))
        )
        rad_soft.append(
            float(radial_concentration_approx(positions, masses, m_cut=m_cut, tau=tau))
        )
        sig_soft.append(
            float(sigma_m_approx(positions, masses, m_cut=m_cut, tau=tau, k=k))
        )

        xy = np.asarray(positions[:, :2])
        massive = np.asarray(masses) > m_cut
        lam_exact.append(
            compute_lambda_msr(
                np.asarray(positions), np.asarray(masses), N_massive=int(massive.sum())
            )[0]
        )
        rad_exact.append(_exact_radial_concentration(xy, massive, m_cut))
        sig_exact.append(_exact_sigma_m(xy, massive, k))

    lam_soft, lam_exact = np.array(lam_soft), np.array(lam_exact)
    rad_soft, rad_exact = np.array(rad_soft), np.array(rad_exact)
    sig_soft, sig_exact = np.array(sig_soft), np.array(sig_exact)

    return {
        "calibration_lambda": float(np.mean(lam_exact) / np.mean(lam_soft)),
        "calibration_radial": float(np.mean(rad_exact) / np.mean(rad_soft)),
        "calibration_sigma": float(
            np.mean(np.abs(sig_exact)) / (np.mean(np.abs(sig_soft)) + 1e-12)
        ),
        "correlation_lambda": float(np.corrcoef(lam_exact, lam_soft)[0, 1]),
        "correlation_radial": float(np.corrcoef(rad_exact, rad_soft)[0, 1]),
        "correlation_sigma": float(np.corrcoef(sig_exact, sig_soft)[0, 1]),
        "n_samples": n_samples,
    }
