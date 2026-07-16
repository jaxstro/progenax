r"""Mass-weighted substructure diagnostic via local stellar surface density (Tier C.2).

Grounded in the held Maschberger & Clarke (2011, MNRAS 416, 541) PDF, §4 / Eq. 4. For each star
the local stellar surface *number* density is the Casertano & Hut (1985) k-th-nearest-neighbour
estimator

    Σ_i = (k − 1) / (π r_{k,i}²),     r_{k,i} = distance to the k-th nearest neighbour  (k = 6),

evaluated in projection (M&C analyse the m–Σ plane in 2-D). Mass segregation / primordial
mass–density correlation is read off the **m–Σ plane**: massive stars sit at systematically higher
Σ. We summarise this with the Spearman rank correlation ρ(m, Σ), the high-vs-low median-Σ ratio, and
the two-sample KS p-value M&C use. **Robust to substructure** (a local density, not a global MST/Q),
so it works in clumpy turbulent-density fields where CW04 Q on small subsets is too noisy.

Non-differentiable (kNN ranking), a diagnostic — numpy/scipy on the analysis side (cf. ``q.py``).
"""

import numpy as np
from scipy.stats import ks_2samp, spearmanr


def local_surface_density(xy, k: int = 6):
    r"""Casertano & Hut (1985) local surface density Σ = (k−1)/(π r_k²) per point (M&C Eq. 4).

    ``xy``: projected positions, shape (n, 2). ``r_k`` is the distance to the k-th nearest
    neighbour (excluding self). Returns Σ (n,) in points per unit area. Requires n > k.
    """
    p = np.asarray(xy, dtype=float)
    n = p.shape[0]
    if n <= k:
        raise ValueError(f"need n > k for the k-th neighbour (n={n}, k={k})")
    d2 = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    r_k2 = np.sort(d2, axis=1)[:, k - 1]          # squared distance to the k-th nearest neighbour
    return (k - 1) / (np.pi * r_k2)


def mass_density_segregation(positions, masses, k: int = 6) -> dict:
    r"""Mass–local-density correlation (M&C 2011 m–Σ diagnostic), robust to substructure.

    Projects ``positions`` (n, 3) to the x–y plane, computes Σ per star (:func:`local_surface_density`),
    and returns:
      - ``rho_m_sigma``      — Spearman ρ(mass, Σ): >0 ⇒ massive stars in denser regions (segregated),
      - ``median_sigma_ratio`` — median Σ of above-median-mass stars / below-median-mass stars,
      - ``ks_p``             — two-sample KS p-value between the high- and low-mass Σ distributions,
      - ``sigma``            — the per-star Σ array.
    """
    pos = np.asarray(positions, dtype=float)
    m = np.asarray(masses, dtype=float)
    sigma = local_surface_density(pos[:, :2], k=k)
    rho, _ = spearmanr(m, sigma)
    med_m = np.median(m)
    high, low = sigma[m > med_m], sigma[m <= med_m]
    ratio = float(np.median(high) / np.median(low))
    ks_p = float(ks_2samp(high, low).pvalue)
    return {"rho_m_sigma": float(rho), "median_sigma_ratio": ratio, "ks_p": ks_p, "sigma": sigma}
