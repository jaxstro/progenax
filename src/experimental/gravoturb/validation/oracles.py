"""INDEPENDENT numpy reference implementations (oracles) for placement validation.

These are deliberately numpy-only re-derivations used to cross-check the JAX
sampling code: no gravoturb realization code is imported here beyond the single
placement-law constant ``FREEFALL_EXPONENT`` (the p in ρ^p — imported so the
oracle and the production PMF share one source of truth for the exponent, while
the *implementation* stays independent).

Callers: tests/experimental/unit/test_multi_freefall.py (envelope control) and
gravoturb.validation.cluster_acceptance (AC-IC7 part a).
"""

import numpy as np

from gravoturb.realization.placement import FREEFALL_EXPONENT  # noqa: F401 (re-export)


def rho_weighted_reference_positions(profile, shape, box_size, p_exp, n, rng):
    """Reference draw of ``n`` positions ∝ profile.density(r)**p_exp on a cell grid.

    Cell-centre grid → density**p_exp cell weights → ``rng.choice`` over cells →
    unravel + uniform in-cell jitter → CENTRED positions (box centred on 0).
    Pass ``p_exp=FREEFALL_EXPONENT`` to oracle the multi-freefall (ρ^{3/2}) law,
    ``p_exp=1.0`` for plain ∝ρ placement.
    """
    n_grid = shape[0]
    ax = (np.arange(n_grid) + 0.5) / n_grid * box_size - box_size / 2
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    r_cell = np.sqrt(X**2 + Y**2 + Z**2).ravel()
    w = np.asarray(profile.density(r_cell), dtype=float) ** p_exp
    idx = rng.choice(w.size, size=n, p=w / w.sum())
    ijk = np.stack(np.unravel_index(idx, shape), axis=-1)
    return (ijk + rng.uniform(size=ijk.shape)) * (box_size / n_grid) - box_size / 2


def ks_two_sample(a, b):
    """Two-sample Kolmogorov–Smirnov statistic max|CDF_a − CDF_b| (searchsorted)."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / a.size
    cdf_b = np.searchsorted(b, allv, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))
