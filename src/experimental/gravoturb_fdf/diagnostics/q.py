"""CW04 Q substructure parameter — the non-differentiable truth metric.

Cartwright & Whitworth 2004 (MNRAS 348, 589). Q distinguishes centrally-concentrated
clusters (Q > ~0.8) from substructured/fractal ones (Q < ~0.8):

    Q = m_bar / s_bar
    m_bar = L_MST / sqrt(N * A)        # normalized mean MST edge length
    s_bar = <r_pairwise> / R_cluster   # normalized mean separation

where L_MST is the total minimum-spanning-tree edge length, R_cluster the max distance
from the centroid, A = pi * R_cluster^2 the cluster area, and positions are 2D-projected
(CW04 methodology). The sqrt(N) normalization of m_bar is MANDATORY -- omitting it was
the root cause of the discredited Q~0.13 headline.

NOTE on A: CW04 normalize m_bar by the *cluster area* A = pi R_cluster^2 (circle of the
max-distance radius). Using the convex-hull area instead (as the legacy released
diagnostics did) biases Q high by ~+0.08; A = pi R_cluster^2 reproduces CW04 Table 1
(3D0->0.79, 3D1->0.84, 3D2->0.93) to <0.01.

numpy/scipy (analysis side); NON-differentiable. Validated against CW04 anchors (AC5).
"""

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


def compute_q_parameter(positions: np.ndarray) -> float:
    """Cartwright & Whitworth (2004) Q parameter.

    Parameters
    ----------
    positions : (N, 2) or (N, 3) array. 3D inputs are projected to the x-y plane.

    Returns
    -------
    Q = m_bar / s_bar (float). Uniform sphere ~ 0.79; fractal D<3 gives Q < 0.79;
    centrally-concentrated profiles give Q > 0.79.
    """
    positions = np.asarray(positions)
    if positions.shape[1] == 3:
        xy = positions[:, :2]
    elif positions.shape[1] == 2:
        xy = positions
    else:
        raise ValueError("positions must be (N, 2) or (N, 3)")

    N = len(xy)
    if N < 3:
        raise ValueError("Q requires at least 3 points")

    centre = xy.mean(axis=0)
    radii = np.linalg.norm(xy - centre, axis=1)
    R_cluster = radii.max()
    if R_cluster <= 0:
        raise ValueError("degenerate cluster (zero radius)")

    pairwise = pdist(xy)
    s_bar = float(np.mean(pairwise)) / R_cluster

    L_MST = float(minimum_spanning_tree(squareform(pairwise)).sum())
    A = float(np.pi * R_cluster**2)  # CW04 cluster area (reproduces Table 1)
    m_bar = L_MST / np.sqrt(N * A)

    return float(m_bar / s_bar)
