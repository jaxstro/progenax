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


def q_components(positions: np.ndarray) -> tuple[float, float, float]:
    """Cartwright & Whitworth (2004) Q with its components: returns ``(Q, m_bar, s_bar)``.

    Reporting the components separates the two effects Q conflates, via the ``(m_bar, s_bar)`` PLANE
    (not either alone): ``s_bar`` (normalized mean separation) is the CONCENTRATION axis (centrally
    concentrated -> small s_bar; uniform & clumpy both ~0.8). ``m_bar`` (normalized MST length) is
    lowered by BOTH concentration and clumpiness, so at FIXED s_bar it isolates SUBSTRUCTURE (uniform
    high m_bar vs clumpy low m_bar at the same s_bar). The three regimes occupy distinct regions:
    uniform (high m_bar, high s_bar), concentrated (low m_bar, LOW s_bar), clumpy (low m_bar, HIGH
    s_bar). Measured (N=500): uniform (0.64, 0.79); concentrated (0.20, 0.15); clumpy (0.25, 0.82).
    NB: m_bar's R_cluster normalization is outlier/tail-sensitive, so m_bar alone is NOT a clean
    substructure measure -- use the plane (or compare at matched s_bar).

    Parameters
    ----------
    positions : (N, 2) or (N, 3) array. 3D inputs are projected to the x-y plane (CW04 methodology).
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

    return float(m_bar / s_bar), float(m_bar), float(s_bar)


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
    return q_components(positions)[0]
