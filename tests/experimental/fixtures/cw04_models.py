"""CW04 analytic radial cluster models — TEST FIXTURE ONLY.

Generates spherical clusters with volume density n(r) ~ r^{-alpha} (Cartwright &
Whitworth 2004, type "3D-alpha", their Eq. 2), used to validate the CW04 Q estimator
against the published Q anchors WITHOUT a (non-differentiable, finicky) box-fractal:

    CW04 Table 1 (3D, projected to 2D, 100 <= N <= 300, 100 realizations):
      3D0 (alpha=0, uniform sphere): Q = 0.79 +- 0.02
      3D1 (alpha=1, n ~ r^-1):       Q = 0.84 +- 0.03
      3D2 (alpha=2, n ~ r^-2):       Q = 0.93 +- 0.03
      3D2.9 (alpha=2.9):             Q = 1.50 +- 0.13

Inverse-CDF sampling: for n ~ r^{-alpha} the enclosed mass ~ r^{3-alpha}, so
r = u^{1/(3-alpha)} (unit sphere; the constant scale is irrelevant to the
scale-invariant Q). Angles isotropic. NOT core code -- the packages ship no cluster
generator. 3D points are projected to 2D by the Q estimator.
"""

import numpy as np


def radial_profile_positions(alpha, n_points, seed):
    """Return (n_points, 3) points with volume density n(r) ~ r^{-alpha} in a sphere.

    alpha=0 -> uniform sphere (CW04 3D0). Requires alpha < 3 for normalizability.
    """
    if alpha >= 3.0:
        raise ValueError("alpha must be < 3 for a normalizable n ~ r^-alpha sphere")
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=n_points)
    r = u ** (1.0 / (3.0 - alpha))  # inverse CDF of M(r) ~ r^{3-alpha}
    cos_theta = 2.0 * rng.uniform(0.0, 1.0, size=n_points) - 1.0
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, 1.0))
    phi = 2.0 * np.pi * rng.uniform(0.0, 1.0, size=n_points)
    return np.column_stack(
        [
            r * sin_theta * np.cos(phi),
            r * sin_theta * np.sin(phi),
            r * cos_theta,
        ]
    )


def uniform_sphere_positions(n_points, seed):
    """Return (n_points, 3) points uniform in the unit sphere (CW04 3D0; alpha=0)."""
    return radial_profile_positions(0.0, n_points, seed)
