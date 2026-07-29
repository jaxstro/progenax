r"""Gas-cloud density profiles for the gravoturbulent natal envelope (ADR-0062).

gravoturb models *natal gas* -- the material an RMHD simulation evolves -- so its
envelope should be a gas equilibrium, not the stellar-dynamics ``PlummerProfile`` it
inherited. This subpackage supplies the two canonical star-formation gas profiles:

- :class:`BonnorEbertProfile` -- pressure-confined isothermal sphere.
- :class:`PolytropeProfile` -- ``P = K rho^gamma``, self-truncating at ``xi_1``.

Both are built on the shared differentiable :mod:`lane_emden` solver core, and both
satisfy the duck-type the gravoturb chain needs (``.density(r)`` for the envelope layer,
``.r_h`` for the magnetic chain), so they drop in without touching ``cluster.py``.

Built here rather than in ``progenax.profiles`` while the physics is being proven; the
hoist is planned and these modules stay dependency-clean for it (ADR-0068).
"""

from gravoturb.profiles.bonnor_ebert import (
    BonnorEbertProfile,
    CriticalSphere,
    critical_sphere,
)
from gravoturb.profiles.polytrope import GAMMA_MIN, PolytropeProfile

# The dimensionless Lane-Emden solvers (solve_isothermal, solve_polytrope,
# polytrope_xi1, LaneEmdenSolution) now live in jaxstro.numerics.lane_emden; import
# them from there rather than through this package.

__all__ = [
    "GAMMA_MIN",
    "BonnorEbertProfile",
    "CriticalSphere",
    "PolytropeProfile",
    "critical_sphere",
]
