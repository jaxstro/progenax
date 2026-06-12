"""Tidal physics utilities for star cluster ICs.

Computes tidal/Jacobi radii and applies tidal truncation.

References:
    King (1962) AJ 67, 471 - Tidal radius definition
    Binney & Tremaine (2008) "Galactic Dynamics" Section 8.3.1
    Baumgardt & Makino (2003) MNRAS 340, 227 - Tidal stripping
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def jacobi_radius(
    M_cluster: float,
    M_galaxy: float,
    R_galactic: float,
) -> float:
    """Compute Jacobi (tidal) radius for a cluster in a galaxy.

    The Jacobi radius is where the cluster's gravitational pull
    equals the tidal force from the galaxy (point mass approximation):

        r_J = R * (M_cluster / (3 * M_galaxy))^(1/3)

    Beyond r_J, stars become unbound from the cluster.

    Args:
        M_cluster: Cluster mass [Msun]
        M_galaxy: Galaxy mass (enclosed within R) [Msun]
        R_galactic: Distance from galactic center [pc]

    Returns:
        Jacobi radius [pc]

    Note:
        This assumes a point-mass galaxy and circular orbit.
        For more realistic models, use galactic potential functions.

    Reference:
        King (1962) AJ 67, 471
        Binney & Tremaine (2008) Eq. 8.91
    """
    r_J = R_galactic * (M_cluster / (3.0 * M_galaxy)) ** (1.0/3.0)
    return r_J


def jacobi_radius_isothermal(
    M_cluster: float,
    V_circ: float,
    R_galactic: float,
    G: float,
) -> float:
    """Compute Jacobi radius for isothermal halo.

    For a singular isothermal sphere with circular velocity V_circ:

        r_J = (G * M_cluster / (2 * Omega^2))^(1/3)

    where Omega = V_circ / R is the angular velocity.

    Units (IMPORTANT): V_circ must be in the SAME length/time units as G, i.e.
    consistent with the rest of the unit system — pc/Myr for STELLAR, NOT km/s.
    The ecosystem's display convention quotes velocities in km/s, but G is in
    pc^3 Msun^-1 Myr^-2, so pass V_circ in pc/Myr (1 km/s = 1.0227 pc/Myr);
    mixing the two biases r_J by ~1.5% per the km/s->pc/Myr factor.

    Args:
        M_cluster: Cluster mass [Msun]
        V_circ: Circular velocity [pc/Myr for STELLAR — same units as G, not km/s]
        R_galactic: Distance from galactic center [pc]
        G: Gravitational constant [pc^3 Msun^-1 Myr^-2 for STELLAR]

    Returns:
        Jacobi radius [pc]

    Reference:
        Binney & Tremaine (2008) Section 8.3.1
    """
    Omega = V_circ / R_galactic
    r_J = (G * M_cluster / (2.0 * Omega**2)) ** (1.0/3.0)
    return r_J


@jax.custom_jvp
def _truncation_weight(
    signed_dist: Float[Array, "N"],
    width: Float[Array, ""],
) -> Float[Array, "N"]:
    """Hard membership weight ``1[signed_dist >= 0]`` with a smooth surrogate grad.

    ``signed_dist = r_t - r_i`` (>=0 inside the truncation radius). The forward
    pass is an EXACT Heaviside step (sharp truncation); the custom JVP replaces
    the (delta-function) true derivative with a logistic surrogate of scale
    ``width`` — a straight-through estimator — so the truncation radius stays
    differentiable. ``width``'s own tangent is intentionally ignored (it is a
    gradient-smoothing hyperparameter, not an inference target).

    Reference:
        Bengio, Léonard & Courville (2013) arXiv:1308.3432 (straight-through estimator)
    """
    return jnp.where(signed_dist >= 0.0, 1.0, 0.0)


@_truncation_weight.defjvp
def _truncation_weight_jvp(primals, tangents):
    signed_dist, width = primals
    sd_dot, _width_dot = tangents
    primal_out = _truncation_weight(signed_dist, width)
    s = jax.nn.sigmoid(signed_dist / width)
    surrogate = s * (1.0 - s) / width  # logistic bump; integrates to 1
    return primal_out, surrogate * sd_dot


def apply_tidal_truncation(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    r_t: float,
    grad_width: float = 0.05,
) -> Tuple[Float[Array, "N 3"], Float[Array, "N 3"], Float[Array, "N"], Float[Array, "N"]]:
    """Sharp tidal truncation — shape-preserving and differentiable in ``r_t``.

    Particles with ``r > r_t`` are "removed" by setting their mass to zero, so
    they contribute nothing to mass-weighted quantities (energy, virial, COM).
    The forward pass is an EXACT hard cut; the backward pass uses a logistic
    straight-through surrogate (width ``grad_width * r_t``) so that ``r_t`` — and
    any upstream parameter feeding it (e.g. via :func:`jacobi_radius`) — remains
    differentiable. Unlike boolean-mask indexing, the output keeps a static
    shape ``N``, so the function is ``jit`` / ``vmap`` / ``grad`` safe.

    .. warning::
       The survivors keep the velocities they were drawn with for the UNtruncated
       potential, so the truncated set is SUPER-VIRIAL w.r.t. its own (now shallower)
       potential — some stars near ``r_t`` are formally unbound and the set is not a
       stationary equilibrium (audit S4). Re-virialize the survivors
       (``virial_scale`` / ``rescale_velocities_to_virial``) or use an r_t-consistent
       equilibrium model (King / LIMEPY) if you need a stationary IC.

    Args:
        positions: Particle positions (N, 3)
        velocities: Particle velocities (N, 3)
        masses: Particle masses (N,)
        r_t: Tidal truncation radius [same length units as ``positions``]
        grad_width: Surrogate-gradient smoothing scale as a fraction of ``r_t``
            (default 0.05). Affects ONLY the gradient w.r.t. ``r_t``, never the
            (exact) forward truncation.

    Returns:
        Tuple ``(positions, velocities, masses_truncated, keep_mask)``, all
        length ``N``:
        - ``positions``, ``velocities``: unchanged (truncated particles are left
          in place — never moved to ``inf``, which would give ``0 * inf = nan``
          in mass-weighted sums).
        - ``masses_truncated``: masses with ``r > r_t`` entries set to 0.
        - ``keep_mask``: boolean (N,), ``True`` where ``r <= r_t`` (use this to
          filter NUMBER-based downstream quantities, which still see the
          zero-mass "ghost" particles).

    Note:
        Sharp cutoff. For a physically smooth truncation consistent with King
        models, use the King profile directly.
    """
    radii = jnp.linalg.norm(positions, axis=1)
    width = grad_width * r_t
    weights = _truncation_weight(r_t - radii, width)
    masses_truncated = masses * weights
    keep_mask = radii <= r_t
    return positions, velocities, masses_truncated, keep_mask


def fill_factor_to_r_h(
    fill_factor: float,
    r_J: float,
) -> float:
    """Convert fill factor to half-mass radius.

    Fill factor = r_h / r_J is the ratio of half-mass radius to Jacobi radius.

    Typical values:
        - fill_factor ~ 0.05-0.15: Compact, tidally underfilling
        - fill_factor ~ 0.15-0.30: Typical globular cluster
        - fill_factor ~ 0.30-0.50: Tidally filling

    Args:
        fill_factor: r_h / r_J ratio (typically 0.05 to 0.5)
        r_J: Jacobi radius [length units]

    Returns:
        Half-mass radius r_h [length units]

    Reference:
        Baumgardt & Makino (2003) MNRAS 340, 227
    """
    return fill_factor * r_J


__all__ = [
    "jacobi_radius",
    "jacobi_radius_isothermal",
    "apply_tidal_truncation",
    "fill_factor_to_r_h",
]
