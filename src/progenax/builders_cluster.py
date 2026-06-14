"""Convenience cluster-IC builders (thin, differentiable sugar over build_spatial_ic).

Public API: build_cluster, matched_velocity_df, RotationSpec, ClusterParams,
build_cluster_from_params, and the named aliases build_{plummer,king,eff,michie,limepy}_cluster.

Design: docs/plans/2026-06-14-cluster-builder-api-design.md (round-2 addendum).
"""
from __future__ import annotations
from typing import Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .defaults import DEFAULT_UNITS
from .protocols import SpatialProfile, VelocityDF
from .builders import ICResult, build_spatial_ic, virial_scale, compute_stellar_radii
from .profiles import (
    PlummerProfile, EFFProfile, KingProfile, MichieProfile, LIMEPYProfile,
)
from .kinematics import (
    PlummerVelocityDF, EFFVelocityDF, KingVelocityDF, MichieVelocityDF, LIMEPYVelocityDF,
    apply_solid_body_rotation, apply_differential_rotation,
)

_ZHAT = jnp.array([0.0, 0.0, 1.0])

__all__ = [
    "build_cluster", "matched_velocity_df", "RotationSpec", "ClusterParams",
    "build_cluster_from_params", "build_plummer_cluster", "build_king_cluster",
    "build_eff_cluster", "build_michie_cluster", "build_limepy_cluster",
]


def matched_velocity_df(
    profile: SpatialProfile,
    anisotropy_radius: Optional[float] = None,
) -> VelocityDF:
    """Return the equilibrium velocity DF whose scale params match `profile` exactly.

    Kills the r_h-desync footgun: the DF reads the profile's OWN scale fields, so a
    Plummer profile can never be paired with a mismatched-r_h DF.

    `anisotropy_radius` (Osipkov-Merritt r_a; β(r)=r²/(r²+r_a²)) is valid ONLY for
    Plummer/EFF (whose base DFs are isotropic and OM-augmentable). King is isotropic;
    Michie/LIMEPY carry their anisotropy intrinsically (set `r_a` on the profile via
    `from_W0_rc(..., r_a=...)`), so passing `anisotropy_radius` for those is an ERROR
    (no silent ignore).

    Caveat: King/Michie/LIMEPY DFs re-solve their ODE at DEFAULT domains (consistent
    with the default profile constructors). A profile built with a custom `xi_max`
    cannot round-trip that domain (not stored as a field) — hand-compose for that case.
    """
    if isinstance(profile, PlummerProfile):
        return PlummerVelocityDF(r_h=profile.r_h, anisotropy_radius=anisotropy_radius)
    if isinstance(profile, EFFProfile):
        return EFFVelocityDF(a=profile.a, gamma=profile.gamma, r_t=profile.r_t,
                             anisotropy_radius=anisotropy_radius)
    # --- isotropic / intrinsically-anisotropic models: anisotropy_radius is invalid ---
    if anisotropy_radius is not None:
        raise ValueError(
            f"anisotropy_radius is only valid for Plummer/EFF DFs; got "
            f"{type(profile).__name__}. King is isotropic; for Michie/LIMEPY set the "
            f"anisotropy radius r_a on the profile (e.g. "
            f"{type(profile).__name__}.from_W0_rc(..., r_a=...))."
        )
    if isinstance(profile, KingProfile):
        return KingVelocityDF(W0=profile.W0, r_c=profile.r_c)
    if isinstance(profile, MichieProfile):
        return MichieVelocityDF(W0=profile.W0, r_c=profile.r_c, r_a=profile.r_a)
    if isinstance(profile, LIMEPYProfile):
        # LIMEPY stores r_a=inf for the isotropic model; the DF wants r_a=None there.
        # Branch on the STATIC is_aniso flag (not a traced jnp.isfinite).
        r_a = profile.r_a if profile.is_aniso else None
        return LIMEPYVelocityDF(W0=profile.W0, g=profile.g, r_c=profile.r_c, r_a=r_a)
    raise TypeError(
        f"matched_velocity_df: unknown profile type {type(profile).__name__}. "
        f"Supported: Plummer, EFF, King, Michie, LIMEPY."
    )


def _resolve_masses(masses, n, imf, key):
    """Return (masses, key_spatial). Split the key ONLY when an IMF draw is needed."""
    if masses is not None:
        if n is not None:
            raise ValueError("pass exactly one of `masses` or `n` (got both).")
        if imf is not None:
            raise ValueError("`imf` requires `n` (the count to sample); pass `n=...`, not `masses=...`.")
        return masses, key
    if n is None:
        raise ValueError("pass exactly one of `masses` or `n` (got neither).")
    if imf is None:
        return jnp.ones(n), key                       # equal 1 M_sun, no PRNG needed
    key_imf, key_spatial = jax.random.split(key)
    return imf.sample(key_imf, n), key_spatial


def build_cluster(
    profile: SpatialProfile,
    *,
    key: PRNGKeyArray,
    masses: Optional[Float[Array, "N"]] = None,
    n: Optional[int] = None,
    imf=None,
    units=None,
    Q: float = 0.5,
    anisotropy_radius: Optional[float] = None,
    tidal_radius: Optional[float] = None,
    rotation: Optional[Union[float, "RotationSpec"]] = None,
    revirialize: bool = False,
    softening: float = 0.0,
) -> ICResult:
    """Build a single-population cluster IC from a profile object (see design doc)."""
    units = DEFAULT_UNITS if units is None else units
    masses, key_spatial = _resolve_masses(masses, n, imf, key)
    df = matched_velocity_df(profile, anisotropy_radius)
    ic = build_spatial_ic(profile, masses, df, key_spatial, G=units.G, Q=Q, softening=softening)

    if tidal_radius is None and rotation is None:
        return ic                                     # base case: bit-identical to build_spatial_ic
    return _apply_modifiers(ic, profile, tidal_radius, rotation, revirialize, Q, units.G, softening)


def _apply_modifiers(ic, profile, tidal_radius, rotation, revirialize, Q, G, softening):
    return ic  # filled in Batch 3
