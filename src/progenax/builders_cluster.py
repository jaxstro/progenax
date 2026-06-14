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
from .tidal import apply_tidal_truncation

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
    Q: Optional[float] = 0.5,
    anisotropy_radius: Optional[Float[Array, ""]] = None,
    tidal_radius: Optional[Float[Array, ""]] = None,
    rotation: Optional[Union[float, RotationSpec]] = None,
    revirialize: bool = False,
    softening: float = 0.0,
) -> ICResult:
    """Build a single-population cluster IC from a profile object (see design doc).

    `Q` is the target virial ratio Q = T/|V| passed through to `build_spatial_ic`:
    `Q=0.5` virial-scales the IC to equilibrium (default); `Q=None` disables the
    rescale entirely, yielding the faithful UNSCALED equilibrium of the matched DF
    (the King/EFF/Michie/LIMEPY true-DF samplers are already in detailed equilibrium,
    so their measured Q lands near 0.5 with no rescale).
    """
    units = DEFAULT_UNITS if units is None else units
    masses, key_spatial = _resolve_masses(masses, n, imf, key)
    df = matched_velocity_df(profile, anisotropy_radius)
    ic = build_spatial_ic(profile, masses, df, key_spatial, G=units.G, Q=Q, softening=softening)

    if tidal_radius is None and rotation is None:
        return ic                                     # base case: bit-identical to build_spatial_ic
    return _apply_modifiers(ic, profile, tidal_radius, rotation, revirialize, Q, units.G, softening)


class RotationSpec(eqx.Module):
    """Rotation overlay spec. kind='solid' uses omega; kind='differential' uses (v_peak, R_peak).

    Additive kinematic overlay — injects L_z and raises Q above 0.5 (audit S3, NOT a stationary
    equilibrium). Differentiable in omega / v_peak / R_peak.
    """
    kind: str = eqx.field(static=True, default="solid")
    omega: Optional[Float[Array, ""]] = None
    v_peak: Optional[Float[Array, ""]] = None
    R_peak: Optional[Float[Array, ""]] = None
    # Array class-defaults are rejected by eqx/dataclasses ("mutable default ... use
    # default_factory"), so the z-hat axis is supplied via a default_factory.
    axis: Float[Array, "3"] = eqx.field(default_factory=lambda: jnp.array([0.0, 0.0, 1.0]))

    def __post_init__(self):
        if self.kind == "solid" and self.omega is None:
            raise ValueError("RotationSpec(kind='solid') requires omega=...")
        if self.kind == "differential" and (self.v_peak is None or self.R_peak is None):
            raise ValueError("RotationSpec(kind='differential') requires v_peak=... and R_peak=...")
        if self.kind not in ("solid", "differential"):
            raise ValueError(f"RotationSpec.kind must be 'solid' or 'differential', got {self.kind!r}")


# All non-Plummer build_cluster profiles carry a native truncation radius r_t, so the
# `tidal_radius` modifier would double-truncate them (audit-S4 inner cut) or be a silent
# outer no-op. `tidal_radius` is therefore valid ONLY for Plummer (the one untruncated
# profile); the truncated families set r_t on the profile (already differentiable for
# inference). King/Michie/LIMEPY derive r_t from where ψ→0; EFF's r_t is a prescribed
# truncation — both make `tidal_radius` illegitimate.
_TRUNCATED_PROFILES = (KingProfile, LIMEPYProfile, MichieProfile, EFFProfile)


def _apply_modifiers(ic, profile, tidal_radius, rotation, revirialize, Q, G, softening):
    positions, velocities, masses = ic.positions, ic.velocities, ic.masses

    if tidal_radius is not None:
        if isinstance(profile, _TRUNCATED_PROFILES):
            raise ValueError(
                f"{type(profile).__name__} is already tidally truncated (native r_t); passing "
                f"tidal_radius would double-truncate. For a stationary truncated equilibrium set "
                f"r_t on the profile instead (it is the recommended route — no audit-S4 issue)."
            )
        if revirialize and Q is None:
            raise ValueError("revirialize=True needs a numeric Q target; got Q=None")
        positions, velocities, masses, _keep = apply_tidal_truncation(
            positions, velocities, masses, tidal_radius)
        if revirialize:
            velocities = virial_scale(positions, velocities, masses, Q, G, softening)

    if rotation is not None:
        spec = RotationSpec(omega=rotation) if not isinstance(rotation, RotationSpec) else rotation
        if spec.kind == "solid":
            velocities = apply_solid_body_rotation(velocities, positions, spec.omega, spec.axis)
        else:
            velocities = apply_differential_rotation(
                velocities, positions, spec.v_peak, spec.R_peak, spec.axis)

    return ICResult(
        positions=positions, velocities=velocities, masses=masses,
        stellar_radii=compute_stellar_radii(masses), ids=ic.ids,
        primordial_system_id=ic.primordial_system_id,
        is_primordial_secondary=ic.is_primordial_secondary, component_id=ic.component_id,
    )


def build_plummer_cluster(*, key, masses=None, n=None, r_h=1.0, imf=None, **kw):
    return build_cluster(PlummerProfile(r_h=r_h), key=key, masses=masses, n=n, imf=imf, **kw)


def build_king_cluster(*, key, masses=None, n=None, W0=7.0, r_c=1.0, imf=None, **kw):
    return build_cluster(KingProfile.from_W0_rc(W0=W0, r_c=r_c), key=key,
                         masses=masses, n=n, imf=imf, **kw)


def build_eff_cluster(*, key, masses=None, n=None, a=1.0, gamma=3.0, r_t=10.0, imf=None, **kw):
    return build_cluster(EFFProfile(a=a, gamma=gamma, r_t=r_t), key=key,
                         masses=masses, n=n, imf=imf, **kw)


def build_michie_cluster(*, key, masses=None, n=None, W0=7.0, r_c=1.0, r_a=8.0, imf=None, **kw):
    return build_cluster(MichieProfile.from_W0_rc(W0=W0, r_c=r_c, r_a=r_a), key=key,
                         masses=masses, n=n, imf=imf, **kw)


def build_limepy_cluster(*, key, masses=None, n=None, W0=5.0, g=1.0, r_c=1.0, r_a=None,
                         imf=None, **kw):
    return build_cluster(LIMEPYProfile.from_W0_rc(W0=W0, g=g, r_c=r_c, r_a=r_a), key=key,
                         masses=masses, n=n, imf=imf, **kw)


class ClusterParams(eqx.Module):
    """Differentiable θ-PyTree: profile + named modifier knobs (see design doc).

    jax.grad over a ClusterParams gives joint gradients over the profile's float leaves AND
    any non-None modifier (the structure declares which params are free). `revirialize`/
    `softening` are static force-model config -> kwargs of build_cluster_from_params, not fields.
    """
    profile: SpatialProfile
    anisotropy_radius: Optional[Float[Array, ""]] = None
    tidal_radius: Optional[Float[Array, ""]] = None
    rotation: Optional[Union[float, RotationSpec]] = None
    Q: Optional[float] = 0.5


def build_cluster_from_params(
    params: ClusterParams, *, key, masses=None, n=None, imf=None, units=None,
    revirialize: bool = False, softening: float = 0.0,
) -> ICResult:
    """Unpack a ClusterParams θ-PyTree into build_cluster (the inference forward map)."""
    return build_cluster(
        params.profile, key=key, masses=masses, n=n, imf=imf, units=units,
        anisotropy_radius=params.anisotropy_radius, tidal_radius=params.tidal_radius,
        rotation=params.rotation, Q=params.Q, revirialize=revirialize, softening=softening)
