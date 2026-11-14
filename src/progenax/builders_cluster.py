"""Convenience cluster-IC builders (thin, differentiable sugar over build_spatial_ic).

Public API: build_cluster, matched_velocity_df, RotationSpec, ClusterParams,
build_cluster_from_params, and the named aliases build_{plummer,king,eff,michie,limepy}_cluster.

Design: internal design note (round-2 addendum).
"""

from __future__ import annotations

from typing import Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from .builders import ICResult, build_spatial_ic, virial_scale
from .defaults import DEFAULT_UNITS
from .kinematics import (
    EFFVelocityDF,
    KingVelocityDF,
    LIMEPYVelocityDF,
    MichieVelocityDF,
    PlummerVelocityDF,
    apply_differential_rotation,
    apply_solid_body_rotation,
)
from .profiles import (
    EFFProfile,
    KingProfile,
    LIMEPYProfile,
    MichieProfile,
    PlummerProfile,
)
from .protocols import SpatialProfile, VelocityDF
from .stellar import zams_radius
from .tidal import apply_tidal_truncation

__all__ = [
    "build_cluster",
    "matched_velocity_df",
    "RotationSpec",
    "ClusterParams",
    "build_cluster_from_params",
    "build_plummer_cluster",
    "build_king_cluster",
    "build_eff_cluster",
    "build_michie_cluster",
    "build_limepy_cluster",
]


def matched_velocity_df(
    profile: SpatialProfile,
    anisotropy_radius: Optional[ArrayLike] = None,
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
        return EFFVelocityDF(
            a=profile.a,
            gamma=profile.gamma,
            r_t=profile.r_t,
            anisotropy_radius=anisotropy_radius,
        )
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


def _resolve_masses(masses, n, imf, key, *, mass_scale_to_msun):
    """Return native masses and spatial key; IMF/equal-star inputs are in M_sun."""
    if masses is not None:
        if n is not None:
            raise ValueError("pass exactly one of `masses` or `n` (got both).")
        if imf is not None:
            raise ValueError(
                "`imf` requires `n` (the count to sample); pass `n=...`, not `masses=...`."
            )
        return masses, key
    if n is None:
        raise ValueError("pass exactly one of `masses` or `n` (got neither).")
    if imf is None:
        return jnp.ones(n) / mass_scale_to_msun, key  # physical 1 M_sun each
    key_imf, key_spatial = jax.random.split(key)
    return imf.sample(key_imf, n) / mass_scale_to_msun, key_spatial


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
    """Build a single-population IC with a scale-matched velocity distribution.

    Resolves masses, pairs the profile through :func:`matched_velocity_df`, and
    delegates sampling, center-of-mass subtraction and optional velocity
    rescaling to :func:`build_spatial_ic`. Tidal cuts and rotation follow.

    Args:
        profile: Plummer, King, EFF, Michie or LIMEPY profile instance. Custom
            profile/DF pairs use :func:`build_spatial_ic` directly.
        key: JAX random key. An IMF draw splits it into mass and spatial keys;
            supplied masses or equal masses pass it directly to spatial sampling.
        masses: Particle masses of shape (N,). Choose exactly one of ``masses``
            and ``n``. Supplied masses use the chosen dynamical mass unit.
        n: Particle count when ``masses`` is omitted; static under JIT. Without
            ``imf``, each star has physical mass 1 M_sun, converted to native units.
        imf: Mass sampler providing ``sample(key, n)`` in M_sun; draws are
            converted to native units. Requires ``n`` and cannot accompany
            supplied ``masses``. Default None.
        units: UnitSystem supplying G and mass_scale_cgs; None resolves to
            DEFAULT_UNITS (STELLAR: M_sun, pc, Myr). Supplied masses, profile
            lengths and modifier parameters must use this system's units.
        Q: Target energy ratio T/abs(V), with T the kinetic energy and V the
            gravitational potential energy. Default 0.5 rescales velocities;
            None skips this rescaling while retaining center-of-mass subtraction.
        anisotropy_radius: Osipkov-Merritt radius in profile length units for
            Plummer/EFF; None is isotropic. Michie/LIMEPY use their profile's r_a.
        tidal_radius: Optional Plummer cut radius in profile length units.
            Stars outside it become zero-mass entries with the same array shape.
            Native truncated profiles reject this additional cut.
        rotation: Optional angular frequency (inverse time units) for a
            solid-body overlay, or a :class:`RotationSpec`. Applied after cuts.
        revirialize: If True, rescale velocities to Q after a tidal cut and
            before rotation. Requires numeric Q when a cut is requested.
            Default False.
        softening: Length used for potential energy during rescaling, default
            0.0. It is not stored on the returned IC.

    Returns:
        :class:`ICResult` with positions and velocities (N, 3), masses (N,)
        and ZAMS stellar radii (N,) [R_sun]. Tidal cuts retain N entries.

    Raises:
        ValueError: Incompatible mass/count/IMF selection, unsupported anisotropy
            or tidal modifier, or a tidal revirialization request with Q=None.
        TypeError: Profile type unsupported by :func:`matched_velocity_df`.

    Notes:
        Supplied masses and profile lengths are not converted. For stellar
        labels, native masses are converted to M_sun before :func:`zams_radius`;
        its output stays in R_sun. Profile/DF domain restrictions still apply.
        A target Q sets an energy ratio; it does not certify stationarity.
        Rotation changes that ratio after any requested rescaling. Hard tidal
        cuts use the existing straight-through gradient surrogate, not the
        derivative of a smooth physical selection.
    """
    units = DEFAULT_UNITS if units is None else units
    mass_scale_to_msun = units.mass_scale_cgs / STELLAR.mass_scale_cgs
    masses, key_spatial = _resolve_masses(
        masses, n, imf, key, mass_scale_to_msun=mass_scale_to_msun
    )
    df = matched_velocity_df(profile, anisotropy_radius)
    ic = build_spatial_ic(
        profile,
        masses,
        df,
        key_spatial,
        G=units.G,
        Q=Q,
        softening=softening,
        mass_scale_to_msun=mass_scale_to_msun,
    )

    if tidal_radius is None and rotation is None:
        return ic  # base case: bit-identical to build_spatial_ic
    return _apply_modifiers(
        ic,
        profile,
        tidal_radius,
        rotation,
        revirialize,
        Q,
        units.G,
        softening,
        mass_scale_to_msun=mass_scale_to_msun,
    )


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
    axis: Float[Array, "3"] = eqx.field(
        default_factory=lambda: jnp.array([0.0, 0.0, 1.0])
    )

    def __post_init__(self):
        if self.kind == "solid" and self.omega is None:
            raise ValueError("RotationSpec(kind='solid') requires omega=...")
        if self.kind == "differential" and (self.v_peak is None or self.R_peak is None):
            raise ValueError(
                "RotationSpec(kind='differential') requires v_peak=... and R_peak=..."
            )
        if self.kind not in ("solid", "differential"):
            raise ValueError(
                f"RotationSpec.kind must be 'solid' or 'differential', got {self.kind!r}"
            )


# All non-Plummer build_cluster profiles carry a native truncation radius r_t, so the
# `tidal_radius` modifier would double-truncate them (audit-S4 inner cut) or be a silent
# outer no-op. `tidal_radius` is therefore valid ONLY for Plummer (the one untruncated
# profile); the truncated families set r_t on the profile (already differentiable for
# inference). King/Michie/LIMEPY derive r_t from where ψ→0; EFF's r_t is a prescribed
# truncation — both make `tidal_radius` illegitimate.
_TRUNCATED_PROFILES = (KingProfile, LIMEPYProfile, MichieProfile, EFFProfile)


def _apply_modifiers(
    ic,
    profile,
    tidal_radius,
    rotation,
    revirialize,
    Q,
    G,
    softening,
    *,
    mass_scale_to_msun,
):
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
            positions, velocities, masses, tidal_radius
        )
        if revirialize:
            velocities = virial_scale(positions, velocities, masses, Q, G, softening)

    if rotation is not None:
        spec = (
            RotationSpec(omega=rotation)
            if not isinstance(rotation, RotationSpec)
            else rotation
        )
        if spec.kind == "solid":
            velocities = apply_solid_body_rotation(
                velocities, positions, spec.omega, spec.axis
            )
        else:
            velocities = apply_differential_rotation(
                velocities, positions, spec.v_peak, spec.R_peak, spec.axis
            )

    return ICResult(
        positions=positions,
        velocities=velocities,
        masses=masses,
        stellar_radii=zams_radius(masses * mass_scale_to_msun),
        ids=ic.ids,
        primordial_system_id=ic.primordial_system_id,
        is_primordial_secondary=ic.is_primordial_secondary,
        component_id=ic.component_id,
    )


def build_plummer_cluster(*, key, masses=None, n=None, r_h=1.0, imf=None, **kw):
    """Build a Plummer IC through :func:`build_cluster`.

    Args:
        key: JAX random key.
        masses: Mass array (N,); choose exactly one of ``masses`` and ``n``.
        n: Static sample count; absent ``imf``, each star has physical mass 1 M_sun.
        r_h: Three-dimensional half-mass radius in length units, default 1.0.
        imf: Optional mass sampler; requires ``n``.
        **kw: :func:`build_cluster` options, including units, Q, anisotropy_radius,
            tidal_radius, rotation, revirialize and softening. Q defaults to 0.5;
            use Q=None to skip velocity rescaling.

    Returns:
        :class:`ICResult` with the units, shapes and modifier behavior documented
        by :func:`build_cluster`.
    """
    return build_cluster(
        PlummerProfile(r_h=r_h), key=key, masses=masses, n=n, imf=imf, **kw
    )


def build_king_cluster(*, key, masses=None, n=None, W0=7.0, r_c=1.0, imf=None, **kw):
    """Build an isotropic King IC through :func:`build_cluster`.

    Args:
        key: JAX random key.
        masses: Mass array (N,); choose exactly one of ``masses`` and ``n``.
        n: Static sample count; absent ``imf``, each star has physical mass 1 M_sun.
        W0: Dimensionless central potential, default 7.0.
        r_c: King core-radius scale in length units, default 1.0.
        imf: Optional mass sampler; requires ``n``.
        **kw: :func:`build_cluster` options. Q defaults to 0.5; Q=None skips
            velocity rescaling. anisotropy_radius and tidal_radius are rejected.

    Returns:
        :class:`ICResult` with the units, shapes and modifier behavior documented
        by :func:`build_cluster`.

    Notes:
        :meth:`KingProfile.from_W0_rc` derives the tidal radius using its default
        solve settings. To select profile solve settings, construct the profile
        explicitly and consult :func:`matched_velocity_df`'s domain limitation.
    """
    return build_cluster(
        KingProfile.from_W0_rc(W0=W0, r_c=r_c),
        key=key,
        masses=masses,
        n=n,
        imf=imf,
        **kw,
    )


def build_eff_cluster(
    *, key, masses=None, n=None, a=1.0, gamma=3.0, r_t=10.0, imf=None, **kw
):
    """Build a truncated EFF IC through :func:`build_cluster`.

    Args:
        key: JAX random key.
        masses: Mass array (N,); choose exactly one of ``masses`` and ``n``.
        n: Static sample count; absent ``imf``, each star has physical mass 1 M_sun.
        a: EFF scale radius in length units, default 1.0.
        gamma: Dimensionless projected-density slope, default 3.0.
        r_t: Prescribed truncation radius in length units, default 10.0.
        imf: Optional mass sampler; requires ``n``.
        **kw: :func:`build_cluster` options, including anisotropy_radius for
            Osipkov-Merritt velocities. Q defaults to 0.5; Q=None skips velocity
            rescaling. The additional tidal_radius modifier is rejected.

    Returns:
        :class:`ICResult` with the units, shapes and modifier behavior documented
        by :func:`build_cluster`. EFF profile/DF boundary limitations still apply.
    """
    return build_cluster(
        EFFProfile(a=a, gamma=gamma, r_t=r_t),
        key=key,
        masses=masses,
        n=n,
        imf=imf,
        **kw,
    )


def build_michie_cluster(
    *, key, masses=None, n=None, W0=7.0, r_c=1.0, r_a=8.0, imf=None, **kw
):
    """Build a Michie-King IC through :func:`build_cluster`.

    Args:
        key: JAX random key.
        masses: Mass array (N,); choose exactly one of ``masses`` and ``n``.
        n: Static sample count; absent ``imf``, each star has physical mass 1 M_sun.
        W0: Dimensionless central potential, default 7.0.
        r_c: Core-radius scale in length units, default 1.0.
        r_a: Intrinsic Michie anisotropy radius in length units, default 8.0.
        imf: Optional mass sampler; requires ``n``.
        **kw: :func:`build_cluster` options. Q defaults to 0.5; Q=None skips
            velocity rescaling. anisotropy_radius and tidal_radius are rejected.

    Returns:
        :class:`ICResult` with the units, shapes and modifier behavior documented
        by :func:`build_cluster`.

    Notes:
        :meth:`MichieProfile.from_W0_rc` derives the tidal radius using its
        default solve settings and concrete-input non-truncation guard.
        Consult :func:`matched_velocity_df` for custom profile solve domains.
    """
    return build_cluster(
        MichieProfile.from_W0_rc(W0=W0, r_c=r_c, r_a=r_a),
        key=key,
        masses=masses,
        n=n,
        imf=imf,
        **kw,
    )


def build_limepy_cluster(
    *, key, masses=None, n=None, W0=5.0, g=1.0, r_c=1.0, r_a=None, imf=None, **kw
):
    """Build a single-component LIMEPY IC through :func:`build_cluster`.

    Args:
        key: JAX random key.
        masses: Mass array (N,); choose exactly one of ``masses`` and ``n``.
        n: Static sample count; absent ``imf``, each star has physical mass 1 M_sun.
        W0: Dimensionless central potential, default 5.0.
        g: Dimensionless lowering index, default 1.0 (King energy cutoff).
        r_c: Core-radius scale in length units, default 1.0.
        r_a: Intrinsic anisotropy radius in length units; default None is
            isotropic. Use this parameter rather than anisotropy_radius.
        imf: Optional mass sampler; requires ``n``.
        **kw: :func:`build_cluster` options. Q defaults to 0.5; Q=None skips
            velocity rescaling. anisotropy_radius and tidal_radius are rejected.

    Returns:
        :class:`ICResult` with the units, shapes and modifier behavior documented
        by :func:`build_cluster`.

    Notes:
        :meth:`LIMEPYProfile.from_W0_rc` uses its default solve settings and
        derives the tidal radius. Consult :func:`matched_velocity_df` for custom
        profile solve domains. Stellar masses share this one spatial/velocity
        population; multicomponent equilibrium uses MultiComponentCluster.
    """
    return build_cluster(
        LIMEPYProfile.from_W0_rc(W0=W0, g=g, r_c=r_c, r_a=r_a),
        key=key,
        masses=masses,
        n=n,
        imf=imf,
        **kw,
    )


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
    params: ClusterParams,
    *,
    key,
    masses=None,
    n=None,
    imf=None,
    units=None,
    revirialize: bool = False,
    softening: float = 0.0,
) -> ICResult:
    """Unpack a ClusterParams θ-PyTree into build_cluster (the inference forward map)."""
    return build_cluster(
        params.profile,
        key=key,
        masses=masses,
        n=n,
        imf=imf,
        units=units,
        anisotropy_radius=params.anisotropy_radius,
        tidal_radius=params.tidal_radius,
        rotation=params.rotation,
        Q=params.Q,
        revirialize=revirialize,
        softening=softening,
    )
