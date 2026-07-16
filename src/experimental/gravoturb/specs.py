r"""Typed parameter specs for the gravoturbulent cluster-IC builder.

Groups :func:`gravoturb.cluster.build_cluster_ic`'s parameters into four physically-coherent
Equinox modules, so the builder signature never grows again as physics modes land
(finalization design 2026-07-16, Phase 0.5):

- :class:`CloudSpec`     — the turbulent cloud: PDF physics (ℳ, b, α) + density spectrum β.
- :class:`GeometrySpec`  — the realization geometry: envelope profile, box, grid.
- :class:`VelocitySpec`  — stellar velocity structure: coherence slope β_v + amplitude mode
  (imposed Q_target, or physical σ_⋆ = η_v·ℳ·c_s with Q emergent).
- :class:`CompositionSpec` — who forms where: placement law + substructure knobs.

Validation is loud and constructor-time (``__check_init__``): a bad parameter fails at spec
construction with a physics message, never deep inside the pipeline. Constructing a spec
from TRACED values (inside jit/vmap/grad sweeps) is supported: value checks are skipped for
tracers (parity with the pre-spec flat signature) — static fields (mode strings, grid shape,
f_sub) must always be concrete. All specs are immutable
PyTrees; grid shape / mode strings are static fields.

Mode fields land with their phases (documented so reviewers see the trajectory):
Phase 1 ``CompositionSpec.placement='multi_freefall'`` (default) with derived f_sub — LANDED;
Phase 2 ``VelocitySpec.mode='physical'`` (σ_⋆ = η_v·ℳ·c_s, Q emergent) — LANDED; Phase 3
``CloudSpec.coupling='helmholtz'`` (β derived = β_v − 2, χ compressive fraction); Phase 4
``CompositionSpec.lambda_corr`` / ``companions``.

JAX-native (Equinox).
"""

import equinox as eqx
import jax.core
import jax.numpy as jnp
from jaxtyping import Array, Float


def _is_traced(value) -> bool:
    return isinstance(value, jax.core.Tracer)


def _positive(name: str, value) -> None:
    if _is_traced(value):  # traced construction: defer to the physics (main parity)
        return
    if not float(value) > 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _non_negative(name: str, value) -> None:
    if _is_traced(value):
        return
    if not float(value) >= 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")


class CloudSpec(eqx.Module):
    r"""Turbulent-cloud parameters: the BM19 density-PDF physics + the density spectrum.

    ``mach`` (rms sonic ℳ), ``b`` (FK10 driving parameter, 1/3 solenoidal → 1 compressive),
    ``alpha`` (BM19 power-law tail slope, must be > 1 or the whole cloud is self-gravitating),
    ``beta`` (density power-spectrum slope P(k) ∝ k^{-β}; Kim & Ryu 2005 regime ≈ 2–11/3).
    """

    mach: Float[Array, ""] | float
    b: Float[Array, ""] | float
    alpha: Float[Array, ""] | float
    beta: Float[Array, ""] | float

    def __check_init__(self):
        _positive("mach", self.mach)
        if not _is_traced(self.b) and not 0.0 < float(self.b) <= 1.0:
            raise ValueError(f"b must be in (0, 1] (FK10 driving parameter), got {self.b}")
        if not _is_traced(self.alpha) and not float(self.alpha) > 1.0:
            raise ValueError(
                f"alpha must be > 1 (BM19: the power-law tail mass diverges as alpha→1), "
                f"got {self.alpha}"
            )
        _positive("beta", self.beta)


class GeometrySpec(eqx.Module):
    r"""Realization geometry: the cluster-shape envelope and the periodic grid.

    ``profile`` is any progenax ``SpatialProfile`` (its ``r_h`` shares length units with
    ``box_size``; NB the realized half-mass radius exceeds the envelope r_h under turbulence —
    AC-IC0). ``shape`` is the FFT grid (static).
    """

    profile: object
    box_size: Float[Array, ""] | float
    shape: tuple[int, int, int] = eqx.field(static=True)

    def __check_init__(self):
        _positive("box_size", self.box_size)
        if len(self.shape) != 3 or any(int(n) < 8 for n in self.shape):
            raise ValueError(f"shape must be a 3-tuple of grid sizes ≥ 8, got {self.shape}")
        if not hasattr(self.profile, "density"):
            raise TypeError(
                f"profile must be a progenax SpatialProfile (needs .density), "
                f"got {type(self.profile).__name__}"
            )


class VelocitySpec(eqx.Module):
    r"""Stellar velocity structure: coherent turbulent field + amplitude mode.

    ``beta_v`` is the velocity-spectrum slope P_v(k) ∝ k^{-β_v} (spatial coherence; larger =
    smoother). The AMPLITUDE is set by ``mode``:

    - ``mode='virial_target'`` (default): impose the virial ratio ``Q_target`` ≡ T/|V|
      (0.5 virial, <0.5 cold / collapsing, >0.5 super-virial) via ``virial_scale`` on the
      actual positions. ``c_s``/``eta_v`` must stay unset.
    - ``mode='physical'`` (Phase 2): the stars inherit the gas turbulence amplitude,
      σ_⋆ = η_v·ℳ·c_s (3-D mass-weighted dispersion; each component carries ~σ_⋆/√3).
      ``c_s`` is the sound speed in **km/s** (literature convention; the builder converts
      via the ``units`` UnitSystem), ``eta_v`` is the stars-inherit-gas efficiency
      (default 1; η_v<1 for subvirial-star studies, cf. Foster+2015). **Q_virial becomes
      an output**; passing ``Q_target`` here is an error (no silent precedence).
    """

    beta_v: Float[Array, ""] | float
    Q_target: Float[Array, ""] | float | None = None
    mode: str = eqx.field(static=True, default="virial_target")
    c_s: Float[Array, ""] | float | None = None    # sound speed [km/s] (physical mode)
    eta_v: Float[Array, ""] | float = 1.0          # σ_⋆ = η_v·ℳ·c_s (physical mode)

    def __check_init__(self):
        _positive("beta_v", self.beta_v)
        if self.mode not in ("virial_target", "physical"):
            raise ValueError(
                f"mode must be 'virial_target' or 'physical', got {self.mode!r}"
            )
        if self.mode == "physical":
            if self.Q_target is not None:
                raise ValueError(
                    "Q_target is EMERGENT under mode='physical' (read ClusterIC.Q_virial); "
                    "pass Q_target only with mode='virial_target' — no silent precedence"
                )
            if self.c_s is None:
                raise ValueError("mode='physical' requires the sound speed c_s [km/s]")
            _positive("c_s", self.c_s)
            _positive("eta_v", self.eta_v)
        else:
            if self.Q_target is None:
                raise ValueError("mode='virial_target' requires Q_target (0.5 = virial)")
            _non_negative("Q_target", self.Q_target)  # Q_target=0: cold collapse (v=0 IC)
            if self.c_s is not None:
                raise ValueError("c_s is a physical-mode knob; unused with mode='virial_target'")
            if not _is_traced(self.eta_v) and float(self.eta_v) != 1.0:
                raise ValueError(
                    "eta_v is a physical-mode knob; unused with mode='virial_target'"
                )


class CompositionSpec(eqx.Module):
    r"""Star placement: which cells form stars, and the dense-tail physics.

    ``placement='multi_freefall'`` (default, Phase 1): the FK12 law p_⋆ ∝ w·ρ_total^{3/2}
    (SFR ∝ ρ/t_ff, gated on the BM19 transition) — the tail-star fraction is DERIVED
    (``ClusterIC.tail_star_fraction`` + the smooth ``collapse_eligible_fraction``),
    not chosen; passing ``f_sub`` here is an error.

    ``placement='two_population'`` (legacy/ablation): ``n_tail = round(f_sub·N)`` stars
    from p ∝ w·ρ, the rest from p ∝ ρ — requires the free ``f_sub`` knob.

    ``mask_sharpness`` is the numerical sigmoid sharpness of the collapse-eligibility
    mask w (κ→∞ recovers the hard s > s_t indicator); distinct from the physical radial
    slope κ = 3/α.
    """

    # Design note: `placement` is kept as an explicit mode string (not derived from
    # `f_sub is None`) deliberately — Phase-4 fields (lambda_corr, companions) attach
    # mode-specific validation (decision: 2026-07-16 review).
    placement: str = eqx.field(static=True, default="multi_freefall")
    f_sub: float | None = eqx.field(static=True, default=None)
    mask_sharpness: Float[Array, ""] | float = 8.0

    def __check_init__(self):
        if self.placement not in ("multi_freefall", "two_population"):
            raise ValueError(
                f"placement must be 'multi_freefall' or 'two_population', "
                f"got {self.placement!r}"
            )
        if self.placement == "multi_freefall" and self.f_sub is not None:
            raise ValueError(
                "f_sub is DERIVED under placement='multi_freefall' "
                "(read ClusterIC.tail_star_fraction); pass f_sub only with "
                "placement='two_population'"
            )
        if self.placement == "two_population":
            if self.f_sub is None:
                raise ValueError("placement='two_population' requires the f_sub knob")
            if not 0.0 <= float(self.f_sub) <= 1.0:
                raise ValueError(f"f_sub must be in [0, 1], got {self.f_sub}")
        _positive("mask_sharpness", self.mask_sharpness)


def cloud_spec_from_larson(
    *,
    M_ecl: Float[Array, ""] | float,
    sfe: Float[Array, ""] | float,
    rho_cl: Float[Array, ""] | float,
    alpha: Float[Array, ""] | float,
    b: Float[Array, ""] | float | None = None,
    c_s: float | None = None,
    sigma_v0: float | None = None,
    alpha_larson: float | None = None,
) -> tuple[CloudSpec, Float[Array, ""]]:
    r"""Close cloud-level inputs through the released Larson chain → (CloudSpec, box_size).

    Given the embedded-cluster mass ``M_ecl`` [M⊙], star-formation efficiency ``sfe``, and
    cloud density ``rho_cl`` [M⊙ pc⁻³], derives the parent-cloud radius
    R_cloud = (3 M_ecl/sfe / 4πρ_cl)^{1/3}, then ℳ = σ_v(R_cloud)/c_s from the Larson
    velocity–size relation, β from the Kim & Ryu (2005) density-spectrum calibration, and
    (unless ``b`` is given) the driving parameter from ``b_from_environment(log₁₀ρ_cl)``.
    ``alpha`` (the BM19 tail slope) stays a free physics input. Returns the CloudSpec plus
    ``box_size = 2 R_cloud`` [pc] so the realization box spans the parent cloud.

    ``c_s``/``sigma_v0``/``alpha_larson`` default to the released
    :mod:`progenax.cluster.constants` values (0.2 km/s cold-GMC sound speed, 1 km/s at
    1 pc, exponent 0.5). Under ``VelocitySpec(mode='physical')`` pass the SAME ``c_s`` so
    σ_⋆ = η_v·ℳ·c_s = η_v·σ_v(R_cloud) — the Larson chain then closes end to end.

    Differentiable in (M_ecl, sfe, rho_cl).
    """
    # Lazy released-core import: keeps the spec module import-light; the dependency
    # direction (experimental -> released) is the cycle-free one (see cluster.py).
    from progenax.cluster.constants import ALPHA_LARSON, C_S_DEFAULT, SIGMA_V0_DEFAULT
    from progenax.cluster.turbulence import (
        b_from_environment,
        cloud_radius_from_density,
        spectral_slope_from_mach,
        turbulent_mach_from_cloud,
    )

    c_s = C_S_DEFAULT if c_s is None else c_s
    sigma_v0 = SIGMA_V0_DEFAULT if sigma_v0 is None else sigma_v0
    alpha_larson = ALPHA_LARSON if alpha_larson is None else alpha_larson

    R_cloud = cloud_radius_from_density(jnp.asarray(M_ecl), sfe, rho_cl)
    mach = turbulent_mach_from_cloud(R_cloud, c_s=c_s, sigma_v0=sigma_v0, alpha=alpha_larson)
    beta = spectral_slope_from_mach(mach)
    if b is None:
        b = b_from_environment(jnp.log10(jnp.asarray(rho_cl)))
    return CloudSpec(mach=mach, b=b, alpha=alpha, beta=beta), 2.0 * R_cloud
