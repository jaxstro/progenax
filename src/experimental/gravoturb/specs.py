r"""Typed parameter specs for the gravoturbulent cluster-IC builder.

Groups :func:`gravoturb.cluster.build_cluster_ic`'s parameters into four physically-coherent
Equinox modules, so the builder signature never grows again as physics modes land
(finalization design 2026-07-16, Phase 0.5):

- :class:`CloudSpec`     — the turbulent cloud: PDF physics (ℳ, b, α) + density spectrum β.
- :class:`GeometrySpec`  — the realization geometry: envelope profile, box, grid.
- :class:`VelocitySpec`  — stellar velocity structure: coherence slope β_v + virial target.
- :class:`CompositionSpec` — who forms where: placement law + substructure knobs.

Validation is loud and constructor-time (``__check_init__``): a bad parameter fails at spec
construction with a physics message, never deep inside the pipeline. All specs are immutable
PyTrees; grid shape / mode strings are static fields.

Planned mode fields (documented so reviewers see the trajectory; each lands with its phase):
Phase 1 ``CompositionSpec.placement='multi_freefall'`` (default) with derived f_sub; Phase 2
``VelocitySpec.mode='physical'`` (σ_⋆ = η_v·ℳ·c_s, Q emergent); Phase 3
``CloudSpec.coupling='helmholtz'`` (β derived = β_v − 2, χ compressive fraction); Phase 4
``CompositionSpec.lambda_corr`` / ``companions``.

JAX-native (Equinox).
"""

import equinox as eqx
from jaxtyping import Array, Float


def _positive(name: str, value) -> None:
    if not float(value) > 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")


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
        if not 0.0 < float(self.b) <= 1.0:
            raise ValueError(f"b must be in (0, 1] (FK10 driving parameter), got {self.b}")
        if not float(self.alpha) > 1.0:
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
    r"""Stellar velocity structure: coherent turbulent field + virial-target amplitude.

    ``beta_v`` is the velocity-spectrum slope P_v(k) ∝ k^{-β_v} (spatial coherence; larger =
    smoother). ``Q_target`` is the imposed virial ratio Q ≡ T/|V| (0.5 virial, <0.5 cold /
    collapsing, >0.5 super-virial); the amplitude is set by ``virial_scale`` on the actual
    positions. (Phase 2 adds ``mode='physical'`` where σ_⋆ = η_v·ℳ·c_s and Q is emergent.)
    """

    beta_v: Float[Array, ""] | float
    Q_target: Float[Array, ""] | float

    def __check_init__(self):
        _positive("beta_v", self.beta_v)
        _positive("Q_target", self.Q_target)


class CompositionSpec(eqx.Module):
    r"""Star placement: which cells form stars, and the dense-tail physics.

    ``placement='multi_freefall'`` (default, Phase 1): the FK12 law p_⋆ ∝ w·ρ_total^{3/2}
    (SFR ∝ ρ/t_ff, gated on the BM19 transition) — the tail-star fraction is DERIVED
    (``ClusterIC.f_sub_derived``), not chosen; passing ``f_sub`` here is an error.

    ``placement='two_population'`` (legacy/ablation): ``n_tail = round(f_sub·N)`` stars
    from p ∝ w·ρ, the rest from p ∝ ρ — requires the free ``f_sub`` knob.

    ``mask_sharpness`` is the numerical sigmoid sharpness of the collapse-eligibility
    mask w (κ→∞ recovers the hard s > s_t indicator); distinct from the physical radial
    slope κ = 3/α.
    """

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
                "(read ClusterIC.f_sub_derived); pass f_sub only with "
                "placement='two_population'"
            )
        if self.placement == "two_population":
            if self.f_sub is None:
                raise ValueError("placement='two_population' requires the f_sub knob")
            if not 0.0 <= float(self.f_sub) <= 1.0:
                raise ValueError(f"f_sub must be in [0, 1], got {self.f_sub}")
        _positive("mask_sharpness", self.mask_sharpness)
