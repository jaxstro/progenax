r"""End-to-end gravoturbulent cluster initial conditions (the forward generative tool).

This is the user-facing assembly that turns natal-turbulence parameters into a complete,
physically-scaled N-body IC, composing the verified subsystem pieces:

  1. ``build_turbulent_field`` — BM19 turbulent log-density box ``s_turb`` (carries β, ℳ, α; ⟨e^{s_turb}⟩=1).
  2. ``apply_spherical_envelope`` — add the radial log-envelope of a progenax ``SpatialProfile``:
     ``s_total = s_turb + ln ρ_env(r)`` (the SHAPE: r_h, concentration).
  3. star placement (per ``CompositionSpec.placement``):
     **default ``'multi_freefall'``** — ``p_⋆ ∝ w(s_turb)·e^{(3/2)s_total}`` (the FK12 Eq. 7
     SFR ∝ ρ/t_ff law, collapse-gated on the BM19 transition; ``ClusterIC`` reports the derived
     ``tail_star_fraction`` and ``collapse_eligible_fraction``);
     legacy ``'two_population'`` — ``n_tail = round(f_sub·N)`` from p ∝ w·ρ, the rest ∝ ρ
     (ablation mode; the dense-tail mask always stays on ``s_turb``, decoupled from the envelope).
  4. ``turbulent_velocity_field`` + ``sample_turbulent_velocities`` — coherent turbulent velocities
     (β_v); nearby stars move together (Goodwin & Whitworth 2004).
  5. core ``to_com_frame``, then the velocity AMPLITUDE per ``VelocitySpec.mode``:
     ``'virial_target'`` — ``virial_scale`` imposes a CHOSEN virial ratio ``Q ≡ T/|V|`` (0.5
     virial, <0.5 collapsing, 0.75 super-virial; the potential uses the actual positions, so the
     envelope's deeper well is auto-accounted for); ``'physical'`` (Phase 2) —
     ``scale_to_dispersion`` sets the mass-weighted 3-D dispersion to σ_⋆ = η_v·ℳ·c_s (stars
     inherit the gas turbulence; **Q_virial is then emergent**, and the BM92-form ``alpha_vir``
     is the consistency diagnostic, reported in both modes).

Units are explicit (pass ``G``; CLAUDE.md mandate): ``box_size`` and the profile's ``r_h`` share length
units (pc for ``STELLAR``), masses are M⊙, and the output velocities come out in the matching system
(pc/Myr for ``STELLAR.G``). Positions/velocities are non-differentiable (categorical placement);
the differentiable interface is the inference layer plus the analytic fraction diagnostics.

JAX-native.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gravoturb.realization.envelope import apply_spherical_envelope
from gravoturb.realization.pipeline import TurbulentField, build_turbulent_field
from gravoturb.realization.placement import (
    collapse_eligible_fraction,
    effective_cell_count,
    multi_freefall_pmf,
    sample_positions,
    sample_positions_multi_freefall,
    tail_star_fraction,
)
from gravoturb.realization.turbulent_velocity import (
    sample_turbulent_velocities,
    scale_to_dispersion,
    turbulent_velocity_field,
)
from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec
from gravoturb.theory.collapse_threshold import virial_parameter

# One-way dependency: released progenax never imports gravoturb (enforced by
# tests/experimental/unit/test_core_severance.py), so this import is cycle-free.
from progenax import (
    compute_kinetic_energy,
    compute_potential_energy,
    to_com_frame,
    virial_scale,
)


class ClusterIC(NamedTuple):
    """A complete gravoturbulent cluster IC (a JAX pytree)."""

    positions: Float[Array, "n 3"]    # COM-centred, length units of box_size (pc for STELLAR)
    velocities: Float[Array, "n 3"]   # COM frame, units set by G (pc/Myr for STELLAR)
    masses: Float[Array, "n"]         # M⊙
    field: TurbulentField             # realized turbulent field (BM19 provenance / diagnostics)
    Q_virial: Float[Array, ""]        # realized virial ratio T/|V|
    tail_star_fraction: Float[Array, ""] | None = None
    #   expected fraction of stars in dense-tail (s>s_t) cells under the ACTUAL
    #   placement PMF — the successor of the legacy f_sub knob (multi_freefall only)
    collapse_eligible_fraction: Float[Array, ""] | None = None
    #   collapse-eligible share of the UNGATED freefall measure — the smooth
    #   differentiable diagnostic; NOT a star fraction (multi_freefall only)
    placement_n_eff: Float[Array, ""] | None = None
    #   effective number of cells the placement PMF spreads stars over (1/Σp²) —
    #   a resolution-monitoring diagnostic (grows with grid size; low values flag
    #   the under-resolved-tail regime; use ≥64³ at ℳ≥8, the AC-IC4 caveat)
    alpha_vir: Float[Array, ""] | None = None
    #   Bertoldi & McKee (1992)-form virial parameter 5σ²r_h/(GM) measured on the
    #   REALIZED cluster (mass-weighted 3-D dispersion, realized half-mass radius) —
    #   the physical-mode consistency diagnostic (reported in both modes; ~2Q for a
    #   uniform sphere, so expect O(1) values, not exact Q agreement)


def build_cluster_ic(
    masses: Float[Array, "n"],
    *,
    cloud: CloudSpec,
    geometry: GeometrySpec,
    velocity: VelocitySpec,
    composition: CompositionSpec,
    G: float,
    key: jax.Array,
    units=None,
) -> ClusterIC:
    r"""Build a spherical, substructured, velocity-scaled gravoturbulent cluster IC.

    ``masses`` (M⊙, masses-first per the ecosystem convention) sets ``n_stars = len(masses)``.
    Parameters are grouped into the four typed specs (:mod:`gravoturb.specs`), each validated
    at construction; ``G`` stays explicit (units mandate). Returns a :class:`ClusterIC`.

    The velocity amplitude follows ``velocity.mode``: ``'virial_target'`` imposes Q via
    ``virial_scale``; ``'physical'`` sets the mass-weighted 3-D dispersion to
    σ_⋆ = η_v·ℳ·c_s (stars inherit the gas turbulence; **Q_virial is then emergent**) and
    additionally requires ``units`` (a jaxstro ``UnitSystem`` consistent with ``G``) to
    convert ``c_s`` from km/s into the G-implied velocity unit (pc/Myr for STELLAR).
    """
    n_stars = int(masses.shape[0])
    k_field, k_vfield, k_pos = jax.random.split(key, 3)

    field = build_turbulent_field(
        cloud.mach, cloud.b, cloud.alpha, cloud.beta, geometry.shape, k_field
    )
    s_total = apply_spherical_envelope(field.s, geometry.profile, geometry.box_size)

    if composition.placement == "multi_freefall":
        positions = sample_positions_multi_freefall(
            field.s, field.s_t, composition.mask_sharpness, n_stars, k_pos,
            box_size=geometry.box_size, s_density=s_total,
        )
        args = (field.s, field.s_t, composition.mask_sharpness)
        f_tail = tail_star_fraction(*args, s_density=s_total)
        f_elig = collapse_eligible_fraction(*args, s_density=s_total)
        n_eff = effective_cell_count(multi_freefall_pmf(*args, s_density=s_total))
    else:  # 'two_population' (legacy/ablation; guarded by CompositionSpec)
        positions = sample_positions(
            field.s, field.s_t, composition.mask_sharpness, composition.f_sub,
            n_stars, k_pos, box_size=geometry.box_size, s_density=s_total,
        )
        f_tail = f_elig = n_eff = None

    v_field = turbulent_velocity_field(geometry.shape, velocity.beta_v, k_vfield)
    v_raw = sample_turbulent_velocities(positions, v_field, box_size=geometry.box_size)

    pos_com, v_com = to_com_frame(positions, v_raw, masses)
    if velocity.mode == "physical":
        if units is None:
            raise ValueError(
                "VelocitySpec(mode='physical') requires units=... (a jaxstro UnitSystem, "
                "e.g. STELLAR): c_s is in km/s and must be converted to the G-consistent "
                "velocity unit"
            )
        if abs(float(units.G) - float(G)) > 1e-9 * abs(float(G)):
            raise ValueError(
                f"units and G disagree (units.G={units.G!r}, G={G!r}); pass a "
                f"consistent pair — no silent precedence"
            )
        # σ_⋆ = η_v·ℳ·c_s, with c_s [km/s] → length/time units of G (0.9778 km/s per
        # pc/Myr for STELLAR); scaling happens AFTER COM removal, so the realized
        # 3-D dispersion equals σ_⋆ exactly and Q_virial below is emergent.
        sigma_star = velocity.eta_v * cloud.mach * velocity.c_s / units.velocity_scale_km_s
        v_scaled = scale_to_dispersion(v_com, masses, sigma_star)
    else:  # 'virial_target' (byte-identical to the pre-Phase-2 pipeline)
        v_scaled = virial_scale(pos_com, v_com, masses, Q_target=velocity.Q_target, G=G)

    T = compute_kinetic_energy(v_scaled, masses)
    V = compute_potential_energy(pos_com, masses, G=G)
    Q_virial = T / jnp.abs(V)

    m_total = jnp.sum(masses)
    sigma_3d = jnp.sqrt(2.0 * T / m_total)  # mass-weighted 3-D dispersion (COM frame)
    alpha_vir = virial_parameter(
        m_total, _half_mass_radius(pos_com, masses), sigma_3d, G=G
    )

    return ClusterIC(
        positions=pos_com, velocities=v_scaled, masses=masses, field=field,
        Q_virial=Q_virial, tail_star_fraction=f_tail,
        collapse_eligible_fraction=f_elig, placement_n_eff=n_eff,
        alpha_vir=alpha_vir,
    )


def _half_mass_radius(
    positions: Float[Array, "n 3"], masses: Float[Array, "n"]
) -> Float[Array, ""]:
    """Realized half-mass radius: smallest star radius enclosing half the total mass.

    Sort-based (non-differentiable — fine here: positions are categorically placed)."""
    r = jnp.linalg.norm(positions, axis=1)
    order = jnp.argsort(r)
    enclosed = jnp.cumsum(masses[order])
    idx = jnp.searchsorted(enclosed, 0.5 * enclosed[-1])
    return r[order][idx]
