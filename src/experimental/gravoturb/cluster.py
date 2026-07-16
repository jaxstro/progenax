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
  5. core ``to_com_frame`` then ``virial_scale`` — centre the cluster and set the velocity amplitude
     to a CHOSEN virial ratio ``Q ≡ T/|V|`` (0.5 virial, <0.5 collapsing, 0.75 super-virial). The
     potential uses the actual positions, so the envelope's deeper well is auto-accounted for.

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
    turbulent_velocity_field,
)
from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec

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


def build_cluster_ic(
    masses: Float[Array, "n"],
    *,
    cloud: CloudSpec,
    geometry: GeometrySpec,
    velocity: VelocitySpec,
    composition: CompositionSpec,
    G: float,
    key: jax.Array,
) -> ClusterIC:
    r"""Build a spherical, substructured, virial-scaled gravoturbulent cluster IC.

    ``masses`` (M⊙, masses-first per the ecosystem convention) sets ``n_stars = len(masses)``.
    Parameters are grouped into the four typed specs (:mod:`gravoturb.specs`), each validated
    at construction; ``G`` stays explicit (units mandate). Returns a :class:`ClusterIC`.
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
    v_scaled = virial_scale(pos_com, v_com, masses, Q_target=velocity.Q_target, G=G)

    T = compute_kinetic_energy(v_scaled, masses)
    V = compute_potential_energy(pos_com, masses, G=G)
    Q_virial = T / jnp.abs(V)

    return ClusterIC(
        positions=pos_com, velocities=v_scaled, masses=masses, field=field,
        Q_virial=Q_virial, tail_star_fraction=f_tail,
        collapse_eligible_fraction=f_elig, placement_n_eff=n_eff,
    )
