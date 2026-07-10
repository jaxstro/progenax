r"""End-to-end FDF cluster initial conditions (Build 4 — the forward generative tool).

This is the user-facing assembly that turns natal-turbulence parameters into a complete,
physically-scaled N-body IC, composing the verified subsystem pieces:

  1. ``build_fdf_field`` — BM19 turbulent log-density box ``s_turb`` (carries β, ℳ, α; ⟨e^{s_turb}⟩=1).
  2. ``apply_spherical_envelope`` — add the radial log-envelope of a progenax ``SpatialProfile``:
     ``s_total = s_turb + ln ρ_env(r)`` (the SHAPE: r_h, concentration).
  3. ``sample_positions`` — star positions with the PLACEMENT density ∝ e^{s_total} (centrally
     concentrated) while the dense-tail mask stays on ``s_turb`` against ``s_t`` (substructure
     decoupled from the envelope; see :func:`gravoturb_fdf.field.sampling.sample_cell_indices`).
  4. ``turbulent_velocity_field`` + ``sample_turbulent_velocities`` — coherent turbulent velocities
     (β_v); nearby stars move together (Goodwin & Whitworth 2004).
  5. core ``to_com_frame`` then ``virial_scale`` — centre the cluster and set the velocity amplitude
     to a CHOSEN virial ratio ``Q ≡ T/|V|`` (0.5 virial, <0.5 collapsing, 0.75 super-virial). The
     potential uses the actual positions, so the envelope's deeper well is auto-accounted for.

Units are explicit (pass ``G``; CLAUDE.md mandate): ``box_size`` and the profile's ``r_h`` share length
units (pc for ``STELLAR``), masses are M⊙, and the output velocities come out in the matching system
(pc/Myr for ``STELLAR.G``). Positions/velocities are non-differentiable (categorical placement, spec §8);
the differentiable interface is the inference layer, not this generator.

JAX-native.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gravoturb_fdf.field.envelope import apply_spherical_envelope
from gravoturb_fdf.field.pipeline import FDFField, build_fdf_field
from gravoturb_fdf.field.sampling import sample_positions
from gravoturb_fdf.field.velocity import (
    sample_turbulent_velocities,
    turbulent_velocity_field,
)


class ClusterIC(NamedTuple):
    """A complete FDF cluster IC (a JAX pytree)."""

    positions: Float[Array, "n 3"]    # COM-centred, length units of box_size (pc for STELLAR)
    velocities: Float[Array, "n 3"]   # COM frame, units set by G (pc/Myr for STELLAR)
    masses: Float[Array, "n"]         # M⊙
    field: FDFField                   # realized turbulent field (BM19 provenance / diagnostics)
    Q: Float[Array, ""]               # realized virial ratio T/|V|


def build_cluster_ic(
    masses: Float[Array, "n"],
    *,
    mach: float,
    b: float,
    alpha: float,
    beta: float,
    profile,
    beta_v: float,
    Q_target: float,
    f_sub: float,
    shape: tuple[int, int, int],
    box_size: float,
    G: float,
    key: jax.Array,
    kappa: float = 8.0,
) -> ClusterIC:
    r"""Build a spherical, substructured, virial-scaled FDF cluster IC.

    ``masses`` (M⊙, masses-first per the ecosystem convention) sets ``n_stars = len(masses)``.
    ``profile`` is any progenax ``SpatialProfile`` (the envelope shape; e.g. ``PlummerProfile``),
    in the same length units as ``box_size``. Returns a :class:`ClusterIC`.
    """
    n_stars = int(masses.shape[0])
    k_field, k_vfield, k_pos = jax.random.split(key, 3)

    field = build_fdf_field(mach, b, alpha, beta, shape, k_field)
    s_total = apply_spherical_envelope(field.s, profile, box_size)

    positions = sample_positions(
        field.s, field.s_t, kappa, f_sub, n_stars, k_pos,
        box_size=box_size, s_density=s_total,
    )

    v_field = turbulent_velocity_field(shape, beta_v, k_vfield)
    v_raw = sample_turbulent_velocities(positions, v_field, box_size=box_size)

    from progenax import (
        compute_kinetic_energy,
        compute_potential_energy,
        to_com_frame,
        virial_scale,
    )

    pos_com, v_com = to_com_frame(positions, v_raw, masses)
    v_scaled = virial_scale(pos_com, v_com, masses, Q_target=Q_target, G=G)

    T = compute_kinetic_energy(v_scaled, masses)
    V = compute_potential_energy(pos_com, masses, G=G)
    Q = T / jnp.abs(V)

    return ClusterIC(
        positions=pos_com, velocities=v_scaled, masses=masses, field=field, Q=Q
    )
