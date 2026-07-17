r"""End-to-end gravoturbulent initial conditions: stars, and optionally residual gas.

The user-facing assembly that turns natal-turbulence parameters into a complete,
physically-scaled IC (the Phase-4a ``TurbulentCloudIC``), composing the verified pieces:

  1. density construction per ``CloudSpec.coupling``: ``'independent'`` — free-β GRF;
     ``'helmholtz'`` — ONE white field yields the velocity realization AND (via linearized
     continuity, ĝ ∝ −∇·v) the density carrier, so β is DERIVED = β_v − 2 (Phase 3).
     The mass-conserving copula imposes the BM19 marginal either way.
  2. ``apply_spherical_envelope`` — ``s_total = s_turb + ln ρ_env(r)`` (the SHAPE).
  3. star placement per ``CompositionSpec.placement`` (default FK12 multi-freefall,
     ``p_⋆ ∝ w·ρ_total^{3/2}``; legacy ``'two_population'`` ablation).
  4. velocities per ``VelocitySpec.mode``:
     ``'virial_target'`` — sample the coherent field, impose Q via core ``virial_scale``
     (byte-identical legacy path; REFUSES a physical gas — the imposed amplitude has no
     cloud meaning);
     ``'physical'`` (Phase 2, **field-first since Phase 4a**) — the velocity GRID is
     normalized so its volume-weighted rms is σ_g = ℳ·c_s (the FK10 convention: ℳ is a
     GAS parameter), stars sample it ×η_v, and the stellar dispersion is EMERGENT with
     sampling scatter (≈ η_v·ℳ·c_s; gate AC-G7 band). Q_virial is emergent; the BM92
     1-D ``alpha_vir`` is the consistency diagnostic.
  5. optional residual gas per ``GasSpec`` (Phase 4a, Aim 2 handoff): the SAME cloud is
     normalized to M_cl = M⋆/ε_global, partitioned by the local free-fall model
     ε⋆ = 1−exp(−τ⋆w/t_ff) (or the uniform ablation), and carried as
     (ρ_cloud, ρ_residual, velocity, pressure) with an exact mass-closure ledger.
  6. ONE frame: with gas, the joint stars+gas COM/momentum frame; star-only, the stellar
     COM frame — recorded exactly in ``FrameTransform`` either way.

Units are explicit (pass ``G``; CLAUDE.md mandate): lengths in pc, masses in M⊙,
velocities in pc/Myr for STELLAR. Positions/velocities are non-differentiable
(categorical placement); the differentiable interface is the inference layer, the
analytic fraction diagnostics, and the τ⋆ root (IFT).

JAX-native.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxstro.units import UnitSystem
from jaxtyping import Array, Float

from gravoturb.realization.envelope import apply_spherical_envelope
from gravoturb.realization.gas import (
    local_freefall_time,
    normalized_cloud_density,
    partition_star_gas,
    solve_tau_star,
)
from gravoturb.realization.helmholtz import (
    coupled_log_density_gaussian,
    helmholtz_velocity_field,
)
from gravoturb.realization.mass_assignment import correlated_mass_assignment
from gravoturb.realization.pipeline import (
    TurbulentField,
    build_turbulent_field,
    turbulent_field_from_gaussian,
)
from gravoturb.realization.placement import (
    collapse_eligible_fraction,
    collapse_weights,
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
from gravoturb.specs import (
    CloudSpec,
    CompositionSpec,
    GasSpec,
    GeometrySpec,
    VelocitySpec,
    validate_spec_bundle,
)
from gravoturb.theory.collapse_threshold import virial_parameter

# One-way dependency: released progenax never imports gravoturb (enforced by
# tests/experimental/unit/test_core_severance.py), so this import is cycle-free.
from progenax import (
    compute_kinetic_energy,
    compute_potential_energy,
    resolve_binary_components,
    to_com_frame,
    virial_scale,
)


class FrameTransform(NamedTuple):
    """The affine map between the cluster (COM) frame and the realization grid frame.

    The builder samples stars in the grid/box frame ([0, box)³, cell centres at
    (i+0.5)·box/n), then shifts to the COM frame (JOINT stars+gas when gas is built,
    stellar otherwise) and applies one global velocity-amplitude factor. This ledger
    records that transformation exactly, so stars and the carried grids are always
    mutually reconcilable:

    - grid frame:    positions_box = positions + origin
                     (grid cell centres in the cluster frame: (i+0.5)·box/n − origin)
    - velocities:    velocities = velocity_scale · (v_raw(positions_box) − bulk_velocity),
                     where v_raw is the trilinear sample of the (unnormalized) velocity
                     grid and bulk_velocity is in the same raw units.

    Reconstruction holds to machine roundoff (floating point: (x−c)+c is not bitwise x).
    ``velocity_scale`` is exact in both modes because both apply one global scalar
    (0 for Q_target=0 cold ICs).
    """

    origin: Float[Array, "3"]         # COM in grid/box coordinates [pc]
    bulk_velocity: Float[Array, "3"]  # subtracted bulk velocity [raw field units]
    velocity_scale: Float[Array, ""]  # global scalar applied around the bulk


class Stars(NamedTuple):
    """Discrete stellar IC (COM frame; pc, pc/Myr, M⊙ for STELLAR).

    ``system_id`` is the primordial-system provenance (Aim-2 hierarchy state):
    singles get their own id; both components of a primordial binary share one id
    (feeds ``progenax.binary_energy_budget`` and downstream startrax hierarchy)."""

    positions: Float[Array, "n 3"]
    velocities: Float[Array, "n 3"]
    masses: Float[Array, "n"]
    system_id: Array | None = None


class GasState(NamedTuple):
    """Residual-gas grids (grid frame + recorded origin; Phase 4a handoff to gravax).

    ``rho_cloud`` is the normalized parent cloud ρ_cl [M⊙/pc³]; ``rho_residual`` the
    post-partition gas ρ_g,0 = (1−ε⋆)ρ_cl; ``velocity`` the physical coherent gas
    velocity in the joint COM frame [pc/Myr]; ``pressure`` the cold isothermal
    P_g,0 = ρ_g,0·c_s² [M⊙ pc⁻¹ Myr⁻²]; ``cell_volume`` [pc³] makes ∫ρdV = Σρ·dV
    explicit."""

    rho_cloud: Float[Array, "nx ny nz"]
    rho_residual: Float[Array, "nx ny nz"]
    velocity: Float[Array, "nx ny nz 3"]
    pressure: Float[Array, "nx ny nz"]
    cell_volume: Float[Array, ""]


class Fields(NamedTuple):
    """The realized dimensionless fields (diagnostics/provenance)."""

    s_turb: TurbulentField            # BM19 log-density box + scalars (⟨e^s⟩=1)
    s_total: Float[Array, "nx ny nz"]  # s_turb + ln ρ_env (enveloped, unnormalized)
    collapse_weight: Float[Array, "nx ny nz"]  # the smooth eligibility w(s_turb) ∈ [0,1]


class Geometry(NamedTuple):
    """Grid geometry; ``origin`` maps grids into the cluster frame (= frame.origin)."""

    shape: tuple[int, int, int]
    box_size: Float[Array, ""] | float
    origin: Float[Array, "3"]


class Physics(NamedTuple):
    """The resolved physical parameters this realization was built from."""

    mach: Float[Array, ""] | float
    b: Float[Array, ""] | float
    alpha: Float[Array, ""] | float
    beta: Float[Array, ""] | float            # resolved (derived β_v−2 under helmholtz)
    beta_v: Float[Array, ""] | float
    chi: Float[Array, ""] | float | None      # compressive fraction (helmholtz only)
    coupling: str
    velocity_mode: str
    placement: str
    eta_v: Float[Array, ""] | float | None    # physical mode only
    c_s: Float[Array, ""] | float | None      # [km/s]; physical mode only
    sfe_global: Float[Array, ""] | float | None   # gas build only
    tau_star: Float[Array, ""] | None         # local_freefall partition only
    gamma: Float[Array, ""] | float | None    # gas build only


class Ledger(NamedTuple):
    """Conservation accounting + realization diagnostics + provenance."""

    M_star: Float[Array, ""]
    M_cl: Float[Array, ""] | None             # gas build only (= M_star/sfe)
    M_gas: Float[Array, ""] | None            # ∫ρ_g,0 dV (gas build only)
    mass_closure_residual: Float[Array, ""] | None
    #   M_cl − (M_star + ∫ρ_g dV); must be ~roundoff (gate AC-G1)
    total_momentum: Float[Array, "3"]         # stars (+ gas) in the adopted frame
    Q_virial: Float[Array, ""]                # stellar T/|V| (emergent in physical mode)
    alpha_vir: Float[Array, ""]               # BM92 1-D convention on the realized stars
    frame: FrameTransform
    gas_included: bool                        # LOUD star-only label (Anna's guard)
    tail_star_fraction: Float[Array, ""] | None = None   # multi_freefall only
    collapse_eligible_fraction: Float[Array, ""] | None = None
    placement_n_eff: Float[Array, ""] | None = None
    n_binaries: int | None = None                        # companions builds only


class TurbulentCloudIC(NamedTuple):
    """The canonical gravoturbulent IC product (a JAX pytree; Phase 4a).

    ``gas`` is ``None`` for star-only builds (no ``GasSpec``) — a first-class,
    zero-cost path; ``ledger.gas_included`` records it loudly so gas-requiring
    pipelines fail immediately instead of treating a star-only IC as a gas-free cloud.
    """

    stars: Stars
    gas: GasState | None
    fields: Fields
    geometry: Geometry
    physics: Physics
    ledger: Ledger


def build_cluster_ic(
    masses: Float[Array, "n"],
    *,
    cloud: CloudSpec,
    geometry: GeometrySpec,
    velocity: VelocitySpec,
    composition: CompositionSpec,
    G: float,
    key: jax.Array,
    units: UnitSystem | None = None,
    gas: GasSpec | None = None,
) -> TurbulentCloudIC:
    r"""Build a spherical, substructured gravoturbulent IC (stars + optional gas).

    ``masses`` (M⊙, masses-first) sets ``n_stars``; with ``gas=GasSpec(sfe=…)`` the
    parent cloud is normalized to M_cl = Σmᵢ/sfe and the residual gas is carried with
    an exact mass-closure ledger. Gas requires ``velocity.mode='physical'`` (an imposed
    Q has no cloud meaning — loud refusal) and ``units``. Returns
    :class:`TurbulentCloudIC`.
    """
    n_stars = int(masses.shape[0])
    k_field, k_vfield, k_pos = jax.random.split(key, 3)

    # ── boundary validation (ADR-0041 + Phase-4a refusals) ──
    beta, chi = validate_spec_bundle(cloud, velocity)
    if units is not None and abs(float(units.G) - float(G)) > 1e-9 * abs(float(G)):
        raise ValueError(
            f"units and G disagree (units.G={units.G!r}, G={G!r}); pass a "
            f"consistent pair — no silent precedence"
        )
    if velocity.mode == "physical" and units is None:
        raise ValueError(
            "VelocitySpec(mode='physical') requires units=... (a jaxstro UnitSystem, "
            "e.g. STELLAR): c_s is in km/s and must be converted to the G-consistent "
            "velocity unit"
        )
    if gas is not None and velocity.mode != "physical":
        raise ValueError(
            "a physical residual gas requires VelocitySpec(mode='physical'): the "
            "virial_target amplitude is an imposed stellar ablation with no cloud "
            "meaning — refusing to label it gas (design 2026-07-16)"
        )

    # ── density + velocity construction (Phase 3 coupling modes) ──
    if cloud.coupling == "helmholtz":
        # ONE white field drives both (ĝ ∝ −∇·v; no new randomness on the compressive
        # channel). k_vfield intentionally unused (key-stream parity across modes).
        bundle = helmholtz_velocity_field(
            geometry.shape, velocity.beta_v, chi, k_field, return_fourier=True
        )
        field = turbulent_field_from_gaussian(
            coupled_log_density_gaussian(bundle), cloud.mach, cloud.b, cloud.alpha
        )
        v_field = bundle.velocity
    else:  # 'independent' (byte-identical to the pre-Phase-3 pipeline)
        field = build_turbulent_field(
            cloud.mach, cloud.b, cloud.alpha, beta, geometry.shape, k_field
        )
        v_field = turbulent_velocity_field(geometry.shape, velocity.beta_v, k_vfield)

    s_total = apply_spherical_envelope(field.s, geometry.profile, geometry.box_size)
    w = collapse_weights(field.s, field.s_t, composition.mask_sharpness)

    # ── star placement ──
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

    # ── Phase 4b: primordial mass segregation (λ_corr; OFF by default) ──
    if composition.lambda_corr is not None:
        # local density at each star's cell (nearest-cell lookup of the enveloped
        # field). Key comes from fold_in of the ORIGINAL key so the existing 3-way
        # split streams (and the byte-identity pins) are untouched.
        n_ax = jnp.asarray(geometry.shape)
        cell = jnp.clip(
            jnp.floor(positions / geometry.box_size * n_ax).astype(jnp.int32),
            0, n_ax - 1,
        )
        local_rho = jnp.exp(s_total)[cell[:, 0], cell[:, 1], cell[:, 2]]
        masses = correlated_mass_assignment(
            masses, local_rho, composition.lambda_corr,
            jax.random.fold_in(key, 4),
        )

    # ── Phase 4b: companion sampling (barycenter-first contract) ──
    # Input masses are PRIMARIES; companions are drawn per primary, and the SYSTEM
    # masses m_sys = m1 + m2 carry every downstream mass role (barycenter dynamics,
    # frame weights, the gas M_cl contract). Components are split only AFTER the
    # velocity amplitude is applied to the barycenters (ratified design).
    if composition.companions is not None:
        if units is None:
            raise ValueError(
                "CompositionSpec(companions=...) requires units=... (a jaxstro "
                "UnitSystem): orbital periods are drawn in days and must be "
                "converted to the G-consistent time unit"
            )
        day_in_time_units = 86400.0 / units.time_scale_cgs
        is_binary, comp_elems = composition.companions.sample(
            jax.random.fold_in(key, 5), masses,
            G=G, day_in_time_units=day_in_time_units,
        )
        m2 = jnp.where(is_binary, comp_elems.m2, 0.0)
        m_sys = masses + m2
    else:
        is_binary = comp_elems = None
        m_sys = masses

    v_raw = sample_turbulent_velocities(positions, v_field, box_size=geometry.box_size)
    m_star = jnp.sum(m_sys)

    # ── gas construction (Phase 4a; before the frame so the frame can be joint) ──
    if gas is not None:
        M_cl = m_star / gas.sfe
        rho_cl, cell_volume = normalized_cloud_density(
            s_total, geometry.box_size, M_cl
        )
        t_ff = local_freefall_time(rho_cl, G=G)
        if gas.partition == "local_freefall":
            tau_star = solve_tau_star(w, t_ff, rho_cl, cell_volume, gas.sfe)
            _, rho_gas = partition_star_gas(rho_cl, w, t_ff, tau_star)
        else:  # 'uniform' (the controlled ablation): ρ_g = (1−ε)ρ_cl
            tau_star = None
            rho_gas = (1.0 - gas.sfe) * rho_cl
        M_gas = jnp.sum(rho_gas) * cell_volume
    else:
        M_cl = M_gas = rho_cl = rho_gas = cell_volume = tau_star = None

    # ── velocity amplitude + ONE frame ──
    if velocity.mode == "physical":
        # FIELD-FIRST (Phase 4a, ratified): normalize the GRID so its volume-weighted
        # rms is σ_g = ℳ·c_s (FK10 convention — ℳ is a gas parameter); stars sample
        # it ×η_v, so the stellar dispersion is EMERGENT with sampling scatter.
        c_s_internal = velocity.c_s / units.velocity_scale_km_s
        sigma_g = cloud.mach * c_s_internal
        rms_grid = jnp.sqrt(jnp.mean(jnp.sum(v_field**2, axis=-1)))
        grid_scale = sigma_g / rms_grid
        star_scale = velocity.eta_v * grid_scale
        v_star_unshifted = star_scale * v_raw
        if gas is not None:
            # joint stars+gas COM and momentum (one frame — design contract 4)
            grid = _cell_centre_grid(geometry.shape, geometry.box_size)
            m_gas_cells = rho_gas * cell_volume
            M_tot = m_star + M_gas
            origin = (
                jnp.sum(positions * m_sys[:, None], axis=0)
                + jnp.sum(grid * m_gas_cells[..., None], axis=(0, 1, 2))
            ) / M_tot
            v_gas_unshifted = grid_scale * v_field
            bulk = (
                jnp.sum(v_star_unshifted * m_sys[:, None], axis=0)
                + jnp.sum(v_gas_unshifted * m_gas_cells[..., None], axis=(0, 1, 2))
            ) / M_tot
            v_gas = v_gas_unshifted - bulk
        else:
            origin = jnp.sum(positions * m_sys[:, None], axis=0) / m_star
            bulk = jnp.sum(v_star_unshifted * m_sys[:, None], axis=0) / m_star
            v_gas = None
        pos_com = positions - origin
        v_scaled = v_star_unshifted - bulk
        # FrameTransform contract: v = scale·(v_raw − bulk_raw) with bulk in RAW units
        frame = FrameTransform(
            origin=origin,
            bulk_velocity=bulk / star_scale,
            velocity_scale=star_scale,
        )
    else:  # 'virial_target' (byte-identical legacy path; gas refused above)
        pos_com, v_com = to_com_frame(positions, v_raw, m_sys)
        origin = jnp.sum(positions * m_sys[:, None], axis=0) / m_star
        frame_bulk_v = jnp.sum(v_raw * m_sys[:, None], axis=0) / m_star
        v_scaled = virial_scale(pos_com, v_com, m_sys, Q_target=velocity.Q_target, G=G)
        T_raw = compute_kinetic_energy(v_com, m_sys)
        T_scaled = compute_kinetic_energy(v_scaled, m_sys)
        frame = FrameTransform(
            origin=origin, bulk_velocity=frame_bulk_v,
            velocity_scale=jnp.sqrt(T_scaled / T_raw),
        )
        v_gas = None

    # ── Phase 4b: split binary components AFTER the barycenter amplitude ──
    if composition.companions is not None:
        resolved = resolve_binary_components(
            pos_com, v_scaled, masses, m2, is_binary,
            comp_elems.a, comp_elems.e, comp_elems.inc,
            comp_elems.Omega, comp_elems.omega, comp_elems.M_anom, G=G,
        )
        # host-level compaction (the builder is eager): drop single-star ghosts
        real = resolved.is_real
        star_pos = resolved.positions[real]
        star_vel = resolved.velocities[real]
        star_masses = resolved.masses[real]
        star_sysid = resolved.primordial_system_id[real]
        n_binaries = int(jnp.sum(is_binary))
    else:
        star_pos, star_vel, star_masses = pos_com, v_scaled, masses
        star_sysid = jnp.arange(n_stars)
        n_binaries = None

    # ── diagnostics + ledger (on the FINAL star set gravax will integrate) ──
    T = compute_kinetic_energy(star_vel, star_masses)
    V = compute_potential_energy(star_pos, star_masses, G=G)
    Q_virial = T / jnp.abs(V)
    sigma_1d = jnp.sqrt(2.0 * T / (3.0 * m_star))  # BM92/Heyer 1-D convention
    alpha_vir = virial_parameter(
        m_star, _half_mass_radius(star_pos, star_masses), sigma_1d, G=G
    )
    p_stars = jnp.sum(star_vel * star_masses[:, None], axis=0)
    if gas is not None:
        total_momentum = p_stars + jnp.sum(
            v_gas * (rho_gas * cell_volume)[..., None], axis=(0, 1, 2)
        )
        closure = M_cl - (m_star + M_gas)
        c_s_internal = velocity.c_s / units.velocity_scale_km_s
        gas_state = GasState(
            rho_cloud=rho_cl, rho_residual=rho_gas, velocity=v_gas,
            pressure=rho_gas * c_s_internal**2, cell_volume=cell_volume,
        )
    else:
        total_momentum = p_stars
        closure = None
        gas_state = None

    return TurbulentCloudIC(
        stars=Stars(positions=star_pos, velocities=star_vel, masses=star_masses,
                    system_id=star_sysid),
        gas=gas_state,
        fields=Fields(s_turb=field, s_total=s_total, collapse_weight=w),
        geometry=Geometry(shape=geometry.shape, box_size=geometry.box_size,
                          origin=frame.origin),
        physics=Physics(
            mach=cloud.mach, b=cloud.b, alpha=cloud.alpha, beta=beta,
            beta_v=velocity.beta_v, chi=chi, coupling=cloud.coupling,
            velocity_mode=velocity.mode, placement=composition.placement,
            eta_v=velocity.eta_v if velocity.mode == "physical" else None,
            c_s=velocity.c_s, sfe_global=gas.sfe if gas is not None else None,
            tau_star=tau_star, gamma=gas.gamma if gas is not None else None,
        ),
        ledger=Ledger(
            M_star=m_star, M_cl=M_cl, M_gas=M_gas, mass_closure_residual=closure,
            total_momentum=total_momentum, Q_virial=Q_virial, alpha_vir=alpha_vir,
            frame=frame, gas_included=gas is not None,
            tail_star_fraction=f_tail, collapse_eligible_fraction=f_elig,
            placement_n_eff=n_eff, n_binaries=n_binaries,
        ),
    )


def _cell_centre_grid(
    shape: tuple[int, int, int], box_size
) -> Float[Array, "nx ny nz 3"]:
    """Cell-centre coordinates (i+0.5)·box/n — the same convention as radius_grid /
    sample_turbulent_velocities."""
    axes = [
        (jnp.arange(n) + 0.5) * (box_size / n) for n in shape
    ]
    X, Y, Z = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([X, Y, Z], axis=-1)


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
