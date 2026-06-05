"""
IC builders for progenax.

Provides:
- ICResult: Dataclass for initial condition outputs
- build_spatial_ic: Generic builder using profile + velocity DF
- to_com_frame: Center-of-mass frame transformation
- virial_scale: Scale velocities to target virial ratio
- compute_stellar_radii: Estimate stellar radii from mass
- compute_kinetic_energy, compute_potential_energy: Energy helpers
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from .protocols import SpatialProfile, VelocityDF
# Single canonical energy implementation lives in dynamics.virial; re-export it
# here so the public API (progenax.compute_*_energy) and virial_scale share one
# gradient-safe source of truth (Batch 0, F1+F2).
from .dynamics.virial import compute_kinetic_energy, compute_potential_energy
from .binaries import resolve_binary_components

# Seconds in one (SI) day — exact; used to convert sampled periods (days) into the
# code time unit via units.time_scale_cgs.
_SECONDS_PER_DAY = 86400.0


# =============================================================================
# Population-size budget targets (Batch 4k)
# =============================================================================


@dataclass(frozen=True)
class Systems:
    """Target a fixed number of stellar *systems* (singles + binaries).

    Companions are **not** counted toward the count — the observational convention of
    Rosen, *Confidently Wrong* (``N`` = observed systems; primaries from the IMF,
    companions attached on top, so total stars = ``n + n_binary``). The only
    **fixed-shape => differentiable** target (supports the masked ``compact=False`` path).
    """

    n: int


@dataclass(frozen=True)
class Stars:
    """Target a fixed number of resolved *stars* (primaries + real secondaries).

    Companions count toward the total — the dynamical-IC convention of McLuster
    (Kuepper+2011 draws ``N`` stars then forms ``N*b/2`` binaries). Draw whole systems
    in draw order until the resolved star count first reaches ``n`` (overshoot <= 1 star
    — a binary is never split, so the result is ``n`` or ``n+1`` stars). The
    data-dependent system count makes this **eager only** (``compact=True``).
    """

    n: int


@dataclass(frozen=True)
class TotalMass:
    """Target a fixed total stellar *mass* Σ(m1+m2) [M_sun] (companions counted).

    Whole-system, McLuster-style mass filling: draw until the cumulative system
    mass first reaches ``m`` (overshoot ≤ one system). **Eager only** (``compact=True``).
    """

    m: float


def _system_star_counts(is_binary: Bool[Array, "N"]) -> Int[Array, "N"]:
    """Resolved stars contributed by each system: 1 (single) or 2 (binary)."""
    return 1 + is_binary.astype(jnp.int32)


def _target_system_mask(
    target, is_binary: Bool[Array, "N"], system_masses: Float[Array, "N"]
) -> Bool[Array, "N"]:
    """Prefix keep-mask over systems (draw order) for a budget target.

    Keeps whole systems until the budget is first reached (the crossing system is
    included), so star/mass budgets overshoot by at most one system. `Systems(n)`
    keeps the first ``n`` slots (all of them when the draw is exactly ``n``).
    """
    n_sys = is_binary.shape[0]
    idx = jnp.arange(n_sys)
    if isinstance(target, Systems):
        return idx < target.n
    if isinstance(target, Stars):
        stars = _system_star_counts(is_binary)
        cum_before = jnp.cumsum(stars) - stars  # exclusive prefix sum
        return cum_before < target.n
    if isinstance(target, TotalMass):
        cum_before = jnp.cumsum(system_masses) - system_masses
        return cum_before < target.m
    raise TypeError(f"Unknown population target: {target!r}")


def _target_satisfied(
    target, is_binary: Bool[Array, "N"], system_masses: Float[Array, "N"]
) -> bool:
    """Whether a draw of these systems is large enough to fill the target."""
    if isinstance(target, Systems):
        return is_binary.shape[0] >= target.n
    if isinstance(target, Stars):
        return int(jnp.sum(_system_star_counts(is_binary))) >= target.n
    if isinstance(target, TotalMass):
        return float(jnp.sum(system_masses)) >= target.m
    raise TypeError(f"Unknown population target: {target!r}")


@dataclass(frozen=True)
class ICResult:
    """
    Result from initial conditions generation — pure physical state.

    Immutable dataclass containing all particle data. No dependency on gravax;
    can be converted to any state format.

    Softening is intentionally **NOT** stored here: it is a force-model /
    integration choice (selected on the integrator, e.g. ε=0 for collisional
    Hermite/IAS15), not a property of the initial conditions.

    Attributes:
        positions: Particle positions (N, 3) [length units]
        velocities: Particle velocities (N, 3) [velocity units]
        masses: Particle masses (N,) [M_sun]
        stellar_radii: Stellar radii (N,) [R_sun]
        ids: Particle IDs (N,) or None
        primordial_system_id: (N,) int — which primordial system each particle
            belongs to (paired particles share an id); None for single-only ICs.
            **PROVENANCE at t=0 only** — goes stale under dynamical evolution
            (ionization / formation / exchange). Measure the *current* binary
            population with `binaries.diagnostics.find_bound_pairs`, not this.
        is_primordial_secondary: (N,) bool — True for the secondary of a
            primordial binary; None for single-only ICs.
    """

    positions: Float[Array, "N 3"]
    velocities: Float[Array, "N 3"]
    masses: Float[Array, "N"]
    stellar_radii: Float[Array, "N"]
    ids: Optional[Float[Array, "N"]] = None
    primordial_system_id: Optional[Int[Array, "N"]] = None
    is_primordial_secondary: Optional[Bool[Array, "N"]] = None


def compute_stellar_radii(masses: Float[Array, "N"]) -> Float[Array, "N"]:
    """
    Estimate stellar radii from mass (main-sequence + brown dwarfs).

    Returns radii in SOLAR RADII (R☉).

    R/R☉ relations (3 regimes):
    - M > 1 M☉: R ∝ M^0.8 (massive stars)
    - 0.08 ≤ M ≤ 1 M☉: R ∝ M^0.57 (low-mass main sequence)
    - M < 0.08 M☉: R ∝ M^0.08 (brown dwarfs, ~0.1 R☉)

    Args:
        masses: Particle masses (N,) [M☉]

    Returns:
        Radii in R☉
    """

    def radius_high_mass(m):
        return jnp.power(m, 0.8)

    def radius_low_mass(m):
        return jnp.power(m, 0.57)

    def radius_brown_dwarf(m):
        return 0.1 * jnp.power(m / 0.08, 0.08)

    radii = jax.vmap(
        lambda m: jax.lax.cond(
            m > 1.0,
            radius_high_mass,
            lambda mv: jax.lax.cond(
                mv >= 0.08, radius_low_mass, radius_brown_dwarf, mv
            ),
            m,
        )
    )(masses)

    return radii


def to_com_frame(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
) -> tuple[Float[Array, "N 3"], Float[Array, "N 3"]]:
    """
    Transform to center-of-mass frame.

    Args:
        positions: Particle positions (N, 3)
        velocities: Particle velocities (N, 3)
        masses: Particle masses (N,)

    Returns:
        (positions_com, velocities_com): Transformed coordinates
    """
    M_total = jnp.sum(masses)

    # COM position and velocity
    r_com = jnp.sum(positions * masses[:, None], axis=0) / M_total
    v_com = jnp.sum(velocities * masses[:, None], axis=0) / M_total

    return positions - r_com, velocities - v_com


def virial_scale(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    Q_target: float,
    G: float,
    softening: float = 0.0,
) -> Float[Array, "N 3"]:
    """
    Scale velocities to achieve target virial ratio Q = T/|V|.

    Physical interpretation:
        - Q = 0.5: Virial equilibrium (2T + V = 0)
        - Q < 0.5: Sub-virial (cold), system will collapse
        - Q > 0.5: Super-virial (hot), system will expand/unbind

    Args:
        positions: Particle positions (N, 3)
        velocities: Particle velocities (N, 3)
        masses: Particle masses (N,)
        Q_target: Target virial ratio (0.5 for equilibrium)
        G: Gravitational constant
        softening: Softening length (default: 0)

    Returns:
        Scaled velocities

    References:
        Goodwin & Whitworth (2004) A&A 413, 929 - Sub-virial clusters
        Baumgardt & Kroupa (2007) MNRAS 380, 1589 - Cluster dissolution
    """
    T = compute_kinetic_energy(velocities, masses)
    V = compute_potential_energy(positions, masses, G=G, softening=softening)

    # Q = T / |V| (NOT 2T / |V|)
    Q_current = T / jnp.abs(V)
    scale = jnp.sqrt(Q_target / Q_current)

    return velocities * scale


def build_spatial_ic(
    profile: SpatialProfile,
    masses: Float[Array, "N"],
    velocity_df: VelocityDF,
    key: PRNGKeyArray,
    G: float,
    Q: Optional[float] = 0.5,
    softening: float = 0.0,
    id_offset: int = 0,
) -> ICResult:
    """
    Build initial conditions from spatial profile and velocity DF.

    Args:
        profile: Spatial density profile (must implement SpatialProfile protocol)
        masses: Particle masses (N,) [M_sun]
        velocity_df: Velocity distribution function (must implement VelocityDF protocol)
        key: JAX random key
        G: Gravitational constant (REQUIRED - no default)
        Q: Virial ratio target Q = T/|V| (0.5 for equilibrium, None to disable)
        softening: Softening length used ONLY to virial-scale the IC under the same
            force law it will be integrated with (default: 0.0 = exact Newtonian,
            matching the analytic equilibrium DFs and collisional integration).
            Pass ε>0 for a collisionless science case so the IC is virialized
            consistently. This is a force-model knob (a future shared `ForceModel`
            will supply it); it is **not** stored on the returned `ICResult`.
        id_offset: Offset for particle IDs (default: 0)

    Returns:
        ICResult (pure physical state — no softening field)
    """
    N = masses.shape[0]

    # Split keys
    key_pos, key_vel = jax.random.split(key)

    # 1. Sample positions
    positions = profile.sample_positions(masses, key_pos)

    # 2. Sample velocities (G must be threaded through — see audit C1; without it
    #    the DF silently falls back to DEFAULT_UNITS.G and ignores the caller's units)
    velocities = velocity_df.sample_velocities(positions, masses, key_vel, G=G)

    # 3. Compute stellar radii
    stellar_radii = compute_stellar_radii(masses)

    # 4. Transform to COM frame
    positions, velocities = to_com_frame(positions, velocities, masses)

    # 5. Apply virial scaling under the given force law (ε=0 by default — exact
    #    Newtonian, consistent with the analytic equilibrium DFs).
    if Q is not None:
        velocities = virial_scale(positions, velocities, masses, Q, G, softening)

    # 6. Generate IDs
    ids = jnp.arange(id_offset, id_offset + N) if id_offset != 0 else None

    return ICResult(
        positions=positions,
        velocities=velocities,
        masses=masses,
        stellar_radii=stellar_radii,
        ids=ids,
    )


_MASS_PRESAMPLE = 8192  # systems presampled to estimate mean system mass for a TotalMass budget.


def _index_companions(comp, keep: Bool[Array, "N"]):
    """Boolean-index every field of a `CompanionElements` (eager budget cut)."""
    return jax.tree_util.tree_map(lambda x: x[keep], comp)


def _concat_companions(a, b):
    """Concatenate two `CompanionElements` along axis 0 (eager TotalMass top-up)."""
    return jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y]), a, b)


def _draw_systems_for_target(target, primary_imf, companion_model, key, *, G, day):
    """Draw enough systems to fill `target` (in draw order).

    `Systems(n)` / `Stars(n)` are filled by a single draw of `n` systems (>=1 star
    per system guarantees >= n stars). `TotalMass(M)` over-draws from a mean-system-
    mass estimate and tops up until the mass budget is reached.

    Returns (m1, is_binary, CompanionElements).
    """
    if isinstance(target, (Systems, Stars)):
        km, kc = jax.random.split(key)
        m1 = primary_imf.sample(km, target.n)
        is_binary, comp = companion_model.sample(kc, m1, G=G, day_in_time_units=day)
        return m1, is_binary, comp

    if isinstance(target, TotalMass):
        key, kp, kc = jax.random.split(key, 3)
        m1_pre = primary_imf.sample(kp, _MASS_PRESAMPLE)
        _, comp_pre = companion_model.sample(kc, m1_pre, G=G, day_in_time_units=day)
        mbar = float(jnp.mean(m1_pre + comp_pre.m2))
        n = int(1.5 * target.m / max(mbar, 1e-12)) + 64
        m1_all = is_binary_all = comp_all = None
        while True:
            key, km, kc = jax.random.split(key, 3)
            m1 = primary_imf.sample(km, n)
            is_binary, comp = companion_model.sample(kc, m1, G=G, day_in_time_units=day)
            if m1_all is None:
                m1_all, is_binary_all, comp_all = m1, is_binary, comp
            else:
                m1_all = jnp.concatenate([m1_all, m1])
                is_binary_all = jnp.concatenate([is_binary_all, is_binary])
                comp_all = _concat_companions(comp_all, comp)
            if _target_satisfied(target, is_binary_all, m1_all + comp_all.m2):
                return m1_all, is_binary_all, comp_all
            n = max(n // 2, 64)  # smaller top-up batches

    raise TypeError(f"Unknown population target: {target!r}")


def build_binary_cluster(
    profile: SpatialProfile,
    velocity_df: VelocityDF,
    primary_imf,
    companion_model,
    target,
    key: PRNGKeyArray,
    *,
    units,
    Q: Optional[float] = 0.5,
    softening: float = 0.0,
    compact: bool = True,
):
    """Assemble a star cluster with a primordial binary population (SoTA composition).

    Composes five independent axes: `profile` x `velocity_df` (spatial), `primary_imf`
    (the **primary**-star IMF — vary alpha freely), `companion_model` (the single owner
    of the binary statistics: f_b -> is_binary AND q -> m2, P -> a, e, orientation), and
    `target` (what the population size holds fixed).

    Pipeline: draw `target`-many systems -> `companion_model.sample` -> system masses
    `m1 + m2` -> budget cut -> `build_spatial_ic` on the system masses (COMs virialized
    treating binaries as point masses, eps=0 by default) -> `resolve_binary_components`
    places each binary's two components around its COM (COM preserved exactly).

    **Conventions.** `primary_imf` is the IMF of *primaries*; companions are generated
    conditionally (`m2 = q*m1`, with `q | M1` from the companion model), so the all-stars
    mass function is a *derived* consequence — not the input IMF (Rosen, *Confidently
    Wrong*, S9.6; the conditional `q | M1` parameterization follows Moe & Di Stefano 2017).
    The COM virialization treats each binary as a single CoM particle and replaces it with
    its two constituents only at the end (the McLuster convention, Kuepper+2011 SA8);
    internal binary binding energy is a separate reservoir untouched by `Q` (measure it
    with `binaries.diagnostics`).

    Args:
        profile, velocity_df: spatial profile + velocity DF for the system COMs.
        primary_imf: primary-star IMF with `sample(key, n) -> m1 [Msun]`.
        companion_model: a `CompanionModel` (e.g. `IndependentCompanions`, `MoeCompanions`)
            owning multiplicity + (q, P, e); `sample(key, m1, *, G, day_in_time_units)
            -> (is_binary, CompanionElements)`. No separate `binary_fraction` arg — f_b
            lives in the model (Moe sets it from the masses).
        target: population-size budget — `Systems(n)` (count systems; companions not
            counted; the only differentiable / `compact=False` target), `Stars(n)` (count
            resolved stars, companions included), or `TotalMass(M)` [Msun]. Stars/TotalMass
            have data-dependent counts and are **eager only** (`compact=True`).
        key: JAX random key.
        units: `UnitSystem` (carries G + the time scale for the day->time-unit conversion).
        Q: system-level virial ratio target (0.5 = equilibrium; None to disable).
        softening: virial-scaling softening for the COM cluster (default 0 = exact; NOT stored).
        compact: True (default) -> eagerly compacted `ICResult`; False -> the masked
            fixed-shape `ResolvedBinaries` (jit/grad-safe; requires a `Systems` target).

    Returns:
        `ICResult` (compact=True) or `ResolvedBinaries` (compact=False).
    """
    G = units.G
    day_in_time_units = _SECONDS_PER_DAY / units.time_scale_cgs

    if not compact and not isinstance(target, Systems):
        raise ValueError(
            "compact=False (masked, differentiable output) requires a Systems(n) target; "
            "Stars/TotalMass have data-dependent counts and are eager-only (compact=True)."
        )

    key_draw, key_spatial = jax.random.split(key)

    # 1. Draw systems for the budget: primaries from the IMF, companions (f_b, q, P, e)
    #    from the companion model (the single owner of the binary statistics).
    m1, is_binary, comp = _draw_systems_for_target(
        target, primary_imf, companion_model, key_draw, G=G, day=day_in_time_units
    )
    system_masses = m1 + comp.m2

    # 2. Budget cut. Systems(n) needs no cut (static shape -> compact=False safe);
    #    Stars/TotalMass keep a whole-system prefix (eager, dynamic shape).
    if not isinstance(target, Systems):
        keep = _target_system_mask(target, is_binary, system_masses)
        m1 = m1[keep]
        is_binary = is_binary[keep]
        comp = _index_companions(comp, keep)
        system_masses = system_masses[keep]

    # 3. System COMs (virialized treating binaries as point masses at COM).
    ic_sys = build_spatial_ic(
        profile, system_masses, velocity_df, key_spatial, G, Q=Q, softening=softening
    )

    # 4. Resolve binaries into the masked 2N representation (COM preserved exactly).
    resolved = resolve_binary_components(
        ic_sys.positions, ic_sys.velocities, m1, comp.m2, is_binary,
        comp.a, comp.e, comp.inc, comp.Omega, comp.omega, comp.M_anom, G=G,
    )

    if not compact:
        return resolved

    # 5. Eager compaction -> real-particle ICResult (one-shot; dynamic shape).
    mask = resolved.is_real
    masses = resolved.masses[mask]
    return ICResult(
        positions=resolved.positions[mask],
        velocities=resolved.velocities[mask],
        masses=masses,
        stellar_radii=compute_stellar_radii(masses),
        ids=jnp.arange(masses.shape[0]),
        primordial_system_id=resolved.primordial_system_id[mask],
        is_primordial_secondary=resolved.is_primordial_secondary[mask],
    )


__all__ = [
    "Systems",
    "Stars",
    "TotalMass",
    "ICResult",
    "compute_stellar_radii",
    "compute_kinetic_energy",
    "compute_potential_energy",
    "to_com_frame",
    "virial_scale",
    "build_spatial_ic",
    "build_binary_cluster",
]
