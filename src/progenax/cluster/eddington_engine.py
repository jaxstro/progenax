# progenax/src/progenax/cluster/eddington_engine.py
"""Engine B state builder: prescribed densities -> shared-Psi Eddington DFs.

The Engine B construction pipeline (design doc 2026-06-10, decisions 1-3):
derive the domain (`derive_r_t`), integrate the prescribed total density to the
shared relative potential (`shared_potential`), then Eddington-invert EACH
component's density in that shared Psi (`eddington_invert`, optionally
Osipkov-Merritt via per-component r_a_j). The result is one `_EngineBState`
field group stored on `MultiComponentCluster`.

Realizability gate (design decision 3, refuse loudly): a genuinely negative
per-component DF (min f_j < -1e-3 * max|f_j|, separating physics from grid
ringing) means the prescribed component CANNOT exist as an equilibrium in this
shared potential. On concrete inputs we raise ValueError naming the component
and the remedy; traced builds (grad/jit) necessarily skip the raise but ALWAYS
store the `f_min_j` diagnostic -- the two-tier pattern EFFVelocityDF uses.
Never clamp silently: a clamped DF integrates back to a *different* density
than the one prescribed.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxtyping import PRNGKeyArray

from progenax.kinematics.eddington import (
    assign_om_directions,
    eddington_invert,
    sample_speed_from_f_table,
)
from progenax.profiles.density_poisson import (
    _density_and_derivative,
    derive_r_t,
    shared_potential,
)

# Remedy text from the design doc (decision 3) -- quoted in the gate's ValueError.
_REMEDY = ("density too shallow to be supported in this shared potential -- "
           "steepen it, raise its mass fraction, or raise r_a_j")

# Genuine-negativity threshold in RELATIVE units (min f / max |f|); anything
# above this is grid-level quadrature ringing, clamped harmlessly at sampling.
_F_MIN_GENUINE = -1e-3


def _is_concrete(x) -> bool:
    """True iff x is a concrete value (codebase idiom, cf. king.py auto-sizing)."""
    try:
        float(x)
        return True
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError,
            TypeError):
        return False


class _EngineBState(eqx.Module):
    """Engine B tables + diagnostics (one field group on MultiComponentCluster).

    Attributes:
        Psi_grid: shared relative potential on the model's `_r_grid` (the
            constructor interpolates the full-resolution Poisson Psi onto the
            downsampled position grid so the sampler can do one jnp.interp).
        E_grid: shared energy grid linspace(1e-4 Psi0, 0.999 Psi0, n_e) -- ONE
            grid for all components (they share Psi0 by construction).
        f_j_grid: per-component Eddington DFs f_j(E) on E_grid, RAW (unclamped;
            the speed sampler clamps grid ringing at use). Amplitudes carry the
            mass-fraction normalization of rho_j (sum_j M_j = 1, G = 1).
        mass_fractions: M_j / M_total amplitudes (design decision 5).
        r_a_j: per-component Osipkov-Merritt anisotropy radii (inf = isotropic,
            the Engine A convention).
        mu: velocity-scale integral sum_j int rho_j r^2 dr (the EFF
            kappa = G M_total / (4 pi mu) pattern). In Engine B mu is
            IDENTICALLY M_tot_dimensionless / (4 pi) = 1/(4 pi): each rho_j is
            mass-normalized to its mass fraction and the fractions sum to 1, so
            no model-parameter dependence flows through mu (kept as a field for
            convention symmetry with EFFVelocityDF, not as a free quantity).
        f_min_j: realizability margin min(f_j) / max|f_j| per component
            (diagnostic; gate threshold -1e-3).
        trunc_frac_j: M_j(<r_t)/M_j(inf) per component (diagnostic; see
            SharedPotential.trunc_frac_j for the EFF gamma <= 3 convention).
        r_poisson, Psi_poisson, dPsi_dr_poisson: FULL-resolution Poisson-grid
            copies (radial grid, shared Psi, dPsi/dr = -M(<r)/r^2) stored for
            the exact-quadrature Q_j oracle (`engine_b_component_virials`) --
            the oracle must integrate the SAME potential the DFs were built
            in, never a re-interpolation of the downsampled sampler grids.
        rho_j_poisson: the prescribed mass-fraction-normalized per-component
            densities on the same grid (density-fidelity diagnostics: compare
            against the DF-side rho_DF,j = 4 pi int w^2 f_j dw to see the
            truncation-edge offset the ergodic DF cannot carry).
        r_t_provenance: static string naming what set the domain (derive_r_t).
    """

    Psi_grid: Float[Array, "n_r"]
    E_grid: Float[Array, "n_e"]
    f_j_grid: Float[Array, "n_comp n_e"]
    mass_fractions: Float[Array, "n_comp"]
    r_a_j: Float[Array, "n_comp"]
    mu: Float[Array, ""]
    f_min_j: Float[Array, "n_comp"]
    trunc_frac_j: Float[Array, "n_comp"]
    r_poisson: Float[Array, "n_r_full"]
    Psi_poisson: Float[Array, "n_r_full"]
    dPsi_dr_poisson: Float[Array, "n_r_full"]
    rho_j_poisson: Float[Array, "n_comp n_r_full"]
    r_t_provenance: str = eqx.field(static=True)


def build_engine_b_state(profiles, mass_fractions, r_a_j, r_t, f_enc,
                         n_r: int, n_e: int):
    """Construct the Engine B state: domain -> shared Psi -> per-component DFs.

    Returns ``(state, pot)`` where `state` is the `_EngineBState` (Psi_grid on
    the FULL Poisson r-grid; the model constructor re-interpolates it onto the
    downsampled `_r_grid`) and `pot` the `SharedPotential` (carries r_grid and
    M_cum_j for the position-CDF assembly).

    Raises ValueError (concrete inputs only) when a component's DF is genuinely
    negative -- the prescribed mix does not exist as an equilibrium. Traced
    builds skip the raise but always store the `f_min_j` diagnostic.
    """
    mass_fractions = jnp.asarray(mass_fractions, dtype=jnp.float64)
    r_a_j = jnp.asarray(r_a_j, dtype=jnp.float64)

    r_t_arr, prov = derive_r_t(profiles, mass_fractions, r_t=r_t, f_enc=f_enc)
    pot = shared_potential(profiles, mass_fractions, r_t_arr, n_r=n_r,
                           r_t_provenance=prov)
    r = pot.r_grid

    # Per-component Eddington inversion in the SHARED Psi (Python loop over the
    # small static component count -- Engine A precedent). Densities re-derive
    # the analytic d rho/dr and rescale to the same mass-fraction amplitude as
    # pot.rho_j_grid (the trapezoid total equals the cumtrap total exactly).
    E_ref = None
    f_rows, f_min_rows = [], []
    for j, p in enumerate(profiles):
        rho_hat, drho_hat = _density_and_derivative(p, r)
        m_hat = 4.0 * jnp.pi * jnp.trapezoid(rho_hat * r**2, r)
        scale = mass_fractions[j] / m_hat
        E_j, f_j = eddington_invert(r, scale * rho_hat, scale * drho_hat,
                                    pot.Psi_grid, pot.dPsi_dr_grid,
                                    r_a=r_a_j[j], n_e=n_e)
        # ONE shared E_grid: all components share Psi0 in one potential, so the
        # per-component grids are the same computation -- assert identity on
        # concrete builds (a mismatch would mean the shared-Psi contract broke).
        if E_ref is None:
            E_ref = E_j
        else:
            diff = jnp.max(jnp.abs(E_j - E_ref))
            if _is_concrete(diff) and float(diff) != 0.0:
                # raise (not assert: assert is stripped under -O) -- a mismatch
                # means the shared-Psi contract broke, never a tolerance issue.
                raise ValueError(
                    f"per-component E_grids must be identical (one shared Psi0); "
                    f"component {j} differs by {float(diff):.3e}")
        f_rows.append(f_j)
        f_min_rows.append(jnp.min(f_j) / jnp.max(jnp.abs(f_j)))

    f_min_j = jnp.stack(f_min_rows)

    # GENUINE-negativity gate (design decision 3): concrete inputs refuse loudly.
    for j in range(len(profiles)):
        fm = f_min_j[j]
        if _is_concrete(fm) and float(fm) < _F_MIN_GENUINE:
            raise ValueError(
                f"Engine B realizability failure: component {j} "
                f"({type(profiles[j]).__name__}) has a genuinely negative "
                f"Eddington DF (min f / max|f| = {float(fm):.3e} < "
                f"{_F_MIN_GENUINE:g}). This component's {_REMEDY}."
            )

    state = _EngineBState(
        Psi_grid=pot.Psi_grid,
        E_grid=E_ref,
        f_j_grid=jnp.stack(f_rows),
        mass_fractions=mass_fractions,
        r_a_j=r_a_j,
        mu=pot.mu,
        f_min_j=f_min_j,
        trunc_frac_j=pot.trunc_frac_j,
        r_poisson=pot.r_grid,
        Psi_poisson=pot.Psi_grid,
        dPsi_dr_poisson=pot.dPsi_dr_grid,
        rho_j_poisson=pot.rho_j_grid,
        r_t_provenance=prov,
    )
    return state, pot


def engine_b_star_velocities(
    state: _EngineBState,
    speed_keys: PRNGKeyArray,
    k_vdir: PRNGKeyArray,
    pos: Float[Array, "N 3"],
    radii: Float[Array, "N"],
    r_grid: Float[Array, "n_grid"],
    c,
    G,
    M_sampled: Float[Array, ""],
) -> Float[Array, "N 3"]:
    """Engine B speed + direction stage of the cluster sampler (jit-safe).

    Per star i with component c_i: Psi_i by ONE jnp.interp on the sampler grid
    (state.Psi_grid was re-interpolated onto `r_grid` at assembly), a
    dimensionless speed s_i ~ s^2 f_{c_i}(Psi_i - s^2/2) from the component's
    Eddington table, then the physical scale and the OM stretched split with
    the PER-STAR anisotropy radius r_a_j[c_i] (inf rows -> stretch 1 ->
    isotropic).

    Velocity-scale convention (the EFF kappa = G M_total/(4 pi mu) pattern):
    the tables are built with G = 1, physical lengths, and total dimensionless
    mass 1, so rho_phys = M_sampled * rho_dimless and Psi_phys = G M_sampled *
    Psi_dimless; hence v = sqrt(G * M_sampled / (4 pi mu)) * s with
    4 pi mu == 1 identically (mass-normalized densities -- see _EngineBState.mu).
    M_sampled = sum_i m_i is the ACTUAL sampled stellar mass, NEVER an input
    M_total (the Engine A lesson: energies must use the realized cluster mass).
    """
    Psi_i = jnp.interp(radii, r_grid, state.Psi_grid)
    s = jax.vmap(
        lambda kk, pp, cc: sample_speed_from_f_table(kk, pp, state.E_grid,
                                                     state.f_j_grid[cc])
    )(speed_keys, Psi_i, c)
    speeds = jnp.sqrt(G * M_sampled / (4.0 * jnp.pi * state.mu)) * s
    return assign_om_directions(k_vdir, pos, speeds, state.r_a_j[c])


def engine_b_component_virials(state: _EngineBState,
                               n_w: int = 400) -> Float[Array, "n_comp"]:
    """EXACT-quadrature per-component virial ratios Q_j = T_j/|W_j| (Engine B).

    The bias-free equilibrium oracle -- independent of every sampled quantity.
    On the full-resolution Poisson grid, per component j:

        m0_j(r) = int w^2 f_j(Psi - w^2/2) dw,   m2_j(r) = int w^4 f_j dw,
        rho_DF,j(r) = 4 pi m0_j(r) / (1 + r^2/r_a_j^2),
        <v^2>_j(r) = (m2_j/m0_j) * (1/3 + (2/3) / (1 + r^2/r_a_j^2)),
        T_j = (1/2) int rho_DF,j <v^2>_j 4 pi r^2 dr,
        W_j = -int rho_DF,j (dPhi/dr) r 4 pi r^2 dr   (Clausius, shared Phi),
        dPhi/dr = -dPsi/dr = +M(<r)/r^2.

    w is the STRETCHED-frame speed: the OM substitution w_t = v_t * sqrt(1 +
    r^2/r_a^2) makes the conditional speed weight isotropic w^2 f_j(Psi - w^2/2)
    -- exactly the frame `sample_speed_from_f_table` + `assign_om_directions`
    draw in -- so <v_r^2> = <w^2>/3, <v_t^2> = (2/3)<w^2>/(1 + r^2/r_a^2), and
    the velocity-space measure carries d^3v = d^3w/(1 + r^2/r_a^2) (the
    rho_DF denominator); r_a = inf gives the isotropic limits exactly.

    TRUNCATION-CONSISTENT weighting (load-bearing): both integrals use
    rho_DF,j -- the density the DF actually represents -- NOT the prescribed
    rho_j. A sharply truncated prescribed density has rho_j(r_t) > 0, an edge
    offset NO ergodic f(E) can carry (the Eddington pair represents
    rho(Psi) - rho(0)), so weighting by the prescribed rho_j mixes that
    unrepresentable offset into T_j and reads ~0.495 for a hard-truncated halo
    -- genuine edge physics (the documented truncated-empirical-profile
    approximation, gated at 0.02 on the SAMPLED global Q), not a table bug.
    With rho_DF the steady-state identity 2T_j + W_j = 4 pi r_t^3 p_j(r_t) = 0
    holds for ANY f(Q)-of-integral table (p_j(r_t) = 0), so Q_j = 0.5 to pure
    quadrature accuracy: any inversion/interpolation/Psi-consistency bug in the
    tables breaks it. Density fidelity (rho_DF vs prescribed rho_j in the
    interior) is gated separately by the sampled-density KS test and the
    Task 5 cross-engine anchors. f is clamped at 0 inside the moments exactly
    as the speed sampler clamps grid ringing. Q_j is dimensionless: the
    velocity scale and the total mass cancel in T_j/|W_j|.
    """
    r, Psi = state.r_poisson, state.Psi_poisson
    dphi_dr = -state.dPsi_dr_poisson           # = +M(<r)/r^2 (G=1)
    Psi_safe = jnp.maximum(Psi, 1e-12)

    def moments(Psi_r, f_row):
        w = jnp.linspace(0.0, jnp.sqrt(2.0 * Psi_r), n_w)
        f_at = jnp.maximum(jnp.interp(Psi_r - 0.5 * w**2, state.E_grid, f_row), 0.0)
        return jnp.trapezoid(w**2 * f_at, w), jnp.trapezoid(w**4 * f_at, w)

    Qs = []
    for j in range(state.f_j_grid.shape[0]):
        m0, m2 = jax.vmap(lambda P: moments(P, state.f_j_grid[j]))(Psi_safe)
        finite = jnp.isfinite(state.r_a_j[j])
        ra_safe = jnp.where(finite, state.r_a_j[j], 1.0)
        inv_st2 = jnp.where(finite, 1.0 / (1.0 + (r / ra_safe) ** 2), 1.0)
        rho_df = jnp.where(Psi > 0.0, 4.0 * jnp.pi * m0 * inv_st2, 0.0)
        v2 = (m2 / (m0 + 1e-300)) * (1.0 / 3.0 + (2.0 / 3.0) * inv_st2)
        T = jnp.trapezoid(0.5 * rho_df * v2 * 4.0 * jnp.pi * r**2, r)
        W = jnp.trapezoid(-rho_df * r * dphi_dr * 4.0 * jnp.pi * r**2, r)
        Qs.append(T / jnp.abs(W))
    return jnp.stack(Qs)


def assemble_engine_b_fields(profiles, mass_fractions, m_j, r_a_j=None,
                             r_t=None, f_enc: float = 0.995, n_r: int = 6000,
                             n_e: int = 1000, n_grid: int = 1000) -> dict:
    """Build the complete MultiComponentCluster field dict for an Engine B model.

    SHARED sampler fields are filled meaningfully: `_r_grid` (the Poisson grid
    downsampled to n_grid), `_cdf_j` (per-component M_j(<r)/M_j(r_t) interpolated
    onto it), `N_frac_j` (proportional to mass_fractions/m_j), `r_t`, `m_j`,
    `residual=0`; `engine_b.Psi_grid` is re-interpolated onto `_r_grid` so the
    sampler does one jnp.interp per star. Engine-A-ONLY fields are NaN tripwires
    (W0, g, r_c, mu_tot scalars; alpha_j/w_j/ra_hat_j arrays; xi_grid/psi_grid
    shape-minimal length-2 arrays): accidental A-path use must poison results
    visibly, never silently.
    """
    mass_fractions = jnp.asarray(mass_fractions, dtype=jnp.float64)
    m_arr = jnp.asarray(m_j, dtype=jnp.float64)
    n_comp = len(profiles)
    ra_arr = (jnp.full((n_comp,), jnp.inf, dtype=jnp.float64) if r_a_j is None
              else jnp.asarray(r_a_j, dtype=jnp.float64))

    state, pot = build_engine_b_state(profiles, mass_fractions, ra_arr,
                                      r_t, f_enc, n_r, n_e)

    r_grid = jnp.linspace(0.0, pot.r_t, n_grid)
    cdf_full = pot.M_cum_j / (pot.M_cum_j[:, -1:] + 1e-30)
    cdf_j = jax.vmap(lambda row: jnp.interp(r_grid, pot.r_grid, row,
                                            left=0.0))(cdf_full)
    state = eqx.tree_at(lambda s: s.Psi_grid, state,
                        jnp.interp(r_grid, pot.r_grid, pot.Psi_grid))
    N_frac = (mass_fractions / m_arr) / jnp.sum(mass_fractions / m_arr)

    nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
    nan_j = jnp.full((n_comp,), jnp.nan, dtype=jnp.float64)
    nan_2 = jnp.full((2,), jnp.nan, dtype=jnp.float64)
    return dict(
        # A-only fields: NaN tripwires (never silently usable in B mode).
        W0=nan, g=nan, r_c=nan, mu_tot=nan,
        alpha_j=nan_j, w_j=nan_j, ra_hat_j=nan_j,
        xi_grid=nan_2, psi_grid=nan_2,
        # Shared fields, filled meaningfully.
        r_t=jnp.asarray(pot.r_t, dtype=jnp.float64), m_j=m_arr,
        N_frac_j=N_frac, residual=jnp.zeros((), dtype=jnp.float64),
        _r_grid=r_grid, _cdf_j=cdf_j,
        # Engine dispatch: is_aniso is the A sampler's static switch only;
        # Engine B dispatches on `engine` (Task 4) and carries r_a_j itself.
        is_aniso=False, engine="B", engine_b=state,
    )


__all__ = [
    "_EngineBState",
    "assemble_engine_b_fields",
    "build_engine_b_state",
    "engine_b_component_virials",
    "engine_b_star_velocities",
]
