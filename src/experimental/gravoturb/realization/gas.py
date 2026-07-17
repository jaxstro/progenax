r"""Physical parent-cloud normalization + local star–gas partition (Phase 4a, Aim 2).

The Aim 2 handoff carries the SAME turbulent cloud realization forward as residual gas
(design addendum 2026-07-16; brain note aim2-coupled-engine.md, "Aim 1 to Aim 2").
This module owns the density-side physics:

1. **Normalization** — the enveloped unnormalized field ρ̃ = e^{s_total} is scaled to a
   physical parent-cloud mass, ρ_cl = M_cl·ρ̃/∫ρ̃dV (discrete cell volume, exact
   closure; invariant to the envelope's arbitrary normalization).
2. **Partition** — the positivity-preserving local star-formation efficiency

       ε⋆(x; τ⋆) = 1 − exp[−τ⋆ w(x)/t_ff(x)],   t_ff = √(3π/32Gρ_cl),

   with w the existing smooth collapse-eligibility weight, gives
   ρ⋆,0 = ε⋆ρ_cl and ρ_g,0 = (1−ε⋆)ρ_cl — pointwise conserving and nonnegative.
   At low efficiency ε⋆ ≃ τ⋆w/t_ff, so ρ⋆,0 ∝ w·ρ_cl^{3/2}: the AC-IC7-gated
   multi-freefall placement law is the low-efficiency limit of this partition,
   not a separate prescription.
3. **Global-SFE closure** — the single scalar τ⋆ solves the monotone constraint
   ∫ρ⋆,0 dV / M_cl = ε_global by FIXED-ITERATION bracketed bisection (``lax.scan``;
   no ``while_loop`` — differentiability mandate) with the implicit-function-theorem
   derivative dτ⋆/dθ = −(∂F/∂θ)/(∂F/∂τ⋆) attached via ``jax.custom_jvp``.

ρ⋆,0 is a SAMPLING INTENSITY only — discrete stars are drawn from it and the
continuous field is never retained as a second gravitating component. Units are
explicit: ρ in M⊙/pc³ with STELLAR G gives t_ff in Myr.

JAX-native; differentiable in (w, t_ff, ρ_cl, sfe_global) through the root.
"""

from functools import partial

import jax
import jax.core
import jax.numpy as jnp
from jaxtyping import Array, Float

# Bisection bracket ceiling and iteration count: τ⋆ = ε_ff·Δt_form is O(0.01–10) Myr
# physically; 2^40 is an intentionally absurd ceiling so the post-solve convergence
# check (not the bracket) is what rejects unreachable SFEs. 120 halvings resolve τ⋆
# to ~1e-24 absolute — far below any physical scale.
_TAU_HI = 2.0**40
_N_ITER = 120


def normalized_cloud_density(
    s_total: Float[Array, "nx ny nz"],
    box_size: Float[Array, ""] | float,
    M_cl: Float[Array, ""] | float,
) -> tuple[Float[Array, "nx ny nz"], Float[Array, ""]]:
    r"""Normalize the enveloped log-density to a physical parent-cloud mass.

    Returns ``(rho_cl, cell_volume)`` with ∫ρ_cl dV = M_cl exact (discrete sum) and
    ρ_cl invariant to any additive constant in ``s_total`` (the envelope's arbitrary
    normalization cancels in ρ̃/Σρ̃)."""
    n_cells = s_total.size
    cell_volume = box_size**3 / n_cells
    rho_tilde = jnp.exp(s_total)
    return M_cl * rho_tilde / (jnp.sum(rho_tilde) * cell_volume), cell_volume


def local_freefall_time(
    rho_cl: Float[Array, "..."], G: float
) -> Float[Array, "..."]:
    r"""t_ff = √(3π/32Gρ) — the local spherical free-fall time (explicit G; Myr for
    STELLAR units with ρ in M⊙/pc³)."""
    return jnp.sqrt(3.0 * jnp.pi / (32.0 * G * rho_cl))


def local_stellar_fraction(
    w: Float[Array, "..."],
    t_ff: Float[Array, "..."],
    tau_star: Float[Array, ""],
) -> Float[Array, "..."]:
    r"""ε⋆ = 1 − exp(−τ⋆ w/t_ff) ∈ [0, 1): positivity-preserving by construction."""
    return -jnp.expm1(-tau_star * w / t_ff)


def partition_star_gas(
    rho_cl: Float[Array, "..."],
    w: Float[Array, "..."],
    t_ff: Float[Array, "..."],
    tau_star: Float[Array, ""],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    r"""(ρ⋆,0, ρ_g,0) = (ε⋆ρ_cl, (1−ε⋆)ρ_cl) — pointwise conserving, nonnegative."""
    eps = local_stellar_fraction(w, t_ff, tau_star)
    return eps * rho_cl, (1.0 - eps) * rho_cl


def _global_sfe(tau, w, t_ff, rho_cl):
    """F(τ) = ∫ε⋆ρ_cl dV / ∫ρ_cl dV (cell volume cancels); monotone ↑ in τ."""
    eps = local_stellar_fraction(w, t_ff, tau)
    return jnp.sum(eps * rho_cl) / jnp.sum(rho_cl)


@partial(jax.custom_jvp, nondiff_argnums=(4,))
def _solve_tau_core(w, t_ff, rho_cl, sfe, n_iter):
    def body(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        below = _global_sfe(mid, w, t_ff, rho_cl) < sfe
        return (jnp.where(below, mid, lo), jnp.where(below, hi, mid)), None

    (lo, hi), _ = jax.lax.scan(
        body, (jnp.asarray(0.0), jnp.asarray(_TAU_HI)), None, length=n_iter
    )
    return 0.5 * (lo + hi)


@_solve_tau_core.defjvp
def _solve_tau_core_jvp(n_iter, primals, tangents):
    # Implicit-function theorem on R(τ, θ) = F(τ; θ) − sfe = 0:
    #   dτ = −(∂R/∂θ · dθ) / (∂R/∂τ)   (the design note's dτ⋆/dθ = −F_θ/F_τ).
    w, t_ff, rho_cl, sfe = primals
    dw, dt_ff, drho, dsfe = tangents
    tau = _solve_tau_core(w, t_ff, rho_cl, sfe, n_iter)
    dR_dtau = jax.grad(_global_sfe, argnums=0)(tau, w, t_ff, rho_cl)
    _, dR_dtheta = jax.jvp(
        lambda w_, t_, r_, s_: _global_sfe(tau, w_, t_, r_) - s_,
        (w, t_ff, rho_cl, sfe),
        (dw, dt_ff, drho, dsfe),
    )
    return tau, -dR_dtheta / dR_dtau


def solve_tau_star(
    w: Float[Array, "..."],
    t_ff: Float[Array, "..."],
    rho_cl: Float[Array, "..."],
    cell_volume: Float[Array, ""] | float,
    sfe_global: Float[Array, ""] | float,
    n_iter: int = _N_ITER,
) -> Float[Array, ""]:
    r"""Solve F(τ⋆) = ε_global for the partition's single scalar (bracketed bisection).

    Loud refusals (concrete inputs; value checks skip tracers — main parity):
    ``sfe_global`` outside (0, 1); empty collapse support (w ≡ 0 exactly — a cloud
    that can form no stars at any τ); a requested SFE above the achievable ceiling
    (the mass fraction with w > 0), reported with the ceiling; and post-solve
    non-convergence (the bracket could not reach the requested SFE). NB the smooth
    sigmoid weight is strictly positive, so realistic fields have ceiling 1; the
    support/ceiling guards bite for hard-masked or degenerate inputs (AC-G8).
    Differentiable in every array argument and in ``sfe_global`` via the IFT JVP.
    """
    del cell_volume  # F is a mass ratio; volume cancels (kept in the signature for
    #                  call-site clarity: the constraint is ∫ρ⋆dV/M_cl)
    traced = any(
        isinstance(x, jax.core.Tracer) for x in (w, t_ff, rho_cl, sfe_global)
    )
    if not traced:
        sfe_val = float(sfe_global)
        if not 0.0 < sfe_val < 1.0:
            raise ValueError(f"sfe_global must be in (0, 1), got {sfe_val}")
        ceiling = float(jnp.sum(jnp.where(w > 0, rho_cl, 0.0)) / jnp.sum(rho_cl))
        if ceiling == 0.0:
            raise ValueError(
                "empty collapse support: w ≡ 0 everywhere — no cell is collapse-"
                "eligible, so no SFE is achievable"
            )
        if sfe_val >= ceiling:
            raise ValueError(
                f"requested sfe_global={sfe_val} is not achievable: the collapse-"
                f"eligible (w > 0) mass fraction caps the SFE at {ceiling:.6f}"
            )
    tau = _solve_tau_core(w, t_ff, rho_cl, jnp.asarray(sfe_global), n_iter)
    if not traced:
        achieved = float(_global_sfe(tau, w, t_ff, rho_cl))
        if abs(achieved - float(sfe_global)) > 1e-6:
            raise RuntimeError(
                f"tau_star solve did not converge: requested SFE {float(sfe_global)}, "
                f"achieved {achieved} at the bracket ceiling — the requested SFE is "
                f"effectively unreachable for this field"
            )
    return tau
