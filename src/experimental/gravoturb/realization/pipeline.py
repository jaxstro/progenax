"""End-to-end cloud→field→stars turbulent-density-field pipeline (spec §8 algorithm, §3.5-3.6).

Steps:
  1. (ℳ,b,α) → σ_s², s_t, f_dense (BM19 1D theory).
  2. Gaussian random field g with P(k)∝k^{-β}.
  3. Rank-copula remap g → s with the BM19 volume marginal (⟨e^s⟩=1).
  4. Realized dense mass fraction f_dense_realized = Σ_{s>s_t} e^s / Σ e^s
     (hard κ→∞ limit of f_tail_actual) — the AC6 cornerstone metric.
  5. (optional) sample N⋆ stars: N_tail=round(f_sub·N⋆) from p_tail∝wρ, rest from ρ.

AC6 (make-or-break): f_dense_realized must reproduce BM19 f_dense — |bias|<5% single,
<1% ensemble. The realized fraction is honest field fidelity, independent of any soft-
mask sharpness.

JAX-native; build_turbulent_field is an eager builder (host-side resolution warning).
"""

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gravoturb.realization.copula import mass_conserving_copula_field
from gravoturb.realization.gaussian_field import (
    gaussian_random_field,
    low_resolution_flag,
)
from gravoturb.realization.placement import sample_positions
from gravoturb.theory.density_pdf import (
    dense_mass_fraction,
    sigma_s_squared,
    transition_density,
)


class TurbulentField(NamedTuple):
    """A realized turbulent log-density field plus its BM19 scalars (a JAX pytree)."""

    # log-density ln(ρ/ρ_0). NOTE ⟨e^s⟩ = mean_density(mach,b,alpha) ≥ 1, NOT 1:
    # ρ_0 is BM19's PRE-COLLAPSE reference density, and the powerlaw tail adds mass
    # above it. BM19 §2 give the mass-conserving alternative as a density shift
    # s_new = s − s_s (their Eq. 3, with e^{s_s} = mean_density); we deliberately do
    # NOT apply it (see realization/copula.py). Downstream this is immaterial:
    # gas.normalized_cloud_density rescales to ∫ρ dV = M_cl exactly and is invariant to any
    # additive constant in s, so the offset cancels. See tests/experimental/unit/
    # test_mass_conservation.py, which pins BOTH the offset and its cancellation.
    s: Float[Array, "nx ny nz"]
    s_t: Float[Array, ""]            # transition log-density (BM19 Eq.2)
    f_dense: Float[Array, ""]        # BM19 theoretical dense mass fraction
    f_dense_realized: Float[Array, ""]  # realized hard mass fraction above s_t
    low_resolution: bool             # < ~5 cells expected above s_t


def build_turbulent_field(
    mach: float,
    b: float,
    alpha: float,
    beta: float,
    shape: tuple[int, int, int],
    key: jax.Array,
) -> TurbulentField:
    r"""Realize the turbulent density field (steps 1-4) and report the AC6 cornerstone metric."""
    g = gaussian_random_field(shape, beta, key)
    return turbulent_field_from_gaussian(g, mach, b, alpha)


def turbulent_field_from_gaussian(
    g: Float[Array, "nx ny nz"],
    mach: float,
    b: float,
    alpha: float,
) -> TurbulentField:
    r"""Impose the BM19 marginal on a PROVIDED Gaussian carrier (the copula side).

    The seam behind :func:`build_turbulent_field` (which draws its own free-β GRF) and
    the Phase-3 Helmholtz path (whose carrier is the coupled ĝ ∝ −∇·v from
    :func:`gravoturb.realization.helmholtz.coupled_log_density_gaussian`, with the
    DERIVED slope β = β_v − 2). The mass-conserving rank copula consumes only the
    carrier's ranks, so AC6 (f_dense fidelity) holds by construction on either carrier;
    the carrier chooses the RANK ARRANGEMENT (spectrum/coupling), the copula the marginal.
    """
    shape = g.shape
    s_t = transition_density(alpha, sigma_s_squared(mach, b))
    f_dense = dense_mass_fraction(mach, b, alpha)

    n_cells = shape[0] * shape[1] * shape[2]
    low_res = bool(low_resolution_flag(n_cells, mach, b, alpha))
    if low_res:
        warnings.warn(
            f"Turbulent density field {shape}: <5 cells expected above s_t "
            f"(tail under-resolved at ℳ={mach}, b={b}, α={alpha}); "
            "increase resolution or soften the tail.",
            stacklevel=2,
        )

    s = mass_conserving_copula_field(g, mach, b, alpha)

    rho = jnp.exp(s)
    above = s > s_t
    f_dense_realized = jnp.sum(jnp.where(above, rho, 0.0)) / jnp.sum(rho)

    return TurbulentField(
        s=s,
        s_t=s_t,
        f_dense=f_dense,
        f_dense_realized=f_dense_realized,
        low_resolution=low_res,
    )


def cloud_to_stars(
    field: TurbulentField,
    f_sub: float,
    n_stars: int,
    key: jax.Array,
    mask_sharpness: float = 8.0,
    box_size: float = 1.0,
) -> Float[Array, "n_stars 3"]:
    r"""Sample ``n_stars`` star positions from a realized turbulent density field (step 5)."""
    return sample_positions(
        field.s, field.s_t, mask_sharpness, f_sub, n_stars, key, box_size=box_size
    )
