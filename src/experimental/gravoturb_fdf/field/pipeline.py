"""End-to-end cloud→field→stars FDF pipeline (spec §8 algorithm, §3.5-3.6).

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

JAX-native; build_fdf_field is an eager builder (host-side resolution warning).
"""

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gravoturb_fdf.field.field import (
    gaussian_random_field,
    low_resolution_flag,
    mass_conserving_copula_field,
)
from gravoturb_fdf.field.sampling import sample_positions
from gravoturb_fdf.theory.bm19 import (
    f_dense_bm19_full,
    sigma_s_squared,
    transition_density,
)


class FDFField(NamedTuple):
    """A realized FDF log-density field plus its BM19 scalars (a JAX pytree)."""

    s: Float[Array, "nx ny nz"]      # log-density ln(ρ/ρ_0), ⟨e^s⟩=1
    s_t: Float[Array, ""]            # transition log-density (BM19 Eq.2)
    f_dense: Float[Array, ""]        # BM19 theoretical dense mass fraction
    f_dense_realized: Float[Array, ""]  # realized hard mass fraction above s_t
    low_resolution: bool             # < ~5 cells expected above s_t


def build_fdf_field(
    mach: float,
    b: float,
    alpha: float,
    beta: float,
    shape: tuple[int, int, int],
    key: jax.Array,
) -> FDFField:
    r"""Realize the FDF field (steps 1-4) and report the AC6 cornerstone metric."""
    s_t = transition_density(alpha, sigma_s_squared(mach, b))
    f_dense = f_dense_bm19_full(mach, b, alpha)

    n_cells = shape[0] * shape[1] * shape[2]
    low_res = bool(low_resolution_flag(n_cells, mach, b, alpha))
    if low_res:
        warnings.warn(
            f"FDF field {shape}: <5 cells expected above s_t "
            f"(tail under-resolved at ℳ={mach}, b={b}, α={alpha}); "
            "increase resolution or soften the tail.",
            stacklevel=2,
        )

    g = gaussian_random_field(shape, beta, key)
    s = mass_conserving_copula_field(g, mach, b, alpha)

    rho = jnp.exp(s)
    above = s > s_t
    f_dense_realized = jnp.sum(jnp.where(above, rho, 0.0)) / jnp.sum(rho)

    return FDFField(
        s=s,
        s_t=s_t,
        f_dense=f_dense,
        f_dense_realized=f_dense_realized,
        low_resolution=low_res,
    )


def cloud_to_stars(
    field: FDFField,
    f_sub: float,
    n_stars: int,
    key: jax.Array,
    kappa: float = 8.0,
    box_size: float = 1.0,
) -> Float[Array, "n_stars 3"]:
    r"""Sample ``n_stars`` star positions from a realized FDF field (step 5)."""
    return sample_positions(
        field.s, field.s_t, kappa, f_sub, n_stars, key, box_size=box_size
    )
