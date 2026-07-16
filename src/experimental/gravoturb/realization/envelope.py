r"""Spherical cluster-shape envelope for the gravoturbulent ICs (separable log-space).

A young cluster forms from a centrally-concentrated turbulent clump: a smooth radial profile
(the SHAPE) modulated by turbulent fluctuations (the SUBSTRUCTURE). We model this by adding a radial
log-envelope to the existing BM19 turbulent log-density field:

    s_total(x) = s_turb(x) + ln rho_env(r),     rho_total = rho_env(r) * exp(s_turb(x)),

where ``s_turb`` is the mass-conserving copula field (⟨e^s⟩=1; carries β, ℳ, α) and ``rho_env(r)`` is
any progenax ``SpatialProfile`` density (Plummer/EFF/King; carries r_h, concentration). The split is
exact, so the BM19 marginal / dense-tail (s_t, f_dense, AC6) stay defined on ``s_turb`` (dense clumps =
LOCAL overdensities), while the envelope only imposes the large-scale spherical shape. Differentiable
in the profile parameters and ``s_turb``.

JAX-native.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

_TINY = 1e-300  # floor for ln(density) where a truncated profile (EFF/King) reaches zero


def radius_grid(
    shape: tuple[int, int, int], box_size: Float[Array, ""] = 1.0
) -> Float[Array, "nx ny nz"]:
    r"""Distance of each cell centre from the box centre, in the same units as ``box_size``.

    Cell centres are at ``(i + 0.5)/n * box_size`` per axis, centred on ``box_size/2`` — so the grid is
    symmetric about the cluster centre and no cell sits exactly at r=0 (avoids a singular centre)."""
    axes = [
        (jnp.arange(n) + 0.5) / n * box_size - 0.5 * box_size for n in shape
    ]
    X, Y, Z = jnp.meshgrid(*axes, indexing="ij")
    return jnp.sqrt(X**2 + Y**2 + Z**2)


def apply_spherical_envelope(
    s_turb: Float[Array, "nx ny nz"], profile, box_size: Float[Array, ""] = 1.0
) -> Float[Array, "nx ny nz"]:
    r"""Add the radial log-envelope of ``profile`` to the turbulent log-density ``s_turb``.

    ``s_total = s_turb + ln rho_env(r)`` with ``rho_env = profile.density(r)`` on the centred grid
    (:func:`radius_grid`). ``profile`` is any progenax ``SpatialProfile`` (uses ``.density``). Only the
    radial SHAPE of the envelope matters for star placement (an overall constant is irrelevant); the
    log is floored at ``_TINY`` so a truncated profile's zero-density exterior maps to ~no stars rather
    than NaN. Differentiable in the profile parameters and ``s_turb``."""
    r = radius_grid(s_turb.shape, box_size)
    rho_env = profile.density(r)
    return s_turb + jnp.log(jnp.clip(rho_env, _TINY, None))
