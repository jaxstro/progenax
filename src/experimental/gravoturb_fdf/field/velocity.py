r"""Turbulent coherent velocities for the FDF ICs (build 2).

A young cluster's stars inherit the local coherent velocity of the natal turbulent gas (Goodwin &
Whitworth 2004 argue coherent, mildly super-virial velocities are the most realistic post-gas state).
G&W realize coherence via their fractal tree; for our GRF density field the faithful analog is a
**turbulent velocity field**: a 3-component Gaussian random field with a power-law spectrum
``P_v(k) ∝ k^{-beta_v}`` (so it is spatially smooth/coherent — nearby stars share velocity), sampled
(trilinearly interpolated) at the star positions. The amplitude is then set by the core
``progenax.virial_scale`` to a CHOSEN virial ratio ``Q ≡ T/|V|`` (0.5 virial, <0.5 collapsing, 0.75
super-virial); ``virial_scale`` uses the actual positions, so the spherical envelope's deeper potential
is automatically accounted for.

``beta_v`` is the velocity-spectrum slope — a turbulence parameter (NOT grounded by G&W, who use no
spectrum). Match it to the target turbulence (e.g. Larson σ–size / compressible-turbulence velocity
spectra; verify against Heyer 2009 / Federrath et al. before treating any default as physical).

JAX-native, differentiable in positions.
"""

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jaxtyping import Array, Float

from gravoturb_fdf.field.field import gaussian_random_field


def turbulent_velocity_field(
    shape: tuple[int, int, int], beta_v: Float[Array, ""], key: jax.Array
) -> Float[Array, "nx ny nz 3"]:
    r"""3-component coherent turbulent velocity GRF with spectrum ``P_v(k) ∝ k^{-beta_v}`` per axis.

    Each Cartesian component is an independent :func:`gaussian_random_field` realization (zero-mean,
    spatially coherent for ``beta_v > 0``); larger ``beta_v`` → smoother (more large-scale) field."""
    kx, ky, kz = jax.random.split(key, 3)
    return jnp.stack(
        [gaussian_random_field(shape, beta_v, k) for k in (kx, ky, kz)], axis=-1
    )


def sample_turbulent_velocities(
    positions: Float[Array, "n 3"],
    v_field: Float[Array, "nx ny nz 3"],
    box_size: Float[Array, ""] = 1.0,
) -> Float[Array, "n 3"]:
    r"""Trilinearly interpolate the velocity field to continuous star ``positions`` (in [0, box)^3).

    Cell centres sit at ``(i+0.5)/n * box`` (matching :func:`gravoturb_fdf.field.envelope.radius_grid`),
    so the grid coordinate is ``c = position/box * n - 0.5``. Periodic (``mode='wrap'``) to match the
    periodic GRF. Differentiable in ``positions``; nearby stars get nearly-equal velocities (coherent)."""
    n_axes = jnp.asarray(v_field.shape[:3])
    coords = (positions / box_size) * n_axes - 0.5          # (n, 3) grid coordinates
    cc = coords.T                                            # (3, n) for map_coordinates
    comps = [map_coordinates(v_field[..., i], cc, order=1, mode="wrap") for i in range(3)]
    return jnp.stack(comps, axis=-1)
