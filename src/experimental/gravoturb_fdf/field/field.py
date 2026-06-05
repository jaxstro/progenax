"""Gaussian random field with turbulent power spectrum P(k) ∝ k^{-β}.

Step 1 of the FDF realization (spec §3.5) and the Lomax+2018 FBM construction:
draw complex Gaussian amplitudes, scale by √P(k) = k^{-β/2}, zero the DC mode, and
inverse-FFT to a real field. The real-output transform ``jnp.fft.irfftn`` enforces
Hermitian symmetry by construction, so the field is real and the requested isotropic
power spectrum is reproduced (in radial-shell average) up to per-mode χ² scatter.

The spectral slope maps to the Hurst exponent via β = E + 2H with embedding E = 3
(Lomax, Bate & Whitworth 2018, MNRAS 480, 371); callers pass β directly.

JAX-native, key-driven.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def gaussian_random_field(
    shape: tuple[int, int, int],
    beta: float,
    key: jax.Array,
) -> Float[Array, "n n n"]:
    r"""Real, zero-mean Gaussian random field with isotropic P(k) ∝ k^{-β}.

    Parameters
    ----------
    shape : (nx, ny, nz)
        Grid shape (cubic recommended; the spectrum is isotropic in integer-|k|).
    beta : float
        Power-spectrum slope. Amplitudes scale as k^{-β/2} so that the power
        |F(k)|² ∝ k^{-β}. For an FBM field, β = 3 + 2H.
    key : jax.Array
        PRNG key (deterministic given the key).

    Returns
    -------
    g : Float[Array, "nx ny nz"]
        Real field, DC mode zeroed (zero spatial mean).
    """
    nx, ny, nz = shape
    # rfft half-space frequency grid (cycles per sample → integer wavenumbers).
    kx = jnp.fft.fftfreq(nx) * nx
    ky = jnp.fft.fftfreq(ny) * ny
    kz = jnp.fft.rfftfreq(nz) * nz
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    kmag = jnp.sqrt(KX**2 + KY**2 + KZ**2)

    # √P(k) = k^{-β/2}; DC (k=0) set to 0 → zero-mean field, no division by zero.
    amplitude = jnp.where(kmag > 0, kmag ** (-0.5 * beta), 0.0)

    # Unit-variance complex white noise on the rfft half-space.
    kr, ki = jax.random.split(key)
    half_shape = KX.shape
    white = (
        jax.random.normal(kr, half_shape)
        + 1j * jax.random.normal(ki, half_shape)
    )
    spectrum = amplitude * white

    # irfftn enforces Hermitian symmetry → real output of the requested shape.
    g = jnp.fft.irfftn(spectrum, s=shape)
    return g
