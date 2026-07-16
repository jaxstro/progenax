"""Gaussian random field with turbulent power spectrum P(k) ∝ k^{-β}.

Step 1 of the turbulent-density-field realization (spec §3.5) and the Lomax+2018 FBM construction:
draw full-grid complex Gaussian amplitudes, scale by √P(k) = k^{-β/2}, zero the DC mode,
and take the real part of the inverse FFT. The full-grid + ``.real`` construction yields a
real, isotropic field whose ensemble power spectrum is |amplitude|² ∝ k^{-β} *exactly*
(amplitude is symmetric in k, so the ``.real`` symmetrization is exact), up to per-mode χ²
scatter. NB an earlier rfft construction filled the half-grid with independent complex
Gaussians and ``irfftn``'d without enforcing Hermitian symmetry on the kz=0/Nyquist planes,
which produced a spurious ~24% high-k power excess (the realized β drifted from the input β);
see tests/experimental/unit/test_projection.py::test_grf_realizes_power_law_spectrum.

The spectral slope maps to the Hurst exponent via β = E + 2H with embedding E = 3
(Lomax, Bate & Whitworth 2018, MNRAS 480, 371); callers pass β directly.

JAX-native, key-driven.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gravoturb.theory.density_cdf import volume_tail_fraction


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
    # Full integer-wavenumber grid (cycles per sample → integer |k|).
    kx = jnp.fft.fftfreq(nx) * nx
    ky = jnp.fft.fftfreq(ny) * ny
    kz = jnp.fft.fftfreq(nz) * nz
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    kmag = jnp.sqrt(KX**2 + KY**2 + KZ**2)

    # √P(k) = k^{-β/2}; DC (k=0) set to 0 → zero-mean field, no division by zero.
    amplitude = jnp.where(kmag > 0, kmag ** (-0.5 * beta), 0.0)

    # Full-grid complex white noise; the real part of the inverse FFT is a real, isotropic
    # GRF whose ensemble power is |amplitude|² ∝ k^{-β} (exact: amplitude is k-symmetric).
    kr, ki = jax.random.split(key)
    white = jax.random.normal(kr, shape) + 1j * jax.random.normal(ki, shape)
    g = jnp.fft.ifftn(amplitude * white).real
    return g


def expected_cells_above_transition(
    n_cells: int,
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Expected number of cells above s_t: ``n_cells × (1 − F(s_t))``.

    Uses the closed-form BM19 volume tail fraction. The dense tail must be resolved by
    enough cells for the rank copula to populate it faithfully (spec §3.5).
    """
    return n_cells * volume_tail_fraction(mach, b, alpha)


def low_resolution_flag(
    n_cells: int,
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
    min_cells: float = 5.0,
):
    r"""True when fewer than ``min_cells`` cells are expected above s_t.

    JIT-safe (returns a JAX boolean); the eager pipeline wrapper converts to a host
    bool and emits a ``warnings.warn`` — JAX cannot warn inside ``jit``.
    """
    return expected_cells_above_transition(n_cells, mach, b, alpha) < min_cells
