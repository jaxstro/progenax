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

from gravoturb_fdf.theory.pdf import bm19_icdf, bm19_volume_tail_fraction


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


def rank_to_uniform(values: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Empirical-CDF (rank) copula: u = (rank(values) + 0.5) / N.

    Double-argsort assigns each element its rank in [0, N); ``(rank+0.5)/N`` is a
    permutation of the uniform plotting positions, so ``u`` is **exactly** uniform on
    (0,1) regardless of the input's realized marginal. Shape-preserving.

    Non-differentiable in ``values`` (argsort); used on a frozen GRF realization, so
    downstream grads in the cloud parameters are unaffected.
    """
    flat = values.ravel()
    n = flat.size
    ranks = jnp.argsort(jnp.argsort(flat))
    u = (ranks + 0.5) / n
    return u.reshape(values.shape)


def rank_copula_field(
    g: Float[Array, "..."],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
) -> Float[Array, "..."]:
    r"""Remap a GRF ``g`` to the BM19 volume marginal via the rank copula.

    ``u = (rank(g)+0.5)/N`` (distribution-free) → ``s = F_BM19^{-1}(u)``, then a
    constant shift enforces the ρ_0 convention ``⟨e^s⟩ = 1`` (ρ_0 = volume-mean
    density). The shift is a constant in (mach, b, alpha)-space, so the marginal shape
    is preserved and the result is differentiable in the cloud parameters through the
    smooth CDF table (the ranks are frozen).

    Returns ``s = ln(ρ/ρ_0)`` with the same shape as ``g``.
    """
    u = rank_to_uniform(g)
    s_raw = bm19_icdf(u.ravel(), mach, b, alpha).reshape(g.shape)
    # Enforce ρ_0 = volume mean: ⟨e^s⟩ = 1  ⇒  s = s_raw − ln⟨e^{s_raw}⟩.
    shift = jnp.log(jnp.mean(jnp.exp(s_raw)))
    return s_raw - shift


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
    return n_cells * bm19_volume_tail_fraction(mach, b, alpha)


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
