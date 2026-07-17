r"""Helmholtz-coupled velocity + log-density construction (Phase 3, design 2026-07-16).

One 3-component complex white field is split per k-mode by the Helmholtz projectors
(P∥ = k̂k̂ᵀ longitudinal/compressive, P⊥ = 1 − k̂k̂ᵀ transverse/solenoidal), re-weighted
so the LONGITUDINAL POWER FRACTION is exactly χ (per-mode weights √(3χ) and
√(3(1−χ)/2): an isotropic white field carries 1/3 of its power in P∥, 2/3 in P⊥), and
spectrally scaled by k^{−β_v/2} (P_v(k) ∝ k^{−β_v}).

The density Gaussian carrier follows LINEARIZED CONTINUITY with no new randomness:
ĝ ∝ −i k·v̂ = −i k·v̂∥, so P_g(k) ∝ k² P∥(k) ∝ k^{−(β_v−2)} — the density slope is
DERIVED, β = β_v − 2, and corr(g, −∇·v) = 1 by construction (perfect correlation on the
compressive channel, none on the solenoidal one — the "frozen flow at star-formation
epoch" limit, claimed as such and no more). The mass-conserving copula then imposes the
BM19 marginal on g, so AC6 holds by construction; what the coupling changes is the RANK
structure, now shared with converging-flow regions.

χ defaults to the PDF-verified F10 relation :func:`gravoturb.theory.driving.chi_f10`
(χ = b/√3; F10 Eqs. 21–22). χ = 0 has no compressive channel: the coupled carrier is
degenerate (ĝ ≡ 0) and is REFUSED loudly (amendment A3) — use coupling='independent'.

Integer-wavenumber convention (fftfreq(n)·n), matching gaussian_field.py and the
inference/theory k-grids; the carrier feeds the rank copula (ranks only), so its overall
normalization is immaterial and it is returned unit-variance for cleanliness.

JAX-native, key-driven; differentiable in (χ, β_v).
"""

from typing import NamedTuple

import jax
import jax.core
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float


class HelmholtzBundle(NamedTuple):
    """A coupled velocity realization plus the spectral pieces the density carrier needs."""

    velocity: Float[Array, "nx ny nz 3"]      # real coherent velocity field
    div_hat: Complex[Array, "nx ny nz"]       # FT of ∇·v = i k·v̂ (integer-k convention)
    chi: Float[Array, ""] | float             # the imposed compressive power fraction


def _k_grids(shape):
    kx = jnp.fft.fftfreq(shape[0]) * shape[0]
    ky = jnp.fft.fftfreq(shape[1]) * shape[1]
    kz = jnp.fft.fftfreq(shape[2]) * shape[2]
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    kmag = jnp.sqrt(KX**2 + KY**2 + KZ**2)
    return (KX, KY, KZ), kmag


def helmholtz_velocity_field(
    shape: tuple[int, int, int],
    beta_v: Float[Array, ""],
    chi: Float[Array, ""],
    key: jax.Array,
    return_fourier: bool = False,
):
    r"""Coherent turbulent velocity field with an imposed compressive fraction χ.

    Per k-mode: v̂ = k^{−β_v/2} [√(3χ)·P∥ŵ + √(3(1−χ)/2)·P⊥ŵ] for one shared complex
    white field ŵ — so different χ at the same key RE-WEIGHT the same phases (no new
    randomness), and E_long/E_tot = χ in ensemble (per-mode exact split). DC zeroed.

    Returns the real (nx,ny,nz,3) field, or a :class:`HelmholtzBundle` carrying the
    spectral divergence for :func:`coupled_log_density_gaussian` when
    ``return_fourier=True``.
    """
    (KX, KY, KZ), kmag = _k_grids(shape)
    safe = jnp.where(kmag > 0, kmag, 1.0)
    khat = jnp.stack([KX, KY, KZ], axis=-1) / safe[..., None]

    kr, ki = jax.random.split(key)
    white = (jax.random.normal(kr, shape + (3,))
             + 1j * jax.random.normal(ki, shape + (3,)))

    w_par = jnp.sum(white * khat, axis=-1)[..., None] * khat
    w_perp = white - w_par

    # Zero the Nyquist planes (even grids): under the Hermitian symmetrization that
    # `.real` applies, a bin with ANY component at ±n/2 mirrors onto k' ≠ −k, which
    # silently mixes longitudinal and transverse power (measured: 2e-4 longitudinal
    # leak at χ=0 and corr(g, −∇·v) = 0.99 instead of 1). Dropping these inherently
    # sign-ambiguous modes restores exact transversality/continuity; the scalar GRF
    # (gaussian_field.py) is immune because its amplitude is k-symmetric. Odd grids
    # have no Nyquist bin (the mask is trivially true).
    no_nyquist = (
        (jnp.abs(KX) < shape[0] / 2)
        & (jnp.abs(KY) < shape[1] / 2)
        & (jnp.abs(KZ) < shape[2] / 2)
    )
    amplitude = jnp.where((kmag > 0) & no_nyquist, safe ** (-0.5 * beta_v), 0.0)
    v_hat = amplitude[..., None] * (
        jnp.sqrt(3.0 * chi) * w_par + jnp.sqrt(3.0 * (1.0 - chi) / 2.0) * w_perp
    )

    velocity = jnp.stack(
        [jnp.fft.ifftn(v_hat[..., i]).real for i in range(3)], axis=-1
    )
    if not return_fourier:
        return velocity
    div_hat = 1j * (KX * v_hat[..., 0] + KY * v_hat[..., 1] + KZ * v_hat[..., 2])
    return HelmholtzBundle(velocity=velocity, div_hat=div_hat, chi=chi)


def coupled_log_density_gaussian(bundle: HelmholtzBundle) -> Float[Array, "nx ny nz"]:
    r"""The coupled log-density Gaussian carrier g ∝ −∇·v (linearized continuity).

    Consumes a :class:`HelmholtzBundle`; returns the unit-variance real carrier whose
    ranks the mass-conserving copula reshapes into the BM19 marginal. Slope is the
    DERIVED β = β_v − 2. Refuses χ = 0 loudly (no compressive channel → ĝ ≡ 0,
    amendment A3): use coupling='independent' for the uncoupled ablation.
    """
    chi = bundle.chi
    if not isinstance(chi, jax.core.Tracer) and float(chi) == 0.0:
        raise ValueError(
            "chi=0 has no compressive channel — the coupled density carrier is "
            "degenerate (g ≡ 0); use coupling='independent' for the uncoupled ablation"
        )
    g = jnp.fft.ifftn(-bundle.div_hat).real
    return g / jnp.std(g)
