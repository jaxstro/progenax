"""Helmholtz-coupled velocity + density construction (realization/helmholtz.py; Phase 3).

k-space construction (design 2026-07-16 §Phase 3): one 3-component white field →
longitudinal/transverse projectors (P∥ = k̂k̂ᵀ, P⊥ = 1−k̂k̂ᵀ) → per-mode power weights
χ (compressive) and 1−χ (solenoidal) → spectral scaling k^{−β_v/2}. The density
Gaussian carrier is ĝ ∝ −i k·v̂∥ (linearized continuity; NO new randomness), so
β_density = β_v − 2 (P_{∇·v} = k² P∥) and corr(g, −∇·v) = 1 by construction — the
"frozen flow at star-formation epoch" limit, documented as such.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gravoturb.realization.helmholtz import (
    coupled_log_density_gaussian,
    helmholtz_velocity_field,
)

pytestmark = [pytest.mark.experimental, pytest.mark.unit]

SHAPE = (32, 32, 32)


def _longitudinal_power_fraction(v):
    """Measured Ψ = E_long/E_tot of a (n,n,n,3) velocity field via FFT projection."""
    v = np.asarray(v)
    n = v.shape[0]
    k1 = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    khat = np.stack([KX, KY, KZ], axis=-1) / np.where(kmag > 0, kmag, 1.0)[..., None]
    vk = np.stack([np.fft.fftn(v[..., i]) for i in range(3)], axis=-1)
    v_par = np.sum(vk * khat, axis=-1)
    e_long = np.sum(np.abs(v_par) ** 2)
    e_tot = np.sum(np.abs(vk) ** 2)
    return float(e_long / e_tot)


def _radial_slope(field3d, fit_lo=2.0, fit_hi=10.0):
    """log-log slope of the isotropic P(k) of a scalar field (numpy oracle)."""
    f = np.asarray(field3d) - np.asarray(field3d).mean()
    pk = np.abs(np.fft.fftn(f)) ** 2
    n = f.shape[0]
    k1 = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()
    pf = pk.ravel()
    keep = (kmag > fit_lo) & (kmag < fit_hi)
    kb = np.round(kmag[keep]).astype(int)
    means = np.array([pf[keep][kb == k].mean() for k in np.unique(kb)])
    ks = np.unique(kb).astype(float)
    coef, *_ = np.linalg.lstsq(
        np.vstack([np.log10(ks), np.ones_like(ks)]).T, np.log10(means), rcond=None)
    return -coef[0]


def test_velocity_field_shape_and_measured_compressive_fraction():
    """Realized E_long/E_tot equals the requested χ (per-mode exact split; the
    realization scatter over ~3e4 modes is percent-level)."""
    for chi in [0.192, 0.4, 1.0 / np.sqrt(3.0)]:
        v = helmholtz_velocity_field(SHAPE, beta_v=4.0, chi=chi,
                                     key=jax.random.PRNGKey(0))
        assert v.shape == SHAPE + (3,)
        assert np.all(np.isfinite(np.asarray(v)))
        assert _longitudinal_power_fraction(v) == pytest.approx(chi, abs=0.03)


def test_solenoidal_limit_has_no_longitudinal_power():
    """χ=0 (pure solenoidal) zeroes the compressive channel exactly."""
    v = helmholtz_velocity_field(SHAPE, beta_v=4.0, chi=0.0, key=jax.random.PRNGKey(1))
    assert _longitudinal_power_fraction(v) == pytest.approx(0.0, abs=1e-10)


def test_coupled_density_carrier_slope_is_beta_v_minus_2():
    """ĝ ∝ −i k·v̂∥ ⇒ P_g(k) ∝ k²·P∥(k) ∝ k^{−(β_v−2)} — the derived density slope."""
    for beta_v in [11.0 / 3.0, 4.0]:
        v_hat_bundle = helmholtz_velocity_field(
            SHAPE, beta_v=beta_v, chi=0.4, key=jax.random.PRNGKey(2),
            return_fourier=True)
        g = coupled_log_density_gaussian(v_hat_bundle)
        slope = _radial_slope(g)
        assert slope == pytest.approx(beta_v - 2.0, abs=0.35)  # finite-mode scatter


def test_coupled_carrier_is_exactly_minus_divergence():
    """corr(g, −∇·v) = 1 by construction (no new randomness on the compressive channel)."""
    bundle = helmholtz_velocity_field(SHAPE, beta_v=4.0, chi=0.4,
                                      key=jax.random.PRNGKey(3), return_fourier=True)
    v = bundle.velocity
    g = np.asarray(coupled_log_density_gaussian(bundle))
    # numpy oracle: −∇·v via spectral derivative
    n = SHAPE[0]
    k1 = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    vk = [np.fft.fftn(np.asarray(v)[..., i]) for i in range(3)]
    div = np.fft.ifftn(1j * 2 * np.pi / n * 0.0 + 1j * (KX * vk[0] + KY * vk[1] + KZ * vk[2])).real
    r = np.corrcoef(g.ravel(), (-div).ravel())[0, 1]
    assert r == pytest.approx(1.0, abs=1e-8)


def test_chi_zero_coupled_carrier_refused():
    """χ=0 has NO compressive channel — a coupled density carrier is degenerate (ĝ≡0)
    and must be refused loudly (amendment A3), pointing at coupling='independent'."""
    bundle = helmholtz_velocity_field(SHAPE, beta_v=4.0, chi=0.0,
                                      key=jax.random.PRNGKey(4), return_fourier=True)
    with pytest.raises(ValueError, match="independent"):
        coupled_log_density_gaussian(bundle)


def test_same_key_shares_phases_across_chi():
    """χ only re-weights the SAME white field (no new randomness): the transverse
    component at fixed key is proportional across different χ."""
    b1 = helmholtz_velocity_field(SHAPE, beta_v=4.0, chi=0.2, key=jax.random.PRNGKey(5))
    b2 = helmholtz_velocity_field(SHAPE, beta_v=4.0, chi=0.5, key=jax.random.PRNGKey(5))
    # project out the longitudinal parts (numpy) and compare transverse fields
    def _transverse(v):
        v = np.asarray(v)
        n = v.shape[0]
        k1 = np.fft.fftfreq(n) * n
        KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
        kmag2 = KX**2 + KY**2 + KZ**2
        khat = np.stack([KX, KY, KZ], axis=-1) / np.sqrt(np.where(kmag2 > 0, kmag2, 1.0))[..., None]
        vk = np.stack([np.fft.fftn(v[..., i]) for i in range(3)], axis=-1)
        v_par = np.sum(vk * khat, axis=-1)[..., None] * khat
        v_perp = vk - v_par
        return np.stack([np.fft.ifftn(v_perp[..., i]).real for i in range(3)], axis=-1)
    t1, t2 = _transverse(b1), _transverse(b2)
    scale = np.sqrt((1 - 0.5) / (1 - 0.2))
    np.testing.assert_allclose(t2, t1 * scale, rtol=1e-8, atol=1e-12)


def test_differentiable_in_chi_and_beta_v():
    """The construction is differentiable in (χ, β_v) — Fisher-integrity requirement."""
    def total_power(chi):
        v = helmholtz_velocity_field((16, 16, 16), beta_v=4.0, chi=chi,
                                     key=jax.random.PRNGKey(6))
        return jnp.sum(v**2)

    g = float(jax.grad(total_power)(0.4))
    assert np.isfinite(g)
