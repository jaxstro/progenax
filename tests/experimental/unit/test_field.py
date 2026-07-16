"""Gaussian random field with a turbulent power spectrum P(k) ∝ k^{-β}.

This is step 1 of the FDF field realization (spec §3.5) and the Lomax+2018 FBM
construction: complex Gaussian amplitudes scaled by √P(k)=k^{-β/2}, Hermitian-
symmetrized (real output via irfftn), DC mode zeroed (zero-mean field).

Tests assert the three defining properties: zero mean (DC=0), a real field, and
a radially-binned power spectrum whose log-log slope recovers -β.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def _measured_power_slope(field_np, k_lo=3.0, k_hi=None):
    """Radially-binned P(k) log-log slope (numpy, test-side analysis).

    Bins |F(k)|^2 in integer-|k| shells, fits log P vs log k over an
    intermediate band, and returns the slope.
    """
    n = field_np.shape[0]
    fk = np.fft.fftn(field_np)
    power = np.abs(fk) ** 2
    kx = np.fft.fftfreq(n) * n  # integer wavenumbers
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kbin = np.rint(kmag).astype(int)
    if k_hi is None:
        k_hi = n // 2 - 1
    ks, ps = [], []
    for k in range(int(k_lo), int(k_hi) + 1):
        sel = kbin == k
        if sel.sum() > 0:
            ks.append(k)
            ps.append(power[sel].mean())
    ks, ps = np.asarray(ks, float), np.asarray(ps, float)
    slope = np.polyfit(np.log(ks), np.log(ps), 1)[0]
    return slope


def test_grf_zero_mean():
    """DC mode is zeroed → field has (near) zero spatial mean."""
    from gravoturb.realization.gaussian_field import gaussian_random_field

    g = gaussian_random_field((32, 32, 32), beta=3.5, key=jax.random.PRNGKey(0))
    assert abs(float(jnp.mean(g))) < 1e-8


def test_grf_is_real():
    """irfftn output is a real-valued float array (Hermitian symmetry enforced)."""
    from gravoturb.realization.gaussian_field import gaussian_random_field

    g = gaussian_random_field((16, 16, 16), beta=3.0, key=jax.random.PRNGKey(1))
    assert jnp.isrealobj(g)
    assert g.shape == (16, 16, 16)
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize("beta", [3.0, 3.667, 4.0])
def test_grf_power_spectrum_slope(beta):
    """Radially-binned P(k) recovers slope ≈ -β over an intermediate k band."""
    from gravoturb.realization.gaussian_field import gaussian_random_field

    g = gaussian_random_field((64, 64, 64), beta=beta, key=jax.random.PRNGKey(7))
    slope = _measured_power_slope(np.asarray(g))
    assert slope == pytest.approx(-beta, abs=0.25)


def test_expected_cells_above_transition():
    """Expected count = n_cells × BM19 volume tail fraction above s_t."""
    from gravoturb.realization.gaussian_field import expected_cells_above_transition
    from gravoturb.theory.density_cdf import bm19_volume_tail_fraction

    n_cells = 128**3
    expected = float(expected_cells_above_transition(n_cells, 6.0, 0.4, 1.8))
    frac = float(bm19_volume_tail_fraction(6.0, 0.4, 1.8))
    assert expected == pytest.approx(n_cells * frac, rel=1e-9)
    assert expected > 100  # 128³ at these params is well-resolved


def test_resolution_guard_flags_small_field():
    """A tiny field with a steep tail expects <5 cells above s_t → low-resolution flag."""
    from gravoturb.realization.gaussian_field import low_resolution_flag

    # 8³=512 cells, steep alpha + modest Mach → ~1.3 cells above s_t
    assert bool(low_resolution_flag(8**3, mach=3.0, b=0.4, alpha=3.0))


def test_resolution_guard_ok_for_large_field():
    """A well-resolved 128³ field at typical params is NOT flagged."""
    from gravoturb.realization.gaussian_field import low_resolution_flag

    assert not bool(low_resolution_flag(128**3, mach=6.0, b=0.4, alpha=1.8))


def test_grf_deterministic_in_key():
    """Same key → identical field (reproducible); different key → different field."""
    from gravoturb.realization.gaussian_field import gaussian_random_field

    a = gaussian_random_field((16, 16, 16), beta=3.5, key=jax.random.PRNGKey(3))
    b = gaussian_random_field((16, 16, 16), beta=3.5, key=jax.random.PRNGKey(3))
    c = gaussian_random_field((16, 16, 16), beta=3.5, key=jax.random.PRNGKey(4))
    assert jnp.allclose(a, b)
    assert not jnp.allclose(a, c)
