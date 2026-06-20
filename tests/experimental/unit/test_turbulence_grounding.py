"""P4 turbulence grounding — citations + density-spectrum slope corrected against PDFs.

These tests pin the 2026-06 grounding corrections (read from the held PDFs, not memory):
- FK10 σ_s²=ln(1+b²M²) is Eq. 19 (the docstring previously mis-cited "Eq. 14", which is
  the Azzalini skewed-lognormal).
- spectral_slope_from_mach must follow the DENSITY power spectrum (Kim & Ryu 2005), which
  FLATTENS with Mach (β decreases ~3.7→2.5 in the P_3D convention), NOT the velocity
  Kolmogorov→Burgers trend (β increasing toward 4) the code used before.

These verify core progenax.cluster.turbulence; kept under the experimental marker so the
released-core test count is unchanged.
"""

import jax
import pytest

pytestmark = pytest.mark.experimental


# ── FK10 citation (Task 4.1) ──
def test_sigma_s_cites_fk10_eq19_not_eq14():
    from progenax.cluster.turbulence import sigma_ln_rho_from_mach

    doc = sigma_ln_rho_from_mach.__doc__
    assert "Eq. 19" in doc
    # the old (wrong) attribution must be gone
    assert "Eq. 14" not in doc


# ── Kim & Ryu density-spectrum slope (Task 4.2) ──
@pytest.mark.parametrize(
    "mach,beta_expected",
    [(3.4, 3.08), (7.3, 2.75), (12.0, 2.52)],  # Kim&Ryu 2005 3D, P_3D = -E_slope + 2
)
def test_spectral_slope_matches_kimryu_anchors(mach, beta_expected):
    from progenax.cluster.turbulence import spectral_slope_from_mach

    beta = float(spectral_slope_from_mach(mach))
    assert beta == pytest.approx(beta_expected, abs=0.15)


def test_spectral_slope_transonic_kolmogorov_ceiling():
    """Transonic (M~1) → ~Kolmogorov 11/3 (Kim&Ryu M=1.2 slope ≈ -5/3)."""
    from progenax.cluster.turbulence import spectral_slope_from_mach

    assert float(spectral_slope_from_mach(1.2)) == pytest.approx(11.0 / 3.0, abs=0.1)


def test_spectral_slope_decreases_with_mach():
    """Density spectrum FLATTENS (β decreases) as Mach rises (the corrected direction)."""
    from progenax.cluster.turbulence import spectral_slope_from_mach

    betas = [float(spectral_slope_from_mach(m)) for m in (2.0, 5.0, 10.0, 20.0)]
    assert all(b2 < b1 for b1, b2 in zip(betas, betas[1:]))


def test_spectral_slope_bounded_and_not_burgers():
    """β stays in [2, 11/3]; supersonic β is shallow (NOT the old velocity-Burgers ~4)."""
    from progenax.cluster.turbulence import spectral_slope_from_mach

    for m in (1.0, 5.0, 12.0, 50.0):
        b = float(spectral_slope_from_mach(m))
        assert 2.0 <= b <= 11.0 / 3.0 + 1e-9
    assert float(spectral_slope_from_mach(10.0)) < 3.2  # would be ~4 under the old code


def test_spectral_slope_differentiable():
    import jax.numpy as jnp

    from progenax.cluster.turbulence import spectral_slope_from_mach

    g = float(jax.grad(spectral_slope_from_mach)(8.0))
    assert jnp.isfinite(g) and g < 0.0  # decreasing
