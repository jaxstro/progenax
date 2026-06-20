import pathlib
import sys

import jax.numpy as jnp

from progenax import ChabrierIMF

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import _demo_selection as sel


def test_apparent_mag_and_distance_modulus():
    # (1) Zero-point uses the NORMAL present-day L_sun: a hypothetical 1.0 L_sun source is
    #     M_bol = 4.74 (IAU 2015). This is the UNIT check -- we do NOT renormalize to L_sun(ZAMS).
    assert jnp.allclose(sel.M_BOL_SUN - 2.5 * jnp.log10(1.0), 4.74)
    # (2) A 1 M_sun ZAMS *star* is sub-luminous (~0.70 L_sun via Tout+1996) -> FAINTER than the
    #     unit Sun. M_bol = 4.74 - 2.5 log10(0.698) ~= 5.13. At 10 pc (DM=0) m_app == M_bol.
    assert sel.abs_bol_mag(jnp.array(1.0)) > 4.74
    assert jnp.allclose(
        sel.apparent_mag(jnp.array(1.0), d_pc=10.0, Z=0.02), 5.13, atol=0.02
    )
    # (3) distance-modulus identity (DM=0 at 10 pc) + farther -> fainter (larger m_app)
    assert jnp.allclose(
        sel.apparent_mag(jnp.array(1.0), 10.0), sel.abs_bol_mag(jnp.array(1.0))
    )
    assert sel.apparent_mag(jnp.array(1.0), d_pc=4000.0) > sel.apparent_mag(
        jnp.array(1.0), d_pc=10.0
    )


def test_m_min_monotonic_in_depth():
    d = 4000.0
    # a brighter (smaller) m_lim admits only higher-mass stars -> larger m_min
    m_min_shallow = sel.m_min(m_lim=10.0, d_pc=d)
    m_min_deep = sel.m_min(m_lim=14.0, d_pc=d)
    assert m_min_deep < m_min_shallow  # deeper reaches lower mass
    # round-trip: a star exactly at m_lim has apparent mag ~ m_lim
    assert jnp.allclose(sel.apparent_mag(m_min_deep, d_pc=d), 14.0, atol=0.1)


def test_photon_noise_grows_faint():
    eps_bright = sel.photon_noise_error(m_app=jnp.array(12.0), eps0=1.0, m_ref=12.0)
    eps_faint = sel.photon_noise_error(m_app=jnp.array(16.0), eps0=1.0, m_ref=12.0)
    assert jnp.allclose(eps_bright, 1.0)  # at m_ref, error = eps0
    assert eps_faint > eps_bright  # 10^{0.2*(16-12)} = 10^0.8 = 6.31x


def test_detectable_fraction_in_unit_interval_and_monotonic():
    imf = ChabrierIMF(m_min=0.08, m_max=100.0)
    f_shallow = sel.detectable_fraction(m_lim=10.0, d_pc=4000.0, imf=imf)
    f_deep = sel.detectable_fraction(m_lim=14.0, d_pc=4000.0, imf=imf)
    assert 0.0 <= f_shallow <= 1.0 and 0.0 <= f_deep <= 1.0
    assert f_deep > f_shallow  # deeper detects a larger IMF fraction
