import jax, jax.numpy as jnp, sys, pathlib
import progenax
from jaxstro.units import STELLAR
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import _demo_oed as oed
import _demo_oed_depth as oed_depth


def test_blocks_from_eps_matches_per_star_blocks():
    th = oed.theta_truth()
    J, sig = oed.jacobian_and_sigma(th, oed.R_BINS, STELLAR.G)   # NEW: (3,K,3),(3,K)
    Mb_ref, _ = oed.per_star_blocks(th, oed.R_BINS, oed.EPS, STELLAR.G)
    Mb = oed.blocks_from_eps(J, sig, oed.EPS)                    # NEW: rebuild at given eps
    assert jnp.allclose(Mb, Mb_ref, atol=1e-12)
    # different eps -> different blocks (larger eps -> smaller information)
    Mb_noisy = oed.blocks_from_eps(J, sig, 2.0 * oed.EPS)
    assert jnp.all(jnp.diagonal(Mb_noisy, axis1=-2, axis2=-1)
                   <= jnp.diagonal(Mb, axis1=-2, axis2=-1) + 1e-12)


def test_eps_eff_rises_with_depth():
    e_shallow = oed_depth.eps_eff(m_lim=10.0)     # (3,) per channel
    e_deep    = oed_depth.eps_eff(m_lim=14.0)
    assert jnp.all(e_deep > e_shallow)            # admitting faint stars raises the mean error


def test_availability_rises_with_depth_and_is_radial():
    a_shallow = oed_depth.avail_bins(m_lim=10.0)  # (K,)
    a_deep    = oed_depth.avail_bins(m_lim=14.0)
    assert jnp.all(a_deep >= a_shallow)           # deeper detects more per bin
    assert a_shallow[0] > a_shallow[-1]           # outskirts star-starved (lower density)


def test_depth_fisher_spd_and_targets_M():
    z = jnp.zeros(3 * oed.R_BINS.shape[0])
    F = oed_depth.depth_fisher(z, m_lim=12.0, N_total=4000.0)
    assert F.shape == (3, 3) and jnp.allclose(F, F.T, atol=1e-10)
    assert jnp.all(jnp.linalg.eigvalsh(F) > 0)
    # c-criterion targeting M (index 1) is finite & positive
    assert oed.c_criterion(F, target=1) > 0
