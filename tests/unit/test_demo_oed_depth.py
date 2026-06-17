import jax, jax.numpy as jnp, sys, pathlib
import progenax
from jaxstro.units import STELLAR
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import _demo_oed as oed


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
