"""Independent power-law companion mean; one vmapped value/gradient program."""

import jax
import jax.numpy as jnp
import numpy as np

from progenax.imf import BinaryIMF, PowerLawIMF
from progenax.imf.binary import ConstantBinaryFraction, PowerLawMassRatio

_PRIMARY = PowerLawIMF([0.0], [], m_min=1.0, m_max=3.0)


def _mean_ratio(gamma, q_min):
    model = BinaryIMF(
        primary_imf=_PRIMARY,
        binary_fraction=ConstantBinaryFraction(1.0),
        q_distribution=PowerLawMassRatio(gamma=gamma, q_min=q_min),
    )
    return model.mean_system_mass() / 2.0  # uniform primary has E[M1]=2


_means_and_slopes = jax.jit(jax.vmap(jax.value_and_grad(_mean_ratio), in_axes=(0, 0)))


def test_powerlaw_system_mean_including_logarithmic_indices():
    """Catch flat-q mean substituted for nonflat laws and lost gamma=-1/-2 limit slopes."""
    gammas = jnp.array([-2.0, -1.0, 0.0, 1.0, 0.0, 1.0])
    lower = jnp.array([0.1, 0.1, 0.1, 0.1, 0.0, 0.0])
    values, slopes = _means_and_slopes(gammas, lower)
    # Independent Gaussian quadrature integrates q^gamma and its log-weighted
    # moment; dE[q]/dgamma=Cov(q,log q). The positive-cutoff integrands are
    # analytic on [.1,1], so 128 points resolve these values to roundoff.
    nodes, weights = np.polynomial.legendre.leggauss(128)
    q = 0.1 + 0.9 * (nodes + 1) / 2
    weights = 0.9 * weights / 2
    refs, derivatives = [], []
    for gamma in [-2.0, -1.0, 0.0, 1.0]:
        p = weights * q**gamma
        p /= p.sum()
        mean = np.sum(p * q)
        refs.append(1 + mean)
        derivatives.append(np.sum(p * q * np.log(q)) - mean * np.sum(p * np.log(q)))
    # At q_min=0 normalizable power laws give E[q]=(gamma+1)/(gamma+2).
    refs.extend([1.5, 5 / 3])
    derivatives.extend([0.25, 1 / 9])
    assert jnp.allclose(values, jnp.array(refs), rtol=3e-14, atol=0.0)
    assert jnp.allclose(slopes, jnp.array(derivatives), rtol=3e-12, atol=0.0)
