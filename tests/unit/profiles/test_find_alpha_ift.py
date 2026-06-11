"""Forward + gradient characterization pins for find_alpha_for_masses.

Written against the CURRENT unrolled implementation so the custom_vjp/IFT
refactor (H2) must preserve the forward value (to ~residual) and match
finite-difference gradients. The IFT gradient is the EXACT fixed-point
gradient; we gate it against CENTRAL FINITE DIFFERENCES (the ground truth).
"""
import jax, jax.numpy as jnp, numpy as np
import progenax  # noqa: F401  (float64)
from progenax.profiles.limepy_multimass import find_alpha_for_masses, _bin_imf
from progenax.imf.smooth import Maschberger

# Converged alpha for _alpha_of(2.3, 5.0, 1.0, 0.4) under the CURRENT 30-step
# unrolled scan (sum=1.0, residual=3.4e-11). H2's adaptive solve must agree.
REF_ALPHA = [
    0.10602922107020678,
    0.16668777334641557,
    0.19602827681429544,
    0.5312547287690823,
]


def _alpha_of(alpha_imf, W0, g, delta):
    imf = Maschberger(alpha=alpha_imf, m_min=0.1, m_max=20.0)
    m_j, M_j = _bin_imf(imf, 4, (0.1, 20.0))
    a, res = find_alpha_for_masses(m_j, M_j, W0, g, delta)
    return a, res


class TestForwardRegression:
    def test_sums_to_one_and_converged(self):
        a, res = _alpha_of(2.3, 5.0, 1.0, 0.4)
        assert abs(float(jnp.sum(a)) - 1.0) < 1e-9
        assert float(res) < 2e-3 and bool(jnp.all(a > 0))

    def test_alpha_value_pinned(self):
        a, _ = _alpha_of(2.3, 5.0, 1.0, 0.4)
        # Fill REF_ALPHA from the first run's print; H2's adaptive solve must
        # agree with this (converged) value to <1e-4 (consistency), and the H2
        # version re-pins to its own value at rtol 1e-8 (self-regression).
        np.testing.assert_allclose(np.asarray(a), np.array(REF_ALPHA), atol=1e-4)


class TestGradientMatchesFD:
    def test_grad_alpha_imf(self):
        def loss(ai): a, _ = _alpha_of(ai, 5.0, 1.0, 0.4); return jnp.sum(a**2)
        g_ad = float(jax.grad(loss)(2.3)); h = 1e-4
        g_fd = float((loss(2.3+h)-loss(2.3-h))/(2*h))
        assert abs(g_ad-g_fd)/(abs(g_fd)+1e-12) < 1e-5

    def test_grad_delta_W0(self):
        def loss(t):
            d, W0 = t; a,_ = _alpha_of(2.3, W0, 1.0, d); return jnp.sum(a**2)
        g_ad = np.asarray(jax.grad(loss)(jnp.array([0.4,5.0]))); h=1e-4
        g_fd=[]
        for i in range(2):
            e=np.zeros(2); e[i]=h
            g_fd.append(float((loss(jnp.array([0.4,5.0])+e)-loss(jnp.array([0.4,5.0])-e))/(2*h)))
        np.testing.assert_allclose(g_ad, np.array(g_fd), rtol=1e-5, atol=1e-7)
