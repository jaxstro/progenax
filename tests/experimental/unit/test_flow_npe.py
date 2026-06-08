"""Unit tests for the flow-based NPE low-N beta likelihood (Path C).

Amortized Neural Posterior Estimation with a flowjax conditional normalizing flow:
q(z | s), z = logit-beta, s = whitened log+ band-power summary. Learns the projected-density marginal
implicitly -- the regime where the analytic shot model hit a wall (var-skew crossover). flowjax needs
new-style typed PRNG keys (jax.random.key).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental

BETA_LO, BETA_HI = 2.0, 11.0 / 3.0


def test_beta_z_transform_roundtrip():
    """z_to_beta(beta_to_z(beta)) == beta across the prior range (log-uniform reparam)."""
    from gravoturb_fdf.inference.flow_npe import beta_to_z, z_to_beta

    beta = jnp.array([2.0, 2.5, 3.0, 3.4, 11.0 / 3.0])
    z = beta_to_z(beta, BETA_LO, BETA_HI)
    np.testing.assert_allclose(np.asarray(z_to_beta(z, BETA_LO, BETA_HI)), np.asarray(beta), rtol=1e-8)


def test_whiten_zero_mean_unit_std():
    """whiten(s; stats) has ~0 mean and ~1 std per summary dimension."""
    from gravoturb_fdf.inference.flow_npe import whiten, whiten_stats

    rng = np.random.default_rng(0)
    s = jnp.asarray(rng.normal(5.0, 2.0, size=(500, 4)))
    mean, std = whiten_stats(s)
    w = np.asarray(whiten(s, mean, std))
    np.testing.assert_allclose(w.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(w.std(axis=0), 1.0, rtol=1e-6)


def test_npe_recovers_synthetic_conditional():
    """Trained conditional flow recovers a known posterior z|s ~ N(s0, 0.3): mean->s_obs0, std->0.3.

    End-to-end test of build + train + sample on a cheap synthetic problem (no simulator), proving the
    NPE machinery learns a conditional density correctly.
    """
    from gravoturb_fdf.inference.flow_npe import build_npe_flow, npe_posterior_z, train_npe

    key = jax.random.key(0)
    ks, kz, kf, kt, ksamp = jax.random.split(key, 5)
    n = 3000
    s = jax.random.normal(ks, (n, 3))
    z = s[:, :1] + 0.3 * jax.random.normal(kz, (n, 1))
    flow = build_npe_flow(kf, summary_dim=3, nn_width=32, flow_layers=4)
    flow = train_npe(kt, flow, z, s, learning_rate=1e-3, max_epochs=80, batch_size=256)
    samp = npe_posterior_z(ksamp, flow, jnp.array([1.5, 0.0, 0.0]), n_samples=3000)
    assert abs(float(samp.mean()) - 1.5) < 0.15
    assert abs(float(samp.std()) - 0.3) < 0.1
