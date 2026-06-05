"""Smooth differentiable Q(f_sub; σ_s, β) surrogate (spec §8 / P3.3).

The CW04 Q estimator and the upstream categorical star sampling are non-differentiable,
so the differentiable interface to the calibration is a smooth parametric surrogate fit
to the measured Q(f_sub; σ_s, β) grid. It must recover a known linear-in-features model
from synthetic data, be differentiable, and decrease in f_sub.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def _synth(true_coeffs, n=200, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.uniform(0, 0.9, n)
    sig = rng.uniform(1.0, 1.8, n)
    beta = rng.uniform(3.0, 4.0, n)
    from gravoturb_fdf.surrogate import q_surrogate

    q = np.array([float(q_surrogate(f[i], sig[i], beta[i], true_coeffs)) for i in range(n)])
    q = q + noise * rng.standard_normal(n)
    return f, sig, beta, q


def test_fit_recovers_known_coeffs():
    """fit_q_surrogate recovers a known linear-in-features model (noiseless)."""
    from gravoturb_fdf.validation.calibration import fit_q_surrogate

    true = np.array([0.9, -0.25, -0.05, -0.03, 0.01, -0.02, 0.0])
    f, sig, beta, q = _synth(true, noise=0.0)
    coeffs = fit_q_surrogate(f, sig, beta, q)
    assert np.allclose(coeffs, true, atol=1e-6)


def test_fit_quality_with_noise():
    """RMS residual of the fit is within the injected noise scale."""
    from gravoturb_fdf.surrogate import q_surrogate
    from gravoturb_fdf.validation.calibration import fit_q_surrogate

    true = np.array([0.85, -0.30, 0.02, -0.04, 0.0, -0.01, 0.0])
    f, sig, beta, q = _synth(true, noise=0.02, seed=1)
    coeffs = fit_q_surrogate(f, sig, beta, q)
    pred = np.array([float(q_surrogate(f[i], sig[i], beta[i], coeffs)) for i in range(len(f))])
    rms = float(np.sqrt(np.mean((pred - q) ** 2)))
    assert rms < 0.03


def test_surrogate_differentiable_and_decreasing():
    """∂Q/∂f_sub via jax.grad is finite and negative for a decreasing fit."""
    from gravoturb_fdf.surrogate import q_surrogate
    from gravoturb_fdf.validation.calibration import fit_q_surrogate

    true = np.array([0.9, -0.25, -0.05, -0.03, 0.01, -0.02, 0.0])
    f, sig, beta, q = _synth(true, noise=0.0)
    coeffs = jnp.asarray(fit_q_surrogate(f, sig, beta, q))
    g = float(jax.grad(lambda x: q_surrogate(x, 1.5, 3.5, coeffs))(0.4))
    assert jnp.isfinite(g) and g < 0.0


def test_persisted_coeffs_monotone_in_band():
    """The persisted surrogate decreases in f_sub and stays in the CW04 substructured band."""
    from gravoturb_fdf.surrogate import PERSISTED_COEFFS, q_surrogate

    fs = jnp.linspace(0.0, 0.8, 9)
    q = jnp.array([q_surrogate(f, 1.68, 3.5, PERSISTED_COEFFS) for f in fs])
    assert float(q[0]) > float(q[-1])
    assert bool(jnp.all((q > 0.4) & (q < 0.85)))
