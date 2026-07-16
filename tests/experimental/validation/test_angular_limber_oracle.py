"""Slow oracle: predicted vs MEASURED projected-density angular band-power SLOPE (A-new1, Piece 3).

The keystone gate for the 2-D Limber predictor: the analytic
``angular_bandpowers_2d_limber`` (exact BM19 density 2-pt -> Limber slab -> 2-D FFT) must
reproduce the SLOPE of the measured projected-density band-powers of the **generative model it
describes**, with NO fitted constant -- so beta is FIT-able against it.

GENERATIVE MAP (Anna-approved 2026-06-07, "Option A"): the inference's generative model is the
**pointwise copula map** ``smooth_copula_field`` (= ``s = s_of_g(g)`` on a unit-variance Gaussian),
for which the d_n Mehler density 2-pt is *exact*. This is the canonical lognormal-copula generative
model (Coles & Jones 1991); validating the predictor against it (and generating SBC mocks from it)
is self-consistent by construction. The mass-conserving copula ``mass_conserving_copula_field`` is
reserved for the IC *realization* / the f_dense (AC6) 1-pt cornerstone -- a separate deliverable.

Measured side: ``measure_angular_bandpowers_2d(exp(s).sum(axis=2), k_edges)`` averaged over
realizations; slope via a log-log LSQ over the signal band. Predicted side: the analytic predictor
at full depth. Config matches the validated _v1b harness (96^3, n_real>=20, full depth, M=8,
b=0.4, alpha=2.5, signal band k in [1,20]). Verified slope |d| under the pointwise generative map:

    beta=2.5 : |d| ~ 0.01-0.05   (PASS <= 0.08)
    beta=3.0 : |d| ~ 0.02-0.04   (PASS <= 0.08)
    beta=3.5 : |d| ~ 0.05        (PASS <= 0.08)  -- the full LogUniform[2, 11/3] prior is covered.

DOCUMENTED MODELLING SYSTEMATIC (evidence this session, deterministic, NOT cosmic-variance): the
two copula maps agree for beta<=3 but DIVERGE at steep spectra -- vs the mass-conserving IC the
d_n predictor's slope error grows to ~0.165 at beta=3.5 (vs ~0.046 vs the pointwise map). Neither
map is "truth" (both are BM19-marginal GRF-copula models); near the Kolmogorov edge the copula-map
choice induces a ~0.12 slope systematic, a known modelling uncertainty to characterise against real
data / turbulence sims. The inference is self-consistent under the pointwise map (this gate); the
map-choice systematic is a model-adequacy caveat, separate from SBC validity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.experimental, pytest.mark.validation, pytest.mark.slow]

_SHAPE = (96, 96, 96)
_DEPTH = 96
_B, _ALPHA, _MACH = 0.4, 2.5, 8.0
_N_REAL = 20
_K_EDGES = np.linspace(1.0, 40.0, 13)
_SIG_LO, _SIG_HI = 1.0, 20.0
_SLOPE_TOL = 0.08


def _slope_in_band(kc, bp):
    """log-log LSQ slope over the signal band [_SIG_LO, _SIG_HI]."""
    m = (kc >= _SIG_LO) & (kc <= _SIG_HI) & (bp > 0) & np.isfinite(bp)
    return float(np.polyfit(np.log(kc[m]), np.log(bp[m]), 1)[0])


def _measured_minus_predicted_slope(beta):
    from gravoturb.realization.gaussian_field import gaussian_random_field
    from gravoturb.inference.covariance import angular_bandpowers_2d_limber
    from gravoturb.validation.measure import (
        measure_angular_bandpowers_2d,
        smooth_copula_field,
    )

    key = jax.random.fold_in(jax.random.PRNGKey(20260607), int(beta * 100))
    rows = []
    for r in range(_N_REAL):
        g = gaussian_random_field(_SHAPE, beta, jax.random.fold_in(key, r))
        # Option A: pointwise copula map (the generative model the d_n predictor is exact for).
        s = smooth_copula_field(g, _MACH, _B, _ALPHA)
        rows.append(measure_angular_bandpowers_2d(np.exp(s).sum(axis=2), _K_EDGES))
    meas = np.mean(rows, axis=0)

    kc, bp, _ = angular_bandpowers_2d_limber(
        _SHAPE,
        jnp.asarray(beta),
        jnp.asarray(_MACH),
        jnp.asarray(_B),
        jnp.asarray(_ALPHA),
        jnp.asarray(float(_DEPTH)),
        jnp.asarray(_K_EDGES),
        14,
        256,
    )
    kc, bp = np.asarray(kc), np.asarray(bp)
    return _slope_in_band(kc, meas), _slope_in_band(kc, bp)


@pytest.mark.parametrize("beta", [2.5, 3.0, 3.5])
def test_predicted_slope_matches_measured(beta):
    """Predicted projected-density band-power slope matches measured to <= 0.08.

    Under the pointwise-copula generative map (Option A) the gate holds across the FULL
    LogUniform[2, 11/3] beta prior, including the steep beta=3.5 (Kolmogorov) edge.
    """
    slope_meas, slope_pred = _measured_minus_predicted_slope(beta)
    d = abs(slope_pred - slope_meas)
    assert d <= _SLOPE_TOL, (
        f"beta={beta}: slope meas={slope_meas:+.4f} pred={slope_pred:+.4f} "
        f"|d|={d:.4f} exceeds {_SLOPE_TOL}"
    )
