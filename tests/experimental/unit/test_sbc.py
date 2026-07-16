"""SBC driver wiring test (Task 6).

Small/fast wiring check for ``sbc_ranks`` -- shape, rank support {0..L}, and param names.
The heavy uniformity assertion (ranks ~ DiscreteUniform) is AC18 / Task 7, not here.

These are slow (each trial is a full NUTS run); marked ``slow`` so the not-slow suites skip
them. The ``experimental`` marker keeps them out of the released-core collection.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gravoturb.inference.hmc import to_unconstrained
from gravoturb.inference.priors import BM19Prior
from gravoturb.inference.sbc import build_logdensity, sbc_ranks

pytestmark = [pytest.mark.experimental, pytest.mark.slow]


def test_build_logdensity_uses_log_count_variance_and_bandpower_blocks():
    """build_logdensity consumes the log-count-variance AND 2-pt band-power keys; finite + diff'able.

    Mirrors the SBC mock bundle with the count keys (Task 7) ``log_count_vars`` (per-cell measured
    Var_cells[log_plus(N)]) + threaded ``var_vs`` (fixed-fiducial estimator variance), PLUS the new
    field-level 2-pt beta channel: ``band_powers`` (measured periodogram on ``_K_EDGES``) fit with a
    fixed-fiducial ``bp_precision``. Asserts logdensity(z) is finite and jax.grad is finite (the
    band-power block must be differentiable in theta via the analytic power_spectrum_bandpowers).
    """
    from gravoturb.realization.gaussian_field import gaussian_random_field
    from gravoturb.realization.copula import rank_copula_field
    from gravoturb.inference.covariance import measured_bandpowers, mock_precision
    from gravoturb.inference.sbc import _ALPHA_FID, _BETA_FID, _K_EDGES, _MACH_FID

    pr = BM19Prior()
    shape = (24, 24, 24)
    cell_sizes = (4,)
    b = 0.4
    # A small, valid POT histogram (empty tail -> contributes exactly 0) + the count keys.
    s_thr, s_max = 5.0, 6.0
    # Measured band powers of a real latent log-density field + fixed-fiducial Hartlap precision.
    key = jax.random.PRNGKey(0)
    s_lo = rank_copula_field(
        gaussian_random_field(shape, _BETA_FID, jax.random.fold_in(key, 1)),
        _MACH_FID,
        b,
        _ALPHA_FID,
    )
    band_powers = measured_bandpowers(np.asarray(s_lo), shape, _K_EDGES)
    k_bp = jax.random.fold_in(key, 2**30)
    bp_rows = [
        measured_bandpowers(
            np.asarray(
                rank_copula_field(
                    gaussian_random_field(
                        shape, _BETA_FID, jax.random.fold_in(k_bp, i)
                    ),
                    _MACH_FID,
                    b,
                    _ALPHA_FID,
                )
            ),
            shape,
            _K_EDGES,
        )
        for i in range(16)
    ]
    bp_precision = mock_precision(bp_rows)

    data = {
        "exc_counts": np.zeros(8, dtype=float),
        "exc_edges": np.linspace(s_thr, s_max, 9),
        "log_count_vars": (0.35,),  # measured Var_cells[log_plus(N)] per cell
        "var_vs": (1e-3,),  # fixed-fiducial estimator variance per cell
        "n_bars": (5.0,),
        "band_powers": band_powers,
    }
    logdensity = build_logdensity(
        pr,
        data,
        b=b,
        s_thr=s_thr,
        s_max=s_max,
        shape=shape,
        cell_sizes=cell_sizes,
        bp_precision=bp_precision,
        n_max=8,
        n_s=256,
    )
    # Evaluate at an in-prior-support point (unconstrained image of a valid theta), as the
    # SBC driver does; z=0 would map outside the BM19 box -> prior is -inf there.
    z = to_unconstrained(jnp.array([8.0, 2.5, 3.0]))
    val = logdensity(z)
    assert jnp.isfinite(val), f"logdensity not finite: {val}"
    g = jax.grad(logdensity)(z)
    assert bool(jnp.all(jnp.isfinite(g))), f"gradient not finite: {g}"


def test_sbc_ranks_shape_and_support():
    pr = BM19Prior()
    out = sbc_ranks(
        pr,
        key=jax.random.PRNGKey(0),
        n_trials=6,
        b=0.4,
        s_thr_margin=0.75,
        shape=(24, 24, 24),
        density_shape=(48, 48, 48),
        n_warmup=80,
        n_samples=120,
        n_thin=4,
        cell_sizes=(2, 4),
        n_stars=4.0e4,
    )
    assert out["ranks"].shape == (6, 3)
    L = out["n_draws"]
    assert bool(jnp.all((out["ranks"] >= 0) & (out["ranks"] <= L)))
    assert out["param_names"] == ["M", "alpha", "beta"]
