"""SBC driver wiring test (Task 6).

Small/fast wiring check for ``sbc_ranks`` -- shape, rank support {0..L}, and param names.
The heavy uniformity assertion (ranks ~ DiscreteUniform) is AC18 / Task 7, not here.

These are slow (each trial is a full NUTS run); marked ``slow`` so the not-slow suites skip
them. The ``experimental`` marker keeps them out of the released-core collection.
"""

import jax
import jax.numpy as jnp
import pytest

from gravoturb_fdf.inference.priors import BM19Prior
from gravoturb_fdf.inference.sbc import sbc_ranks

pytestmark = [pytest.mark.experimental, pytest.mark.slow]


def test_sbc_ranks_shape_and_support():
    pr = BM19Prior()
    out = sbc_ranks(pr, key=jax.random.PRNGKey(0), n_trials=6,
                    b=0.4, s_thr_margin=0.75,
                    shape=(24, 24, 24), density_shape=(48, 48, 48),
                    n_warmup=80, n_samples=120, n_thin=4,
                    cell_sizes=(2, 4), n_stars=4.0e4)
    assert out["ranks"].shape == (6, 3)
    L = out["n_draws"]
    assert bool(jnp.all((out["ranks"] >= 0) & (out["ranks"] <= L)))
    assert out["param_names"] == ["M", "alpha", "beta"]
