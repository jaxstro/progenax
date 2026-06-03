"""BM19 dense-tail field validation (audit M3).

The Gaussian-copula field generator must reproduce the BM19 dense-tail MASS fraction
f_dense at the *physical* spectral slope beta=4. The legacy normal-CDF copula
(u = Phi(g)) assumes g ~ N(0,1); at beta=4 the realized GRF is non-Gaussian, so the
tail collapses and f_tail undershoots with large realization scatter. The rank /
empirical-CDF copula forces the marginal exactly at any beta.

Regime: Mach=2, alpha=2 -> f_dense = 0.0568 (the audit's ~0.057), tail resolvable at
N=48. Oracle: f_dense_bm19_full (analytic BM19 mass integral).
"""

import warnings

import jax
import numpy as np
import pytest

import progenax  # noqa: F401  (enables float64)
from progenax.gravoturb import bm19_pipeline
from progenax.gravoturb.bm19_model import f_dense_bm19_full
from progenax.cluster.fdf_density import init_bm19_density_field
from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19


def _measure_f_tail(seeds, grid_size, beta, **kw):
    r = bm19_pipeline(2.0, 2.0)
    sigma_s_sq, s_t, alpha = float(r.sigma_s_sq), float(r.s_t), 2.0
    f_dense = float(f_dense_bm19_full(sigma_s_sq, s_t, alpha))
    vals = []
    for s in range(seeds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fld = init_bm19_density_field(
                jax.random.PRNGKey(s), sigma_s_sq, s_t, alpha,
                grid_size=grid_size, beta=beta, **kw,
            )
        vals.append(float(compute_tail_pmfs_bm19(fld.rho_grid, s_t).f_tail_actual))
    return np.array(vals), f_dense


class TestBM19FieldDenseTail:
    """Realized dense-tail mass fraction matches BM19 theory at beta=4 (M3)."""

    def test_default_copula_matches_f_dense_at_beta4(self):
        """Default copula: realized f_tail -> f_dense within ~15%, near-zero scatter, beta=4."""
        vals, f_dense = _measure_f_tail(seeds=4, grid_size=48, beta=4.0)
        assert vals.mean() > 0.85 * f_dense, (
            f"f_tail={vals.mean():.4f} < 0.85*f_dense={0.85 * f_dense:.4f} "
            f"(dense tail undersampled at beta=4, audit M3)"
        )
        assert vals.std() < 0.1 * f_dense, (
            f"f_tail scatter {vals.std():.4f} too large vs f_dense {f_dense:.4f} "
            f"(copula marginal not exact)"
        )

    def test_phi_copula_reproduces_the_bug(self):
        """Regression guard: the legacy phi copula undersamples with large scatter at
        beta=4 (the audit-M3 symptom) -- this is *why* rank is the default."""
        vals, f_dense = _measure_f_tail(seeds=4, grid_size=48, beta=4.0, copula="phi")
        # The defining symptom is large realization-to-realization scatter (vs the
        # rank copula's near-zero), plus an undersampled mean.
        assert vals.std() > 0.2 * f_dense, "phi copula scatter unexpectedly small"
        assert vals.mean() < 0.85 * f_dense, "phi copula unexpectedly matched f_dense"

    def test_resolution_guard_warns_on_unresolved_tail(self):
        """An extreme s_t (tail count-prob << 1/N^3) warns the user to raise grid_size."""
        r = bm19_pipeline(10.0, 2.0)  # s_t ~ 9 -> tail unresolvable at N=64
        with pytest.warns(UserWarning, match="under-resolved"):
            init_bm19_density_field(
                jax.random.PRNGKey(0), float(r.sigma_s_sq), float(r.s_t), 2.0,
                grid_size=64, beta=4.0,
            )

    def test_resolution_guard_silent_when_resolvable(self):
        """No under-resolution warning in the resolvable regime (Mach=2, N=48)."""
        r = bm19_pipeline(2.0, 2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # any UserWarning -> failure
            init_bm19_density_field(
                jax.random.PRNGKey(0), float(r.sigma_s_sq), float(r.s_t), 2.0,
                grid_size=48, beta=4.0,
            )
