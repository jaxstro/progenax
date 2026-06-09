"""
Physics validation for the azimuthal-density-variation substructure diagnostic.

sigma_Sigma/<Sigma> (Kupper et al. 2011) is the relative scatter of star counts in
azimuthal sectors -- a cheap O(N) substructure proxy that rises with clumpiness and
falls toward the Poisson floor for a smooth axisymmetric cluster. Kupper give a
linear relation to fractal dimension, sigma_Sigma/<Sigma> ~ -0.46 D + 1.45 (D in
1.5-3.0), so the metric spans ~[0.07, 0.76].

The fractal *generator* was removed in the 2026-06 rewrite, so we do NOT reproduce
the D-slope directly; we validate the metric's floor, monotonicity, its span over
the Kupper range, and that it anti-correlates with the CW04 Q diagnostic.

Reference: Kupper et al. (2011), MNRAS 417, 2300.
"""
import numpy as np
import pytest

from progenax.diagnostics.substructure import (
    compute_azimuthal_variation,
    compute_q_parameter,
)

N_BINS = 12


def _smooth(n, seed):
    """Axisymmetric (uniform-azimuth) disc-projected cluster -> Poisson floor."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-1, 1, n)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _azimuthal_clumps(n, seed, n_clumps=4, frac=0.7, width=0.15):
    """Fraction `frac` of stars concentrated into `n_clumps` narrow azimuthal
    sectors, the rest uniform: a controllable azimuthal-substructure knob."""
    rng = np.random.default_rng(seed)
    n_cl = int(frac * n)
    centers = rng.uniform(0, 2 * np.pi, n_clumps)
    phi_cl = (centers[rng.integers(0, n_clumps, n_cl)]
              + rng.normal(0, width, n_cl))
    phi_un = rng.uniform(0, 2 * np.pi, n - n_cl)
    phi = np.concatenate([phi_cl, phi_un])
    r = np.sqrt(rng.uniform(0, 1, n))
    z = rng.uniform(-1, 1, n)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


class TestAzimuthalVariation:
    def test_smooth_near_poisson_floor(self):
        """A smooth axisymmetric cluster sits near the Poisson floor
        sqrt(n_bins/N)."""
        N = 3000
        vals = [compute_azimuthal_variation(_smooth(N, s), n_bins=N_BINS) for s in range(8)]
        floor = np.sqrt(N_BINS / N)
        mean = np.mean(vals)
        assert mean < 3 * floor, f"smooth sigma_Sigma/<Sigma>={mean:.3f} vs floor {floor:.3f}"

    def test_clumps_raise_metric(self):
        """Azimuthal clumps give a much larger sigma_Sigma/<Sigma> than smooth."""
        sm = np.mean([compute_azimuthal_variation(_smooth(2000, s)) for s in range(5)])
        cl = np.mean([compute_azimuthal_variation(_azimuthal_clumps(2000, s)) for s in range(5)])
        assert cl > 3 * sm and cl > 0.3, f"clumpy={cl:.3f} should exceed smooth={sm:.3f}"

    def test_monotonic_with_clumpiness(self):
        """The metric rises monotonically as the clump fraction increases."""
        fracs = [0.0, 0.3, 0.6, 0.9]
        vals = [np.mean([compute_azimuthal_variation(_azimuthal_clumps(2000, s, frac=f))
                         for s in range(4)]) for f in fracs]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), \
            f"not monotonic in clump fraction: {vals}"

    def test_spans_kupper_range(self):
        """Smooth -> clumpy spans roughly the Kupper [0.07, 0.76] range."""
        sm = np.mean([compute_azimuthal_variation(_smooth(2000, s)) for s in range(5)])
        cl = np.mean([compute_azimuthal_variation(
            _azimuthal_clumps(2000, s, n_clumps=3, frac=0.85, width=0.1)) for s in range(5)])
        assert sm < 0.2 and cl > 0.5, f"range not spanned: smooth={sm:.3f}, clumpy={cl:.3f}"

    def test_anticorrelates_with_cw04_Q(self):
        """sigma_Sigma/<Sigma> and CW04 Q both detect substructure, so they
        anti-correlate across a smooth->clumpy sequence."""
        sigmas, Qs = [], []
        for f in (0.0, 0.4, 0.8):
            for s in range(3):
                p = _azimuthal_clumps(600, s, frac=f)
                sigmas.append(compute_azimuthal_variation(p))
                Qs.append(compute_q_parameter(p))
        corr = np.corrcoef(sigmas, Qs)[0, 1]
        assert corr < -0.5, f"sigma_Sigma vs Q correlation = {corr:.2f} (expected strongly negative)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
