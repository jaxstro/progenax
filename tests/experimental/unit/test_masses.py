"""Density-correlated mass placement for the FDF cluster IC (Tier C).

``correlated_mass_assignment`` reorders an IMF mass sample so that massive stars sit
preferentially in the dense BM19 clumps (physical: massive stars form in dense gas —
competitive accretion / dense cores), with a tunable strength λ_corr ∈ [0, 1]:
  λ_corr = 0 → random placement (no primordial segregation),
  λ_corr = 1 → density-rank ↔ mass-rank (most massive in densest cell).
It is a permutation of the input masses (mass-conserving), built on the tested McLuster
partial-shuffle primitive. Non-differentiable (categorical), like the rest of star placement.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import spearmanr

pytestmark = pytest.mark.experimental


def _setup(n=300, seed=0):
    rng = np.random.default_rng(seed)
    masses = jnp.asarray(rng.random(n) ** -0.5)      # a spread of masses (IMF-like)
    local_density = jnp.asarray(rng.random(n))        # density at each star's position
    return masses, local_density


def test_lambda1_perfect_density_mass_correlation():
    """λ_corr=1 → most-massive stars at densest positions (Spearman ρ = 1)."""
    from gravoturb_fdf.masses import correlated_mass_assignment

    masses, dens = _setup()
    assigned = correlated_mass_assignment(masses, dens, lambda_corr=1.0, key=jax.random.PRNGKey(1))
    rho, _ = spearmanr(np.asarray(assigned), np.asarray(dens))
    assert rho > 0.999, f"λ=1 should give rank-perfect correlation, got ρ={rho:.4f}"


def test_lambda0_no_correlation():
    """λ_corr=0 → random placement, mean |ρ| ≈ 0 over seeds."""
    from gravoturb_fdf.masses import correlated_mass_assignment

    masses, dens = _setup()
    rhos = []
    for s in range(20):
        a = correlated_mass_assignment(masses, dens, lambda_corr=0.0, key=jax.random.PRNGKey(s))
        rhos.append(spearmanr(np.asarray(a), np.asarray(dens))[0])
    assert abs(float(np.mean(rhos))) < 0.1, f"λ=0 mean ρ={np.mean(rhos):.3f}, expected ≈0"


def test_correlation_monotonic_in_lambda():
    """Density–mass correlation increases monotonically with λ_corr."""
    from gravoturb_fdf.masses import correlated_mass_assignment

    masses, dens = _setup()

    def mean_rho(lam):
        return float(np.mean([
            spearmanr(np.asarray(correlated_mass_assignment(
                masses, dens, lambda_corr=lam, key=jax.random.PRNGKey(s))), np.asarray(dens))[0]
            for s in range(12)]))

    rhos = [mean_rho(l) for l in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(rhos[i] <= rhos[i + 1] + 0.05 for i in range(len(rhos) - 1)), f"not monotonic: {rhos}"
    assert rhos[-1] - rhos[0] > 0.5, f"λ should span a wide ρ range, got {rhos}"


def test_mass_conserving_permutation():
    """Output is a permutation of the input masses (mass-conserving) for any λ_corr."""
    from gravoturb_fdf.masses import correlated_mass_assignment

    masses, dens = _setup(n=200)
    for lam in (0.0, 0.5, 1.0):
        a = correlated_mass_assignment(masses, dens, lambda_corr=lam, key=jax.random.PRNGKey(3))
        assert jnp.allclose(jnp.sort(a), jnp.sort(masses)), f"masses not conserved at λ={lam}"


def test_reproducible_with_key():
    """Same key → identical assignment."""
    from gravoturb_fdf.masses import correlated_mass_assignment

    masses, dens = _setup(n=150)
    a = correlated_mass_assignment(masses, dens, lambda_corr=0.6, key=jax.random.PRNGKey(7))
    b = correlated_mass_assignment(masses, dens, lambda_corr=0.6, key=jax.random.PRNGKey(7))
    assert jnp.array_equal(a, b)
