"""Mass-weighted substructure metric (Tier C.2) — Maschberger & Clarke (2011) local Σ.

Grounded in the held M&C 2011 (MNRAS 416, 541) PDF, §4 / Eq. 4: the local stellar surface
number density Σ = (k−1)/(π r_k²) with r_k the distance to the k-th nearest neighbour (k=6;
Casertano & Hut 1985). Mass segregation / primordial mass–density correlation is read off the
m–Σ plane: massive stars sit at systematically higher Σ. This is robust to substructure (works
in clumpy fields), unlike CW04 Q on small subsets. Non-differentiable (kNN ranking), a diagnostic.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def test_local_surface_density_exact_formula():
    """Σ_center = (k−1)/(π r_k²) exactly (M&C Eq. 4): center + 6 collinear neighbours, r₆=6."""
    from gravoturb.diagnostics.mass_density import local_surface_density

    xy = np.array([[0.0, 0.0]] + [[d, 0.0] for d in (1, 2, 3, 4, 5, 6)])  # 7 points
    sigma = np.asarray(local_surface_density(jnp.asarray(xy), k=6))
    expected = 5.0 / (np.pi * 36.0)  # (6-1)/(π·6²)
    assert sigma[0] == pytest.approx(expected, rel=1e-6)


def test_local_surface_density_recovers_uniform_density():
    """Median Σ of a uniform field ≈ true surface density N/A (Casertano–Hut ~unbiased)."""
    from gravoturb.diagnostics.mass_density import local_surface_density

    rng = np.random.default_rng(0)
    L, N = 10.0, 2000
    xy = rng.uniform(0, L, (N, 2))
    sigma = np.asarray(local_surface_density(jnp.asarray(xy), k=6))
    true = N / L**2
    assert 0.6 * true < np.median(sigma) < 1.4 * true


def test_dense_region_has_higher_sigma():
    """Stars in a tight clump have higher Σ than stars in a sparse halo."""
    from gravoturb.diagnostics.mass_density import local_surface_density

    rng = np.random.default_rng(1)
    clump = rng.normal(0, 0.1, (100, 2))
    halo = rng.uniform(-3, 3, (100, 2))
    xy = np.vstack([clump, halo])
    sigma = np.asarray(local_surface_density(jnp.asarray(xy), k=6))
    assert np.median(sigma[:100]) > 5 * np.median(sigma[100:])


def test_no_correlation_for_random_masses():
    """Random masses ⇒ m–Σ Spearman ρ ≈ 0 and high/low median-Σ ratio ≈ 1."""
    from gravoturb.diagnostics.mass_density import mass_density_segregation

    rng = np.random.default_rng(2)
    pos = rng.normal(size=(400, 3))
    rhos, ratios = [], []
    for s in range(15):
        masses = jnp.asarray(np.random.default_rng(s).random(400) ** -0.5)
        r = mass_density_segregation(jnp.asarray(pos), masses, k=6)
        rhos.append(r["rho_m_sigma"])
        ratios.append(r["median_sigma_ratio"])
    assert abs(float(np.mean(rhos))) < 0.1
    assert 0.8 < float(np.mean(ratios)) < 1.25


def test_primordial_correlation_detected():
    """When massive stars are placed in dense clumps (via correlated_mass_assignment λ=1),
    the metric detects it: ρ(m,Σ) > 0 and massive-vs-low median Σ ratio > 1."""
    from gravoturb.diagnostics.mass_density import (
        local_surface_density,
        mass_density_segregation,
    )
    from gravoturb.realization.mass_assignment import correlated_mass_assignment

    rng = np.random.default_rng(3)
    clump = rng.normal(0, 0.15, (150, 3))
    halo = rng.uniform(-3, 3, (150, 3))
    pos = jnp.asarray(np.vstack([clump, halo]))
    masses = jnp.asarray(rng.random(300) ** -0.5)
    local_dens = local_surface_density(pos[:, :2], k=6)  # density proxy at each star
    assigned = correlated_mass_assignment(
        masses, local_dens, lambda_corr=1.0, key=jax.random.PRNGKey(4)
    )
    r = mass_density_segregation(pos, assigned, k=6)
    assert r["rho_m_sigma"] > 0.5, f"ρ(m,Σ)={r['rho_m_sigma']:.3f}, expected strong +"
    assert r["median_sigma_ratio"] > 1.5, (
        f"ratio={r['median_sigma_ratio']:.2f}, expected >1"
    )


def test_reproducible_and_finite():
    from gravoturb.diagnostics.mass_density import mass_density_segregation

    rng = np.random.default_rng(5)
    pos = jnp.asarray(rng.normal(size=(200, 3)))
    m = jnp.asarray(rng.random(200) ** -0.5)
    a = mass_density_segregation(pos, m, k=6)
    b = mass_density_segregation(pos, m, k=6)
    assert a["rho_m_sigma"] == b["rho_m_sigma"]
    assert np.isfinite(a["rho_m_sigma"]) and np.isfinite(a["median_sigma_ratio"])
