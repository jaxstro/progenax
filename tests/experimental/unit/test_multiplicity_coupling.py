"""Environment-coupled multiplicity (λ_mult): the blended system↔position placement primitive.

Systems (self-consistent (m1, m2, orbit) units) are assigned to fixed spatial positions by a
Gaussian-copula affinity that couples system MASS to local density with strength λ_corr and system
MULTIPLICITY/compactness to local density with an independent strength λ_mult. Reassigning whole
systems preserves every marginal; λ_mult adds a controlled density→multiplicity correlation beyond
the emergent mass channel (CAREER birth variable, spec C).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def _spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a)))
    rb = np.argsort(np.argsort(np.asarray(b)))
    return float(np.corrcoef(ra, rb)[0, 1])


def test_placement_returns_permutation():
    """The placement returns a permutation of the systems (whole systems reassigned — set preserved)."""
    from gravoturb.realization.mass_assignment import blended_system_placement

    n = 400
    key = jax.random.PRNGKey(0)
    density = jax.random.uniform(key, (n,))
    mass_score = jax.random.uniform(jax.random.fold_in(key, 1), (n,))
    mult_score = jax.random.uniform(jax.random.fold_in(key, 2), (n,))
    perm = blended_system_placement(density, mass_score, mult_score,
                                    lambda_corr=0.5, lambda_mult=0.5, key=key)
    assert np.array_equal(np.sort(np.asarray(perm)), np.arange(n))  # a true permutation


def test_lambda_corr_couples_mass_to_density():
    """With λ_mult=0, higher λ_corr → stronger mass–density rank correlation at the placed positions."""
    from gravoturb.realization.mass_assignment import blended_system_placement

    n = 3000
    key = jax.random.PRNGKey(1)
    density = jax.random.uniform(key, (n,))
    mass_score = jax.random.uniform(jax.random.fold_in(key, 1), (n,))
    mult_score = jax.random.uniform(jax.random.fold_in(key, 2), (n,))

    def placed_mass_density_corr(lc):
        perm = blended_system_placement(density, mass_score, mult_score,
                                        lambda_corr=lc, lambda_mult=0.0, key=key)
        return _spearman(mass_score[np.asarray(perm)], density)

    c_lo = placed_mass_density_corr(0.2)
    c_hi = placed_mass_density_corr(0.9)
    assert c_hi > c_lo > 0.0
    assert c_hi > 0.6


def test_lambda_mult_adds_multiplicity_coupling_beyond_baseline():
    """At fixed λ_corr, raising λ_mult increases the multiplicity–density correlation beyond λ_mult=0."""
    from gravoturb.realization.mass_assignment import blended_system_placement

    n = 3000
    key = jax.random.PRNGKey(2)
    density = jax.random.uniform(key, (n,))
    mass_score = jax.random.uniform(jax.random.fold_in(key, 1), (n,))
    mult_score = jax.random.uniform(jax.random.fold_in(key, 2), (n,))

    def placed_mult_density_corr(lm):
        perm = blended_system_placement(density, mass_score, mult_score,
                                        lambda_corr=0.5, lambda_mult=lm, key=key)
        return _spearman(mult_score[np.asarray(perm)], density)

    base = placed_mult_density_corr(0.0)      # emergent-only (mult uncorrelated with mass here → ~0)
    coupled = placed_mult_density_corr(0.5)
    assert coupled > base + 0.1


# ---------------------------------------------------------------------------
# Cluster-level: λ_mult raises the f_bin–density coupling beyond the emergent baseline.
# ---------------------------------------------------------------------------

def _fbin_by_density_split(ic):
    """(f_bin in low-density half, f_bin in high-density half) over resolved systems."""
    sysid = np.asarray(ic.stars.system_id)
    pos = np.asarray(ic.stars.positions)
    order = np.argsort(sysid, kind="stable")
    sysid, pos = sysid[order], pos[order]
    uniq, idx, counts = np.unique(sysid, return_index=True, return_counts=True)
    is_binary = counts == 2
    sys_pos = np.array([pos[i:i + c].mean(axis=0) for i, c in zip(idx, counts)])
    s_total = np.asarray(ic.fields.s_total)
    origin = np.asarray(ic.geometry.origin)
    box = float(ic.geometry.box_size)
    nax = np.asarray(ic.geometry.shape)
    cell = np.clip(np.floor((sys_pos + origin) / box * nax).astype(int), 0, nax - 1)
    rho = s_total[cell[:, 0], cell[:, 1], cell[:, 2]]
    med = np.median(rho)
    return is_binary[rho <= med].mean(), is_binary[rho > med].mean()


def _build(lambda_mult):
    import jax.numpy as jnp
    from jaxstro.units import STELLAR
    from gravoturb.cluster import build_cluster_ic
    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec
    from progenax import PlummerProfile
    from progenax.imf import Maschberger
    from progenax.binaries.companions import MoeCompanions

    masses = Maschberger().sample(jax.random.PRNGKey(7), 2500)
    return build_cluster_ic(
        masses,
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=0.7), box_size=6.0, shape=(32, 32, 32)),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
        composition=CompositionSpec(placement="two_population", f_sub=0.3,
                                    lambda_corr=0.5, lambda_mult=lambda_mult,
                                    companions=MoeCompanions()),
        G=STELLAR.G, units=STELLAR, key=jax.random.PRNGKey(0),
    )


def test_lambda_mult_raises_fbin_density_coupling_in_cluster():
    """In a full build, λ_mult>0 gives a larger f_bin(high ρ) − f_bin(low ρ) split than λ_mult=0."""
    lo0, hi0 = _fbin_by_density_split(_build(0.0))
    lo1, hi1 = _fbin_by_density_split(_build(0.9))
    assert (hi1 - lo1) > (hi0 - lo0) + 0.03    # extra coupling beyond the mass-channel baseline
    assert hi1 > lo1                            # denser → more binary


def test_lambda_mult_preserves_population_marginals():
    """λ_mult only re-PLACES systems: the population (binary count + mass multiset) is identical."""
    ic0 = _build(0.0)
    ic1 = _build(0.9)
    # f_bin marginal: same number of primordial binaries regardless of placement
    assert int(ic0.ledger.n_binaries) == int(ic1.ledger.n_binaries)
    # mass marginal: the multiset of resolved component masses is unchanged (a permutation of place)
    m0 = np.sort(np.asarray(ic0.stars.masses))
    m1 = np.sort(np.asarray(ic1.stars.masses))
    assert np.allclose(m0, m1, rtol=1e-10)
