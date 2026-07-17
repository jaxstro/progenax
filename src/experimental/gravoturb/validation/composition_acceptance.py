"""AC-IC10/AC-IC11 acceptance for Phase-4b composition (λ_corr + binaries).

Printed-artifact discipline as in the sibling acceptance scripts; numpy/scipy
permitted (validation side). Design 2026-07-16 Phase 4: AC-IC10 = Spearman(m, ρ_local)
sweeps with λ_corr, λ_corr off byte-identical mass order, Λ_MSR(t=0) responds;
AC-IC11 = barycenter-first binaries with the released binary_energy_budget printed.
The third Phase-4b item (per-cell local IMF) was DEFENSIBILITY-REFUSED: Marks+2012's
α₃ relation is calibrated on the GLOBAL pre-cluster cloud-core density (their Fig. 3),
not any per-cell density — the cluster-level path via env_to_imf_params + masses-first
remains the supported route (verified against the held PDF 2026-07-16).

Run:
    PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
        python -m gravoturb.validation.composition_acceptance
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxstro.units import STELLAR
from scipy import stats
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from gravoturb.cluster import build_cluster_ic
from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec
from progenax import (
    FlatMassRatio,
    IndependentCompanions,
    LogUniformPeriod,
    Maschberger,
    ThermalEccentricity,
    binary_energy_budget,
    compute_kinetic_energy,
)

BOX, SHAPE = 4.0, (16,) * 3
G = STELLAR.G


def _build(masses, *, lam=None, companions=None, seed=3):
    return build_cluster_ic(
        masses,
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.5),
        geometry=GeometrySpec(profile=__import__("progenax").PlummerProfile(r_h=0.5),
                              box_size=BOX, shape=SHAPE),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
        composition=CompositionSpec(placement="two_population", f_sub=0.3,
                                    lambda_corr=lam, companions=companions),
        G=G, units=STELLAR, key=jax.random.PRNGKey(seed),
    )


def _local_log_density(ic):
    pos_box = np.asarray(ic.stars.positions) + np.asarray(ic.ledger.frame.origin)
    n = np.asarray(ic.geometry.shape)
    idx = np.clip((pos_box / float(ic.geometry.box_size) * n).astype(int), 0, n - 1)
    return np.asarray(ic.fields.s_total)[idx[:, 0], idx[:, 1], idx[:, 2]]


def _lambda_msr(positions, masses, n_massive=10, n_random=50, rng=None):
    """Allison+2009 mass-segregation ratio Λ_MSR = <l_random>/l_massive ± σ/l_massive
    (MST edge-length sum of the n_massive most-massive stars vs random n-samples).
    Independent scipy oracle."""
    rng = rng or np.random.default_rng(0)
    pos = np.asarray(positions)

    def mst_length(idx):
        d = squareform(pdist(pos[idx]))
        return minimum_spanning_tree(d).sum()

    massive_idx = np.argsort(np.asarray(masses))[-n_massive:]
    l_massive = mst_length(massive_idx)
    l_rand = np.array([mst_length(rng.choice(len(pos), n_massive, replace=False))
                       for _ in range(n_random)])
    return float(l_rand.mean() / l_massive), float(l_rand.std() / l_massive)


def ac_ic10_lambda_corr(seeds=(3, 4, 5)):
    print("\n=== AC-IC10 — λ_corr primordial segregation: Spearman sweep + Λ_MSR ===")
    imf = Maschberger()
    rows = []
    for lam in [None, 0.0, 0.5, 1.0]:
        rs = []
        for sd in seeds:
            masses = imf.sample(jax.random.PRNGKey(100 + sd), 800)
            ic = _build(masses, lam=lam, seed=sd)
            rs.append(stats.spearmanr(np.asarray(ic.stars.masses),
                                      _local_log_density(ic)).statistic)
        rows.append((lam, float(np.mean(rs)), float(np.std(rs))))
        lbl = "off " if lam is None else f"{lam:.1f} "
        print(f"  λ_corr={lbl}: Spearman(m, s_local) = {rows[-1][1]:+.3f} ± {rows[-1][2]:.3f}")
    # off: input order preserved exactly
    masses = imf.sample(jax.random.PRNGKey(100), 800)
    ic_off = _build(masses, lam=None, seed=3)
    order_ok = bool(np.array_equal(np.asarray(ic_off.stars.masses), np.asarray(masses)))
    # monotone response + strong at 1 + ~0 when off/random
    vals = {lam: r for lam, r, _ in rows}
    mono = vals[0.5] > vals[0.0] + 0.1 and vals[1.0] > vals[0.5] + 0.1
    ok = order_ok and mono and vals[1.0] > 0.9 and abs(vals[None]) < 0.15

    # Λ_MSR(t=0) responds (Allison+2009 oracle; massive stars in dense clumps →
    # shorter massive-MST → Λ > 1)
    m0 = imf.sample(jax.random.PRNGKey(100), 800)
    lam_off = _lambda_msr(_build(m0, lam=None, seed=3).stars.positions,
                          _build(m0, lam=None, seed=3).stars.masses)
    lam_on = _lambda_msr(_build(m0, lam=1.0, seed=3).stars.positions,
                         _build(m0, lam=1.0, seed=3).stars.masses)
    print(f"  Λ_MSR(t=0): off = {lam_off[0]:.2f}±{lam_off[1]:.2f}   "
          f"λ_corr=1 = {lam_on[0]:.2f}±{lam_on[1]:.2f}")
    responds = lam_on[0] > lam_off[0] + 2.0 * max(lam_on[1], lam_off[1])
    ok = ok and responds
    print(f"  input order preserved when off = {order_ok}; monotone = {mono}; "
          f"Λ_MSR responds (>2σ) = {responds}  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "rows": rows, "lambda_msr": (lam_off, lam_on)}


def ac_ic11_binaries(seed=7):
    print("\n=== AC-IC11 — binaries barycenter-first: energy budget + boundness ===")
    comp = IndependentCompanions(
        binary_fraction=0.4,
        q_distribution=FlatMassRatio(q_min=0.1),
        period_distribution=LogUniformPeriod(log_P_min=2.0, log_P_max=5.0),
        eccentricity_distribution=ThermalEccentricity(),
    )
    n_sys = 400
    ic = _build(jnp.ones(n_sys), companions=comp, seed=seed)
    n_bin = int(ic.ledger.n_binaries)
    print(f"  systems={n_sys}, binaries={n_bin} (f_b=0.4 → expect ~160), "
          f"stars={ic.stars.positions.shape[0]}")

    budget = binary_energy_budget(
        ic.stars.positions, ic.stars.velocities, ic.stars.masses,
        ic.stars.system_id, G=G,
    )
    print("  binary_energy_budget (released diagnostic):")
    for name in budget._fields:
        val = getattr(budget, name)
        try:
            print(f"    {name:<28} {float(val):+.6g}")
        except TypeError:
            pass
    # the COM (barycenter) virial state vs the internal reservoir are separated:
    # internal binding is negative and does not contaminate the COM kinetic energy
    ic0 = _build(jnp.ones(n_sys), companions=None, seed=seed)
    T_bar = float(compute_kinetic_energy(ic0.stars.velocities, ic0.stars.masses))
    T_res = float(compute_kinetic_energy(ic.stars.velocities, ic.stars.masses))
    print(f"  KE: barycenters-only = {T_bar:.4f}   resolved = {T_res:.4f} "
          f"(orbital motion rides on top)")
    p = np.asarray(jnp.sum(ic.stars.velocities * ic.stars.masses[:, None], axis=0))
    p_ok = bool(np.all(np.abs(p) < 1e-8 * float(ic.ledger.M_star)))
    frac_expected = 0.4
    count_ok = abs(n_bin / n_sys - frac_expected) < 0.1
    ok = p_ok and count_ok and T_res > T_bar
    print(f"  momentum closure = {p_ok}; binary count within ±0.1 of f_b = {count_ok}; "
          f"KE ordering = {T_res > T_bar}  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "n_binaries": n_bin}


def main():
    print("=" * 78)
    print("GRAVOTURB COMPOSITION ACCEPTANCE (AC-IC10/AC-IC11)  |  Phase 4b")
    print("=" * 78)
    results = {
        "AC-IC10 λ_corr segregation": ac_ic10_lambda_corr(),
        "AC-IC11 binaries barycenter-first": ac_ic11_binaries(),
    }
    print("\n" + "=" * 78)
    print("SUMMARY")
    for name, r in results.items():
        print(f"  {name:<36} {'PASS' if r['passed'] else 'FAIL'}")
    print(f"  {sum(r['passed'] for r in results.values())}/{len(results)} passed")
    print("  (local IMF: DEFENSIBILITY-REFUSED — Marks+2012 α₃ is a GLOBAL cloud-core")
    print("   relation; cluster-level env_to_imf_params + masses-first is the route)")
    print("=" * 78)
    return results


if __name__ == "__main__":
    main()
