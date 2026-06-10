# tests/validation/test_engine_b_physics.py
"""Physics validation: Engine B (prescribed-density shared-Psi Eddington) anchors.

Task 5 (2c-iii) of the Engine B plan -- the cross-engine and cross-family trust
anchors that carry the inversion-correctness burden the exact-quadrature Q_j
oracle cannot (that oracle is NECESSARY, not sufficient: 2T_j + W_j = 0 holds
for ANY positive f in a consistent (Psi, dPsi/dr) pair):

  - King A-vs-B: the SAME physical model built by two INDEPENDENT engines
    (Engine A: lowered-isothermal DF + coupled ODE; Engine B: prescribed King
    density + Poisson quadrature + Eddington inversion) must agree in r_t,
    theory Q_j, sampled sigma_1d(r), and sampled radial CDF. THE trust anchor.
  - EFF(gamma=5) == Plummer: a closed-form cross-family identity (the gamma=5
    EFF IS the Plummer density with a_Pl = a_EFF), so the two Engine B builds
    must agree to numerical-identity tolerances.
  - Plummer halo + EFF core: the science headline -- a two-family prescribed
    mix is a true shared-potential equilibrium, globally AND per component,
    UNSCALED.
  - OM anisotropy: a finite r_a on the halo realizes beta(r) = r^2/(r^2+r_a^2)
    in the sampled velocities while the isotropic core stays beta ~ 0.
  - DF-density fidelity: rho_DF,j reconstructed from the f_j tables matches
    the prescribed rho_j in each component's interior -- the direct
    inversion-fidelity gate the DF-weighted virial oracle does not cover.

The gates are the contract: if the King A-vs-B anchor disagrees beyond its
gates, one engine has a physics inconsistency -- STOP and diagnose, never
tune tolerances.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.units import STELLAR

from progenax import EFFProfile, KingProfile, PlummerProfile
from progenax.cluster.multicomponent import MultiComponentCluster

G = STELLAR.G


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _headline_model(**kw):
    """The science-headline mix: Plummer halo + EFF(gamma=5) core.

    The core scale a=0.8 is a REALIZABILITY constraint (a=0.4 has NO
    equilibrium: the cored halo density in the concentrated core's potential
    has a genuinely negative Eddington DF -- two independent oracles agree;
    see TestEngineB in tests/unit/cluster/test_multicomponent.py).
    """
    cfg = dict(
        profiles=[PlummerProfile(r_h=2.0), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)],
        mass_fractions=jnp.array([0.6, 0.4]),
        m_j=jnp.array([0.5, 1.0]),
    )
    cfg.update(kw)
    return MultiComponentCluster.from_density_profiles(**cfg)


def _com_arrays(ic):
    """COM-frame numpy (positions, velocities, masses) from an ICResult."""
    p = np.asarray(ic.positions - jnp.average(ic.positions, axis=0, weights=ic.masses))
    v = np.asarray(ic.velocities - jnp.average(ic.velocities, axis=0, weights=ic.masses))
    return p, v, np.asarray(ic.masses)


def _chunked_potential(pos, mass, chunk=2000):
    """Exact pairwise potential, row-chunked (a 30k^2 block is ~22 GB).

    Sums over ORDERED pairs i != j, so V = -G/2 * sum_{i!=j} m_i m_j / r_ij --
    the math of compute_potential_energy without the (N, N) materialization.
    """
    V2 = 0.0
    for i0 in range(0, pos.shape[0], chunk):
        p = pos[i0:i0 + chunk]
        d = np.sqrt(((p[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
        rows = np.arange(p.shape[0])
        d[rows, i0 + rows] = np.inf  # drop self-pairs
        V2 += float(np.sum(mass[i0:i0 + chunk, None] * mass[None, :] / d))
    return -0.5 * G * V2


def _chunked_accelerations(pos, mass, chunk=1000):
    """Direct-summation a_i = -G sum_k m_k (r_i - r_k)/r_ik^3, row-chunked."""
    acc = np.zeros_like(pos)
    for i0 in range(0, pos.shape[0], chunk):
        d = pos[i0:i0 + chunk, None, :] - pos[None, :, :]
        r2 = (d**2).sum(axis=2)
        rows = np.arange(d.shape[0])
        r2[rows, i0 + rows] = np.inf  # self term -> 0
        acc[i0:i0 + chunk] = -G * np.sum(mass[None, :, None] * d * r2[:, :, None] ** -1.5,
                                         axis=1)
    return acc


def _sampled_component_Q(model, seed, n_stars):
    """Sampled per-component Q_j = T_j/|W_j| in the TOTAL field (Clausius)."""
    ic = model.sample_cluster(jax.random.PRNGKey(seed), n_stars=n_stars, G=G)
    p, v, mass = _com_arrays(ic)
    cid = np.asarray(ic.component_id)
    a = _chunked_accelerations(p, mass)
    T_i = 0.5 * mass * np.sum(v**2, axis=1)
    W_i = mass * np.sum(p * a, axis=1)
    return np.array([T_i[cid == j].sum() / abs(W_i[cid == j].sum())
                     for j in range(int(cid.max()) + 1)])


# ---------------------------------------------------------------------------
# 1. THE trust anchor: King by Engine A vs King by Engine B
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_king_density_engine_b_matches_engine_a():
    """One physical model, two INDEPENDENT engines.

    Engine A: W0=5, g=1 lowered-isothermal DF through the coupled Poisson ODE.
    Engine B: the SAME King model as a prescribed density (KingProfile,
    W0=5, r_c=1) -> quadrature Psi -> Eddington inversion. Every shared
    construction step differs (ODE vs cumulative trapezoid; closed-form DF vs
    Abel integral), so agreement here is a genuine cross-validation of both.

    Gates (the contract -- never loosen):
      - r_t agreement rtol 1e-3 (same physical r_c = 1; A's natural ODE r_t
        vs B's KingProfile r_t);
      - theory Q_j = 0.5 +- 3e-3 in BOTH engines;
      - sampled sigma_1d(r): |sigma_B/sigma_A - 1| < 0.02 in interior bins
        (N = 2e4 each, same seed -- the shared position/speed key structure
        makes the draws strongly correlated, isolating model differences);
      - sampled radial CDFs: two-sample KS distance < 0.02.
    """
    king = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)
    mB = MultiComponentCluster.from_density_profiles(
        [king], jnp.array([1.0]), m_j=jnp.array([1.0]))
    mA = MultiComponentCluster.from_components(
        alpha_j=jnp.array([1.0]), w_j=jnp.array([1.0]), m_j=jnp.array([1.0]),
        W0=5.0, g=1.0, r_c=1.0)

    # Domain: like with like -- one King model, one tidal radius.
    np.testing.assert_allclose(float(mA.r_t), float(mB.r_t), rtol=1e-3,
                               err_msg="A and B disagree on the King r_t")

    for m, name in ((mA, "A"), (mB, "B")):
        Qj = np.asarray(m.component_virial_ratios())
        np.testing.assert_allclose(
            Qj, 0.5, atol=3e-3, err_msg=f"engine {name} theory Q_j = {Qj}")

    N = 20000
    key = jax.random.PRNGKey(0)
    icA = mA.sample_cluster(key, n_stars=N, G=G)
    icB = mB.sample_cluster(key, n_stars=N, G=G)
    _, vA, _ = _com_arrays(icA)
    _, vB, _ = _com_arrays(icB)
    rA = np.asarray(jnp.linalg.norm(icA.positions, axis=1))
    rB = np.asarray(jnp.linalg.norm(icB.positions, axis=1))
    v2A = np.sum(vA**2, axis=1)
    v2B = np.sum(vB**2, axis=1)

    # Sampled radial CDFs: two-sample KS distance.
    grid = np.sort(np.concatenate([rA, rB]))
    ks = float(np.max(np.abs(
        np.searchsorted(np.sort(rA), grid, side="right") / N
        - np.searchsorted(np.sort(rB), grid, side="right") / N)))
    assert ks < 0.02, f"King A-vs-B radial KS distance {ks:.4f} >= 0.02"

    # sigma_1d(r) in interior quantile bins (5%-90% of the A radii).
    edges = np.quantile(rA, np.linspace(0.05, 0.90, 7))
    for lo, hi in zip(edges[:-1], edges[1:]):
        selA = (rA >= lo) & (rA < hi)
        selB = (rB >= lo) & (rB < hi)
        sigA = np.sqrt(v2A[selA].mean() / 3.0)
        sigB = np.sqrt(v2B[selB].mean() / 3.0)
        dev = abs(sigB / sigA - 1.0)
        assert dev < 0.02, (
            f"sigma_1d mismatch in bin [{lo:.2f}, {hi:.2f}): "
            f"sigma_B/sigma_A - 1 = {sigB / sigA - 1.0:+.4f}")


# ---------------------------------------------------------------------------
# 2. Cross-family identity: EFF(gamma=5) IS Plummer
# ---------------------------------------------------------------------------


def test_eff_gamma5_single_component_matches_plummer():
    """gamma=5 EFF == Plummer with a_Pl = a_EFF (closed-form identity).

    rho_EFF = (1 + r^2/a^2)^{-5/2} is EXACTLY the Plummer density shape, so a
    single-EFF Engine B build and a single-Plummer Engine B build truncated at
    the same r_t are the SAME prescribed model up to normalization (which the
    mass-fraction scaling removes): the shared machinery must produce
    numerically identical Psi grids (rtol 1e-8) and f tables (rtol 1e-5).
    The Plummer r_h is the exact inverse of PlummerProfile's a(r_h) relation
    a = r_h sqrt(2^(2/3) - 1); the explicit r_t=12 override truncates the
    (infinite) Plummer at the EFF extent (the King-conflict rule does not
    apply to Plummer).
    """
    a_eff = 1.0
    r_h = a_eff / float(jnp.sqrt(2.0 ** (2.0 / 3.0) - 1.0))
    plummer = PlummerProfile(r_h=r_h)
    np.testing.assert_allclose(float(plummer.a), a_eff, rtol=1e-12)

    m_eff = MultiComponentCluster.from_density_profiles(
        [EFFProfile(a=a_eff, gamma=5.0, r_t=12.0)],
        jnp.array([1.0]), m_j=jnp.array([1.0]))
    m_pl = MultiComponentCluster.from_density_profiles(
        [plummer], jnp.array([1.0]), m_j=jnp.array([1.0]), r_t=12.0)

    assert float(m_eff.r_t) == 12.0 and float(m_pl.r_t) == 12.0

    np.testing.assert_allclose(
        np.asarray(m_pl.engine_b.Psi_poisson),
        np.asarray(m_eff.engine_b.Psi_poisson), rtol=1e-8,
        err_msg="EFF(gamma=5) and Plummer shared-Psi grids differ")

    f_eff = np.asarray(m_eff.engine_b.f_j_grid)
    f_pl = np.asarray(m_pl.engine_b.f_j_grid)
    np.testing.assert_allclose(
        f_pl, f_eff, rtol=1e-5, atol=1e-10 * np.max(np.abs(f_eff)),
        err_msg="EFF(gamma=5) and Plummer Eddington f tables differ")

    Q_eff = np.asarray(m_eff.component_virial_ratios())
    Q_pl = np.asarray(m_pl.component_virial_ratios())
    np.testing.assert_allclose(Q_eff, 0.5, atol=3e-3)
    np.testing.assert_allclose(Q_pl, 0.5, atol=3e-3)
    np.testing.assert_allclose(Q_pl, Q_eff, atol=1e-8,
                               err_msg="theory Q_j differs between the twins")


# ---------------------------------------------------------------------------
# 3. The science headline: Plummer halo + EFF core equilibrium
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_plummer_halo_eff_core_equilibrium():
    """The headline: a two-family prescribed mix is a TRUE equilibrium.

    Gates:
      - theory Q_j = 0.5 +- 3e-3 for both components (exact-quadrature oracle);
      - sampled global Q within 0.02 of 0.5 UNSCALED at N=30k (the expected
        value is ~0.496-0.498: genuine truncation-edge physics of the
        hard-truncated halo -- the documented truncated-empirical-profile
        approximation -- not a pipeline bias);
      - sampled per-component Q_j converge toward the 0.5 theory with N
        (loose trend over two N values, seed-averaged).
    """
    m = _headline_model()

    Qj = np.asarray(m.component_virial_ratios())
    np.testing.assert_allclose(Qj, 0.5, atol=3e-3,
                               err_msg=f"headline theory Q_j = {Qj}")

    ic = m.sample_cluster(jax.random.PRNGKey(0), n_stars=30000, G=G)
    p, v, mass = _com_arrays(ic)
    T = 0.5 * float(np.sum(mass * np.sum(v**2, axis=1)))
    V = _chunked_potential(p, mass)
    Q = T / abs(V)
    assert abs(Q - 0.5) < 0.02, f"headline global Q = {Q:.4f} (unscaled)"

    # Loose convergence trend: per-component sampled Q_j error shrinks (or
    # stays within noise) from N=4k to N=16k, and is small at N=16k.
    errs = {}
    for n in (4000, 16000):
        Q_seeds = np.stack([_sampled_component_Q(m, seed, n) for seed in (1, 2)])
        errs[n] = np.abs(Q_seeds.mean(axis=0) - 0.5)
    assert np.all(errs[16000] < errs[4000] + 0.01), (
        f"per-component |Q_j - 0.5| did not converge: "
        f"N=4000 -> {errs[4000]}, N=16000 -> {errs[16000]}")
    assert np.all(errs[16000] < 0.04), (
        f"per-component |Q_j - 0.5| at N=16000: {errs[16000]}")


# ---------------------------------------------------------------------------
# 4. Osipkov-Merritt anisotropy realized in the sampled velocities
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_om_beta_profile_realized():
    """OM on the halo only: sampled beta_halo(r) == r^2/(r^2 + r_a^2).

    Realizability (probed before writing the gate): r_a = 3.0 on the a=0.8
    halo+core mix gives f_min_j = [+0.085, +1.2e-4] -- comfortably realizable,
    so the planned r_a = 3.0 is used as-is (no relaxation needed).

    Gates: pooled over 4 seeds x 20k stars, the halo's sampled
    beta = 1 - <v_t^2>/(2 <v_r^2>) tracks the OM curve within 0.05 in every
    resolved interior bin; the isotropic core stays |beta| < 0.05.
    """
    r_a = 3.0
    m = _headline_model(r_a_j=jnp.array([r_a, jnp.inf]))

    r_all, vr2_all, vt2_all, cid_all = [], [], [], []
    for seed in range(4):
        ic = m.sample_cluster(jax.random.PRNGKey(seed), n_stars=20000, G=G)
        pos = np.asarray(ic.positions)
        vel = np.asarray(ic.velocities)
        r = np.linalg.norm(pos, axis=1)
        v_r = np.sum(pos * vel, axis=1) / np.maximum(r, 1e-30)
        v2 = np.sum(vel**2, axis=1)
        r_all.append(r)
        vr2_all.append(v_r**2)
        vt2_all.append(v2 - v_r**2)
        cid_all.append(np.asarray(ic.component_id))
    r = np.concatenate(r_all)
    vr2 = np.concatenate(vr2_all)
    vt2 = np.concatenate(vt2_all)
    cid = np.concatenate(cid_all)

    def beta_in_bins(sel, edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = sel & (r >= lo) & (r < hi)
            assert s.sum() > 2000, f"unresolved bin [{lo:.2f}, {hi:.2f})"
            beta = 1.0 - vt2[s].mean() / (2.0 * vr2[s].mean())
            out.append((beta, r[s]))
        return out

    # Halo: 8 interior quantile bins (5%-95% of the pooled halo radii).
    halo = cid == 0
    edges = np.quantile(r[halo], np.linspace(0.05, 0.95, 9))
    for beta, r_bin in beta_in_bins(halo, edges):
        expect = float(np.mean(r_bin**2 / (r_bin**2 + r_a**2)))
        assert abs(beta - expect) < 0.05, (
            f"halo beta at <r>={r_bin.mean():.2f}: measured {beta:+.3f}, "
            f"OM expects {expect:+.3f}")

    # Core (r_a = inf): isotropic everywhere it is resolved.
    core = cid == 1
    edges = np.quantile(r[core], np.linspace(0.05, 0.95, 5))
    for beta, r_bin in beta_in_bins(core, edges):
        assert abs(beta) < 0.05, (
            f"core beta at <r>={r_bin.mean():.2f}: measured {beta:+.3f}, "
            f"expected ~0 (isotropic)")


# ---------------------------------------------------------------------------
# 5. Direct inversion-fidelity gate: rho_DF,j == rho_presc,j in the interior
# ---------------------------------------------------------------------------


def test_df_density_fidelity_interior():
    """The DF must integrate back to the density it was inverted from.

    For the headline mix, reconstruct on the full Poisson grid

        rho_DF,j(r) = (1 + r^2/r_a_j^2)^{-1} 4 pi int_0^sqrt(2 Psi) w^2
                      f_j(Psi(r) - w^2/2) dw

    (w the OM stretched-frame speed; the factor is the d^3v = d^3w/(1+r^2/r_a^2)
    measure, = 1 here since the mix is isotropic) and gate
    |rho_DF,j/rho_presc,j - 1| < 5e-3 for r < r_h of EACH component -- the
    known constant rho(r_t) offset (the edge term no ergodic f(E) can carry)
    dominates only near the truncation edge. This is the direct
    inversion-fidelity gate the DF-weighted virial oracle does not cover.
    """
    m = _headline_model()
    st = m.engine_b
    r = np.asarray(st.r_poisson)
    Psi = st.Psi_poisson
    n_w = 400

    def rho_df_row(f_row):
        def m0(Psi_r):
            w = jnp.linspace(0.0, jnp.sqrt(2.0 * jnp.maximum(Psi_r, 1e-12)), n_w)
            f_at = jnp.maximum(
                jnp.interp(Psi_r - 0.5 * w**2, st.E_grid, f_row), 0.0)
            return jnp.trapezoid(w**2 * f_at, w)
        return 4.0 * np.pi * np.asarray(jax.vmap(m0)(Psi))

    for j in range(2):
        ra = float(st.r_a_j[j])
        inv_st2 = 1.0 / (1.0 + (r / ra) ** 2) if np.isfinite(ra) else 1.0
        rho_df = rho_df_row(st.f_j_grid[j]) * inv_st2
        rho_presc = np.asarray(st.rho_j_poisson[j])

        # component half-mass radius from its own position CDF
        r_h_j = float(jnp.interp(0.5, m._cdf_j[j], m._r_grid))
        sel = r < r_h_j
        dev = np.max(np.abs(rho_df[sel] / rho_presc[sel] - 1.0))
        assert dev < 5e-3, (
            f"component {j}: max |rho_DF/rho_presc - 1| = {dev:.2e} "
            f"for r < r_h = {r_h_j:.3f}")
