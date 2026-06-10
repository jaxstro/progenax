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
    mix is a true shared-potential equilibrium, globally UNSCALED; the sampled
    per-component Q_j is gated against its exact-quadrature hybrid prediction
    (the halo's truncation-edge plateau below 0.5 is verified physics).
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
    """Sampled per-component Q_j = T_j/|W_j| in the TOTAL field (Clausius).

    Estimator definition (the prediction in _predicted_component_Q MUST match
    it): T_j = (1/2) sum_{i in j} m_i v_i^2 with v_i drawn from f_j at the
    star's radius; W_j = sum_{i in j} m_i r_i . a_i with a_i the direct
    pairwise acceleration of ALL sampled stars -- whose continuum limit is the
    PRESCRIBED-total field G M_presc(<r)/r^2, because positions are drawn from
    the prescribed rho_j (via _cdf_j), not from rho_DF,j.
    """
    ic = model.sample_cluster(jax.random.PRNGKey(seed), n_stars=n_stars, G=G)
    p, v, mass = _com_arrays(ic)
    cid = np.asarray(ic.component_id)
    a = _chunked_accelerations(p, mass)
    T_i = 0.5 * mass * np.sum(v**2, axis=1)
    W_i = mass * np.sum(p * a, axis=1)
    return np.array([T_i[cid == j].sum() / abs(W_i[cid == j].sum())
                     for j in range(int(cid.max()) + 1)])


def _predicted_component_Q(model, n_w=400):
    """Exact-quadrature HYBRID expectation of _sampled_component_Q.

    Engine B samples hybrid clusters: positions from the prescribed rho_j,
    speeds from the Eddington f_j in the shared Psi. The continuum expectation
    of the sampled estimator is therefore

        T_j^pred = (1/2) int rho_presc,j(r) <v^2>_DF,j(r) 4 pi r^2 dr,
        W_j^pred = -int rho_presc,j(r) r (M_presc(<r)/r^2) 4 pi r^2 dr,

    with <v^2>_DF,j = (m2/m0)(1/3 + (2/3)/(1 + r^2/r_a_j^2)) from the f_j
    speed moments m0 = int w^2 f_j dw, m2 = int w^4 f_j dw (w the OM
    stretched-frame speed, f clamped at 0 exactly as the speed sampler clamps
    grid ringing). This differs from engine_b_component_virials ONLY in the
    weights: rho_presc,j and the prescribed-total dPhi/dr, not rho_DF,j --
    because that is what the sampler realizes. For a hard-truncated component
    rho_presc(r_t) > 0 is an edge offset NO ergodic f(E) can carry, so
    Q_j^pred sits genuinely below 0.5: a quantitative prediction, not a bias.
    """
    st = model.engine_b
    r, Psi = st.r_poisson, st.Psi_poisson
    dphi_dr = -st.dPsi_dr_poisson           # = +M_presc(<r)/r^2 (G=1)
    Psi_safe = jnp.maximum(Psi, 1e-12)

    def moments(Psi_r, f_row):
        w = jnp.linspace(0.0, jnp.sqrt(2.0 * Psi_r), n_w)
        f_at = jnp.maximum(jnp.interp(Psi_r - 0.5 * w**2, st.E_grid, f_row), 0.0)
        return jnp.trapezoid(w**2 * f_at, w), jnp.trapezoid(w**4 * f_at, w)

    out = []
    for j in range(st.f_j_grid.shape[0]):
        m0, m2 = jax.vmap(lambda P: moments(P, st.f_j_grid[j]))(Psi_safe)
        finite = jnp.isfinite(st.r_a_j[j])
        ra_safe = jnp.where(finite, st.r_a_j[j], 1.0)
        inv_st2 = jnp.where(finite, 1.0 / (1.0 + (r / ra_safe) ** 2), 1.0)
        v2 = (m2 / (m0 + 1e-300)) * (1.0 / 3.0 + (2.0 / 3.0) * inv_st2)
        rho_p = st.rho_j_poisson[j]
        T = jnp.trapezoid(0.5 * rho_p * v2 * 4.0 * jnp.pi * r**2, r)
        W = jnp.trapezoid(-rho_p * r * dphi_dr * 4.0 * jnp.pi * r**2, r)
        out.append(float(T / jnp.abs(W)))
    return np.array(out)


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
        vs B's KingProfile r_t). NOTE: this sub-check is partially CORRELATED
        -- both r_t values descend from the same King-ODE machinery -- so it
        is a consistency check only; the sigma_1d and KS gates below carry
        the independent-anchor weight;
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
    Near-bit-identity is EXPECTED here -- at gamma=5 both builds execute the
    same arithmetic through the same shared machinery -- so this is a
    branch-reduction + domain-path regression test, NOT an
    independent-machinery anchor (that role belongs to King A-vs-B).
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
      - theory Q_j = 0.5 +- 3e-3 for both components (DF-weighted
        exact-quadrature oracle: the DF is consistent with the potential);
      - sampled global Q within 0.02 of 0.5 UNSCALED at N=30k;
      - PREDICT-THE-OFFSET gate (amended 2026-06-10, Anna-approved): the
        sampled per-component Q_j does NOT converge to 0.5. Engine B samples
        a HYBRID: positions from the prescribed rho_j, speeds from f_j in the
        shared Psi. The hard-truncated Plummer halo has rho(r_t) > 0 -- a
        constant edge offset NO ergodic f(E) can carry (the Eddington pair
        represents rho(Psi) - rho(0)) -- so rho_DF,halo != rho_presc,halo
        near the edge and the halo's sampled Q_j plateaus BELOW 0.5. The
        plateau is a VERIFIED PREDICTION: _predicted_component_Q computes the
        exact-quadrature expectation of the sampled estimator (rho_presc
        weights x DF speed moments x prescribed-total Clausius field) and the
        sampled values must match IT, not 0.5. Evidence (18 seeds, N=16k):
        halo predicted 0.4953 vs sampled 0.4947 +- 0.0014 (sem, 0.4 sigma);
        core predicted 0.4985 vs sampled 0.5007 +- 0.0018 (1.2 sigma);
        per-seed scatter sigma ~ 0.006 (halo) / 0.008 (core). The historical
        "plateau ~ 0.484" was a 2-seed downward fluctuation, not physics.
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

    # The physics statement, gated on the PREDICTION itself: the nearly
    # untruncated core (rho_EFF(r_t)/rho_EFF(0) ~ 5e-6) is at 0.5; the
    # hard-truncated halo is visibly below (offset >= 1e-3, well above the
    # ~4e-4 quadrature accuracy the DF-weighted oracle demonstrates).
    Q_pred = _predicted_component_Q(m)
    assert abs(Q_pred[1] - 0.5) < 5e-3, (
        f"core predicted Q = {Q_pred[1]:.4f}, expected ~0.5")
    assert 0.47 < Q_pred[0] < 0.499, (
        f"halo predicted Q = {Q_pred[0]:.4f}, expected visibly below 0.5 "
        f"(truncation-edge offset)")

    # Sampled-vs-predicted: 3 seeds at N=16k. Tolerance 0.012 ~ 3 sigma on
    # the 3-seed mean (sem ~ 0.0035-0.0046 from the measured per-seed
    # scatter above) -- shot noise only, never a tuned offset.
    Q_seeds = np.stack([_sampled_component_Q(m, seed, 16000)
                        for seed in (1, 2, 3)])
    Q_meas = Q_seeds.mean(axis=0)
    np.testing.assert_allclose(
        Q_meas, Q_pred, atol=0.012,
        err_msg=(f"sampled per-component Q_j {Q_meas} does not match the "
                 f"hybrid exact-quadrature prediction {Q_pred}"))


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
    """The DF must integrate back to the AUGMENTED density it was inverted from.

    The Eddington/OM pair represents rho_Q(Psi) - rho_Q(0), where
    rho_Q(r) = (1 + r^2/r_a^2) rho(r) is the Osipkov-Merritt augmented density
    (factor 1 on isotropic components). For a hard-truncated component
    rho_Q(0) = rho_Q(r_t) = (1 + r_t^2/r_a^2) rho(r_t) is an edge offset NO
    ergodic f can carry -- and the OM augmentation AMPLIFIES the isotropic
    edge offset by (1 + r_t^2/r_a^2): for the halo (r_t = 9, r_a = 3) that is
    a 10x amplification. The earlier form of this test (divide out the OM
    measure, compare to rho_presc directly) failed at 6.6e-3 on the OM halo
    purely from this PREDICTED offset: the measured raw deficit
    rho_Q,presc - rho_Q,DF ~ 4.1-5.0e-5 across interior radii matches the
    predicted constant rho_Q(r_t) = 5.54e-5 (verified ~constant in r). So the
    gate must be stated against the continuum statement the DF can actually
    satisfy:

        rho_Q,DF(r) = 4 pi int_0^sqrt(2 Psi) w^2 f_j(Psi(r) - w^2/2) dw
                      (w the OM stretched-frame speed; NO measure division)
        ==  rho_Q,presc(r) - rho_Q,presc(r_t),
        rho_Q,presc = (1 + r^2/r_a_j^2) rho_presc,j.

    The SAME form is used for both builds and both components: for the
    isotropic build and the near-untruncated core
    (rho_EFF(r_t)/rho_EFF(0) ~ 5e-6) the subtraction is negligible, so this
    is one uniform statement, not a special case. n_w = 1200: at n_w = 400
    the w_max-endpoint under-resolution partially MASKED the deficit (the
    measured deviation grows 6.6e-3 -> 9.0e-3 with n_w under the old form);
    the corrected statement needs the honest quadrature. Gate
    |rho_Q,DF/(rho_Q,presc - rho_Q,presc(r_t)) - 1| < 5e-3 for r < r_h of
    EACH component -- UNCHANGED from the old form, now with ~5x margin:
    measured worst case 1.06e-3 (OM build), 2.4e-4 (isotropic build, same
    form). This is the direct inversion-fidelity gate the DF-weighted virial
    oracle does not cover.
    """
    n_w = 1200
    builds = (
        ("isotropic", _headline_model()),
        ("OM r_a=3.0", _headline_model(r_a_j=jnp.array([3.0, jnp.inf]))),
    )
    for label, m in builds:
        st = m.engine_b
        r = np.asarray(st.r_poisson)
        Psi = st.Psi_poisson

        def rho_q_df_row(f_row, E_grid=st.E_grid, Psi=Psi):
            def m0(Psi_r):
                w = jnp.linspace(0.0, jnp.sqrt(2.0 * jnp.maximum(Psi_r, 1e-12)),
                                 n_w)
                f_at = jnp.maximum(
                    jnp.interp(Psi_r - 0.5 * w**2, E_grid, f_row), 0.0)
                return jnp.trapezoid(w**2 * f_at, w)
            return 4.0 * np.pi * np.asarray(jax.vmap(m0)(Psi))

        for j in range(2):
            ra = float(st.r_a_j[j])
            # OM augmentation (1 + r^2/r_a^2); factor 1 for infinite r_a
            # (no division by inf -- the isotropic path needs the unit factor).
            aug = (1.0 + (r / ra) ** 2) if np.isfinite(ra) else np.ones_like(r)
            rho_q_df = rho_q_df_row(st.f_j_grid[j])
            rho_q_presc = aug * np.asarray(st.rho_j_poisson[j])
            # Truncation-consistent target: the DF represents
            # rho_Q(Psi) - rho_Q(0), so subtract the edge value at r_t.
            rho_q_target = rho_q_presc - rho_q_presc[-1]

            # component half-mass radius from its own position CDF
            r_h_j = float(jnp.interp(0.5, m._cdf_j[j], m._r_grid))
            sel = r < r_h_j
            dev = np.max(np.abs(rho_q_df[sel] / rho_q_target[sel] - 1.0))
            assert dev < 5e-3, (
                f"[{label}] component {j}: max |rho_Q,DF/(rho_Q,presc - "
                f"rho_Q,presc(r_t)) - 1| = {dev:.2e} for r < r_h = {r_h_j:.3f}")


# ---------------------------------------------------------------------------
# 6. Differentiability contract: AD == FD through the full Engine B build
# ---------------------------------------------------------------------------


def test_gradients_ad_vs_fd():
    """jax.grad through the FULL Engine B construction (Poisson quadrature +
    Eddington inversion) matches central finite differences to rtol 1e-3
    (the contract; measured agreement is ~1e-6 or better) for THREE physical
    parameters, each finite AND nonzero:

      (a) halo r_h        -- the headline Plummer+EFF mix; grad of the halo's
                             Plummer scale through density -> Psi -> f;
      (b) mass fraction t -- reparametrized fractions [t, 1-t] (sum-to-1 by
                             construction) on a KING + Plummer mix, so the
                             Poisson-identity King dW/dr path
                             (_king_drho_dW + the cumtrap dpsi/dxi route in
                             density_poisson._density_and_derivative) is
                             INSIDE the differentiated graph (Task 5 review);
      (c) r_a_j[0] = 3.0  -- finite OM anisotropy radius on the King
                             component of the same mix (grad through the
                             augmented-density weight in eddington_invert).

    Scalar: mean(Psi_poisson) + mean(f_j_grid[0]) -- smooth, and it exercises
    BOTH the shared potential and the component-0 inversion ((c)'s Psi does
    not depend on r_a, so the f term carries that gradient). Resolution is
    reduced to (n_r, n_e) = (3000, 500): AD-vs-FD identity is a property of
    the computational graph, not of quadrature convergence, and both sides
    use the same grids. FD step: central differences, h = 1e-4 |x0| (the
    test_limepy_tables FD pattern). Both mixes are realizable at the FD-
    perturbed points (f_min_j > 0 measured), so the concrete-path gate stays
    silent. NEVER weaken the 1e-3 rtol -- a failure here means a non-smooth
    op entered the build graph.
    """
    n_r, n_e = 3000, 500
    king = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)

    def scalar(state):
        return jnp.mean(state.Psi_poisson) + jnp.mean(state.f_j_grid[0])

    def loss_rh(x):
        from progenax.cluster.eddington_engine import build_engine_b_state
        state, _ = build_engine_b_state(
            [PlummerProfile(r_h=x), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)],
            jnp.array([0.6, 0.4]), jnp.array([jnp.inf, jnp.inf]),
            None, 0.995, n_r, n_e)
        return scalar(state)

    def loss_t(t):
        from progenax.cluster.eddington_engine import build_engine_b_state
        state, _ = build_engine_b_state(
            [king, PlummerProfile(r_h=2.0)], jnp.stack([t, 1.0 - t]),
            jnp.array([3.0, jnp.inf]), None, 0.995, n_r, n_e)
        return scalar(state)

    def loss_ra(ra):
        from progenax.cluster.eddington_engine import build_engine_b_state
        state, _ = build_engine_b_state(
            [king, PlummerProfile(r_h=2.0)], jnp.array([0.5, 0.5]),
            jnp.stack([ra, jnp.inf]), None, 0.995, n_r, n_e)
        return scalar(state)

    for name, loss, x0 in (("halo r_h", loss_rh, 2.0),
                           ("mass fraction t", loss_t, 0.5),
                           ("r_a_j[0]", loss_ra, 3.0)):
        ad = float(jax.grad(loss)(jnp.asarray(x0)))
        assert np.isfinite(ad), f"{name}: AD gradient not finite ({ad})"
        assert ad != 0.0, f"{name}: AD gradient is exactly zero"
        h = 1e-4 * abs(x0)
        fd = (float(loss(jnp.asarray(x0 + h)))
              - float(loss(jnp.asarray(x0 - h)))) / (2.0 * h)
        np.testing.assert_allclose(
            ad, fd, rtol=1e-3,
            err_msg=f"{name}: AD {ad:.10e} vs FD {fd:.10e}")
