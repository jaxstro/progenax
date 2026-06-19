"""Phase-0 (Task 0.2) de-risk tests for the binary-misspecification OED arc.

Demo-harness tier — NOT released-core (scripts-only, no ``src/progenax/`` change).
These PIN the two velocity scales of the binary-inflation forward model and confirm
the H1 bias can bite at the YMC operating point:

* ``V_BIN`` — the flux-weighted binary blend-velocity variance Var(K_orb) for the
  OBSERVED massive-primary population (Moe & Di Stefano P-q-e, M1 >= 2 Msun), built
  ONCE at import; and
* ``sigma_cluster_ref()`` — the central (peak) EFF-OM line-of-sight dispersion of the
  fiducial young-massive cluster, in km/s.

H1 needs ``sigma_bin / sigma_cluster`` large enough that binaries rival cluster heat;
the gate is ``ratio > 0.5``. The threshold is NOT to be weakened — if the physical YMC
operating point gives ``ratio <= 0.5`` the test must FAIL and the number is reported as
a finding (the Phase-4 sweep spans the ratio).

See docs/plans/2026-06-19-oed-binary-misspecification-{plan,design}.md.
"""
import os
import sys
import pathlib

import jax.numpy as jnp
import progenax  # noqa: F401  -- enables float64 at import
from jaxstro.units import STELLAR

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import _demo_oed as oed  # noqa: E402  -- the Stage-1 c-criterion (reused by the marg gates)
import _demo_oed_binary as oedb  # noqa: E402


def test_vbin_and_sigma_ratio_bites():
    """V_bin sane (finite, positive) and sigma_bin / sigma_cluster > 0.5 (H1 can bite).

    sigma_bin = sqrt(Var(K_orb)) for the massive-primary Moe population; sigma_cluster
    is the central (peak) EFF-OM sigma_los of the fiducial YMC. Conservative choice
    (central is the LARGEST sigma_los), so if H1 bites at the centre it bites everywhere.
    """
    V_bin = oedb.V_BIN
    assert jnp.isfinite(V_bin) and V_bin > 0.0
    sig_bin = float(jnp.sqrt(V_bin))
    sig_cluster = float(oedb.sigma_cluster_ref())
    ratio = sig_bin / sig_cluster
    assert ratio > 0.5, f"sigma_bin/sigma_cluster={ratio:.3f} too small for H1 to bite"


def test_eff_profile_feeds_project_dispersion_positive_finite():
    """The eff_profile(...) helper feeds project_dispersion without error, and the
    line-of-sight dispersion is positive and finite at every fiducial R bin."""
    from progenax import project_dispersion

    prof = oedb.eff_profile()
    pd = project_dispersion(prof, oedb.R_A_FID, oedb.R_BINS, oedb.M_FID, STELLAR.G)
    sig_los = jnp.asarray(pd.sigma_los)
    assert sig_los.shape == (oedb.R_BINS.shape[0],)
    assert bool(jnp.all(jnp.isfinite(sig_los)))
    assert bool(jnp.all(sig_los > 0.0))


# ---------------------------------------------------------------------------
# Task 1.1: EFF-OM RV-only cluster forward model (cluster_sigma_los)
# ---------------------------------------------------------------------------
def test_cluster_sigma_los_matches_project_dispersion():
    """Per-bin cluster_sigma_los(theta_clusteronly, R, G) equals the direct
    project_dispersion(...).sigma_los oracle (km/s), is everywhere positive, and the
    sigma_los profile DECLINES from the dense core to the outskirts (radial leverage:
    M info lives in the high-sigma core, the flat binary pedestal dominates in the
    low-sigma outskirts)."""
    from progenax import project_dispersion

    th = oedb.theta_truth_clusteronly()           # (M, r_a, gamma, a)
    R = oedb.R_BINS
    sig = oedb.cluster_sigma_los(th, R, STELLAR.G)  # (K,) km/s, RV channel only

    # Oracle: project_dispersion on the SAME EFF-OM model, converted pc/Myr -> km/s.
    prof = oedb.eff_profile(gamma=oedb.th_gamma(th), a=oedb.th_a(th), r_t=oedb.R_T_FID)
    sig_ref = oedb.kms(
        project_dispersion(prof, oedb.th_ra(th), R, oedb.th_M(th), STELLAR.G).sigma_los
    )

    assert sig.shape == (R.shape[0],)
    assert jnp.allclose(sig, sig_ref, rtol=1e-10)
    assert bool(jnp.all(sig > 0.0))
    # Radial leverage: the PROJECTED sigma_los of an EFF-OM model has a mild interior
    # peak (the deep-core LOS integral samples the full radial column, so the central
    # projected dispersion is slightly BELOW the peak at R ~ a), then DECLINES steeply
    # to the outskirts. The leverage that breaks M<->f_bin is this large core->outskirt
    # contrast: the inner bins are hot (M-dominated), the outer bins are cold (where the
    # flat binary pedestal dominates). Assert (i) a big drop core->outskirts and
    # (ii) strict decline past the interior peak (NOT strict monotonicity from bin 0,
    # which would encode the wrong physics).
    assert float(sig[0]) > 5.0 * float(sig[-1])         # >5x core->outskirt contrast
    peak = int(jnp.argmax(sig))
    assert peak < sig.shape[0] - 1                       # the peak is interior, not the last bin
    assert bool(jnp.all(jnp.diff(sig[peak:]) < 0.0))     # strictly declining past the peak


def test_sigma_cluster_ref_is_max_of_cluster_sigma_los():
    """sigma_cluster_ref() is the single source of truth = max over bins of
    cluster_sigma_los at the fiducial theta (the central / peak dispersion)."""
    th = oedb.theta_truth_clusteronly()
    sig = oedb.cluster_sigma_los(th, oedb.R_BINS, STELLAR.G)
    assert jnp.isclose(oedb.sigma_cluster_ref(), jnp.max(sig), rtol=1e-12)


# ---------------------------------------------------------------------------
# Task 1.2: binary-inflated observable + ONE jacrev (ln-theta, ADR 0011)
# ---------------------------------------------------------------------------
def test_predict_sigma_obs_adds_binary_pedestal():
    """sigma_obs^2 = sigma_cluster^2 + f_bin * V_BIN (the flat binary pedestal),
    exactly (rtol 1e-10), and the observable is everywhere larger than the bare
    cluster dispersion (binaries inflate the second moment)."""
    th = oedb.theta_truth()                       # (M, r_a, gamma, a, f_bin)
    R = oedb.R_BINS
    sig_obs = oedb.predict_sigma_obs(th, R, STELLAR.G)        # (K,) km/s
    sig_cluster = oedb.cluster_sigma_los(th[:4], R, STELLAR.G)
    expected2 = sig_cluster ** 2 + oedb.th_fbin(th) * oedb.V_BIN
    assert sig_obs.shape == (R.shape[0],)
    assert jnp.allclose(sig_obs ** 2, expected2, rtol=1e-10)
    assert bool(jnp.all(sig_obs > sig_cluster))   # pedestal strictly inflates


def test_jacobian_lntheta_shape_and_fbin_concentrates_in_outskirts():
    """The ONE reverse-mode jacrev J = d sigma_obs / d ln theta is (K, 5), finite,
    and the f_bin column concentrates info in the OUTSKIRTS: |J[:, IDX_FBIN]| is
    larger at the outermost (cold) bin than the innermost (hot) bin, because
    d sigma_obs / d ln f_bin = f_bin * V_BIN / (2 sigma_obs) grows as sigma_obs
    falls toward the outskirts -- the leverage that breaks M<->f_bin."""
    th = oedb.theta_truth()
    R = oedb.R_BINS
    J = oedb.jacobian_lntheta(th, R, STELLAR.G)   # (K, 5)
    assert J.shape == (R.shape[0], 5)
    assert bool(jnp.all(jnp.isfinite(J)))
    fcol = J[:, oedb.IDX_FBIN]
    assert float(jnp.abs(fcol[-1])) > float(jnp.abs(fcol[0]))


def test_jacobian_lntheta_clusteronly_chain_rule_to_full():
    """The cluster-only jacrev (d sigma_CLUSTER / d ln theta, the binary-free model)
    relates to the first four columns of the full jacrev (d sigma_OBS / d ln theta) by
    the exact chain-rule factor sigma_cluster / sigma_obs, since
    sigma_obs^2 = sigma_cluster^2 + f_bin*V_bin =>
    d sigma_obs / d ln theta_i = (sigma_cluster / sigma_obs) * d sigma_cluster / d ln theta_i.
    They are deliberately NOT equal when f_bin > 0; the binary-free Fisher (Task 1.3)
    caches the cluster-only jacrev (the binary-free model's sensitivity)."""
    th = oedb.theta_truth()
    R = oedb.R_BINS
    J_full = oedb.jacobian_lntheta(th, R, STELLAR.G)                 # (K, 5)
    J_bf = oedb.jacobian_lntheta_clusteronly(th[:4], R, STELLAR.G)   # (K, 4)
    assert J_bf.shape == (R.shape[0], 4)
    sig_cluster = oedb.cluster_sigma_los(th[:4], R, STELLAR.G)
    sig_obs = oedb.predict_sigma_obs(th, R, STELLAR.G)
    factor = (sig_cluster / sig_obs)[:, None]                        # (K, 1)
    assert jnp.allclose(J_full[:, :4], J_bf * factor, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------------
# Task 1.3: additive Fisher (RV-only) + binary-free c-optimal-for-M design
# ---------------------------------------------------------------------------
import jax  # noqa: E402  -- needed for PRNGKey in the optimizer tests


def test_binary_free_fisher_symmetric_and_spd():
    """The binary-free additive Fisher (single RV channel, ln-theta, +prior) is
    symmetric and strictly positive-definite for a uniform design."""
    F = oedb.fisher_binary_free(oedb.uniform_design(), oedb.N_TOTAL)
    assert F.shape == (4, 4)
    assert jnp.allclose(F, F.T)
    assert bool(jnp.all(jnp.linalg.eigvalsh(F) > 0.0))


def test_optimize_design_M_normalized_and_positive_sigma():
    """optimize_design_M returns per-bin counts summing to N_total (rtol 1e-4) and a
    positive c-optimal fractional precision sigma(M)/M."""
    res = oedb.optimize_design_M(oedb.N_TOTAL, key=jax.random.PRNGKey(0))
    assert jnp.isclose(jnp.sum(res.n_eff), oedb.N_TOTAL, rtol=1e-4)
    assert res.sigma_M_over_M > 0.0
    assert res.n_eff.shape == (oedb.R_BINS.shape[0],)
    assert bool(jnp.all(res.n_eff > 0.0))     # softmax keeps every bin populated


def test_c_optimal_beats_uniform_for_M():
    """The c-optimal-for-M design achieves sigma(M)/M no worse than the uniform
    design (optimization helps): optimizing the radial allocation tightens M."""
    res = oedb.optimize_design_M(oedb.N_TOTAL, key=jax.random.PRNGKey(0))
    F_unif = oedb.fisher_binary_free(oedb.uniform_design(), oedb.N_TOTAL)
    sigma_M_unif = float(jnp.sqrt(oedb.c_criterion_M(F_unif)))
    assert res.sigma_M_over_M <= sigma_M_unif


# ---------------------------------------------------------------------------
# Task 1.4: cross-model bias harness (Route-1 forward-model-consistent mock).
# Build-once: the per-bin truth sigma_los (one project_dispersion call) + the K_orb
# blend-velocity pool; the per-bin cluster velocities are drawn directly from
# Normal(0, sig_model^2) (NOT an EFF particle sampler), with per-star Bernoulli binary
# contamination. lax.map over draws, jit per-draw, LM GN MAP binary-free fit. FAST smoke
# (n_draws=4): the headline H1 statistic must be finite. NOT @slow.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.mark.parametrize("n_draws", [4])
def test_cross_model_bias_runs_and_is_finite(n_draws):
    """The cross-model bias harness runs end-to-end (generate WITH Moe binaries on a
    design, fit the binary-free model WITHOUT binaries) and returns a finite fractional
    bias + std at a small n_draws (smoke; the @slow gate in Task 1.5 runs the full MC)."""
    design = oedb.optimize_design_M(oedb.N_TOTAL, key=jax.random.PRNGKey(0)).n_eff
    out = oedb.cross_model_bias(design, n_draws=n_draws, key=jax.random.PRNGKey(1))
    assert jnp.isfinite(out.bias_M_frac)
    assert jnp.isfinite(out.std_M_frac)
    assert out.std_M_frac >= 0.0
    assert jnp.isfinite(out.mhat_mean) and out.mhat_mean > 0.0


# ---------------------------------------------------------------------------
# Refinement (2026-06-19): honest-analyst weighting + drop-empty-bins.
# The fit weight is the analyst's OWN realized scatter se[b] = sigma_hat[b]/sqrt(2 n_b)
# (NOT the truth sig_model), and only the bins the c-optimal design actually populates
# (n_b = round(design_n_eff[b]) >= N_MIN_FIT) enter the mock+fit; the rest are MASKED OUT
# of the GN residual entirely (no 2-star filler). This is the bias a real analyst incurs.
# ---------------------------------------------------------------------------
def test_per_bin_counts_drops_empty_bins_no_floor():
    """_per_bin_star_counts returns counts = round(design_n_eff) (NO 2-star floor) and a
    keep mask = counts >= N_MIN_FIT. For the c-optimal-for-M design only the few bins the
    design populates survive (the cold near-empty bins are DROPPED, not floored to 2)."""
    design = oedb.optimize_design_M(oedb.N_TOTAL, key=jax.random.PRNGKey(0)).n_eff
    counts, n_max, keep = oedb._per_bin_star_counts(design)
    counts = jnp.asarray(counts)
    keep = jnp.asarray(keep)
    K = oedb.R_BINS.shape[0]
    assert counts.shape == (K,) and keep.shape == (K,)
    # counts are the rounded design counts, NOT floored to 2.
    assert jnp.allclose(counts.astype(float), jnp.round(jnp.asarray(design)))
    # keep == (counts >= N_MIN_FIT); every kept bin has at least N_MIN_FIT stars.
    assert bool(jnp.all(keep == (counts >= oedb.N_MIN_FIT)))
    assert bool(jnp.all(counts[keep] >= oedb.N_MIN_FIT))
    # The c-optimal design populates only a few bins -> most are dropped.
    assert int(jnp.sum(keep)) < K
    assert int(jnp.sum(keep)) >= 1            # at least one bin survives
    # n_max is the width over KEPT bins (the array the masked sigma_hat reduction uses).
    assert n_max == int(jnp.max(counts[keep]))


def test_draw_binned_sigma_hat_uses_realized_scatter_se():
    """The Route-1 mock's per-bin SE is the HONEST-ANALYST realized scatter
    se[b] = sigma_hat[b] / sqrt(2 n_b) (n_b the actual kept count), NOT the truth
    sig_model. An analyst weights by what they OBSERVE; the contaminated cold outskirts
    then have large sigma_hat -> large se -> appropriately DOWN-weighted."""
    design = oedb.optimize_design_M(oedb.N_TOTAL, key=jax.random.PRNGKey(0)).n_eff
    counts, n_max, keep = oedb._per_bin_star_counts(design)
    sig_model = oedb.cluster_sigma_los(oedb.theta_truth_clusteronly(), oedb.R_BINS, STELLAR.G)
    korb = jnp.zeros(8)  # zero-Delta pool: with f_bin contamination off, isolate the SE form
    sigma_hat, se = oedb._draw_binned_sigma_hat(
        jax.random.PRNGKey(7), sig_model, counts, n_max, keep, korb, 0.0
    )
    keepb = jnp.asarray(keep)
    n_b = jnp.asarray(counts).astype(jnp.float64)
    # Realized SE == sigma_hat / sqrt(2 n_b) on the KEPT bins (the honest-analyst weight).
    expected_se = sigma_hat / jnp.sqrt(2.0 * jnp.maximum(n_b, 1.0))
    assert jnp.allclose(se[keepb], expected_se[keepb], rtol=1e-10)


def test_cross_model_bias_only_fits_kept_bins():
    """The harness fits ONLY the bins the design populates: the nuisance/M bias must be
    independent of how many EMPTY trailing bins the (masked) machinery carries. Smoke that
    the masked GN fit converges (dropped bins contribute nothing) at the c-optimal design."""
    design = oedb.optimize_design_M(oedb.N_TOTAL, key=jax.random.PRNGKey(0)).n_eff
    out = oedb.cross_model_bias(design, n_draws=4, key=jax.random.PRNGKey(2))
    assert jnp.isfinite(out.bias_M_frac)
    assert jnp.isfinite(jnp.asarray(out.bias_other)).all()


# ---------------------------------------------------------------------------
# Task 1.5: H1 gate -- the @slow, env-gated calibration MC (OUT of CI).
#
# Pre-registration LOCKED 2026-06-19 (design doc): ACCEPT H1 iff the naive
# (binary-free) c-optimal-for-M design, fit with the binary-free model on
# binary-contaminated mocks, biases M_hat HIGH by MORE than 2x its own forecast
# sigma(M)/M (false confidence). The threshold is NOT to be weakened: a reject ->
# H0 -> DESCOPE is a valid finding (null-result integrity).
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("PROGENAX_RUN_OED_BINARY"),
    reason="env-gated cross-model MC (set PROGENAX_RUN_OED_BINARY=1)",
)
def test_H0_no_binary_baseline_is_unbiased():
    """Route-1 PROOF: with f_bin_truth = 0 the cross-model mock is generated from the
    SAME forward model the fit uses (cluster_sigma_los + eps in quadrature), so there
    is NO sampler<->fit misspecification (review issue I1). M must then be recovered
    UNBIASED within the design's own forecast sigma -- the no-binary baseline that
    isolates the pure binary effect in H1.

    If this FAILS the mock and the fit are still inconsistent (likely the eps term or
    a sigma-vs-sigma^2 mismatch) -- FIX the harness, do NOT weaken the test."""
    res = oedb.run_H1(n_draws=oedb.N_DRAWS_H1, key=jax.random.PRNGKey(0), f_bin_truth=0.0)
    # sampler == fit-model: with zero binaries, M is recovered unbiased within forecast.
    assert abs(res.bias_M_frac) < 2.0 * res.forecast_sigma_M_frac


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("PROGENAX_RUN_OED_BINARY"),
    reason="env-gated cross-model MC (set PROGENAX_RUN_OED_BINARY=1)",
)
def test_H1_naive_design_biased_beyond_forecast():
    """The naive (binary-free) c-optimal-for-M design + binary-free fit on
    binary-contaminated RV data biases M_hat beyond its OWN forecast sigma(M):
    pre-registered ACCEPT H1 iff bias_M_frac > 2 * forecast_sigma_M_frac.

    Pre-registered (LOCKED 2026-06-19); do NOT weaken. A reject is a reportable
    finding, not a test to relax."""
    res = oedb.run_H1(n_draws=oedb.N_DRAWS_H1, key=jax.random.PRNGKey(0))
    assert res.bias_M_frac > 2.0 * res.forecast_sigma_M_frac   # pre-registered; do NOT weaken
    assert res.accept


# ---------------------------------------------------------------------------
# Task 1.6: gated CLI smoke test (scripts/demo_oed_binary.py).
#
# The CLI's --quick path computes ONLY the cheap parts -- the binary-free
# c-optimal-for-M design (dialed-down multi-start Adam) + the forecast + the no-MC
# mechanism figure. It does NOT run the env-gated cross-model calibration MC (that is
# the @slow gate above). So this smoke test is FAST and needs no env var: it asserts
# rc == 0 and that the mechanism PNG (the figure that needs no MC) lands in --outdir.
# It also guards the Stage-3 CLI lesson: the run-record path derives from --outdir, so a
# smoke run into tmp_path NEVER clobbers the committed full-quality run-record/figures.
# ---------------------------------------------------------------------------
def test_cli_binary_quick_smoke(tmp_path):
    """The CLI --quick path runs end-to-end (design + forecast + the no-MC mechanism
    figure) and exits 0 WITHOUT the cross-model MC or its env var.

    Dialed down via --quick (few Adam starts/steps). Asserts rc == 0, that the mechanism
    PNG lands in --outdir, that the false-confidence (MC-only) figure is NOT produced in
    --quick mode, and that the run-record was written under --outdir (NOT the committed
    FIGURE_DIR) -- the Stage-3 fixed-path guard.
    """
    import importlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    cli = importlib.import_module("demo_oed_binary")
    rc = cli.main(["--quick", "--outdir", str(tmp_path)])
    assert rc == 0
    # The mechanism figure (no MC) MUST be produced; the false-confidence figure needs the
    # MC and so is absent in --quick mode.
    assert (tmp_path / "demo_oedb_mechanism.png").exists()
    assert not (tmp_path / "demo_oedb_false_confidence.png").exists()
    # The run-record path derives from --outdir (Stage-3 lesson): it lands in tmp_path,
    # never the committed FIGURE_DIR.
    assert (tmp_path / "demo_oed_binary_run_record.json").exists()


# ===========================================================================
# Phase 2 -- Marginalize fix (deterministic group: T2.1 + T2.2 + H2 + H3)
# ===========================================================================
#
# Task 2.1: marginalized (5-param) Fisher + f_bin prior + AD-vs-FD f_bin gate.
# The marginalized model carries f_bin as a FREE nuisance: theta = (M, r_a, gamma,
# a, f_bin). The Fisher denominator uses the binary-INFLATED observed dispersion
# (sigma_obs^2 + eps^2), and the f_bin column of J carries the radial leverage that
# breaks M<->f_bin. Build-once at the full truth (no re-jacrev in the optimizer loop).
# ---------------------------------------------------------------------------
def test_marginalized_fisher_symmetric_and_spd():
    """The marginalized 5-param additive Fisher (single RV channel, ln-theta, +prior)
    is symmetric and strictly positive-definite for a uniform design."""
    F = oedb.fisher_marginalized(oedb.uniform_design(), oedb.N_TOTAL)
    assert F.shape == (5, 5)
    assert jnp.allclose(F, F.T)
    assert bool(jnp.all(jnp.linalg.eigvalsh(F) > 0.0))


def test_prior_diag_marg_shape_and_fbin_weak():
    """PRIOR_DIAG_MARG is len-5 ln-theta fractional precisions: M=0 (target, no prior),
    r_a weak, (gamma, a) tight photometric, f_bin WEAK (a data-driven nuisance
    constrained by radial leverage, not by a prior). The f_bin precision must be SMALL
    relative to the photometric (gamma, a) precision -- f_bin is NOT photometrically
    pinned; the design's core<->outskirts contrast is what constrains it."""
    pd = oedb.PRIOR_DIAG_MARG
    assert pd.shape == (5,)
    assert float(pd[oedb.IDX_M]) == 0.0                       # target: no prior
    # f_bin is a weak (data-driven) nuisance: its prior precision is far below the tight
    # photometric (gamma, a) precision (1/0.1^2 = 100) -- the radial leverage constrains it.
    assert float(pd[oedb.IDX_FBIN]) < float(pd[oedb.IDX_GAMMA])
    assert float(pd[oedb.IDX_FBIN]) < float(pd[oedb.IDX_A])
    # ... and it matches the binary-free r_a weak prior structure (a kinematic nuisance).
    assert float(pd[oedb.IDX_RA]) > 0.0 and float(pd[oedb.IDX_FBIN]) > 0.0


def test_jacobian_lntheta_fbin_column_ad_vs_fd():
    """AD-vs-FD gate on the f_bin column of jacobian_lntheta (gradient-validation).

    The analytic ln-theta sensitivity d sigma_obs / d ln f_bin = f_bin * V_BIN /
    (2 sigma_obs) (verified 4e-16 against its closed form in Task 1.2) is here
    cross-checked by an EXPLICIT central finite difference of the FULL observable
    predict_sigma_obs wrt ln f_bin at the truth. Reverse-mode AD vs central FD must
    agree to rel < 1e-3 (the design-doc validation-gate threshold; central FD is
    O(h^2)-truncation-limited, so 1e-3 is the achievable accuracy at a sane step)."""
    th = oedb.theta_truth()
    R = oedb.R_BINS
    J = oedb.jacobian_lntheta(th, R, STELLAR.G)               # (K, 5)
    ad_fcol = J[:, oedb.IDX_FBIN]                             # d sigma_obs / d ln f_bin (AD)

    # Central FD of sigma_obs(exp(lnth)) wrt the f_bin log-coordinate, at ln(truth).
    lnth = jnp.log(th)
    h = 1e-4                                                  # log-step
    e_fbin = jnp.zeros(5).at[oedb.IDX_FBIN].set(1.0)
    sig_plus = oedb.predict_sigma_obs(jnp.exp(lnth + h * e_fbin), R, STELLAR.G)
    sig_minus = oedb.predict_sigma_obs(jnp.exp(lnth - h * e_fbin), R, STELLAR.G)
    fd_fcol = (sig_plus - sig_minus) / (2.0 * h)             # d sigma_obs / d ln f_bin (FD)

    rel = jnp.max(jnp.abs(ad_fcol - fd_fcol) / jnp.abs(fd_fcol))
    assert float(rel) < 1e-3, f"AD-vs-FD f_bin column rel-err {float(rel):.2e} >= 1e-3"


def test_marginalized_fisher_uses_inflated_denominator():
    """The marginalized Fisher denominator uses the binary-INFLATED observed dispersion
    (_SIG_MARG = predict_sigma_obs at the full truth), NOT the bare cluster sigma. The
    cached _SIG_MARG must equal the binary-inflated observable and strictly exceed the
    binary-free _SIG_BF (the f_bin pedestal inflates the observed second moment)."""
    sig_obs = oedb.predict_sigma_obs(oedb.theta_truth(), oedb.R_BINS, STELLAR.G)
    assert jnp.allclose(oedb._SIG_MARG, sig_obs, rtol=1e-12)
    assert bool(jnp.all(oedb._SIG_MARG >= oedb._SIG_BF))      # inflated >= bare cluster
    # _J_MARG is the full 5-column jacrev at the truth (build-once, no re-jacrev in loop).
    assert oedb._J_MARG.shape == (oedb.R_BINS.shape[0], 5)
    assert jnp.allclose(
        oedb._J_MARG, oedb.jacobian_lntheta(oedb.theta_truth(), oedb.R_BINS, STELLAR.G)
    )


# ---------------------------------------------------------------------------
# Task 2.2: binary-aware c-optimal-for-M design + H2 (OED payoff) + H3 (allocation).
# H2/H3 are DETERMINISTIC pre-registered gates (Fisher/design only, no MC). Thresholds
# LOCKED 2026-06-19 (design doc): H2 precision_gain >= 1.3x; H3 non-monotone allocation.
# A reject is a reportable finding -- do NOT weaken (null-result integrity).
# ---------------------------------------------------------------------------
def test_optimize_design_M_marg_normalized_and_positive_sigma():
    """optimize_design_M_marg returns per-bin counts summing to N_total (rtol 1e-4) and a
    positive marginalized c-optimal fractional precision sigma(M)/M over the 5-param
    fisher_marginalized."""
    res = oedb.optimize_design_M_marg(oedb.N_TOTAL, key=jax.random.PRNGKey(0))
    assert jnp.isclose(jnp.sum(res.n_eff), oedb.N_TOTAL, rtol=1e-4)
    assert res.sigma_M_over_M > 0.0
    assert res.n_eff.shape == (oedb.R_BINS.shape[0],)
    assert bool(jnp.all(res.n_eff > 0.0))     # softmax keeps every bin populated


def test_marg_design_costs_info_vs_binary_free():
    """Physical sanity: marginalizing over the unknown f_bin COSTS information, so the
    binary-aware design's marginalized sigma(M)/M is LARGER than the binary-free design's
    (over-confident) binary-free sigma(M)/M = 0.0447. You pay for not knowing f_bin."""
    res_marg = oedb.optimize_design_M_marg(oedb.N_TOTAL, key=jax.random.PRNGKey(0))
    res_bf = oedb.optimize_design_M(oedb.N_TOTAL, key=jax.random.PRNGKey(0))
    assert res_marg.sigma_M_over_M > res_bf.sigma_M_over_M


def test_marg_c_optimal_beats_uniform():
    """The marginalized c-optimal-for-M design achieves sigma(M)/M no worse than the
    uniform design under the SAME marginalized Fisher (optimization helps)."""
    res = oedb.optimize_design_M_marg(oedb.N_TOTAL, key=jax.random.PRNGKey(0))
    F_unif = oedb.fisher_marginalized(oedb.uniform_design(), oedb.N_TOTAL)
    sigma_M_unif = float(jnp.sqrt(oed.c_criterion(F_unif, target=oedb.IDX_M)))
    assert res.sigma_M_over_M <= sigma_M_unif


def test_H2_precision_gain_meets_threshold():
    """H2 (OED payoff, pre-registered >= 1.3x): under the binary-AWARE (marginalized)
    Fisher, the binary-aware design tightens sigma(M) by precision_gain =
    sigmaM_under_marg(binary_free_z) / sigmaM_under_marg(binary_aware_z) >= 1.3. The
    binary-free design is over-confident: scored under the marginalized Fisher (which it
    did not optimize for) its sigma(M) is ~2x its own naive forecast, and the
    binary-aware design recovers a 1.3x precision gain. LOCKED -- do NOT weaken; a
    reject is a reportable null finding (binary-free was accidentally near-optimal)."""
    gain = oedb.h2_precision_gain(oedb.N_TOTAL, key=jax.random.PRNGKey(0))
    assert gain >= 1.3, f"H2 precision_gain {gain:.3f} < 1.3 (pre-registered) -- report as null"


def test_H3_allocation_non_monotone():
    """H3 (non-obvious allocation, pre-registered): the binary-aware per-bin allocation is
    NOT a monotone rescaling of the binary-free one. A monotone rescaling preserves the
    per-bin weight RANK ORDER and gives a near-1 cosine similarity of the (normalized)
    weight vectors; the binary-aware design instead REORDERS the bins (it pulls budget
    toward the f_bin-constraining radii) -- so the ranks differ AND the cosine similarity
    is far below 1. LOCKED -- a monotone result would be the null finding, reported not
    weakened."""
    h3 = oedb.h3_allocation_comparison(oedb.N_TOTAL, key=jax.random.PRNGKey(0))
    # Non-monotone signature 1: the per-bin weight rank order changes.
    assert h3.ranks_differ, "binary-aware allocation preserved the binary-free rank order"
    # Non-monotone signature 2: the weight-vector cosine similarity is far below 1 (a
    # monotone rescaling would give cosine ~ 1). Documented threshold: < 0.9.
    assert h3.cosine_similarity < 0.9, (
        f"weight-vector cosine {h3.cosine_similarity:.3f} >= 0.9 -- too close to a "
        "monotone rescaling"
    )


# ===========================================================================
# Phase 2 -- Marginalize fix (MC group: Task 2.5 -- the fix REMOVES the bias).
#
# The @slow, env-gated counterpart to the H1 gate: the binary-AWARE FIT (5-param,
# f_bin FREE) on the SAME binary-contaminated mocks recovers M_hat UNBIASED -- the
# "fix works" headline. Phase 1 showed the binary-FREE fit gives M_hat ~ 2.85x truth
# (bias +185%); here the binary-aware fit on the binary-aware design removes it.
#
# Pre-registration (design doc Phase 2): ACCEPT iff |bias_M_frac| < 2 * sigma_M_marg
# (the binary-aware forecast). The threshold is NOT to be weakened: if M_hat is NOT
# recovered unbiased, INVESTIGATE the harness (f_bin identifiability on the few
# populated bins, or an eps/sigma-vs-sigma^2 inconsistency); a genuine inability to
# debias would be a reportable finding.
# ===========================================================================
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("PROGENAX_RUN_OED_BINARY"),
    reason="env-gated cross-model MC (set PROGENAX_RUN_OED_BINARY=1)",
)
def test_fix_binary_aware_fit_is_unbiased():
    """The fix: the binary-AWARE fit (5-param, f_bin free) on the binary-aware design
    removes the +185% M bias of the binary-FREE fit. Pre-registered ACCEPT iff
    |bias_M_frac| < 2 * sigma_M_marg (the binary-aware forecast). LOCKED -- do NOT
    weaken; if it cannot be made unbiased, fix the harness (or report as a finding)."""
    res = oedb.run_fix(n_draws=oedb.N_DRAWS_H1, key=jax.random.PRNGKey(0))
    assert abs(res.bias_M_frac) < 2.0 * res.sigma_M_marg   # the binary-aware fit removes the bias
    # Regression guard #2 (review): the binary-aware fit RECOVERS the binary fraction (guards
    # the MECHANISM, not just the M symptom). f_bin_hat must agree with the truth within 3 sigma
    # of its own draw-to-draw scatter -- the radial leverage identifies f_bin, not just the prior.
    assert abs(res.fbin_hat_mean - oedb.F_BIN_TRUTH) < 3.0 * res.fbin_hat_std


# ===========================================================================
# Phase 3 -- min-max / maximin design (deterministic; T3.1 + T3.2).
#
# The marginalize design optimizes at the ASSUMED truth f_bin = 0.5. The MAXIMIN
# design instead minimizes the WORST-CASE marginalized sigma(M) over f_bin in
# [0, F_MAX] -- robust to an UNMODELED / untrusted binary fraction ("don't trust
# Moe's f_bin"). Cheap: a build-once f_bin GRID of jacrevs (the f_bin column AND the
# sigma_obs denominator both depend on f_bin; EFF is ODE-free so the grid is cheap),
# re-used in the optimizer loop with NO re-jacrev. The optimizer uses a SMOOTH max
# (logsumexp) for stable gradients; the REPORTED worst-case is the true jnp.max.
# ===========================================================================
def test_fbin_grid_caches_shapes_and_endpoints():
    """The build-once f_bin grid of jacrevs / sigma_obs has the documented shape and
    spans [0, F_MAX]. At f_bin = 0 the f_bin column of the jacrev vanishes (the pedestal
    f_bin*V_bin = 0, so d sigma_obs / d ln f_bin = f_bin*V_bin/(2 sigma_obs) = 0) -- the
    f_bin parameter is then prior-only (correct)."""
    assert oedb.N_FBIN_GRID >= 3
    assert float(oedb.F_BIN_GRID[0]) == 0.0
    assert jnp.isclose(oedb.F_BIN_GRID[-1], oedb.F_MAX)
    assert oedb.F_BIN_GRID.shape == (oedb.N_FBIN_GRID,)
    K = oedb.R_BINS.shape[0]
    assert oedb._J_MARG_GRID.shape == (oedb.N_FBIN_GRID, K, 5)
    assert oedb._SIG_MARG_GRID.shape == (oedb.N_FBIN_GRID, K)
    # f_bin = 0 endpoint: the f_bin sensitivity column is ~0 (pedestal vanishes).
    assert jnp.allclose(oedb._J_MARG_GRID[0, :, oedb.IDX_FBIN], 0.0, atol=1e-12)


def test_worstcase_sigmaM_spd_at_each_grid_point():
    """Every grid-point Fisher (5-param marginalized, at that f_bin's cached jacrev +
    sigma_obs) is symmetric and SPD for a uniform design -- so each inverse (F^-1)[M, M]
    is well-defined and the worst-case max is over finite, positive sigma(M)."""
    z = oedb.uniform_design()
    for i in range(oedb.N_FBIN_GRID):
        F = oedb.fisher_marginalized(
            z, oedb.N_TOTAL, J=oedb._J_MARG_GRID[i], sig=oedb._SIG_MARG_GRID[i]
        )
        assert jnp.allclose(F, F.T)
        assert bool(jnp.all(jnp.linalg.eigvalsh(F) > 0.0))


def test_worstcase_sigmaM_bounds_each_grid_point():
    """worstcase_sigmaM(z) is finite and >= the single-point marginalized sigma(M) at
    ANY interior grid f_bin (the max bounds every point it maximizes over)."""
    z = oedb.uniform_design()
    wc = oedb.worstcase_sigmaM(z, oedb.N_TOTAL)
    assert jnp.isfinite(wc) and wc > 0.0
    for i in range(oedb.N_FBIN_GRID):
        sigmaM_i = float(jnp.sqrt(oed.c_criterion(
            oedb.fisher_marginalized(
                z, oedb.N_TOTAL, J=oedb._J_MARG_GRID[i], sig=oedb._SIG_MARG_GRID[i]
            ),
            target=oedb.IDX_M,
        )))
        assert wc >= sigmaM_i - 1e-9


def test_maximin_design_wins_at_worstcase():
    """The DEFINING property of the maximin design: its worst-case sigma(M) over the
    f_bin grid is <= the marginalize design's worst-case (the maximin design hedges to
    lower the worst case). LOCKED -- a violation means the optimizer failed to find the
    maximin optimum (investigate), do NOT weaken."""
    mm = oedb.optimize_design_maximin(oedb.N_TOTAL, key=jax.random.PRNGKey(0))
    assert jnp.isclose(jnp.sum(mm.n_eff), oedb.N_TOTAL, rtol=1e-4)
    assert mm.worstcase_sigma_M > 0.0
    marg = oedb.optimize_design_M_marg(oedb.N_TOTAL, key=jax.random.PRNGKey(0))
    wc_marg = float(oedb.worstcase_sigmaM(marg.z, oedb.N_TOTAL))
    # maximin's worst-case must not exceed the marginalize design's worst-case (small
    # numerical slack for the two independent Adam optima).
    assert mm.worstcase_sigma_M <= wc_marg + 1e-6


def test_marg_fit_is_data_driven_not_pinned_at_truth():
    """Regression guard #1 (review): the binary-AWARE fit is DATA-DRIVEN, not pinned at the
    truth init. Build ONE fixed binary-contaminated mock (deterministic) and fit the 5-param
    _fit_theta_marg_gn from several PERTURBED starts (M x2, M x0.4; f_bin 0.2, 0.9). All
    starts must converge to the SAME data minimum -- recovered M_hat and f_bin_hat AGREE
    across inits to a tolerance -- proving the fit chases the data, not the starting value.
    Cheap: one mock, two kept bins, a modest iteration budget.
    """
    G = STELLAR.G
    # A tiny TWO-bin design (a core bin + a cold outskirts bin) so f_bin has radial leverage
    # (the core<->outskirts contrast breaks M<->f_bin) while staying cheap. Counts well above
    # N_MIN_FIT so both bins are kept.
    K = oedb.R_BINS.shape[0]
    design = jnp.zeros(K).at[2].set(2000.0).at[K - 1].set(1500.0)   # core-ish + outermost bin
    counts, n_max, keep = oedb._per_bin_star_counts(design)
    assert int(jnp.sum(keep)) == 2                                   # exactly the two populated bins

    sig_model = oedb.cluster_sigma_los(oedb.theta_truth_clusteronly(), oedb.R_BINS, G)
    # Build-once K_orb pool exactly as the MC does (Var == V_BIN), so the mock is a faithful
    # binary-contaminated draw at the truth f_bin.
    korb_raw = jnp.asarray(oedb.binaries.sample_blend_velocities(
        jax.random.PRNGKey(oedb.V_BIN_SEED + 1), oedb._KORB_POOL_N,
        imf=oedb.massive_primary_imf(), Z=oedb.V_BIN_Z,
    ))
    korb_centered = korb_raw - jnp.mean(korb_raw)
    korb_pool = korb_centered * jnp.sqrt(oedb.V_BIN / jnp.var(korb_centered, ddof=1))

    sigma_hat, se = oedb._draw_binned_sigma_hat(
        jax.random.PRNGKey(123), sig_model, counts, n_max, keep, korb_pool, oedb.F_BIN_TRUTH
    )

    th = oedb.theta_truth()
    # Perturbed starts: move M far (x2, x0.4) and f_bin far (0.2, 0.9) from the truth. The
    # nuisance (r_a, gamma, a) starts are kept at truth (they are well-pinned photometrically).
    inits = [
        th.at[oedb.IDX_M].mul(2.0).at[oedb.IDX_FBIN].set(0.2),
        th.at[oedb.IDX_M].mul(0.4).at[oedb.IDX_FBIN].set(0.9),
        th.at[oedb.IDX_M].mul(2.0).at[oedb.IDX_FBIN].set(0.9),
    ]
    fits = [
        oedb._fit_theta_marg_gn(sigma_hat, se, keep, G, n_iter=120, theta_init=ti)[0]
        for ti in inits
    ]
    m_hats = jnp.array([oedb.th_M(f) for f in fits])
    f_hats = jnp.array([oedb.th_fbin(f) for f in fits])
    # Agreement across inits -> the fit converged to the SAME data minimum (data-driven).
    assert float(jnp.max(m_hats) / jnp.min(m_hats) - 1.0) < 0.02, f"M_hat spread {m_hats}"
    assert float(jnp.max(f_hats) - jnp.min(f_hats)) < 0.02, f"f_bin_hat spread {f_hats}"


def test_compare_maximin_vs_marginalize_is_a_hedge():
    """Task 3.2 comparison: the maximin design HEDGES -- it sacrifices a little sigma(M) at
    the truth f_bin = 0.5 to lower the worst-case sigma(M) over the grid. The marginalize
    design (optimized at the truth) is best AT f_bin = 0.5; the maximin design is best at
    the worst case. Both summary fractions are non-negative (sacrifice at truth >= 0, gain
    at worst >= 0) -- the textbook robust-design trade. Shapes/finiteness sanity too."""
    cmp = oedb.compare_maximin_vs_marginalize(oedb.N_TOTAL, key=jax.random.PRNGKey(0))
    G = oedb.N_FBIN_GRID
    K = oedb.R_BINS.shape[0]
    assert cmp.grid_sigmaM_marg.shape == (G,) and cmp.grid_sigmaM_mm.shape == (G,)
    assert cmp.n_eff_marg.shape == (K,) and cmp.n_eff_mm.shape == (K,)
    # marginalize wins AT the truth f_bin: its sigma(M) there is <= maximin's.
    assert cmp.sigmaM_truth_marg <= cmp.sigmaM_truth_mm + 1e-9
    # maximin wins at the WORST case: its worst-case sigma(M) is <= marginalize's.
    assert cmp.sigmaM_worst_mm <= cmp.sigmaM_worst_marg + 1e-9
    # the hedge: a (non-negative) sacrifice at truth buys a (non-negative) worst-case gain.
    assert cmp.sacrifice_at_truth_frac >= -1e-9
    assert cmp.gain_at_worst_frac >= -1e-9


# ===========================================================================
# Phase 4 -- Task T4.1: the sigma_bin/sigma_cluster sweep across system mass.
#
# The single-point H1 headline (M_hat = 2.85x at sigma_bin/sigma_cluster = 1.08) and
# H2's thin 1.33x margin are CONTEXTUALIZED as CURVES across system mass. The Moe
# massive-primary population (V_BIN, sigma_bin = 9.73 km/s) is FIXED; the cluster mass M
# varies -> sigma_cluster varies -> sigma_bin/sigma_cluster sweeps (smaller M = colder
# cluster = worse contamination). ADDITIVE-ONLY: per-mass J/sig are computed and PASSED
# to the existing override-capable Fisher/optimizer; the module-level M_FID caches and
# every default are byte-unaffected.
#
# T4.1a (deterministic sweep -- fast): no MC.
# ---------------------------------------------------------------------------
def test_deterministic_sweep_runs_monotone_finite():
    """The deterministic sweep (T4.1a) runs over a fine mass grid and returns, per mass:
    sigma_bin/sigma_cluster, the binary-FREE forecast sigma(M)/M, the marginalized
    (binary-AWARE) forecast sigma(M)/M, and the H2 precision-gain. All finite and positive;
    sigma_bin/sigma_cluster is MONOTONE DECREASING in M (colder low-mass clusters are worse
    contaminated). Cheap (no MC); a dialed-down optimizer keeps it fast."""
    sw = oedb.deterministic_sweep(
        n_mass=6, key=jax.random.PRNGKey(0), n_starts=2, n_steps=120
    )
    n = sw.M_grid.shape[0]
    assert n == 6
    # sigma_bin is the FIXED massive-primary scale (mass-independent).
    assert jnp.isclose(sw.sigma_bin_kms, jnp.sqrt(oedb.V_BIN))
    # all arrays finite + positive.
    for arr in (sw.M_grid, sw.sigma_cluster_kms, sw.ratio, sw.sigmaM_bf,
                sw.sigmaM_marg, sw.h2_gain):
        assert arr.shape == (n,)
        assert bool(jnp.all(jnp.isfinite(arr)))
        assert bool(jnp.all(arr > 0.0))
    # M increasing -> sigma_cluster increasing -> ratio strictly DECREASING.
    assert bool(jnp.all(jnp.diff(sw.M_grid) > 0.0))
    assert bool(jnp.all(jnp.diff(sw.ratio) < 0.0))
    # the ratio range brackets the fiducial 1.08 operating point.
    assert float(sw.ratio[0]) > 1.08 > float(sw.ratio[-1])


def test_deterministic_sweep_span_and_marg_costs_info():
    """The mass grid spans sigma_cluster ~3 -> ~15 km/s (sigma_bin/sigma_cluster ~3 ->
    ~0.6, bracketing the fiducial 1.08), and AT EVERY mass the marginalized (binary-aware)
    forecast sigma(M)/M is LARGER than the binary-free one -- marginalizing the unknown
    f_bin always costs information (the honest binary-aware precision)."""
    sw = oedb.deterministic_sweep(
        n_mass=8, key=jax.random.PRNGKey(0), n_starts=2, n_steps=120
    )
    # sigma_cluster span: ~3 km/s (coldest) -> ~15 km/s (hottest).
    assert float(sw.sigma_cluster_kms[0]) < 4.0
    assert float(sw.sigma_cluster_kms[-1]) > 13.0
    # marginalizing f_bin costs info at every mass (binary-aware sigma(M) > binary-free).
    assert bool(jnp.all(sw.sigmaM_marg > sw.sigmaM_bf))
    # H2 gain is the binary-free design's marginalized sigma(M) over the binary-aware
    # design's: >= 1 (the binary-aware design is at least as good under its own Fisher).
    assert bool(jnp.all(sw.h2_gain >= 1.0 - 1e-6))


# ---------------------------------------------------------------------------
# T4.1b (MC bias sweep -- @slow, env-gated). A tiny smoke (n_draws=4, 2 anchor
# masses) confirms the sweep returns finite bias arrays for BOTH the binary-free
# design+fit and the binary-aware design+fit. Marked @slow + env-gated because it
# runs the cross-model MC (the project_dispersion GN-fit tape per draw).
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("PROGENAX_RUN_OED_BINARY"),
    reason="env-gated cross-model MC (set PROGENAX_RUN_OED_BINARY=1)",
)
def test_mc_bias_sweep_runs_and_is_finite():
    """The MC bias sweep (T4.1b) runs the cross-model MC at a few anchor masses and
    returns finite binary-FREE-fit bias and binary-AWARE-fit residual-bias arrays. Tiny
    (n_draws=4, 2 masses) smoke -- the full sweep is the reported result, not gated."""
    sw = oedb.mc_bias_sweep(
        n_mass=2, n_draws=4, key=jax.random.PRNGKey(0), n_starts=2, n_steps=120
    )
    n = sw.M_grid.shape[0]
    assert n == 2
    for arr in (sw.M_grid, sw.ratio, sw.bias_bf, sw.bias_marg):
        assert arr.shape == (n,)
        assert bool(jnp.all(jnp.isfinite(arr)))
    assert bool(jnp.all(jnp.diff(sw.ratio) < 0.0))
