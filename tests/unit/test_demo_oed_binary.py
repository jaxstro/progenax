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
# Task 1.4: cross-model bias harness (build-once EFF sampler + K_orb pool,
# lax.map over draws, jit per-draw, LM GN MAP binary-free fit). FAST smoke
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
