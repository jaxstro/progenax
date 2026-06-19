"""Unit tests for the W0-OED concentration demo core (scripts/_demo_oed_concentration.py).

Task 1 (this file, for now): the load-bearing sampler-equals-Fisher-model check.
The W0-OED calibration gate rests on a particle sampler whose binned dispersion
equals what `progenax.project_dispersion(profile, r_a)` predicts (the Fisher
forward model). `project_dispersion` is Osipkov-Merritt-only, so the sampler must
draw the SAME model it projects: "that profile's density under OM", for BOTH King
(Engine B `from_density_profiles`) and Michie (hand-rolled `eddington_invert` on
the Michie density -- Engine B does not ingest MichieProfile, and we must NOT use
`MichieVelocityDF`'s NATIVE anisotropy, which would mismatch the OM projection).
"""
import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import pytest

import progenax  # noqa: F401  enables float64 at import
from jaxstro.units import STELLAR

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import _demo_oed_concentration as oedc  # noqa: E402


@pytest.mark.parametrize("model", ["king", "michie"])
def test_om_sampler_matches_project_dispersion(model):
    """Binned dispersion of the OM sampler must match project_dispersion (the Fisher
    forward model) -- otherwise the calibration gate would test mismatched models.

    Probes ALL THREE channels (LOS, PM_R, PM_T) at a MID (R~2 r_c) and an OUTER
    (R~6 r_c) radius: the outer/PM regimes are exactly what the OED design will
    exploit (outskirts PM allocation), so the regression net must cover them, not
    just the inner LOS bin (review M3)."""
    key = jax.random.PRNGKey(0)
    W0, r_a, M = 6.0, 6.0, 1e5
    R, v_los, v_pm_r, v_pm_t = oedc.sample_om_cluster(
        model, W0, r_a, M, n_stars=200_000, key=key
    )
    channels = {"los": v_los, "pm_r": v_pm_r, "pm_t": v_pm_t}
    # THIN absolute annulus (+-0.5 r_c), NOT a fractional one: at the outer radius
    # sigma(R) falls steeply, so a wide annulus's surface-density weighting biases the
    # binned std toward the inner (higher-sigma) edge (~5% at +-20% R=6). A thin
    # annulus keeps sigma ~ const across it; ~8000 stars remain at R=6 (ample).
    for R_center in (2.0, 6.0):
        prof = oedc.build_profile(W0, r_a, model)
        pd = progenax.project_dispersion(prof, r_a, jnp.array([R_center]), M, STELLAR.G)
        pred = {"los": pd.sigma_los[0], "pm_r": pd.sigma_pm_r[0], "pm_t": pd.sigma_pm_t[0]}
        sel = (R > R_center - 0.5) & (R < R_center + 0.5)
        assert sel.sum() > 2000, (model, R_center, int(sel.sum()))
        for name, v in channels.items():
            meas = jnp.std(v[sel], ddof=1)
            assert jnp.abs(meas - pred[name]) / pred[name] < 0.05, (
                model, R_center, name, float(meas), float(pred[name])
            )


def test_michie_om_df_is_positive():
    """The hand-rolled OM-Michie Eddington DF must be realizable (f >= 0) at the
    demo's truth (W0=6, r_a=6) -- the King Engine B path self-gates this, but the
    Michie path's positivity needs an explicit assertion (plan Step 5; review M2).
    Mirrors the EFF DF realizability guard (eff_df.py:135)."""
    f_min, f_max = oedc.michie_om_table_diagnostics(W0=6.0, r_a=6.0)
    assert f_min > -1e-3 * f_max, (f_min, f_max)


# ---------------------------------------------------------------------------
# Task 2: forward observable predict_sigma + ln-theta Jacobian jacobian_and_sigma.
# theta = (W0, r_a, M); index map W0=0 (TARGET), r_a=1, M=2.
# ---------------------------------------------------------------------------


def test_predict_sigma_shape_and_bins_bound():
    """predict_sigma returns finite, positive (3, K) channel dispersions and every
    R_BINS bin is dynamically BOUND (r_t > R_BINS[-1]) for BOTH models -- otherwise
    the outer bins probe unbound radii where project_dispersion is undefined."""
    for model in ("king", "michie"):
        th = oedc.theta_truth()                       # (3,) = (W0, r_a, M)
        sig = oedc.predict_sigma(th, oedc.R_BINS, STELLAR.G, model)
        assert sig.shape == (3, oedc.R_BINS.shape[0])
        assert jnp.all(jnp.isfinite(sig)) and jnp.all(sig > 0)
        prof = oedc.build_profile(th[0], th[1], model)
        assert float(prof.r_t) > float(oedc.R_BINS[-1]), (model, float(prof.r_t))


def test_jacobian_lntheta_shape_and_W0_column_nonzero():
    """jacobian_and_sigma returns the dimensionless ln-theta Jacobian (3, K, 3) and
    the W0 column (index 0) carries nonzero signal -- the design's target gradient."""
    for model in ("king", "michie"):
        th = oedc.theta_truth()
        J, sig = oedc.jacobian_and_sigma(th, oedc.R_BINS, STELLAR.G, model)
        K = oedc.R_BINS.shape[0]
        assert J.shape == (3, K, 3)                    # channel, bin, param
        assert jnp.all(jnp.isfinite(J))
        assert jnp.any(jnp.abs(J[:, :, 0]) > 0)        # W0 column carries signal


# ---------------------------------------------------------------------------
# Task 3: AD-vs-FD gate on d sigma / d ln W0 (the W0 column of the Jacobian).
#
# This gates the WHOLE POINT of the arc at the gradient level: that the King /
# Michie -> project_dispersion forward model is W0-DIFFERENTIABLE and the AD
# gradient ON THE W0 PARAMETER is the h->0 truth. We differentiate the per-bin,
# per-channel observable w.r.t. theta[0] (= W0) and scale by W0 to get the
# DIMENSIONLESS d sigma / d ln W0 (the same ln-theta metric the Fisher uses,
# jacobian_and_sigma / ADR-0011).
#
# METHODOLOGY (ADR-0016, mirroring the repo's ratified Michie Richardson idiom
# tests/unit/kinematics/test_dispersion.py::test_grad_jeans_michie_high_W0_ad_correct):
# a fixed-step central FD is only a faithful truth-proxy where d sigma/d W0 has
# mild curvature. Near a high-curvature bin (project_dispersion's B&M82 projection
# weights a thin radial shell where sigma_r(W0) bends sharply -- the same r_t(W0)
# proximity effect as ADR-0016) the fixed-step FD's O(h^2 sigma''') truncation
# error dominates, so a fixed `rel < 1e-3` floor would mis-flag a CORRECT AD
# gradient. The faithful proxy there is the Richardson TREND: the central-FD
# estimate must CONVERGE toward AD as h shrinks (FD -> AD), proving AD is the h->0
# limit. Both the fixed-step floor (on the FD-reliable bins) and the Richardson
# trend (everywhere) are measured empirically (Task 3 characterization), never
# tuned to pass.

# Step sizes for the central-FD Richardson sweep (coarse -> fine).
_FD_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
# Fixed-step floor used on the FD-reliable bins (design / Stage-1 precedent, never
# weakened): at this step the central FD is a faithful truth-proxy for those bins.
_FD_FLOOR_H = 1e-4
_FD_FLOOR = 1e-3


def _dsigma_dlnW0(model, channel, R_bins):
    """Return f(W0) = d-able per-bin observable for one channel, and the AD value.

    f maps a scalar W0 -> (K,) channel dispersion at the truth (r_a, M) fixed; the
    AD d sigma / d ln W0 is jacrev(f)(W0) * W0 (chain rule for the ln-theta metric)."""
    th = oedc.theta_truth()
    W0 = th[0]

    def f(w0):
        return oedc.predict_sigma(th.at[0].set(w0), R_bins, STELLAR.G, model)[channel]

    J_ad = jax.jacrev(f)(W0) * W0          # (K,) d sigma / d ln W0
    return f, W0, J_ad


def _central_fd_lnW0(f, W0, h):
    """Central-difference estimate of d sigma / d ln W0 = (d sigma/d W0) * W0."""
    return (f(W0 + h) - f(W0 - h)) / (2.0 * h) * W0


def _assert_ad_is_hto0_limit(model, channel, R_bins):
    """At EVERY bin, assert AD is the h->0 FD limit (ADR-0016 Richardson proof):

      (a) AD matches a CONVERGED (fine-step) FD to < 1e-3  -> AD value is correct;
      (b) the fixed-step FD CONVERGES toward AD as h shrinks (rel_fine < rel_coarse)
          -> any coarse-h gap is the FD's own truncation error, not a gradient defect.

    This is the load-bearing gate (it holds at high-curvature bins where a single
    fixed-step floor would mis-flag the CORRECT AD). Returns the per-bin fixed-step
    (h=1e-4) rel-errs so the caller can additionally floor the FD-RELIABLE bins.
    """
    f, W0, J_ad = _dsigma_dlnW0(model, channel, R_bins)
    assert jnp.all(jnp.isfinite(J_ad)), (model, channel, J_ad)   # AD finite everywhere

    fd_by_h = {h: _central_fd_lnW0(f, W0, h) for h in _FD_STEPS}
    rel_by_h = {
        h: jnp.abs(J_ad - fd) / (jnp.abs(fd) + 1e-30) for h, fd in fd_by_h.items()
    }
    rel_coarse = rel_by_h[_FD_STEPS[0]]                 # h = 1e-2
    rel_fine = rel_by_h[_FD_STEPS[-1]]                  # h = 1e-6 (converged proxy)

    # (a) AD == converged FD, every bin.
    assert jnp.all(rel_fine < _FD_FLOOR), (model, channel, "AD!=converged-FD", rel_fine)
    # (b) FD -> AD as h shrinks, every bin (Richardson trend).
    assert jnp.all(rel_fine < rel_coarse), (
        model, channel, "FD did not converge to AD as h shrank (would be a real defect)",
        rel_coarse, rel_fine,
    )
    return rel_by_h[_FD_FLOOR_H]                         # (K,) fixed-step rel at h=1e-4


def test_grad_sigma_W0_king_AD_vs_FD():
    """OM-King d sigma/d ln W0 AD-vs-FD, all bins, all 3 channels (ADR-0016).

    AD is the h->0 truth at EVERY bin (converged-FD match + Richardson trend). On the
    FD-RELIABLE bins (where the fixed-step central FD is a faithful proxy) the
    design's strict `rel < 1e-3` floor holds at h=1e-4. A SMALL set of mid-radius
    bins (R~2.2-3.1 r_c) sit where sigma_r(W0) has sharp curvature; there the
    FIXED-STEP h=1e-4 FD has O(h^2) truncation error ~1e-2 -- NOT a code defect: the
    Richardson sweep (above) shows that FD -> AD as h shrinks (rel ~1e-6 at h=1e-6),
    so AD is correct. We therefore floor only the FD-reliable bins and Richardson-gate
    the rest, per the measured characterization (design table probed only R=0.5,2,8
    and so did not surface these mid-radius high-curvature bins; AD remains correct).
    """
    for channel in range(3):
        rel_h4 = _assert_ad_is_hto0_limit("king", channel, oedc.R_BINS)
        # Strict design floor on the FD-RELIABLE bins (the vast majority).
        reliable = rel_h4 < _FD_FLOOR
        assert reliable.sum() >= oedc.R_BINS.shape[0] - 2, (channel, rel_h4)
        assert jnp.all(rel_h4[reliable] < _FD_FLOOR), (channel, rel_h4)


def test_grad_sigma_W0_michie_inner_AD_vs_FD():
    """OM-Michie d sigma/d ln W0 AD-vs-FD at R <= r_a, all 3 channels (ADR-0016).

    AD is the h->0 truth at every inner bin (converged-FD + Richardson trend). The
    design's strict `rel < 1e-3` fixed-step floor holds on the FD-reliable inner bins;
    a couple of inner bins near R~3-4.4 r_c sit at sharp sigma_r(W0) curvature where
    the fixed-step h=1e-4 FD truncation dominates (rel ~1e-2), but FD -> AD as h
    shrinks -- so AD is correct (NOT a defect). Floor the reliable inner bins,
    Richardson-gate the rest (measured, not tuned).
    """
    th = oedc.theta_truth()
    R_inner = oedc.R_BINS[oedc.R_BINS <= th[1]]        # R <= r_a (= 6 r_c)
    for channel in range(3):
        rel_h4 = _assert_ad_is_hto0_limit("michie", channel, R_inner)
        reliable = rel_h4 < _FD_FLOOR
        # The K-2 allowance is EXACTLY saturated here: the worst Michie-inner channel
        # has 2 high-curvature bins (R~3.1, 4.4 r_c) over the fixed-step floor, so the
        # margin is fully consumed. A FUTURE 3rd break (e.g. a profile-solver resolution
        # change) should be read as "investigate the AD gradient", NOT "bump K-2" -- the
        # Richardson trend above already proves AD is correct, so a new fixed-step break
        # means the curvature/FD characterization moved and must be re-measured.
        assert reliable.sum() >= R_inner.shape[0] - 2, (channel, rel_h4)
        assert jnp.all(rel_h4[reliable] < _FD_FLOOR), (channel, rel_h4)


def test_grad_sigma_W0_michie_outer_richardson():
    """OM-Michie d sigma/d ln W0 at the OUTERMOST bin: Richardson convergence (ADR-0016).

    At R = R_BINS[-1] (= 12 r_c, well beyond r_a = 6) Michie's r_t(W0) near-divergence
    makes a FIXED-STEP central FD a poor truth-proxy (design: ~8e-3 at R=8, h=1e-4),
    so asserting a fixed `rel < 1e-3` floor here would REINTRODUCE the exact problem and
    mis-flag a CORRECT AD. Instead we assert AD is the h->0 limit, mirroring the repo's
    ratified pattern (tests/unit/kinematics/test_dispersion.py
    ::test_grad_jeans_michie_high_W0_ad_correct): AD is finite, AD matches the converged
    fine-step FD, and the central FD CONVERGES toward AD as h shrinks (rel_fine <
    rel_coarse) -- the coarse-h gap is the FD's own O(h^2 sigma''') truncation error.

    NOTE (where the fixed-step floor actually breaks): after the C1 PCHIP fix the
    OUTERMOST bin (R=12) is itself FD-clean at h=1e-4 (measured rel ~5-8e-5); the
    FD-unreliable Michie bins are the MID-radius ones (R~3-4.4 r_c) where sigma_r(W0)
    bends sharply (a fixed h=1e-4 floor there breaks at ~1e-2 to ~1.5e-1). The
    Richardson trend is the faithful, curvature-agnostic gate and is what is asserted
    here and (per-bin) in the King/Michie-inner tests above; a naive ALL-BINS fixed
    h=1e-4 floor demonstrably FAILS (Task 3 characterization), confirming these gates
    have teeth.
    """
    R_outer = jnp.array([oedc.R_BINS[-1]])              # outermost bin, R = 12 r_c
    for channel in range(3):
        f, W0, J_ad = _dsigma_dlnW0("michie", channel, R_outer)
        assert jnp.isfinite(J_ad[0]), (channel, J_ad)   # AD finite (not a divergence)

        rels = [
            float(jnp.abs(J_ad[0] - _central_fd_lnW0(f, W0, h)[0])
                  / (jnp.abs(_central_fd_lnW0(f, W0, h)[0]) + 1e-30))
            for h in _FD_STEPS
        ]
        # AD matches the converged (finest-step) FD ...
        assert rels[-1] < _FD_FLOOR, (channel, "AD != converged-FD", rels)
        # ... and the FD CONVERGES toward AD as h shrinks (Richardson; ADR-0016).
        assert rels[-1] < rels[0], (
            channel, "FD did not converge to AD as h shrank -- a real gradient defect", rels
        )


# ---------------------------------------------------------------------------
# Task 4: per-star Fisher blocks, additive design Fisher, c/D/A criteria, optimizer.
#
# The per-star blocks Mb (3, K, 3, 3) and the additive design Fisher F = Sum n*c*M
# are reused from Stage-1 (model-agnostic once the (J, sigma) are built). The W0-OED
# arc threads its OWN prior: PRIOR_DIAG = [0, 0, 1/0.3**2] -- a prior on M (index 2,
# external integrated-light x M/L constraint) ONLY; W0 (target) and r_a (anisotropy,
# constrained by kinematics alone) carry ZERO prior. The (W0, r_a) 2-block must then
# be SPD from the DATA alone, which is the load-bearing SPD check below.
# ---------------------------------------------------------------------------


def test_blocks_shape_symmetry_and_fisher_spd():
    """per_star_blocks returns symmetric (3, K, 3, 3) blocks, and the additive design
    Fisher F (with the M-only PRIOR_DIAG) is SPD for BOTH models at a uniform design --
    i.e. the data constrains (W0, r_a) on its own (no prior needed on those)."""
    for model in ("king", "michie"):
        th = oedc.theta_truth()
        Mb, sig = oedc.per_star_blocks(th, oedc.R_BINS, oedc.EPS, STELLAR.G, model)
        K = oedc.R_BINS.shape[0]
        assert Mb.shape == (3, K, 3, 3)
        assert jnp.allclose(Mb, jnp.swapaxes(Mb, -1, -2), atol=1e-12)
        z = jnp.zeros(3 * K)
        F = oedc.fisher(z, Mb, oedc.completeness(oedc.R_BINS), 1000.0, oedc.PRIOR_DIAG)
        evals = jnp.linalg.eigvalsh(F)
        assert jnp.all(evals > 0), (model, evals)        # SPD with M-only prior


def test_fisher_spd_over_random_designs():
    """SPD escalation guard (measure, don't assume): with the locked PRIOR_DIAG, F must
    stay SPD across ~10 random design vectors z for BOTH models -- not just the uniform
    design. A singular/indefinite F at some generic z would mean the (W0, r_a) data-block
    is rank-deficient for that design and the prior must be escalated (recorded reason)."""
    for model in ("king", "michie"):
        th = oedc.theta_truth()
        Mb, _ = oedc.per_star_blocks(th, oedc.R_BINS, oedc.EPS, STELLAR.G, model)
        cb = oedc.completeness(oedc.R_BINS)
        K = oedc.R_BINS.shape[0]
        for s in range(10):
            z = jax.random.normal(jax.random.PRNGKey(100 + s), (3 * K,)) * 1.0
            F = oedc.fisher(z, Mb, cb, 1000.0, oedc.PRIOR_DIAG)
            evals = jnp.linalg.eigvalsh(F)
            assert jnp.all(evals > 0), (model, s, evals)


def test_c_criterion_targets_W0_and_grad_AD_vs_FD():
    """c_criterion with target=0 (W0) is finite and positive (a fractional variance),
    and grad of the criterion w.r.t. the design vector z is AD-vs-FD consistent
    (rtol 1e-4) -- pure 3x3 linalg, model-agnostic, so reverse-mode flows cleanly."""
    for model in ("king", "michie"):
        th = oedc.theta_truth()
        Mb, _ = oedc.per_star_blocks(th, oedc.R_BINS, oedc.EPS, STELLAR.G, model)
        cb = oedc.completeness(oedc.R_BINS)
        K = oedc.R_BINS.shape[0]
        loss = lambda z: oedc.c_criterion(
            oedc.fisher(z, Mb, cb, 1000.0, oedc.PRIOR_DIAG), target=0
        )
        z = jax.random.normal(jax.random.PRNGKey(1), (3 * K,)) * 0.5
        c0 = loss(z)
        assert jnp.isfinite(c0) and c0 > 0, (model, float(c0))
        g_ad = jax.grad(loss)(z)
        i = 5
        eps = 1e-4
        g_fd = (loss(z.at[i].add(eps)) - loss(z.at[i].add(-eps))) / (2 * eps)
        assert jnp.allclose(g_ad[i], g_fd, rtol=1e-4, atol=1e-8), (
            model, float(g_ad[i]), float(g_fd)
        )


# ===========================================================================
# Task 5: real-star calibration gate (KING ONLY, opt-in; kept OUT of CI)
# ===========================================================================


# Why BOTH @slow AND the env-skip (do not drop either): the FULL CI gate runs
# `pytest tests/unit tests/integration tests/validation -q -n auto` with NO
# `-m "not slow"` filter (see progenax/CLAUDE.md "FULL GATE"), so `@slow` ALONE would
# NOT keep this expensive (~minutes) MLE-calibration MC out of CI. The env-skip does:
# it runs only when explicitly opted in via PROGENAX_RUN_OED_CALIB=1 (figures / the
# eventual informax port). This matches Anna's "no expensive demos in CI" decision.
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PROGENAX_RUN_OED_CALIB") != "1",
    reason="expensive OED calibration; opt-in via PROGENAX_RUN_OED_CALIB=1 "
    "(figures/informax), kept out of CI",
)
def test_W0_fisher_calibration_matches_realized_scatter():
    """The HEADLINE gate: the design Fisher's predicted fractional variance of W0 must
    match the REALIZED fractional scatter of ln(W0_hat) over independent mock catalogs,
    each sampled from the OM model (sampler == Fisher forward model) and MAP-fit in the
    ln-theta Gauss-Newton metric started at the truth.

    KING ONLY (Option A): Michie's machinery is validated by the (cheap) forward
    (test_predict_sigma_*, test_jacobian_*), W0-gradient (test_grad_sigma_W0_michie_*),
    and Task-1 sampler-match (test_om_sampler_matches_project_dispersion) tests -- all
    still parametrized king/michie. Michie's expensive MLE-calibration MC is intentionally
    NOT run: it adds no new anisotropy physics over King, only ~2x cost (the Michie
    sampler/fitter code remains, exercised by those cheaper tests).

    Both quantities are fractional/ln variances (ADR-0011). The tolerance band is the
    Monte-Carlo error on a variance estimated from n_draws draws (~2 sqrt(2/n_draws));
    if the ratio is outside the band the design Fisher is wrong -- root-cause it, do NOT
    widen the band."""
    key = jax.random.PRNGKey(7)
    K = oedc.R_BINS.shape[0]
    cal = oedc.calibrate_fisher_W0(
        z=jnp.zeros(3 * K), N_total=400.0, n_draws=48, key=key, model="king"
    )
    band = 2.0 * (2.0 / 48) ** 0.5
    ratio = cal.realized_var_W0 / cal.fisher_var_W0
    assert jnp.abs(ratio - 1.0) < band, ("king", ratio)
