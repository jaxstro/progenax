import jax, jax.numpy as jnp, sys, pathlib
import pytest
import progenax
from jaxstro.units import STELLAR
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import _demo_oed as oed
import _demo_oed_depth as oed_depth


def test_blocks_from_eps_matches_per_star_blocks():
    th = oed.theta_truth()
    J, sig = oed.jacobian_and_sigma(th, oed.R_BINS, STELLAR.G)   # NEW: (3,K,3),(3,K)
    Mb_ref, _ = oed.per_star_blocks(th, oed.R_BINS, oed.EPS, STELLAR.G)
    Mb = oed.blocks_from_eps(J, sig, oed.EPS)                    # NEW: rebuild at given eps
    assert jnp.allclose(Mb, Mb_ref, atol=1e-12)
    # different eps -> different blocks (larger eps -> smaller information)
    Mb_noisy = oed.blocks_from_eps(J, sig, 2.0 * oed.EPS)
    assert jnp.all(jnp.diagonal(Mb_noisy, axis1=-2, axis2=-1)
                   <= jnp.diagonal(Mb, axis1=-2, axis2=-1) + 1e-12)


def test_eps_eff_rises_with_depth():
    e_shallow = oed_depth.eps_eff(m_lim=10.0)     # (3,) per channel
    e_deep    = oed_depth.eps_eff(m_lim=14.0)
    assert jnp.all(e_deep > e_shallow)            # admitting faint stars raises the mean error


def test_availability_rises_with_depth_and_is_radial():
    a_shallow = oed_depth.avail_bins(m_lim=10.0)  # (K,)
    a_deep    = oed_depth.avail_bins(m_lim=14.0)
    assert jnp.all(a_deep >= a_shallow)           # deeper detects more per bin
    assert a_shallow[0] > a_shallow[-1]           # outskirts star-starved (lower density)


def test_depth_fisher_spd_and_targets_M():
    z = jnp.zeros(3 * oed.R_BINS.shape[0])
    F = oed_depth.depth_fisher(z, m_lim=12.0, N_total=4000.0)
    assert F.shape == (3, 3) and jnp.allclose(F, F.T, atol=1e-10)
    assert jnp.all(jnp.linalg.eigvalsh(F) > 0)
    # c-criterion targeting M (index 1) is finite & positive
    assert oed.c_criterion(F, target=1) > 0


def test_depth_criterion_grad_AD_vs_FD():
    z = jax.random.normal(jax.random.PRNGKey(0), (3 * oed.R_BINS.shape[0],)) * 0.1
    u = jnp.array(0.3)                              # m_lim via expit into [m_lo, m_hi]
    loss = lambda zz, uu: oed.c_criterion(oed_depth.depth_fisher_u(zz, uu, 4000.0), target=1)
    g_ad = jax.grad(loss, argnums=(0, 1))(z, u)
    eps = 1e-5
    # FD on m_lim (the new dimension) and a few z coords
    g_fd_u = (loss(z, u + eps) - loss(z, u - eps)) / (2 * eps)
    assert jnp.allclose(g_ad[1], g_fd_u, rtol=1e-4, atol=1e-8)
    for i in (0, 17, 31):
        zp = z.at[i].add(eps); zm = z.at[i].add(-eps)
        assert jnp.allclose(g_ad[0][i], (loss(zp, u) - loss(zm, u)) / (2 * eps),
                            rtol=1e-4, atol=1e-8)


def test_joint_optimizer_beats_fixed_depth():
    # N_total=400 is in the SELECTIVELY-BINDING regime: the availability cap binds
    # (ratio > 1) at shallow m_lim<=12 but is loose (ratio < 1) at deep m_lim>=13,
    # so depth is a genuine trade. (4000 would saturate everywhere -- degenerate.)
    res = oed_depth.optimize_depth_design(target=1, N_total=400.0,
                                          key=jax.random.PRNGKey(1), n_starts=6, n_steps=400)
    # the jointly-optimised design beats a shallow and a very-deep fixed depth
    assert res.criterion < oed_depth.crit_at_fixed_depth(m_lim=10.0, target=1, N_total=400.0)
    assert res.criterion < oed_depth.crit_at_fixed_depth(m_lim=16.0, target=1, N_total=400.0)


def test_sigma_M_has_interior_optimum_in_depth():
    # N_total=400 is in the SELECTIVELY-BINDING regime (Task-4 review I2); 4000 saturates
    # the availability cap everywhere and the depth trade degenerates. sigma(M)/M vs m_lim
    # has an INTERIOR minimum: too-shallow surveys are supply-starved (few bright stars,
    # esp. in the outskirts), too-deep ones are photon-noise-limited.
    m_grid = jnp.linspace(oed_depth.M_LIM_LO, oed_depth.M_LIM_HI, 13)
    sigM = oed_depth.sigma_M_vs_depth(m_grid, target=1, N_total=400.0, n_starts=3, n_steps=250)
    i = int(jnp.argmin(sigM))
    assert 0 < i < len(m_grid) - 1                 # INTERIOR minimum, not an endpoint
    assert sigM[i] < sigM[0] and sigM[i] < sigM[-1]


@pytest.mark.slow
def test_depth_fisher_calibration_is_validated_and_bounded():
    """The magnitude-selected Monte-Carlo calibration VALIDATES the depth Fisher for M_dyn: the
    realized sigma(M) matches the Fisher prediction to within ~15%, with no significant systematic bias.

    Measured with the corrected Gauss-Newton ln-theta fit (_fit_theta_gn), the variance ratio
    realized/predicted is DESIGN-dependent and brackets 1.0: ~0.84 +/- 0.08 at one optimal design
    (design seed 1), ~1.05 +/- 0.09 at another (design seed 0) -- i.e. consistent with 1.0 to ~15%,
    no consistent conservative OR anti-conservative bias. (An earlier "~19% conservative" figure was an
    artefact of an under-converged physical-theta Adam fit, since replaced by GN; once corrected, the
    apparent bias was design/seed-specific MC scatter, not a systematic.)

    Both the depth Fisher and the calibration fit use the M-free Stage-2 prior (PRIOR_DIAG_M) and the
    ln-theta metric (ADR 0011), so the two are consistent by construction.

    This gate is DETERMINISTIC (fixed design seed 1, calib seed 0 -> ratio ~0.79) and asserts the ratio
    lies in a sanity band [0.25, 1.15]. The bounds are NOT a tight calibration claim (the cross-design
    range needs more seeds); they verify the Fisher is the right order of magnitude and, crucially, the
    LOWER bound guards against a regression to the old physical-theta-fit bug (which pinned M_hat ->
    ratio ~ 0).
    """
    key = jax.random.PRNGKey(0)
    res = oed_depth.optimize_depth_design(target=1, N_total=400.0, key=jax.random.PRNGKey(1),
                                          n_starts=6, n_steps=400)
    n_draws = 64
    realized, predicted = oed_depth.calibrate_depth_fisher(res.z, res.m_lim, 400.0, n_draws, key)
    ratio = realized / predicted                   # variance ratio; brackets 1.0 across designs
    assert 0.25 < ratio < 1.15                     # sanity band; lower bound guards the old M-pinned bug


def test_cli_dynamical_mass_smoke(tmp_path):
    """Smoke-test the Stage-2 gated CLI: run main() end to end with a SMALL config and assert it
    exits 0 (all gates pass) and writes a run-record JSON with the expected schema.

    The heavy knobs (joint-optimiser starts/steps, sweep density, calibration draws) are dialled
    down via the CLI module's own constants so the test stays in the fast suite, but NOT so far that
    the gates become meaningless: the interior-argmin and beats-fixed-depth gates still exercise the
    real depth physics, and the calibration gate still runs a (small) magnitude-selected ensemble.
    """
    import importlib

    cli = importlib.import_module("demo_oed_dynamical_mass")
    # Dial down the cost; keep the gates real (interior sweep + small calibration ensemble).
    cli.N_STARTS, cli.N_STEPS = 3, 200
    cli.N_SWEEP_QUICK = 9
    cli.SWEEP_STARTS, cli.SWEEP_STEPS = 2, 200
    cli.N_DRAWS_QUICK = 8
    cli.N_CAL_SEEDS_QUICK = 2
    # This is a PLUMBING smoke test (CLI runs end-to-end + writes a valid record + the deterministic
    # interior/beats-fixed gates hold). At 2x8 draws the calibration ratio carries ~50% MC noise, so we
    # widen ITS band here to tolerate that while still catching a catastrophic regression (the old
    # M-pinned-fit bug -> ratio ~ 0). The calibration PHYSICS is validated separately, at 64 draws, by
    # test_depth_fisher_calibration_is_validated_and_bounded (NOT weakened).
    cli.CAL_RATIO_LO, cli.CAL_RATIO_HI = 0.05, 10.0

    out = tmp_path / "stage2_smoke_record.json"
    rc = cli.main(["--seed", "0", "--out", str(out)])
    assert rc == 0, "Stage-2 CLI gates failed in the smoke config"
    assert out.exists()

    import json
    rec = json.loads(out.read_text())
    assert rec["all_pass"] is True
    for key in ("demo", "timestamp_utc", "params", "results", "gates", "design_z"):
        assert key in rec
    res = rec["results"]
    for key in ("joint_optimum", "contrast", "depth_trade", "calibration"):
        assert key in res
    assert "m_lim" in res["joint_optimum"] and "frac_sigma_M" in res["joint_optimum"]
    assert "variance_ratio" in res["calibration"]
    assert 0 < res["contrast"]["sweep_argmin_index"] < cli.N_SWEEP_QUICK - 1
