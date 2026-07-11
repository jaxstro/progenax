"""Basic pins for the retained binary-robustness OED demo (scripts/demo_oed_binary.py).

Demo-harness tier — NOT released-core (scripts-only, no ``src/progenax/`` change).

2026-07 test trim (quality-over-quantity pass): the OED tooling migrated to
informax, and this file was cut from 36 tests to the load-bearing pins for the
ONE demo progenax retains (binary RVs vs. the inferred cluster mass). The full
inference / design-machinery gates (H0/H1/H2/H3 MC, marginalized Fisher, maximin,
sweeps) now live in informax; the science they certified is frozen in the arc's
ADRs (0019/0020) and the shipped report. Kept here:

* ``test_vbin_and_sigma_ratio_bites`` — the headline's honesty gate: the
  flux-weighted binary blend-velocity scale rivals the cluster dispersion at the
  YMC operating point (``sigma_bin / sigma_cluster > 0.5``). The threshold is NOT
  to be weakened — if the physical operating point stops biting, the demo's
  premise is dead and the test must fail loudly.
* ``test_cluster_sigma_los_matches_project_dispersion`` — forward-model
  integrity: the demo's cluster channel equals the released
  ``project_dispersion`` oracle, with the core->outskirt radial leverage that
  breaks the M <-> f_bin degeneracy.
* ``test_sigma_cluster_ref_is_max_of_cluster_sigma_los`` — the single source of
  truth behind the ratio gate's denominator.
* ``test_predict_sigma_obs_adds_binary_pedestal`` — the misspecification itself:
  sigma_obs^2 = sigma_cluster^2 + f_bin * V_BIN, exactly.
* ``test_cli_binary_quick_smoke`` (@slow) — the demo CLI runs end-to-end.
"""

import pathlib
import sys

import jax.numpy as jnp
import pytest
from jaxstro.units import STELLAR

import progenax  # noqa: F401  -- enables float64 at import

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
# The demo helper imports optax (experimental extra). Skip the whole module when
# optax is absent so released-core runs on --extra dev.
pytest.importorskip("optax")
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


def test_cluster_sigma_los_matches_project_dispersion():
    """Per-bin cluster_sigma_los(theta_clusteronly, R, G) equals the direct
    project_dispersion(...).sigma_los oracle (km/s), is everywhere positive, and the
    sigma_los profile DECLINES from the dense core to the outskirts (radial leverage:
    M info lives in the high-sigma core, the flat binary pedestal dominates in the
    low-sigma outskirts)."""
    from progenax import project_dispersion

    th = oedb.theta_truth_clusteronly()  # (M, r_a, gamma, a)
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
    assert float(sig[0]) > 5.0 * float(sig[-1])  # >5x core->outskirt contrast
    peak = int(jnp.argmax(sig))
    assert peak < sig.shape[0] - 1  # the peak is interior, not the last bin
    assert bool(jnp.all(jnp.diff(sig[peak:]) < 0.0))  # strictly declining past the peak


def test_sigma_cluster_ref_is_max_of_cluster_sigma_los():
    """sigma_cluster_ref() is the single source of truth = max over bins of
    cluster_sigma_los at the fiducial theta (the central / peak dispersion)."""
    th = oedb.theta_truth_clusteronly()
    sig = oedb.cluster_sigma_los(th, oedb.R_BINS, STELLAR.G)
    assert jnp.isclose(oedb.sigma_cluster_ref(), jnp.max(sig), rtol=1e-12)


def test_predict_sigma_obs_adds_binary_pedestal():
    """sigma_obs^2 = sigma_cluster^2 + f_bin * V_BIN (the flat binary pedestal),
    exactly (rtol 1e-10), and the observable is everywhere larger than the bare
    cluster dispersion (binaries inflate the second moment)."""
    th = oedb.theta_truth()  # (M, r_a, gamma, a, f_bin)
    R = oedb.R_BINS
    sig_obs = oedb.predict_sigma_obs(th, R, STELLAR.G)  # (K,) km/s
    sig_cluster = oedb.cluster_sigma_los(th[:4], R, STELLAR.G)
    expected2 = sig_cluster**2 + oedb.th_fbin(th) * oedb.V_BIN
    assert sig_obs.shape == (R.shape[0],)
    assert jnp.allclose(sig_obs**2, expected2, rtol=1e-10)
    assert bool(jnp.all(sig_obs > sig_cluster))  # pedestal strictly inflates


@pytest.mark.slow  # demo CLI end-to-end; full/release gate only
def test_cli_binary_quick_smoke(tmp_path):
    """The CLI --quick path runs end-to-end (design + forecast + the no-MC mechanism
    figure) and exits 0 WITHOUT the cross-model MC or its env var.

    Dialed down via --quick (few Adam starts/steps). Asserts rc == 0, that the mechanism
    PNG lands in --outdir, that the false-confidence (MC-only) figure is NOT produced in
    --quick mode, and that the run-record was written under --outdir (NOT the committed
    FIGURE_DIR) -- the Stage-3 fixed-path guard.
    """
    # The CLI plots via matplotlib (pulled in through jaxstroviz, not a progenax dep).
    # Skip this smoke when matplotlib is absent (e.g. released-core --extra dev env).
    pytest.importorskip("matplotlib")
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
