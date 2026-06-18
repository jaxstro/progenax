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
