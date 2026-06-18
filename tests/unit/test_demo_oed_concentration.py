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
    """Binned LOS dispersion of the OM sampler must match project_dispersion (the
    Fisher forward model) -- otherwise the calibration gate would test mismatched
    models. Anchored at a mid radius R ~ 2 r_c."""
    key = jax.random.PRNGKey(0)
    W0, r_a, M = 6.0, 6.0, 1e5
    R, v_los, v_pm_r, v_pm_t = oedc.sample_om_cluster(
        model, W0, r_a, M, n_stars=200_000, key=key
    )
    prof = oedc.build_profile(W0, r_a, model)
    R_probe = jnp.array([2.0])
    pred = progenax.project_dispersion(prof, r_a, R_probe, M, STELLAR.G).sigma_los[0]
    sel = (R > 1.6) & (R < 2.4)
    meas = jnp.std(v_los[sel], ddof=1)
    assert sel.sum() > 2000, sel.sum()
    assert jnp.abs(meas - pred) / pred < 0.05, (model, float(meas), float(pred))
