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
