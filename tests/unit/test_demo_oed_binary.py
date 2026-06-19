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
