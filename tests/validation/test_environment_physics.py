"""
Physics validation for the environment-dependent IMF (Marks+2012 / Jeřábková+2018).

These tests assert the *published* anchor values for the top-heavy IMF slope alpha3
and the metallicity-dependent low-mass slopes against the implemented mapping --
i.e. external published-table oracles, not self-consistency. The same oracles are
exercised more granularly in tests/unit/imf/test_environment.py; this validation
tier pins the headline published results (Marks+2012 Table 1 globular clusters and
Table 4 low-mass slopes) and the qualitative physics (density dominates metallicity)
so they are surfaced in the publication validation suite and figures
(scripts/validate_environment.py, docs/website/50-validation/environment-imf.md).

References:
    Marks et al. (2012), MNRAS 422, 2246 -- Table 1 (GC birth conditions),
        Table 4 (low-mass slopes), Eq. 12, Eq. 14-15 (Fundamental Plane).
    Jeřábková et al. (2018), A&A 620, A39 -- IGIMF alpha3 mapping (Eq. 6-7).
"""

import jax.numpy as jnp
import pytest

from progenax.imf.environment.mapping import (
    alpha3_marks_plane,
    lowmass_slopes_metallicity,
)

# Marks+2012 Table 1: (name, [Fe/H], rho_cl [1e6 Msun/pc^3], published alpha3).
# Table 1 caption notes scatter ~0.15 about the best-fit Fundamental Plane.
MARKS_TABLE1_GCS = [
    ("NGC 104", -0.76, 9.54, 1.34),
    ("NGC 6341", -2.28, 66.03, 1.11),
    ("NGC 6752", -1.56, 31.78, 1.27),
    ("NGC 7078", -2.16, 258.13, 0.76),
]

# Marks+2012 Table 4: [Fe/H] -> (alpha1, alpha2) low-mass slopes (Eq. 12).
MARKS_TABLE4_SLOPES = [
    (-2.0, 0.30, 1.30),
    (-1.5, 0.55, 1.55),
    (-1.0, 0.80, 1.80),
    (0.0, 1.30, 2.30),
    (0.5, 1.55, 2.55),
]


class TestMarksTable1GlobularClusters:
    """alpha3 from the Marks+2012 Fundamental Plane vs Table 1 GC values."""

    @pytest.mark.parametrize("name,FeH,rho_1e6,alpha3_pub", MARKS_TABLE1_GCS)
    def test_gc_alpha3_matches_published(self, name, FeH, rho_1e6, alpha3_pub):
        log_rho_6 = jnp.log10(jnp.asarray(rho_1e6))
        computed = float(alpha3_marks_plane(log_rho_6, jnp.asarray(FeH)))
        # atol 0.20 absorbs the ~0.15 intrinsic scatter about the best-fit plane.
        assert abs(computed - alpha3_pub) < 0.20, (
            f"{name}: alpha3 computed {computed:.3f} vs published {alpha3_pub} "
            f"(|delta|={abs(computed - alpha3_pub):.3f} > 0.20)"
        )

    def test_ngc7078_is_most_top_heavy(self):
        """NGC 7078 (M15) is the most top-heavy GC in Marks+2012 Table 1."""
        a3 = {
            name: float(alpha3_marks_plane(jnp.log10(jnp.asarray(rho)), jnp.asarray(feh)))
            for name, feh, rho, _ in MARKS_TABLE1_GCS
        }
        assert a3["NGC 7078"] == min(a3.values()), a3
        # and it is genuinely top-heavy (alpha3 < canonical Salpeter/Kroupa 2.3)
        assert a3["NGC 7078"] < 1.0

    def test_density_dominates_metallicity(self):
        """Increasing density at fixed [Fe/H] makes alpha3 more top-heavy faster
        than the same dex change in [Fe/H] (Fundamental-Plane sin/cos ~ 7)."""
        feh0, lr0 = -1.5, jnp.log10(jnp.asarray(30.0))
        base = float(alpha3_marks_plane(lr0, jnp.asarray(feh0)))
        d_rho = base - float(alpha3_marks_plane(lr0 + 1.0, jnp.asarray(feh0)))
        d_feh = base - float(alpha3_marks_plane(lr0, jnp.asarray(feh0 + 1.0)))
        assert d_rho > 0.0, "more density should lower alpha3 (more top-heavy)"
        assert abs(d_rho) > 3.0 * abs(d_feh), (
            f"density response {d_rho:.3f} should dominate metallicity {d_feh:.3f}"
        )


class TestMarksTable4LowMassSlopes:
    """Metallicity-dependent low-mass slopes vs Marks+2012 Table 4 (Eq. 12)."""

    @pytest.mark.parametrize("FeH,a1_pub,a2_pub", MARKS_TABLE4_SLOPES)
    def test_lowmass_slopes_match_table4(self, FeH, a1_pub, a2_pub):
        a1, a2 = lowmass_slopes_metallicity(jnp.asarray(FeH))
        assert abs(float(a1) - a1_pub) < 0.02, f"alpha1({FeH}) {float(a1):.3f} vs {a1_pub}"
        assert abs(float(a2) - a2_pub) < 0.02, f"alpha2({FeH}) {float(a2):.3f} vs {a2_pub}"

    def test_slopes_steepen_with_metallicity(self):
        """Both low-mass slopes increase monotonically with [Fe/H]."""
        fehs = jnp.linspace(-2.0, 0.5, 6)
        a1 = jnp.array([lowmass_slopes_metallicity(f)[0] for f in fehs])
        a2 = jnp.array([lowmass_slopes_metallicity(f)[1] for f in fehs])
        assert bool(jnp.all(jnp.diff(a1) > 0)), "alpha1 should increase with [Fe/H]"
        assert bool(jnp.all(jnp.diff(a2) > 0)), "alpha2 should increase with [Fe/H]"


class TestErratumCorrectedPlane:
    """The 2014 erratum: the Marks Fundamental Plane uses threshold x_hat >= -0.87,
    so it coincides with the Jerabkova (2018) IGIMF density relation (which adopts the
    same corrected form). The originally printed +0.87 was a missing-minus-sign typo.
    """

    def test_marks_threshold_is_erratum_value(self):
        from progenax.imf.environment.coefficients import MARKS_COEFFICIENTS as M
        assert M["x_hat_threshold"] == -0.87, "must use the 2014-erratum threshold"
        # the corrected line meets canonical 2.3 continuously at the threshold
        knee = M["alpha3_slope"] * M["x_hat_threshold"] + M["alpha3_intercept"]
        assert abs(knee - 2.3) < 0.02, "erratum threshold must give a continuous knee"

    def test_corrected_marks_equals_jerabkova(self):
        from progenax.imf.environment.mapping import (
            alpha3_marks_plane, alpha3_jerabkova_rho,
        )
        lr = jnp.linspace(-1.0, 3.0, 50)
        for feh in (-2.0, -1.0, 0.0):
            am = alpha3_marks_plane(lr, jnp.full_like(lr, feh))
            aj = alpha3_jerabkova_rho(lr, jnp.full_like(lr, feh))
            gap = float(jnp.max(jnp.abs(am - aj)))
            # identical up to the -0.4072-vs-(-0.41) slope rounding
            assert gap < 0.05, f"[Fe/H]={feh}: corrected Marks vs Jerabkova gap {gap:.3f}"

    def test_canonical_recovered_at_low_density(self):
        """Below the knee (x_hat < -0.87) the corrected plane returns canonical 2.3."""
        from progenax.imf.environment.mapping import alpha3_marks_plane
        # FeH=0, log_rho_6 = -2 -> x_hat = -1.98 < -0.87
        a3 = float(alpha3_marks_plane(jnp.asarray(-2.0), jnp.asarray(0.0)))
        assert abs(a3 - 2.3) < 0.01, f"diffuse field must be canonical, got {a3:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
