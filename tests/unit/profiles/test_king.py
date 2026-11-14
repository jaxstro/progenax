# progenax/tests/unit/profiles/test_king.py
"""
Unit tests for KingProfile audit invariants.

Covers the unit-specific guards: r_t boundary-pinning (audit J4), the
r_t-consistency UserWarning (audit S1), and the differentiable tidal radius.

ODE-solution and profile-property physics (truncation, isotropy, concentration,
density, equilibrium DF, King Table II) is covered more thoroughly in the
validation tier: tests/validation/test_king_physics.py. The redundant unit
duplicates were removed in the 2026-06 pre-release test consolidation.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.profiles import KingProfile, solve_king_profile


class TestRtBoundaryPinning:
    """Audit J4: a too-small ODE domain leaves psi(xi)>0 everywhere, so
    _find_tidal_radius silently pins xi_t to the grid boundary (a wrong r_t).
    Concrete construction must REFUSE loudly; traced construction can't raise
    on a traced bool, so it stores an r_t_is_pinned diagnostic instead (the
    Engine-B two-tier concrete/traced pattern)."""

    def test_healthy_solve_not_flagged(self):
        prof = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
        assert not bool(prof.r_t_is_pinned)

    def test_concrete_pinned_solve_raises(self):
        """Eager construction with a domain that pins xi_t refuses loudly."""
        with pytest.raises(ValueError, match="pinned|xi_max"):
            KingProfile.from_W0_rc(W0=12.0, r_c=1.0, xi_max=50.0, n_ode_points=500)

    def test_traced_pinned_solve_is_flagged_not_raised(self):
        """Under tracing the raise is impossible (traced bool); the flag is set."""
        prof = jax.jit(
            lambda w: KingProfile.from_W0_rc(
                W0=w, r_c=1.0, xi_max=50.0, n_ode_points=500
            )
        )(12.0)
        assert bool(prof.r_t_is_pinned)

    def test_traced_healthy_solve_constructs(self):
        prof = jax.jit(lambda w: KingProfile.from_W0_rc(W0=w, r_c=1.0))(7.0)
        assert jnp.isfinite(prof.r_t) and not bool(prof.r_t_is_pinned)


class TestRtConsistencyWarning:
    """Audit S1: the direct KingProfile constructor accepts an arbitrary r_t
    inconsistent with c(W0) — KingProfile(W0=7, r_c=1, r_t=10) silently builds
    a non-self-consistent, non-equilibrium model. Concrete inputs now warn."""

    def test_inconsistent_r_t_warns(self):
        xi_grid, psi_grid, _ = solve_king_profile(7.0)  # c(7): r_t/r_c ~ 30, not 10
        with pytest.warns(UserWarning, match="inconsistent"):
            KingProfile(W0=7.0, r_c=1.0, r_t=10.0, xi_grid=xi_grid, psi_grid=psi_grid)

    def test_consistent_r_t_no_warning(self, recwarn):
        KingProfile.from_W0_rc(W0=7.0, r_c=1.0)  # derives r_t -> self-consistent
        assert not any("inconsistent" in str(w.message) for w in recwarn.list)


class TestTidalRadiusVacuumReference:
    """The zero of the exterior A+B/r potential independently checks the root."""

    @pytest.mark.parametrize("W0", [6.0, 7.0, 8.0, 9.0])
    def test_root_matches_exterior_vacuum_solution(self, W0):
        from progenax.profiles.king import _find_tidal_radius, solve_king_profile

        xi, _, raw = solve_king_profile(W0, xi_max=400.0, n_points=8000)
        # Both samples must be outside the density support for the vacuum relation.
        xa, xb = xi[6000], xi[-1]
        ya, yb = raw[6000], raw[-1]
        assert float(ya) < 0 and float(yb) < 0
        coefficient = (ya - yb) / (1 / xa - 1 / xb)
        offset = ya - coefficient / xa
        reference = -coefficient / offset
        actual = _find_tidal_radius(xi, raw)
        assert float(actual) == pytest.approx(float(reference), rel=1e-5)


class TestKingInputValidation:
    """Invalid concrete inputs must raise eagerly (audit S7): W0<=0 silently gave a
    degenerate r_t=1e-6 r_c model; r_c<=0 is unphysical."""

    def test_nonpositive_W0_raises(self):
        import pytest

        from progenax import KingProfile

        with pytest.raises(ValueError, match="W0"):
            KingProfile.from_W0_rc(-1.0, 1.0)

    def test_nonpositive_r_c_raises(self):
        import pytest

        from progenax import KingProfile

        with pytest.raises(ValueError, match="r_c"):
            KingProfile.from_W0_rc(7.0, -1.0)
