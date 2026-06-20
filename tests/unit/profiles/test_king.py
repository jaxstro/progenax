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


class TestDifferentiableTidalRadius:
    """Audit Task 1.2b: KingProfile.from_W0_rc(...).r_t is differentiable in W0.

    The original code fed CLAMPED psi (psi>=0) to _find_tidal_radius, so the
    zero-crossing node had psi1=0 exactly -> interpolation fraction t=1 -> xi_t
    SNAPPED to the grid node, with d(xi_t)/dW0 = 0 by autodiff (a silent-zero
    hazard: the value still moved with W0 via the grid, but the gradient was 0).
    The fix feeds UNCLAMPED psi (psi_raw, the 3rd return): psi1<0 at the crossing, so
    xi_t is a true linear interpolation and d(r_t)/dW0 flows through the diffrax
    solve (the implicit-function-theorem result to grid accuracy).
    """

    # Forward-VALUE pin (gradient-only change must NOT silently corrupt the value).
    #
    # NOTE (Task 1.2b, Option A — forward value is NOT bit-for-bit identical to
    # the PRE-fix code, by design): the pre-fix clamped path SNAPPED xi_t to the
    # grid node (psi1 clamped to 0 -> t=1). At W0=8 (xi_max=400, n=8000) the
    # pre-fix value was r_t = 68.158520644580577; the fix interpolates the true
    # crossing -> 68.146780078164710 (psi1 ~ -7e-5 -> t ~ 0.765), a ~0.0117 shift
    # (~23% of one grid cell, ~6e-4 relative — far below ODE/grid accuracy). The
    # unclamped interpolation is BOTH differentiable AND a more accurate crossing
    # than the grid-snap, so these NEW values are the honest regression baseline.
    # PRE-fix (snapped) values, for the record: W0 6->17.99225..., 7->33.75422...,
    # 8->68.15852..., 9->131.41642... .
    _EXPECTED_RT = {  # post-fix (true linear-interp crossing); xi_max=400, n=8000
        6.0: 17.991633186676829,
        7.0: 33.708576645085365,
        8.0: 68.146780078164710,
        9.0: 131.38070080188584,
    }

    @pytest.mark.parametrize("W0,r_t_expected", sorted(_EXPECTED_RT.items()))
    def test_forward_r_t_value_pinned(self, W0, r_t_expected):
        """r_t value is pinned (the fix is gradient-correctness, not a physics
        rewrite). Tight 1e-10 absolute pin proves the value is reproducible."""
        prof = KingProfile.from_W0_rc(W0=W0, r_c=1.0, xi_max=400.0, n_ode_points=8000)
        assert float(prof.r_t) == pytest.approx(r_t_expected, abs=1e-10)

    # AD-vs-FD for KingProfile.r_t(W0) is owned by the grad-audit registry
    # (tests/validation/grad_audit/registry.py :: KingProfile.r_t, same xi_max=400/
    # n_ode_points=8000 config); see docs/website/50-validation/differentiability-audit.md.
    # The former test_r_t_grad_is_nonzero_and_fd_consistent was removed here (audit T6
    # consolidation; registry is SoT). The forward-VALUE pin above stays (unique regression).
