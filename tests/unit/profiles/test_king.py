# progenax/tests/unit/profiles/test_king.py
"""
Unit tests for KingProfile.

Physics tests only - ODE solution and profile properties.
"""

import jax
import jax.numpy as jnp
import pytest
from progenax.profiles import KingProfile, solve_king_profile


class TestSolveKingProfile:
    """Test solve_king_profile() ODE solver."""

    def test_boundary_conditions(self):
        """ψ(0) ≈ W0, ψ → 0 at tidal radius."""
        xi_grid, psi_grid, _ = solve_king_profile(W0=7.0, xi_max=50.0, n_points=500)

        # Central potential should be close to W0
        assert jnp.isclose(psi_grid[0], 7.0, atol=0.1)

        # Potential should decay to zero
        assert psi_grid[-1] < 0.5  # Should be nearly zero at large xi

    def test_monotonic_decrease(self):
        """ψ(ξ) decreases monotonically."""
        xi_grid, psi_grid, _ = solve_king_profile(W0=5.0, xi_max=30.0, n_points=500)

        # Check that potential is non-increasing
        diff = jnp.diff(psi_grid)
        assert jnp.all(diff <= 1e-6)  # Allow small numerical noise

    def test_different_W0(self):
        """Higher W0 gives steeper profile (tidal radius at larger xi)."""
        xi1, psi1, _ = solve_king_profile(W0=3.0, xi_max=20.0, n_points=500)
        xi2, psi2, _ = solve_king_profile(W0=7.0, xi_max=50.0, n_points=500)

        # Find where ψ drops to 0.1
        idx1 = jnp.argmax(psi1 < 0.1)
        idx2 = jnp.argmax(psi2 < 0.1)

        # Higher W0 should have larger dimensionless tidal radius
        assert xi2[idx2] > xi1[idx1]

    def test_non_negative_potential(self):
        """All ψ values are non-negative."""
        xi_grid, psi_grid, _ = solve_king_profile(W0=7.0, xi_max=50.0, n_points=500)
        assert jnp.all(psi_grid >= 0.0)


class TestKingPhysics:
    """Test KingProfile physical properties."""

    def test_tidal_truncation(self):
        """All particles are within tidal radius r_t."""
        # self-consistent constructor (recommended API); r_t derived from W0
        profile = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
        masses = jnp.ones(1000)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        radii = jnp.linalg.norm(positions, axis=1)
        assert jnp.all(radii <= float(profile.r_t) * 1.01)  # Allow small numerical tolerance

    def test_isotropy(self):
        """Angular distribution is isotropic."""
        # self-consistent constructor (recommended API); r_t derived from W0
        profile = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)
        N = 1000
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        # Check mean position is near origin (within ~3σ/√N tolerance)
        mean_pos = jnp.mean(positions, axis=0)
        assert jnp.all(jnp.abs(mean_pos) < 0.1), f"Mean pos {mean_pos} too far from origin"

        # Check each axis has similar spread (ratio < 1.3)
        stds = jnp.array([
            jnp.std(positions[:, 0]),
            jnp.std(positions[:, 1]),
            jnp.std(positions[:, 2]),
        ])
        ratio = jnp.max(stds) / jnp.min(stds)
        assert ratio < 1.3, f"Std ratio {float(ratio):.3f} > 1.3 (not isotropic)"

    def test_concentration_effect(self):
        """Higher W0 gives more concentrated distribution."""
        # self-consistent constructor (recommended API); r_t derived from W0
        # Low concentration
        profile1 = KingProfile.from_W0_rc(W0=3.0, r_c=1.0)

        # High concentration
        profile2 = KingProfile.from_W0_rc(W0=9.0, r_c=1.0)

        masses = jnp.ones(5000)

        # Use different seeds to ensure independent samples
        pos1 = profile1.sample_positions(masses, jax.random.PRNGKey(42))
        pos2 = profile2.sample_positions(masses, jax.random.PRNGKey(43))

        radii1 = jnp.linalg.norm(pos1, axis=1)
        radii2 = jnp.linalg.norm(pos2, axis=1)

        # Both distributions should be reasonable
        median_r1 = jnp.median(radii1)
        median_r2 = jnp.median(radii2)

        assert median_r1 > 0.0
        assert median_r2 > 0.0
        assert median_r1 < float(profile1.r_t)
        assert median_r2 < float(profile2.r_t)

    def test_characteristic_radius_returns_r_t(self):
        """characteristic_radius() returns r_t (tidal radius)."""
        # self-consistent constructor (recommended API); r_t derived from W0
        profile = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
        assert jnp.isclose(profile.characteristic_radius(), profile.r_t)

    def test_jit_compatible(self):
        """sample_positions() works with JIT compilation."""
        # self-consistent constructor (recommended API); r_t derived from W0
        profile = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
        masses = jnp.ones(100)
        key = jax.random.PRNGKey(42)

        @jax.jit
        def sample_and_sum(m, k):
            pos = profile.sample_positions(m, k)
            return jnp.sum(pos**2)

        result = sample_and_sum(masses, key)
        assert jnp.isfinite(result)


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
