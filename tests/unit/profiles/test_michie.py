"""Michie-King anisotropic model (Michie 1963 anisotropy + King 1966 cutoff).

Built test-first in stages. Stage 1: the anisotropic density integral michie_density(W, s),
s = r/r_a, must reduce to the King lowered-Maxwellian density at s=0 (the isotropic limit).
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.profiles.king import king_lowered_maxwellian_density

G = None  # filled per-test where needed


class TestMichieDensityIsotropicLimit:
    def test_s0_matches_king_shape(self):
        """At s=0 (r_a -> inf) the Michie-King density is proportional to King's."""
        from progenax.profiles.michie import michie_density

        Ws = [1.0, 3.0, 5.0, 7.0, 9.0]
        michie = jnp.array([float(michie_density(W, 0.0)) for W in Ws])
        king = king_lowered_maxwellian_density(jnp.array(Ws))
        ratio = michie / king
        assert jnp.all(jnp.abs(ratio - ratio[0]) < 1e-3 * ratio[0]), (
            f"michie_density(W,0)/king(W) not constant: {ratio}"
        )

    def test_s0_constant_is_sqrt_2pi(self):
        """The proportionality constant is sqrt(2 pi) (spherical reduction of the
        (u_r, u_t) integral vs King's normalised volume density)."""
        from progenax.profiles.michie import michie_density

        ratio = float(michie_density(5.0, 0.0)) / float(
            king_lowered_maxwellian_density(jnp.array(5.0))
        )
        assert abs(ratio - jnp.sqrt(2.0 * jnp.pi)) < 1e-3 * jnp.sqrt(2.0 * jnp.pi)

    def test_anisotropy_lowers_density(self):
        """s>0 (finite r_a) suppresses tangential velocities, lowering rho_hat(W,s)."""
        from progenax.profiles.michie import michie_density

        assert float(michie_density(7.0, 1.0)) < float(michie_density(7.0, 0.0))

    def test_zero_below_boundary(self):
        from progenax.profiles.michie import michie_density

        assert float(michie_density(0.0, 0.5)) == 0.0
        assert float(michie_density(-1.0, 0.5)) == 0.0


class TestSolveMichieProfileIsotropicLimit:
    @pytest.mark.parametrize("W0", [3.0, 5.0, 7.0])
    def test_large_r_a_matches_king(self, W0):
        """ra_hat -> inf reduces the anisotropic ODE to King's; psi(xi) must match."""
        from progenax.profiles.michie import solve_michie_profile
        from progenax.profiles.king import solve_king_profile

        # Match King's grid (xi_max, n_points) so psi can be compared pointwise.
        xi_m, psi_m, _ = solve_michie_profile(W0, ra_hat=1e5, xi_max=300.0, n_points=2000)
        xi_k, psi_k, _ = solve_king_profile(W0)
        assert jnp.allclose(xi_m, xi_k), "grids must align for comparison"
        assert jnp.allclose(psi_m, psi_k, atol=5e-3, rtol=5e-3), (
            f"max |dpsi| = {float(jnp.max(jnp.abs(psi_m - psi_k))):.2e}"
        )

    def test_boundary_and_monotonic(self):
        from progenax.profiles.michie import solve_michie_profile

        # ra_hat=10 is a physically valid (finitely truncated) anisotropic model.
        xi, psi, _ = solve_michie_profile(7.0, ra_hat=10.0)
        assert abs(float(psi[0]) - 7.0) < 1e-2, "psi(0) = W0"
        # psi monotonically decreases from the centre outward (until it hits 0)
        dpsi = jnp.diff(psi)
        assert jnp.all(dpsi <= 1e-6), "psi must be non-increasing outward"
        assert float(psi[-1]) == 0.0, "psi -> 0 at/after the tidal radius"


class TestMichieMaxAnisotropy:
    """Refuse an over-anisotropic model that has no finite tidal radius."""

    def test_too_anisotropic_raises(self):
        from progenax.profiles.michie import solve_michie_profile

        with pytest.raises(ValueError, match="does not truncate"):
            solve_michie_profile(7.0, ra_hat=2.0)  # 1/r^2 tail, no finite r_t

    def test_valid_anisotropy_truncates(self):
        from progenax.profiles.michie import solve_michie_profile

        xi, psi, _ = solve_michie_profile(7.0, ra_hat=10.0)
        assert float(psi[-1]) == 0.0, "a valid model truncates within xi_max"


class TestMichieProfile:
    def test_constructs_and_samples_within_rt(self):
        from progenax.profiles.michie import MichieProfile

        prof = MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=10.0)
        assert float(prof.r_t) > 0.0
        masses = jnp.ones(3000)
        pos = prof.sample_positions(masses, jax.random.PRNGKey(0))
        assert pos.shape == (3000, 3)
        r = jnp.linalg.norm(pos, axis=1)
        assert jnp.all(r <= float(prof.r_t) * 1.001), "all particles within r_t"

    def test_isotropic_limit_matches_king_rt(self):
        from progenax.profiles.michie import MichieProfile
        from progenax.profiles.king import KingProfile

        m = MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=1e5)
        k = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
        assert abs(float(m.r_t) - float(k.r_t)) < 0.05 * float(k.r_t), (
            f"Michie r_t={float(m.r_t):.2f} vs King r_t={float(k.r_t):.2f}"
        )

    def test_more_anisotropic_is_more_extended(self):
        """Stronger anisotropy (smaller r_a) gives a larger tidal radius."""
        from progenax.profiles.michie import MichieProfile

        r_t_strong = float(MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=6.0).r_t)
        r_t_weak = float(MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=20.0).r_t)
        assert r_t_strong > r_t_weak

    def test_too_anisotropic_raises(self):
        from progenax.profiles.michie import MichieProfile

        with pytest.raises(ValueError, match="does not truncate"):
            MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=2.0)
