"""Tests for tidal physics utilities (progenax.tidal)."""

import jax
import jax.numpy as jnp

from progenax.tidal import (
    _truncation_weight,
    apply_tidal_truncation,
    fill_factor_to_r_h,
    jacobi_radius,
    jacobi_radius_isothermal,
)


class TestJacobiRadius:
    """r_J = R * (M_cl / 3 M_gal)^(1/3)  (King 1962; BT2008 Eq 8.91)."""

    def test_jacobi_radius_formula(self):
        M_cluster, M_galaxy, R = 1e4, 1e11, 8000.0
        r_J = jacobi_radius(M_cluster, M_galaxy, R)
        # defining relation, independent of internal arithmetic ordering
        assert jnp.isclose(r_J**3, R**3 * M_cluster / (3.0 * M_galaxy), rtol=1e-12)

    def test_scales_with_cluster_mass(self):
        r_small = jacobi_radius(1e3, 1e11, 8000.0)
        r_large = jacobi_radius(1e5, 1e11, 8000.0)
        assert jnp.isclose(r_large / r_small, (1e5 / 1e3) ** (1.0 / 3.0), rtol=1e-3)

    def test_scales_with_distance(self):
        assert jacobi_radius(1e4, 1e11, 8000.0) > jacobi_radius(1e4, 1e11, 4000.0)


class TestJacobiRadiusIsothermal:
    """r_J = (G M_cl R^2 / 2 V^2)^(1/3); V_circ must share G's units (pc/Myr)."""

    def test_satisfies_defining_relation(self):
        M, V, R, G = 1e4, 225.0, 8000.0, 0.00450  # V in pc/Myr (consistent with G)
        r_J = jacobi_radius_isothermal(M, V, R, G)
        # Omega = V/R  =>  r_J^3 = G M / (2 Omega^2) = G M R^2 / (2 V^2)
        assert jnp.isclose(r_J**3, G * M * R**2 / (2.0 * V**2), rtol=1e-12)

    def test_mass_cube_root_scaling(self):
        G = 0.00450
        r1 = jacobi_radius_isothermal(1e4, 225.0, 8000.0, G)
        r8 = jacobi_radius_isothermal(8e4, 225.0, 8000.0, G)
        assert jnp.isclose(r8 / r1, 2.0, rtol=1e-3)

    def test_finite_positive_for_consistent_units(self):
        # 220 km/s expressed in pc/Myr (the unit the function requires)
        r_J = jacobi_radius_isothermal(1e4, 220.0 * 1.0227121651, 8000.0, 0.00450)
        assert jnp.isfinite(r_J) and r_J > 0.0


class TestTidalTruncation:
    """Hybrid: exact hard cut (zero-mass), shape-preserving, differentiable in r_t."""

    def _system(self, N=200, seed=42):
        pos = jax.random.normal(jax.random.PRNGKey(seed), (N, 3)) * 5.0
        vel = jax.random.normal(jax.random.PRNGKey(seed + 1), (N, 3))
        return pos, vel, jnp.ones(N)

    def test_shape_preserving(self):
        pos, vel, m = self._system()
        p, v, mt, mask = apply_tidal_truncation(pos, vel, m, 3.0)
        assert p.shape == pos.shape and v.shape == vel.shape
        assert mt.shape == m.shape and mask.shape == (m.shape[0],)
        # positions/velocities returned unchanged (truncated left in place)
        assert jnp.allclose(p, pos) and jnp.allclose(v, vel)

    def test_forward_is_exact_hard_cut(self):
        pos, vel, m = self._system()
        r_t = 3.0
        _, _, mt, mask = apply_tidal_truncation(pos, vel, m, r_t)
        radii = jnp.linalg.norm(pos, axis=1)
        assert jnp.all(mt[radii > r_t] == 0.0)  # outside -> massless
        assert jnp.allclose(mt[radii <= r_t], m[radii <= r_t])  # inside -> unchanged
        assert jnp.array_equal(mask, radii <= r_t)

    def test_zero_mass_ghosts_are_inert_in_potential_energy(self):
        from progenax.dynamics.virial import compute_potential_energy

        pos, vel, m = self._system(N=60)
        _, _, mt, _ = apply_tidal_truncation(pos, vel, m, 3.0)
        assert jnp.isfinite(compute_potential_energy(pos, mt, 0.00450, 0.0))

    def test_jit_compatible(self):
        pos, vel, m = self._system()
        _, _, mt_j, _ = jax.jit(apply_tidal_truncation)(pos, vel, m, 3.0)
        _, _, mt, _ = apply_tidal_truncation(pos, vel, m, 3.0)
        assert jnp.allclose(mt_j, mt)

    def test_vmap_over_r_t_monotone(self):
        pos, vel, m = self._system(N=40)
        f = lambda rt: apply_tidal_truncation(pos, vel, m, rt)[2].sum()
        retained = jax.vmap(f)(jnp.array([2.0, 3.0, 4.0]))
        assert retained.shape == (3,)
        assert retained[0] <= retained[1] <= retained[2]

    def test_retained_mass_grad_wrt_r_t_finite_positive(self):
        """The capability a hard cut lacks: nonzero gradient w.r.t. r_t."""
        pos, vel, m = self._system(N=200)
        g = jax.grad(lambda r_t: apply_tidal_truncation(pos, vel, m, r_t)[2].sum())(3.0)
        assert jnp.isfinite(g)
        assert g > 0.0  # widening the truncation radius retains more mass

    def test_surrogate_matches_logistic_derivative(self):
        """Self-consistency: d(weight)/d(r_t) == sigma(1-sigma)/w (the surrogate)."""
        r, w = 2.0, 0.1
        g = jax.grad(
            lambda r_t: _truncation_weight(jnp.asarray(r_t - r), jnp.asarray(w))
        )(2.05)
        s = jax.nn.sigmoid((2.05 - r) / w)
        assert jnp.isclose(g, s * (1.0 - s) / w, rtol=1e-6)


class TestFillFactor:
    """r_h = fill_factor * r_J (Baumgardt & Makino 2003)."""

    def test_formula(self):
        assert jnp.isclose(fill_factor_to_r_h(0.2, 10.0), 2.0, atol=1e-12)

    def test_monotone_and_bounded(self):
        r_J = 10.0
        assert fill_factor_to_r_h(0.1, r_J) < fill_factor_to_r_h(0.5, r_J)
        assert 0 < fill_factor_to_r_h(0.1, r_J) < r_J
