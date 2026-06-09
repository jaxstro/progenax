"""
Physics validation for tidal truncation (Jacobi radius + apply_tidal_truncation).

The Jacobi radius is validated against INDEPENDENT oracles, not its own closed form:
  - the tidal force-balance condition  G m / r_J^2 = 3 Omega^2 r_J  (point-mass host),
  - the collinear L1 Lagrange point of the full restricted three-body problem
    (solved numerically), which the Hill/King formula r_J = R (m/3 M_g)^(1/3)
    approximates and converges to as m / M_g -> 0,
  - the flat-rotation-curve (isothermal) vs Keplerian tidal-tensor factor (3/2)^(1/3).
apply_tidal_truncation is checked for exact bound-mass accounting and differentiability
of the bound mass in the truncation radius (the straight-through surrogate gradient).

References:
    King (1962) AJ 67, 471; Binney & Tremaine (2008) Sec. 8.3.1 (Eq. 8.91);
    Spitzer (1987) "Dynamical Evolution of Globular Clusters".
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize

from jaxstro.units import STELLAR
from progenax.tidal import (
    apply_tidal_truncation,
    jacobi_radius,
    jacobi_radius_isothermal,
)

G = STELLAR.G  # pc^3 Msun^-1 Myr^-2


def _l1_distance_from_cluster(m, M_g, R, G):
    """Distance from the cluster centre to the inner Lagrange point L1, from the
    FULL collinear restricted-three-body force balance (independent of the Hill
    formula). Cluster m at x=0, galaxy M_g at x=R, rotating about the COM with
    Omega^2 = G (M_g + m) / R^3. Returns the root of the rotating-frame radial
    force in (0, R).
    """
    Omega2 = G * (M_g + m) / R**3
    x_com = M_g * R / (M_g + m)  # COM measured from the cluster

    def F(x):  # rotating-frame radial force on a test star at x in (0, R)
        return -G * m / x**2 + G * M_g / (R - x) ** 2 + Omega2 * (x - x_com)

    # bracket the root around the Hill estimate
    r_hill = R * (m / (3.0 * M_g)) ** (1.0 / 3.0)
    return scipy.optimize.brentq(F, 1e-6 * r_hill, 0.9 * R)


class TestJacobiTidalBalance:
    """r_J satisfies the defining tidal force balance G m / r_J^2 = 3 Omega^2 r_J."""

    @pytest.mark.parametrize("m,M_g,R", [(1e4, 1e11, 8000.0), (1e6, 5e10, 4000.0),
                                         (1e3, 1e12, 15000.0)])
    def test_self_gravity_equals_tidal_plus_centrifugal(self, m, M_g, R):
        r_J = float(jacobi_radius(m, M_g, R))
        Omega2 = G * M_g / R**3            # point-mass host: Omega^2 = G M_g / R^3
        self_grav = G * m / r_J**2
        tidal_centrifugal = 3.0 * Omega2 * r_J
        rel = abs(self_grav - tidal_centrifugal) / self_grav
        assert rel < 1e-10, f"tidal balance off by {rel:.2e}"


class TestJacobiL1Lagrange:
    """The Hill/King formula equals the full restricted-3-body L1 point in the
    m/M_g -> 0 limit (the independent physics oracle)."""

    def test_matches_l1_for_realistic_cluster(self):
        m, M_g, R = 1e4, 1e11, 8000.0           # m/M_g = 1e-7 (Galactic GC)
        r_J = float(jacobi_radius(m, M_g, R))
        l1 = _l1_distance_from_cluster(m, M_g, R, G)
        rel = abs(r_J - l1) / l1
        assert rel < 1e-2, f"r_J vs L1 differ by {rel:.2e} at m/M_g=1e-7"

    def test_hill_approximation_converges(self):
        """As m/M_g -> 0 the Hill formula approaches the exact L1 point."""
        M_g, R = 1e11, 8000.0
        rels = []
        for m in (1e8, 1e6, 1e4, 1e2):          # decreasing m/M_g
            r_J = float(jacobi_radius(m, M_g, R))
            l1 = _l1_distance_from_cluster(m, M_g, R, G)
            rels.append(abs(r_J - l1) / l1)
        # monotone improvement as the mass ratio shrinks
        assert all(rels[i + 1] < rels[i] for i in range(len(rels) - 1)), rels
        assert rels[-1] < 1e-3


class TestKeplerianVsIsothermal:
    """The flat-rotation-curve (isothermal) tidal radius is (3/2)^(1/3) larger than
    the Keplerian/point-mass one at matched orbital frequency Omega -- the tidal
    tensor carries 2 Omega^2 (flat) vs 3 Omega^2 (point mass)."""

    def test_isothermal_keplerian_ratio(self):
        m, R = 1e4, 8000.0
        V = 220.0 * 1.0227121651            # km/s -> pc/Myr (units consistent with G)
        Omega = V / R
        M_g_equiv = Omega**2 * R**3 / G     # point-mass host with the same Omega
        r_point = float(jacobi_radius(m, M_g_equiv, R))
        r_iso = float(jacobi_radius_isothermal(m, V, R, G))
        ratio = r_iso / r_point
        assert np.isclose(ratio, (3.0 / 2.0) ** (1.0 / 3.0), rtol=1e-6), ratio


class TestTruncationBoundMass:
    """apply_tidal_truncation: exact bound-mass accounting + differentiability in r_t."""

    def _plummer(self, n=4000, r_h=5.0, seed=0):
        from progenax.profiles import PlummerProfile
        prof = PlummerProfile(r_h=r_h)
        masses = jnp.ones(n)
        pos = prof.sample_positions(masses, jax.random.PRNGKey(seed))
        vel = jnp.zeros((n, 3))
        return pos, vel, masses

    def test_bound_mass_equals_mass_within_rt(self):
        pos, vel, m = self._plummer()
        r_t = 12.0
        _, _, mt, mask = apply_tidal_truncation(pos, vel, m, r_t)
        radii = jnp.linalg.norm(pos, axis=1)
        expected = float(jnp.sum(m[radii <= r_t]))
        assert float(jnp.sum(mt)) == pytest.approx(expected, abs=1e-9)
        assert bool(jnp.array_equal(mask, radii <= r_t))

    def test_bound_mass_matches_analytic_plummer_enclosed(self):
        """Truncated bound mass tracks the analytic Plummer enclosed-mass profile
        M(<r) = N r^3 / (r^2 + a^2)^{3/2}, a = r_h sqrt(2^{2/3}-1) (independent oracle)."""
        n, r_h = 8000, 5.0
        pos, vel, m = self._plummer(n=n, r_h=r_h)
        a = r_h * np.sqrt(2.0 ** (2.0 / 3.0) - 1.0)
        rts = np.array([5.0, 10.0, 20.0, 40.0])
        bound = np.array([float(jnp.sum(apply_tidal_truncation(pos, vel, m, rt)[2]))
                          for rt in rts])
        frac_analytic = rts**3 / (rts**2 + a**2) ** 1.5
        assert all(bound[i + 1] >= bound[i] for i in range(len(bound) - 1))  # monotone
        # finite-N: binomial std ~ sqrt(f(1-f)/n) <~ 0.006; allow 0.02
        assert np.max(np.abs(bound / n - frac_analytic)) < 0.02, (
            f"bound fractions {bound / n} vs analytic Plummer {frac_analytic}"
        )

    def test_bound_mass_differentiable_in_rt(self):
        """d(bound mass)/d(r_t) is finite and positive (straight-through surrogate)."""
        pos, vel, m = self._plummer()

        def bound_mass(r_t):
            return jnp.sum(apply_tidal_truncation(pos, vel, m, r_t)[2])

        g = jax.grad(bound_mass)(10.0)
        assert jnp.isfinite(g) and float(g) > 0.0, f"d(M_bound)/d(r_t) = {g}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
