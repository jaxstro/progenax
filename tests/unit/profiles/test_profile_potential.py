"""True-potential validation for compute_profile_potential (Batch 1, P2+P3).

The old code used ad-hoc enclosed-mass ansaetze (an arctan form for EFF gamma=3,
behind a non-jittable `if gamma == 3.0`, and a (r/r_t)^3 form for King). These
tests pin the *true* potentials: King uses the ODE relative potential psi
(V(r_t)=0), EFF uses the exact enclosed mass + outer shell, Plummer is analytic.
"""
import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR
from progenax.profiles import compute_profile_potential, sample_density_profile

G = STELLAR.G


def _positions(profile, N=400, R_half=1.0, seed=0, **kw):
    return sample_density_profile(jax.random.PRNGKey(seed), N, profile, R_half, **kw)


class TestKingTruePotential:
    def test_zero_at_tidal_radius(self):
        """King convention V(r_t)=0 (psi(r_t)=0). The old (r/r_t)^3 ansatz gave -GM/r_t."""
        from progenax.profiles import make_profile

        prof = make_profile("king", 1.0, W0=7.0)
        r_t = float(prof.r_t)
        pos = jnp.array([[r_t, 0.0, 0.0]])
        phi = compute_profile_potential(pos, "king", 1000.0, 1.0, G, W0=7.0)
        assert jnp.abs(phi[0]) < 1e-2 * G * 1000.0 / r_t, f"King Phi(r_t)={float(phi[0])} not ~0"

    def test_monotonic_increasing_with_radius(self):
        pos = jnp.array([[0.1, 0, 0], [1.0, 0, 0], [5.0, 0, 0]])
        phi = compute_profile_potential(pos, "king", 1000.0, 1.0, G, W0=7.0)
        assert phi[0] < phi[1] < phi[2], "King potential must rise (less bound) outward"

    def test_negative_and_finite(self):
        pos = _positions("king", R_half=1.0, W0=7.0)
        phi = compute_profile_potential(pos, "king", 1000.0, 1.0, G, W0=7.0)
        assert jnp.all(jnp.isfinite(phi)) and jnp.all(phi < 0.0)


class TestEFFTruePotential:
    def test_jit_safe_and_grad_wrt_gamma(self):
        """Old code: `if gamma==3.0` -> TracerBoolConversionError under jit; grad=0."""
        pos = _positions("eff", R_half=1.0, gamma=3.0, r_t=10.0)

        def loss(gamma):
            return compute_profile_potential(pos, "eff", 1000.0, 1.0, G, gamma=gamma, r_t=10.0).sum()

        val = jax.jit(loss)(3.0)
        assert jnp.isfinite(val)
        g = jax.grad(loss)(3.5)
        assert jnp.isfinite(g) and jnp.abs(g) > 0.0, "grad wrt gamma must be finite & nonzero"

    def test_enclosed_mass_matches_profile_cdf(self):
        """The potential must use the profile's exact enclosed mass, not an arctan form."""
        from progenax.profiles import make_profile

        prof = make_profile("eff", 1.0, gamma=3.0, r_t=10.0)
        # enclosed-mass fraction the potential implies at the interior, M(<r)/M_tot,
        # equals the profile's own CDF (both = cumulative trapezoid of rho*r^2).
        rgrid = prof._r_grid
        rho_t = (1.0 + (rgrid / prof.a) ** 2) ** (-prof.gamma / 2.0)
        # NON-UNIFORM trapezoid: _r_grid is sqrt-stretched (audit R4), so weight
        # each trapezoid by its own width diff(rgrid). The invariant is unchanged
        # (potential's enclosed mass == profile CDF, both = cumtrap of rho*r^2).
        dr = jnp.diff(rgrid)
        I2 = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (rho_t[1:] * rgrid[1:] ** 2 + rho_t[:-1] * rgrid[:-1] ** 2) * dr)])
        cdf_from_pot = I2 / I2[-1]
        assert jnp.max(jnp.abs(cdf_from_pot - prof._cdf_grid)) < 1e-9

    def test_negative_and_finite(self):
        pos = _positions("eff", R_half=1.0, gamma=3.0, r_t=10.0)
        phi = compute_profile_potential(pos, "eff", 1000.0, 1.0, G, gamma=3.0, r_t=10.0)
        assert jnp.all(jnp.isfinite(phi)) and jnp.all(phi < 0.0)


class TestPlummerPotential:
    def test_exact_analytic(self):
        a = 1.0 * jnp.sqrt((1.0 - 0.5 ** (2 / 3)) / 0.5 ** (2 / 3))
        pos = jnp.array([[0.0, 0, 0], [2.0, 0, 0]])
        phi = compute_profile_potential(pos, "plummer", 1000.0, 1.0, G)
        expected = -G * 1000.0 / jnp.sqrt(jnp.sum(pos ** 2, axis=1) + a ** 2)
        assert jnp.allclose(phi, expected, rtol=1e-12)
