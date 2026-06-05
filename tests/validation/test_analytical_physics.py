"""Self-validating physics for the analytical oracles (Batch 5).

The figure-eight is verified by **integration** (the Chenciner & Montgomery 2000 paper
is not held): integrate one period and check orbit closure + zero total angular
momentum + energy conservation. A wrong initial condition — e.g. a 3-fold *spatial*
rotation instead of the canonical Chenciner–Montgomery–Simó collinear configuration —
fails closure and carries L ≠ 0, which these tests catch.
"""

import jax
import jax.numpy as jnp
import pytest


def _accel(pos, masses, G):
    diff = pos[None, :, :] - pos[:, None, :]  # r_j - r_i
    r = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    r = jnp.where(jnp.eye(pos.shape[0], dtype=bool), jnp.inf, r)
    return G * jnp.sum(masses[None, :, None] * diff / r[:, :, None] ** 3, axis=1)


def _verlet(pos0, vel0, masses, G, T, n):
    """Velocity-Verlet integration of an N-body system for time T in n steps."""
    dt = T / n

    def step(c, _):
        p, v = c
        v = v + 0.5 * dt * _accel(p, masses, G)
        p = p + dt * v
        v = v + 0.5 * dt * _accel(p, masses, G)
        return (p, v), None

    (p, v), _ = jax.lax.scan(step, (pos0, vel0), None, length=n)
    return p, v


def _energy(pos, vel, masses, G):
    KE = 0.5 * jnp.sum(masses * jnp.sum(vel**2, axis=1))
    diff = pos[None, :, :] - pos[:, None, :]
    r = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    r = jnp.where(jnp.eye(pos.shape[0], dtype=bool), jnp.inf, r)
    PE = -0.5 * G * jnp.sum(masses[:, None] * masses[None, :] / r)
    return KE + PE


def _angular_momentum(pos, vel, masses):
    return jnp.sum(masses[:, None] * jnp.cross(pos, vel), axis=0)


class TestFigureEightSelfValidating:
    def test_closes_after_one_period(self):
        """The defining property: integrate one period, return to the start."""
        from progenax.analytical import three_body_figure_eight

        ic = three_body_figure_eight(mass=1.0, scale=1.0, G=1.0)
        p, _ = _verlet(ic.positions, ic.velocities, ic.masses, 1.0, ic.period, 400000)
        closure = jnp.max(jnp.linalg.norm(p - ic.positions, axis=1))
        assert closure < 1e-6, f"figure-eight did not close: residual {closure:.3e}"

    def test_zero_total_angular_momentum(self):
        """The Chenciner–Montgomery figure-eight has L = 0."""
        from progenax.analytical import three_body_figure_eight

        ic = three_body_figure_eight(G=1.0)
        L = jnp.linalg.norm(_angular_momentum(ic.positions, ic.velocities, ic.masses))
        assert L < 1e-10, f"|L| = {L:.3e} (figure-eight requires L = 0)"

    def test_energy_conserved_over_period(self):
        from progenax.analytical import three_body_figure_eight

        ic = three_body_figure_eight(G=1.0)
        E0 = _energy(ic.positions, ic.velocities, ic.masses, 1.0)
        p, v = _verlet(ic.positions, ic.velocities, ic.masses, 1.0, ic.period, 400000)
        E1 = _energy(p, v, ic.masses, 1.0)
        assert jnp.abs((E1 - E0) / E0) < 1e-4, f"ΔE/E = {(E1 - E0) / E0:.3e}"

    def test_com_stationary_at_origin(self):
        from progenax.analytical import three_body_figure_eight

        ic = three_body_figure_eight(G=1.0)
        com = jnp.sum(ic.positions * ic.masses[:, None], axis=0) / jnp.sum(ic.masses)
        pcom = jnp.sum(ic.masses[:, None] * ic.velocities, axis=0)
        assert jnp.allclose(com, 0.0, atol=1e-8) and jnp.allclose(pcom, 0.0, atol=1e-8)

    def test_closes_when_rescaled(self):
        """The rescaling (lengths ×scale, velocities ×√(G·m/scale), period
        T₀√(scale³/(G·m))) must keep the orbit closed for non-default (scale, G, m)."""
        from progenax.analytical import three_body_figure_eight

        ic = three_body_figure_eight(mass=1.0, scale=1.5, G=2.0)
        p, _ = _verlet(ic.positions, ic.velocities, ic.masses, 2.0, ic.period, 400000)
        closure = jnp.max(jnp.linalg.norm(p - ic.positions, axis=1))
        assert closure < 1e-5, f"rescaled figure-eight did not close: residual {closure:.3e}"


class TestTwoBodyConservation:
    def test_energy_formula(self):
        from progenax.analytical import two_body_kepler

        G, M1, M2, a = 1.0, 1.0, 0.5, 2.0
        ic = two_body_kepler(M1=M1, M2=M2, a=a, e=0.3, G=G)
        assert jnp.isclose(ic.energy, -G * M1 * M2 / (2 * a))  # E independent of e

    def test_kepler_third_law_period(self):
        from progenax.analytical import two_body_kepler

        G, M1, M2, a = 1.0, 1.0, 0.5, 2.0
        ic = two_body_kepler(M1=M1, M2=M2, a=a, e=0.0, G=G)
        assert jnp.isclose(ic.period, 2 * jnp.pi * jnp.sqrt(a**3 / (G * (M1 + M2))))

    def test_eccentric_orbit_closes_and_conserves(self):
        """Integrate an eccentric Kepler orbit one period: closes + conserves E, L."""
        from progenax.analytical import two_body_kepler

        G = 1.0
        ic = two_body_kepler(M1=1.0, M2=0.1, a=1.0, e=0.5, G=G)
        E0 = _energy(ic.positions, ic.velocities, ic.masses, G)
        L0 = _angular_momentum(ic.positions, ic.velocities, ic.masses)
        p, v = _verlet(ic.positions, ic.velocities, ic.masses, G, ic.period, 600000)
        E1 = _energy(p, v, ic.masses, G)
        L1 = _angular_momentum(p, v, ic.masses)
        closure = jnp.max(jnp.linalg.norm(p - ic.positions, axis=1))
        assert closure < 1e-4, f"ellipse did not close: {closure:.3e}"
        assert jnp.abs((E1 - E0) / E0) < 1e-5
        assert jnp.linalg.norm(L1 - L0) < 1e-6


class TestSolarSystemPhysics:
    @pytest.mark.parametrize("builder,n", [("solar_system_inner_4", 5), ("solar_system_full", 9)])
    def test_barycentric_and_finite(self, builder, n):
        import progenax.analytical as A

        ic = getattr(A, builder)(G=39.478)  # PLANETARY units (AU^3/Msun/yr^2)
        assert ic.masses.shape[0] == n
        com = jnp.sum(ic.positions * ic.masses[:, None], axis=0) / jnp.sum(ic.masses)
        pcom = jnp.sum(ic.masses[:, None] * ic.velocities, axis=0)
        assert jnp.linalg.norm(com) < 1e-12 and jnp.linalg.norm(pcom) < 1e-12
        assert jnp.all(jnp.isfinite(ic.positions)) and jnp.all(jnp.isfinite(ic.velocities))

    def test_planet_mass_ratios_match_iau(self):
        """Planet/Sun mass ratios match the IAU 2009 / JPL best estimates."""
        from progenax.analytical import get_planet

        # M_sun / M_planet to ~5 significant figures (IAU 2009; cf. Prša 2016 nominal GM).
        assert jnp.isclose(1.0 / get_planet("Earth")["M"], 332946.0, rtol=1e-4)
        assert jnp.isclose(1.0 / get_planet("Jupiter")["M"], 1047.35, rtol=1e-4)
        assert jnp.isclose(1.0 / get_planet("Saturn")["M"], 3497.9, rtol=1e-3)

    def test_inner4_shares_table(self):
        """solar_system_inner_4 is built from the first four SOLAR_SYSTEM_PLANETS (DRY)."""
        from progenax.analytical import SOLAR_SYSTEM_PLANETS, solar_system_inner_4

        ic = solar_system_inner_4(G=39.478)
        # planet masses (rows 1..4) equal the table's first four, in order
        assert jnp.allclose(ic.masses[1:], jnp.array([p["M"] for p in SOLAR_SYSTEM_PLANETS[:4]]))
