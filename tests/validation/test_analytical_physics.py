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
        assert closure < 1e-5, (
            f"rescaled figure-eight did not close: residual {closure:.3e}"
        )


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
    @pytest.mark.parametrize(
        "builder,n", [("solar_system_inner_4", 5), ("solar_system_full", 9)]
    )
    def test_barycentric_and_finite(self, builder, n):
        import progenax.analytical as A

        ic = getattr(A, builder)(G=39.478)  # PLANETARY units (AU^3/Msun/yr^2)
        assert ic.masses.shape[0] == n
        com = jnp.sum(ic.positions * ic.masses[:, None], axis=0) / jnp.sum(ic.masses)
        pcom = jnp.sum(ic.masses[:, None] * ic.velocities, axis=0)
        assert jnp.linalg.norm(com) < 1e-12 and jnp.linalg.norm(pcom) < 1e-12
        assert jnp.all(jnp.isfinite(ic.positions)) and jnp.all(
            jnp.isfinite(ic.velocities)
        )

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
        assert jnp.allclose(
            ic.masses[1:], jnp.array([p["M"] for p in SOLAR_SYSTEM_PLANETS[:4]])
        )


# PLANETARY units (AU^3 / Msun / yr^2): G = 4 pi^2 so a 1-AU orbit about 1 Msun has T = 1 yr.
_G_PLANETARY = 39.478


def _eccentricity_vector(r_rel, v_rel, mu):
    """Laplace-Runge-Lenz eccentricity from a relative orbit (Murray & Dermott 1999, Eq.2.42).

    e_vec = ((v^2 - mu/r) r - (r.v) v) / mu, with mu = G(M1+M2) the standard gravitational
    parameter of the relative two-body motion. |e_vec| is the orbital eccentricity.
    """
    r = jnp.linalg.norm(r_rel)
    v2 = jnp.sum(v_rel**2)
    rdotv = jnp.sum(r_rel * v_rel)
    return ((v2 - mu / r) * r_rel - rdotv * v_rel) / mu


class TestEarthSunTwoBody:
    """earth_sun_2body: a BOUND, barycentric Sun-Earth circular two-body system.

    Physics anchors: Kepler's third law (T = 2 pi sqrt(a^3 / (G M_tot)) ~ 1 yr at a = 1 AU
    in PLANETARY units), bound total energy E < 0 equal to the vis-viva closed form
    -G M1 M2 / (2 a) (Murray & Dermott 1999, Ch.2), zero barycentric COM and momentum,
    and total mass = M_sun + M_earth (1 + 3.0035e-6 Msun, JPL/IAU mass ratio).
    """

    def test_two_bodies_total_mass(self):
        from progenax.analytical import earth_sun_2body

        ic = earth_sun_2body(G=_G_PLANETARY)
        assert ic.masses.shape[0] == 2
        # M_sun = 1.0 Msun, M_earth = 3.0035e-6 Msun (IAU 2009 / JPL mass ratio).
        assert jnp.isclose(jnp.sum(ic.masses), 1.0 + 3.0035e-6)

    def test_barycentric_com_and_momentum_zero(self):
        from progenax.analytical import earth_sun_2body

        ic = earth_sun_2body(G=_G_PLANETARY)
        com = jnp.sum(ic.positions * ic.masses[:, None], axis=0) / jnp.sum(ic.masses)
        pcom = jnp.sum(ic.masses[:, None] * ic.velocities, axis=0)
        assert jnp.linalg.norm(com) < 1e-12 and jnp.linalg.norm(pcom) < 1e-12

    def test_bound_energy_matches_vis_viva(self):
        from progenax.analytical import earth_sun_2body

        ic = earth_sun_2body(G=_G_PLANETARY)
        E = _energy(ic.positions, ic.velocities, ic.masses, _G_PLANETARY)
        assert E < 0.0, f"Earth-Sun must be bound (E < 0), got E = {E:.3e}"
        # Closed form E = -G M1 M2 / (2 a), a = 1 AU (Murray & Dermott 1999).
        E_vis_viva = -_G_PLANETARY * ic.masses[0] * ic.masses[1] / (2.0 * 1.0)
        assert jnp.isclose(E, E_vis_viva, rtol=1e-10)

    def test_kepler_third_law_period_one_year(self):
        from progenax.analytical import earth_sun_2body

        ic = earth_sun_2body(G=_G_PLANETARY)
        # Kepler III at a = 1 AU about (M_sun + M_earth): T ~ 1 yr (the tiny Earth mass
        # shortens it by ~1.5e-6). Assert against the exact two-body period for a = 1.
        T_kepler = 2.0 * jnp.pi * jnp.sqrt(1.0**3 / (_G_PLANETARY * jnp.sum(ic.masses)))
        assert jnp.isclose(ic.period, T_kepler, rtol=1e-12)
        assert jnp.isclose(ic.period, 1.0, rtol=1e-4)  # ~1 yr in PLANETARY units


class TestEarthSunEccentric:
    """earth_sun_eccentric: bound Sun-Earth orbit at Earth's TRUE eccentricity e = 0.0167.

    Physics anchor: the Laplace-Runge-Lenz eccentricity recovered from the realized
    relative orbit must equal the factory's specified e = 0.0167 (JPL J2000.0); the orbit
    is bound (E < 0) and barycentric (COM = 0). Recovering e from (r, v) is an independent
    check that the orbital state was built with the right shape, not just the right energy
    (E is e-independent, so it alone cannot catch a wrong eccentricity).
    """

    def test_recovers_specified_eccentricity(self):
        from progenax.analytical import earth_sun_eccentric

        ic = earth_sun_eccentric(G=_G_PLANETARY)
        mu = _G_PLANETARY * jnp.sum(ic.masses)  # G M_tot for the relative motion
        r_rel = ic.positions[1] - ic.positions[0]
        v_rel = ic.velocities[1] - ic.velocities[0]
        e = jnp.linalg.norm(_eccentricity_vector(r_rel, v_rel, mu))
        assert jnp.isclose(e, 0.0167, atol=1e-6), (
            f"recovered e = {e:.6f}, expected 0.0167"
        )
        # Vis-viva semi-major axis 1/a = 2/r - v^2/mu must recover a = 1 AU.
        a = 1.0 / (2.0 / jnp.linalg.norm(r_rel) - jnp.sum(v_rel**2) / mu)
        assert jnp.isclose(a, 1.0, rtol=1e-10)

    def test_bound_and_barycentric(self):
        from progenax.analytical import earth_sun_eccentric

        ic = earth_sun_eccentric(G=_G_PLANETARY)
        E = _energy(ic.positions, ic.velocities, ic.masses, _G_PLANETARY)
        assert E < 0.0, f"eccentric Earth-Sun must be bound (E < 0), got {E:.3e}"
        com = jnp.sum(ic.positions * ic.masses[:, None], axis=0) / jnp.sum(ic.masses)
        assert jnp.linalg.norm(com) < 1e-12


class TestSunEarthJupiterThreeBody:
    """sun_earth_jupiter_3body: a bound, barycentric hierarchical 3-body system.

    Physics anchors: exactly 3 bodies with total mass = M_sun + M_earth + M_jupiter
    (1 + 3.0035e-6 + 9.548e-4 Msun, the factory's hardcoded JPL masses), zero barycentric
    COM and momentum (the Sun is placed to enforce sum m_i q_i = 0), a bound configuration
    (E < 0), and finite state throughout.
    """

    def test_three_bodies_total_mass(self):
        from progenax.analytical import sun_earth_jupiter_3body

        ic = sun_earth_jupiter_3body(G=_G_PLANETARY)
        assert ic.masses.shape[0] == 3
        # M_sun + M_earth + M_jupiter (factory-hardcoded JPL/IAU masses).
        assert jnp.isclose(jnp.sum(ic.masses), 1.0 + 3.0035e-6 + 9.548e-4)

    def test_barycentric_com_and_momentum_zero(self):
        from progenax.analytical import sun_earth_jupiter_3body

        ic = sun_earth_jupiter_3body(G=_G_PLANETARY)
        com = jnp.sum(ic.positions * ic.masses[:, None], axis=0) / jnp.sum(ic.masses)
        pcom = jnp.sum(ic.masses[:, None] * ic.velocities, axis=0)
        assert jnp.linalg.norm(com) < 1e-12 and jnp.linalg.norm(pcom) < 1e-12

    def test_bound_and_finite(self):
        from progenax.analytical import sun_earth_jupiter_3body

        ic = sun_earth_jupiter_3body(G=_G_PLANETARY)
        E = _energy(ic.positions, ic.velocities, ic.masses, _G_PLANETARY)
        assert E < 0.0, f"Sun-Earth-Jupiter must be bound (E < 0), got {E:.3e}"
        assert jnp.all(jnp.isfinite(ic.positions)) and jnp.all(
            jnp.isfinite(ic.velocities)
        )


class TestHarmonicOscillator:
    """harmonic_oscillator: a single particle whose IC encodes simple harmonic motion.

    Physics anchors (x(t) = A cos(wt + phi)): initial state x0 = A cos(phi),
    v0 = -A w sin(phi); period T = 2 pi / w (amplitude-independent); and total energy
    E = 1/2 m v^2 + 1/2 m w^2 x^2 = 1/2 m w^2 A^2, the conserved SHO energy.
    """

    def test_initial_state_matches_sho(self):
        from progenax.analytical import harmonic_oscillator

        amp, omega, phase, mass = 2.0, 3.0, 0.4, 1.5
        ic = harmonic_oscillator(amplitude=amp, omega=omega, phase=phase, mass=mass)
        assert ic.masses.shape[0] == 1
        assert jnp.isclose(ic.positions[0, 0], amp * jnp.cos(phase))
        assert jnp.isclose(ic.velocities[0, 0], -amp * omega * jnp.sin(phase))
        assert jnp.isclose(ic.period, 2.0 * jnp.pi / omega)

    def test_energy_is_sho_constant(self):
        from progenax.analytical import harmonic_oscillator

        amp, omega, phase, mass = 2.0, 3.0, 0.4, 1.5
        ic = harmonic_oscillator(amplitude=amp, omega=omega, phase=phase, mass=mass)
        x0, v0 = ic.positions[0, 0], ic.velocities[0, 0]
        # E = 1/2 m v^2 + 1/2 m w^2 x^2 must equal the closed-form 1/2 m w^2 A^2.
        E_state = 0.5 * mass * v0**2 + 0.5 * mass * omega**2 * x0**2
        E_closed = 0.5 * mass * omega**2 * amp**2
        assert jnp.isclose(E_state, E_closed)
        assert jnp.isclose(ic.energy, E_closed)


class TestHarmonicSolution:
    """harmonic_solution: the closed-form SHO trajectory x(t) = A cos(wt + phi).

    The defining check (the "analytical solution satisfies its ODE" anchor): x(t) must
    satisfy xddot = -w^2 x. We compare a central-difference second derivative to -w^2 x at
    several t. A second test pins x(0), v(0) to harmonic_oscillator's IC (same A, w, phi).
    """

    def test_satisfies_sho_ode(self):
        from progenax.analytical import harmonic_solution

        amp, omega, phase = 2.0, 3.0, 0.4
        h = 1e-4
        for t in (0.0, 0.37, 1.1, 2.5, 4.2):
            pos_p, _ = harmonic_solution(t + h, amp, omega, phase)
            pos_0, _ = harmonic_solution(t, amp, omega, phase)
            pos_m, _ = harmonic_solution(t - h, amp, omega, phase)
            xddot = (pos_p[0, 0] - 2.0 * pos_0[0, 0] + pos_m[0, 0]) / h**2
            rhs = -(omega**2) * pos_0[0, 0]
            assert jnp.isclose(xddot, rhs, rtol=1e-5, atol=1e-6), (
                f"ODE residual at t={t}: xddot={xddot:.6f} vs -w^2 x={rhs:.6f}"
            )

    def test_matches_oscillator_ic_at_t0(self):
        from progenax.analytical import harmonic_oscillator, harmonic_solution

        amp, omega, phase = 2.0, 3.0, 0.4
        ic = harmonic_oscillator(amplitude=amp, omega=omega, phase=phase)
        pos0, vel0 = harmonic_solution(0.0, amp, omega, phase)
        assert jnp.allclose(pos0, ic.positions) and jnp.allclose(vel0, ic.velocities)


class TestFigureEightPeriod:
    """figure_eight_period: the Chenciner-Montgomery-Simo figure-eight orbital period.

    Physics anchor: in the module's dimensionless normalization (G = 1, m = 1, scale = 1)
    the period is the literature constant T0 = 6.3259... (Chenciner & Montgomery 2000,
    Ann. Math. 152, 881; Simo numerical coefficients). Cross-check: the value must equal
    the period stamped into three_body_figure_eight (which consumes this helper) AND obey
    the scaling law T = T0 sqrt(scale^3 / (G m)).
    """

    def test_default_is_chenciner_montgomery_constant(self):
        from progenax.analytical import figure_eight_period

        # Chenciner & Montgomery (2000) figure-eight period in dimensionless units.
        assert jnp.isclose(figure_eight_period(), 6.32591398, rtol=1e-8)

    def test_consistent_with_three_body_factory(self):
        from progenax.analytical import figure_eight_period, three_body_figure_eight

        # Same helper drives the factory's `.period`; they must agree at matched params.
        for scale, G, mass in ((1.0, 1.0, 1.0), (1.5, 2.0, 1.0), (2.0, 1.0, 3.0)):
            ic = three_body_figure_eight(mass=mass, scale=scale, G=G)
            assert jnp.isclose(
                ic.period, figure_eight_period(scale, G, mass), rtol=1e-12
            )

    def test_scaling_law(self):
        from progenax.analytical import figure_eight_period

        # T = T0 sqrt(scale^3 / (G m)): a dimensional consistency anchor (period grows as
        # the dynamical time sqrt(L^3/GM) of the rescaled orbit).
        for scale, G, mass in ((1.5, 2.0, 1.0), (0.5, 0.5, 2.0)):
            expected = 6.32591398 * jnp.sqrt(scale**3 / (G * mass))
            assert jnp.isclose(
                figure_eight_period(scale, G, mass), expected, rtol=1e-10
            )
