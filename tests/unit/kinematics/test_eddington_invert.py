"""Generic Eddington inversion: extraction regression + Plummer analytic oracle."""

import jax.numpy as jnp
import numpy as np
import pytest

# Exact-value pins from _eff_eddington_table(1.0, 4.0, 12.0, r_a)
# (full repr precision). Re-capture (never loosen) on JAX upgrades or
# DELIBERATE op-order changes only.
# Re-captured 2026-06-10 (consolidation Task 8): the table's cumulative
# trapezoid moved onto progenax.numerics.cumulative_trapezoid (scalar dx
# OUTSIDE the cumsum; the old inline used the non-uniform jnp.diff(r)
# INSIDE), shifting every pinned value by ~1 ulp (max rel. diff ~9e-16).
PINS = {
    "iso": (
        5.5042467299701805,
        0.702448236726355,
        [
            0.0001243105130388584,
            0.0007435780468229333,
            0.0043761938193880206,
            0.01734566382207258,
            0.05622049408621226,
        ],
    ),
    "om": (
        5.5042467299701805,
        0.702448236726355,
        [
            0.00227502604753575,
            0.001516930253864098,
            0.0050208984460832105,
            0.01595233272056175,
            0.046931165805733546,
        ],
    ),
}


class TestExtractionRegression:
    @pytest.mark.parametrize("tag,ra", [("iso", None), ("om", 2.0)])
    def test_eff_table_bit_identical_after_refactor(self, tag, ra):
        """The refactored _eff_eddington_table (now calling eddington_invert)
        reproduces the pre-refactor values EXACTLY (same grids, same ops)."""
        from progenax.kinematics.eff_df import _eff_eddington_table

        r, Psi, E, f, mu = _eff_eddington_table(1.0, 4.0, 12.0, ra)
        Psi0, mu_pin, f_pins = PINS[tag]
        assert float(Psi[0]) == Psi0 and float(mu) == mu_pin
        for idx, pin in zip((1, 250, 500, 750, 998), f_pins):
            assert float(f[idx]) == pin, f"f[{idx}] drifted"


def _plummer_grids(a=1.0, rt=100.0, n=20000):
    """Analytic (unnormalized, G=1) Plummer inputs for the inverter.

    rho = (1 + x^2)^{-5/2},  M(<r) = (4 pi a^3 / 3) x^3 (1 + x^2)^{-3/2},
    M_tot = (4 pi a^3 / 3),  Phi = -M_tot / sqrt(r^2 + a^2),
    dPsi/dr = -M(<r)/r^2 (zero-point independent).
    """
    r = jnp.linspace(1e-5, rt, n)
    x = r / a
    rho = (1.0 + x**2) ** (-2.5)
    drho = -5.0 * (r / a**2) * (1.0 + x**2) ** (-3.5)
    Mr = (4.0 * jnp.pi * a**3 / 3.0) * x**3 / (1.0 + x**2) ** 1.5
    M_tot = 4.0 * jnp.pi * a**3 / 3.0
    Phi = -M_tot / jnp.sqrt(r**2 + a**2)
    dPsi_dr = -Mr / r**2
    return r, rho, drho, Phi, dPsi_dr, M_tot


class TestPlummerAnalyticOracle:
    def test_isotropic_plummer_f_propto_E_3p5(self):
        """Feed the inverter the ANALYTIC Plummer (rho, Psi, dPsi/dr) with the
        UNTRUNCATED zero point Psi = -Phi (Phi(inf) = 0), where the ergodic DF
        is exactly f(E) propto E^{7/2} (BT2008 Eq. 4.83), and check
        f(E)/f(E_ref) == (E/E_ref)^{3.5} to rtol 1e-3 on interior energies
        E in [0.1, 0.8] Psi0. Bypasses all of our own potential numerics.

        Zero-point physics: with the truncated zero point Psi = Phi(r_t) - Phi
        the energies shift by c = M/sqrt(r_t^2 + a^2) and the E^{7/2} law picks
        up an O(3.5 c/E) ~ 30% deviation at E = 0.1 Psi0 even for r_t = 100 a;
        that case is covered EXACTLY by the truncated closed-form oracle below.
        """
        from progenax.kinematics.eddington import eddington_invert

        r, rho, drho, Phi, dPsi_dr, _ = _plummer_grids()
        Psi = -Phi  # untruncated zero point: Psi(inf) = 0
        E_grid, f_grid = eddington_invert(r, rho, drho, Psi, dPsi_dr)

        Psi0 = float(Psi[0])
        sel = (np.asarray(E_grid) > 0.1 * Psi0) & (np.asarray(E_grid) < 0.8 * Psi0)
        E = np.asarray(E_grid)[sel]
        f = np.asarray(f_grid)[sel]
        i_ref = len(E) // 2
        np.testing.assert_allclose(
            f / f[i_ref],
            (E / E[i_ref]) ** 3.5,
            rtol=1e-3,
            err_msg="inverter does not reproduce the Plummer E^{7/2} law",
        )

    def test_truncated_plummer_exact_closed_form(self):
        """Even stronger: with the truncated zero point Psi = Phi(r_t) - Phi,
        rho(Psi) = k (Psi + c)^5 exactly (k = (a/M)^5, c = M/sqrt(r_t^2+a^2)),
        and the Eddington integral has a closed form INCLUDING the truncation
        boundary term (b = E + c):

            f(E) = [20k (2 b^3 sqrt(E) - 2 b^2 E^{3/2} + (6/5) b E^{5/2}
                         - (2/7) E^{7/2}) + 5 k c^4 / sqrt(E)] / (sqrt(8) pi^2).

        Tests amplitude AND shape (not just the power law) to rtol 1e-4."""
        from progenax.kinematics.eddington import eddington_invert

        a, rt = 1.0, 100.0
        r, rho, drho, Phi, dPsi_dr, M_tot = _plummer_grids(a, rt)
        Psi = Phi[-1] - Phi  # truncated zero point: Psi(r_t) = 0
        E_grid, f_grid = eddington_invert(r, rho, drho, Psi, dPsi_dr)

        c = float(M_tot) / np.sqrt(rt**2 + a**2)
        k = (a / float(M_tot)) ** 5
        E = np.asarray(E_grid)
        b = E + c
        integral = (
            2.0 * b**3 * np.sqrt(E)
            - 2.0 * b**2 * E**1.5
            + 1.2 * b * E**2.5
            - (2.0 / 7.0) * E**3.5
        )
        f_exact = (20.0 * k * integral + 5.0 * k * c**4 / np.sqrt(E)) / (
            np.sqrt(8.0) * np.pi**2
        )

        Psi0 = float(Psi[0])
        sel = (E > 0.1 * Psi0) & (E < 0.8 * Psi0)
        np.testing.assert_allclose(
            np.asarray(f_grid)[sel],
            f_exact[sel],
            rtol=1e-4,
            err_msg="inverter does not match the exact truncated Plummer DF",
        )

    def test_om_reduces_to_iso_at_infinite_ra(self):
        """r_a = inf augmentation weight is exactly 1 -> identical tables."""
        from progenax.kinematics.eddington import eddington_invert

        r = jnp.linspace(1e-5, 30.0, 4000)
        rho = (1.0 + r**2) ** (-2.5)
        drho = -5.0 * r * (1.0 + r**2) ** (-3.5)
        Mr = (4.0 * jnp.pi / 3.0) * r**3 / (1.0 + r**2) ** 1.5
        Phi = -(4.0 * jnp.pi / 3.0) / jnp.sqrt(r**2 + 1.0)
        Psi = Phi[-1] - Phi
        dPsi = -Mr / r**2
        E1, f1 = eddington_invert(r, rho, drho, Psi, dPsi, r_a=None)
        E2, f2 = eddington_invert(r, rho, drho, Psi, dPsi, r_a=jnp.inf)
        np.testing.assert_array_equal(np.asarray(f1), np.asarray(f2))


class TestSpeedSamplerScaleRelativeThresholds:
    """F3 (code-review batch 2026-06-10): sample_speed_from_f_table thresholds.

    The Psi floor and the zero-speed cutoff must be RELATIVE to the table's
    energy scale E_grid[-1] (~0.999 Psi0), never absolute: Engine B tables are
    in physical units where Psi0 ~ 1/length (PlummerProfile(r_h=1e4 pc) ->
    Psi0 = 1.2e-4), so the old absolute cutoff Psi_r > 1e-6 silently zeroed
    speeds. Sampling must be exactly equivariant under a uniform energy
    rescale: E' = lam*E, Psi' = lam*Psi  =>  s' = sqrt(lam) s.
    """

    def _table(self, lam):
        Psi0 = 1.0 * lam
        E = jnp.linspace(1e-4 * Psi0, 0.999 * Psi0, 400)
        f = (E / Psi0) ** 3.5  # Plummer-shaped ergodic DF (shape-only)
        return E, f

    @pytest.mark.parametrize("lam", [1e-8, 1e-4, 1e8])
    def test_speeds_scale_exactly_with_sqrt_energy(self, lam):
        import jax
        from progenax.kinematics.eddington import sample_speed_from_f_table

        keys = jax.random.split(jax.random.PRNGKey(0), 64)
        E1, f1 = self._table(1.0)
        El, fl = self._table(lam)
        Psi1 = jnp.linspace(0.05, 0.95, 64)  # interior binding potentials
        s1 = jax.vmap(
            lambda k, p: sample_speed_from_f_table(k, p, E1, f1))(keys, Psi1)
        sl = jax.vmap(
            lambda k, p: sample_speed_from_f_table(k, p, El, fl))(keys, lam * Psi1)
        assert bool(jnp.all(s1 > 0.0))
        assert bool(jnp.all(sl > 0.0)), "rescaled table produced spurious zero speeds"
        np.testing.assert_allclose(np.asarray(sl),
                                   np.asarray(jnp.sqrt(lam) * s1), rtol=1e-10)
