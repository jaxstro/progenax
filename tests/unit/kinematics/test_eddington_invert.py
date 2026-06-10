"""Generic Eddington inversion: extraction regression + Plummer analytic oracle."""

import jax.numpy as jnp
import numpy as np
import pytest

# Pre-refactor pins from _eff_eddington_table(1.0, 4.0, 12.0, r_a)
# (captured at Task 1 Step 1, full repr precision).
PINS = {
    "iso": (
        5.504246729970181,
        0.7024482367263554,
        [
            0.00012431051303885793,
            0.0007435780468229273,
            0.0043761938193880336,
            0.017345663822072446,
            0.056220494086212325,
        ],
    ),
    "om": (
        5.504246729970181,
        0.7024482367263554,
        [
            0.002275026047535748,
            0.001516930253864097,
            0.005020898446083242,
            0.01595233272056164,
            0.046931165805733276,
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
