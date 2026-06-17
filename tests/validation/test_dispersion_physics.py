"""Physics validation for the differentiable Jeans dispersion forward model.

Phase 0 Task 3 — the **Plummer dispersion anchor**. A 3-way + convergence
validation of ``jeans_dispersion``:

1. ``test_plummer_isotropic_jeans_vs_analytic`` — the TIGHT, load-bearing oracle:
   isotropic Jeans sigma_r vs the EXACT closed form
   ``sigma_r^2(r) = G M / (6 sqrt(r^2 + a^2))`` (isotropic Plummer; sigma_r = sigma_1d),
   rtol 1e-3 at r = [0.3, 0.7, 1.0, 2.0].
2. ``test_jeans_quadrature_convergence`` — order-of-accuracy: the error vs that
   analytic oracle falls ~4x per s-grid doubling (trapezoid O(h^2)). Verifies the
   *method*, not one resolution.
3. ``test_plummer_isotropic_jeans_equals_ftable`` — cross-check vs the
   ``s^2 f(Psi - s^2/2)`` speed-second-moment over the isotropic Plummer DF table.
4. ``test_plummer_om_jeans_matches_sampler`` — the OM anchor: Jeans sigma_r/sigma_t
   vs the empirical std of a sampled OM Plummer population (5% MC tol).
5. ``test_plummer_om_jeans_vs_analytic`` — the TIGHT OM oracle: OM Jeans sigma_r
   vs the EXACT closed form ``_plummer_om_sigma_r2_analytic`` (derived from the OM
   anisotropic Jeans equation; reduces to the isotropic form as r_a -> ∞ and matches
   the numerical OM Jeans to ~1e-6), rtol 1e-3 at r_a ∈ {1, 2, 5}.

Physics references
------------------
- Binney & Tremaine (2008) sec. 4.8.3 — anisotropic Jeans equation.
- Merritt (1985) AJ 90, 1027 — isotropic Plummer DF f(E) ∝ (-E)^(7/2) (Eq. 42),
  OM anisotropy beta = r^2/(r^2 + r_a^2).
- Plummer isotropic dispersion sigma_r^2(r) = GM/(6 sqrt(r^2 + a^2)) — standard
  result (e.g. Binney & Tremaine 2008, Table 4.1 / Plummer 1911).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax import project_dispersion
from progenax.profiles import PlummerProfile, EFFProfile, MichieProfile
from progenax.kinematics import (
    PlummerVelocityDF,
    EFFVelocityDF,
    MichieVelocityDF,
    jeans_dispersion,
    df_moment_dispersion,
)
from progenax.kinematics.dispersion import ftable_sigma_r_isotropic
from progenax.kinematics.eff_df import _eff_eddington_table

G = 0.00449


def _plummer_isotropic_sigma_r2_analytic(r, M, a):
    """EXACT isotropic Plummer radial dispersion sigma_r^2(r) = GM/(6 sqrt(r^2+a^2)).

    Standard closed form (Plummer 1911; Binney & Tremaine 2008). For the isotropic
    model sigma_r = sigma_t = sigma_1d, so this is also sigma_1d^2.
    """
    return G * M / (6.0 * jnp.sqrt(r**2 + a**2))


def _plummer_om_sigma_r2_analytic(r, M, a, r_a):
    """EXACT Osipkov-Merritt Plummer radial dispersion sigma_r^2(r).

        sigma_r^2(r) = G M (a^2 + 3 r^2 + 2 r_a^2)
                       / [ 12 (r^2 + r_a^2) sqrt(a^2 + r^2) ]

    DERIVATION (not a literature transcription — Plummer is not an Osipkov-Merritt
    gamma-model, so Carollo, de Zeeuw & van der Marel 1995, MNRAS 276, 1131 give the
    OM *framework* and beta(r) = r^2/(r^2 + r_a^2) (their Eq. 5) but not this Plummer
    closed form). Integrate the OM anisotropic Jeans equation (Binney & Tremaine 2008
    sec. 4.8.3), d(rho sigma_r^2)/dr + (2 beta/r) rho sigma_r^2 = -rho dPhi/dr, with
    integrating factor (r^2 + r_a^2):

        (r^2 + r_a^2) rho sigma_r^2 = G M ∫_r^∞ (r'^2 + r_a^2) rho(r') r'/(a^2+r'^2)^{3/2} dr'.

    For Plummer rho ∝ (a^2 + r'^2)^{-5/2}, Phi = -GM/sqrt(a^2+r'^2), the integral is
    elementary (substitute u = r'^2) and yields the closed form above.

    VERIFIED this session (2026-06-17): reduces to the isotropic Dejonghe-1987 form
    GM/(6 sqrt(a^2+r^2)) as r_a -> ∞ (machine precision, 2e-16), and matches the
    package's independent numerical OM Jeans integration to ~1e-6 across
    r_a ∈ {1, 2, 5} pc and r ∈ [0.3, 10] pc. For the isotropic model sigma_r = sigma_t;
    here sigma_t^2 = (1 - beta) sigma_r^2 with beta = r^2/(r^2 + r_a^2).
    """
    return (
        G * M * (a**2 + 3.0 * r**2 + 2.0 * r_a**2)
        / (12.0 * (r**2 + r_a**2) * jnp.sqrt(a**2 + r**2))
    )


def test_plummer_isotropic_jeans_vs_analytic():
    """TIGHT oracle: isotropic Jeans sigma_r vs the exact closed form, rtol 1e-3."""
    prof = PlummerProfile(r_h=1.0)
    M = 400.0
    r = jnp.array([0.3, 0.7, 1.0, 2.0])
    dp = jeans_dispersion(prof, None, r, M=M, G=G)
    truth = jnp.sqrt(_plummer_isotropic_sigma_r2_analytic(r, M, prof.a))
    # Isotropic: sigma_r == sigma_1d == sigma_t.
    assert jnp.allclose(dp.sigma_r, truth, rtol=1e-3)
    assert jnp.allclose(dp.sigma_1d, truth, rtol=1e-3)
    assert jnp.allclose(dp.sigma_r, dp.sigma_t, rtol=1e-6)


def test_jeans_quadrature_convergence():
    """Order-of-accuracy: the trapezoid s-quadrature converges at O(h^2) (~4x/doubling).

    SELF-CONVERGENCE (Cauchy) study, NOT error-vs-analytic. As of Task C,
    ``jeans_dispersion`` no longer truncates the Plummer tail: the outward Jeans
    integral is now evaluated on an algebraically compactified s-grid
    (s = a t/(1-t), the full semi-infinite tail mapped to a finite uniform-t grid),
    so there is no n_s-independent truncation floor to remove. The self-difference
    ``d_n = |sigma(n) - sigma(2n)|`` is therefore a DEFENSIVE O(h^2) probe: with the
    truncation floor gone, both |sigma(n) - analytic| AND d_n are dominated by the
    pure discretisation term, and d_n falls ~4x per doubling (textbook trapezoid).
    Using the self-difference (rather than the absolute error) keeps the probe robust
    to any residual sub-leading modelling term and to interpolation noise at the
    finest grids. (numerical-method-validation: self-convergence as an order-of-
    accuracy check that is insensitive to the absolute-error constant.)

    The tight absolute accuracy vs the exact closed form is gated separately, at the
    production resolution, by ``test_plummer_isotropic_jeans_vs_analytic`` (rtol 1e-3).
    """
    prof = PlummerProfile(r_h=1.0)
    M = 400.0
    r = jnp.array([1.0])

    ns = [500, 1000, 2000, 4000, 8000]
    vals = [
        float(jeans_dispersion(prof, None, r, M=M, G=G, n_s=n_s).sigma_r[0]) for n_s in ns
    ]
    # Cauchy differences d_n = |sigma(n) - sigma(2n)|: the truncation bias cancels.
    diffs = [abs(vals[i] - vals[i + 1]) for i in range(len(vals) - 1)]
    ratios = [diffs[i] / diffs[i + 1] for i in range(len(diffs) - 1)]
    for ratio in ratios:
        # O(h^2) -> ratio ~ 4. Allow [3.0, 5.0]: the finest pair approaches float
        # noise / sub-leading terms (interp), the rest are textbook ~4.
        assert 3.0 <= ratio <= 5.0, (
            f"self-convergence ratio {ratio:.3f} not ~4 (O(h^2)); diffs={diffs}"
        )


def test_plummer_isotropic_jeans_equals_ftable():
    """Cross-check: isotropic Jeans sigma_r == f-table speed second moment.

    ``ftable_sigma_r_isotropic`` computes sigma_r^2 = <s^2>/3 over the isotropic
    Plummer DF speed pdf ``s^2 f(Psi - s^2/2)`` directly (a different code path from
    the Jeans integral). They must agree to TABLE RESOLUTION.

    Tolerance: 1e-2. This is a *table-resolution floor*, NOT a physics tolerance.
    The f-table path interpolates the DF on an n_e=2000 energy grid and does a
    256-point speed quadrature; the residual is the two methods' independent
    discretisation error, which shrinks with grid refinement (it is a numerical
    floor, not a model disagreement). It must NOT be loosened to hide a physics bug.
    """
    prof = PlummerProfile(r_h=1.0)
    M = 400.0
    r = jnp.array([0.3, 0.7, 1.0, 2.0])

    # Isotropic Plummer DF table (dimensionless, Merritt 1985 Eq. 42): f(E) ∝ E^(7/2).
    # sigma0^2 = G M / (6 a), psi(r) = 6 / sqrt(1 + r^2/a^2) (units of sigma0^2).
    n_e = 2000
    E_grid = jnp.linspace(0.0, 6.0, n_e)  # x = Q/sigma0^2 in [0, psi_max=6]
    f_grid = E_grid**3.5
    sigma0_2 = G * M / (6.0 * prof.a)
    Psi_r = 6.0 / jnp.sqrt(1.0 + (r / prof.a) ** 2)

    # f-table sigma_r (dimensionless) -> physical via sigma0.
    sigma_r2_ft = jax.vmap(
        lambda psi: ftable_sigma_r_isotropic(E_grid, f_grid, psi)
    )(Psi_r) * sigma0_2
    sigma_r_ft = jnp.sqrt(sigma_r2_ft)

    sigma_r_jeans = jeans_dispersion(prof, None, r, M=M, G=G).sigma_r

    assert jnp.allclose(sigma_r_ft, sigma_r_jeans, rtol=1e-2)


@pytest.mark.slow
def test_plummer_isotropic_tail_machine_precision():
    """Compactified Plummer Jeans matches GM/(6 sqrt(r^2+a^2)) at OUTER radii.

    The old s-grid truncated the semi-infinite Plummer outward Jeans integral at
    r_max=30a, leaving a fixed n_s-independent tail bias that grows outward (~8.6e-4
    rel. at r=20). Task C maps the semi-infinite domain to t in [_T_MIN, _T_MAX] via
    s = a t/(1-t) (Jacobian ds/dt = a/(1-t)^2), capturing the full tail, so the
    outer-radius residual collapses far below the old truncation floor. The TIGHT
    inner anchor (rtol 1e-3 at r<=2) is gated by
    test_plummer_isotropic_jeans_vs_analytic; this probes the OUTER radii the
    truncation bit.
    """
    prof = PlummerProfile(r_h=1.0)
    M = 400.0
    r = jnp.array([2.0, 5.0, 10.0, 20.0])  # outer radii where the 30a tail bit
    # With compactification the residual is now PURE O(h^2) discretisation (no
    # truncation floor): clean ~4x/doubling, e.g. n_s=4000 -> 9.8e-5, 8000 -> 2.2e-5,
    # 16000 -> 6.2e-6 at r=20. The < 5e-5 gate is met at n_s=8000 (the resolution this
    # anchor pins); this is a resolution CHOICE, not a tolerance loosening — the old
    # 8.6e-4 was an n_s-independent truncation bias that NO resolution could remove.
    dp = jeans_dispersion(prof, None, r, M=M, G=G, n_s=8000)
    truth = jnp.sqrt(G * M / (6.0 * jnp.sqrt(r**2 + prof.a**2)))
    max_rel = float(jnp.max(jnp.abs(dp.sigma_1d / truth - 1.0)))
    assert max_rel < 5e-5, (
        f"Plummer tail residual {max_rel:.2e} >= 5e-5 (was ~8.6e-4 truncated at 30a); "
        f"sigma_1d={dp.sigma_1d} truth={truth}"
    )


@pytest.mark.slow
def test_plummer_om_jeans_matches_sampler():
    """OM anchor: Jeans sigma_r/sigma_t vs the empirical std of an OM-sampled population.

    Sample N stars at fixed radii r0 (all positions on the x-axis), draw OM Plummer
    velocities, and compare the std of v_x (radial here) and of the tangential plane
    (v_y, v_z) to ``jeans_dispersion(...).sigma_r/.sigma_t``. Within 5% MC tol.
    """
    r_h = 1.0
    r_a = 2.0
    M = 400.0
    N = 200_000
    prof = PlummerProfile(r_h=r_h)
    df = PlummerVelocityDF(r_h=r_h, anisotropy_radius=r_a)

    for r0 in (0.5, 1.0, 2.0):
        # All stars at (r0, 0, 0): radial direction = x_hat, tangential = (y, z).
        positions = jnp.zeros((N, 3)).at[:, 0].set(r0)
        masses = jnp.full((N,), M / N)
        key = jax.random.PRNGKey(int(r0 * 1000) + 7)
        v = df.sample_velocities(positions, masses, key, G=G)

        sigma_r_emp = jnp.std(v[:, 0])
        # Tangential 1-component dispersion: var over the 2 tangential axes,
        # sigma_t^2 = (var(v_y) + var(v_z)) / 2.
        sigma_t_emp = jnp.sqrt(0.5 * (jnp.var(v[:, 1]) + jnp.var(v[:, 2])))

        dp = jeans_dispersion(prof, r_a, jnp.array([r0]), M=M, G=G)
        assert jnp.allclose(dp.sigma_r[0], sigma_r_emp, rtol=0.05), (
            f"sigma_r r0={r0}: jeans={float(dp.sigma_r[0]):.4f} emp={float(sigma_r_emp):.4f}"
        )
        assert jnp.allclose(dp.sigma_t[0], sigma_t_emp, rtol=0.05), (
            f"sigma_t r0={r0}: jeans={float(dp.sigma_t[0]):.4f} emp={float(sigma_t_emp):.4f}"
        )


def test_plummer_om_jeans_vs_analytic():
    """TIGHT OM oracle: OM Jeans sigma_r vs the derived closed form, rtol 1e-3.

    Completes the OM anchor begun by ``test_plummer_om_jeans_matches_sampler`` (which
    checks against a sampled population at 5% MC tol). Here the OM Jeans sigma_r is
    checked against the EXACT closed form ``_plummer_om_sigma_r2_analytic`` (derived
    from the OM anisotropic Jeans equation; see that helper's docstring) at three
    anisotropy radii. The measured agreement at the default s-grid is ~1e-6 across
    these radii, so the 1e-3 gate (mirroring the isotropic anchor) is robust, not
    tuned — the residual is pure O(h^2) quadrature, the formula is exact.

    Also pins the by-construction OM tangential relation sigma_t^2 = (1 - beta)
    sigma_r^2 with beta(r) = r^2/(r^2 + r_a^2).
    """
    prof = PlummerProfile(r_h=1.0)
    M = 400.0
    a = prof.a
    r = jnp.array([0.3, 0.5, 1.0, 2.0])
    for r_a in (1.0, 2.0, 5.0):
        dp = jeans_dispersion(prof, r_a, r, M=M, G=G)
        truth_r = jnp.sqrt(_plummer_om_sigma_r2_analytic(r, M, a, r_a))
        assert jnp.allclose(dp.sigma_r, truth_r, rtol=1e-3), (
            f"OM sigma_r r_a={r_a}: jeans={dp.sigma_r} analytic={truth_r}"
        )
        # sigma_t^2 = (1 - beta) sigma_r^2 by OM construction.
        beta = r**2 / (r**2 + r_a**2)
        truth_t = jnp.sqrt((1.0 - beta)) * truth_r
        assert jnp.allclose(dp.sigma_t, truth_t, rtol=1e-3), (
            f"OM sigma_t r_a={r_a}: jeans={dp.sigma_t} analytic={truth_t}"
        )


# =============================================================================
# Phase 0 Task 4 — EFF + Michie 3-D anchors
# =============================================================================
#
# ``jeans_dispersion`` already works for any profile exposing ``.density(r)``
# (and ``.r_t`` for the finite outward-integral extent), so this is mostly
# *tests*. Two model families:
#
#   EFF (truncated power law, Elson-Fall-Freeman 1987): finite ``r_t`` -> the
#   Jeans s-grid is finite (no Plummer-style truncation-tail bias). With
#   gamma=5 the EFF reduces to Plummer and the sharp-truncation virial offset
#   is small (~1%), so it is a near-equilibrium sampler anchor. The isotropic
#   EFF DF stores ``(E_grid, f_grid, Psi_grid, mu)`` -> we can cross-check the
#   isotropic Jeans sigma_r against ``ftable_sigma_r_isotropic`` (a different
#   code path: speed second moment over the Eddington f-table).
#
#   Michie (Michie 1963 anisotropy + King 1966 cutoff): INTRINSICALLY
#   anisotropic, so its stored f-table is NOT isotropic and
#   ``ftable_sigma_r_isotropic`` does not apply. Crucially, the Michie-King
#   anisotropy law is NOT identical to the Osipkov-Merritt beta = r^2/(r^2+r_a^2)
#   that ``jeans_dispersion`` assumes; the two agree well in the core / inner
#   region but diverge in the far outskirts (the anisotropy *profiles* differ).
#   So Michie is validated by (a) the sampler in the inner region where OM is a
#   good model, and (b) the King/isotropic limit (large r_a -> beta -> 0).


@pytest.mark.slow
def test_eff_isotropic_jeans_matches_sampler():
    """EFF isotropic Jeans sigma_r vs empirical std of an EFF-sampled population (5% MC).

    gamma=5 (Plummer-reducing, mild truncation) is a near-equilibrium EFF, so the
    Eddington sampler is a faithful realisation of the same density the Jeans
    integral uses. Sample N stars at fixed radii (all on the x-axis: radial = x),
    inside r_t, and compare std(v_x) to ``jeans_dispersion(...).sigma_r``.
    """
    a, gamma, r_t = 1.0, 5.0, 10.0
    M = 400.0
    N = 200_000
    prof = EFFProfile(a=a, gamma=gamma, r_t=r_t)
    df = EFFVelocityDF(a=a, gamma=gamma, r_t=r_t)  # isotropic (anisotropy_radius=None)

    for r0 in (0.5, 1.0, 2.0):
        positions = jnp.zeros((N, 3)).at[:, 0].set(r0)
        masses = jnp.full((N,), M / N)  # sum == M, the mass passed to jeans
        key = jax.random.PRNGKey(int(r0 * 1000) + 11)
        v = df.sample_velocities(positions, masses, key, G=G)

        sigma_r_emp = jnp.std(v[:, 0])
        dp = jeans_dispersion(prof, None, jnp.array([r0]), M=M, G=G)
        assert jnp.allclose(dp.sigma_r[0], sigma_r_emp, rtol=0.05), (
            f"EFF sigma_r r0={r0}: jeans={float(dp.sigma_r[0]):.4f} "
            f"emp={float(sigma_r_emp):.4f}"
        )


def test_eff_om_beta_identity():
    """OM EFF: beta(r) == r^2/(r^2+r_a^2) (rtol 1e-6) and sigma_r >= sigma_t.

    For an Osipkov-Merritt model the realised anisotropy IS the OM identity by
    construction; this asserts ``jeans_dispersion`` propagates it correctly for a
    truncated (EFF) profile and that the radial dispersion is the larger one (the
    radial-anisotropy signature, beta >= 0).
    """
    a, gamma, r_t = 1.0, 5.0, 10.0
    M = 400.0
    r_a = 4.0
    prof = EFFProfile(a=a, gamma=gamma, r_t=r_t)
    r = jnp.array([0.5, 1.0, 2.0, 4.0])

    dp = jeans_dispersion(prof, r_a, r, M=M, G=G)
    beta_id = r**2 / (r**2 + r_a**2)
    assert jnp.allclose(dp.beta, beta_id, rtol=1e-6)
    assert jnp.all(dp.sigma_r >= dp.sigma_t)


def test_eff_isotropic_jeans_equals_ftable():
    """Cross-check: isotropic EFF Jeans sigma_r == Eddington f-table speed 2nd moment.

    The EFF isotropic DF stores the Eddington f(E) (``E_grid, f_grid``), the
    dimensionless relative potential ``Psi(r)`` (``r_grid, Psi_grid``), and the mass
    integral ``mu``. The speed-second-moment ``<s^2>/3`` over the (dimensionless)
    speed pdf ``s^2 f(Psi(r) - s^2/2)`` is sigma_r^2 in DF units; the physical scale
    is ``kappa = G M / (4 pi mu)`` (the same self-consistent velocity scale the
    sampler uses). This is a fully INDEPENDENT code path from the Jeans integral
    (the Eddington inversion of the same density), so agreement validates both.

    We restrict to the core/mid radii [0.5, 1.0, 1.5] (well inside r_t=10): near
    the truncation edge both the Eddington-table accuracy and the EFF sharp-cutoff
    non-stationarity degrade, which is a known EFF limitation, not a Jeans bug.

    Tolerance 2e-2 is a TABLE-RESOLUTION floor, NOT a physics tol: the residual is
    the two methods' independent discretisation error (Eddington f(E) grid + speed
    quadrature vs the Jeans s-grid). The resolution-refinement block below is the
    numerical-vs-bug discriminator: as the speed quadrature n_s is refined from
    coarse to its plateau the gap SHRINKS (~3x per step) toward a fixed
    Jeans-vs-Eddington-table floor of ~2-3e-4 — a converging numerical residual,
    not a model disagreement. It must NOT be loosened to hide a physics bug.
    """
    a, gamma, r_t = 1.0, 5.0, 10.0
    M = 400.0
    r = jnp.array([0.5, 1.0, 1.5])  # core/mid, away from the r_t truncation edge

    prof = EFFProfile(a=a, gamma=gamma, r_t=r_t)
    df = EFFVelocityDF(a=a, gamma=gamma, r_t=r_t)  # isotropic Eddington DF

    # Physical velocity scale kappa = G M / (4 pi mu) (sampler's self-consistent scale).
    kappa = G * M / (4.0 * jnp.pi * df.mu)
    Psi_r = jnp.interp(r, df.r_grid, df.Psi_grid, left=df.Psi_grid[0], right=0.0)

    sigma_r_ft = jnp.sqrt(
        jax.vmap(lambda psi: ftable_sigma_r_isotropic(df.E_grid, df.f_grid, psi))(Psi_r)
        * kappa
    )
    sigma_r_jeans = jeans_dispersion(prof, None, r, M=M, G=G).sigma_r
    assert jnp.allclose(sigma_r_ft, sigma_r_jeans, rtol=2e-2)

    # --- Resolution-refinement check (the numerical-vs-bug discriminator) ---
    # Fixed high-resolution Eddington table + fixed high-resolution Jeans reference;
    # refine ONLY the f-table speed quadrature n_s. A genuine numerical floor SHRINKS
    # as n_s grows (until it hits the fixed Jeans-vs-table residual); a physics bug
    # would leave a non-vanishing, n_s-independent gap.
    rr, Psi, E_grid, f_grid, mu = _eff_eddington_table(
        a, gamma, r_t, None, n_r=12000, n_e=4000
    )
    kappa_hi = G * M / (4.0 * jnp.pi * mu)
    Psi_hi = jnp.interp(r, rr, Psi, left=Psi[0], right=0.0)
    sigma_jeans_hi = jeans_dispersion(prof, None, r, M=M, G=G, n_s=8000).sigma_r

    def ft_gap(n_s):
        sft = jnp.sqrt(
            jax.vmap(
                lambda psi: ftable_sigma_r_isotropic(E_grid, f_grid, psi, n_s=n_s)
            )(Psi_hi)
            * kappa_hi
        )
        return float(jnp.max(jnp.abs(sft - sigma_jeans_hi) / sigma_jeans_hi))

    gaps = [ft_gap(n_s) for n_s in (64, 128, 256)]
    # Monotone shrink as the speed quadrature is refined toward the floor.
    assert gaps[0] > gaps[1] > gaps[2], (
        f"f-table gap did not shrink with n_s refinement: {gaps} "
        f"(non-converging gap -> suspect a physics bug, not a numerical floor)"
    )
    # Converged gap is a small numerical floor (Jeans s-grid vs Eddington table).
    assert gaps[-1] < 1e-3, f"converged f-table gap {gaps[-1]:.2e} too large"


@pytest.mark.slow
def test_michie_jeans_matches_sampler():
    """Michie OM-Jeans sigma_r/sigma_t vs empirical std of a Michie-sampled population.

    The Jeans solver assumes Osipkov-Merritt beta = r^2/(r^2+r_a^2); the Michie-King
    DF has its OWN (similar but not identical) anisotropy law. They agree to MC tol
    in the inner region (r << r_t) where OM is a good model of the Michie anisotropy,
    and diverge in the far outskirts (a real OM-vs-Michie-King model difference, NOT
    a Jeans bug — documented in the module note above). We therefore validate at
    inner radii r in [0.5, 3.0] for a model with r_t ~ 28 (so r < ~0.1 r_t).
    """
    W0, r_c, r_a = 6.0, 1.0, 5.0
    M = 400.0
    N = 200_000
    prof = MichieProfile.from_W0_rc(W0=W0, r_c=r_c, r_a=r_a)
    df = MichieVelocityDF(W0=W0, r_c=r_c, r_a=r_a)

    for r0 in (0.5, 1.0, 2.0, 3.0):
        positions = jnp.zeros((N, 3)).at[:, 0].set(r0)
        masses = jnp.full((N,), M / N)
        key = jax.random.PRNGKey(int(r0 * 1000) + 3)
        v = df.sample_velocities(positions, masses, key, G=G)

        sigma_r_emp = jnp.std(v[:, 0])
        sigma_t_emp = jnp.sqrt(0.5 * (jnp.var(v[:, 1]) + jnp.var(v[:, 2])))

        # Michie's anisotropy radius is ``r_a`` (length), exposed as prof.r_a.
        dp = jeans_dispersion(prof, float(prof.r_a), jnp.array([r0]), M=M, G=G)
        assert jnp.allclose(dp.sigma_r[0], sigma_r_emp, rtol=0.05), (
            f"Michie sigma_r r0={r0}: jeans={float(dp.sigma_r[0]):.4f} "
            f"emp={float(sigma_r_emp):.4f}"
        )
        assert jnp.allclose(dp.sigma_t[0], sigma_t_emp, rtol=0.05), (
            f"Michie sigma_t r0={r0}: jeans={float(dp.sigma_t[0]):.4f} "
            f"emp={float(sigma_t_emp):.4f}"
        )


def test_michie_beta_increases_outward():
    """Michie OM-Jeans beta(r) grows with radius (radial anisotropy builds outward)."""
    W0, r_c, r_a = 6.0, 1.0, 5.0
    M = 400.0
    prof = MichieProfile.from_W0_rc(W0=W0, r_c=r_c, r_a=r_a)
    r = jnp.linspace(0.3, 0.3 * float(prof.r_t), 8)
    dp = jeans_dispersion(prof, float(prof.r_a), r, M=M, G=G)
    assert float(dp.beta[-1]) > float(dp.beta[0])
    assert jnp.all(jnp.diff(dp.beta) >= 0.0)  # monotone non-decreasing (OM identity)


def test_michie_isotropic_limit():
    """King/isotropic limit: large r_a -> beta ~ 0 across r, sigma_r ~ sigma_t.

    With r_a >> the query radii the Michie model approaches the isotropic King limit,
    and the OM beta = r^2/(r^2+r_a^2) collapses toward 0 (the realised Michie
    anisotropy likewise vanishes). Asserts beta < a few e-2 everywhere.
    """
    W0, r_c, r_a = 7.0, 1.0, 50.0  # r_a >> the [0.5, 4] query radii
    M = 400.0
    prof = MichieProfile.from_W0_rc(W0=W0, r_c=r_c, r_a=r_a)
    r = jnp.array([0.5, 1.0, 2.0, 4.0])
    dp = jeans_dispersion(prof, float(prof.r_a), r, M=M, G=G)
    assert jnp.allclose(dp.beta, 0.0, atol=2e-2)
    assert jnp.allclose(dp.sigma_r, dp.sigma_t, rtol=2e-2)


# =============================================================================
# Phase 0 Task 5 — project_dispersion (Binney & Mamon 1982 LOS projection)
# =============================================================================
#
# project_dispersion projects the 3-D anisotropic Jeans model onto the sky
# (Binney & Mamon 1982, MNRAS 200, 361), returning the OBSERVED sigma_los (RV
# channel), sigma_pm,R / sigma_pm,T (proper-motion radial/tangential), and the
# projected surface density Sigma. The line-of-sight integral
#   int_R^inf g(r) r/sqrt(r^2-R^2) dr
# is singular at r=R; the singularity is removed ANALYTICALLY by the
# substitution r^2 = R^2 + u^2 (so r dr/sqrt(r^2-R^2) = du), giving a smooth,
# differentiable u-quadrature with NO 1/sqrt(r^2-R^2) ever evaluated.
#
# B&M82 kernels (beta = OM r^2/(r^2+r_a^2)):
#   Sigma          = 2 int (rho)                    du
#   Sigma sig_los^2= 2 int (1 - beta R^2/r^2) rho sig_r^2 du
#   Sigma sig_pmR^2= 2 int (1 - beta + beta R^2/r^2) rho sig_r^2 du
#   Sigma sig_pmT^2= 2 int (1 - beta) rho sig_r^2 du


def test_projection_isotropic_all_equal():
    """Load-bearing structural gate: isotropic (beta=0) -> all three projected
    dispersions are IDENTICAL.

    With beta=0 every B&M82 kernel collapses to 1, so sigma_los, sigma_pm,R and
    sigma_pm,T are the SAME u-integral of (rho sigma_r^2) over (rho). No external
    formula is needed — this is a pure algebraic identity of the kernels, so any
    mismatch exposes a bug in the kernel construction or the u-substitution. Tight
    rtol 1e-3.
    """
    prof = PlummerProfile(r_h=1.0)
    R = jnp.array([0.5, 1.0, 2.0, 4.0])
    pj = project_dispersion(prof, None, R, 400.0, G)  # r_a=None -> beta=0
    assert jnp.allclose(pj.sigma_los, pj.sigma_pm_r, rtol=1e-3)
    assert jnp.allclose(pj.sigma_los, pj.sigma_pm_t, rtol=1e-3)


def test_projection_eff_finite_rt():
    """Finite-r_t projection branch: project an isotropic EFF (King-like cutoff).

    Both isotropic projection oracles above use Plummer, which takes the
    SEMI-INFINITE (compactified-tau) u-grid branch of ``project_dispersion``. A
    finite-``r_t`` profile (EFF / King) instead takes the uniform-``u`` branch
    (``u_max = sqrt(r_edge^2 - R^2)``, ``jac = 1``) — exercised here for the first
    time. No closed-form oracle is asserted (none is source-verified for the
    truncated EFF projection); this is a STRUCTURAL gate: finite, positive outputs
    plus the isotropic kernel identity. With beta=0 (r_a=None) every B&M82 kernel
    collapses to 1, so sigma_los == sigma_pm_r == sigma_pm_t exactly (rtol 1e-3) —
    the same algebraic identity as ``test_projection_isotropic_all_equal``, now on
    the finite-r_t code path.
    """
    a, gamma, r_t = 1.0, 5.0, 12.0
    M = 400.0
    prof = EFFProfile(a=a, gamma=gamma, r_t=r_t)
    R = jnp.array([1.0, 3.0, 6.0])  # all well inside r_t

    pj = project_dispersion(prof, None, R, M, G)  # isotropic -> beta=0

    # Shapes propagate the input R.
    assert pj.sigma_los.shape == R.shape
    assert pj.Sigma.shape == R.shape
    # Surface density strictly positive.
    assert jnp.all(pj.Sigma > 0.0)
    # All three dispersions finite and strictly positive.
    for sig in (pj.sigma_los, pj.sigma_pm_r, pj.sigma_pm_t):
        assert jnp.all(jnp.isfinite(sig))
        assert jnp.all(sig > 0.0)
    # Isotropic identity (beta=0): the three kernels collapse to 1.
    assert jnp.allclose(pj.sigma_los, pj.sigma_pm_r, rtol=1e-3)
    assert jnp.allclose(pj.sigma_los, pj.sigma_pm_t, rtol=1e-3)


def test_projection_isotropic_plummer_los_oracle():
    """TIGHT absolute oracle: isotropic-Plummer projected LOS dispersion has the
    EXACT closed form sigma_los^2(R) = (3 pi / 64) G M / sqrt(a^2 + R^2).

    Source: Dejonghe (1987, MNRAS 224, 13) gives the projected dispersion of the
    Plummer family; for the isotropic member it reduces to the (3 pi / 64) form
    above. INDEPENDENTLY confirmed in-session by a scipy.integrate.quad evaluation
    of the B&M82 LOS integral (r^2=R^2+u^2 substitution), which reproduces this
    closed form to 6+ significant figures across R in [1e-4, 4] a — so it is
    source-verified AND numerically cross-checked, not fabricated.

    rtol 3e-3: the projected sigma^2 ratio (numerator/denominator share the same
    r_max=30a tail) is far less truncation-sensitive than the absolute Sigma; the
    independent reference quadrature on the same truncated grid agrees with the
    analytic oracle to ~1e-5, leaving comfortable margin under 3e-3.
    """
    prof = PlummerProfile(r_h=1.0)
    M = 400.0
    R = jnp.array([0.5, 1.0, 2.0])
    pj = project_dispersion(prof, None, R, M, G)
    los2_oracle = (3.0 * jnp.pi / 64.0) * G * M / jnp.sqrt(prof.a**2 + R**2)
    assert jnp.allclose(pj.sigma_los, jnp.sqrt(los2_oracle), rtol=3e-3)


@pytest.mark.slow
def test_projection_plummer_los_oracle_converges():
    """The compactified projection u-grid kills the outer-R truncation floor.

    Before Task C (cont.), ``project_dispersion``'s OWN outward u-quadrature
    truncated the Plummer tail at u_max = sqrt((30a)^2 - R^2), leaving a FIXED,
    n_u-INDEPENDENT truncation floor of 1.634e-4 (rel.) in sigma_los at the outer
    radius R=4a vs the exact Dejonghe (1987) isotropic oracle
    sigma_los^2(R) = (3 pi / 64) G M / sqrt(a^2 + R^2). Replacing the truncated
    u = linspace(0, u_max, n_u) with an algebraic compactification of the
    semi-infinite u in [0, inf) (u = u_c t/(1-t)) integrates the full tail, so the
    residual is now PURE O(h^2) trapezoid error: it DECREASES as n_u grows (the
    old floor was n_u-independent — that is the proof the floor is gone).
    """
    prof = PlummerProfile(r_h=1.0)
    M = 400.0
    R = jnp.array([0.5, 1.0, 2.0, 4.0])
    oracle = jnp.sqrt((3.0 * jnp.pi / 64.0) * G * M / jnp.sqrt(prof.a**2 + R**2))

    pj_2k = project_dispersion(prof, None, R, M, G, n_u=2000)
    pj_8k = project_dispersion(prof, None, R, M, G, n_u=8000)
    err_2k = float(jnp.max(jnp.abs(pj_8k.sigma_los / oracle - 1.0)))
    err_8k_pointwise = jnp.abs(pj_8k.sigma_los / oracle - 1.0)
    err_2k_pointwise = jnp.abs(pj_2k.sigma_los / oracle - 1.0)

    # 1. Floor is GONE: max rel err << the old 1.634e-4 truncation floor.
    assert err_2k < 2e-5, (
        f"projection LOS max rel err {err_2k:.3e} not < 2e-5 — truncation floor "
        f"still present (was 1.634e-4 n_u-independent before compactification)"
    )
    # 2. Error DECREASES with n_u (n_u=2000 -> 8000), proving the residual is now
    #    pure O(h^2) discretisation, not an n_u-independent truncation floor.
    assert float(jnp.max(err_8k_pointwise)) < float(jnp.max(err_2k_pointwise)), (
        f"projection LOS error did not decrease with n_u "
        f"(2000: {float(jnp.max(err_2k_pointwise)):.3e}, "
        f"8000: {float(jnp.max(err_8k_pointwise)):.3e}) — floor not removed"
    )


def test_projection_anisotropy_signature():
    """Radial-anisotropy signature in projection: with a small OM r_a the model is
    radially biased, and on-sky the TANGENTIAL proper motion is suppressed relative
    to the line-of-sight channel in the outskirts.

    Physical reasoning (B&M82 kernels): for beta>0,
      sigma_pm,T^2 / sigma_los^2 = <(1-beta)> / <(1 - beta R^2/r^2)>   (rho sig_r^2 weighted)
    Along a sightline at projected radius R the integration runs over r >= R, where
    beta R^2/r^2 <= beta, so the LOS kernel (1 - beta R^2/r^2) >= (1 - beta) = the
    tangential-PM kernel pointwise. Hence sigma_pm,T < sigma_los whenever beta>0
    anywhere along the sightline. We probe the outskirts (R = [2,4]) where the
    radial bias beta(r) is largest. r_a=1.0 a >= 0.75 a satisfies the Plummer OM
    validity bound (Merritt 1985 Eq. 46).

    Sanity ordering also asserted: sigma_pm,R >= sigma_los >= sigma_pm,T (the PM
    radial channel ADDS the +beta R^2/r^2 term the LOS channel SUBTRACTS).
    """
    prof = PlummerProfile(r_h=1.0)
    R = jnp.array([2.0, 4.0])
    pj = project_dispersion(prof, 1.0, R, 400.0, G)  # radial bias, r_a = a
    assert jnp.all(pj.sigma_pm_t < pj.sigma_los)
    assert jnp.all(pj.sigma_pm_r >= pj.sigma_los)
    assert jnp.all(pj.sigma_los >= pj.sigma_pm_t)


def test_projection_jit_and_grad():
    """Differentiability smoke: project_dispersion is jit-able and grad-able.

    jax.jit wraps a sigma_los evaluation (finite output); jax.grad of
    sum(sigma_los) w.r.t. the OM anisotropy radius r_a is finite and NONZERO
    (a zero gradient would flag a silent stop-gradient / clamped-interp bug).
    """
    prof = PlummerProfile(r_h=1.0)
    R = jnp.array([1.0, 2.0])
    M, Gc = 400.0, G

    @jax.jit
    def los(r_a):
        return project_dispersion(prof, r_a, R, M, Gc).sigma_los

    out = los(2.0)
    assert jnp.all(jnp.isfinite(out))

    def loss(r_a):
        return jnp.sum(project_dispersion(prof, r_a, R, M, Gc).sigma_los)

    g = jax.grad(loss)(2.0)
    assert jnp.isfinite(g)
    assert jnp.abs(g) > 1e-9


# =============================================================================
# Phase 0 Task 6 — projected EMPIRICAL anchor + beta recovery
# =============================================================================
#
# The closing leg of the projection anchor: sample a REAL anisotropic Plummer
# population, project it to the sky EXACTLY as an observer would, and confirm
# ``project_dispersion`` reproduces the empirical line-of-sight + proper-motion
# dispersions and that the projected observables CARRY the input anisotropy
# (the OED's whole premise: RV<->sigma_los, PM<->sigma_pm,R/T encode beta).
#
# Projection geometry (line of sight = the x-axis):
#   - on-sky (projected) radius:  R = sqrt(y^2 + z^2)   (positions[:,1], [:,2])
#   - sigma_los:                  std of v_x            (velocities[:,0])
#   - on-sky radial unit vector:  e_R = (y, z) / R
#   - on-sky tangential unit vec: e_T = (-z, y) / R     (perp to e_R, in-plane)
#   - v_pmR = (v_y, v_z) . e_R ;  v_pmT = (v_y, v_z) . e_T
#   - sigma_pm,R = std(v_pmR) ;   sigma_pm,T = std(v_pmT)
#
# We sample BOTH positions (Plummer density) and velocities (OM Plummer DF),
# i.e. the same full realisation an N-body IC builder produces, then bin in
# projected R and compare to ``project_dispersion`` at the bin's mean R.
#
# MC-noise note (root-caused, NOT a tolerance loosening): a projected annulus
# holds far fewer stars than a 3-D shell, and the dispersion VARIES across a
# finite-width bin, so we (a) use a large N, (b) keep bins narrow at
# small/intermediate R where they are well populated, and (c) compare the
# prediction at the bin's MEAN R. The outermost annuli (large R) thin out and
# carry larger MC scatter; we restrict the tight assertion to the
# well-populated bins (each >= a few thousand stars) and document the per-bin
# counts in the assertion messages. The residual is sampling noise, not model
# error (the analytic legs — isotropic-all-equal, the (3 pi/64) LOS oracle, the
# anisotropy-signature kernel ordering — are gated tightly above).

# Shared geometry / binning configuration for the empirical projection anchor.
_PROJ_R_A = 1.5  # OM anisotropy radius (>= 0.75 a; a ~ 0.766 -> 0.75a ~ 0.575)
_PROJ_M = 400.0
_PROJ_N = 500_000  # 5e5: projected annuli hold fewer stars than 3-D shells; this
#                    gives >= a few thousand stars/bin out to R ~ 3 with <5% MC.
_PROJ_BIN_EDGES = jnp.array([0.3, 0.5, 0.8, 1.2, 1.8, 3.0])


def _project_population(positions, velocities):
    """Project a 3-D (pos, vel) population to the sky with LOS = x-axis.

    Returns (R, v_los, v_pmR, v_pmT): the on-sky radius and the three projected
    velocity components per star (see the module note for the geometry).
    """
    y, z = positions[:, 1], positions[:, 2]
    R = jnp.sqrt(y**2 + z**2)
    R_safe = jnp.maximum(R, 1e-12)
    e_Ry, e_Rz = y / R_safe, z / R_safe           # on-sky radial unit vector
    e_Ty, e_Tz = -z / R_safe, y / R_safe          # on-sky tangential (perp)
    v_los = velocities[:, 0]                        # LOS = x
    v_y, v_z = velocities[:, 1], velocities[:, 2]
    v_pmR = v_y * e_Ry + v_z * e_Rz
    v_pmT = v_y * e_Ty + v_z * e_Tz
    return R, v_los, v_pmR, v_pmT


def _bin_empirical(R, v_los, v_pmR, v_pmT, edges):
    """Per-annulus empirical (R_center, n, sigma_los, sigma_pm,R, sigma_pm,T)."""
    centers, counts, slos, spmr, spmt = [], [], [], [], []
    for i in range(len(edges) - 1):
        m = (R >= edges[i]) & (R < edges[i + 1])
        centers.append(float(jnp.mean(R[m])))
        counts.append(int(jnp.sum(m)))
        slos.append(float(jnp.std(v_los[m])))
        spmr.append(float(jnp.std(v_pmR[m])))
        spmt.append(float(jnp.std(v_pmT[m])))
    return (
        jnp.array(centers),
        counts,
        jnp.array(slos),
        jnp.array(spmr),
        jnp.array(spmt),
    )


def _sample_om_plummer_population(r_a, M, N, seed):
    """Sample an OM Plummer population (positions from rho, velocities from the OM DF)."""
    prof = PlummerProfile(r_h=1.0)
    df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=r_a)
    key = jax.random.PRNGKey(seed)
    key_pos, key_vel = jax.random.split(key)
    masses = jnp.full((N,), M / N)               # sum == M (the mass jeans uses)
    positions = prof.sample_positions(masses, key_pos)
    velocities = df.sample_velocities(positions, masses, key_vel, G=G)
    return prof, positions, velocities


def test_projection_isotropic_geometry_sanity():
    """SANITY (cheap, not slow): isotropic population -> the three projected
    dispersions are empirically EQUAL in every bin.

    This pins the projection GEOMETRY (LOS axis, R = sqrt(y^2+z^2), the
    radial/tangential PM decomposition) BEFORE the anisotropic assertions: with
    beta=0 there is no preferred on-sky direction, so sigma_los, sigma_pm,R and
    sigma_pm,T must agree to MC tol. If the decomposition were swapped or wrong,
    this isotropic check is what would catch it (per the Task 6 brief). Uses a
    smaller N (this is a structural geometry check, not a precision anchor).
    """
    prof, pos, vel = _sample_om_plummer_population(r_a=None, M=_PROJ_M, N=200_000, seed=0)
    R, v_los, v_pmR, v_pmT = _project_population(pos, vel)
    centers, counts, slos, spmr, spmt = _bin_empirical(
        R, v_los, v_pmR, v_pmT, _PROJ_BIN_EDGES
    )
    # Isotropic: all three projected dispersions equal per bin (3% MC, finite N).
    assert jnp.allclose(slos, spmr, rtol=0.03), (
        f"isotropic geometry: sigma_los vs sigma_pm,R differ "
        f"(los={slos}, pmR={spmr}, counts={counts})"
    )
    assert jnp.allclose(slos, spmt, rtol=0.03), (
        f"isotropic geometry: sigma_los vs sigma_pm,T differ "
        f"(los={slos}, pmT={spmt}, counts={counts})"
    )


@pytest.mark.slow
def test_projection_empirical_los_and_pm():
    """Empirical projected anchor: project_dispersion vs the binned empirical
    sigma_los / sigma_pm,R / sigma_pm,T of a sampled OM Plummer population (5% MC).

    Sample N=5e5 stars (positions from the Plummer density, velocities from the
    OM Plummer DF, r_a=1.5), project along x (R = sqrt(y^2+z^2)), bin in
    projected R, and compare each empirical bin dispersion to
    ``project_dispersion(...)`` evaluated at the bin's MEAN R. Within 5% MC.

    Bin edges = [0.3, 0.5, 0.8, 1.2, 1.8, 3.0] (the per-bin counts are reported in
    the assertion messages). Every bin here holds >= a few thousand stars; the
    residual is sampling noise (the tight ANALYTIC legs are gated separately
    above), so the tolerance is NOT loosened to mask model error.
    """
    prof, pos, vel = _sample_om_plummer_population(
        r_a=_PROJ_R_A, M=_PROJ_M, N=_PROJ_N, seed=7
    )
    R, v_los, v_pmR, v_pmT = _project_population(pos, vel)
    centers, counts, slos, spmr, spmt = _bin_empirical(
        R, v_los, v_pmR, v_pmT, _PROJ_BIN_EDGES
    )

    # Require each bin to be well populated (>= a few thousand) for a 5% MC anchor.
    assert min(counts) >= 3000, f"under-populated bin: counts={counts}"

    pj = project_dispersion(prof, _PROJ_R_A, centers, _PROJ_M, G)

    assert jnp.allclose(pj.sigma_los, slos, rtol=0.05), (
        f"sigma_los: pred={pj.sigma_los} emp={slos} "
        f"R={centers} counts={counts}"
    )
    assert jnp.allclose(pj.sigma_pm_r, spmr, rtol=0.05), (
        f"sigma_pm,R: pred={pj.sigma_pm_r} emp={spmr} "
        f"R={centers} counts={counts}"
    )
    assert jnp.allclose(pj.sigma_pm_t, spmt, rtol=0.05), (
        f"sigma_pm,T: pred={pj.sigma_pm_t} emp={spmt} "
        f"R={centers} counts={counts}"
    )


@pytest.mark.slow
def test_projection_recovers_anisotropy():
    """beta-recovery: the projected observables CARRY the input radial bias, and
    project_dispersion predicts the SAME anisotropy signature (within MC tol).

    The OED's premise is that the on-sky PM/RV ratio encodes beta. With a
    radially-biased OM model (r_a=1.5) the on-sky tangential PM is suppressed
    relative to the LOS channel, GROWING outward as beta(r)=r^2/(r^2+r_a^2)
    builds. We assert, EMPIRICALLY (from the sampled population):

      1. sigma_pm,T < sigma_los in every well-populated bin (radial bias present),
      2. the ratio sigma_pm,T/sigma_los DECREASES outward (bias builds with R),
      3. project_dispersion reproduces that empirical ratio per bin (5% MC).

    (3) is the load-bearing claim: the analytic forward model and the sampled sky
    carry the SAME beta signature, so an OED inversion of the ratio recovers r_a.
    """
    prof, pos, vel = _sample_om_plummer_population(
        r_a=_PROJ_R_A, M=_PROJ_M, N=_PROJ_N, seed=7
    )
    R, v_los, v_pmR, v_pmT = _project_population(pos, vel)
    centers, counts, slos, spmr, spmt = _bin_empirical(
        R, v_los, v_pmR, v_pmT, _PROJ_BIN_EDGES
    )
    assert min(counts) >= 3000, f"under-populated bin: counts={counts}"

    ratio_emp = spmt / slos
    pj = project_dispersion(prof, _PROJ_R_A, centers, _PROJ_M, G)
    ratio_pred = pj.sigma_pm_t / pj.sigma_los

    # (1) radial-anisotropy signature: tangential PM suppressed below LOS.
    assert jnp.all(ratio_emp < 1.0), (
        f"expected sigma_pm,T < sigma_los (radial bias); ratio_emp={ratio_emp} "
        f"R={centers}"
    )
    # (2) the suppression DEEPENS outward (beta builds with radius).
    assert ratio_emp[-1] < ratio_emp[0], (
        f"anisotropy ratio should drop outward; ratio_emp={ratio_emp} R={centers}"
    )
    # (3) the analytic forward model carries the SAME beta signature (5% MC).
    assert jnp.allclose(ratio_emp, ratio_pred, rtol=0.05), (
        f"beta signature mismatch: emp={ratio_emp} pred={ratio_pred} "
        f"R={centers} counts={counts}"
    )


# =============================================================================
# Phase 0.5 Task D3 — Michie DF-moment (Tier B) all-radii anchor
# =============================================================================
#
# ``df_moment_dispersion`` (Tier B) is the EXACT second-moment dispersion of the
# Michie (1963) anisotropic DF: it integrates the DF's own velocity moments at
# each radius, so it is correct at ALL radii — including the far outskirts where
# the OM-Jeans approximation (``jeans_dispersion`` with an OM ``r_a``) is 6-8%
# wrong because the Michie-King anisotropy law diverges from the OM
# beta = r^2/(r^2+r_a^2). This multi-leg anchor proves that claim:
#
#   Leg 1 (headline) — Tier B vs the Michie SAMPLER at ALL radii (r in [0.5, 8]),
#     including the outer radii r=4,8 where OM-Jeans diverges. 5% MC tol.
#   Leg 2 — Tier-A / Tier-B consistency: a TRUE equilibrium satisfies the Jeans
#     equation, so feeding the DF's OWN beta(r) (from Tier B) into the general-beta
#     Jeans solver must reproduce Tier B's sigma_r (a pure numerical-consistency
#     check, 2% tol — NOT a fitted tolerance).
#   Leg 3 — beta(r) vs the established Michie beta oracle (the same 2-D moment
#     integral used in test_michie_physics.py), atol 1e-2.
#
# (Leg 4, a zeroth-moment / density self-consistency check, is INTENTIONALLY
# OMITTED: ``df_moment_dispersion`` returns only ``(sigma_r, sigma_t, sigma_1d,
# beta)`` and does NOT expose its internal velocity-space normalisation, so a
# zeroth-moment vs ``michie_density`` comparison would require either a contrived
# re-implementation of the integral or a src-side accessor. Density
# self-consistency is already covered by ``test_density_matches_king_at_large_ra``
# in test_michie_physics.py plus the all-radii sampler leg below, so no contrived
# test is added — per the Task D3 brief.)


def _michie_beta_oracle(W, s, n=400):
    """Analytic Michie anisotropy beta(W, s) from the 2nd moments of the EXACT
    sampled density p(u_r, u_t) ∝ u_t exp(-s^2 u_t^2/2)[exp(W-(u_r^2+u_t^2)/2)-1]
    on u_r^2+u_t^2 < 2W.

    This is the model's OWN beta (the lowering term breaks the pure f(Q) form, so
    it sits below the OM ceiling r^2/(r^2+r_a^2)). Replicated verbatim from
    test_michie_physics.py::_michie_beta_oracle (numpy helper, independent of the
    JAX df_moment_dispersion code path it validates).
    """
    if W <= 0:
        return 0.0
    umax = np.sqrt(2.0 * W)
    ur = np.linspace(-umax, umax, n)
    ut = np.linspace(0.0, umax, n)
    UR, UT = np.meshgrid(ur, ut)
    bound = UR**2 + UT**2 < 2.0 * W
    w = UT * np.exp(-(s**2) * UT**2 / 2.0) * (np.exp(W - (UR**2 + UT**2) / 2.0) - 1.0)
    w = np.where(bound, np.maximum(w, 0.0), 0.0)
    norm = w.sum()
    ur2 = (w * UR**2).sum() / norm
    ut2 = (w * UT**2).sum() / norm
    return 1.0 - ut2 / (2.0 * ur2)


@pytest.mark.slow
def test_df_moment_matches_sampler_all_radii():
    """HEADLINE: Tier B (DF-moment) vs the Michie sampler at ALL radii (incl. outer).

    Place N stars at fixed radii r0 on the x-axis, draw Michie velocities, and
    compare the empirical std of v_x (radial) and of the tangential plane to
    ``df_moment_dispersion(...).sigma_r/.sigma_t`` within 5% MC. The point is the
    OUTER radii r=4,8 where the OM-Jeans model is 6-8% wrong — the EXACT DF moment
    tracks the sampler there too (orchestrator measured 0.0-0.3% actual), so this is
    the test that KILLS the 'inner-region-only' limitation of OM-Jeans.
    """
    W0, r_c, r_a, M, N = 6.0, 1.0, 5.0, 400.0, 200_000
    df = MichieVelocityDF(W0=W0, r_c=r_c, r_a=r_a)
    for r0 in (0.5, 1.0, 2.0, 4.0, 8.0):  # outer radii where OM-Jeans diverged
        positions = jnp.zeros((N, 3)).at[:, 0].set(r0)
        masses = jnp.full((N,), M / N)
        key = jax.random.PRNGKey(int(r0 * 1000) + 7)
        v = df.sample_velocities(positions, masses, key, G=G)
        sr_emp = jnp.std(v[:, 0])
        st_emp = jnp.sqrt(0.5 * (jnp.var(v[:, 1]) + jnp.var(v[:, 2])))
        dp = df_moment_dispersion(df, jnp.array([r0]), M, G)
        assert jnp.allclose(dp.sigma_r[0], sr_emp, rtol=0.05), (
            f"Tier-B sigma_r r0={r0}: df_moment={float(dp.sigma_r[0]):.4f} "
            f"emp={float(sr_emp):.4f} rel={float(abs(dp.sigma_r[0]/sr_emp - 1)):.3e}"
        )
        assert jnp.allclose(dp.sigma_t[0], st_emp, rtol=0.05), (
            f"Tier-B sigma_t r0={r0}: df_moment={float(dp.sigma_t[0]):.4f} "
            f"emp={float(st_emp):.4f} rel={float(abs(dp.sigma_t[0]/st_emp - 1)):.3e}"
        )


@pytest.mark.slow
def test_tierA_jeans_consistent_with_tierB():
    """Tier-A / Tier-B consistency: a true equilibrium satisfies Jeans.

    Build the DF's OWN beta(r) from Tier B on a grid, feed it to the general-beta
    Jeans solver (Tier A), and assert the resulting sigma_r matches Tier B's sigma_r
    within rtol 0.02. This is a NUMERICAL-CONSISTENCY check (the integrating-factor
    Jeans solve vs the direct DF moment), not a fitted tolerance: a true equilibrium
    must satisfy its own Jeans equation with its own anisotropy. The actual max rel
    achieved is reported in the assertion message.
    """
    W0, r_c, r_a, M = 6.0, 1.0, 5.0, 400.0
    df = MichieVelocityDF(W0=W0, r_c=r_c, r_a=r_a)
    prof = MichieProfile.from_W0_rc(W0=W0, r_c=r_c, r_a=r_a)
    s = jnp.linspace(0.1, 0.9 * float(prof.r_t), 60)
    b = df_moment_dispersion(df, s, M, G).beta
    beta_fn = lambda rr: jnp.interp(rr, s, b)
    A = jeans_dispersion(prof, None, s, M, G, beta_fn=beta_fn).sigma_r
    B = df_moment_dispersion(df, s, M, G).sigma_r
    max_rel = float(jnp.max(jnp.abs(A / B - 1.0)))
    assert max_rel < 0.02, (
        f"Tier-A (Jeans w/ DF's own beta) vs Tier-B sigma_r max rel {max_rel:.3e} "
        f">= 0.02 (numerical-consistency tol — investigate, do NOT loosen blindly)"
    )


def test_df_moment_beta_matches_michie_oracle():
    """Tier B beta(r) vs the established Michie beta oracle (atol 1e-2).

    The 2-D moment integral ``_michie_beta_oracle(W(r), r/r_a)`` (numpy, the same
    helper that anchors test_michie_physics.py) is an INDEPENDENT computation of the
    model's own anisotropy from the EXACT sampled DF density. ``df_moment_dispersion``
    must reproduce it (orchestrator measured 1.5e-4). W(r) is read off the stored
    Michie ODE solution, interp at r/r_c on xi_grid/psi_grid, clamped >= 0.
    """
    W0, r_c, r_a, M = 6.0, 1.0, 5.0, 400.0
    df = MichieVelocityDF(W0=W0, r_c=r_c, r_a=r_a)
    radii = (0.5, 1.0, 2.0, 4.0, 8.0)
    max_dev = 0.0
    for r0 in radii:
        W = jnp.interp(r0 / r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)
        W = float(jnp.maximum(W, 0.0))
        oracle = float(_michie_beta_oracle(W, r0 / r_a))
        beta = float(df_moment_dispersion(df, jnp.array([r0]), M, G).beta[0])
        dev = abs(beta - oracle)
        max_dev = max(max_dev, dev)
        assert dev < 1e-2, (
            f"Tier-B beta r0={r0}: df_moment={beta:.5f} oracle={oracle:.5f} "
            f"dev={dev:.3e} >= 1e-2"
        )
