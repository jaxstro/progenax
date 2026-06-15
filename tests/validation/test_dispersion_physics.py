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
5. ``test_plummer_om_jeans_vs_analytic`` — DEFERRED/SKIPPED: no source-verified
   closed-form OM-Plummer sigma_r oracle is encoded (honesty over coverage). OM is
   validated by the sampler (4) + the convergence study (2).

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
import pytest

from progenax.profiles import PlummerProfile, EFFProfile, MichieProfile
from progenax.kinematics import (
    PlummerVelocityDF,
    EFFVelocityDF,
    MichieVelocityDF,
    jeans_dispersion,
)
from progenax.kinematics.dispersion import jeans_sigma_r, ftable_sigma_r_isotropic
from progenax.kinematics.eff_df import _eff_eddington_table

G = 0.00449


def _plummer_isotropic_sigma_r2_analytic(r, M, a):
    """EXACT isotropic Plummer radial dispersion sigma_r^2(r) = GM/(6 sqrt(r^2+a^2)).

    Standard closed form (Plummer 1911; Binney & Tremaine 2008). For the isotropic
    model sigma_r = sigma_t = sigma_1d, so this is also sigma_1d^2.
    """
    return G * M / (6.0 * jnp.sqrt(r**2 + a**2))


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

    SELF-CONVERGENCE (Cauchy) study, NOT error-vs-analytic. Subtlety (root-caused,
    not papered over): ``jeans_dispersion`` truncates the outward Jeans integral at
    ``r_max = 30 a`` (Plummer has no finite cutoff), which leaves a FIXED,
    n_s-INDEPENDENT tail bias of ~4e-4 (rel.) in sigma_r at r=1. That truncation
    floor swamps the trapezoid error once h is small, so |sigma(n) - analytic|
    plateaus and its ratio is NOT 4. The honest order-of-accuracy probe is the
    self-difference ``d_n = |sigma(n) - sigma(2n)|``: the fixed tail bias CANCELS in
    the difference, isolating the pure O(h^2) discretisation term, so ``d_n`` falls
    ~4x per doubling. (numerical-method-validation: self-convergence when the cheap
    oracle carries a separate, non-vanishing modelling/truncation error.)

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


@pytest.mark.skip(
    reason="OM-Plummer analytic oracle deferred: not source-verified. The closed-form "
    "OM-Plummer sigma_r^2(r) (Carollo et al. 1995 / Merritt 1985-type reduction) was "
    "not encoded because it could not be independently verified against a primary "
    "source in-session; fabricating it would violate the no-guessed-formula rule. OM "
    "is validated via the empirical sampler (test_plummer_om_jeans_matches_sampler) "
    "and the O(h^2) convergence study (test_jeans_quadrature_convergence)."
)
def test_plummer_om_jeans_vs_analytic():
    """DEFERRED — see skip reason. Placeholder for a future source-verified OM oracle."""
    raise NotImplementedError


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
