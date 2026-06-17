"""Unit tests for the differentiable dispersion forward models.

Phase 0 Task 1: scaffold — exports present + NamedTuple field layout.
Phase 0 Task 2: jeans_dispersion 3-D — isotropic closed form, GM scaling
invariants, r_a domain guard, jit smoke.
"""

import jax
import jax.numpy as jnp
import pytest
import progenax
from progenax import jeans_dispersion, project_dispersion
from progenax.kinematics.dispersion import DispersionProfile, ProjectedDispersion
from progenax.profiles import PlummerProfile, EFFProfile

G_STELLAR = 0.00449


def _assert_ad_fd(f, theta0, *, h=None, name=""):
    """AD-vs-FD differentiability gate for a scalar-valued ``f(theta)``.

    Asserts the reverse-mode gradient matches a central finite difference
    (FD-consistent, rel-error < 1e-3) AND is not a silent/blocked zero
    (|g_ad| > 1e-9). The OED Fisher rides entirely on these gradients, so a
    silent zero (interp edge-clamp, ``where`` dead branch, sqrt-at-0 NaN
    masked to 0) is a HARD failure here, not a tolerance to relax.
    """
    theta0 = float(theta0)
    if h is None:
        h = 1e-4 * abs(theta0) if theta0 != 0.0 else 1e-4
    g_ad = float(jax.grad(f)(theta0))
    g_fd = float((f(theta0 + h) - f(theta0 - h)) / (2.0 * h))
    rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-12)
    assert rel < 1e-3, f"{name}: AD-vs-FD inconsistent g_ad={g_ad!r} g_fd={g_fd!r} rel={rel!r}"
    assert abs(g_ad) > 1e-9, f"{name}: silent/blocked-zero gradient g_ad={g_ad!r}"
    return g_ad, g_fd, rel


class TestPchipInterp:
    def test_passes_through_nodes(self):
        x = jnp.linspace(0.5, 4.0, 12); y = jnp.sin(x)
        from progenax.kinematics.dispersion import _pchip_interp
        assert jnp.allclose(_pchip_interp(x, x, y), y, atol=1e-12)

    def test_exact_on_linear(self):
        from progenax.kinematics.dispersion import _pchip_interp
        x = jnp.linspace(0.0, 10.0, 9); y = 3.0 * x - 1.0
        xq = jnp.linspace(0.3, 9.7, 50)
        assert jnp.allclose(_pchip_interp(xq, x, y), 3.0 * xq - 1.0, atol=1e-10)

    def test_c1_no_slope_jump_across_node(self):
        # central FD of the interpolant straddling an interior node must agree
        # left vs right (C1); linear interp would show a jump here.
        from progenax.kinematics.dispersion import _pchip_interp
        x = jnp.linspace(0.0, 6.0, 13); y = jnp.exp(-x)        # smooth, monotone
        node = x[6]; e = 1e-3
        d_left  = (_pchip_interp(jnp.array([node - e]), x, y) - _pchip_interp(jnp.array([node - 2*e]), x, y)) / e
        d_right = (_pchip_interp(jnp.array([node + 2*e]), x, y) - _pchip_interp(jnp.array([node + e]), x, y)) / e
        assert abs(float(d_left[0] - d_right[0])) < 1e-2 * abs(float(d_right[0]))

    def test_differentiable_in_data(self):
        from progenax.kinematics.dispersion import _pchip_interp
        x = jnp.linspace(0.5, 4.0, 10)
        def loss(scale):
            return jnp.sum(_pchip_interp(jnp.array([1.3, 2.7]), x, scale * jnp.cos(x)))
        g = float(jax.grad(loss)(1.0)); assert abs(g) > 1e-9 and jnp.isfinite(g)

    def test_nan_safe_grad_on_degenerate_data(self):
        # Non-monotone data with a local max (sign-changing secants -> the
        # Fritsch-Carlson `same` guard hits its False branch) AND a flat run
        # (zero secants), exercising the double-`where` NaN guard. jax.grad must
        # stay finite (a naive 1/0 in the harmonic-mean slope would NaN here).
        from progenax.kinematics.dispersion import _pchip_interp
        x = jnp.linspace(0.0, 6.0, 13)
        base = jnp.array([0.,1.,2.,3.,2.,1.,1.,1.,1.,2.,3.,4.,5.])
        def loss(c):
            return jnp.sum(_pchip_interp(jnp.array([1.5, 3.0, 4.5]), x, c * base))
        g = float(jax.grad(loss)(1.0)); assert jnp.isfinite(g)


def test_exports_and_namedtuples():
    assert {"jeans_dispersion", "project_dispersion"} <= set(progenax.__all__)
    assert DispersionProfile._fields == ("r", "sigma_r", "sigma_t", "sigma_1d", "beta")
    assert ProjectedDispersion._fields == ("R", "sigma_los", "sigma_pm_r", "sigma_pm_t", "Sigma")


def test_plummer_isotropic_closed_form():
    prof = PlummerProfile(r_h=1.0)
    r = jnp.array([0.3, 0.7, 1.0, 2.0])
    dp = jeans_dispersion(prof, None, r, M=400.0, G=0.00449)
    truth = jnp.sqrt(0.00449 * 400.0 / (6.0 * jnp.sqrt(r**2 + prof.a**2)))
    assert jnp.allclose(dp.sigma_1d, truth, rtol=3e-3)
    assert jnp.allclose(dp.beta, 0.0, atol=1e-10)
    assert jnp.allclose(dp.sigma_r, dp.sigma_t, rtol=1e-6)


def test_gm_scaling_invariants():
    prof = PlummerProfile(r_h=1.0)
    r = jnp.array([1.0])
    s1 = jeans_dispersion(prof, 2.0, r, 400.0, 0.00449).sigma_r
    assert jnp.allclose(
        jeans_dispersion(prof, 2.0, r, 800.0, 0.00449).sigma_r**2, 2 * s1**2, rtol=1e-4
    )
    assert jnp.allclose(
        jeans_dispersion(prof, 2.0, r, 400.0, 2 * 0.00449).sigma_r**2, 2 * s1**2, rtol=1e-4
    )


def test_r_a_domain_guard():
    import pytest

    prof = PlummerProfile(r_h=1.0)
    with pytest.raises(ValueError):  # r_a < 0.75 a is unphysical for Plummer OM
        jeans_dispersion(prof, 0.1 * prof.a, jnp.array([1.0]), 400.0, 0.00449)


def test_jit_smoke():
    prof = PlummerProfile(r_h=1.0)
    f = jax.jit(lambda ra: jeans_dispersion(prof, ra, jnp.array([1.0]), 400.0, 0.00449).sigma_r)
    assert jnp.isfinite(f(2.0)).all()


# --- Task 7: AD-vs-FD differentiability gate (the OED Fisher rides on these) ---
#
# Query radii r/R in [0.5, 2.0] sit well inside the Plummer grid extent
# (r_max = 30 a ~= 23 pc for r_h=1.0) and away from the inner edge (s_min =
# 1e-4 r_max), so the jnp.interp edge-clamp and the I(r_max)=0 endpoint zero
# (the known silent-zero hazards) are NOT probed — we test the INTERIOR
# gradient, which must be FD-consistent and non-zero.

R_QUERY = jnp.array([0.5, 1.0, 2.0])


def test_grad_jeans_dispersion_wrt_r_a():
    """1. d(sum sigma_r)/d(r_a) — Plummer Osipkov-Merritt."""
    prof = PlummerProfile(r_h=1.0)

    def f(r_a):
        return jnp.sum(jeans_dispersion(prof, r_a, R_QUERY, 400.0, G_STELLAR).sigma_r)

    _assert_ad_fd(f, 2.0, name="jeans sigma_r / r_a")


def test_grad_jeans_dispersion_wrt_M():
    """2. d(sum sigma_r)/d(M) — Plummer OM."""
    prof = PlummerProfile(r_h=1.0)

    def f(M):
        return jnp.sum(jeans_dispersion(prof, 2.0, R_QUERY, M, G_STELLAR).sigma_r)

    _assert_ad_fd(f, 400.0, name="jeans sigma_r / M")


def test_grad_jeans_dispersion_wrt_r_h():
    """3. d(sum sigma_r)/d(r_h) — through the Plummer profile param."""

    def f(r_h):
        prof = PlummerProfile(r_h=r_h)
        return jnp.sum(jeans_dispersion(prof, 2.0, R_QUERY, 400.0, G_STELLAR).sigma_r)

    _assert_ad_fd(f, 1.0, name="jeans sigma_r / r_h")


def test_grad_jeans_dispersion_wrt_gamma():
    """4. d(sum sigma_r)/d(gamma) — through EFFProfile (isotropic, mild trunc)."""
    # a=1.0, r_t=30.0 (wide truncation); query radii [0.5,2.0] well inside.
    # Isotropic (r_a=None) so no OM validity domain to worry about for EFF.
    def f(gamma):
        prof = EFFProfile(a=1.0, gamma=gamma, r_t=30.0)
        return jnp.sum(jeans_dispersion(prof, None, R_QUERY, 400.0, G_STELLAR).sigma_r)

    _assert_ad_fd(f, 5.0, name="jeans sigma_r / gamma (EFF)")


def test_grad_project_sigma_los_wrt_r_a():
    """5. d(sum sigma_los)/d(r_a) — Plummer OM (RV channel)."""
    prof = PlummerProfile(r_h=1.0)

    def f(r_a):
        return jnp.sum(project_dispersion(prof, r_a, R_QUERY, 400.0, G_STELLAR).sigma_los)

    _assert_ad_fd(f, 2.0, name="project sigma_los / r_a")


def test_grad_project_sigma_los_wrt_M():
    """6. d(sum sigma_los)/d(M) — Plummer OM."""
    prof = PlummerProfile(r_h=1.0)

    def f(M):
        return jnp.sum(project_dispersion(prof, 2.0, R_QUERY, M, G_STELLAR).sigma_los)

    _assert_ad_fd(f, 400.0, name="project sigma_los / M")


def test_grad_project_sigma_pm_t_wrt_r_a():
    """7. d(sum sigma_pm_t)/d(r_a) — Plummer OM (the beta-carrying, OED-critical grad)."""
    prof = PlummerProfile(r_h=1.0)

    def f(r_a):
        return jnp.sum(project_dispersion(prof, r_a, R_QUERY, 400.0, G_STELLAR).sigma_pm_t)

    _assert_ad_fd(f, 2.0, name="project sigma_pm_t / r_a")


# --- Phase 0.5 Task B: regression-gate profile-parameter dispersion gradients ---
#
# EFF (prescribed r_t/gamma, no ODE solve) gradients are CLEAN (measured rel
# ~1e-8). The SOLVED-equilibrium W0 gradients (King/Michie, both via
# solve_*_profile + _find_tidal_radius) are measured honestly here:
#   - King W0:  rel ~9e-5 at the helper's default FD step; AD->FD CONVERGES as
#     h shrinks (1.4e-3 @ h=1e-2 -> 2e-5 @ h=1e-4) => genuinely CLEAN, gated.
#   - Michie W0: rel ~3.5e-4, CLEAN gate. The former ~5e-3 inconsistency was NOT an
#     ODE-solver defect (earlier mis-attribution) but the C⁰ jnp.interp back-interp
#     in _sigma_r2_from_tables: as r_t(W0) moved the master s-grid nodes, the
#     piecewise-linear bracket switched and kinked ∂σ/∂W0. The C¹ PCHIP back-interp
#     (ADR-0016) removes the slope-jump => gate now clean. (Beyond W0≈7 at r_a=5 the
#     Michie model nears its mass-divergence, r_t->∞; the gradient stays correct but
#     a fixed-step FD is a poor truth-proxy there — see the high-W0 Richardson test.)
# Both King and Michie share solve_*/_find_tidal_radius and are FD-consistent here.


def test_grad_jeans_eff_wrt_r_t():
    """EFF sigma_r gradient w.r.t. truncation radius r_t (prescribed, no ODE; clean)."""

    def f(r_t):
        return jnp.sum(
            jeans_dispersion(
                EFFProfile(a=1.0, gamma=4.0, r_t=r_t), None, jnp.array([1.0]), 400.0, G_STELLAR
            ).sigma_r
        )

    _assert_ad_fd(f, 8.0, name="jeans EFF sigma_r / r_t")  # measured rel ~4.8e-8


def test_grad_jeans_eff_wrt_gamma():
    """EFF sigma_r gradient w.r.t. outer slope gamma (prescribed, no ODE; clean)."""

    def f(g):
        return jnp.sum(
            jeans_dispersion(
                EFFProfile(a=1.0, gamma=g, r_t=8.0), None, jnp.array([1.0]), 400.0, G_STELLAR
            ).sigma_r
        )

    _assert_ad_fd(f, 4.0, name="jeans EFF sigma_r / gamma")  # measured rel ~5.7e-9


def test_grad_jeans_king_wrt_W0():
    """King sigma_r gradient w.r.t. W0 (solved equilibrium; FD-consistent, gated clean).

    Despite King sharing solve_king_profile + _find_tidal_radius with the
    FD-inconsistent Michie path, the King W0 gradient CONVERGES to FD as the
    step shrinks (rel ~9e-5 at the default step) -- a genuine clean gate, not
    the deferred limitation.
    """
    from progenax.profiles import KingProfile

    def f(W0):
        return jnp.sum(
            jeans_dispersion(
                KingProfile.from_W0_rc(W0=W0, r_c=1.0), None, jnp.array([1.0]), 400.0, G_STELLAR
            ).sigma_r
        )

    _assert_ad_fd(f, 6.0, name="jeans King sigma_r / W0")  # measured rel ~9.1e-5


def test_grad_jeans_michie_wrt_W0():
    """Michie sigma_r gradient w.r.t. W0 (solved equilibrium; clean gate).

    The C¹ PCHIP back-interpolation in _sigma_r2_from_tables (ADR-0016) removes
    the bracket-crossing slope-jump that the old C⁰ jnp.interp injected into
    ∂σ/∂W0 as r_t(W0) moved the s-grid nodes, so this AD-vs-FD gate is now clean
    (was a deferred xfail at ~5e-3).
    """
    from progenax.profiles import MichieProfile

    def f(W0):
        return jnp.sum(
            jeans_dispersion(
                MichieProfile.from_W0_rc(W0=W0, r_c=1.0, r_a=5.0),
                None,
                jnp.array([1.0]),
                400.0,
                G_STELLAR,
            ).sigma_r
        )

    _assert_ad_fd(f, 6.0, name="jeans Michie sigma_r / W0")  # measured rel ~3.5e-4 < 1e-3


def test_grad_jeans_michie_high_W0_ad_correct():
    """High-W0 Michie gradient is AD-correct; the coarse-FD disagreement is an FD artifact.

    At W0=7, r_a=5 the Michie model is near its mass-divergence (r_t ~ 545; no finite
    truncation past W0~7.1), so sigma_r(W0) has near-singular curvature and a single
    coarse central FD is an unreliable truth-proxy. This pins that the reverse-mode AD
    gradient is nonetheless CORRECT: a central FD CONVERGES to AD as the step shrinks
    (rel ~3e-3 @ h=1e-3 -> ~2e-5 @ h=1e-6), so the coarse-step inconsistency is the FD's
    own O(h^2 f''') truncation error, not a gradient defect. The GATED Michie-W0 test
    (test_grad_jeans_michie_wrt_W0) runs in the well-truncated W0=6 regime; see ADR-0016
    and the jeans_dispersion docstring.
    """
    from progenax.profiles import MichieProfile

    def f(W0):
        return jnp.sum(
            jeans_dispersion(
                MichieProfile.from_W0_rc(W0=W0, r_c=1.0, r_a=5.0),
                None,
                jnp.array([1.0]),
                400.0,
                G_STELLAR,
            ).sigma_r
        )

    W0 = 7.0
    g_ad = float(jax.grad(f)(W0))

    def rel(h):
        g_fd = float((f(W0 + h) - f(W0 - h)) / (2.0 * h))
        return abs(g_ad - g_fd) / (abs(g_fd) + 1e-12)

    rel_coarse, rel_fine = rel(1e-3), rel(1e-6)
    # AD matches a converged (fine-step) FD ...
    assert rel_fine < 1e-3, f"AD vs fine-FD rel {rel_fine:.2e} not < 1e-3 at W0=7"
    # ... and the FD CONVERGES toward AD as h shrinks (the coarse-h gap is the FD artifact).
    assert rel_fine < rel_coarse, (
        f"FD did not converge to AD as h shrank: coarse(1e-3)={rel_coarse:.2e}, "
        f"fine(1e-6)={rel_fine:.2e} — would indicate a real gradient defect, not an FD artifact"
    )


# --- Phase 0.5 Task A: tabulate-once project_dispersion equivalence regression ---
#
# Baseline pins the EXACT project_dispersion output for PlummerProfile(r_h=1.0),
# r_a=2.0, R=[0.5,1.0,2.0,4.0], M=400.0, G=0.00449, to rtol 1e-9 — a mismatch means
# a structural change silently altered the physics. NEVER loosen this tolerance.
#
# Task C compactification intentionally shifted Plummer values toward the analytic
# oracle; baseline re-captured TWICE. (1) Task C compactified the master-Jeans
# s-grid (s = a t/(1-t)). (2) Task C (cont.) ALSO compactified project_dispersion's
# OWN outward u-quadrature (u = u_c tau/(1-tau)), which had still truncated the
# Plummer tail at u_max = sqrt((30a)^2 - R^2) and left an n_u-INDEPENDENT
# truncation floor of 1.634e-4 (rel.) in the projected sigma_los at the outer
# radius R=4a. PROOF each re-capture is an improvement (isotropic Dejonghe oracle
# (3pi/64) GM/sqrt(a^2+R^2), same R): the isotropic sigma_los's max relative error
# to the oracle fell 8.58e-4 (pre-Task-C, the master-Jeans truncation floor) ->
# 1.634e-4 (after the master-Jeans compactification, now limited by the projection
# u-truncation) -> 7.10e-6 (after compactifying the projection u-grid too),
# i.e. ~23x closer at the worst R. So each re-captured baseline is MORE accurate,
# not a regression.
# (3) 2026-06-17: gradient-motivated re-capture. _sigma_r2_from_tables now uses a
# C¹ PCHIP back-interp (ADR-0016) instead of C⁰ jnp.interp, to remove the
# bracket-crossing slope-jump in ∂σ/∂W0 (see test_grad_jeans_michie_wrt_W0). Shared
# by project_dispersion, this shifts the pinned values ~1.6e-6 (rel) — above the
# 1e-9 pin, so re-captured here. It ALSO improves oracle accuracy: the isotropic
# sigma_los max relative error to the Dejonghe oracle falls 7.10e-6 -> 5.16e-6
# (uniformly closer at every R), so this re-capture is again an improvement, not a
# regression. Tolerance UNCHANGED at rtol=1e-9.
_BL_LOS = jnp.array(
    [0.5526179475472416, 0.45025550482131277, 0.3009530819950546, 0.17215507175857458]
)
_BL_PMT = jnp.array(
    [0.5368408461106755, 0.42825244807500384, 0.2671191845762194, 0.1255501505502787]
)


def test_project_equivalence_after_tabulate():
    """project_dispersion output is pinned (re-captured after Task C compactification)."""
    prof = PlummerProfile(r_h=1.0)
    R = jnp.array([0.5, 1.0, 2.0, 4.0])
    pj = project_dispersion(prof, 2.0, R, 400.0, G_STELLAR)
    assert jnp.allclose(pj.sigma_los, _BL_LOS, rtol=1e-9)
    assert jnp.allclose(pj.sigma_pm_t, _BL_PMT, rtol=1e-9)


# --- Phase 0.5 Task D1: general-beta jeans_dispersion (Tier A) ---
#
# A beta_fn callable lets jeans_dispersion use an ARBITRARY anisotropy beta(r)
# via the general integrating factor f(r)=exp(2 int beta(s)/s ds). The OM/
# isotropic default (beta_fn=None) is bit-preserved. The OM-reduction test below
# is THE correctness proof: a beta_fn reproducing the OM law must reproduce the
# analytic OM sigma_r (the numerical integrating factor reduces to f=r^2+r_a^2).


def test_beta_fn_om_equals_default():
    """beta_fn reproducing OM matches the analytic OM default (numerical f -> OM)."""
    prof = PlummerProfile(r_h=1.0)
    r = jnp.array([0.5, 1.0, 2.0])
    r_a = 2.0
    default = jeans_dispersion(prof, r_a, r, 400.0, G_STELLAR)
    beta_om = lambda rr: rr**2 / (rr**2 + r_a**2)
    general = jeans_dispersion(prof, None, r, 400.0, G_STELLAR, beta_fn=beta_om)
    assert jnp.allclose(default.sigma_r, general.sigma_r, rtol=1e-4)
    assert jnp.allclose(general.beta, beta_om(r), rtol=1e-6)


def test_grad_jeans_beta_fn_wrt_M():
    """AD-vs-FD clean for d(sum sigma_r)/d(M) on the beta_fn path (OM beta_fn)."""
    beta_om = lambda rr: rr**2 / (rr**2 + 4.0)

    def f(M):
        return jnp.sum(
            jeans_dispersion(
                PlummerProfile(r_h=1.0), None, jnp.array([1.0]), M, G_STELLAR, beta_fn=beta_om
            ).sigma_r
        )

    _assert_ad_fd(f, 400.0, name="general-beta jeans / M")


def test_beta_fn_constant_sigma_ratio():
    """Non-OM beta_fn anchor: a CONSTANT beta(r)=0.5 -> sigma_t/sigma_r == sqrt(1-0.5).

    The D1 general-beta path accepts an ARBITRARY anisotropy law, not just the OM
    family. With a constant beta the Jeans tangential relation
    ``sigma_t^2 = (1 - beta) sigma_r^2`` is exact and radius-independent, so
    ``sigma_t/sigma_r = sqrt(1 - beta) = 1/sqrt(2)`` at EVERY radius. This pins the
    genuinely-new arbitrary-beta capability at the unit tier, independent of D3's
    Michie validation legs (which exercise the OM-like / true-Michie laws).
    """
    prof = PlummerProfile(r_h=1.0)
    r = jnp.array([0.5, 1.0, 2.0, 4.0])
    beta_const = lambda rr: 0.5 * jnp.ones_like(rr)
    dp = jeans_dispersion(prof, None, r, 400.0, G_STELLAR, beta_fn=beta_const)
    assert jnp.allclose(dp.sigma_t / dp.sigma_r, 1.0 / jnp.sqrt(2.0), rtol=1e-6)
    assert jnp.allclose(dp.beta, 0.5, rtol=1e-6)


def test_jit_both_forward_models():
    """jax.jit compiles both forward models and returns finite arrays (OED jits these)."""
    prof = PlummerProfile(r_h=1.0)

    jeans_jit = jax.jit(
        lambda r_a: jeans_dispersion(prof, r_a, R_QUERY, 400.0, G_STELLAR).sigma_r
    )
    proj_jit = jax.jit(
        lambda r_a: project_dispersion(prof, r_a, R_QUERY, 400.0, G_STELLAR).sigma_los
    )

    sr = jeans_jit(2.0)
    slos = proj_jit(2.0)
    assert jnp.isfinite(sr).all() and sr.shape == R_QUERY.shape
    assert jnp.isfinite(slos).all() and slos.shape == R_QUERY.shape


# ---------------------------------------------------------------------------
# Task D2 — df_moment_dispersion (Tier B exact Michie DF second moment)
# ---------------------------------------------------------------------------


def test_df_moment_export_and_shapes():
    """df_moment_dispersion is exported; returns finite, positive, in-range fields."""
    from progenax import df_moment_dispersion
    from progenax.kinematics import MichieVelocityDF

    assert "df_moment_dispersion" in set(progenax.__all__)
    df = MichieVelocityDF(W0=6.0, r_c=1.0, r_a=5.0)
    dp = df_moment_dispersion(df, jnp.array([0.5, 1.0, 2.0]), 400.0, G_STELLAR)
    assert dp.sigma_r.shape == (3,)
    assert jnp.all(dp.sigma_r > 0)
    assert jnp.all(dp.sigma_t > 0)
    assert jnp.all(jnp.isfinite(dp.beta))
    assert jnp.all(dp.beta >= -1.0) and jnp.all(dp.beta < 1.0)


def test_df_moment_isotropic_limit_beta_near_zero():
    """Large r_a -> isotropic Michie/King limit -> beta ~ 0 at interior radii."""
    from progenax import df_moment_dispersion
    from progenax.kinematics import MichieVelocityDF

    df = MichieVelocityDF(W0=6.0, r_c=1.0, r_a=1e4)
    dp = df_moment_dispersion(df, jnp.array([0.5, 1.0, 2.0]), 400.0, G_STELLAR)
    assert jnp.allclose(dp.beta, 0.0, atol=5e-2)


def test_df_moment_grad_finite_beyond_r_t():
    """grad of df_moment_dispersion is finite at/beyond r_t (NaN-safe outer sqrt).

    Regression for the Phase 0.5 D2 review defect: the outer
    ``sqrt(maximum(sigma_r2, 0))`` had derivative ``1/(2 sqrt(0)) = inf`` when
    ``sigma_r2 == 0`` exactly (r >= r_t, clamped W -> 0), giving ``inf * 0 = NaN``
    on the backward pass. A single beyond-r_t point in a jacrev/grad over a
    radial grid then poisoned the ENTIRE OED Fisher result to NaN. The forward
    value at r >= r_t is a clean 0.0 and must stay exactly 0.0.
    """
    from progenax import df_moment_dispersion
    from progenax.kinematics import MichieVelocityDF

    df = MichieVelocityDF(W0=6.0, r_c=1.0, r_a=5.0)  # r_t ~ 27.89

    # Forward sigma_r at r=30 (> r_t) is exactly 0.0.
    dp = df_moment_dispersion(df, jnp.array([30.0]), 400.0, G_STELLAR)
    assert dp.sigma_r[0] == 0.0

    # (a) grad of a single beyond-r_t point wrt M is finite.
    g_single = jax.grad(
        lambda M: df_moment_dispersion(df, jnp.array([30.0]), M, G_STELLAR).sigma_r[0]
    )(400.0)
    assert jnp.isfinite(g_single)

    # (b) grad over a radial grid that SPANS r_t (the Fisher case) is finite.
    g_grid = jax.grad(
        lambda M: jnp.sum(
            df_moment_dispersion(
                df, jnp.array([1.0, 5.0, 15.0, 30.0]), M, G_STELLAR
            ).sigma_r
        )
    )(400.0)
    assert jnp.isfinite(g_grid)


# ---------------------------------------------------------------------------
# Phase 0.5 final-review — generalize the NaN-safe outer sqrt to jeans + project.
#
# The SAME sqrt(0) -> inf-derivative defect that D2 fixed in df_moment_dispersion
# ALSO lived in jeans_dispersion (_sigma_components: bare sqrt(sigma_r2) and
# sqrt((sigma_r2 + 2 sigma_t2)/3)) and project_dispersion (sqrt(maximum(S/Sigma,
# 0)) — maximum(x, 0) does NOT fix the sqrt-at-0 gradient: 1/(2 sqrt(0)) = inf,
# inf * 0 = NaN). For finite-r_t profiles (King/Michie/EFF), differentiating
# either model wrt a PROFILE parameter over a radial grid that includes points
# r/R >= r_t returned a NaN gradient that poisoned the whole Fisher. Forward
# values at r/R >= r_t are an exact 0.0; only the gradient was broken.
# ---------------------------------------------------------------------------


def test_grad_jeans_finite_rt_grad_finite_beyond_r_t():
    """jeans_dispersion grad is finite at/beyond r_t for finite-r_t profiles.

    Regression for the Phase 0.5 final-review defect: _sigma_components used a
    bare ``jnp.sqrt(sigma_r2)`` whose derivative is ``1/(2 sqrt(0)) = inf`` where
    ``sigma_r2 == 0`` exactly (r >= r_t), giving ``inf * 0 = NaN`` on the backward
    pass. One beyond-r_t point in a grad/jacrev over a radial grid then poisoned
    the whole result to NaN. Forward sigma_r at r >= r_t must stay exactly 0.0.
    """
    from progenax.profiles import KingProfile

    # EFF, r_t = 8.0 (a constructor arg). Forward sigma_r at r=10 (> r_t) is 0.0.
    prof = EFFProfile(a=1.0, gamma=4.0, r_t=8.0)
    dp = jeans_dispersion(prof, None, jnp.array([10.0]), 400.0, G_STELLAR)
    assert dp.sigma_r[0] == 0.0

    # (a) grad of a single beyond-r_t point wrt M is finite.
    g_single = jax.grad(
        lambda M: jeans_dispersion(prof, None, jnp.array([10.0]), M, G_STELLAR).sigma_r[0]
    )(400.0)
    assert jnp.isfinite(g_single)

    # (b) grad over an EFF grid SPANNING r_t wrt gamma (rebuild profile inside) is finite.
    def f_eff_gamma(gamma):
        p = EFFProfile(a=1.0, gamma=gamma, r_t=8.0)
        return jnp.sum(jeans_dispersion(p, None, jnp.array([1.0, 5.0, 10.0]), 400.0, G_STELLAR).sigma_r)

    assert jnp.isfinite(jax.grad(f_eff_gamma)(4.0))

    # (c) King: grad over a grid SPANNING r_t wrt r_c (rebuild profile inside) is finite.
    def f_king_rc(r_c):
        p = KingProfile.from_W0_rc(W0=6.0, r_c=r_c)  # r_t ~ a few r_c
        return jnp.sum(jeans_dispersion(p, None, jnp.array([1.0, 5.0, 50.0]), 400.0, G_STELLAR).sigma_r)

    assert jnp.isfinite(jax.grad(f_king_rc)(1.0))


def test_grad_project_finite_rt_grad_finite_beyond_r_t():
    """project_dispersion grad is finite at/beyond r_t for finite-r_t profiles.

    Regression for the Phase 0.5 final-review defect: project_dispersion built
    sigma_los/pm via ``sqrt(maximum(S/Sigma, 0))`` — ``maximum(x, 0)`` clamps the
    VALUE but NOT the sqrt-at-0 gradient (1/(2 sqrt(0)) = inf, inf * 0 = NaN), so
    a grad/jacrev over an on-sky grid spanning r_t returned NaN. _safe_sqrt fixes
    both the value (exact 0) and the gradient (finite 0) at the argument's 0.
    """
    prof = EFFProfile(a=1.0, gamma=5.0, r_t=8.0)

    def f_eff_a(a):
        p = EFFProfile(a=a, gamma=5.0, r_t=8.0)
        return jnp.sum(project_dispersion(p, None, jnp.array([2.0, 6.0, 10.0]), 400.0, G_STELLAR).sigma_los)

    assert jnp.isfinite(jax.grad(f_eff_a)(1.0))


def test_grad_project_king_rc_finite_beyond_r_t():
    """project_dispersion grad is finite when r_t MOVES with the differentiated param.

    Second regression for the Phase 0.5 final-review defect: the EFF case above keeps
    r_t fixed (constructor arg), so it does NOT exercise the ``u_max = sqrt(r_edge^2 -
    R^2)`` endpoint, whose r_edge moves with King r_c/W0. Differentiating w.r.t. King
    r_c over an on-sky grid spanning r_t hit that second sqrt-at-0 (inf*0=NaN) until
    u_max was switched to _safe_sqrt. The jacrev of the sigma_los VECTOR (the OED Fisher
    pattern) must be all-finite, with the beyond-r_t bin carrying a clean 0.
    """
    from progenax.profiles import KingProfile

    def f_king_rc(r_c):
        p = KingProfile.from_W0_rc(W0=6.0, r_c=r_c)
        return project_dispersion(p, None, jnp.array([2.0, 6.0, 40.0]), 400.0, G_STELLAR).sigma_los

    jac = jax.jacrev(lambda rc: jnp.sum(f_king_rc(rc)))(1.0)
    assert jnp.isfinite(jac)
    vec_jac = jax.jacrev(f_king_rc)(1.0)
    assert jnp.all(jnp.isfinite(vec_jac))
