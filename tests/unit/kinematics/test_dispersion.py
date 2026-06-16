"""Unit tests for the differentiable dispersion forward models.

Phase 0 Task 1: scaffold — exports present + NamedTuple field layout.
Phase 0 Task 2: jeans_dispersion 3-D — isotropic closed form, GM scaling
invariants, r_a domain guard, jit smoke.
"""

import jax
import jax.numpy as jnp
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


def test_grad_jeans_sigma_r_wrt_r_a():
    """1. d(sum sigma_r)/d(r_a) — Plummer Osipkov-Merritt."""
    prof = PlummerProfile(r_h=1.0)

    def f(r_a):
        return jnp.sum(jeans_dispersion(prof, r_a, R_QUERY, 400.0, G_STELLAR).sigma_r)

    _assert_ad_fd(f, 2.0, name="jeans sigma_r / r_a")


def test_grad_jeans_sigma_r_wrt_M():
    """2. d(sum sigma_r)/d(M) — Plummer OM."""
    prof = PlummerProfile(r_h=1.0)

    def f(M):
        return jnp.sum(jeans_dispersion(prof, 2.0, R_QUERY, M, G_STELLAR).sigma_r)

    _assert_ad_fd(f, 400.0, name="jeans sigma_r / M")


def test_grad_jeans_sigma_r_wrt_r_h():
    """3. d(sum sigma_r)/d(r_h) — through the Plummer profile param."""

    def f(r_h):
        prof = PlummerProfile(r_h=r_h)
        return jnp.sum(jeans_dispersion(prof, 2.0, R_QUERY, 400.0, G_STELLAR).sigma_r)

    _assert_ad_fd(f, 1.0, name="jeans sigma_r / r_h")


def test_grad_jeans_sigma_r_wrt_gamma():
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


# --- Phase 0.5 Task A: tabulate-once project_dispersion equivalence regression ---
#
# Baseline captured from the PRE-refactor project_dispersion (main@54f5437) for
# PlummerProfile(r_h=1.0), r_a=2.0, R=[0.5,1.0,2.0,4.0], M=400.0, G=0.00449.
# The Task-A refactor (one master Jeans solve, interpolated per-R, instead of a
# per-R re-solve) is PURE code-motion: the math/operations are identical, so the
# outputs must be bit-equivalent to rtol 1e-9. NEVER loosen this tolerance — a
# mismatch means the refactor changed the physics.
_BL_LOS = jnp.array(
    [0.5530860064022423, 0.450633657883771, 0.301176907787219, 0.17198970776636632]
)
_BL_PMT = jnp.array(
    [0.5372960641506553, 0.4286141692075936, 0.267333398791882, 0.12558509737916926]
)


def test_project_equivalence_after_tabulate():
    """project_dispersion is numerically unchanged by the tabulate-once refactor."""
    prof = PlummerProfile(r_h=1.0)
    R = jnp.array([0.5, 1.0, 2.0, 4.0])
    pj = project_dispersion(prof, 2.0, R, 400.0, G_STELLAR)
    assert jnp.allclose(pj.sigma_los, _BL_LOS, rtol=1e-9)
    assert jnp.allclose(pj.sigma_pm_t, _BL_PMT, rtol=1e-9)


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
