"""BM19 1D gravoturbulent density-PDF theory — physics tests.

Grounded in Burkhart & Mocz 2019 (ApJ 879, 129), verified against the held PDF
(docs/core-papers/Burkhart_2019_ApJ_879_129.pdf) this session:
  Eq. 1   sigma_s^2 = ln(1 + b^2 M^2)
  Eq. 2   s_t = (alpha - 1/2) sigma_s^2
  Eq. 19/20 f_dense = M_PL / (M_LN + M_PL)  (mass-weighted piecewise integral)
  kappa = 3/alpha  (radial density slope <-> PDF powerlaw slope)
"""

import math

import jax
import pytest

pytestmark = pytest.mark.experimental


# ── Task 1.1: BM19 Eq. 1 — lognormal width sigma_s^2 = ln(1 + b^2 M^2) ──
def test_sigma_s_squared_known_value():
    from gravoturb.theory.density_pdf import sigma_s_squared

    # b=0.4, M=5 -> ln(1 + 0.16*25) = ln(5) = 1.6094379124341003
    val = float(sigma_s_squared(mach=5.0, b=0.4))
    assert val == pytest.approx(math.log(5.0), abs=1e-12)


def test_sigma_s_squared_zero_mach():
    from gravoturb.theory.density_pdf import sigma_s_squared

    # No turbulence -> delta-function density -> zero width.
    assert float(sigma_s_squared(mach=0.0, b=0.4)) == pytest.approx(0.0, abs=1e-12)


def test_sigma_s_squared_differentiable():
    from gravoturb.theory.density_pdf import sigma_s_squared

    g = jax.grad(lambda m: sigma_s_squared(m, 0.4))(5.0)
    # d/dM ln(1+b^2 M^2) = 2 b^2 M / (1 + b^2 M^2) > 0
    assert float(g) > 0.0


# ── Task 1.2: BM19 Eq. 2 — transition density s_t = (alpha - 1/2) sigma_s^2 ──
def test_transition_density_known_value():
    from gravoturb.theory.density_pdf import transition_density

    # alpha=2, sigma_s^2=ln(5) -> s_t = 1.5 * ln(5)
    s_t = float(transition_density(alpha=2.0, sigma_s_sq=math.log(5.0)))
    assert s_t == pytest.approx(1.5 * math.log(5.0), abs=1e-12)


def test_transition_density_alpha_1p5_identity():
    # BM19 Eq. 16: for alpha=1.5, s_t = sigma_s^2 exactly.
    from gravoturb.theory.density_pdf import transition_density

    s2 = math.log(5.0)
    assert float(transition_density(1.5, s2)) == pytest.approx(s2, abs=1e-12)


# ── Task 1.4: radial slope kappa = 3/alpha (BM19 §2) ──
def test_pdf_slope_to_radial():
    from gravoturb.theory.density_pdf import pdf_slope_to_radial

    # alpha=2 -> kappa=1.5; alpha=1.5 -> kappa=2 (rho ~ r^-2 isothermal core, Shu 1977)
    assert float(pdf_slope_to_radial(2.0)) == pytest.approx(1.5, abs=1e-12)
    assert float(pdf_slope_to_radial(1.5)) == pytest.approx(2.0, abs=1e-12)


# ── Task 1.3: f_dense (BM19 Eq. 17-20) + AC1 (limit) + AC2 (mass conservation) ──
def _numpy_eq18_f_dense(mach, b, alpha):
    """Independent reference: numerically integrate BM19 Eq. 18 (mass-weighted)
    using a mass-conserving lognormal body (mean s0 = -sigma_s^2/2) and a
    continuity-matched powerlaw tail p_PL(s) = C e^{-alpha s}, C = p_LN(s_t) e^{alpha s_t}.
    numpy quadrature is allowed in tests (analysis side)."""
    import numpy as np
    from scipy.special import erf  # noqa: F401  (used implicitly via trapz check)

    s2 = math.log(1.0 + (b * mach) ** 2)
    sig = math.sqrt(s2)
    s0 = -0.5 * s2
    s_t = (alpha - 0.5) * s2

    def p_ln(s):
        return np.exp(-((s - s0) ** 2) / (2 * s2)) / np.sqrt(2 * np.pi * s2)

    C = p_ln(s_t) * np.exp(alpha * s_t)

    # M_LN = int_{-inf}^{s_t} e^s p_LN ds ; M_PL = int_{s_t}^{inf} e^s C e^{-alpha s} ds
    s_lo = np.linspace(s0 - 12 * sig, s_t, 200_000)
    M_LN = np.trapezoid(np.exp(s_lo) * p_ln(s_lo), s_lo)
    s_hi = np.linspace(s_t, s_t + 200.0, 400_000)
    M_PL = np.trapezoid(np.exp(s_hi) * C * np.exp(-alpha * s_hi), s_hi)
    return M_PL / (M_LN + M_PL)


def test_mass_conservation_lognormal():  # AC2
    """The mass-conserving lognormal (mean s0=-sigma_s^2/2) integrates e^s p_LN ds = 1."""
    import numpy as np

    s2 = math.log(1.0 + (0.4 * 5.0) ** 2)
    sig, s0 = math.sqrt(s2), -0.5 * s2
    s = np.linspace(s0 - 15 * sig, s0 + 15 * sig, 400_000)
    p = np.exp(-((s - s0) ** 2) / (2 * s2)) / np.sqrt(2 * np.pi * s2)
    assert np.trapezoid(np.exp(s) * p, s) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize(
    "mach,b,alpha", [(5.0, 0.4, 2.0), (10.0, 1.0 / 3, 1.6), (8.0, 0.5, 1.8)]
)
def test_f_dense_matches_eq18_quadrature(mach, b, alpha):  # AC1
    """Closed-form f_dense (Eq. 19/20) matches direct quadrature of Eq. 18."""
    from gravoturb.theory.density_pdf import f_dense_bm19_full

    closed = float(f_dense_bm19_full(mach=mach, b=b, alpha=alpha))
    ref = _numpy_eq18_f_dense(mach, b, alpha)
    assert closed == pytest.approx(ref, rel=1e-4)


def test_f_dense_lognormal_limit_formula():  # AC1
    """f_dense_lognormal_limit == 1/2 erfc((s_t - sigma_s^2/2)/(sqrt2 sigma_s)),
    the dense-mass fraction of a *pure* lognormal above s_t (BM19 comparison form)."""
    from gravoturb.theory.density_pdf import f_dense_lognormal_limit
    from scipy.special import erfc

    mach, b, alpha = 5.0, 0.4, 1.8
    s2 = math.log(1.0 + (b * mach) ** 2)
    s_t = (alpha - 0.5) * s2
    z = (s_t - 0.5 * s2) / (math.sqrt(2.0) * math.sqrt(s2))
    assert float(f_dense_lognormal_limit(mach=mach, b=b, alpha=alpha)) == pytest.approx(
        0.5 * erfc(z), rel=1e-6
    )


def test_f_dense_exceeds_lognormal_only():  # AC1
    """The shallower continuity-matched powerlaw tail adds dense gas beyond the
    pure-lognormal fraction: f_dense_full > f_dense_lognormal_limit."""
    from gravoturb.theory.density_pdf import f_dense_bm19_full, f_dense_lognormal_limit

    for alpha in (1.6, 2.0):
        full = float(f_dense_bm19_full(mach=5.0, b=0.4, alpha=alpha))
        lim = float(f_dense_lognormal_limit(mach=5.0, b=0.4, alpha=alpha))
        assert full > lim


def test_f_dense_bounds_and_monotonic():
    """0 < f_dense < 1; decreases with Mach and with alpha (BM19 Fig. 5)."""
    from gravoturb.theory.density_pdf import f_dense_bm19_full

    f = lambda M, a: float(f_dense_bm19_full(mach=M, b=1.0 / 3, alpha=a))
    assert 0.0 < f(5.0, 2.0) < 1.0
    assert f(5.0, 2.0) < f(5.0, 1.5)  # shallower tail -> more dense gas
    assert f(20.0, 1.8) < f(5.0, 1.8)  # higher Mach -> less dense gas
