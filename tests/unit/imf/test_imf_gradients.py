"""Gradient correctness for IMF sampling: FD-vs-autodiff + boundary NaN-grad.

The IMF ppf/sample entry points must be differentiable w.r.t. their *parameters*
(for gradient-based IMF inference), not just w.r.t. the uniform draws u. These
tests pin that the autodiff gradient matches a central finite difference (so a
silently-zero or wrong gradient is caught) and that no NaN/Inf appears at the
u -> 0 / u -> 1 boundaries (the sqrt(0) / 1-over-r NaN-grad trap).
"""
import jax
import jax.numpy as jnp
import pytest

from progenax.imf import ChabrierIMF, Maschberger, Schechter, PowerLawIMF

U = jnp.array([0.2, 0.4, 0.6, 0.8])


def _central_fd(f, x, h):
    """Central finite difference of a scalar->scalar function."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _assert_grad_matches_fd(f, x0, h=1e-5, rtol=1e-5, atol=1e-9):
    """Autodiff grad of f at x0 matches the central FD (and is finite/non-zero)."""
    g = jax.grad(f)(x0)
    g_fd = _central_fd(f, x0, h)
    assert jnp.isfinite(g), f"autodiff grad is {g}"
    assert jnp.abs(g) > 1e-6, f"grad effectively zero ({g}); FD says {g_fd}"
    assert jnp.abs(g - g_fd) <= rtol * jnp.abs(g_fd) + atol, (
        f"autodiff {float(g):.6e} vs FD {float(g_fd):.6e} "
        f"(rel {float(jnp.abs(g - g_fd) / (jnp.abs(g_fd) + 1e-12)):.2e})"
    )


class TestFDvsAutodiff:
    """Autodiff ppf-parameter gradients match central finite differences."""

    def test_chabrier_ppf_grad_alpha(self):
        _assert_grad_matches_fd(lambda a: jnp.sum(ChabrierIMF(alpha=a).ppf(U)), 2.3)

    def test_chabrier_ppf_grad_sigma(self):
        _assert_grad_matches_fd(lambda s: jnp.sum(ChabrierIMF(sigma=s).ppf(U)), 0.69)

    def test_chabrier_ppf_grad_mc(self):
        _assert_grad_matches_fd(lambda mc: jnp.sum(ChabrierIMF(m_c=mc).ppf(U)), 0.08)

    def test_maschberger_ppf_grad_mu(self):
        _assert_grad_matches_fd(lambda mu: jnp.sum(Maschberger(mu=mu).ppf(U)), 0.2)

    def test_maschberger_ppf_grad_alpha(self):
        _assert_grad_matches_fd(lambda a: jnp.sum(Maschberger(alpha=a).ppf(U)), 2.3)

    def test_maschberger_ppf_grad_beta(self):
        _assert_grad_matches_fd(lambda b: jnp.sum(Maschberger(beta=b).ppf(U)), 1.4)

    def test_schechter_ppf_grad_alpha(self):
        _assert_grad_matches_fd(lambda a: jnp.sum(Schechter(alpha=a).ppf(U)), 2.3)

    def test_powerlaw_ppf_grad_exponent(self):
        """Salpeter single-segment: grad flows through the analytic ppf w.r.t. the slope."""
        _assert_grad_matches_fd(
            lambda a: jnp.sum(
                PowerLawIMF(exponents=[a], breakpoints=[], m_min=0.1, m_max=100.0).ppf(U)
            ),
            2.35,
        )

    def test_powerlaw_ppf_grad_mmin(self):
        _assert_grad_matches_fd(
            lambda mm: jnp.sum(
                PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=mm, m_max=100.0).ppf(U)
            ),
            0.1,
        )


class TestBoundaryGradients:
    """No NaN/Inf in d(ppf)/du at the u -> 0 / u -> 1 boundaries."""

    @pytest.mark.parametrize(
        "imf",
        [
            ChabrierIMF(),
            Maschberger(),
            Schechter(),
            PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=0.1, m_max=100.0),
        ],
    )
    def test_grad_finite_at_boundary(self, imf):
        u_boundary = jnp.array([1e-10, 1.0 - 1e-10])
        g = jax.grad(lambda u: jnp.sum(imf.ppf(u)))(u_boundary)
        assert jnp.all(jnp.isfinite(g)), f"non-finite boundary grad: {g}"


class TestParameterGradients:
    """Gradients flow through IMF parameters (finiteness + non-zero)."""

    def test_chabrier_grad_wrt_alpha(self):
        """Gradient w.r.t. alpha is finite and non-zero through ppf."""
        def loss(alpha):
            return jnp.sum(ChabrierIMF(alpha=alpha).ppf(U))

        grad_val = jax.grad(loss)(2.3)
        assert jnp.isfinite(grad_val), f"Gradient is {grad_val}"
        assert jnp.abs(grad_val) > 1e-6, f"Gradient is effectively zero: {grad_val}"

    def test_chabrier_grad_wrt_sigma(self):
        """Gradient w.r.t. sigma is finite and non-zero through ppf."""
        def loss(sigma):
            return jnp.sum(ChabrierIMF(sigma=sigma).ppf(U))

        grad_val = jax.grad(loss)(0.69)
        assert jnp.isfinite(grad_val), f"Gradient is {grad_val}"
        assert jnp.abs(grad_val) > 1e-6, f"Gradient is effectively zero: {grad_val}"

    def test_maschberger_grad_wrt_mu(self):
        """Gradient w.r.t. mu is finite and non-zero through ppf."""
        def loss(mu):
            return jnp.sum(Maschberger(mu=mu).ppf(U))

        grad_val = jax.grad(loss)(0.2)
        assert jnp.isfinite(grad_val), f"Gradient is {grad_val}"
        assert jnp.abs(grad_val) > 1e-6, f"Gradient is effectively zero: {grad_val}"


class TestAlphaOneGradients:
    """Audit R10: bare where(|1-a|<eps, log, pow/e) backprops 0*NaN at a=1.

    The exp_safe double-where fix (ported from imf/differentiable.py:47-54 to
    PowerLawIMF's 4 power/division sites) makes the gradient FINITE everywhere
    and FD-EXACT in a neighborhood of a=1. At EXACTLY a=1 the double-where
    selects the alpha-INDEPENDENT log branch, so AD there is branch-limited
    (grad 0 for ppf; off for mean_mass via dZ/da=0) — a measure-zero point the
    function is still smooth through. The exp_safe pattern guarantees finiteness,
    not point-FD (same property as the reference differentiable.py, whose tests
    likewise check finiteness, not point-FD).

    These tests pin the achievable + honest guarantee, asserting MORE than the
    existing convention: finite AT a=1, FD-exact at a=1±1e-3, and a smooth
    (kink-free) forward value through a=1.
    """

    @staticmethod
    def _imf(alpha):
        from progenax.imf import PowerLawIMF
        return PowerLawIMF(exponents=[alpha], breakpoints=[], m_min=0.1, m_max=100.0)

    @staticmethod
    def _stat_fn(stat):
        if stat == "mean_mass":
            return lambda a: TestAlphaOneGradients._imf(a).mean_mass()
        return lambda a: TestAlphaOneGradients._imf(a).ppf(jnp.array(0.5))

    @pytest.mark.parametrize("stat", ["mean_mass", "ppf"])
    def test_grad_finite_at_exactly_alpha_one(self, stat):
        """No NaN/Inf at exactly alpha=1 — the audit R10 failure mode."""
        g = jax.grad(self._stat_fn(stat))(1.0)
        assert jnp.isfinite(g), f"grad({stat}) at alpha=1 is {g}"

    @pytest.mark.parametrize("stat", ["mean_mass", "ppf"])
    @pytest.mark.parametrize("a0", [1.0 - 1e-3, 1.0 + 1e-3])
    def test_grad_fd_exact_near_alpha_one(self, stat, a0):
        """In a neighborhood of the removable singularity the gradient is exact
        (the regular branch is active and FD-matches to <1e-4)."""
        f = self._stat_fn(stat)
        g = float(jax.grad(f)(a0))
        h = 1e-4
        fd = float((f(a0 + h) - f(a0 - h)) / (2 * h))
        assert abs(g / fd - 1.0) < 1e-4, f"{stat} a={a0}: AD={g} FD={fd}"

    @pytest.mark.parametrize("stat", ["mean_mass", "ppf"])
    def test_value_smooth_through_alpha_one(self, stat):
        """Forward value is continuous, monotone, and kink-free through alpha=1
        (the singularity is removable — the value never had the NaN, only the
        gradient). The near-zero second difference confirms no kink."""
        f = lambda a: float(self._stat_fn(stat)(a))
        lo, mid, hi = f(0.999), f(1.0), f(1.001)
        assert lo > mid > hi  # both stats decrease as alpha steepens
        second_diff = abs((lo - mid) - (mid - hi))
        assert second_diff < 0.05 * abs(lo - hi)  # near-linear: no kink at a=1

    def test_sample_statistic_grad_finite_at_alpha_one(self):
        def loss(a):
            m = self._imf(a).sample(jax.random.PRNGKey(0), 500)
            return jnp.mean(jnp.log(m))
        assert jnp.isfinite(jax.grad(loss)(1.0))
