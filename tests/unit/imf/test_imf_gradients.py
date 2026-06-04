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
