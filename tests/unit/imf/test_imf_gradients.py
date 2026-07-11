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

from progenax.imf import ChabrierIMF, Maschberger, PowerLawIMF, Schechter

# The AD-vs-FD ppf-parameter gradient checks (ChabrierIMF.ppf m_c/sigma/alpha,
# Maschberger.ppf mu/alpha/beta, Schechter.ppf alpha, PowerLawIMF.ppf[Salpeter]
# alpha, and PowerLawIMF.ppf[m_min]) are owned by the grad-audit registry
# (tests/validation/grad_audit/registry.py :: ChabrierIMF.ppf / Maschberger.ppf /
# Schechter.ppf / PowerLawIMF.ppf[Salpeter] / PowerLawIMF.ppf[m_min]);
# see docs/website/50-validation/differentiability-audit.md. The former
# TestFDvsAutodiff class was removed here (audit T6 consolidation; registry is SoT).


class TestBoundaryGradients:
    """No NaN/Inf in d(ppf)/du at the u -> 0 / u -> 1 boundaries."""

    @pytest.mark.parametrize(
        "imf",
        [
            ChabrierIMF(),
            Maschberger(),
            Schechter(),
            PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=0.1, m_max=100.0),
            # Multi-segment Kroupa: pins d(ppf)/du finiteness across the piecewise
            # breakpoints too (the broken-power-law inverse-CDF has a distinct
            # segment-selection path from the single-segment Salpeter form above).
            # This preserves the multi-segment du-grad coverage from the deleted
            # finite-only du-monotonicity smokes (4.2c review note).
            PowerLawIMF.kroupa(),
        ],
    )
    def test_grad_finite_at_boundary(self, imf):
        u_boundary = jnp.array([1e-10, 1.0 - 1e-10])
        g = jax.grad(lambda u: jnp.sum(imf.ppf(u)))(u_boundary)
        assert jnp.all(jnp.isfinite(g)), f"non-finite boundary grad: {g}"


# The IMF-parameter ppf gradients (ChabrierIMF.ppf alpha/sigma, Maschberger.ppf mu)
# are FD-audited by the grad-audit registry (tests/validation/grad_audit/registry.py ::
# ChabrierIMF.ppf [alpha] / [sigma], Maschberger.ppf [mu]); see
# docs/website/50-validation/differentiability-audit.md. The former finite-only
# TestParameterGradients (isfinite + non-zero, NO FD) was removed (audit T6: a
# silently-zeroed grad would PASS isfinite; the registry FD cases are strictly
# stronger; registry is SoT).


class TestAlphaOneGradients:
    """Audit S4 (supersedes the R10 branch-limited pin): the expm1-stable form
    (lo**e * D * phi(e*D), phi = expm1(x)/x; sibling psi = log1p(y)/y for the
    inverse) is ONE smooth expression in e = 1 - alpha, so the autodiff gradient
    is CORRECT (FD-exact) everywhere INCLUDING exactly alpha=1.

    The old exp_safe double-where kept the VJP finite but selected an
    alpha-INDEPENDENT log branch at exactly alpha=1: AD(ppf)=0 vs FD=-1.384e4,
    AD(mean_mass)=-52.2 vs FD=-35.6 (audit S4 measurements). These tests pin the
    strengthened guarantee: AD == FD at exactly a=1 (and a=2 for mean_mass's
    numerator singularity), plus the original finiteness/smoothness properties.
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
    @pytest.mark.parametrize("a0", [1.0 - 1e-3, 1.0, 1.0 + 1e-3])
    def test_grad_fd_exact_at_and_near_alpha_one(self, stat, a0):
        """AD == central FD at AND around the removable singularity — including
        exactly alpha=1 (the S4 fix; the old form silently returned AD=0 there)."""
        f = self._stat_fn(stat)
        g = float(jax.grad(f)(a0))
        h = 1e-4
        fd = float((f(a0 + h) - f(a0 - h)) / (2 * h))
        assert abs(g / fd - 1.0) < 1e-4, f"{stat} a={a0}: AD={g} FD={fd}"

    def test_mean_mass_grad_fd_exact_at_alpha_two(self):
        """mean_mass's NUMERATOR exponent e = 2 - alpha hits its removable
        singularity at alpha=2 — the same stable form must carry that gradient."""
        f = self._stat_fn("mean_mass")
        g = float(jax.grad(f)(2.0))
        h = 1e-4
        fd = float((f(2.0 + h) - f(2.0 - h)) / (2 * h))
        assert abs(g / fd - 1.0) < 1e-4, f"mean_mass a=2: AD={g} FD={fd}"

    def test_ppf_forward_regression_vs_closed_form(self):
        """Forward regression: at alpha != 1 the stable form must agree with the
        textbook closed form (u*(hi**e - lo**e) + lo**e)**(1/e) to < 1e-10 rel."""
        import numpy as np

        lo, hi, a = 0.1, 100.0, 2.35
        e = 1.0 - a
        u = np.linspace(0.01, 0.99, 21)
        expected = (u * (hi**e - lo**e) + lo**e) ** (1.0 / e)
        got = np.asarray(self._imf(a).ppf(jnp.asarray(u)))
        np.testing.assert_allclose(got, expected, rtol=1e-10)

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


class TestAlphaOneGradientsDifferentiableIMF:
    """S4 sibling for imf/differentiable.py (the IMFParams inference path):
    AD == FD at exactly alpha_j = 1 through _compute_normalization,
    _compute_cdf_at_breaks, and the per-segment inverse CDF."""

    @staticmethod
    def _params(a1):
        from progenax.imf.params import IMFParams

        return IMFParams(
            alpha0=jnp.array(0.3),
            alpha1=a1,
            alpha2=jnp.array(2.3),
            alpha3=jnp.array(2.3),
        )

    def test_nll_grad_fd_exact_at_alpha_one(self):
        from progenax.imf.differentiable import log_prob_masses

        masses = PowerLawIMF.kroupa().sample(jax.random.PRNGKey(3), 200)
        f = lambda a1: -jnp.sum(log_prob_masses(masses, self._params(a1)))
        g = float(jax.grad(f)(1.0))
        h = 1e-4
        fd = float((f(1.0 + h) - f(1.0 - h)) / (2 * h))
        assert abs(g / fd - 1.0) < 1e-4, f"NLL a1=1: AD={g} FD={fd}"

    def test_sampled_mass_grad_fd_exact_at_alpha_one(self):
        from progenax.imf.differentiable import sample_masses_from_params

        u = jax.random.uniform(jax.random.PRNGKey(4), (500,))
        f = lambda a1: jnp.mean(sample_masses_from_params(self._params(a1), u))
        g = float(jax.grad(f)(1.0))
        h = 1e-4
        fd = float((f(1.0 + h) - f(1.0 - h)) / (2 * h))
        assert abs(g / fd - 1.0) < 1e-4, f"sample a1=1: AD={g} FD={fd}"
