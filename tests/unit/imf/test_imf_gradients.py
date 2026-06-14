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
