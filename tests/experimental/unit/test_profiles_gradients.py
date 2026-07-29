r"""AC-BE5 — gradient validation for the gas-envelope profiles.

A correct forward value does not imply a correct gradient, so every user-facing knob is
grad-checked against a central finite difference with a step sweep (float64 is on
project-wide via ``jaxstro.jaxconfig``).

The knobs, and why each matters:

    BonnorEbertProfile.r_h      physical scale
    BonnorEbertProfile.xi_max   SHAPE -- the stored primary (ADR-0066)
    PolytropeProfile.r_h        physical scale
    PolytropeProfile.gamma      the equation of state, the whole point of ADR-0065
    polytrope_xi1 in n          the diffrax.Event root, via the implicit function theorem

:class:`TestNoZeroGradientTrap` is a regression guard for a real bug caught during this
phase. The half-mass inversion originally bisected for ``xi_h``; bisection builds its
answer purely from arithmetic on the bracket endpoints, and the mass table enters only
through the comparison ``m(mid) < m_half`` -- a hard threshold with zero derivative. The
forward value was excellent while ``d xi_h / d(shape)`` was silently **zero**, which would
have corrupted every density derivative through ``r_0 = r_h / xi_h``. The fix bisects
under ``stop_gradient`` and takes one Newton step, so the gradient comes from the implicit
function theorem. These tests fail loudly if that ever regresses.
"""

import jax
import jax.numpy as jnp
import pytest
from gravoturb.profiles import BonnorEbertProfile, PolytropeProfile
from jaxstro.numerics.lane_emden import polytrope_xi1

# Skill threshold: relative error below 1e-5 after sweeping the step size.
GRAD_RTOL = 1e-5
_STEPS = (1e-4, 1e-5, 1e-6, 1e-7)

# Small grids keep the gate fast; accuracy is dominated by Tsit5, not n_points.
N_POINTS = 400


def grad_check(fn, x):
    """Return ``(g_ad, g_fd, rel_err)`` using the best step in a sweep.

    A single step size can fail spuriously -- too large is truncation error, too small is
    round-off -- so the sweep takes the best and the caller asserts on that.
    """
    g_ad = float(jax.grad(fn)(x))
    best = None
    for h in _STEPS:
        g_fd = float((fn(x + h) - fn(x - h)) / (2.0 * h))
        rel = abs(g_ad - g_fd) / (abs(g_ad) + abs(g_fd) + 1e-30)
        if best is None or rel < best[2]:
            best = (g_ad, g_fd, rel)
    return best


def _be_density_sum(r_h, xi_max, radii):
    p = BonnorEbertProfile(r_h=r_h, xi_max=xi_max, n_points=N_POINTS)
    return jnp.sum(p.density(radii))


def _poly_density_sum(r_h, gamma, radii):
    p = PolytropeProfile(r_h=r_h, gamma=gamma, n_points=N_POINTS)
    return jnp.sum(p.density(radii))


class TestBonnorEbertGradients:
    @pytest.mark.parametrize("r_h", [0.7, 2.3])
    def test_density_wrt_r_h(self, r_h):
        radii = jnp.array([0.3, 0.6, 1.1])
        g_ad, g_fd, rel = grad_check(lambda x: _be_density_sum(x, 6.0, radii), r_h)
        assert rel < GRAD_RTOL, f"AD={g_ad:.6e} FD={g_fd:.6e} rel={rel:.2e}"
        assert jnp.isfinite(g_ad)

    @pytest.mark.parametrize("xi_max", [3.0, 6.0, 9.0])
    def test_density_wrt_xi_max(self, xi_max):
        """The shape knob -- and the one the bisection bug would have silently zeroed.

        Needs a finer grid than the other gates, for a reason specific to THIS knob:
        perturbing ``xi_max`` moves the ``linspace`` nodes, so the two finite-difference
        evaluations are tabulated on different grids and their PCHIP interpolation error
        does not cancel. The residual is an artifact of the comparison, not of the
        gradient -- it falls off as the grid refines while AD converges:

            n_points   AD              best rel-err
              200      3.2052802e-02     5.8e-05
              400      3.2049352e-02     3.1e-05
              800      3.2050567e-02     9.2e-08
             1600      3.2050614e-02     1.2e-06   (finite-difference round-off floor)

        So the grid is made adequate for the quantity being measured; the 1e-5 criterion
        is unchanged.
        """
        radii = jnp.array([0.3, 0.6])
        g_ad, g_fd, rel = grad_check(
            lambda x: jnp.sum(
                BonnorEbertProfile(r_h=1.0, xi_max=x, n_points=800).density(radii)
            ),
            xi_max,
        )
        assert rel < GRAD_RTOL, f"AD={g_ad:.6e} FD={g_fd:.6e} rel={rel:.2e}"

    def test_mass_enclosed_wrt_r_h(self):
        def f(r_h):
            p = BonnorEbertProfile(r_h=r_h, xi_max=6.0, n_points=N_POINTS)
            return p.mass_enclosed(jnp.array(0.8))

        g_ad, g_fd, rel = grad_check(f, 1.0)
        assert rel < GRAD_RTOL, f"AD={g_ad:.6e} FD={g_fd:.6e} rel={rel:.2e}"

    def test_mu_be_wrt_xi_max(self):
        """The derived diagnostic must be differentiable too, not just reportable."""

        def f(xi_max):
            return BonnorEbertProfile(r_h=1.0, xi_max=xi_max, n_points=N_POINTS).mu_BE

        g_ad, g_fd, rel = grad_check(f, 4.0)
        assert rel < GRAD_RTOL, f"AD={g_ad:.6e} FD={g_fd:.6e} rel={rel:.2e}"


class TestPolytropeGradients:
    @pytest.mark.parametrize("gamma", [1.35, 5.0 / 3.0, 2.0])
    def test_density_wrt_gamma(self, gamma):
        """The equation of state as a differentiable, testable parameter."""
        radii = jnp.array([0.3, 0.6])
        g_ad, g_fd, rel = grad_check(lambda x: _poly_density_sum(1.0, x, radii), gamma)
        assert rel < GRAD_RTOL, f"AD={g_ad:.6e} FD={g_fd:.6e} rel={rel:.2e}"

    @pytest.mark.parametrize("r_h", [0.7, 2.3])
    def test_density_wrt_r_h(self, r_h):
        radii = jnp.array([0.3, 0.6, 1.1])
        g_ad, g_fd, rel = grad_check(
            lambda x: _poly_density_sum(x, 5.0 / 3.0, radii), r_h
        )
        assert rel < GRAD_RTOL, f"AD={g_ad:.6e} FD={g_fd:.6e} rel={rel:.2e}"

    def test_xi1_wrt_n(self):
        """The diffrax.Event root, differentiated via the implicit function theorem."""
        g_ad, g_fd, rel = grad_check(lambda n: polytrope_xi1(n), 1.5)
        assert rel < GRAD_RTOL, f"AD={g_ad:.6e} FD={g_fd:.6e} rel={rel:.2e}"
        assert g_ad > 0.0, "a softer EOS (larger n) must push the edge outward"


class TestNoZeroGradientTrap:
    """Regression guard: these gradients must be NONZERO, not merely finite."""

    def test_be_scale_responds_to_shape(self):
        """d r_0 / d xi_max was silently zero under plain bisection."""

        def f(xi_max):
            return BonnorEbertProfile(r_h=1.0, xi_max=xi_max, n_points=N_POINTS).r_0

        g = float(jax.grad(f)(6.0))
        assert abs(g) > 1e-6, f"d r_0/d xi_max = {g:.3e} -- the inversion lost its gradient"

    def test_polytrope_scale_responds_to_gamma(self):
        def f(gamma):
            return PolytropeProfile(r_h=1.0, gamma=gamma, n_points=N_POINTS).r_0

        g = float(jax.grad(f)(5.0 / 3.0))
        assert abs(g) > 1e-6, f"d r_0/d gamma = {g:.3e} -- the inversion lost its gradient"

    def test_be_central_density_responds_to_shape(self):
        def f(xi_max):
            return BonnorEbertProfile(r_h=1.0, xi_max=xi_max, n_points=N_POINTS).rho_c

        assert abs(float(jax.grad(f)(6.0))) > 1e-6


class TestGradientFiniteness:
    """No NaN/Inf at the awkward points: the origin, and the truncation edge."""

    @pytest.mark.parametrize("frac", [1e-4, 0.999, 1.5])
    def test_be_density_gradient_finite_across_edge(self, frac):
        p0 = BonnorEbertProfile(r_h=1.0, xi_max=6.0, n_points=N_POINTS)
        r = float(p0.r_edge) * frac

        def f(r_h):
            return BonnorEbertProfile(r_h=r_h, xi_max=6.0, n_points=N_POINTS).density(
                jnp.array(r)
            )

        assert jnp.isfinite(jax.grad(f)(1.0))

    @pytest.mark.parametrize("gamma", [1.35, 2.5])
    @pytest.mark.parametrize("frac", [1e-4, 0.999, 1.5])
    def test_polytrope_density_gradient_finite_across_edge(self, gamma, frac):
        """theta^n at the edge is the classic pow-at-zero trap; gamma>2 gives n<1."""
        p0 = PolytropeProfile(r_h=1.0, gamma=gamma, n_points=N_POINTS)
        r = float(p0.r_edge) * frac

        def f(r_h):
            return PolytropeProfile(r_h=r_h, gamma=gamma, n_points=N_POINTS).density(
                jnp.array(r)
            )

        assert jnp.isfinite(jax.grad(f)(1.0))


class TestJitCompatibility:
    def test_be_density_under_jit(self):
        @jax.jit
        def f(r_h, xi_max):
            return _be_density_sum(r_h, xi_max, jnp.array([0.3, 0.6]))

        eager = _be_density_sum(1.0, 6.0, jnp.array([0.3, 0.6]))
        assert float(f(1.0, 6.0)) == pytest.approx(float(eager), rel=1e-10)

    def test_polytrope_grad_under_jit(self):
        @jax.jit
        def g(gamma):
            return jax.grad(
                lambda x: _poly_density_sum(1.0, x, jnp.array([0.3, 0.6]))
            )(gamma)

        assert jnp.isfinite(g(5.0 / 3.0))
