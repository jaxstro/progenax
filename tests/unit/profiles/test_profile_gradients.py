"""FD-vs-autodiff gradient checks on profile samplers (Batch 1, P4).

The existing grad tests assert only finiteness; these pin the gradient *value*
of a sampler observable (mean sampled radius) against a central finite
difference, so a silently wrong or stop_gradient'd sampler gradient is caught.
"""
import jax
import jax.numpy as jnp

from progenax import PlummerProfile, EFFProfile, KingProfile

_MASSES = jnp.ones(300)
_KEY = jax.random.PRNGKey(0)


def _mean_radius(profile):
    pos = profile.sample_positions(_MASSES, _KEY)
    return jnp.mean(jnp.linalg.norm(pos, axis=1))


def _central_fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


class TestSamplerGradients:
    def test_plummer_grad_rh(self):
        f = lambda r_h: _mean_radius(PlummerProfile(r_h=r_h))
        ad, fd = jax.grad(f)(1.5), _central_fd(f, 1.5, 1e-4)
        assert jnp.isfinite(ad) and jnp.isclose(ad, fd, rtol=1e-3)

    def test_eff_grad_a(self):
        f = lambda a: _mean_radius(EFFProfile(a=a, gamma=3.0, r_t=10.0))
        ad, fd = jax.grad(f)(1.0), _central_fd(f, 1.0, 1e-4)
        assert jnp.isfinite(ad) and jnp.isclose(ad, fd, rtol=5e-3)

    def test_eff_grad_gamma(self):
        f = lambda g: _mean_radius(EFFProfile(a=1.0, gamma=g, r_t=10.0))
        ad, fd = jax.grad(f)(3.0), _central_fd(f, 3.0, 1e-4)
        assert jnp.isfinite(ad) and jnp.isclose(ad, fd, rtol=1e-2)

    def test_king_grad_rc(self):
        # r_c is a pure length scale: mean radius is linear in r_c.
        f = lambda r_c: _mean_radius(KingProfile.from_W0_rc(7.0, r_c))
        ad, fd = jax.grad(f)(1.0), _central_fd(f, 1.0, 1e-4)
        assert jnp.isfinite(ad) and jnp.isclose(ad, fd, rtol=5e-3)

    def test_king_grad_w0_through_ode(self):
        # Gradient flows through the diffrax King ODE + tidal radius + CDF.
        f = lambda W0: _mean_radius(KingProfile.from_W0_rc(W0, 1.0))
        ad, fd = jax.grad(f)(7.0), _central_fd(f, 7.0, 1e-3)
        assert jnp.isfinite(ad) and jnp.isclose(ad, fd, rtol=3e-2)
