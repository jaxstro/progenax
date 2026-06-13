"""FD-vs-autodiff gradient checks on profile samplers (Batch 1, P4).

The existing grad tests assert only finiteness; these pin the gradient *value*
of a sampler observable (mean sampled radius) against a central finite
difference, so a silently wrong or stop_gradient'd sampler gradient is caught.
"""
import jax
import jax.numpy as jnp

from progenax import EFFProfile

_MASSES = jnp.ones(300)
_KEY = jax.random.PRNGKey(0)


def _mean_radius(profile):
    pos = profile.sample_positions(_MASSES, _KEY)
    return jnp.mean(jnp.linalg.norm(pos, axis=1))


def _central_fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


class TestSamplerGradients:
    # AD-vs-FD for the profile sample_positions FD cases is owned by the grad-audit
    # registry (tests/validation/grad_audit/registry.py :: PlummerProfile.sample_positions
    # [r_h], EFFProfile.sample_positions [gamma, r_t], KingProfile.sample_positions [r_c, W0]);
    # see docs/website/50-validation/differentiability-audit.md. (audit T6 consolidation; registry is SoT)
    #
    # TODO(grad-audit): EFFProfile.sample_positions(a) — the EFF scale-radius channel on the
    # POSITION sampler — is NOT directly owned by the registry. The registry has
    # EFFProfile.sample_positions(gamma) + (r_t) and EFFVelocityDF.sample_velocities(a) [the
    # VELOCITY observable], but no EFFProfile.sample_positions(a) mean-radius case. Kept here
    # (the safety interlock against silent coverage loss) pending an equal/stronger registry Case.
    def test_eff_grad_a(self):
        f = lambda a: _mean_radius(EFFProfile(a=a, gamma=3.0, r_t=10.0))
        ad, fd = jax.grad(f)(1.0), _central_fd(f, 1.0, 1e-4)
        assert jnp.isfinite(ad) and jnp.isclose(ad, fd, rtol=5e-3)
