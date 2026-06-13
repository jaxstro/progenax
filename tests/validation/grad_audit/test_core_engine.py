"""The grad-audit engine must CATCH broken gradients (NaN / silent-zero / wrong-by-2x)
and correctly classify the intentional-limitation expect-classes. These toy functions are
the engine's RED proof — they do not touch progenax physics."""
import jax
import jax.numpy as jnp
import progenax  # noqa: F401  (float64)

from tests.validation.grad_audit.core import Case, audit_entry_point


def _case(fn, expect="consistent", tol=1e-5, reduce=jnp.sum):
    return Case(id="toy", direction="params->IC", fn=fn, param="x",
                theta0=2.0, reduce=reduce, expect=expect, tol=tol)


def test_clean_linear_is_clean():
    r = audit_entry_point(_case(lambda x: jnp.array([3.0 * x, 5.0 * x])))
    assert r.finite and r.status == "clean"
    assert abs(r.ratio - 1.0) < 1e-6 and r.abs_ad > 1e-9


def test_silent_zero_is_hazard():
    # stop_gradient -> AD is 0 but FD is non-zero: the headline failure mode.
    r = audit_entry_point(_case(lambda x: jax.lax.stop_gradient(3.0 * x) * jnp.ones(2)))
    assert r.status == "hazard" and r.abs_ad < 1e-12


def test_nan_grad_is_hazard():
    # sqrt(x - x) has a 0/0 grad -> NaN; the engine must not call it clean.
    r = audit_entry_point(_case(lambda x: jnp.sqrt(x - x) * x * jnp.ones(1)))
    assert (not r.finite) and r.status == "hazard"


def test_wrong_by_two_is_hazard():
    # AD double-counts: a custom_jvp that lies about the derivative.
    @jax.custom_jvp
    def f(x):
        return x * jnp.ones(1)
    f.defjvp(lambda p, t: (f(p[0]), 2.0 * t[0] * jnp.ones(1)))  # claims 2x
    r = audit_entry_point(_case(f))
    assert r.status == "hazard" and abs(r.ratio - 2.0) < 1e-5


def test_known_zero_pins_zero_gradient():
    # An intentionally constant-in-x output: AD=0, FD=0 -> known-limitation, NOT hazard.
    r = audit_entry_point(_case(lambda x: jnp.ones(2), expect="known_zero"))
    assert r.status == "known-limitation"


def test_known_zero_flags_if_gradient_appears():
    # If a 'known_zero' case SUDDENLY has a gradient, that is a hazard (unannounced change).
    r = audit_entry_point(_case(lambda x: 3.0 * x * jnp.ones(2), expect="known_zero"))
    assert r.status == "hazard"


def test_known_blocked_requires_only_finite():
    r = audit_entry_point(_case(lambda x: jax.lax.stop_gradient(x) * jnp.ones(2),
                                expect="known_blocked"))
    assert r.status == "known-limitation"  # finite AD (0) is acceptable for a blocked site
