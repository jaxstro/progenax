"""Shared numerical primitives.

Both generic kernels now live in jaxstro as the ecosystem's single source of
truth and are re-exported here so the existing ``from progenax.numerics import
...`` call sites keep working unchanged:

- ``cumulative_trapz`` (``jaxstro.numerics.integration``): cumulative trapezoid,
  standardized on the dx-OUTSIDE ordering (pairwise average -> cumsum -> scale by
  scalar dx once -> leading zero). The jaxstro signature
  ``cumulative_trapz(y, x=None, *, dx, axis)`` is keyword-compatible with every
  progenax call site (all pass ``dx=``/``axis=``).
- ``inverse_cdf_draw`` (``jaxstro.numerics.sampling``): differentiable inverse-CDF
  (PPF) draw from an unnormalized weight on a uniform grid. Hoisted byte-for-byte
  from this module's former local kernel (parity-verified in jaxstro's
  ``tests/test_sampling.py`` against a copy of this origin), so all 16 call sites
  retain identical numerics.

The former dx-INSIDE sites (``profiles/density_poisson.py``, ``profiles/api.py``,
``kinematics/eff_df.py``) multiplied dx inside the cumsum; against the dx-outside
form they agree only to ~1 ulp (measured: 124/257 elements differ, max rel. diff
8.9e-16), within their existing test budgets.
"""

import math

import jax.numpy as jnp
from jaxstro.numerics.integration import cumulative_trapz
from jaxstro.numerics.sampling import inverse_cdf_draw

__all__ = [
    "cumulative_trapz",
    "inverse_cdf_draw",
    "power_integral_stable",
    "power_ppf_stable",
    "require_positive",
]

# Switch to the Taylor branch of phi/psi below this |argument|. At 1e-6 the
# 3-term Taylor truncation error (~x^3/24 for phi, ~y^3/4 for psi) is < 1e-18,
# below float64 eps, so value AND gradient are seamless across the switch.
_TAYLOR_THRESHOLD = 1e-6


def _phi(x):
    """expm1(x)/x, smooth (value 1, slope 1/2) through x=0.

    Double-where with a finite substitute in the masked branch so the VJP is
    finite at exactly x=0 (a bare where would backprop 0*NaN from d(1/x)).
    """
    small = jnp.abs(x) < _TAYLOR_THRESHOLD
    x_safe = jnp.where(small, 1.0, x)
    taylor = 1.0 + x / 2.0 + x * x / 6.0
    return jnp.where(small, taylor, jnp.expm1(x_safe) / x_safe)


def _psi(y):
    """log1p(y)/y, smooth (value 1, slope -1/2) through y=0. Same double-where
    pattern as _phi."""
    small = jnp.abs(y) < _TAYLOR_THRESHOLD
    y_safe = jnp.where(small, 1.0, y)
    taylor = 1.0 - y / 2.0 + y * y / 3.0
    return jnp.where(small, taylor, jnp.log1p(y_safe) / y_safe)


def power_integral_stable(lo, hi, e):
    """(hi**e - lo**e)/e with the CORRECT autodiff gradient through e=0.

    The power-law segment integral int_lo^hi m**(e-1) dm has a removable
    singularity at e=0 (value log(hi/lo)). The historical exp_safe double-where
    guarded the NaN but selected an e-INDEPENDENT log branch at exactly e=0,
    silently zeroing d/de (audit S4: AD=0 vs FD=-4.92e0-scale). This form,

        (hi**e - lo**e)/e = lo**e * D * phi(e*D),  D = log(hi) - log(lo),
        phi(x) = expm1(x)/x,

    is one smooth expression in e, so jax.grad is FD-exact everywhere,
    including exactly e=0 (its e=0 slope D*(log hi + log lo)/2 is the true
    derivative). Requires lo > 0, hi > 0.
    """
    delta = jnp.log(hi) - jnp.log(lo)
    return lo**e * delta * _phi(e * delta)


def power_ppf_stable(lo, t, e):
    """(lo**e + t*e)**(1/e) with the CORRECT autodiff gradient through e=0.

    The inverse of ``power_integral_stable``: solving (m**e - lo**e)/e = t for
    m. Computed as

        m = exp(log(lo) + s * psi(e*s)),  s = t * lo**(-e),  psi(y) = log1p(y)/y,

    one smooth expression in e whose e=0 limit lo*exp(t) is the alpha=1
    log-segment inverse. Requires lo > 0 and lo**e + t*e > 0 (i.e. t a valid
    partial integral for the segment).
    """
    s = t * lo ** (-e)
    return jnp.exp(jnp.log(lo) + s * _psi(e * s))


def require_positive(value, name: str) -> None:
    """Raise ``ValueError`` if a CONCRETE ``value`` is a finite, non-positive number.

    Eager input validation for constructors (audit S7): a *finite* negative/zero
    scale silently produces garbage (a negative Plummer scale radius flips
    positions; W0<=0 gives a degenerate King r_t). Non-finite (NaN/inf) input is
    deliberately NOT caught here — it propagates loudly (NaN outputs) and is
    handled by downstream realizability gates (e.g. Engine-B's non-finite-DF
    check); intercepting it here would mask those more specific messages.
    Skipped under tracing — ``float()`` raises on a tracer, so jit/grad see no
    host-side control flow.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return  # traced: cannot check at trace time
    if math.isfinite(v) and v <= 0.0:
        raise ValueError(f"{name} must be > 0 (got {v:g}).")
