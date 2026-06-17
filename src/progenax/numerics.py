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

from jaxstro.numerics.integration import cumulative_trapz
from jaxstro.numerics.sampling import inverse_cdf_draw

__all__ = ["cumulative_trapz", "inverse_cdf_draw"]
