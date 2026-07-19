r"""Gas-envelope fidelity: does the realized gas follow the prescribed profile? (ADR-0069)

The Bonnor-Ebert / polytrope models describe the **gas**, not the stars, so the validation
question is whether the realized gas density reproduces the prescribed envelope shape.

Two properties of the problem dictate how this must be measured:

1. **Ensemble, not single realization.** A single realization's shell density scatters by
   130-620% (1 sigma). The BM19 field is heavy-tailed (the top 0.1% of cells carry ~10% of
   the mass) and ``beta=3.5`` is a steep spectrum, so a shell holds many *cells* but few
   independent *modes*. Refining the grid adds cells, not modes, and does **not** reduce
   the bias. Only averaging over realizations does.

2. **No catch-all bin.** ``jnp.clip(jnp.searchsorted(edges, r) - 1, 0, n-1)`` silently
   dumps every cell outside the range into the end bins. With a cloud occupying a few
   percent of the box that puts ~99% of the grid -- essentially empty space -- into the
   outermost shell, which then reads ~0.03 of the envelope and looks like catastrophic
   disagreement. :func:`shell_mean` drops out-of-range cells instead.

Shape, not amplitude, is what is being tested: the prescribed profile is normalised to unit
total mass while the realized gas carries physical units, so the two differ by a constant.
:func:`envelope_fidelity` divides the ratio by its own mean, leaving a curve that is flat at
1.0 exactly when the shape matches.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float

# Below this a shell's mean is dominated by whether it happened to catch a tail cell.
MIN_CELLS_PER_SHELL = 200

# Stay inside the truncation edge: the density cliff there is not what we are testing.
DEFAULT_R_MIN_FRAC = 0.15
DEFAULT_R_MAX_FRAC = 0.75


class FidelityResult(NamedTuple):
    """Shape agreement between realized gas and its prescribed envelope.

    Attributes:
        r: shell centres.
        counts: cells contributing to each shell (never a catch-all).
        ratio: realized / prescribed, carrying the arbitrary amplitude constant.
        shape: ``ratio`` normalised by its own mean -- flat at 1.0 iff the shape matches.
        max_deviation: ``max|shape - 1|``, the headline number.
    """

    r: Float[Array, " k"]
    counts: Float[Array, " k"]
    ratio: Float[Array, " k"]
    shape: Float[Array, " k"]
    max_deviation: Float[Array, ""]


def shell_edges(
    r_edge,
    n_shells: int = 5,
    r_min_frac: float = DEFAULT_R_MIN_FRAC,
    r_max_frac: float = DEFAULT_R_MAX_FRAC,
):
    """Log-spaced shell edges spanning a fixed fraction of the truncation radius."""
    return jnp.geomspace(r_min_frac * r_edge, r_max_frac * r_edge, n_shells + 1)


def shell_mean(field, radii, edges):
    """Mean of ``field`` in each shell, **dropping** cells outside ``edges``.

    Returns:
        ``(centres, mean, counts)``. Cells outside ``[edges[0], edges[-1]]`` contribute to
        no shell -- deliberately, so the end bins are not catch-alls (see module docstring).
    """
    f = jnp.asarray(field).ravel()
    r = jnp.asarray(radii).ravel()
    n = edges.shape[0] - 1

    raw = jnp.searchsorted(edges, r) - 1
    valid = (raw >= 0) & (raw < n)
    idx = jnp.where(valid, raw, 0)
    w = valid.astype(f.dtype)

    total = jnp.zeros(n).at[idx].add(f * w)
    counts = jnp.zeros(n).at[idx].add(w)
    centres = jnp.sqrt(edges[1:] * edges[:-1])
    return centres, total / jnp.maximum(counts, 1.0), counts


def envelope_fidelity(rho_realized, radii, profile, *, n_shells: int = 5) -> FidelityResult:
    """Compare a realized gas density grid to its prescribed envelope, in shape.

    Args:
        rho_realized: gas density on the grid (``rho_cloud`` or ``rho_residual``).
        radii: cell-centre radii, from ``gravoturb.realization.envelope.radius_grid``.
        profile: the prescribed profile (needs ``.density`` and ``.r_edge``).
        n_shells: number of log-spaced shells.

    Returns:
        :class:`FidelityResult`. Feed a per-realization ``ratio`` into an ensemble mean
        before reading ``shape`` -- a single realization is far too noisy (ADR-0069).
    """
    edges = shell_edges(profile.r_edge, n_shells=n_shells)
    centres, mean, counts = shell_mean(rho_realized, radii, edges)
    ratio = mean / profile.density(centres)
    shape = ratio / jnp.mean(ratio)
    return FidelityResult(
        r=centres,
        counts=counts,
        ratio=ratio,
        shape=shape,
        max_deviation=jnp.max(jnp.abs(shape - 1.0)),
    )
