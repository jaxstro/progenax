r"""Fit an EFF profile to the EMERGENT stellar distribution (ADR-0062 validation hook).

gravoturb never imposes a stellar profile: stars are placed by gas-density rank, so the
young cluster's structure is *emergent* from the natal gas plus turbulence. This module
measures what came out, by fitting the Elson-Fall-Freeman form to the realized 3-D stellar
density:

    rho(r) = rho_0 (1 + r^2/a^2)^{-gamma/2}

**Convention.** ``progenax.EFFProfile.gamma`` -- and therefore ``gamma`` here -- is the
**3-D** density slope. Elson, Fall & Freeman (1987), ApJ 323, 54 quote *projected* slopes,
and for a power law the Abel projection gives ``Sigma ~ R^{-(gamma_3D - 1)}``. So

    gamma_3D = gamma_surface + 1

EFF87's abstract reports ``2.2 <~ gamma_surface <~ 3.2`` with a median of ``2.6`` for 10
young LMC clusters (individual fits in their Figs. 1-4 include NGC 1818 gamma=2.45,
NGC 1866 gamma=2.55, NGC 2004 gamma=2.20, NGC 1831 gamma=3.35). In OUR 3-D convention that
band is

    gamma_3D in [3.2, 4.2],  median 3.6      (EFF87 observed young LMC clusters)

Note that a Plummer sphere is *exactly* EFF with ``gamma_3D = 5``, which makes the
Plummer-envelope control a sharp quantitative reference rather than a vague one.

The fit is least-squares in log-density over log-spaced shells, solved with
``optimistix.LevenbergMarquardt`` -- JAX-native and differentiable in the star positions.
"""

from typing import NamedTuple

import jax.numpy as jnp
import optimistix
from jaxtyping import Array, Float

# EFF87 observed young-LMC-cluster band, converted to the 3-D slope convention.
EFF87_GAMMA3D_MIN = 3.2
EFF87_GAMMA3D_MAX = 4.2
EFF87_GAMMA3D_MEDIAN = 3.6

# A Plummer sphere is EFF with this 3-D slope, exactly.
PLUMMER_GAMMA3D = 5.0


class EFFFit(NamedTuple):
    """Best-fit EFF parameters for a realized stellar distribution.

    Attributes:
        a: scale radius, same length units as the input positions.
        gamma: 3-D density slope (see the module docstring for the EFF87 conversion).
        r: shell centres used in the fit.
        log_rho: measured log density in those shells.
        log_rho_model: the fitted model in those shells.
    """

    a: Float[Array, ""]
    gamma: Float[Array, ""]
    r: Float[Array, " k"]
    log_rho: Float[Array, " k"]
    log_rho_model: Float[Array, " k"]

    def residual_rms(self) -> Float[Array, ""]:
        """RMS of the log-density residual -- a goodness-of-fit handle."""
        return jnp.sqrt(jnp.mean((self.log_rho - self.log_rho_model) ** 2))


def radial_density_profile(
    positions: Float[Array, "n 3"],
    masses: Float[Array, " n"],
    *,
    n_bins: int = 24,
    r_min_frac: float = 0.02,
    r_max_frac: float = 0.98,
):
    """Spherical-shell mass density of a star field, on a log-spaced radial grid.

    The radial range is set from quantiles of the star radii rather than absolute numbers,
    so the same call works for any cluster size. Empty shells are dropped by the caller via
    the returned finite mask.

    Returns:
        ``(r_centre, rho, occupied)`` -- shell centres, density, and a boolean mask of
        shells containing at least one star.
    """
    r = jnp.linalg.norm(positions, axis=1)
    r_lo = jnp.quantile(r, r_min_frac)
    r_hi = jnp.quantile(r, r_max_frac)
    edges = jnp.geomspace(r_lo, r_hi, n_bins + 1)

    idx = jnp.clip(jnp.searchsorted(edges, r) - 1, 0, n_bins - 1)
    inside = (r >= edges[0]) & (r <= edges[-1])
    shell_mass = jnp.zeros(n_bins).at[idx].add(jnp.where(inside, masses, 0.0))
    counts = jnp.zeros(n_bins).at[idx].add(jnp.where(inside, 1.0, 0.0))

    volume = (4.0 / 3.0) * jnp.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = shell_mass / volume
    r_centre = jnp.sqrt(edges[1:] * edges[:-1])  # geometric centre, log-grid appropriate
    return r_centre, rho, counts > 0


def surface_density_profile(
    positions: Float[Array, "n 3"],
    masses: Float[Array, " n"],
    *,
    axis: int = 2,
    n_bins: int = 24,
    r_min_frac: float = 0.02,
    r_max_frac: float = 0.98,
):
    """Projected surface density in log-spaced annuli, viewing along ``axis``.

    This is what Elson, Fall & Freeman actually measured. Projection integrates through
    the cloud, which averages over 3-D turbulent clumps that no smooth profile can
    represent -- so it is the fair comparison, not merely a convenience.

    Returns:
        ``(R_centre, sigma, occupied)``.
    """
    keep = [i for i in range(3) if i != axis]
    xy = positions[:, jnp.array(keep)]
    R = jnp.linalg.norm(xy, axis=1)

    edges = jnp.geomspace(jnp.quantile(R, r_min_frac), jnp.quantile(R, r_max_frac), n_bins + 1)
    idx = jnp.clip(jnp.searchsorted(edges, R) - 1, 0, n_bins - 1)
    inside = (R >= edges[0]) & (R <= edges[-1])
    ann_mass = jnp.zeros(n_bins).at[idx].add(jnp.where(inside, masses, 0.0))
    counts = jnp.zeros(n_bins).at[idx].add(jnp.where(inside, 1.0, 0.0))

    area = jnp.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    R_centre = jnp.sqrt(edges[1:] * edges[:-1])
    return R_centre, ann_mass / area, counts > 0


def fit_eff_projected(
    positions: Float[Array, "n 3"],
    masses: Float[Array, " n"],
    *,
    axis: int = 2,
    n_bins: int = 24,
    a_init: float | None = None,
    gamma_init: float = 2.6,
) -> EFFFit:
    r"""Fit EFF87's own Eq. 1, ``Sigma = Sigma_0 (1 + R^2/a^2)^{-gamma/2}``, to a projection.

    The returned ``gamma`` is therefore the **surface** slope, directly comparable to
    EFF87's quoted ``2.2 <~ gamma <~ 3.2`` with no Abel conversion. A Plummer sphere
    projects to ``Sigma ~ (1 + R^2/a^2)^{-2}``, i.e. ``gamma_surface = 4`` exactly, which
    is the control's oracle.
    """
    R, sigma, occupied = surface_density_profile(
        positions, masses, axis=axis, n_bins=n_bins
    )
    if a_init is None:
        keep = [i for i in range(3) if i != axis]
        a_init = float(jnp.median(jnp.linalg.norm(positions[:, jnp.array(keep)], axis=1)))
    return _fit_log_profile(R, sigma, occupied, a_init, gamma_init)


def fit_eff(
    positions: Float[Array, "n 3"],
    masses: Float[Array, " n"],
    *,
    n_bins: int = 24,
    a_init: float | None = None,
    gamma_init: float = 3.5,
) -> EFFFit:
    r"""Fit ``rho = rho_0 (1 + r^2/a^2)^{-gamma/2}`` to a realized star field in 3-D.

    The returned ``gamma`` is the **3-D** slope; see the module docstring for the
    ``gamma_3D = gamma_surface + 1`` conversion to EFF87's quoted values. For a comparison
    in EFF87's own convention use :func:`fit_eff_projected`.

    Args:
        positions: star positions, shape (n, 3), centred on the cluster.
        masses: star masses, shape (n,).
        n_bins: number of log-spaced radial shells.
        a_init: initial scale radius; defaults to the median star radius.
        gamma_init: initial 3-D slope.

    Returns:
        :class:`EFFFit`.
    """
    r, rho, occupied = radial_density_profile(positions, masses, n_bins=n_bins)
    if a_init is None:
        a_init = float(jnp.median(jnp.linalg.norm(positions, axis=1)))
    return _fit_log_profile(r, rho, occupied, a_init, gamma_init)


def _fit_log_profile(r, rho, occupied, a_init, gamma_init) -> EFFFit:
    """Shared least-squares core for the 3-D and projected fits.

    Both fit ``(1 + r^2/a^2)^{-gamma/2}`` in LOG space -- the right metric for a profile
    spanning orders of magnitude, since a linear-space fit would be dominated entirely by
    the innermost bins. The amplitude is not a free parameter: for fixed ``(a, gamma)`` the
    optimal log-offset is the weighted mean residual, so it is profiled out analytically
    and only the two shape parameters are optimized.
    """
    # Drop empty bins by weighting rather than slicing, so shapes stay static.
    weight = jnp.where(occupied & (rho > 0), 1.0, 0.0)
    log_rho = jnp.log(jnp.where(rho > 0, rho, 1.0))

    def shape(params):
        """Unit-amplitude log EFF shape, with log a as the free variable (keeps a > 0)."""
        log_a, gamma = params
        return -0.5 * gamma * jnp.log1p((r / jnp.exp(log_a)) ** 2)

    def residuals(params, _):
        model = shape(params)
        # Profile out the amplitude: the optimal offset is the weighted mean residual.
        offset = jnp.sum(weight * (log_rho - model)) / jnp.sum(weight)
        return weight * (log_rho - (model + offset))

    solution = optimistix.least_squares(
        residuals,
        optimistix.LevenbergMarquardt(rtol=1e-8, atol=1e-10),
        jnp.array([jnp.log(a_init), gamma_init]),
        args=None,
        max_steps=256,
        throw=False,
    )
    log_a, gamma = solution.value
    model = shape(solution.value)
    offset = jnp.sum(weight * (log_rho - model)) / jnp.sum(weight)

    return EFFFit(
        a=jnp.exp(log_a),
        gamma=gamma,
        r=r,
        log_rho=jnp.where(weight > 0, log_rho, jnp.nan),
        log_rho_model=model + offset,
    )
