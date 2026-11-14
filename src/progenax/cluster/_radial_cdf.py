"""Engine A radial CDF and inverse from nonnegative linear density segments.

The grid must be nondecreasing; repeated nodes have zero width and mass.
Cumulative values must be its density's
trapezoidal integral, with positive total mass and the same normalization.
"""

import jax.numpy as jnp


def radial_grid(potential_radii, tidal_radius, n_intervals, *, subdivisions=16):
    """Merge requested radial resolution with subdivided potential intervals.

    Nodes beyond the tidal radius collapse onto the boundary to retain a static
    shape. The requested grid supplies at least n_intervals positive-width
    cells; potential nodes and subdivisions add resolution where needed.
    """
    if n_intervals < 1:
        raise ValueError("n_grid must be a positive interval count")
    edges = jnp.concatenate((jnp.zeros(1), jnp.minimum(potential_radii, tidal_radius)))
    fractions = jnp.arange(subdivisions) / float(subdivisions)
    mandatory = (edges[:-1, None] + jnp.diff(edges)[:, None] * fractions).ravel()
    requested = tidal_radius * jnp.linspace(0.0, 1.0, n_intervals + 1) ** 2
    return jnp.sort(jnp.concatenate((mandatory, requested, edges[-1:])))


def radial_cdf(radius, grid, density, cumulative):
    """Integrate radial density within a cell; arrays share one normalization."""
    r = jnp.clip(radius, grid[0], grid[-1])
    i = jnp.clip(jnp.searchsorted(grid, r, side="right") - 1, 0, grid.size - 2)
    d = r - grid[i]
    width = grid[i + 1] - grid[i]
    slope = (density[i + 1] - density[i]) / jnp.where(width > 0.0, width, 1.0)
    value = (cumulative[i] + density[i] * d + 0.5 * slope * d**2) / cumulative[-1]
    return jnp.where(r <= grid[0], 0.0, jnp.where(r >= grid[-1], 1.0, value))


def radial_quantile(probability, grid, density, cumulative):
    """Invert the integrated linear density, including support endpoints.

    Interior derivatives retain grid and density dependence. Endpoint derivatives
    with respect to probability are not covered by this numerical contract.
    """
    target = jnp.clip(probability, 0.0, 1.0) * cumulative[-1]
    i = jnp.clip(
        jnp.searchsorted(cumulative, target, side="right") - 1, 0, grid.size - 2
    )
    area = target - cumulative[i]
    width = grid[i + 1] - grid[i]
    slope = (density[i + 1] - density[i]) / jnp.where(width > 0.0, width, 1.0)
    discriminant = density[i] ** 2 + 2 * slope * area
    # Avoid an inactive sqrt(0) cotangent at a zero-density support endpoint.
    root = jnp.sqrt(jnp.where(discriminant > 0.0, discriminant, 1.0))
    root = jnp.where(discriminant > 0.0, root, 0.0)
    denominator = density[i] + root
    d = 2 * area / jnp.where(denominator > 0.0, denominator, 1.0)
    return jnp.where(
        probability <= 0.0,
        grid[0],
        jnp.where(probability >= 1.0, grid[-1], grid[i] + d),
    )


def potential_interp(query, radii, potential, *, left, right):
    """Monotone cubic potential with explicit exterior values."""
    from jaxstro.numerics.interpolation import monotone_cubic_interp

    value = monotone_cubic_interp(radii, potential, query)
    return jnp.where(query < radii[0], left, jnp.where(query > radii[-1], right, value))


def radial_second_moments(grid, density):
    """Exact normalized second moments of piecewise-linear radial densities."""
    left, right = grid[:-1], grid[1:]
    width = right - left
    integral = jnp.sum(
        width
        / 12
        * (
            density[:, :-1] * (3 * left**2 + 2 * left * right + right**2)
            + density[:, 1:] * (left**2 + 2 * left * right + 3 * right**2)
        ),
        axis=1,
    )
    return integral / jnp.trapezoid(density, grid, axis=1)


def inverse_cdf_second_moments(grid, cumulative):
    """Integrate squared linear inverse CDFs for uniform probabilities on [0, 1].

    Rows must be nondecreasing CDFs in [0, 1]. Repeated CDF knots carry
    zero probability; endpoint terms match ``jnp.interp`` constant tails.
    """
    left, right = grid[:-1], grid[1:]
    interval_mean = (left**2 + left * right + right**2) / 3
    return (
        jnp.sum(jnp.diff(cumulative, axis=1) * interval_mean, axis=1)
        + cumulative[:, 0] * grid[0] ** 2
        + (1 - cumulative[:, -1]) * grid[-1] ** 2
    )


def component_radial_quantile(probability, component, grid, density, cumulative):
    """Batched inverse without gathering a complete CDF row for every star."""
    import jax

    target = jnp.clip(probability, 0.0, 1.0) * cumulative[component, -1]
    size = grid.size

    def bisect(_, bounds):
        low, high = bounds
        middle = (low + high) // 2
        value = cumulative[component, jnp.minimum(middle, size - 1)]
        move_right = value <= target
        return jnp.where(move_right, middle + 1, low), jnp.where(
            move_right, high, middle
        )

    low, _ = jax.lax.fori_loop(
        0,
        size.bit_length(),
        bisect,
        (jnp.zeros_like(component), jnp.full_like(component, size)),
    )
    index = jnp.clip(low - 1, 0, size - 2)
    area = target - cumulative[component, index]
    width = grid[index + 1] - grid[index]
    left_density = density[component, index]
    slope = (density[component, index + 1] - left_density) / jnp.where(
        width > 0.0, width, 1.0
    )
    discriminant = left_density**2 + 2 * slope * area
    root = jnp.sqrt(jnp.where(discriminant > 0.0, discriminant, 1.0))
    root = jnp.where(discriminant > 0.0, root, 0.0)
    denominator = left_density + root
    offset = 2 * area / jnp.where(denominator > 0.0, denominator, 1.0)
    return jnp.where(
        probability <= 0.0,
        grid[0],
        jnp.where(probability >= 1.0, grid[-1], grid[index] + offset),
    )


def component_linear_quantile(probability, component, grid, cumulative):
    """Linear inverse CDF using indexed knots, without per-draw row gathers."""
    import jax

    size = grid.size

    def step(_, bounds):
        low, high = bounds
        middle = (low + high) // 2
        right = cumulative[component, jnp.minimum(middle, size - 1)] <= probability
        return jnp.where(right, middle + 1, low), jnp.where(right, high, middle)

    low, _ = jax.lax.fori_loop(
        0,
        size.bit_length(),
        step,
        (jnp.zeros_like(component), jnp.full_like(component, size)),
    )
    i = jnp.clip(low - 1, 0, size - 2)
    left = cumulative[component, i]
    width = cumulative[component, i + 1] - left
    value = grid[i] + (probability - left) * (grid[i + 1] - grid[i]) / jnp.where(
        width != 0, width, 1.0
    )
    value = jnp.where(width == 0, grid[i], value)
    return jnp.where(
        probability < cumulative[component, 0],
        grid[0],
        jnp.where(probability > cumulative[component, -1], grid[-1], value),
    )
