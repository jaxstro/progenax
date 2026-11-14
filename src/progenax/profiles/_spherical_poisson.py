"""Regular center limits for the King-radius spherical Poisson equation."""

import jax.numpy as jnp


def regular_center_state(W0, xi_start, central_source=1.0):
    """Return W and W' through order xi^2 and xi for a regular central source.

    For W'' + 2 W'/xi = -9 f, spherical regularity gives W''(0)=-3 f(0).
    The omitted terms are O(xi^4) in W and O(xi^3) in W'. This requires a
    smooth spherical source and a start small compared with its variation scale.
    """
    return jnp.stack(
        (W0 - 1.5 * central_source * xi_start**2, -3.0 * central_source * xi_start)
    )


def poisson_second_derivative(xi, dW_dxi, source):
    """Evaluate W'' with the regular -3 f(0) limit at exactly zero radius.

    Use the ordinary equation at every positive radius. The safe denominator
    also keeps the inactive branch finite under JVP, VJP, JIT, and vmap.
    At xi=0 the state is assumed regular; this is not an extension to arbitrary
    nonsymmetric data or a contract for derivatives in radius at the origin.
    """
    xi = jnp.asarray(xi)
    safe_xi = jnp.where(xi != 0.0, xi, 1.0)
    ordinary = -9.0 * source - 2.0 * dW_dxi / safe_xi
    return jnp.where(xi == 0.0, -3.0 * source, ordinary)
