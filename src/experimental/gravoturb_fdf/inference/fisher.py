r"""Fisher forecast for gravoturb_fdf (Milestone 1) -- the first science deliverable.

``F = J^T Cinv J``, ``J = d(data_vector)/d theta`` via ``jax.jacobian`` at the fiducial theta;
``Cinv`` is the fixed mock precision (Hartlap-corrected). Marginal errors
``sigma(theta_i) = sqrt[(F^-1)_ii]``. They scale as ``1/sqrt(V_survey)`` (independent survey
volumes add Fisher information).

Identifiability note: the predicted statistics depend on (mach,b) ONLY through
``sigma_s^2 = ln(1+(b*mach)^2)``, so the full 4-param Fisher is rank-3 singular -- the data
cannot break the mach-b degeneracy. Forecasts therefore fix b (the driving parameter) and
constrain (mach, alpha, beta) -- the "beta -> mach given driving" science framing -- via the
``free`` parameter-index selection.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gravoturb_fdf.inference.likelihood import data_vector


def fisher_matrix(
    theta_fid: Float[Array, " 4"],
    precision: Float[Array, " d d"],
    free: tuple[int, ...] = (0, 1, 2, 3),
    **cfg,
) -> Float[Array, " f f"]:
    r"""Fisher matrix ``F = J^T Cinv J`` over the free parameters ``free`` (indices into
    theta=(mach,b,alpha,beta)), holding the rest at ``theta_fid``. ``cfg`` are the
    :func:`data_vector` keyword args. Symmetric; positive-definite iff the free params are
    identifiable (fix b to break the mach-b degeneracy)."""
    theta_fid = jnp.asarray(theta_fid)
    idx = jnp.asarray(free)

    def dv(free_vals):
        return data_vector(theta_fid.at[idx].set(free_vals), **cfg)

    J = jax.jacobian(dv)(theta_fid[idx])  # (d, n_free)
    return J.T @ (precision @ J)


def marginal_errors(F: Float[Array, " f f"]) -> Float[Array, " f"]:
    r"""Marginalised 1-sigma errors ``sigma_i = sqrt[(F^-1)_ii]`` (the Cramer-Rao bound)."""
    return jnp.sqrt(jnp.diag(jnp.linalg.inv(F)))
