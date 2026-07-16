r"""Fisher forecast for gravoturb (Milestone 1) -- the first science deliverable.

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

from gravoturb.inference.likelihood import data_vector


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


def alpha_fisher_info(alpha: Float[Array, ""], L: Float[Array, ""]) -> Float[Array, ""]:
    r"""Per-exceedance Fisher information for alpha of the truncated exponential on ``[0, L]``.

    The POT exceedance model ``p(x) = alpha e^{-alpha x} / (1 - e^{-alpha L})`` has ``log p`` linear
    in x, so the Fisher info is ``-d^2 log p / d alpha^2`` (no expectation integral):

        I(alpha) = 1/alpha^2  -  L^2 e^{-alpha L} / (1 - e^{-alpha L})^2.

    As ``L -> inf`` this relaxes to ``1/alpha^2`` (the EVT/Hill asymptote ``sigma(alpha)=alpha/sqrt(N)``);
    as ``L -> 0`` it vanishes (a squished tail carries no slope information). The truncation term is
    written with ``expm1`` so the denominator ``(1 - e^{-alpha L})^2`` stays accurate at small
    ``alpha*L``. At the demo grids ``L ~ 2-3`` the correction inflates sigma(alpha) by ~2-4%, so the
    forecast MUST use this corrected form, not the bare ``alpha/sqrt(N_tail)`` asymptote.
    """
    aL = alpha * L
    denom = jnp.expm1(-aL)  # = e^{-aL} - 1 = -(1 - e^{-aL}); denom^2 = (1 - e^{-aL})^2
    trunc = (L**2) * jnp.exp(-aL) / denom**2
    return 1.0 / alpha**2 - trunc


def sigma_alpha(
    alpha: Float[Array, ""], L: Float[Array, ""], n_tail: Float[Array, ""]
) -> Float[Array, ""]:
    r"""Forecast 1-sigma error on alpha from ``n_tail`` exceedances: ``1/sqrt(n_tail * I(alpha, L))``.

    The truncation-corrected EVT forecast (the honest "how big a gas map measures the natal tail
    slope" curve). ``sigma(alpha)*sqrt(n_tail) = 1/sqrt(I)`` -> ``alpha`` as ``L -> inf``.
    """
    return 1.0 / jnp.sqrt(n_tail * alpha_fisher_info(alpha, L))
