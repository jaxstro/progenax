r"""Proper priors over the BM19 free inference parameters (M, alpha, beta), for SBC.

Simulation-Based Calibration (Talts et al. 2018) requires drawing theta from a *proper*
prior; the existing HMC driver (``hmc.py``) carries only the bounded reparametrization +
``log_jacobian`` (implicitly flat-in-theta = improper). :class:`BM19Prior` supplies that
proper prior over the three free parameters ``(M, alpha, beta)`` with the turbulence driving
parameter ``b`` held *fixed* (the data constrains (M, b) only through sigma_s^2, so b is not
identifiable on its own and is fixed in this inference layer).

Priors (Option B — unconditional; Anna-approved 2026-06-06)
-----------------------------------------------------------
- ``M     ~ LogUniform[m_lo, m_hi]``        (sonic Mach number; default 4 .. 20)
- ``alpha ~ Uniform[alpha_lo, alpha_hi]``   (BM19 PDF tail slope; default 1.1 .. 4.0)
- ``beta  ~ LogUniform[beta_lo, beta_hi]``  (GRF power-spectrum slope; default 2 .. 11/3)

The ``M`` lower bound is 4 because the count/ℳ channel is calibrated for ℳ≥4; below that
the field is transonic and shot-noise-dominated (dense-core regime), outside the supersonic
GMC range (ℳ~5–20) and not analytically captured by the cell-density model.

The alpha prior is **unconditional** (a plain box ``[alpha_lo, alpha_hi]``, no M-coupling).
POT-validity of the tail block (the requirement ``s_t(theta) = (alpha - 1/2) sigma_s^2 <=
s_thr``) is **not** enforced here: the SBC driver (``sbc.py``) sets the exceedance threshold
adaptively per trial as ``s_thr = s_t(theta*) + margin`` (exactly as AC16 does), and the POT
block (``tail_exceedance_loglike``) is **shift-immune** in ``s_thr`` (the lognormal norm
cancels), so no validity barrier is needed in the log-density. Folding POT-validity into the
prior instead would make ``alpha`` conditional on
``M`` and collapse the alpha range at high Mach number (wide sigma_s^2 -> low transition),
so it is deliberately left out of the prior. ``alpha > 1`` keeps the mass-weighted tail
integral convergent.

JAX-native, differentiable, immutable (Equinox module). ``sample`` uses inverse-CDF sampling
so it is ``jax.vmap``-able over PRNG keys, and ``logpdf`` is finite & differentiable strictly
*inside* the support (only out-of-support points return ``-inf``) for use as an HMC prior term.
"""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class BM19Prior(eqx.Module):
    r"""Unconditional proper prior over ``(M, alpha, beta)`` (see module docstring).

    Attributes
    ----------
    m_range : (m_lo, m_hi) support of the log-uniform Mach-number prior.
    alpha_range : (alpha_lo, alpha_hi) support of the uniform BM19 tail-slope prior.
    beta_range : (beta_lo, beta_hi) support of the log-uniform beta prior.
    """

    m_range: Tuple[float, float]
    alpha_range: Tuple[float, float]
    beta_range: Tuple[float, float]

    def __init__(
        self,
        m_range: Tuple[float, float] = (4.0, 20.0),
        alpha_range: Tuple[float, float] = (1.1, 4.0),
        beta_range: Tuple[float, float] = (2.0, 11.0 / 3.0),
    ):
        # Static python float tuples (used for the box support + log-CDF normalization).
        self.m_range = (float(m_range[0]), float(m_range[1]))
        self.alpha_range = (float(alpha_range[0]), float(alpha_range[1]))
        self.beta_range = (float(beta_range[0]), float(beta_range[1]))

    def sample(self, key: PRNGKeyArray) -> Float[Array, " 3"]:
        r"""Draw one ``(M, alpha, beta)`` via inverse-CDF (``jax.vmap``-able over keys).

        ``M`` and ``beta`` are log-uniform: ``x = lo * (hi/lo)^u`` with ``u ~ U(0, 1)``.
        ``alpha`` is uniform: ``alpha = alpha_lo + u (alpha_hi - alpha_lo)``.
        """
        k_m, k_a, k_b = jax.random.split(key, 3)
        u_m, u_a, u_b = (
            jax.random.uniform(k_m),
            jax.random.uniform(k_a),
            jax.random.uniform(k_b),
        )

        m_lo, m_hi = self.m_range
        a_lo, a_hi = self.alpha_range
        beta_lo, beta_hi = self.beta_range

        M = m_lo * (m_hi / m_lo) ** u_m
        alpha = a_lo + u_a * (a_hi - a_lo)
        beta = beta_lo * (beta_hi / beta_lo) ** u_b

        return jnp.array([M, alpha, beta])

    def logpdf(self, theta: Float[Array, " 3"]) -> Float[Array, ""]:
        r"""Log prior density at ``theta = (M, alpha, beta)``.

        ``theta`` is the **3 free params** ``[M, alpha, beta]`` — NOT the 4-vector
        ``[M, b, alpha, beta]`` the likelihood/barrier take (``b`` is fixed and lives
        outside the prior). The SBC driver assembles both from ``to_constrained(z)``.

        Sum of per-parameter log-densities:

        - log-uniform on ``x``: ``-log(x) - log(log(hi) - log(lo))`` for ``x in [lo, hi]``;
        - uniform on ``alpha in [alpha_lo, alpha_hi]``: ``-log(alpha_hi - alpha_lo)``.

        Returns ``-inf`` for any out-of-support point (``M``, ``alpha`` or ``beta`` outside
        its box). The in-support branch is finite and cleanly differentiable: the finite
        log-density is computed on *clamped, strictly-positive* arguments so the masked
        ``-inf`` branch contributes no ``nan`` gradient (the standard double-``where`` trick).
        """
        M, alpha, beta = theta[0], theta[1], theta[2]
        m_lo, m_hi = self.m_range
        a_lo, a_hi = self.alpha_range
        beta_lo, beta_hi = self.beta_range

        in_support = (
            (M >= m_lo)
            & (M <= m_hi)
            & (alpha >= a_lo)
            & (alpha <= a_hi)
            & (beta >= beta_lo)
            & (beta <= beta_hi)
        )

        # Clamp arguments that feed logs so the finite branch never sees a non-positive
        # value (which would inject nan into the gradient via the unused -inf branch).
        M_safe = jnp.clip(M, m_lo, m_hi)
        beta_safe = jnp.clip(beta, beta_lo, beta_hi)

        log_norm_m = jnp.log(jnp.log(m_hi) - jnp.log(m_lo))
        log_norm_beta = jnp.log(jnp.log(beta_hi) - jnp.log(beta_lo))

        logpdf_m = -jnp.log(M_safe) - log_norm_m
        logpdf_alpha = -jnp.log(a_hi - a_lo)
        logpdf_beta = -jnp.log(beta_safe) - log_norm_beta

        logpdf_finite = logpdf_m + logpdf_alpha + logpdf_beta

        return jnp.where(in_support, logpdf_finite, -jnp.inf)
