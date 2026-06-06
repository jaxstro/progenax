r"""Thin blackjax NUTS driver + bounded reparametrization for gravoturb_fdf inference (M2).

We sample the free parameters (mach, alpha, beta) with b fixed (the mach-b degeneracy: the
data constrains (mach,b) only via sigma_s^2). HMC runs in UNCONSTRAINED space; the bounds
mach>0, alpha>1 (tail integrable), beta>0 are imposed by a log / log-shift reparametrization,
with the log-Jacobian added to the log-density so the posterior is correct.

``run_nuts`` is a thin wrapper over blackjax window-adaptation + NUTS (verified against the
installed blackjax 1.5 API). numpyro is reserved for the later hierarchical population model
(Decision #7). JAX-native; the sampling loop is a ``jax.lax.scan``.
"""

import blackjax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def to_unconstrained(theta_c: Float[Array, " 3"]) -> Float[Array, " 3"]:
    r"""(mach, alpha, beta) -> unconstrained z = (log mach, log(alpha-1), log beta)."""
    mach, alpha, beta = theta_c
    return jnp.array([jnp.log(mach), jnp.log(alpha - 1.0), jnp.log(beta)])


def to_constrained(z: Float[Array, " 3"]) -> Float[Array, " 3"]:
    r"""Inverse map z -> (mach, alpha, beta) = (exp z0, 1+exp z1, exp z2); enforces the bounds."""
    return jnp.array([jnp.exp(z[0]), 1.0 + jnp.exp(z[1]), jnp.exp(z[2])])


def log_jacobian(z: Float[Array, " 3"]) -> Float[Array, ""]:
    r"""Log|d theta_c / d z| for the reparametrization = log(mach)+log(alpha-1)+log(beta) = sum(z).

    Added to the unconstrained log-density so the sampled posterior matches the prior+likelihood
    in the constrained (mach,alpha,beta) space."""
    return jnp.sum(z)


def run_nuts(logdensity_fn, init_position, key, n_warmup: int = 500, n_samples: int = 1000):
    r"""Sample ``logdensity_fn`` with blackjax NUTS (window adaptation -> sampling).

    ``logdensity_fn`` maps a position (any pytree/array) to a scalar log-density (differentiable).
    Returns an array of ``n_samples`` positions (leading axis = draws). Adaptation tunes the step
    size + diagonal mass matrix; the sampling loop is a fixed-length scan."""
    warmup_key, sample_key = jax.random.split(key)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
    (state, params), _ = warmup.run(warmup_key, init_position, num_steps=n_warmup)
    kernel = blackjax.nuts(logdensity_fn, **params)

    def one_step(state, k):
        state, _info = kernel.step(k, state)
        return state, state.position

    keys = jax.random.split(sample_key, n_samples)
    _, positions = jax.lax.scan(one_step, state, keys)
    return positions
