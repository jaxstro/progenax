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


def run_nuts_diagnostic(
    logdensity_fn,
    init_position,
    key,
    n_warmup: int = 500,
    n_samples: int = 1000,
    n_chains: int = 4,
):
    r"""Multi-chain blackjax NUTS that KEEPS the per-step diagnostic ``info``.

    A sibling of :func:`run_nuts` (which discards ``info`` and runs a single chain). This driver
    runs ``n_chains`` independently-warmed-up chains from *dispersed* initial positions (so that
    R-hat across chains is meaningful), and retains the per-step NUTS diagnostics needed by the
    convergence checks (Task 3 / AC19): divergences, tree depth, and the Hamiltonian energy
    (for BFMI). ``run_nuts`` is unchanged -- AC16 depends on it.

    Dispersion: chain ``c`` starts at ``init_position + 0.5 * N(0, 1)`` with a distinct per-chain
    fold-in, so the chains do not all begin at the same point (a degenerate start would make R-hat
    artificially small).

    blackjax 1.x ``NUTSInfo`` field mapping (verified against the installed version via
    ``info._fields``): divergences <- ``is_divergent`` (boolean), tree_depth <-
    ``num_trajectory_expansions`` (number of trajectory doublings = the tree depth), energy <-
    ``energy``. The configured maximum tree depth is blackjax's NUTS default ``max_num_doublings``.

    JAX-native: per-chain warmup + a fixed-length ``jax.lax.scan`` sampling loop, ``jax.vmap``-ed
    over the dispersed per-chain keys and inits (no python loop in the hot path, no mutation).

    Parameters
    ----------
    logdensity_fn : callable
        position (pytree/array) -> scalar log-density (differentiable).
    init_position : Array
        Central initial position; per-chain inits are dispersed around it.
    key : PRNGKey
    n_warmup, n_samples : int
        Window-adaptation steps and post-warmup samples per chain.
    n_chains : int
        Number of independent chains (>= 4 for AC19).

    Returns
    -------
    dict with keys:
        ``positions``      (n_chains, n_samples, *position.shape) float
        ``divergences``    (n_chains, n_samples) bool         -- ``is_divergent``
        ``tree_depth``     (n_chains, n_samples) int          -- ``num_trajectory_expansions``
        ``energy``         (n_chains, n_samples) float        -- ``energy``
        ``step_size``      (n_chains,) float                  -- warmup-tuned step size per chain
        ``max_tree_depth`` int                                -- blackjax NUTS max_num_doublings
    """
    max_tree_depth = 10  # blackjax NUTS default `max_num_doublings` (verified against installed API)

    chain_keys = jax.random.split(key, n_chains)

    def run_one_chain(chain_key, init_pos):
        disperse_key, warmup_key, sample_key = jax.random.split(chain_key, 3)
        init = init_pos + 0.5 * jax.random.normal(disperse_key, jnp.shape(init_pos))

        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
        (state, params), _ = warmup.run(warmup_key, init, num_steps=n_warmup)
        kernel = blackjax.nuts(logdensity_fn, **params)

        def one_step(state, k):
            state, info = kernel.step(k, state)
            diagnostics = (
                state.position,
                info.is_divergent,
                info.num_trajectory_expansions,
                info.energy,
            )
            return state, diagnostics

        keys = jax.random.split(sample_key, n_samples)
        _, (positions, divergences, tree_depth, energy) = jax.lax.scan(one_step, state, keys)
        return positions, divergences, tree_depth, energy, params["step_size"]

    # Distinct per-chain fold-in for the dispersed inits + warmup/sampling streams.
    init_keys = jax.vmap(lambda c: jax.random.fold_in(key, c))(jnp.arange(n_chains))
    positions, divergences, tree_depth, energy, step_size = jax.vmap(
        run_one_chain
    )(init_keys, jnp.broadcast_to(init_position, (n_chains, *jnp.shape(init_position))))

    return {
        "positions": positions,
        "divergences": divergences,
        "tree_depth": tree_depth,
        "energy": energy,
        "step_size": step_size,
        "max_tree_depth": max_tree_depth,
    }
