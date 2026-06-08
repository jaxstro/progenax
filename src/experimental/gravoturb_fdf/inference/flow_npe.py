r"""Flow-based Neural Posterior Estimation (NPE) for the low-N beta likelihood (Path C).

Where the analytic shot model hits a wall (the projected-density marginal's var-skew relation crosses
over with beta, so no simple positive family fits -- see the projected-beta-inference theory page and
the retrospective), a conditional normalizing flow LEARNS the marginal implicitly. We train an
amortized posterior

    q_phi(z | s),    z = logit-reparam of beta,    s = whitened log+ band-power summary,

on (beta*, s) pairs simulated from the prior at the survey's N_stars (the SAME observable as the
analytic path, so the two are directly comparable). For an observed ``s`` the flow IS the beta
posterior (amortized: train once, apply to every SBC trial). SBC validates calibration at low N.

Thin wrappers over ``flowjax`` (conditional masked-autoregressive flow). NOTE: flowjax requires
NEW-STYLE typed JAX PRNG keys (``jax.random.key``), unlike the old ``jax.random.PRNGKey`` used by the
simulator -- pass typed keys to the functions here.

JAX-native; the flow is differentiable. Experimental, repo-only.
"""

import jax.numpy as jnp
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
from jax.nn import sigmoid
from jax.scipy.special import logit
from jaxtyping import Array, Float


def beta_to_z(beta: Float[Array, " ..."], beta_lo: float, beta_hi: float) -> Float[Array, " ..."]:
    r"""Unconstrained logit reparam of a log-uniform beta: ``z = logit((ln b - ln b_lo)/Δln b)``."""
    u = (jnp.log(beta) - jnp.log(beta_lo)) / (jnp.log(beta_hi) - jnp.log(beta_lo))
    return logit(u)


def z_to_beta(z: Float[Array, " ..."], beta_lo: float, beta_hi: float) -> Float[Array, " ..."]:
    r"""Inverse of :func:`beta_to_z`: ``beta = b_lo (b_hi/b_lo)^{sigmoid(z)}`` (log-uniform support)."""
    u = sigmoid(z)
    return jnp.exp(jnp.log(beta_lo) + u * (jnp.log(beta_hi) - jnp.log(beta_lo)))


def whiten_stats(s_train: Float[Array, " n d"]) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
    r"""Per-dimension (mean, std) of the training summaries (truth-independent normalisation)."""
    return jnp.mean(s_train, axis=0), jnp.std(s_train, axis=0)


def whiten(
    s: Float[Array, " ... d"], mean: Float[Array, " d"], std: Float[Array, " d"]
) -> Float[Array, " ... d"]:
    r"""Standardise summaries with fixed ``(mean, std)`` (computed once on the training set)."""
    return (s - mean) / std


def build_npe_flow(key, summary_dim: int, nn_width: int = 64, flow_layers: int = 6):
    r"""Conditional masked-autoregressive flow ``q(z | s)`` over 1-D ``z`` conditioned on ``s``.

    ``key`` must be a new-style typed PRNG key (``jax.random.key``). Standard-normal base; the
    conditioner is an MLP of width ``nn_width``. Returns a flowjax distribution to be trained."""
    return masked_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(1)),
        cond_dim=summary_dim,
        nn_width=nn_width,
        flow_layers=flow_layers,
    )


def train_npe(
    key,
    flow,
    z_train: Float[Array, " n 1"],
    s_train: Float[Array, " n d"],
    learning_rate: float = 1e-3,
    max_epochs: int = 200,
    batch_size: int = 256,
):
    r"""Fit the conditional flow to ``(z, s)`` pairs by max-likelihood (``flowjax.train.fit_to_data``).

    Returns the trained flow. ``key`` typed; ``z_train`` shape ``(n,1)``, ``s_train`` shape ``(n,d)``."""
    flow, _losses = fit_to_data(
        key,
        flow,
        data=(z_train, s_train),
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        show_progress=False,
    )
    return flow


def npe_posterior_z(key, flow, s_obs: Float[Array, " d"], n_samples: int = 2000) -> Float[Array, " m"]:
    r"""Draw ``n_samples`` posterior ``z`` for a (whitened) observed summary ``s_obs`` (typed ``key``)."""
    return flow.sample(key, (n_samples,), condition=s_obs).reshape(-1)
