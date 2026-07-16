import jax
import jax.numpy as jnp
import pytest
from gravoturb.inference.priors import BM19Prior

pytestmark = pytest.mark.experimental


def _prior(**kw):
    return BM19Prior(
        m_range=(4.0, 20.0), alpha_range=(1.5, 3.0), beta_range=(2.0, 11 / 3), **kw
    )


def test_sample_within_support():
    pr = _prior()
    keys = jax.random.split(jax.random.PRNGKey(0), 2000)
    thetas = jax.vmap(pr.sample)(keys)  # (2000, 3) = (M, alpha, beta)
    M, alpha, beta = thetas[:, 0], thetas[:, 1], thetas[:, 2]
    assert jnp.all((M >= 4.0) & (M <= 20.0))
    assert jnp.all((alpha >= 1.5) & (alpha <= 3.0))
    assert jnp.all((beta >= 2.0) & (beta <= 11 / 3 + 1e-9))


def test_logdensity_finite_inside_minus_inf_outside():
    pr = _prior()
    th_in = jnp.array([5.0, 2.0, 3.0])  # (M, alpha, beta) all in range
    assert jnp.isfinite(pr.logpdf(th_in))
    for bad in (
        jnp.array([1.0, 2.0, 3.0]),  # M below range
        jnp.array([5.0, 0.9, 3.0]),  # alpha below range
        jnp.array([5.0, 4.5, 3.0]),  # alpha above range
        jnp.array([5.0, 2.0, 5.0]),
    ):  # beta above range
        assert pr.logpdf(bad) == -jnp.inf


def test_logpdf_grad_finite_inside():
    pr = _prior()
    g = jax.grad(lambda th: pr.logpdf(th))(jnp.array([5.0, 2.0, 3.0]))
    assert jnp.all(jnp.isfinite(g))


def test_logpdf_grad_finite_outside_support():
    # Locks the load-bearing double-where clamp: HMC can approach/exit the box, so the
    # gradient must stay finite (not nan) even where logpdf == -inf. Without the clamp on
    # the log arguments, all the other tests still pass while grad silently NaNs here.
    pr = _prior()
    for bad in (
        jnp.array([1e-6, 2.0, 3.0]),  # M far below range
        jnp.array([5.0, 0.5, 3.0]),  # alpha below range
        jnp.array([5.0, 4.5, 3.0]),  # alpha above range
        jnp.array([5.0, 2.0, 1e-6]),
    ):  # beta far below range
        g = jax.grad(lambda th: pr.logpdf(th))(bad)
        assert jnp.all(jnp.isfinite(g))


def test_sampled_M_loguniform_smoke():
    # log-uniform M => CDF values ~ Uniform(0,1) (sanity for the inverse-CDF sampler)
    pr = _prior()
    keys = jax.random.split(jax.random.PRNGKey(1), 5000)
    M = jax.vmap(pr.sample)(keys)[:, 0]
    u = (jnp.log(M) - jnp.log(4.0)) / (jnp.log(20.0) - jnp.log(4.0))
    us = jnp.sort(u)
    emp = jnp.arange(1, us.size + 1) / us.size
    assert jnp.max(jnp.abs(emp - us)) < 0.05
