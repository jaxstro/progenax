"""Public model boundaries reject failed numerical states under tracing."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.imf import Maschberger
from progenax.profiles import EFFProfile, PlummerProfile
from progenax.profiles.limepy_multimass import _bin_imf, find_alpha_for_masses


def _imf_model(alpha, *, n_iter, adjoint=None, fit_tol=1e-6):
    return MultiComponentCluster.from_imf(
        Maschberger(alpha=alpha, m_min=0.1, m_max=20.0),
        n_comp=2,
        W0=5.0,
        g=1.0,
        delta=0.4,
        m_range=(0.1, 20.0),
        xi_max=30.0,
        n_ode_points=500,
        n_grid=100,
        n_iter=n_iter,
        fit_tol=fit_tol,
        adjoint=adjoint,
    )


@pytest.mark.parametrize("mode", ["eager", "jit", "forward", "reverse"])
def test_failed_imf_fit_cannot_return_model_fields(mode):
    """An exhausted fit must fail even when the caller reads an unrelated field."""
    adjoint = diffrax.ForwardMode() if mode == "forward" else None

    def construct(alpha):
        return _imf_model(alpha, n_iter=0, adjoint=adjoint).r_c

    transform = {
        "eager": lambda f: f,
        "jit": jax.jit,
        "forward": jax.jacfwd,
        "reverse": jax.jacrev,
    }[mode]
    with pytest.raises((ValueError, RuntimeError), match="fit did not converge"):
        jax.block_until_ready(transform(construct)(jnp.array(2.3)))


def test_failed_fit_diagnostics_remain_available():
    """The low-level solver retains the failed fractions and measured residual."""
    masses, budgets = _bin_imf(
        Maschberger(alpha=2.3, m_min=0.1, m_max=20.0), 2, (0.1, 20.0)
    )
    alpha, residual = find_alpha_for_masses(
        masses, budgets, 5.0, 1.0, 0.4, n_iter=0, xi_max=30.0, n_points=500
    )
    assert bool(jnp.all(jnp.isfinite(alpha)))
    assert float(residual) > 1e-6


@pytest.mark.parametrize("fit_tol", [jnp.nan, jnp.inf, -1.0])
@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_invalid_fit_tolerance_rejected(fit_tol, mode):
    def construct(tol):
        return _imf_model(jnp.array(2.3), n_iter=0, fit_tol=tol).r_c

    transform = jax.jit if mode == "jit" else lambda f: f
    with pytest.raises((ValueError, RuntimeError), match="fit_tol must"):
        jax.block_until_ready(transform(construct)(jnp.array(fit_tol)))


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_zero_fit_tolerance_requires_exact_convergence(mode):
    def construct(tol):
        return _imf_model(jnp.array(2.3), n_iter=0, fit_tol=tol).r_c

    transform = jax.jit if mode == "jit" else lambda f: f
    with pytest.raises((ValueError, RuntimeError), match="fit did not converge"):
        jax.block_until_ready(transform(construct)(jnp.array(0.0)))


@pytest.mark.parametrize("mode", ["eager", "jit", "forward", "reverse"])
def test_converged_imf_model_preserves_low_level_fractions(mode):
    adjoint = diffrax.ForwardMode() if mode == "forward" else None

    def raw_fractions(alpha):
        masses, budgets = _bin_imf(
            Maschberger(alpha=alpha, m_min=0.1, m_max=20.0), 2, (0.1, 20.0)
        )
        fractions, _ = find_alpha_for_masses(
            masses,
            budgets,
            5.0,
            1.0,
            0.4,
            n_iter=60,
            xi_max=30.0,
            n_points=500,
            adjoint=adjoint,
        )
        return fractions

    def model_fractions(alpha):
        return _imf_model(alpha, n_iter=60, adjoint=adjoint).alpha_j

    transform = {
        "eager": lambda f: f,
        "jit": jax.jit,
        "forward": jax.jacfwd,
        "reverse": jax.jacrev,
    }[mode]
    actual = transform(model_fractions)(jnp.array(2.3))
    expected = transform(raw_fractions)(jnp.array(2.3))
    np.testing.assert_array_equal(actual, expected)


def _density_model(*, fractions=None, ra=None, halo_radius=2.0):
    return MultiComponentCluster.from_density_profiles(
        [PlummerProfile(r_h=halo_radius), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)],
        jnp.array([0.6, 0.4]) if fractions is None else fractions,
        jnp.array([0.5, 1.0]),
        r_a_j=ra,
        n_r=500,
        n_e=100,
        n_grid=100,
    )


@pytest.mark.parametrize("mode", ["jit", "forward", "reverse"])
def test_unrealizable_density_model_cannot_return_unrelated_field(mode):
    def construct(ra):
        return _density_model(ra=jnp.stack([ra, jnp.inf])).r_t

    transform = {"jit": jax.jit, "forward": jax.jacfwd, "reverse": jax.jacrev}[mode]
    with pytest.raises((ValueError, RuntimeError), match="component 0"):
        jax.block_until_ready(transform(construct)(jnp.array(0.05)))


@pytest.mark.parametrize(
    "fractions, message",
    [
        ([0.0, 1.0], "strictly positive"),
        ([-0.1, 1.1], "strictly positive"),
        ([jnp.nan, 0.4], "strictly positive"),
        ([jnp.inf, 0.4], "strictly positive"),
        ([0.6, 0.6], "sum to 1"),
    ],
)
def test_invalid_density_amplitudes_rejected_under_jit(fractions, message):
    @jax.jit
    def construct(fractions):
        return _density_model(fractions=fractions).r_t

    with pytest.raises((ValueError, RuntimeError), match=message):
        jax.block_until_ready(construct(jnp.array(fractions)))


def test_zero_density_anisotropy_radius_rejected_under_jit():
    @jax.jit
    def construct(ra):
        return _density_model(ra=jnp.stack([ra, jnp.inf])).r_t

    with pytest.raises((ValueError, RuntimeError), match="r_a_j.*> 0"):
        jax.block_until_ready(construct(jnp.array(0.0)))


def test_nonfinite_density_df_rejected_under_jit():
    @jax.jit
    def construct(radius):
        return _density_model(halo_radius=radius).r_t

    with pytest.raises((ValueError, RuntimeError), match="component 0"):
        jax.block_until_ready(construct(jnp.array(jnp.nan)))


@pytest.mark.parametrize("mode", ["eager", "jit", "forward", "reverse"])
def test_realizable_density_model_preserves_raw_df(mode):
    from progenax.cluster.eddington_engine import build_engine_b_state

    def raw_df(ra):
        state, _ = build_engine_b_state(
            [PlummerProfile(r_h=2.0), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)],
            jnp.array([0.6, 0.4]),
            jnp.stack([ra, jnp.inf]),
            None,
            0.995,
            500,
            100,
        )
        return state.f_j_grid[0]

    def model_df(ra):
        return _density_model(ra=jnp.stack([ra, jnp.inf])).engine_b.f_j_grid[0]

    transform = {
        "eager": lambda f: f,
        "jit": jax.jit,
        "forward": jax.jacfwd,
        "reverse": jax.jacrev,
    }[mode]
    actual = transform(model_df)(jnp.array(5.0))
    expected = transform(raw_df)(jnp.array(5.0))
    np.testing.assert_array_equal(actual, expected)
