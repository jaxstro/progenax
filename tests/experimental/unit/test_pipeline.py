"""End-to-end FDF field pipeline + AC6 cornerstone smoke (spec §8, §3.5-3.6).

build_fdf_field composes GRF(β) → rank-copula → BM19 marginal and reports the
realized dense mass fraction (hard s>s_t indicator, the κ→∞ limit of f_tail_actual).
AC6 asks: does that realized fraction reproduce BM19 f_dense? Single 64³ here
(|bias|<5%); the 128³ ensemble (<1%) lives in the acceptance script.
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.experimental


def test_build_fdf_field_struct():
    """Field carries s (right shape), s_t and f_dense matching the theory functions."""
    from gravoturb_fdf.field.pipeline import build_fdf_field
    from gravoturb_fdf.theory.bm19 import (
        f_dense_bm19_full,
        sigma_s_squared,
        transition_density,
    )

    fld = build_fdf_field(
        mach=10.0,
        b=0.4,
        alpha=2.0,
        beta=3.667,
        shape=(32, 32, 32),
        key=jax.random.PRNGKey(0),
    )
    assert fld.s.shape == (32, 32, 32)
    s_t = float(transition_density(2.0, sigma_s_squared(10.0, 0.4)))
    assert float(fld.s_t) == pytest.approx(s_t, rel=1e-9)
    assert float(fld.f_dense) == pytest.approx(
        float(f_dense_bm19_full(10.0, 0.4, 2.0)), rel=1e-9
    )


def test_build_fdf_field_marginal_mean_density():
    """Mass-conserving copula gives ⟨e^s⟩ = bm19_mean_density (BM19-consistent ρ_0, ≳1)."""
    from gravoturb_fdf.field.pipeline import build_fdf_field
    from gravoturb_fdf.theory.pdf import bm19_mean_density

    fld = build_fdf_field(
        mach=8.0,
        b=0.5,
        alpha=1.8,
        beta=3.5,
        shape=(48, 48, 48),
        key=jax.random.PRNGKey(1),
    )
    assert float(jnp.mean(jnp.exp(fld.s))) == pytest.approx(
        float(bm19_mean_density(8.0, 0.5, 1.8)), rel=1e-6
    )


def test_cornerstone_single_64():
    """AC6 single-realization: |f_dense_realized − f_dense| < 0.5% at 64³ (mass-conserving)."""
    from gravoturb_fdf.field.pipeline import build_fdf_field

    fld = build_fdf_field(
        mach=10.0,
        b=0.4,
        alpha=2.0,
        beta=3.667,
        shape=(64, 64, 64),
        key=jax.random.PRNGKey(7),
    )
    rel_bias = abs(float(fld.f_dense_realized) - float(fld.f_dense)) / float(
        fld.f_dense
    )
    assert rel_bias < 0.005


def test_cloud_to_stars_end_to_end():
    """Full cloud→stars: build field then sample N⋆ positions in the box."""
    from gravoturb_fdf.field.pipeline import build_fdf_field, cloud_to_stars

    fld = build_fdf_field(
        mach=8.0,
        b=0.5,
        alpha=1.8,
        beta=3.5,
        shape=(32, 32, 32),
        key=jax.random.PRNGKey(2),
    )
    pos = cloud_to_stars(fld, f_sub=0.3, n_stars=800, key=jax.random.PRNGKey(3))
    assert pos.shape == (800, 3)
    assert jnp.all(jnp.isfinite(pos))
