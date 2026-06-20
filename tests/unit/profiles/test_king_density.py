"""King lowered-Maxwellian volume density rho_hat(W) (Batch 2, B2.0).

The King Poisson equation needs the *volume* density obtained by integrating the
lowered-Maxwellian DF over velocity:

    rho(W) ∝ int_0^{sqrt(2W)} v^2 (e^{W - v^2/2} - 1) dv
           = const * [ e^W erf(sqrt W) - (2/sqrt(pi)) sqrt(W) (1 + 2W/3) ].

The earlier code used King's K-function (the incomplete-gamma / projected form) here,
which over-extends the profile by 2-30x in the outer regions. This pins the correct
relation against the direct velocity integral (a literature-free oracle).

Reference: King (1966), AJ 71, 64; Binney & Tremaine (2008) Eq. 4.131.
"""

import jax
import jax.numpy as jnp
import pytest

import progenax  # noqa: F401  (float64)
from progenax.profiles.king import king_lowered_maxwellian_density


def _direct_density_integral(W, nv=400_000):
    """Lowered-Maxwellian volume density at local potential W, by direct integration."""
    W = jnp.asarray(float(W))
    v = jnp.linspace(0.0, jnp.sqrt(2.0 * W), nv)
    return jnp.trapezoid(v**2 * (jnp.exp(W - v**2 / 2.0) - 1.0), v)


@pytest.mark.parametrize("W", [0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0])
def test_density_shape_matches_direct_velocity_integral(W):
    """The normalized rho_hat(W)/rho_hat(W_ref) must equal the direct integral ratio."""
    W_ref = 5.0
    analytic = float(
        king_lowered_maxwellian_density(W) / king_lowered_maxwellian_density(W_ref)
    )
    direct = float(_direct_density_integral(W) / _direct_density_integral(W_ref))
    rel = abs(analytic - direct) / (abs(direct) + 1e-30)
    assert rel < 1e-5, (
        f"W={W}: rho_hat ratio {analytic} vs direct {direct} (rel {rel:.2e})"
    )


def test_density_zero_and_gradient_finite_at_zero():
    assert float(king_lowered_maxwellian_density(0.0)) == 0.0
    g = jax.grad(king_lowered_maxwellian_density)(0.0)
    assert jnp.isfinite(g), f"grad rho_hat(0) non-finite: {g}"


def test_density_positive_and_increasing():
    Ws = jnp.array([0.5, 1.0, 2.0, 4.0, 7.0, 10.0])
    rho = jax.vmap(king_lowered_maxwellian_density)(Ws)
    assert jnp.all(rho > 0.0)
    assert jnp.all(jnp.diff(rho) > 0.0), "rho_hat(W) must increase with W"
