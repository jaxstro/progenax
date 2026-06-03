"""King K-function gradient & JIT-safety (audit finding C2).

C2: ``king_K_function`` used ``sqrt(maximum(W, 0))`` guarded only on the *value*
(``where(W_safe < 1e-10, 0, K)``). The ``1/(2*sqrt(W))`` derivative of ``sqrt`` at
W=0 still flowed through the dead branch (the classic where-NaN trap), so
``jax.grad(king_K_function)(0.0)`` returned ``nan``. The argument ``W0 - psi`` hits
exactly 0 at the cluster center and at the tidal radius, so the singularity is
unavoidable in any King-based inference.

Separately, ``_find_tidal_radius`` returned ``float(xi_t)``, concretizing the value
and making ``KingProfile.from_W0_rc`` non-JIT-able.

Analytic derivative used as ground truth:
    K(W)   = erf(sqrt(W)) - (2/sqrt(pi)) sqrt(W) e^{-W}
    dK/dW  = (2/sqrt(pi)) sqrt(W) e^{-W}        (so dK/dW = 0 exactly at W=0)
"""

import jax
import jax.numpy as jnp
import pytest

import progenax  # noqa: F401  (enables float64 at import)
from progenax.profiles.king import KingProfile, king_K_function


def _analytic_dKdW(W):
    return (2.0 / jnp.sqrt(jnp.pi)) * jnp.sqrt(W) * jnp.exp(-W)


def test_king_K_gradient_finite_at_zero():
    g = jax.grad(king_K_function)(0.0)
    assert jnp.isfinite(g), f"grad K at W=0 is non-finite: {g} (where-NaN trap, C2)"


def test_king_K_gradient_is_zero_at_zero():
    # dK/dW = (2/sqrt(pi)) sqrt(W) e^{-W} -> 0 as W -> 0
    g = float(jax.grad(king_K_function)(0.0))
    assert abs(g) < 1e-12, f"grad K at W=0 should be 0, got {g}"


def test_king_K_gradient_finite_at_negative_W():
    # W < 0 is unphysical; K=0 there, gradient must still be finite (not nan)
    g = jax.grad(king_K_function)(-1.0)
    assert jnp.isfinite(g), f"grad K at W=-1 is non-finite: {g}"


@pytest.mark.parametrize("W", [0.5, 1.0, 3.0, 7.0])
def test_king_K_gradient_matches_finite_difference_and_analytic(W):
    g_ad = float(jax.grad(king_K_function)(W))
    g_analytic = float(_analytic_dKdW(W))

    # central finite difference with an h-sweep; take the best match
    best = min(
        abs(g_ad - (float(king_K_function(W + h)) - float(king_K_function(W - h))) / (2 * h))
        for h in (1e-4, 1e-5, 1e-6)
    )
    rel = abs(g_ad - g_analytic) / (abs(g_ad) + abs(g_analytic) + 1e-30)
    assert rel < 1e-5, f"W={W}: AD grad {g_ad} vs analytic {g_analytic} (rel {rel:.2e})"
    assert best < 1e-5, f"W={W}: AD grad disagrees with finite difference (best abs err {best:.2e})"


def test_from_W0_rc_is_jittable():
    # Must not raise a ConcretizationTypeError (was blocked by float(xi_t))
    r_t = jax.jit(lambda w: KingProfile.from_W0_rc(w, 1.0).r_t)(7.0)
    assert jnp.isfinite(r_t) and float(r_t) > 0.0


def test_from_W0_rc_jit_matches_eager():
    eager = float(KingProfile.from_W0_rc(7.0, 1.0).r_t)
    jitted = float(jax.jit(lambda w: KingProfile.from_W0_rc(w, 1.0).r_t)(7.0))
    assert abs(eager - jitted) < 1e-9, f"jit {jitted} != eager {eager}"


def test_grad_of_psi_through_solve_king_profile_is_finite():
    """Audit C2 (second symptom): grad(sum psi) wrt W0 through the King ODE was nan,
    because the ODE RHS evaluates king_K_function(W0 - psi), which reaches W=0 at the
    cluster center (psi(0)=W0). The K-function fix makes the whole solve grad finite."""
    from progenax.profiles.king import solve_king_profile

    def loss(w0):
        _, psi = solve_king_profile(w0, n_points=200)
        return jnp.sum(psi)

    g = jax.grad(loss)(7.0)
    assert jnp.isfinite(g), f"grad(sum psi) wrt W0 is non-finite: {g} (C2)"
