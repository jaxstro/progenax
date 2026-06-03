"""KingProfile JIT-safety & ODE gradient flow (audit finding C2, residual half).

C2 had two symptoms. The first — a where-NaN trap in the King K-function used as
the density source — is moot now that the density-potential relation was corrected
to the lowered-Maxwellian form (Batch 2) and the unused K-function was removed
(Batch 4). The gradient-safety of the W=0 clamp now lives on the density itself
and is checked in ``test_king_density.py``. The two *residual* C2 symptoms this
file still guards:

1. ``_find_tidal_radius`` returned ``float(xi_t)``, concretizing the value and
   making ``KingProfile.from_W0_rc`` non-JIT-able.
2. ``grad(sum psi)`` through the King ODE was ``nan`` because the Poisson RHS
   evaluates the density at W=0 (the cluster center, psi(0)=W0); the density's
   double-``where`` clamp keeps that gradient finite.
"""

import jax
import jax.numpy as jnp

import progenax  # noqa: F401  (enables float64 at import)
from progenax.profiles.king import KingProfile


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
    because the Poisson RHS evaluates the lowered-Maxwellian density at W=0 (the
    cluster center, psi(0)=W0). The density's double-where clamp keeps the whole
    solve's gradient finite."""
    from progenax.profiles.king import solve_king_profile

    def loss(w0):
        _, psi = solve_king_profile(w0, n_points=200)
        return jnp.sum(psi)

    g = jax.grad(loss)(7.0)
    assert jnp.isfinite(g), f"grad(sum psi) wrt W0 is non-finite: {g} (C2)"
