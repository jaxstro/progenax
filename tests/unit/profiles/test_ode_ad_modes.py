"""Forward/reverse AD agreement through the lowered-profile ODE owners."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.profiles.king import solve_king_profile
from progenax.profiles.limepy import solve_limepy_profile
from progenax.profiles.michie import solve_michie_profile


@pytest.mark.parametrize("family", ["king", "michie", "limepy"])
def test_profile_ode_forward_reverse_duality(family):
    def values(theta, adjoint=None):
        if family == "king":
            result = solve_king_profile(
                theta[0], xi_max=300.0, n_points=300, adjoint=adjoint
            )
        elif family == "michie":
            result = solve_michie_profile(
                theta[0], theta[1], xi_max=300.0, n_points=300, adjoint=adjoint
            )
        else:
            result = solve_limepy_profile(
                theta[0], theta[1], xi_max=300.0, n_points=300, adjoint=adjoint
            )
        return result[2][jnp.array([1, 3, 8])]

    theta = jnp.array([5.0, 10.0 if family == "michie" else 1.0])
    forward = jax.jit(jax.jacfwd(lambda t: values(t, diffrax.ForwardMode())))(theta)
    reverse = jax.jit(jax.jacrev(values))(theta)
    assert bool(jnp.all(jnp.isfinite(forward)))
    # Shared PhysicsTolerances.EXACT from conftest: arithmetic-equivalence check.
    np.testing.assert_allclose(forward, reverse, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("family", ["king", "michie", "limepy"])
@pytest.mark.parametrize("kind", ["profile", "df"])
def test_public_constructor_threads_forward_adjoint(family, kind):
    from progenax.kinematics.king_df import KingVelocityDF
    from progenax.kinematics.limepy_df import LIMEPYVelocityDF
    from progenax.kinematics.michie_df import MichieVelocityDF
    from progenax.profiles.king import KingProfile
    from progenax.profiles.limepy import LIMEPYProfile
    from progenax.profiles.michie import MichieProfile

    profiles = dict(king=KingProfile, michie=MichieProfile, limepy=LIMEPYProfile)
    dfs = dict(king=KingVelocityDF, michie=MichieVelocityDF, limepy=LIMEPYVelocityDF)

    def value(W0, adjoint):
        kwargs = dict(W0=W0, r_c=1.0, xi_max=300.0, n_ode_points=300, adjoint=adjoint)
        if family == "michie":
            kwargs["r_a"] = 10.0
        elif family == "limepy":
            kwargs["g"] = 1.0
        if kind == "profile":
            model = profiles[family].from_W0_rc(**kwargs, n_grid=300)
        else:
            model = dfs[family](**kwargs, speed_method="quadrature")
        return jnp.sum(model.psi_grid)

    forward = jax.jit(jax.jacfwd(lambda w: value(w, diffrax.ForwardMode())))(5.0)
    reverse = jax.jit(jax.grad(lambda w: value(w, None)))(5.0)
    assert bool(jnp.isfinite(forward))
    np.testing.assert_allclose(forward, reverse, rtol=0.003, atol=0.0)
