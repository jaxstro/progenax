"""The grad-audit case registry: every public entry point x direction x param.
Tiers are added incrementally (see the implementation plan)."""
import jax, jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import (  # noqa: F401-adjacent — carries float64 on import
    KingProfile,
    KingVelocityDF,
    PlummerProfile,
    PlummerVelocityDF,
)
from tests.validation.grad_audit.core import Case, EdgeConfig
from tests.validation.grad_audit.reductions import mean_radius, mean_speed

_KEY = jax.random.PRNGKey(0)
_MASSES = jnp.ones(400)
# Split the frozen key so positions and velocities never reuse the same PRNG draw
# (repo anti-pattern: one key for pos AND vel correlates the two samplings).
_KEY_POS, _KEY_VEL = jax.random.split(_KEY)

# Plummer scale radius from r_h (a = r_h * sqrt(2^(2/3)-1)); the Merritt (1985, Eq. 46)
# Osipkov-Merritt bound is r_a >= 0.75 a, below which the OM phase-space DF goes negative.
_PLUMMER_A_FACTOR = float(jnp.sqrt(2 ** (2 / 3) - 1))


def _plummer_positions(r_h):
    return PlummerProfile(r_h=r_h).sample_positions(_MASSES, _KEY_POS)


def _plummer_velocities(r_h):
    positions = PlummerProfile(r_h=r_h).sample_positions(_MASSES, _KEY_POS)
    df = PlummerVelocityDF(r_h=r_h)
    return df.sample_velocities(positions, _MASSES, _KEY_VEL, G=STELLAR.G)


def _plummer_velocities_om(r_a):
    # Isotropic positions (fixed r_h=1) with an Osipkov-Merritt anisotropic velocity DF;
    # the audited parameter is the anisotropy radius r_a (constructor kwarg anisotropy_radius).
    positions = PlummerProfile(r_h=1.0).sample_positions(_MASSES, _KEY_POS)
    df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=r_a)
    return df.sample_velocities(positions, _MASSES, _KEY_VEL, G=STELLAR.G)


# Edge value just above the Merritt bound 0.75 a (for r_h=1): a small margin keeps the
# symmetric FD lower probe (r_a - h) on the valid side of the bound while still probing it.
_OM_BOUND = 0.75 * _PLUMMER_A_FACTOR  # ~= 0.57482 for r_h=1
_OM_EDGE = _OM_BOUND + 2e-3           # ~= 0.57682, lower FD probe (h=1e-4) stays >= bound


# ---------------------------------------------------------------------------
# King (1966) profile + DF (Task 1.2)
# ---------------------------------------------------------------------------
# The King ODE auto-sizes its integration domain from a CONCRETE W0
# (_auto_ode_domain), but under jax.grad W0 is a tracer and the domain falls
# back to the historical default (xi_max=300, n=2000) -- which is too small for
# W0 >= ~10 and trips the eager pinned-r_t ValueError. So the W0-parameterised
# closure passes an EXPLICIT domain sized for the largest probed W0 (the W0=12
# edge needs xi_t ~ 548), keeping both the W0=7 baseline and the W0=12 edge valid
# under tracing. The r_c closure fixes W0=7.0 (concrete) so it can auto-size.
_KING_XI_MAX = 700.0       # > xi_t(W0=12) ~ 548, with margin
_KING_N_ODE = 5000


def _king_positions_W0(W0):
    profile = KingProfile.from_W0_rc(
        W0=W0, r_c=1.0, xi_max=_KING_XI_MAX, n_ode_points=_KING_N_ODE
    )
    return profile.sample_positions(_MASSES, _KEY_POS)


def _king_positions_rc(r_c):
    # W0 fixed (concrete) -> auto-sized ODE domain is valid; r_c is the audited param.
    profile = KingProfile.from_W0_rc(W0=7.0, r_c=r_c)
    return profile.sample_positions(_MASSES, _KEY_POS)


def _king_velocities_W0(W0):
    profile = KingProfile.from_W0_rc(
        W0=W0, r_c=1.0, xi_max=_KING_XI_MAX, n_ode_points=_KING_N_ODE
    )
    positions = profile.sample_positions(_MASSES, _KEY_POS)
    df = KingVelocityDF(W0=W0, r_c=1.0, xi_max=_KING_XI_MAX, n_ode_points=_KING_N_ODE)
    return df.sample_velocities(positions, _MASSES, _KEY_VEL, G=STELLAR.G)


def _king_r_t(W0):
    # The King tidal radius r_t = r_c * xi_t, with xi_t from _find_tidal_radius's
    # linear-interpolation crossing. Task 1.2b feeds UNCLAMPED psi to the interp,
    # so d r_t/dW0 now flows through the diffrax solve (the implicit-function-
    # theorem result to grid accuracy). A FINE explicit grid (xi_max=400,
    # n=8000) makes the linear-interp FD estimate grid-converged so AD~FD.
    return jnp.atleast_1d(
        KingProfile.from_W0_rc(
            W0=W0, r_c=1.0, xi_max=400.0, n_ode_points=8000
        ).r_t
    )

REGISTRY: list[Case] = [
    Case(id="PlummerProfile.sample_positions", direction="params->IC",
         fn=_plummer_positions, param="r_h", theta0=1.0, reduce=mean_radius,
         expect="consistent", tol=1e-5),
    Case(id="PlummerVelocityDF.sample_velocities", direction="params->IC",
         fn=_plummer_velocities, param="r_h", theta0=1.0, reduce=mean_speed,
         expect="consistent", tol=1e-3),
    Case(id="PlummerVelocityDF+OM.sample_velocities", direction="params->IC",
         fn=_plummer_velocities_om, param="r_a", theta0=2.0, reduce=mean_speed,
         expect="consistent", tol=1e-3,
         edges=(EdgeConfig("r_a=0.75a", _OM_EDGE),)),
    # King profile positions: differentiable in W0 (profile SHAPE) and r_c.
    Case(id="KingProfile.sample_positions", direction="params->IC",
         fn=_king_positions_W0, param="W0", theta0=7.0, reduce=mean_radius,
         expect="consistent", tol=1e-3,
         edges=(EdgeConfig("W0=12", 12.0),)),
    Case(id="KingProfile.sample_positions", direction="params->IC",
         fn=_king_positions_rc, param="r_c", theta0=1.0, reduce=mean_radius,
         expect="consistent", tol=1e-3),
    # King lowered-Maxwellian velocity DF: differentiable in W0.
    Case(id="KingVelocityDF.sample_velocities", direction="params->IC",
         fn=_king_velocities_W0, param="W0", theta0=7.0, reduce=mean_speed,
         expect="consistent", tol=1e-3),
    # The King tidal radius r_t: now differentiable in W0 (Task 1.2b) via the
    # unclamped-psi linear-interp crossing in _find_tidal_radius. d r_t/dW0 ~ 48
    # at W0=8 (smooth, large; measured AD 47.999 vs FD 48.024); AD~FD to grid
    # accuracy. tol=1e-2 is the honest FD
    # band for a linear-interp estimate of a smooth derivative (measured ratio
    # within ~1e-3; margin to 1e-2).
    Case(id="KingProfile.r_t", direction="params->IC",
         fn=_king_r_t, param="W0", theta0=8.0, reduce=jnp.sum,
         expect="consistent", tol=1e-2),
]
