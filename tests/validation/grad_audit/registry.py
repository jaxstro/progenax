"""The grad-audit case registry: every public entry point x direction x param.
Tiers are added incrementally (see the implementation plan)."""
import jax, jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import PlummerProfile, PlummerVelocityDF  # noqa: F401-adjacent — carries float64 on import
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
]
