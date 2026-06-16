"""Stage-1 OED demo core: additive Fisher over (radius x channel), c/D/A criteria,
optax optimizer, sky projection + calibration. Consumer of progenax.project_dispersion.
See docs/plans/2026-06-16-oed-demo-stage1-design.md.

Task 1 (this commit): the predicted observable g(theta) and the design-INDEPENDENT
per-star Fisher blocks M_{c,b} = 2 J J^T / (sigma^2 + eps_c^2), computed via ONE
reverse-mode jacrev through project_dispersion (the only place the forward model is
differentiated). We use jacrev (reverse-mode) by policy: it is the supported/tested
gradient path for project_dispersion across ALL profiles, and it keeps the demo robust
to a future King/Michie swap, where the equilibrium-solver profiles hit custom_vjp ODE
solvers with no jvp rule so forward-mode (jacfwd/hessian) would genuinely crash. For the
Plummer profile used here the quadratures are plain jnp (no ODE / custom_vjp), so
forward-mode also happens to work -- but reverse-mode is the canonical choice. See the
src/progenax/kinematics/dispersion.py module docstring for the per-profile forward-mode
support matrix.
"""
import jax
import jax.numpy as jnp

from progenax import PlummerProfile, project_dispersion
from jaxstro.units import STELLAR  # noqa: F401  -- re-exported for the demo's callers

# --- Unit conversions (STELLAR: M_sun, pc, Myr; project_dispersion returns pc/Myr) ---
# 1 km/s = 1 / 0.977792 pc/Myr (1 pc/Myr = 0.977792 km/s).
KMS_PER_PC_PER_MYR = 0.977792


def kms_to_pcMyr(v_kms):
    """km/s -> pc/Myr (the native velocity unit of project_dispersion under STELLAR)."""
    return v_kms / KMS_PER_PC_PER_MYR


def pm_masyr_to_kms(mu, d_kpc):
    """Proper motion [mas/yr] at distance d [kpc] -> transverse velocity [km/s]."""
    return 4.74047 * mu * d_kpc


# --- Mock cluster (generic GC-scale, unnamed -- no overclaim). theta = (r_a, M, r_h). ---
MOCK = dict(M=1e5, r_h=3.0, r_a=6.0, d_kpc=4.0, eps_RV_kms=1.0, eps_PM_masyr=0.05)

# K=12 log-spaced on-sky bin-centre radii out to ~3 r_h.
R_BINS = jnp.logspace(jnp.log10(0.3 * MOCK["r_h"]), jnp.log10(3.0 * MOCK["r_h"]), 12)

# Per-channel per-star measurement error eps_c = (eps_RV, eps_PM, eps_PM) [pc/Myr].
# Both PM axes (pm_r, pm_t) share the single astrometric error.
_eps_RV = kms_to_pcMyr(MOCK["eps_RV_kms"])
_eps_PM = kms_to_pcMyr(pm_masyr_to_kms(MOCK["eps_PM_masyr"], MOCK["d_kpc"]))
EPS = jnp.array([_eps_RV, _eps_PM, _eps_PM])      # (3,) [pc/Myr]


def theta_truth():
    """Truth parameter vector theta = (r_a, M, r_h) -- index 0 = r_a (TARGET)."""
    return jnp.array([MOCK["r_a"], MOCK["M"], MOCK["r_h"]])


def predict_sigma(theta, R_bins, G):
    """Predicted observable g(theta): (3, K) dispersions, rows = (los, pm_r, pm_t).

    Channels in pc/Myr at the K on-sky bin-centre radii R_bins, via the
    Binney & Mamon (1982) projection of the OM-Plummer Jeans model.
    """
    r_a, M, r_h = theta[0], theta[1], theta[2]
    prof = PlummerProfile(r_h=r_h)
    pd = project_dispersion(prof, r_a, R_bins, M, G)
    return jnp.stack([pd.sigma_los, pd.sigma_pm_r, pd.sigma_pm_t])   # (3, K)


def per_star_blocks(theta, R_bins, eps, G):
    """Design-INDEPENDENT per-star Fisher blocks M_{c,b} = 2 J J^T / (sigma^2 + eps_c^2).

    A dispersion measured from n stars (per-star error eps, predicted dispersion
    sigma) has Gaussian error delta_sigma^2 = (sigma^2 + eps^2) / (2 n), so the
    per-star Fisher contribution of channel c, bin b is the rank-1 3x3 block
    M_{c,b} = 2 J_{c,b} J_{c,b}^T / (sigma_{c,b}^2 + eps_c^2), with
    J_{c,b} = d sigma_pred,{c,b} / d theta. The full design Fisher is then the linear
    sum F = sum_{c,b} n_eff,{c,b} M_{c,b} (Task 2), so this jacrev is computed ONCE
    and the optimization is pure 3x3 linear algebra.

    Reverse-mode jacrev by policy (the supported/tested grad path for all profiles;
    forward-mode also works for the analytic-density Plummer path but would crash through
    the King/Michie equilibrium-solver custom_vjp ODEs -- see the module docstring).

    Returns (Mb (3, K, 3, 3): channel x bin x P x P, sigma (3, K)).
    """
    sig = predict_sigma(theta, R_bins, G)                       # (3, K)
    J = jax.jacrev(predict_sigma, argnums=0)(theta, R_bins, G)  # (3, K, 3) -- wrt theta
    denom = sig**2 + (eps[:, None])**2                          # (3, K)
    Mb = 2.0 * jnp.einsum("ckp,ckq->ckpq", J, J) / denom[..., None, None]
    return Mb, sig
