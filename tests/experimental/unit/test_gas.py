"""Stars+gas partition physics core (realization/gas.py; Phase 4a, Aim 2 handoff).

Physical contracts (design addendum 2026-07-16, ratified):
- parent-cloud normalization ρ_cl = M_cl·ρ̃/∫ρ̃dV (envelope-normalization invariant,
  exact discrete mass closure);
- local free-fall time t_ff = √(3π/32Gρ_cl);
- positivity-preserving local SFE ε⋆ = 1 − exp(−τ⋆ w/t_ff), partition
  ρ⋆,0 = ε⋆ρ_cl, ρ_g,0 = (1−ε⋆)ρ_cl (pointwise conserving);
- τ⋆ from the monotone global-SFE constraint by fixed-iteration bracketed bisection
  (lax.scan, NO while_loop) with the implicit-function-theorem derivative;
- low efficiency ⇒ ρ⋆,0 ∝ w·ρ_cl^{3/2} (the AC-IC7-gated multi-freefall law);
- loud refusals: impossible SFE, empty collapse support.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gravoturb.realization.gas import (
    local_freefall_time,
    local_stellar_fraction,
    normalized_cloud_density,
    partition_star_gas,
    solve_tau_star,
)
from jaxstro.units import STELLAR

pytestmark = [pytest.mark.experimental, pytest.mark.unit]

G = STELLAR.G
BOX = 4.0  # pc


def _fields(shape=(16, 16, 16), seed=0):
    """A small realistic (s_total, w) pair from the existing verified pieces."""
    from gravoturb.realization.envelope import apply_spherical_envelope
    from gravoturb.realization.pipeline import build_turbulent_field
    from gravoturb.realization.placement import collapse_weights

    from progenax import PlummerProfile

    fld = build_turbulent_field(8.0, 0.5, 1.8, 3.0, shape, jax.random.PRNGKey(seed))
    s_total = apply_spherical_envelope(fld.s, PlummerProfile(r_h=0.5), BOX)
    w = collapse_weights(fld.s, fld.s_t, 8.0)
    return s_total, w


def test_cloud_normalization_exact_and_envelope_invariant():
    """∫ρ_cl dV = M_cl exactly; adding a constant to s_total (an arbitrary envelope
    normalization) changes NOTHING."""
    s_total, _ = _fields()
    M_cl = 5000.0
    rho, dv = normalized_cloud_density(s_total, BOX, M_cl)
    assert float(jnp.sum(rho) * dv) == pytest.approx(M_cl, rel=1e-12)
    assert float(dv) == pytest.approx((BOX / 16) ** 3, rel=1e-12)
    rho2, _ = normalized_cloud_density(s_total + 3.7, BOX, M_cl)
    np.testing.assert_allclose(np.asarray(rho2), np.asarray(rho), rtol=1e-12)
    assert bool(jnp.all(rho > 0))


def test_local_freefall_time_formula():
    """t_ff = √(3π/32Gρ) elementwise (STELLAR units: ρ in M⊙/pc³ → t_ff in Myr)."""
    rho = jnp.array([1.0, 100.0, 1e4])
    t = local_freefall_time(rho, G=G)
    expected = np.sqrt(3.0 * np.pi / (32.0 * G * np.asarray(rho)))
    np.testing.assert_allclose(np.asarray(t), expected, rtol=1e-12)
    assert float(t[1]) == pytest.approx(0.81, rel=0.02)  # 100 M⊙/pc³ → ~0.8 Myr


def test_partition_conserves_pointwise_and_is_positive():
    s_total, w = _fields()
    rho, dv = normalized_cloud_density(s_total, BOX, 5000.0)
    t_ff = local_freefall_time(rho, G=G)
    rho_star, rho_gas = partition_star_gas(rho, w, t_ff, tau_star=0.3)
    np.testing.assert_allclose(np.asarray(rho_star + rho_gas), np.asarray(rho),
                               rtol=1e-12)
    assert bool(jnp.all(rho_star >= 0)) and bool(jnp.all(rho_gas >= 0))
    eps = local_stellar_fraction(w, t_ff, 0.3)
    assert bool(jnp.all(eps >= 0)) and bool(jnp.all(eps < 1.0))


@pytest.mark.parametrize("sfe", [0.05, 0.2, 0.5])
def test_solve_tau_star_reproduces_global_sfe(sfe):
    s_total, w = _fields()
    rho, dv = normalized_cloud_density(s_total, BOX, 5000.0)
    t_ff = local_freefall_time(rho, G=G)
    tau = solve_tau_star(w, t_ff, rho, dv, sfe_global=sfe)
    rho_star, _ = partition_star_gas(rho, w, t_ff, tau)
    achieved = float(jnp.sum(rho_star) * dv / (jnp.sum(rho) * dv))
    assert achieved == pytest.approx(sfe, abs=1e-8)
    assert float(tau) > 0.0


def test_low_efficiency_limit_recovers_multi_freefall_law():
    """As sfe→0, ρ⋆ ∝ w·ρ_cl/t_ff ∝ w·ρ_cl^{3/2}: the pointwise ratio to the
    AC-IC7-gated placement law becomes constant."""
    s_total, w = _fields()
    rho, dv = normalized_cloud_density(s_total, BOX, 5000.0)
    t_ff = local_freefall_time(rho, G=G)
    tau = solve_tau_star(w, t_ff, rho, dv, sfe_global=1e-4)
    rho_star, _ = partition_star_gas(rho, w, t_ff, tau)
    law = w * rho ** 1.5
    mask = np.asarray(law) > np.asarray(law).max() * 1e-6  # avoid 0/0 in dead cells
    ratio = np.asarray(rho_star)[mask] / np.asarray(law)[mask]
    assert np.std(ratio) / np.mean(ratio) < 1e-3  # constant ratio ⇒ same law


def test_tau_star_ift_derivative_matches_fd():
    """dτ⋆/d(sfe) via the implicit-function-theorem custom derivative vs central FD."""
    s_total, w = _fields()
    rho, dv = normalized_cloud_density(s_total, BOX, 5000.0)
    t_ff = local_freefall_time(rho, G=G)

    def tau_of_sfe(sfe):
        return solve_tau_star(w, t_ff, rho, dv, sfe_global=sfe)

    g = float(jax.grad(tau_of_sfe)(0.2))
    eps = 1e-6
    fd = (float(tau_of_sfe(0.2 + eps)) - float(tau_of_sfe(0.2 - eps))) / (2 * eps)
    assert g == pytest.approx(fd, rel=1e-5)
    assert g > 0.0  # more stars need more time-efficiency


def test_loud_refusals():
    s_total, w = _fields()
    rho, dv = normalized_cloud_density(s_total, BOX, 5000.0)
    t_ff = local_freefall_time(rho, G=G)
    with pytest.raises(ValueError, match="sfe"):
        solve_tau_star(w, t_ff, rho, dv, sfe_global=0.0)
    with pytest.raises(ValueError, match="sfe"):
        solve_tau_star(w, t_ff, rho, dv, sfe_global=1.0)
    # empty collapse support: w ≡ 0 can form no stars at ANY tau
    with pytest.raises(ValueError, match="support"):
        solve_tau_star(jnp.zeros_like(w), t_ff, rho, dv, sfe_global=0.2)
    # unreachable SFE: only a tiny w-support mass fraction, but sfe demands more
    w_tiny = jnp.where(jnp.arange(w.size).reshape(w.shape) == 0, 1.0, 0.0)
    with pytest.raises(ValueError, match="achievable"):
        solve_tau_star(w_tiny, t_ff, rho, dv, sfe_global=0.9)
