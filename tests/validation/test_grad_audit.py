"""The release gradient-gate: every registered entry point is finite + FD-consistent
(or a documented, pinned limitation). Numbers are emitted to JSON by scripts/audit_gradients.py;
this gate asserts on the same engine. See docs/plans/2026-06-13-...-design.md.

Confirmed-but-unfixed hazards are pinned with strict-xfail MARKERS (not imperative
pytest.xfail) so the assertion still runs: a hazard fails -> the marker converts it to XFAIL,
but once the hazard is FIXED the assertion passes -> XPASS -> strict-FAIL, forcing the
marker (and hazard_id) to be removed (design D6 self-cleaning ratchet)."""
import jax
import jax.numpy as jnp
import pytest
import progenax  # noqa: F401  (float64)

from tests.validation.grad_audit.core import audit_entry_point
from tests.validation.grad_audit.registry import REGISTRY


def _case_params():
    params = []
    for c in REGISTRY:
        marks = ()
        if c.hazard_id:
            marks = (pytest.mark.xfail(strict=True,
                     reason=f"HAZARD {c.hazard_id}: confirmed, pending triage"),)
        params.append(pytest.param(c, marks=marks, id=c.id))
    return params


@pytest.mark.parametrize("case", _case_params())
def test_gradient_audit(case):
    """Baseline (generic) params: assert the computed status is acceptable."""
    r = audit_entry_point(case)
    assert r.status in ("clean", "known-limitation"), (
        f"{case.id} [{case.param}] -> {r.status}: AD={r.ad:.3e} FD={r.fd:.3e} "
        f"ratio={r.ratio:.6f} finite={r.finite}"
    )


def _edge_params():
    params = []
    for c in REGISTRY:
        for e in c.edges:
            marks = ()
            if e.hazard_id:
                marks = (pytest.mark.xfail(strict=True,
                         reason=f"HAZARD {e.hazard_id}: confirmed at {e.label}, pending triage"),)
            params.append(pytest.param(c, e, marks=marks, id=f"{c.id}::{e.label}"))
    return params


_EDGE_PARAMS = _edge_params()


@pytest.mark.skipif(not _EDGE_PARAMS, reason="no edges registered yet")
@pytest.mark.parametrize("case,edge", _EDGE_PARAMS)
def test_gradient_audit_edges(case, edge):
    """Edge/boundary params: the hazard probes."""
    r = audit_entry_point(case, theta=edge.theta0,
                          tol=edge.tol or case.tol, expect=edge.expect or case.expect)
    assert r.status in ("clean", "known-limitation"), (
        f"{case.id}::{edge.label} -> {r.status}: AD={r.ad:.3e} FD={r.fd:.3e} ratio={r.ratio:.6f}"
    )


def test_binned_sigma_mutation_has_teeth():
    r"""The binned_sigma1d Fisher case genuinely tests the params->summary gradient.

    Proof-of-teeth (Task 3.1, Part B2): the LIVE case differentiates the binned
    velocity-dispersion summary through the sampled (pos, vel). Wrapping (pos, vel)
    in ``jax.lax.stop_gradient`` before the binner must collapse the autodiff
    gradient to ~0 (the summary becomes a constant in r_h) while a finite-difference
    probe stays live and non-zero. Were the live case secretly differentiating a
    constant, the mutation would change nothing — so the stopped-AD -> 0 with a
    live FD is the discriminating evidence that d(summary)/d(r_h) really flows.

    MEASURED (h_rel=1e-5, the case's FD step, N=2000, K=7 frozen edges):
        live AD     = -3.345969e+00   (non-zero, the real gradient)
        live FD     = -3.345969e+00   (matches AD to machine precision)
        stopped AD  =  0.000000e+00   (exact collapse — the mutation kills the channel)
    """
    from jaxstro.units import STELLAR
    from progenax import PlummerProfile, PlummerVelocityDF, build_spatial_ic
    from tests.validation.grad_audit.binners import binned_sigma1d
    from tests.validation.grad_audit.registry import (
        _BK_GROUP, _BK_MASSES, _BK_N_MIN, _BK_R_EDGES, _KEY, _binned_sigma1d_rh,
    )
    from tests.validation.grad_audit.reductions import identity_sum

    theta0 = jnp.asarray(1.0)
    h = 1e-5  # the case's FD step (off the r=8 edge-crossing)

    # LIVE: the real params->summary gradient (the registered case fn).
    live_loss = lambda r_h: identity_sum(_binned_sigma1d_rh(r_h))
    live_ad = float(jax.grad(live_loss)(theta0))
    g = lambda x: float(live_loss(jnp.asarray(x)))
    live_fd = (g(1.0 + h) - g(1.0 - h)) / (2.0 * h)

    # MUTATED: stop_gradient on (pos, vel) before the binner -> AD must die.
    def stopped_loss(r_h):
        profile = PlummerProfile(r_h=r_h)
        df = PlummerVelocityDF(r_h=r_h)
        ic = build_spatial_ic(profile, _BK_MASSES, df, _KEY, G=STELLAR.G)
        pos = jax.lax.stop_gradient(ic.positions)
        vel = jax.lax.stop_gradient(ic.velocities)
        sig_hat, _se, _w, _n = binned_sigma1d(
            pos, vel, _BK_GROUP, 1, _BK_R_EDGES, n_min=_BK_N_MIN)
        return identity_sum(sig_hat)

    stopped_ad = float(jax.grad(stopped_loss)(theta0))

    # The live channel is non-trivially live (the case is not a constant)...
    assert abs(live_ad) > 1e-3, f"live AD unexpectedly ~0: {live_ad:.3e}"
    assert abs(live_fd) > 1e-3, f"live FD unexpectedly ~0: {live_fd:.3e}"
    # ...and the mutation collapses the gradient to ~0 (the teeth).
    assert abs(stopped_ad) < 1e-12, (
        f"stop_gradient did NOT kill the gradient (stopped AD={stopped_ad:.3e}); "
        f"the case may be differentiating a constant, not the params->summary channel"
    )
