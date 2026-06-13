"""The release gradient-gate: every registered entry point is finite + FD-consistent
(or a documented, pinned limitation). Numbers are emitted to JSON by scripts/audit_gradients.py;
this gate asserts on the same engine. See docs/plans/2026-06-13-...-design.md.

Confirmed-but-unfixed hazards are pinned with strict-xfail MARKERS (not imperative
pytest.xfail) so the assertion still runs: a hazard fails -> the marker converts it to XFAIL,
but once the hazard is FIXED the assertion passes -> XPASS -> strict-FAIL, forcing the
marker (and hazard_id) to be removed (design D6 self-cleaning ratchet)."""
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
