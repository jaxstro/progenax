"""The release gradient-gate: every registered entry point is finite + FD-consistent
(or a documented, pinned limitation). Numbers are emitted to JSON by scripts/audit_gradients.py;
this gate asserts on the same engine. See docs/plans/2026-06-13-...-design.md."""
import pytest
import progenax  # noqa: F401  (float64)

from tests.validation.grad_audit.core import audit_entry_point
from tests.validation.grad_audit.registry import REGISTRY

_IDS = [c.id for c in REGISTRY]


@pytest.mark.parametrize("case", REGISTRY, ids=_IDS)
def test_gradient_audit(case):
    """Baseline (generic) params: assert the computed status is acceptable."""
    r = audit_entry_point(case)
    if getattr(case, "hazard_id", None):
        pytest.xfail(f"HAZARD {case.hazard_id}: confirmed, pending triage")
    assert r.status in ("clean", "known-limitation"), (
        f"{case.id} [{case.param}] -> {r.status}: AD={r.ad:.3e} FD={r.fd:.3e} "
        f"ratio={r.ratio:.6f} finite={r.finite}"
    )


def _edge_cases():
    out = []
    for c in REGISTRY:
        for e in c.edges:
            out.append((c, e))
    return out


@pytest.mark.parametrize("case,edge", _edge_cases(),
                         ids=[f"{c.id}::{e.label}" for c, e in _edge_cases()])
def test_gradient_audit_edges(case, edge):
    """Edge/boundary params: the hazard probes."""
    if edge.hazard_id:
        pytest.xfail(f"HAZARD {edge.hazard_id}: confirmed at {edge.label}, pending triage")
    r = audit_entry_point(case, theta=edge.theta0,
                          tol=edge.tol or case.tol, expect=edge.expect or case.expect)
    assert r.status in ("clean", "known-limitation"), (
        f"{case.id}::{edge.label} -> {r.status}: AD={r.ad:.3e} FD={r.fd:.3e} ratio={r.ratio:.6f}"
    )
