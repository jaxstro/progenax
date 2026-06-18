"""progenax grad-audit engine — re-export of the shared ``jaxstro.testing`` engine.

The AD-vs-FD gradient-audit engine (``audit_entry_point``, ``Case``, ``AuditResult``,
``EdgeConfig``) was deduplicated into jaxstro (it was a byte-identical copy in progenax and
fluxax). It now lives in ``jaxstro.testing.grad_audit``; this module re-exports those names
so existing callsites
(``from tests.validation.grad_audit.core import audit_entry_point`` / ``Case, EdgeConfig``)
keep working unchanged. The progenax CASE REGISTRY stays local (``registry.py``); only the
engine is shared.

progenax differentiates N-body IC / summary entry points, so its registry uses the direction
labels ``"params->IC"`` / ``"params->summary"`` — the shared engine treats ``direction`` as a
free string, so no per-package label override is needed.
"""
from jaxstro.testing.grad_audit import (  # noqa: F401  (re-exported public surface)
    AuditResult,
    Case,
    Direction,
    EdgeConfig,
    Expect,
    audit_entry_point,
)

__all__ = [
    "audit_entry_point",
    "Case",
    "AuditResult",
    "EdgeConfig",
    "Direction",
    "Expect",
]
