"""Layer 3 coverage ratchet (design D2/D3): the registry must cover every MUST_AUDIT entry,
and the manifest must categorize every public symbol. A deleted case or a new ungated public
entry point -> RED."""
import progenax  # noqa: F401
import pytest
from tests.validation.grad_audit.manifest import (
    AUDITED, MUST_AUDIT, PARAM_ALLOWLIST, SYMBOL_CATEGORY,
)
from tests.validation.grad_audit.registry import REGISTRY

_REGISTRY_KEYS = {(c.id, c.param) for c in REGISTRY}


def test_every_must_audit_entry_is_covered():
    missing = sorted(k for k in MUST_AUDIT if k not in _REGISTRY_KEYS)
    assert not missing, (
        f"MUST_AUDIT entries with NO registry case (coverage ratchet RED): {missing}. "
        f"Add the Case to registry.py (or, if intentionally removing coverage, remove the "
        f"manifest entry WITH Anna's sign-off)."
    )
