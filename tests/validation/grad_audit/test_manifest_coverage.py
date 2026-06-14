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


def test_symbol_category_covers_all_public_symbols_exactly():
    public = set(progenax.__all__)
    mapped = set(SYMBOL_CATEGORY)
    unmapped = sorted(public - mapped)   # NEW public symbol not categorized -> RED
    stale = sorted(mapped - public)      # category for a removed/renamed symbol -> RED
    assert not unmapped, (
        f"public symbols missing from SYMBOL_CATEGORY (categorize each as AUDITED or EXEMPT_*): "
        f"{unmapped}")
    assert not stale, f"SYMBOL_CATEGORY entries no longer in __all__: {stale}"


def test_every_audited_symbol_has_a_registry_case():
    audited = {s for s, cat in SYMBOL_CATEGORY.items() if cat == AUDITED}
    covered_ids = {cid for (cid, _p) in _REGISTRY_KEYS}
    # An AUDITED symbol must own at least one registry id (id may be "Class.method[...]").
    uncovered = sorted(s for s in audited
                       if not any(cid == s or cid.startswith(s + ".") or cid.startswith(s + "[")
                                  for cid in covered_ids))
    assert not uncovered, (
        f"AUDITED symbols with no registry case: {uncovered}. Either add a case or "
        f"re-categorize as EXEMPT_COVERED_ELSEWHERE with Anna's sign-off.")


def test_param_allowlist_entries_are_real_registry_cases():
    # Allowlist must reference cases that actually exist (no stale pins).
    stale = sorted(k for k in PARAM_ALLOWLIST if k not in _REGISTRY_KEYS)
    assert not stale, f"PARAM_ALLOWLIST entries not in REGISTRY: {stale}"
