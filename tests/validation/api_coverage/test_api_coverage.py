"""API-coverage ratchet (Phase 2 / Task 2.1).

The manifest must partition EVERY ``progenax.__all__`` symbol into exactly one of
SYMBOL_TESTS / EXEMPT / UNTESTED, with no overlaps and no stale entries. A new public
symbol with no categorization -> RED; a categorized symbol removed from ``__all__`` -> RED.
The UNTESTED holes are tracked (xfail) until Task 2.3 fills them; the line-coverage floor
SKIPs until the committed full-suite coverage.json lands (Task 2.2).
"""
import json
from pathlib import Path

import progenax
import pytest

from tests.validation.api_coverage.manifest import (
    EXEMPT,
    FULL_SELECTOR,
    LINE_COV_FLOOR,
    SYMBOL_TESTS,
    UNTESTED,
)

# Repo-root-relative path to the committed full-suite coverage artifact (Task 2.2).
_COVERAGE_JSON = Path(__file__).resolve().parents[3] / "validation" / "data" / "coverage.json"


def test_every_public_symbol_is_categorized():
    """Every __all__ symbol lands in EXACTLY ONE of the three dicts (no gaps, no overlaps)."""
    public = set(progenax.__all__)
    st, ex, un = set(SYMBOL_TESTS), set(EXEMPT), set(UNTESTED)
    union = st | ex | un

    missing = sorted(public - union)
    assert not missing, (
        f"public symbols not categorized in any of SYMBOL_TESTS/EXEMPT/UNTESTED: {missing}")

    # Disjointness: a symbol must not appear in two dicts (else its status is ambiguous).
    assert not (st & ex), f"symbols in BOTH SYMBOL_TESTS and EXEMPT: {sorted(st & ex)}"
    assert not (st & un), f"symbols in BOTH SYMBOL_TESTS and UNTESTED: {sorted(st & un)}"
    assert not (ex & un), f"symbols in BOTH EXEMPT and UNTESTED: {sorted(ex & un)}"


def test_no_stale_mappings():
    """No dict references a symbol that is no longer in __all__ (catches deletions/renames)."""
    public = set(progenax.__all__)
    union = set(SYMBOL_TESTS) | set(EXEMPT) | set(UNTESTED)
    stale = sorted(union - public)
    assert not stale, (
        f"manifest entries for symbols no longer in progenax.__all__ (remove them): {stale}")


@pytest.mark.xfail(
    reason="holes pending Task 2.3 + Anna per-item approval",
    strict=False,
)
def test_no_untested_holes():
    """Task 2.3 fills UNTESTED, then this flips to a hard, always-passing assert (strict)."""
    assert not UNTESTED, (
        f"public symbols with NO asserting test (real holes for Task 2.3): {sorted(UNTESTED)}")


def test_line_coverage_above_floor():
    """Enforce the ratchet-up line-coverage floor against the COMMITTED full-suite artifact.

    SKIPs (green) until the full-suite ``--cov`` run commits ``validation/data/coverage.json``
    (Task 2.2). Once committed, this becomes a hard floor: the artifact must come from the FULL
    suite (a ``-m "not slow"`` pass understates coverage), and total >= LINE_COV_FLOOR.
    """
    if not _COVERAGE_JSON.exists():
        pytest.skip("coverage.json not committed yet — Task 2.2")
    cov = json.loads(_COVERAGE_JSON.read_text())
    selector = cov["coverage_provenance"]["selector"]
    assert selector == FULL_SELECTOR, (
        f"coverage.json selector={selector!r} (not {FULL_SELECTOR!r}) — regenerate with the "
        f"FULL suite; a partial selector understates coverage and would game the floor.")
    total = float(cov["total_percent"])
    assert total >= LINE_COV_FLOOR, (
        f"line coverage {total:.1f}% < floor {LINE_COV_FLOOR:.1f}% — raise coverage, do not "
        f"lower the floor (ratchet-up-only).")


def test_exempt_crosscheck_grad_audit():
    """C4 anti-drift: the api-coverage EXEMPT set vs the grad-audit SYMBOL_CATEGORY partition.

    The two registries answer DIFFERENT questions, so divergence is EXPECTED and not, by
    itself, an error:
      - api-coverage asks "is the symbol exercised by an ASSERTING test?" A typing Protocol
        asserted via a conformance test (``assert isinstance(x, Proto)``) IS exercised here,
        so it lives in SYMBOL_TESTS — while grad-audit (which asks "is it a Fisher/gradient
        entry point?") legitimately marks the same Protocol EXEMPT_PROTOCOL.
    The ONE combination that is a likely mistake (and the only hard-fail here): a symbol we
    mark EXEMPT while grad-audit AUDITED it with a real registry case. If grad-audit found a
    differentiable, FD-consistent entry point worth a registry case, an asserting test almost
    certainly exists, so EXEMPT here is suspect. Everything else is reported, not failed.
    """
    from tests.validation.grad_audit.manifest import AUDITED, SYMBOL_CATEGORY

    hard_fail = sorted(
        s for s in EXEMPT
        if SYMBOL_CATEGORY.get(s) == AUDITED
    )
    assert not hard_fail, (
        f"symbols EXEMPT in api-coverage but AUDITED (with a registry case) in grad-audit — "
        f"a grad-audited entry point should have an asserting test, so EXEMPT is suspect: "
        f"{hard_fail}. Map them in SYMBOL_TESTS or justify with Anna.")

    # Informational: where the two partitions disagree (documented, non-fatal). Surfaces drift
    # for human review without making the suite brittle to each registry's distinct lens.
    divergences = []
    for s in progenax.__all__:
        ga = SYMBOL_CATEGORY.get(s, "<absent>")
        if s in SYMBOL_TESTS:
            api = "SYMBOL_TESTS"
        elif s in EXEMPT:
            api = "EXEMPT"
        else:
            api = "UNTESTED"
        ga_is_exempt = ga.startswith("EXEMPT")
        # Disagreement of interest: grad-audit EXEMPT but api-coverage maps it (or vice-versa).
        if (ga_is_exempt and api == "SYMBOL_TESTS") or (ga == AUDITED and api != "SYMBOL_TESTS"):
            divergences.append(f"{s}: grad_audit={ga} api_coverage={api}")
    # Not asserted — printed for the record (visible with -s / on failure of a sibling test).
    if divergences:
        print(
            "\n[api-coverage vs grad-audit divergences — EXPECTED (different lenses), "
            f"documented for C4 anti-drift review]:\n  " + "\n  ".join(divergences)
        )
