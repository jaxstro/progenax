"""API-coverage ratchet (Phase 2 / Task 2.1).

The manifest must partition EVERY ``progenax.__all__`` symbol into exactly one of
SYMBOL_TESTS / EXEMPT / UNTESTED, with no overlaps and no stale entries. A new public
symbol with no categorization -> RED; a categorized symbol removed from ``__all__`` -> RED.
As of Task 2.3 the UNTESTED holes are filled (six analytical-IC factories/helpers each got
an asserting physics test), so ``test_no_untested_holes`` is a HARD assertion (no longer
xfail) and ``UNTESTED`` is empty; the line-coverage floor is HARD against the committed
full-suite coverage.json (Task 2.2).
"""
import json
from pathlib import Path

import progenax

from jaxstro.testing.ratchet import assert_no_stale, assert_partition
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
    """Every __all__ symbol lands in EXACTLY ONE of the three dicts (no gaps, no overlaps).

    Delegated to the canonical ``jaxstro.testing.ratchet.assert_partition`` primitive
    (Task 2.2): coverage (no uncategorized symbol), disjointness (no symbol in two buckets),
    and no-stale (no bucket key absent from ``__all__``) in one self-tested mechanism.
    """
    assert_partition(
        set(progenax.__all__),
        SYMBOL_TESTS,
        EXEMPT,
        UNTESTED,
        label="api_coverage.partition",
    )


def test_no_stale_mappings():
    """No dict references a symbol that is no longer in __all__ (catches deletions/renames).

    Delegated to ``assert_no_stale`` over the merged manifest mapping; redundant with the
    no-stale leg of ``assert_partition`` above, kept as a distinct, narrowly-scoped failure
    signal for a deleted/renamed symbol still referenced by the manifest.
    """
    merged = {**SYMBOL_TESTS, **EXEMPT, **UNTESTED}
    assert_no_stale(merged, set(progenax.__all__), label="api_coverage.no_stale")


def test_no_untested_holes():
    """HARD as of Task 2.3: every public symbol has an asserting test or is justified EXEMPT.

    The six analytical-IC holes (earth_sun_2body, earth_sun_eccentric,
    sun_earth_jupiter_3body, harmonic_oscillator, harmonic_solution, figure_eight_period)
    were filled with asserting physics tests in tests/validation/test_analytical_physics.py
    and moved to SYMBOL_TESTS, so UNTESTED is empty. A NEW honest hole (a public symbol with
    no asserting test) re-populates UNTESTED and turns this RED until it is filled.
    """
    assert not UNTESTED, (
        f"public symbols with NO asserting test (real holes — fill them with an asserting "
        f"test and move to SYMBOL_TESTS, or justify EXEMPT with Anna): {sorted(UNTESTED)}")


def test_line_coverage_above_floor():
    """Enforce the ratchet-up line-coverage floor against the COMMITTED full-suite artifact.

    HARD as of Task 2.2: the committed ``validation/data/coverage.json`` exists, so there is no
    skip path. The artifact must come from the FULL suite (a ``-m "not slow"`` pass understates
    coverage and would game the floor), and its recorded ``total_percent`` must be
    >= LINE_COV_FLOOR. Both fields are read from the ``coverage_provenance`` stamp written by
    ``write_coverage_json`` (selector / total_percent live together there).
    """
    assert _COVERAGE_JSON.exists(), (
        f"committed full-suite coverage.json missing at {_COVERAGE_JSON} — regenerate it with "
        f"`build_test_dashboard.py --stamp-coverage <raw --cov json>` (Task 2.2).")
    cov = json.loads(_COVERAGE_JSON.read_text())
    provenance = cov["coverage_provenance"]
    selector = provenance["selector"]
    assert selector == FULL_SELECTOR, (
        f"coverage.json selector={selector!r} (not {FULL_SELECTOR!r}) — regenerate with the "
        f"FULL suite; a partial selector understates coverage and would game the floor.")
    total = float(provenance["total_percent"])
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
            "documented for C4 anti-drift review]:\n  " + "\n  ".join(divergences)
        )
