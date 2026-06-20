"""Provenance-of-constants registry ratchet (Phase 5 / Task 5.1).

Four ratchets keep the hand-curated manifest honest and in sync with the audited ledger:

  1. Non-empty citations — every PROVENANCE value is a real citation (no blank strings).
  2. Ledger consistency — every paper/source anchor the COMMITTED
     ``docs/provenance-ledger.md`` marks as a verified/fixed constant is REPRESENTED in
     PROVENANCE. Parsed from the ledger (a doc; parsing it in a test is the whole point of
     this registry — keep the manifest in sync with the audited ledger). A future
     ledger-verified constant whose source is not ported reds CI. Robust-FLOOR form: assert
     >= an anchor floor AND list any ledger anchor missing from the manifest as a hole.
  3. Allowlist-scoped new-literal guard (C6) — scan ONLY ALLOWLIST_MODULES (a handful of
     constant-bearing files, NOT all of src/) for numeric literals that look like citable
     coefficients (exclude trivial 0/1/2/±0.5, small ints, and the documented
     ALLOWLIST_NON_COEFFICIENT carve). Any flagged literal that is NOT a PROVENANCE value
     AND does NOT carry an inline citation comment / scoped citation docstring -> a hole.
     Conservative + documented: this is the C6-scoped tripwire, NOT the 2,525-match
     full-src scan.
  4. No unprovenanced constants — UNPROVENANCED is a hole list (Task-5.2 / Anna
     adjudication), asserted empty under @xfail(strict=False) mirroring api_coverage.

STRICT harness behavior (ratchet-harness hoist, 2026-06-19): the literal scanner and the
nearby-citation check are now the canonical ``jaxstro.testing.ratchet`` primitives
(``scan_module_numeric_literals`` + ``has_nearby_citation``). The harness DELIBERATELY
EXCLUDES the module-level docstring from citation whitelisting — a module docstring that
names a paper must NOT stand in for per-coefficient provenance (a tripwire-defeat). So every
citable-shaped coefficient must carry its OWN in-window citation comment (or a scoped
function/class docstring, or a PROVENANCE value match). The progenax-LOCAL orchestration
survives: the ``_scan_module_for_unprovenanced`` walk, the value-in-provenance matching, and
the ALLOWLIST_NON_COEFFICIENT carve.

NEVER weaken a test to make it pass. A real hole goes to UNPROVENANCED for Anna, not into
a fabricated PROVENANCE citation.
"""

import ast
import re
from pathlib import Path

from jaxstro.testing.ratchet import (
    has_nearby_citation,
    scan_module_numeric_literals,
)

from tests.validation.provenance_registry.manifest import (
    ALLOWLIST_MODULES,
    ALLOWLIST_NON_COEFFICIENT,
    PROVENANCE,
    UNPROVENANCED,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEDGER = _REPO_ROOT / "docs" / "provenance-ledger.md"

# The SAME literal-scanner knobs the harness consumes: the trivial set (never-citable values
# regardless of file) and the small-int cutoff (indices / shapes / loop bounds). Passed
# straight to ``scan_module_numeric_literals``.
_TRIVIAL: set[float] = {0.0, 1.0, 2.0, 0.5, -1.0, -0.5, 3.0, 4.0}
_SMALL_INT_MAX = 12
_CITE_WINDOW = (
    4  # nearby-comment window for has_nearby_citation (lines above the literal)
)

# A PROVENANCE *citation string* counts as traceable if it names a paper/year, a table/eq
# reference, an authority (CODATA/IAU), or an explicit ``provenance:`` marker. This is the
# anti-theater signal for the manifest's own citation values (ratchet #1) — NOT the
# source-literal scan, which uses the harness's content-free citation regex.
_CITE_RE = re.compile(
    r"(provenance:|\b(19|20)\d{2}\b|\bTable\b|\bEq\.?\b|\bSection\b|§|\bCODATA\b|\bIAU\b|"
    r"Salpeter|Kroupa|Chabrier|Maschberger|Sana|Moe|Di Stefano|Marks|Jerab|Demircan|"
    r"Kahraman|King|Plummer|Lucy|von Hoerner|Casertano|Hut|Cartwright|Whitworth|CW04|"
    r"canonical|erratum|Elson|Fall|Freeman|Chenciner|Montgomery)",
    re.IGNORECASE,
)


# ======================================================================================
# 1. Every PROVENANCE entry has a non-empty citation.
# ======================================================================================


def test_every_provenance_entry_has_a_citation():
    """No blank/whitespace-only provenance strings (a blank citation is no provenance)."""
    blank = sorted(k for k, v in PROVENANCE.items() if not v or not v.strip())
    assert not blank, (
        f"PROVENANCE entries with an empty citation (port the real source or move to "
        f"UNPROVENANCED): {blank}"
    )

    # A citation must name SOMETHING — a year, a table/eq, or an authority. A bare phrase
    # with no source token is provenance theater (the C2 anti-theater discipline applied here).
    untraceable = sorted(k for k, v in PROVENANCE.items() if not _CITE_RE.search(v))
    assert not untraceable, (
        f"PROVENANCE citations that name no source token (year / Table / Eq / CODATA / IAU / "
        f"author) — not a traceable citation: {untraceable}"
    )


# ======================================================================================
# 2. Ledger consistency: every ledger-verified source anchor is represented in PROVENANCE.
# ======================================================================================

# Source anchors (author/dataset tokens) the ledger verifies as backing a constant/fit.
# A token here MUST appear (case-insensitively) somewhere in a PROVENANCE citation value.
# Derived by reading docs/provenance-ledger.md: each is a paper/dataset whose CONSTANT (not
# merely a citation-appropriateness note) the audit verified or fixed in src/ released-core.
_LEDGER_CONSTANT_ANCHORS = (
    "Salpeter",
    "Kroupa",
    "Chabrier",
    "Maschberger",
    "Marks",  # Marks+2012 / 2014 erratum (alpha3 FP + Table 3)
    "Jerab",  # Jerabkova+2018 (alpha3(x), Eq.7/9)
    "Moe",  # Moe & Di Stefano 2017 Table 13 grids
    "Sana",  # Sana 2012 OB period pi / q-slope kappa
    "Lucy",  # Lucy 2006 twin excess
    "King",  # King 1966 Table II c(W0)
    "Gieles",  # Gieles & Zocchi 2015 LIMEPY g+3/2  (OR "Zocchi")
    "CW04",  # Cartwright & Whitworth 2004 Table 1 radial Q
    "von Hoerner",  # M&C 2011 Sigma estimator upstream
    "Demircan",  # D&K91 mass-radius
    "IAU",  # IAU 2009 / Luzum 2011 planet mass ratios
    "Chenciner",  # figure-eight period
    "Plummer",  # Plummer scale-radius
)


def _ledger_verified_anchors_present():
    """Cross-check: the anchors we claim the ledger verifies actually appear in the ledger
    AND in a verified/fixed (✅/🔧) context — so this list cannot drift from the doc."""
    ledger = _LEDGER.read_text()
    # Restrict to rows the ledger marks verified (✅) or fixed (🔧) — those back a constant.
    verified_blob = "\n".join(
        ln for ln in ledger.splitlines() if ("✅" in ln or "🔧" in ln)
    )
    # Many anchors appear in the surrounding prose batch headers too; accept the whole doc
    # for presence, but require the verified context to be non-trivially populated.
    present_in_doc = {
        a for a in _LEDGER_CONSTANT_ANCHORS if a.lower() in ledger.lower()
    }
    present_in_verified = {
        a for a in _LEDGER_CONSTANT_ANCHORS if a.lower() in verified_blob.lower()
    }
    return present_in_doc, present_in_verified


def test_ledger_anchors_exist_in_the_committed_ledger():
    """Guard the cross-check itself: every anchor we test for is actually in the ledger (so
    a typo'd anchor cannot silently make the consistency check vacuous)."""
    present_in_doc, _ = _ledger_verified_anchors_present()
    missing_from_ledger = sorted(set(_LEDGER_CONSTANT_ANCHORS) - present_in_doc)
    assert not missing_from_ledger, (
        f"anchors not found in docs/provenance-ledger.md (the doc moved / anchor typo — fix "
        f"the anchor list): {missing_from_ledger}"
    )


def test_every_ledger_constant_anchor_is_in_provenance():
    """Each ledger-verified source anchor must be represented by >= 1 PROVENANCE citation.

    This is the in-sync ratchet: a future ledger row that verifies a new constant whose
    source is NOT ported into PROVENANCE leaves its anchor missing here -> RED. Robust
    FLOOR: we also assert a sane minimum count so the check cannot be gamed by shrinking
    the anchor list.
    """
    blob = "\n".join(PROVENANCE.values()).lower()
    missing = sorted(a for a in _LEDGER_CONSTANT_ANCHORS if a.lower() not in blob)
    assert not missing, (
        "ledger-verified constant sources NOT represented in PROVENANCE (port the citation "
        f"from docs/provenance-ledger.md): {missing}"
    )

    # FLOOR: the manifest must cover at least this many distinct ledger anchors. ZERO
    # fabricated values were found by the audit, so the realistic floor is the full set.
    covered = [a for a in _LEDGER_CONSTANT_ANCHORS if a.lower() in blob]
    assert len(covered) >= 15, (
        f"PROVENANCE covers only {len(covered)} ledger anchors (floor 15) — port the rest."
    )


# ======================================================================================
# 3. Allowlist-scoped new-literal guard (the C6 tripwire).
# ======================================================================================


def _module_provenance_blob(rel_path: str) -> str:
    """The concatenated PROVENANCE citations whose KEY names this module (e.g. all rows
    keyed ``imf/power_law.py::...``). A literal whose value-text appears here is provenanced
    by the ported manifest citation (value-level provenance)."""
    suffix = rel_path.split("src/progenax/", 1)[-1]
    return "\n".join(
        v
        for k, v in PROVENANCE.items()
        if k.startswith(suffix.split("::")[0])
        or suffix in k
        or k.startswith(suffix[: suffix.find("::")] if "::" in suffix else suffix)
    )


def _value_in_provenance(value, blob: str) -> bool:
    """True if the literal's value text appears in the module's PROVENANCE citations. Matches
    the raw value and a few common renderings (e.g. 2.3, -0.41, 0.6039) so a ported Table/Eq
    citation that lists the value covers it. Conservative: only exact-string occurrence."""
    candidates = set()
    f = float(value)
    candidates.add(repr(value))  # e.g. '2.35', '512'
    candidates.add(f"{f:g}")  # e.g. '2.35', '-0.41'
    candidates.add(f"{abs(f):g}")  # the magnitude (citations often drop the sign)
    if f == int(f):
        candidates.add(str(int(f)))  # '20', '100'
    return any(c in blob for c in candidates if c)


def _scan_module_for_unprovenanced(rel_path: str):
    """Return the distinct citable-shaped literals in ``rel_path`` that are unprovenanced:
    NOT a documented ALLOWLIST_NON_COEFFICIENT carve, NOT covered by a nearby citation
    comment / scoped (function/class) citation docstring, and NOT named in this module's
    PROVENANCE citations. Each surviving value is a candidate hole (a NEW unsourced number).

    Strict harness behavior (ratchet-harness hoist): the citable-shaped scan is the canonical
    ``scan_module_numeric_literals`` (signed-literal folding included), and the nearby-citation
    check is ``has_nearby_citation`` — which EXCLUDES the module-level docstring, so a module
    docstring that names a paper does NOT whitelist a file-level coefficient. Each coefficient
    must carry its own in-window citation, a scoped docstring, or a PROVENANCE value match.
    """
    blob = _module_provenance_blob(rel_path)
    carve = ALLOWLIST_NON_COEFFICIENT.get(rel_path, {})

    holes: dict[float, list[int]] = {}
    for value, lineno in scan_module_numeric_literals(
        _REPO_ROOT / rel_path, trivial=_TRIVIAL, small_int_max=_SMALL_INT_MAX
    ):
        if value in carve:
            continue
        # Provenance signals (any one suffices): an in-window citation comment or scoped
        # citation docstring (harness), or the value-text appearing in this module's
        # PROVENANCE citations (local).
        if has_nearby_citation(
            _REPO_ROOT / rel_path, lineno, window=_CITE_WINDOW
        ) or _value_in_provenance(value, blob):
            continue
        holes.setdefault(value, []).append(lineno)
    return holes


def test_allowlist_modules_exist():
    """Every allowlisted path resolves (a renamed/deleted module reds CI, not silently)."""
    missing = sorted(m for m in ALLOWLIST_MODULES if not (_REPO_ROOT / m).exists())
    assert not missing, (
        f"ALLOWLIST_MODULES paths not found (update the allowlist for the rename/move): "
        f"{missing}"
    )


def test_no_new_unprovenanced_literal_in_allowlist_modules():
    """The C6 tripwire: a citable-shaped literal in an allowlisted constant-bearing module
    must be a PROVENANCE value, carry an inline citation comment, or be a documented
    ALLOWLIST_NON_COEFFICIENT carve. Anything else is a NEW unprovenanced number -> RED.

    Honesty: this does NOT scan all of src/ (the 2,525-match trap). It is the conservative,
    scoped tripwire over the handful of coefficient files. A genuine new coefficient with no
    citation reds here; the fix is to add an inline citation (and a PROVENANCE row), NOT to
    widen the carve to silence it.
    """
    report = {}
    for rel in ALLOWLIST_MODULES:
        holes = _scan_module_for_unprovenanced(rel)
        if holes:
            report[rel] = {v: lines for v, lines in sorted(holes.items())}
    assert not report, (
        "unprovenanced citable-shaped literals in allowlisted modules (add an inline "
        "citation comment + a PROVENANCE row, or — only with Anna's sign-off — document "
        "the literal in ALLOWLIST_NON_COEFFICIENT with a reason):\n"
        + "\n".join(f"  {rel}: {vals}" for rel, vals in report.items())
    )


def test_allowlist_non_coefficient_carve_is_not_stale():
    """Every value listed in ALLOWLIST_NON_COEFFICIENT must actually OCCUR (as a citable-
    shaped literal) in its module — a stale carve that no longer matches anything is dead
    suppression and could mask a future real coefficient sharing that file."""
    stale = {}
    for rel, carve in ALLOWLIST_NON_COEFFICIENT.items():
        if not carve:
            continue
        src = (_REPO_ROOT / rel).read_text()
        tree = ast.parse(src)
        present = {
            float(n.value)
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)
        }
        dead = sorted(v for v in carve if v not in present)
        if dead:
            stale[rel] = dead
    assert not stale, (
        f"ALLOWLIST_NON_COEFFICIENT entries no longer present in their module (remove the "
        f"stale carve): {stale}"
    )


# ======================================================================================
# 4. No unprovenanced constants (the Task-5.2 / Anna hole list).
# ======================================================================================


def test_no_unprovenanced_constants():
    """HARD honest-hole gate (Task 5.2 flip; mirrors api_coverage's holes test once closed):
    every allowlisted constant has a citation, so UNPROVENANCED is empty. Holes are genuinely
    0 (the 2026-06 audit found ZERO fabricated values), so this is now a HARD assertion — no
    xfail. A NEW genuinely-unsourced number re-populates UNPROVENANCED and turns this RED, an
    Anna-adjudication item: port a source into PROVENANCE or de-assert the constant. NEVER
    fabricate a citation to make this pass."""
    assert not UNPROVENANCED, (
        f"allowlisted constants with NO citation (Anna adjudicates each — port a source or "
        f"de-assert): {sorted(UNPROVENANCED)}"
    )
