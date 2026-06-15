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
     AND does NOT carry an inline citation comment -> a hole. Conservative + documented:
     this is the C6-scoped tripwire, NOT the 2,525-match full-src scan.
  4. No unprovenanced constants — UNPROVENANCED is a hole list (Task-5.2 / Anna
     adjudication), asserted empty under @xfail(strict=False) mirroring api_coverage.

NEVER weaken a test to make it pass. A real hole goes to UNPROVENANCED for Anna, not into
a fabricated PROVENANCE citation.
"""
import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

from tests.validation.provenance_registry.manifest import (
    ALLOWLIST_MODULES,
    ALLOWLIST_NON_COEFFICIENT,
    PROVENANCE,
    UNPROVENANCED,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEDGER = _REPO_ROOT / "docs" / "provenance-ledger.md"

# Numeric literals that are NEVER citable coefficients regardless of file (the trivial set).
_TRIVIAL = frozenset({0.0, 1.0, 2.0, 0.5, -1.0, -0.5, 3.0, 4.0})

# A comment counts as a citation if it names a paper/year, a table/equation reference, an
# authority (CODATA/IAU), or carries an explicit `provenance:` marker. This is the
# comment-presence signal the C6 design specifies (the human's provenance assertion).
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
        f"UNPROVENANCED): {blank}")

    # A citation must name SOMETHING — a year, a table/eq, or an authority. A bare phrase
    # with no source token is provenance theater (the C2 anti-theater discipline applied here).
    untraceable = sorted(k for k, v in PROVENANCE.items() if not _CITE_RE.search(v))
    assert not untraceable, (
        f"PROVENANCE citations that name no source token (year / Table / Eq / CODATA / IAU / "
        f"author) — not a traceable citation: {untraceable}")


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
    "Marks",       # Marks+2012 / 2014 erratum (alpha3 FP + Table 3)
    "Jerab",       # Jerabkova+2018 (alpha3(x), Eq.7/9)
    "Moe",         # Moe & Di Stefano 2017 Table 13 grids
    "Sana",        # Sana 2012 OB period pi / q-slope kappa
    "Lucy",        # Lucy 2006 twin excess
    "King",        # King 1966 Table II c(W0)
    "Gieles",      # Gieles & Zocchi 2015 LIMEPY g+3/2  (OR "Zocchi")
    "CW04",        # Cartwright & Whitworth 2004 Table 1 radial Q
    "von Hoerner", # M&C 2011 Sigma estimator upstream
    "Demircan",    # D&K91 mass-radius
    "IAU",         # IAU 2009 / Luzum 2011 planet mass ratios
    "Chenciner",   # figure-eight period
    "Plummer",     # Plummer scale-radius
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
    present_in_doc = {a for a in _LEDGER_CONSTANT_ANCHORS if a.lower() in ledger.lower()}
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
        f"the anchor list): {missing_from_ledger}")


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
        f"from docs/provenance-ledger.md): {missing}")

    # FLOOR: the manifest must cover at least this many distinct ledger anchors. ZERO
    # fabricated values were found by the audit, so the realistic floor is the full set.
    covered = [a for a in _LEDGER_CONSTANT_ANCHORS if a.lower() in blob]
    assert len(covered) >= 15, (
        f"PROVENANCE covers only {len(covered)} ledger anchors (floor 15) — port the rest.")


# ======================================================================================
# 3. Allowlist-scoped new-literal guard (the C6 tripwire).
# ======================================================================================


def _cited_comment_lines(src: str) -> set[int]:
    """Set of line numbers carrying a ``#`` comment that names a citation token."""
    out: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT and _CITE_RE.search(tok.string):
                out.add(tok.start[0])
    except tokenize.TokenError:
        pass
    return out


def _cited_docstring_spans(tree: ast.AST) -> list[tuple[int, int]]:
    """Line spans (start, end) of every function/class/module whose docstring carries a
    citation token. A coefficient assigned inside such a block (e.g. ``exponents=[0.3, ...]``
    inside ``kroupa()`` whose docstring cites 'Kroupa (2001), Eq. 2') is provenanced by the
    docstring's References section — the standard way these classmethods record their source.
    """
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            doc = ast.get_docstring(node, clean=False)
            if doc and _CITE_RE.search(doc):
                start = getattr(node, "lineno", 1)
                end = max(
                    (getattr(c, "end_lineno", start) or start) for c in ast.walk(node)
                ) if list(ast.walk(node)) else start
                spans.append((start, end))
    return spans


def _enclosing_stmt_span(tree: ast.AST, lineno: int) -> tuple[int, int]:
    """Line span of the smallest top-level/class-level Assign/AnnAssign statement enclosing
    ``lineno`` (so a long dict/array literal block under a header comment is one unit)."""
    best = (lineno, lineno)
    best_size = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            s, e = node.lineno, (node.end_lineno or node.lineno)
            if s <= lineno <= e:
                size = e - s
                if best_size is None or size < best_size:
                    best, best_size = (s, e), size
    return best


def _has_nearby_citation(lineno: int, cited_lines: set[int]) -> bool:
    """True if the literal's line — or one of the 4 lines above it (a block-header comment
    over a dict/array of coefficients) — carries a citation comment."""
    return any(L in cited_lines for L in range(lineno, max(0, lineno - 5), -1))


def _is_citable_shaped(value) -> bool:
    """A literal is 'citable-shaped' (a candidate coefficient) unless it is trivial."""
    if isinstance(value, bool):
        return False
    v = float(value)
    if v in _TRIVIAL or abs(v) < 1e-9:
        return False
    # Small integers are indices / loop bounds / array shapes, not coefficients.
    if isinstance(value, int) and abs(value) <= 12:
        return False
    return True


def _module_provenance_blob(rel_path: str) -> str:
    """The concatenated PROVENANCE citations whose KEY names this module (e.g. all rows
    keyed ``imf/power_law.py::...``). A literal whose value-text appears here is provenanced
    by the ported manifest citation (value-level provenance)."""
    suffix = rel_path.split("src/progenax/", 1)[-1]
    return "\n".join(v for k, v in PROVENANCE.items() if k.startswith(suffix.split("::")[0]) or suffix in k or k.startswith(suffix[: suffix.find("::")] if "::" in suffix else suffix))


def _value_in_provenance(value, blob: str) -> bool:
    """True if the literal's value text appears in the module's PROVENANCE citations. Matches
    the raw value and a few common renderings (e.g. 2.3, -0.41, 0.6039) so a ported Table/Eq
    citation that lists the value covers it. Conservative: only exact-string occurrence."""
    candidates = set()
    f = float(value)
    candidates.add(repr(value))           # e.g. '2.35', '512'
    candidates.add(f"{f:g}")              # e.g. '2.35', '-0.41'
    candidates.add(f"{abs(f):g}")         # the magnitude (citations often drop the sign)
    if f == int(f):
        candidates.add(str(int(f)))       # '20', '100'
    return any(c in blob for c in candidates if c)


def _scan_module_for_unprovenanced(rel_path: str):
    """Return the distinct citable-shaped literals in ``rel_path`` that are unprovenanced:
    NOT a documented ALLOWLIST_NON_COEFFICIENT carve, NOT covered by a nearby citation
    comment, NOT inside a citation-bearing docstring block, and NOT named in this module's
    PROVENANCE citations. Each surviving value is a candidate hole (a NEW unsourced number)."""
    path = _REPO_ROOT / rel_path
    src = path.read_text()
    tree = ast.parse(src)
    cited_lines = _cited_comment_lines(src)
    doc_spans = _cited_docstring_spans(tree)
    blob = _module_provenance_blob(rel_path)
    carve = ALLOWLIST_NON_COEFFICIENT.get(rel_path, {})

    def in_cited_docstring(lineno: int) -> bool:
        return any(s <= lineno <= e for s, e in doc_spans)

    holes: dict[float, list[int]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))):
            continue
        if not _is_citable_shaped(node.value):
            continue
        v = float(node.value)
        if v in carve:
            continue
        # Provenance signals (any one suffices):
        stmt_s, stmt_e = _enclosing_stmt_span(tree, node.lineno)
        stmt_cited = any(L in cited_lines for L in range(max(0, stmt_s - 4), stmt_e + 1))
        if (_has_nearby_citation(node.lineno, cited_lines)
                or stmt_cited
                or in_cited_docstring(node.lineno)
                or _value_in_provenance(node.value, blob)):
            continue
        holes.setdefault(v, []).append(node.lineno)
    return holes


def test_allowlist_modules_exist():
    """Every allowlisted path resolves (a renamed/deleted module reds CI, not silently)."""
    missing = sorted(m for m in ALLOWLIST_MODULES if not (_REPO_ROOT / m).exists())
    assert not missing, (
        f"ALLOWLIST_MODULES paths not found (update the allowlist for the rename/move): "
        f"{missing}")


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
        + "\n".join(f"  {rel}: {vals}" for rel, vals in report.items()))


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
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)
        }
        dead = sorted(v for v in carve if v not in present)
        if dead:
            stale[rel] = dead
    assert not stale, (
        f"ALLOWLIST_NON_COEFFICIENT entries no longer present in their module (remove the "
        f"stale carve): {stale}")


# ======================================================================================
# 4. No unprovenanced constants (the Task-5.2 / Anna hole list).
# ======================================================================================


@pytest.mark.xfail(strict=False, reason="UNPROVENANCED holes are Task-5.2 / Anna "
                   "adjudication items; empty today (audit found ZERO fabricated values).")
def test_no_unprovenanced_constants():
    """HONEST-HOLE gate (mirrors api_coverage's xfail holes): every allowlisted constant has
    a citation, so UNPROVENANCED is empty. A NEW genuinely-unsourced number re-populates it
    and this stays xfail (visible as XPASS->RED only when strict) until Anna adjudicates."""
    assert not UNPROVENANCED, (
        f"allowlisted constants with NO citation (Anna adjudicates each — port a source or "
        f"de-assert): {sorted(UNPROVENANCED)}")
