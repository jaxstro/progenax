"""Characterization safety-net for the ``jaxstro.testing.ratchet`` hoist (Task 2.1).

Phase 2 of the registry-harness hoist deletes each progenax registry's *inline*
partition / staleness / literal-scanner / citation-proximity / node-id helpers and calls
``jaxstro.testing.ratchet`` instead (Tasks 2.2-2.5). This module pins the *current
observable behavior* as a FROZEN GOLDEN so that refactor is provably behavior-preserving.

By construction it depends ONLY on ``jaxstro.testing.ratchet`` + frozen expected data +
the four manifests — NOT on any inline helper (those get deleted in 2.2-2.5). It therefore
stays valid BOTH before and after the refactor: the golden was captured ONCE from the
current source tree and the harness primitives must keep reproducing it.

What is pinned
--------------
1. Literal-scanner output (the critical one): per ALLOWLIST_MODULE, a frozen (count,
   sha256) fingerprint of ``scan_module_numeric_literals`` over the SAME trivial set /
   small_int_max the provenance registry uses (FAST: a handful of files only).
2. Citation-proximity verdicts: representative (module, lineno) pairs exercising BOTH the
   harness True path (scoped function/class docstring; comment-in-window) and the harness
   False path — INCLUDING the "module-level docstring does NOT whitelist a literal"
   tripwire-defeat fix (stellar.py / moe_di_stefano.py coefficient rows that the inline
   scanner whitelisted via the module docstring but the harness, correctly, does not).
3. Partition / staleness pass on the CURRENT manifests for the three ``__all__``-keyed
   registries (api_coverage, physics_registry, grad_audit).
4. End-to-end provenance verdict = ZERO holes, reproduced with harness primitives + the
   LOCAL orchestration that survives the refactor (the module-docstring-inclusive whitelist
   + value-in-provenance carve) — the proof the subtle docstring/sign semantics are
   preserved.
5. A representative ``test_body_has_assert`` + ``resolve_node_ids`` snapshot.

Harness-vs-inline divergences (DELIBERATE, documented — see ``test_documented_harness_vs_
inline_divergences``): the harness FOLDS signed literals (``-2.0`` survives where the inline
raw-``Constant`` walk saw a trivial ``2.0``) and EXCLUDES the module-level docstring from
citation whitelisting. Both are the SoTA fixes the hoist design intends; neither changes the
registry's observable zero-holes verdict because the affected literals are covered by
``value-in-provenance`` and the local module-docstring whitelist.

NOTE on collection: ``ratchet.test_body_has_assert`` is ``test_``-prefixed; it is reached
ONLY through the ``ratchet.`` namespace here (never imported as a bare name), so pytest does
not try to collect it as a test.
"""

import ast
import hashlib
from pathlib import Path

import progenax

from jaxstro.testing import ratchet
from tests.validation.provenance_registry.manifest import (
    ALLOWLIST_MODULES,
    ALLOWLIST_NON_COEFFICIENT,
    PROVENANCE,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The SAME knobs the provenance registry's inline scanner uses
# (tests/validation/provenance_registry/test_provenance_coverage.py::_TRIVIAL and the
# ``abs(value) <= 12`` small-int rule). The harness must reproduce that policy exactly.
_TRIVIAL: set[float] = {0.0, 1.0, 2.0, 0.5, -1.0, -0.5, 3.0, 4.0}
_SMALL_INT_MAX = 12
_CITE_WINDOW = (
    4  # the provenance registry's nearby-comment window (_has_nearby_citation)
)


def _scan(rel: str) -> list[tuple[float, int]]:
    """Sorted, de-duplicated harness literal scan of an allowlist module."""
    return sorted(
        set(
            ratchet.scan_module_numeric_literals(
                rel, trivial=_TRIVIAL, small_int_max=_SMALL_INT_MAX
            )
        )
    )


def _fingerprint(rel: str) -> tuple[int, str]:
    """(count, sha256) fingerprint of the sorted harness literal scan of a module."""
    lits = _scan(rel)
    return len(lits), hashlib.sha256(repr(lits).encode()).hexdigest()


# ======================================================================================
# 1. Literal-scanner equivalence (FROZEN GOLDEN — the critical characterization).
# ======================================================================================
#
# Frozen ONCE from the current source tree (2026-06-19). Each value is
# (distinct (value, lineno) count, sha256 of the sorted list). The harness
# scan_module_numeric_literals must reproduce this EXACTLY for every allowlist module. A
# drift here means the refactored provenance registry would scan a different literal set
# than the inline scanner did — the exact regression this safety-net exists to catch.
_GOLDEN_LITERAL_FINGERPRINTS: dict[str, tuple[int, str]] = {
    "src/progenax/imf/power_law.py": (
        14,
        "ff38228a8f550095c9bdfb0bbfea09ea0d5bb8fd0588192cdc1ba9ded7cad815",
    ),
    "src/progenax/imf/chabrier.py": (
        10,
        "6dc46f2a95a58c57063b76bfebbad80b0564e79a922524828eb6b0236e928d8b",
    ),
    "src/progenax/imf/smooth.py": (
        15,
        "00d11d4fd51f0fa15132fe8d6712b22870f6115cb7212799823c8df43041b4c8",
    ),
    "src/progenax/imf/environment/coefficients.py": (
        26,
        "6158cad4187f4ebfd4b82840a55899fe69fdba915a49feef4aae40b7bae64d21",
    ),
    "src/progenax/imf/environment/mapping.py": (
        27,
        "35b9f86d0f49a58948819ed023efa94536872469b9f0f6d1ab5acb8604dd2566",
    ),
    "src/progenax/imf/binary/moe_di_stefano.py": (
        74,
        "952cd80104797582b24b56c2806428d809e7716f9e9070078ab6b74d6d7dbe45",
    ),
    "src/progenax/imf/binary/mass_ratio.py": (
        6,
        "bacc5d0fb3c714721d548cad1623c9b6832360917105b90ed3814f68fa3abb44",
    ),
    "src/progenax/binaries/period.py": (
        15,
        "70c381bbebd070f8648d267bb5cbe3108c3f349fd7f329ca1766da6c49ac6e50",
    ),
    "src/progenax/stellar.py": (
        103,
        "82fa256012737390e82d46eed9c63c6b63c76f6121ec32318ea096d912e692f0",
    ),
}


def test_allowlist_modules_match_golden_inventory():
    """The frozen golden covers EXACTLY the registry's ALLOWLIST_MODULES (no drift in the
    scanned file set itself — a renamed/added allowlist module must be re-frozen here)."""
    assert set(_GOLDEN_LITERAL_FINGERPRINTS) == set(ALLOWLIST_MODULES), (
        "ALLOWLIST_MODULES drifted from the frozen golden inventory — re-capture the "
        "fingerprints for the new module set:\n"
        f"  only in golden: {sorted(set(_GOLDEN_LITERAL_FINGERPRINTS) - set(ALLOWLIST_MODULES))}\n"
        f"  only in manifest: {sorted(set(ALLOWLIST_MODULES) - set(_GOLDEN_LITERAL_FINGERPRINTS))}"
    )


def test_harness_literal_scan_matches_frozen_golden():
    """``scan_module_numeric_literals`` reproduces the frozen (count, sha256) per module.

    This is THE characterization: the harness scanner output is byte-for-byte what the
    provenance registry will consume post-refactor, pinned against what it produces today.
    """
    drift = {}
    for rel in ALLOWLIST_MODULES:
        got = _fingerprint(rel)
        if got != _GOLDEN_LITERAL_FINGERPRINTS[rel]:
            drift[rel] = {"golden": _GOLDEN_LITERAL_FINGERPRINTS[rel], "got": got}
    assert not drift, (
        "harness scan_module_numeric_literals drifted from the frozen golden (the refactor "
        "would change the scanned literal set — investigate before re-freezing):\n"
        + "\n".join(f"  {rel}: {d}" for rel, d in drift.items())
    )


# Hand-picked representative literals proving the signed-folding edge (the harness folds
# ``-2.0`` to a SIGNED citable literal where the trivial set excludes only ``+2.0``). These
# are exact (value, lineno) members of the frozen scan — a compact, legible spot-check on top
# of the hash, and the precise place inline-vs-harness behavior diverges.
_GOLDEN_REPRESENTATIVE_LITERALS: dict[str, list[tuple[float, int]]] = {
    # Salpeter alpha, signed Marks/Jerabkova FP coefficients, the Moe Table-13 -2.0 tail,
    # the Sana OB period index, and a Tout ZAMS coefficient row.
    "src/progenax/imf/power_law.py": [(2.35, 147)],
    "src/progenax/imf/environment/coefficients.py": [(-0.4072, 54), (-0.87, 32)],
    "src/progenax/imf/binary/moe_di_stefano.py": [
        (-2.0, 191),
        (-2.0, 192),
        (-1.1, 192),
    ],
    "src/progenax/binaries/period.py": [(-0.55, 138)],
    "src/progenax/stellar.py": [(-48.96066856, 52)],
}


def test_representative_signed_literals_present():
    """Each hand-picked representative (value, lineno) — including the signed ``-2.0`` Moe
    tail and signed Marks coefficients — is in the harness scan for its module."""
    for rel, expected in _GOLDEN_REPRESENTATIVE_LITERALS.items():
        got = set(_scan(rel))
        missing = sorted(p for p in expected if p not in got)
        assert not missing, (
            f"representative literals missing from {rel} scan: {missing}"
        )


# ======================================================================================
# 2. Citation-proximity equivalence (FROZEN GOLDEN verdicts).
# ======================================================================================
#
# (module, lineno, expected has_nearby_citation(window=4)). Captured from the current tree.
#   True  : a scoped function/class docstring (or in-window comment) cites the literal.
#   False : NO scoped citation in window. CRUCIALLY the False cases below sit inside a
#           MODULE-level docstring that names a paper (Tout 1996 / Moe Table 13) — the inline
#           _cited_docstring_spans whitelisted them via ast.Module, but the harness EXCLUDES
#           the module docstring (the tripwire-defeat fix), so it returns False. This pins
#           that the "module docstring does NOT whitelist a literal" semantics is live.
_GOLDEN_CITATION_VERDICTS: list[tuple[str, int, bool]] = [
    # True via the scoped METHOD docstring (Salpeter (1955) in the classmethod docstring).
    ("src/progenax/imf/power_law.py", 147, True),
    # True via the scoped CLASS docstring (SanaOBPeriod cites Sana et al. (2012)).
    ("src/progenax/binaries/period.py", 138, True),
    # False: the Moe Table-13 -2.0 tail row is whitelisted ONLY by the module docstring
    # (excluded by the harness) — no scoped/in-window citation.
    ("src/progenax/imf/binary/moe_di_stefano.py", 191, False),
    # False: a Tout ZAMS coefficient row; the "Tout+1996 Table 1" header comment is >4 lines
    # above, and only the module docstring otherwise cites it (excluded by the harness).
    ("src/progenax/stellar.py", 37, False),
]


def test_citation_proximity_matches_frozen_verdicts():
    """``has_nearby_citation`` reproduces the frozen True/False verdicts (same window=4),
    including the module-docstring-does-NOT-whitelist tripwire-defeat case."""
    mismatches = []
    for rel, lineno, expected in _GOLDEN_CITATION_VERDICTS:
        got = ratchet.has_nearby_citation(rel, lineno, window=_CITE_WINDOW)
        if got != expected:
            mismatches.append(f"{rel}:{lineno} expected={expected} got={got}")
    assert not mismatches, (
        "has_nearby_citation drifted from frozen verdicts:\n  "
        + "\n  ".join(mismatches)
    )


def test_module_docstring_does_not_whitelist_a_literal():
    """Explicit tripwire-defeat regression pin: a coefficient literal whose ONLY citation is
    the MODULE docstring must NOT register as cited (the harness excludes ast.Module)."""
    # stellar.py module docstring cites Tout 1996; line 37 is a coefficient row with no
    # scoped/in-window citation of its own.
    assert (
        ratchet.has_nearby_citation("src/progenax/stellar.py", 37, window=_CITE_WINDOW)
        is False
    )


# ======================================================================================
# 3. Partition / staleness pass on the CURRENT manifests (the registries are full today).
# ======================================================================================


def _public() -> set[str]:
    return set(progenax.__all__)


def test_api_coverage_partition_passes_today():
    """The api_coverage manifest exactly partitions __all__ under the harness primitive."""
    from tests.validation.api_coverage.manifest import EXEMPT, SYMBOL_TESTS, UNTESTED

    ratchet.assert_partition(
        _public(), SYMBOL_TESTS, EXEMPT, UNTESTED, label="api_coverage.partition"
    )


def test_physics_registry_partition_passes_today():
    """The physics_registry manifest exactly partitions __all__ under the harness."""
    from tests.validation.physics_registry.manifest import (
        EXEMPT_NON_EQUILIBRIUM_MODEL,
        EXEMPT_NON_MODEL,
        MODEL_INVARIANTS,
        UNTESTED_MODELS,
    )

    ratchet.assert_partition(
        _public(),
        MODEL_INVARIANTS,
        EXEMPT_NON_MODEL,
        EXEMPT_NON_EQUILIBRIUM_MODEL,
        UNTESTED_MODELS,
        label="physics_registry.partition",
    )


def test_grad_audit_partition_and_staleness_pass_today():
    """grad_audit's SYMBOL_CATEGORY partitions __all__ and carries no stale key."""
    from tests.validation.grad_audit.manifest import SYMBOL_CATEGORY

    public = _public()
    ratchet.assert_partition(public, SYMBOL_CATEGORY, label="grad_audit.partition")
    ratchet.assert_no_stale(SYMBOL_CATEGORY, public, label="grad_audit.symbol_category")


# ======================================================================================
# 4. End-to-end provenance verdict = ZERO holes (harness primitives + local orchestration).
# ======================================================================================


def _module_provenance_blob(rel: str) -> str:
    """LOCAL orchestration (survives the refactor): the PROVENANCE citations keyed to this
    module — a literal whose value-text appears here is provenanced. Mirrors the registry's
    inline ``_module_provenance_blob``; kept here so the characterization does not import the
    soon-deleted inline helper."""
    suffix = rel.split("src/progenax/", 1)[-1]
    head = suffix[: suffix.find("::")] if "::" in suffix else suffix
    return "\n".join(
        v
        for k, v in PROVENANCE.items()
        if k.startswith(suffix.split("::")[0]) or suffix in k or k.startswith(head)
    )


def _value_in_provenance(value: float, blob: str) -> bool:
    """LOCAL orchestration: the literal's value-text (signed repr, %g, magnitude, or int
    form) occurs in the module's PROVENANCE citations. Mirrors the registry's inline
    ``_value_in_provenance``."""
    candidates = {repr(value), f"{value:g}", f"{abs(value):g}"}
    if value == int(value):
        candidates.add(str(int(value)))
    return any(c in blob for c in candidates if c)


def _cited_docstring_spans_module_inclusive(tree: ast.AST) -> list[tuple[int, int]]:
    """LOCAL orchestration: the provenance registry whitelists a literal that sits inside a
    citation-bearing docstring of ANY scope INCLUDING the module (its inline
    ``_cited_docstring_spans`` walks ``ast.Module`` too). The harness deliberately drops the
    module scope; the registry layers the module whitelist back on locally. This reproduces
    the registry's inline span set so the end-to-end verdict matches."""
    import re

    cite_re = re.compile(
        r"(provenance:|\b(18|19|20)\d{2}\b|\bTable\b|\bEq\.?\b|\bSection\b|§|"
        r"\bCODATA\b|\bIAU\b|\bp(?:p|g|age)?\.?\s*\d|"
        r"Salpeter|Kroupa|Chabrier|Maschberger|Sana|Moe|Di Stefano|Marks|Jerab|Demircan|"
        r"Kahraman|King|Plummer|Lucy|von Hoerner|Casertano|Hut|Cartwright|Whitworth|CW04|"
        r"canonical|erratum|Elson|Fall|Freeman|Chenciner|Montgomery|Tout|Pols|Eggleton)",
        re.IGNORECASE,
    )
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc and cite_re.search(doc):
                start = getattr(node, "lineno", 1)
                ends = [getattr(c, "end_lineno", None) for c in ast.walk(node)]
                end = max([e for e in ends if e is not None], default=start)
                spans.append((start, end))
    return spans


def _harness_orchestrated_holes(rel: str) -> dict[float, list[int]]:
    """Reproduce the provenance registry's hole detection with the HARNESS scanner +
    HARNESS nearby-citation, plus the LOCAL carve / value-in-provenance / module-inclusive
    docstring whitelist (the orchestration that stays progenax-local through the refactor)."""
    src = (_REPO_ROOT / rel).read_text()
    tree = ast.parse(src)
    blob = _module_provenance_blob(rel)
    carve = ALLOWLIST_NON_COEFFICIENT.get(rel, {})
    doc_spans = _cited_docstring_spans_module_inclusive(tree)

    holes: dict[float, list[int]] = {}
    for value, lineno in ratchet.scan_module_numeric_literals(
        rel, trivial=_TRIVIAL, small_int_max=_SMALL_INT_MAX
    ):
        if value in carve:
            continue
        if ratchet.has_nearby_citation(rel, lineno, window=_CITE_WINDOW):
            continue
        if any(s <= lineno <= e for s, e in doc_spans):
            continue
        if _value_in_provenance(value, blob):
            continue
        holes.setdefault(value, []).append(lineno)
    return holes


def test_harness_orchestration_reproduces_zero_holes():
    """The registry reports ZERO unprovenanced holes today. Reproducing the orchestration
    with harness primitives (+ the local carve / value-in-provenance / module-docstring
    whitelist) must ALSO yield zero holes — the proof the refactor preserves the observable
    verdict despite the harness's stricter signed-literal and module-docstring semantics."""
    report = {
        rel: h for rel in ALLOWLIST_MODULES if (h := _harness_orchestrated_holes(rel))
    }
    assert not report, (
        "harness-orchestrated provenance scan found holes the inline registry does not "
        "(parity broken — the local orchestration is not faithfully reproducing the inline "
        "semantics):\n" + "\n".join(f"  {rel}: {h}" for rel, h in report.items())
    )


# ======================================================================================
# 5. Documented harness-vs-inline divergences (pinned so the SoTA deltas stay deliberate).
# ======================================================================================


def test_documented_harness_vs_inline_divergences():
    """Pin the two DELIBERATE harness-vs-inline differences so they cannot silently change:

    (a) Signed-literal folding — the harness yields the Moe Table-13 ``-2.0`` tail (lines
        191-192) as a citable literal; the inline raw-``Constant`` walk saw only a trivial
        ``+2.0`` and dropped it. The harness is stricter (closes a blind spot); the value is
        provenanced, so the verdict is unchanged.
    (b) Module-docstring exclusion — the harness excludes ``ast.Module`` from citation
        whitelisting; the inline ``_cited_docstring_spans`` included it.
    """
    rel = "src/progenax/imf/binary/moe_di_stefano.py"

    # (a) signed -2.0 IS in the harness scan ...
    harness_lits = set(_scan(rel))
    assert (-2.0, 191) in harness_lits and (-2.0, 192) in harness_lits

    # ... and the inline raw-Constant walk would NOT see it (it sees a trivial +2.0).
    tree = ast.parse((_REPO_ROOT / rel).read_text())
    raw_constants_at_191_192 = {
        (float(n.value), n.lineno)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, (int, float))
        and not isinstance(n.value, bool)
        and n.lineno in (191, 192)
    }
    # The raw constants are +2.0 (trivial); the SIGNED -2.0 only appears via UnaryOp folding.
    assert (2.0, 191) in raw_constants_at_191_192
    assert (-2.0, 191) not in raw_constants_at_191_192

    # (b) the harness scoped-docstring spans EXCLUDE the whole-file (1, N) module span; the
    # module-inclusive local reproduction INCLUDES it.
    harness_spans = ratchet._cited_docstring_spans(tree)
    local_spans = _cited_docstring_spans_module_inclusive(tree)
    assert not any(s == 1 for s, _ in harness_spans), (
        "harness unexpectedly included the module-level docstring span"
    )
    assert any(s == 1 for s, _ in local_spans), (
        "local module-inclusive reproduction lost the module-level docstring span"
    )


# ======================================================================================
# 6. Representative node-id resolution + asserts-behavior snapshot.
# ======================================================================================

_REPRESENTATIVE_NODE_ID = (
    "tests/unit/binaries/test_binaries.py::"
    "TestKeplerElements::test_circular_orbit_creation"
)


def test_representative_node_id_resolves_and_asserts():
    """A known-good cited node id resolves under ``resolve_node_ids`` and its body asserts
    behavior under ``ratchet.test_body_has_assert`` (called through the namespace so pytest
    does not collect the ``test_``-prefixed helper)."""
    resolved = ratchet.resolve_node_ids(
        [_REPRESENTATIVE_NODE_ID], rootdir=str(_REPO_ROOT)
    )
    assert resolved == {_REPRESENTATIVE_NODE_ID}, (
        f"representative node id did not resolve: {_REPRESENTATIVE_NODE_ID}"
    )
    assert ratchet.test_body_has_assert(_REPRESENTATIVE_NODE_ID) is True
