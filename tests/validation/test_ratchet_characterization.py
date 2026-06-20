"""Characterization safety-net for the ``jaxstro.testing.ratchet`` hoist (Task 2.1/2.4).

Phase 2 of the registry-harness hoist deletes each progenax registry's *inline*
partition / staleness / literal-scanner / citation-proximity / node-id helpers and calls
``jaxstro.testing.ratchet`` instead (Tasks 2.2-2.5). This module pins the *observable
behavior* of the harness primitives as a FROZEN GOLDEN so the refactor is provably correct.

By construction it depends ONLY on ``jaxstro.testing.ratchet`` + frozen expected data +
the four manifests — NOT on any inline helper (those are deleted). The harness primitives
must keep reproducing the golden.

STRICT-MODE ADOPTION (Task 2.4, 2026-06-19)
-------------------------------------------
progenax has ADOPTED the harness's stricter SoTA provenance behavior: the provenance
registry now uses ``has_nearby_citation`` directly, which DELIBERATELY EXCLUDES the
module-level docstring from citation whitelisting (a module docstring naming a paper must
NOT stand in for per-coefficient provenance — the tripwire-defeat fix). To re-green under
that stricter rule, the already-PDF-verified Tout+1996 (``stellar.py``) and Moe & Di Stefano
2017 (``moe_di_stefano.py``) coefficient arrays now carry their OWN in-window citation
comments, so each array is provenanced by a scoped/in-window citation rather than by the
module docstring. The end-to-end registry verdict is therefore STILL ZERO holes — but now
under strict semantics with real per-array citations, NOT the old module-docstring whitelist.

What is pinned
--------------
1. Literal-scanner output (the critical one): per ALLOWLIST_MODULE, a frozen (count,
   sha256) fingerprint of ``scan_module_numeric_literals`` over the SAME trivial set /
   small_int_max the provenance registry uses (FAST: a handful of files only).
2. Citation-proximity verdicts: representative (module, lineno) pairs exercising the harness
   True path (scoped function/class docstring; in-window comment — INCLUDING the Tout / Moe
   coefficient rows that now carry their own in-window citation comment post-adoption), plus
   a synthetic-fixture proof that a MODULE-level docstring does NOT whitelist a literal.
3. Partition / staleness pass on the CURRENT manifests for the three ``__all__``-keyed
   registries (api_coverage, physics_registry, grad_audit).
4. End-to-end provenance verdict = ZERO holes, reproduced with the SAME strict orchestration
   the registry now runs (harness scanner + harness ``has_nearby_citation`` — NO module-
   docstring whitelist — + the local value-in-provenance carve).
5. A representative ``test_body_has_assert`` + ``resolve_node_ids`` snapshot.

Harness signed-literal folding (DELIBERATE, documented — see ``test_documented_harness_
signed_literal_folding``): the harness FOLDS signed literals (``-2.0`` survives where a raw-
``Constant`` walk saw a trivial ``2.0``). It is a SoTA fix that closes a blind spot; the
value is provenanced, so the registry verdict is unchanged.

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
# Frozen from the current source tree (2026-06-19, re-frozen post strict-mode adoption for
# stellar.py + moe_di_stefano.py, which gained per-array in-window citation comments). Each
# value is (distinct (value, lineno) count, sha256 of the sorted list). The harness
# scan_module_numeric_literals must reproduce this EXACTLY for every allowlist module. A
# drift here means the refactored provenance registry would scan a different literal set —
# the exact regression this safety-net exists to catch.
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
        "2d5984c84a11a94b29ab8b4125c4bf6a9a243285fdc2ed8d8168a3f5ae1fc60c",
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
    "src/progenax/stellar.py": [(-48.96066856, 57)],
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
# All True: a scoped function/class docstring OR an in-window citation comment cites the
# literal. The Tout (stellar.py) and Moe (moe_di_stefano.py) coefficient rows are True
# because — under strict-mode adoption (Task 2.4) — each array now carries its OWN in-window
# citation comment ("Tout+1996 Table 1/2", "Moe & Di Stefano (2017) ... Table 13"), NOT
# because the module docstring whitelists them (the harness excludes ast.Module; that
# tripwire-defeat exclusion is proved on a synthetic fixture below).
_GOLDEN_CITATION_VERDICTS: list[tuple[str, int, bool]] = [
    # True via the scoped METHOD docstring (Salpeter (1955) in the classmethod docstring).
    ("src/progenax/imf/power_law.py", 147, True),
    # True via the scoped CLASS docstring (SanaOBPeriod cites Sana et al. (2012)).
    ("src/progenax/binaries/period.py", 138, True),
    # True via the per-array in-window citation comment added at strict-mode adoption: the
    # Moe Table-13 gamma_largeq rows (lines 191-192) carry "Moe & Di Stefano (2017) ... Table
    # 13" on the array-opening line within window=4.
    ("src/progenax/imf/binary/moe_di_stefano.py", 191, True),
    ("src/progenax/imf/binary/moe_di_stefano.py", 192, True),
    # True via the per-row in-window citation comment: each Tout L/R coefficient row now
    # carries "Tout+1996 Table 1/2" inline.
    ("src/progenax/stellar.py", 40, True),
    ("src/progenax/stellar.py", 57, True),
]


def test_citation_proximity_matches_frozen_verdicts():
    """``has_nearby_citation`` reproduces the frozen True verdicts (same window=4) for the
    scoped-docstring and (post-adoption) per-array in-window-comment citation paths."""
    mismatches = []
    for rel, lineno, expected in _GOLDEN_CITATION_VERDICTS:
        got = ratchet.has_nearby_citation(rel, lineno, window=_CITE_WINDOW)
        if got != expected:
            mismatches.append(f"{rel}:{lineno} expected={expected} got={got}")
    assert not mismatches, (
        "has_nearby_citation drifted from frozen verdicts:\n  "
        + "\n  ".join(mismatches)
    )


def test_module_docstring_does_not_whitelist_a_literal(tmp_path):
    """Tripwire-defeat regression pin (now strict-mode REALITY, not just harness theory): a
    literal whose ONLY citation is the MODULE-level docstring must NOT register as cited —
    the harness excludes ``ast.Module``.

    Proved on a SYNTHETIC fixture (decoupled from progenax src, where every real coefficient
    now carries its own in-window citation): a module whose docstring cites "Table 1 (1996)"
    but whose coefficient row has no scoped/in-window citation of its own -> False. The
    contrasting positive control (the same literal WITH an in-window comment) -> True, so the
    test cannot pass vacuously."""
    fixture = tmp_path / "moddocstring_only.py"
    fixture.write_text(
        '"""A module whose docstring cites Tout (1996), Table 1 — but no scoped citation."""\n'
        "VALUE = 0.39704170\n"  # uncited coefficient row (only the module docstring cites)
    )
    # line 2 = the uncited coefficient; the module docstring (line 1) must NOT whitelist it.
    assert ratchet.has_nearby_citation(fixture, 2, window=_CITE_WINDOW) is False

    cited = tmp_path / "inwindow_cited.py"
    cited.write_text(
        '"""Plain module docstring, no citation."""\n'
        "VALUE = 0.39704170  # Tout (1996), Table 1\n"  # in-window comment cites the row
    )
    assert ratchet.has_nearby_citation(cited, 2, window=_CITE_WINDOW) is True


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


def _harness_orchestrated_holes(rel: str) -> dict[float, list[int]]:
    """Reproduce the provenance registry's STRICT hole detection: HARNESS scanner + HARNESS
    nearby-citation (which excludes the module docstring) + the LOCAL carve and value-in-
    provenance check. This mirrors the post-adoption registry orchestration EXACTLY — there
    is NO module-docstring whitelist anymore (Task 2.4 strict-mode adoption)."""
    blob = _module_provenance_blob(rel)
    carve = ALLOWLIST_NON_COEFFICIENT.get(rel, {})

    holes: dict[float, list[int]] = {}
    for value, lineno in ratchet.scan_module_numeric_literals(
        _REPO_ROOT / rel, trivial=_TRIVIAL, small_int_max=_SMALL_INT_MAX
    ):
        if value in carve:
            continue
        if ratchet.has_nearby_citation(_REPO_ROOT / rel, lineno, window=_CITE_WINDOW):
            continue
        if _value_in_provenance(value, blob):
            continue
        holes.setdefault(value, []).append(lineno)
    return holes


def test_harness_orchestration_reproduces_zero_holes():
    """The registry reports ZERO unprovenanced holes under STRICT semantics. Reproducing the
    orchestration with harness primitives (+ the local carve / value-in-provenance check, and
    NO module-docstring whitelist) must ALSO yield zero holes — the proof that the strict
    refactor + the per-array in-window citations added at adoption truly close every hole."""
    report = {
        rel: h for rel in ALLOWLIST_MODULES if (h := _harness_orchestrated_holes(rel))
    }
    assert not report, (
        "harness-orchestrated provenance scan found holes the registry does not (parity "
        "broken — the local orchestration is not faithfully reproducing the strict registry "
        "semantics):\n" + "\n".join(f"  {rel}: {h}" for rel, h in report.items())
    )


# ======================================================================================
# 5. Documented harness SoTA behaviors (pinned so the deltas stay deliberate).
# ======================================================================================


def test_documented_harness_signed_literal_folding():
    """Pin the harness's DELIBERATE signed-literal folding so it cannot silently change:
    the harness yields the Moe Table-13 ``-2.0`` tail (lines 191-192) as a SIGNED citable
    literal, whereas a raw-``Constant`` walk sees only a trivial ``+2.0`` (the ``-`` lives in
    a ``UnaryOp(USub)``). The harness is stricter (closes a sign blind spot); the value is
    provenanced, so the registry verdict is unchanged."""
    rel = "src/progenax/imf/binary/moe_di_stefano.py"

    # signed -2.0 IS in the harness scan ...
    harness_lits = set(_scan(rel))
    assert (-2.0, 191) in harness_lits and (-2.0, 192) in harness_lits

    # ... and a raw-Constant walk would NOT see it (it sees a trivial +2.0).
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


def test_harness_excludes_module_docstring_span():
    """Pin the harness's module-docstring EXCLUSION (the strict-mode behavior progenax has
    adopted): ``_cited_docstring_spans`` must NOT emit a whole-file ``(1, N)`` span for a
    module whose docstring cites a paper. moe_di_stefano.py's module docstring cites Moe
    (2017), yet no scoped-docstring span starts at line 1."""
    rel = "src/progenax/imf/binary/moe_di_stefano.py"
    tree = ast.parse((_REPO_ROOT / rel).read_text())
    harness_spans = ratchet._cited_docstring_spans(tree, ratchet.DEFAULT_CITE_RE)
    assert not any(s == 1 for s, _ in harness_spans), (
        "harness unexpectedly included the module-level docstring span (tripwire-defeat "
        "regression — a module docstring would whitelist the whole file)"
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
