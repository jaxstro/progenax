"""Provenance-cards enforcement test — the 5th registry (ADR-0034, design §5).

For every model card in ``docs/provenance/registry/*.yaml``, assert:

1. every ``sources[].bibkey`` resolves in ``docs/website/references.bib`` AND that
   bib entry carries a public pointer (``doi`` or ``eprint``; ``adsurl`` accepted
   for pre-DOI literature — pre-2001 A&A has no DOIs).
2. every ``code_ref`` (``relpath-under-src/progenax::qualname``) imports and the
   qualname resolves.
3. every ``validation`` pytest node id resolves under collection AND its body
   asserts (anti-theater: reuse ``jaxstro.testing.ratchet``).
4. ``status: verified`` => >=1 code_ref AND >=1 validation entry.
5. coverage ratchet: every ``manifest.CARDED_MODELS`` entry resolves to a card,
   and once ``REGISTRY_FULL`` flips, every physics-registry model is carded.
6. glossary freshness: the committed ``docs/website/15-model-reference/*.md``
   pages byte-match a fresh ``--emit`` regeneration (docs cannot drift).

This complements the per-CONSTANT provenance ratchet (which guards magic numbers);
this registry guards model-level equations/parameters/sources.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from jaxstro.testing.ratchet import (
    resolve_node_ids,
)
from jaxstro.testing.ratchet import (
    test_body_has_assert as _body_has_assert,
)

from .manifest import CARDED_MODELS, REGISTRY_FULL

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts"))

from build_provenance_registry import (  # noqa: E402
    GLOSSARY_DIR,
    REGISTRY_DIR,
    load_cards,
    render_family,
)

# ---------------------------------------------------------------------------
# Shared fixtures (module scope: parse once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def families():
    """{family stem -> [card dict, ...]} for every registry YAML file."""
    fams = load_cards()
    assert fams, f"no card files found under {REGISTRY_DIR}"
    return fams


@pytest.fixture(scope="module")
def cards(families):
    """Flat {card id -> card dict}."""
    flat = {}
    for fam, cardlist in families.items():
        for c in cardlist:
            assert c["model"] not in flat, f"duplicate card id {c['model']}"
            flat[c["model"]] = c
    return flat


@pytest.fixture(scope="module")
def bib_pointers():
    """{bibkey -> True iff the entry has doi/eprint/adsurl} from references.bib."""
    text = (_REPO / "docs/website/references.bib").read_text()
    out = {}
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, re.S):
        out[m.group(1)] = bool(
            re.search(r"^\s*(doi|eprint|adsurl)\s*=", m.group(2), re.M)
        )
    return out


# ---------------------------------------------------------------------------
# 1. bibkeys resolve + carry a public pointer
# ---------------------------------------------------------------------------


def test_bibkeys_resolve_with_public_pointer(cards, bib_pointers):
    problems = []
    for cid, c in cards.items():
        for src in c.get("sources", []):
            key = src["bibkey"]
            if key not in bib_pointers:
                problems.append(f"{cid}: bibkey {key!r} not in references.bib")
            elif not bib_pointers[key]:
                problems.append(f"{cid}: bibkey {key!r} has no doi/eprint/adsurl")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# 2. code_refs import (module::qualname under src/progenax)
# ---------------------------------------------------------------------------


def _resolve_code_ref(ref: str):
    relpath, qualname = ref.split("::")
    module = "progenax." + relpath.removesuffix(".py").replace("/", ".")
    import importlib

    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def test_code_refs_import(cards):
    problems = []
    for cid, c in cards.items():
        for ref in c.get("code_refs", []):
            try:
                _resolve_code_ref(ref)
            except Exception as e:  # noqa: BLE001 — report, don't crash the sweep
                problems.append(f"{cid}: code_ref {ref!r} failed: {e!r}")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# 3. validation node ids resolve + their bodies assert (anti-theater)
# ---------------------------------------------------------------------------


def test_validation_node_ids_resolve_and_assert(cards):
    all_ids = sorted(
        {v for c in cards.values() for v in c.get("validation", [])}
    )
    resolved = resolve_node_ids(all_ids, rootdir=str(_REPO))
    missing = [i for i in all_ids if i not in resolved]
    assert not missing, f"validation node ids not collectable: {missing}"
    # Strip parametrize suffixes ("::test_x[case]") — the AST helper resolves the
    # FUNCTION; collection above already proved the specific parametrization exists.
    toothless = [i for i in all_ids if not _body_has_assert(i.split("[")[0])]
    assert not toothless, f"validation tests without an assert: {toothless}"


# ---------------------------------------------------------------------------
# 4. verified => >=1 code_ref AND >=1 validation
# ---------------------------------------------------------------------------


def test_verified_status_requires_code_and_validation(cards):
    problems = []
    for cid, c in cards.items():
        if c.get("status") == "verified":
            if not c.get("code_refs"):
                problems.append(f"{cid}: verified but no code_refs")
            if not c.get("validation"):
                problems.append(f"{cid}: verified but no validation entries")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# 5. coverage ratchet vs the physics registry
# ---------------------------------------------------------------------------


def test_coverage_ratchet(cards):
    from tests.validation.physics_registry.manifest import MODEL_INVARIANTS

    stale = [m for m in CARDED_MODELS if m not in MODEL_INVARIANTS]
    assert not stale, f"CARDED_MODELS entries not in MODEL_INVARIANTS: {stale}"
    uncarded = [
        (m, cid) for m, cid in CARDED_MODELS.items() if cid not in cards
    ]
    assert not uncarded, f"manifest entries without a card: {uncarded}"
    if REGISTRY_FULL:
        gap = sorted(set(MODEL_INVARIANTS) - set(CARDED_MODELS))
        assert not gap, f"REGISTRY_FULL=True but models lack cards: {gap}"


# ---------------------------------------------------------------------------
# 6. glossary freshness (committed pages == fresh regeneration)
# ---------------------------------------------------------------------------


def test_glossary_pages_fresh(families):
    problems = []
    for fam, cardlist in families.items():
        page = GLOSSARY_DIR / f"{fam}.md"
        if not page.exists():
            problems.append(f"missing committed glossary page: {page}")
            continue
        fresh = render_family(fam, cardlist)
        if page.read_text() != fresh:
            problems.append(
                f"stale glossary page {page} — regenerate with "
                f"`python scripts/build_provenance_registry.py --emit` and commit"
            )
    assert not problems, "\n".join(problems)
