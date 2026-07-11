"""Build the provenance model-card glossary (ADR-0034).

Reads the canonical model cards from ``docs/provenance/registry/*.yaml`` and
generates one MyST glossary page per family under
``docs/website/15-model-reference/`` (plus the section index). The pages are
GENERATED — never hand-edit them; the enforcement registry
(``tests/validation/provenance_cards/``) asserts committed == regenerated.

Usage:
    python scripts/build_provenance_registry.py --emit    # write the pages
    python scripts/build_provenance_registry.py --check   # diff vs committed (CI/gate)
    python scripts/build_provenance_registry.py --digests # emit Brain digest drafts
                                                          # to .brain-drafts/ (gitignored)

Idempotent and pure (no timestamps/randomness): same YAML -> same bytes, the
same idiom as build_test_dashboard.py / build_api_reference.py. Plain script,
not core library code (the JAX-native constraint does not apply here).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = _REPO_ROOT / "docs" / "provenance" / "registry"
GLOSSARY_DIR = _REPO_ROOT / "docs" / "website" / "15-model-reference"
DIGEST_DIR = _REPO_ROOT / ".brain-drafts" / "equation-digests"
_GITHUB_BLOB = "https://github.com/jaxstro/progenax/blob/main/src/progenax"

# Family stem -> display title for the generated page (ordering = site order).
FAMILY_TITLES: dict[str, str] = {
    "spatial_profiles": "Spatial profiles",
    "velocity_dfs": "Velocity distribution functions",
    "imfs": "Initial mass functions",
    "binaries": "Binary populations & orbits",
    "populations": "Multi-component cluster engines",
    "tidal_diagnostics": "Tidal physics & diagnostics",
    "builders_stellar": "Cluster builders & stellar relations",
}

_STATUS_BADGE = {
    "verified": "✅ verified",
    "needs-check": "🔶 needs-check",
    "unverifiable-scanned": "🔸 unverifiable (scanned source)",
}


def load_cards() -> dict[str, list[dict]]:
    """{family stem -> [card, ...]} for every YAML file in the registry dir."""
    families: dict[str, list[dict]] = {}
    for path in sorted(REGISTRY_DIR.glob("*.yaml")):
        cards = yaml.safe_load(path.read_text())
        if not isinstance(cards, list):
            raise ValueError(f"{path}: expected a top-level LIST of cards")
        for c in cards:
            _validate_card(c, path)
        families[path.stem] = cards
    return families


_REQUIRED_FIELDS = (
    "model",
    "description",
    "when_to_use",
    "parameters",
    "sources",
    "equations",
    "code_refs",
    "validation",
    "status",
    "deviations",
)


def _validate_card(card: dict, path: Path) -> None:
    missing = [f for f in _REQUIRED_FIELDS if f not in card]
    if missing:
        raise ValueError(f"{path}: card {card.get('model', '?')} missing {missing}")
    if card["status"] not in _STATUS_BADGE:
        raise ValueError(
            f"{path}: card {card['model']} has unknown status {card['status']!r}"
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_card(card: dict) -> str:
    lines: list[str] = []
    title = card.get("family_display", card["model"])
    lines.append(f"(card-{card['model']})=")
    lines.append(f"## {title}")
    lines.append("")
    # Compact meta row: status · counts · API backlink (from the first code_ref).
    meta = [_STATUS_BADGE[card["status"]]]
    n_eq = len(card["equations"])
    n_src = len(card["sources"])
    meta.append(f"{n_eq} equation{'s' if n_eq != 1 else ''}")
    meta.append(f"{n_src} source{'s' if n_src != 1 else ''}")
    if card["code_refs"]:
        relpath, qualname = card["code_refs"][0].split("::")
        module_short = relpath.split("/")[0].removesuffix(".py")
        # module aliases: builders_cluster documents under builders
        module_short = {"builders_cluster": "builders"}.get(module_short, module_short)
        symbol = qualname.split(".")[0]
        api_page = _REPO_ROOT / "docs" / "website" / "30-api" / f"{module_short}.md"
        if api_page.exists():  # e.g. stellar.py has no API page yet
            meta.append(
                f"[API: `{symbol}`](../30-api/{module_short}.md"
                f"#api-{module_short}-{symbol.lower()})"
            )
    lines.append(" · ".join(meta))
    lines.append("")
    lines.append(" ".join(card["description"].split()))
    lines.append("")

    # When-to-use table
    wtu = card["when_to_use"]
    lines.append("| Use it for | Not for |")
    lines.append("|---|---|")
    good, bad = wtu.get("good_for", []), wtu.get("not_for", [])
    for i in range(max(len(good), len(bad))):
        g = f"✓ {good[i]}" if i < len(good) else ""
        b = f"✗ {bad[i]}" if i < len(bad) else ""
        lines.append(f"| {g} | {b} |")
    lines.append("")

    # Parameters
    lines.append("### Parameters")
    lines.append("")
    lines.append("| Name | Meaning | Units | Typical range | Code |")
    lines.append("|---|---|---|---|---|")
    for p in card["parameters"]:
        lines.append(
            f"| `{p['name']}` | {p['meaning']} | {p.get('units', '—')} | "
            f"{p.get('typical_range', '—')} | `{p['code_arg']}` |"
        )
    lines.append("")

    # Equations
    lines.append("### Equations")
    lines.append("")
    for eq in card["equations"]:
        lines.append("```{math}")
        lines.append(f":label: card-{eq['label']}")
        lines.append(eq["latex"])
        lines.append("```")
        lines.append("")
        sym_bits = [
            (
                f"${info['latex']}$" if "latex" in info else f"`{name}`"
            )
            + f": {info['meaning']} [{info.get('units', '—')}]"
            for name, info in eq.get("symbols", {}).items()
        ]
        if sym_bits:
            lines.append("*Symbols:* " + "; ".join(sym_bits) + ".")
        if eq.get("assumptions"):
            lines.append("*Assumes:* " + "; ".join(eq["assumptions"]) + ".")
        if eq.get("theory_ref"):
            page, _, anchor = eq["theory_ref"].partition("#")
            # glossary pages live one level below the site root; theory_ref is root-relative
            rel = "../" + page
            lines.append(f"*Derivation:* [theory page]({rel}#{anchor}).")
        lines.append("")

    # Sources
    lines.append("### Sources")
    lines.append("")
    for s in card["sources"]:
        loc = "; ".join(s.get("locators", []))
        lines.append(
            f"- {{cite:t}}`{s['bibkey']}` — {s['provides']}"
            + (f" *({loc})*" if loc else "")
        )
    lines.append("")

    # Code + validation
    lines.append("### Code & validation")
    lines.append("")
    for ref in card["code_refs"]:
        relpath, qualname = ref.split("::")
        lines.append(f"- code: [`{qualname}`]({_GITHUB_BLOB}/{relpath}) (`src/progenax/{relpath}`)")
    for v in card["validation"]:
        lines.append(f"- validation: `{v}`")
    lines.append("")

    # Deviations
    if card["deviations"]:
        lines.append(":::{admonition} Deviations from the source")
        lines.append(":class: caution")
        for d in card["deviations"]:
            lines.append(f"- {d}")
        lines.append(":::")
        lines.append("")

    return "\n".join(lines)


_HEADER = """\
---
title: "{title}"
description: "GENERATED provenance glossary — model cards for the {title_lower} family: sources with DOIs, equations, parameters, code and validation cross-references."
---

<!-- GENERATED by scripts/build_provenance_registry.py from
     docs/provenance/registry/{stem}.yaml — DO NOT hand-edit.
     Regenerate:  python scripts/build_provenance_registry.py --emit  -->

# {title} — model cards

Machine-generated from the provenance registry (ADR-0034): every model's
sources (with public DOI/arXiv/ADS pointers), governing equations, parameter
meanings, code entry points, and the validation tests that pin them. The
hand-authored theory pages hold the derivations; these cards are the citable
single source of truth.

"""


def render_family(stem: str, cards: list[dict]) -> str:
    title = FAMILY_TITLES.get(stem, stem.replace("_", " ").title())
    out = _HEADER.format(title=title, title_lower=title.lower(), stem=stem)
    out += "\n\n".join(_render_card(c) for c in cards)
    out += "\n"
    return out


def render_index(families: dict[str, list[dict]]) -> str:
    lines = [
        "---",
        'title: "Model reference"',
        'description: "GENERATED provenance glossary index — one model-card page per model family (ADR-0034)."',
        "---",
        "",
        "<!-- GENERATED by scripts/build_provenance_registry.py — DO NOT hand-edit. -->",
        "",
        "# Model reference (provenance glossary)",
        "",
        "One generated page per model family. Each card: description, when to use,",
        "parameters, governing equations, sources (DOI/arXiv/ADS), code entry points,",
        "and the validation tests that pin the physics.",
        "",
    ]
    for stem, cards in families.items():
        title = FAMILY_TITLES.get(stem, stem)
        models = ", ".join(c.get("family_display", c["model"]) for c in cards)
        lines.append(f"- [{title}]({stem}.md) — {models}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Brain equation-digest drafts (design §7) — staging only; Anna commits to ~/brain
# ---------------------------------------------------------------------------


def render_digest(card: dict) -> str:
    lines = [
        f"# Equation digest — {card.get('family_display', card['model'])}",
        "",
        f"*Draft generated from progenax `docs/provenance/registry` (ADR-0034); "
        f"card `{card['model']}`, status {card['status']}.*",
        "",
        "## Sources",
        "",
    ]
    for s in card["sources"]:
        loc = "; ".join(s.get("locators", []))
        lines.append(f"- `{s['bibkey']}` — {s['provides']}" + (f" ({loc})" if loc else ""))
    lines += ["", "## Equations", ""]
    for eq in card["equations"]:
        lines.append(f"### {eq['label']}")
        lines.append("")
        lines.append("$$" + eq["latex"] + "$$")
        lines.append("")
        for name, info in eq.get("symbols", {}).items():
            lines.append(f"- ${name}$: {info['meaning']} [{info.get('units', '—')}]")
        if eq.get("assumptions"):
            lines.append(f"- assumes: {'; '.join(eq['assumptions'])}")
        lines.append("")
    lines += ["## Verification", ""]
    for v in card["validation"]:
        lines.append(f"- `{v}`")
    if card["deviations"]:
        lines += ["", "## Deviations", ""]
        for d in card["deviations"]:
            lines.append(f"- {d}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", action="store_true", help="write the glossary pages")
    ap.add_argument("--check", action="store_true", help="fail if committed pages are stale")
    ap.add_argument("--digests", action="store_true", help="emit Brain digest drafts to .brain-drafts/")
    args = ap.parse_args(argv)

    families = load_cards()
    rendered = {f"{stem}.md": render_family(stem, cards) for stem, cards in families.items()}
    rendered["index.md"] = render_index(families)

    if args.emit:
        GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
        for name, text in rendered.items():
            (GLOSSARY_DIR / name).write_text(text)
            print(f"wrote {GLOSSARY_DIR / name}")

    if args.check:
        stale = [
            name
            for name, text in rendered.items()
            if not (GLOSSARY_DIR / name).exists()
            or (GLOSSARY_DIR / name).read_text() != text
        ]
        if stale:
            print(f"STALE glossary pages (run --emit and commit): {stale}", file=sys.stderr)
            return 1
        print("glossary fresh")

    if args.digests:
        DIGEST_DIR.mkdir(parents=True, exist_ok=True)
        for cards in families.values():
            for c in cards:
                out = DIGEST_DIR / f"{c['model'].replace('_', '-')}.md"
                out.write_text(render_digest(c))
                print(f"wrote {out}")

    if not (args.emit or args.check or args.digests):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
