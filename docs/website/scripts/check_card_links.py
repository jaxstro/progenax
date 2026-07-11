#!/usr/bin/env python3
"""Post-build gate: every card-links-plugin href resolves in the built site.

The card-links plugin (plugins/card-links.mjs) rewrites [↗ model card](#card-…)
cross-references into raw <a target="_blank"> anchors, computing each URL as
BASE + /<html-normalized glossary filename stem>#<anchor>. That computation
assumes (a) MyST's root-flat slug scheme and (b) that the glossary page WINS any
slug-dedup contest (e.g. glossary binaries.md vs API binaries.md both want
/binaries; today the loser gets /binaries-1 — and the winner is build-order-
dependent). Neither assumption is enforced by `myst build`, so a violation
ships as a silent 404.

This check closes that hole: it extracts every plugin-emitted href from the
built page JSONs and asserts the (page-url, fragment) pair exists in the
build's own myst.xref.json. Run from docs/website/ AFTER `myst build --html`
(check_docs.sh step 3). Exits 1 on any unresolvable link.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

HTML_DIR = Path("_build/html")
# Matches the plugin's emitted anchor; the title attribute is its signature.
# Page JSONs store the html node's markup as a JSON string, so every quote
# appears backslash-escaped (href=\"...\") — hence the \\?" tolerance.
HREF_RE = re.compile(
    r'<a href=\\?"([^"\\#]*)#([^"\\]+)\\?" target=\\?"_blank\\?" '
    r'rel=\\?"noopener\\?" title=\\?"opens the model-card glossary'
)


def main() -> int:
    xref_path = HTML_DIR / "myst.xref.json"
    if not xref_path.exists():
        print(f"  card-link gate: SKIP — {xref_path} not found (build first)")
        return 1

    base = os.environ.get("BASE_URL", "").rstrip("/")
    refs = json.loads(xref_path.read_text())["references"]
    page_urls = {r["url"] for r in refs if r.get("kind") == "page"}
    anchors = {
        (r["url"], r.get("html_id") or r["identifier"])
        for r in refs
        if r.get("identifier")
    }

    found: set[tuple[str, str]] = set()
    for page_json in HTML_DIR.glob("*.json"):
        if page_json.name.startswith("myst."):
            continue
        for url, frag in HREF_RE.findall(page_json.read_text()):
            found.add((url, frag))

    if not found:
        print("  card-link gate: FAIL — no plugin-emitted links found at all")
        print("    (plugin not loaded? check myst.yml project.plugins)")
        return 1

    bad = []
    for url, frag in sorted(found):
        path = url[len(base):] if base and url.startswith(base) else url
        if path not in page_urls:
            bad.append(f"{url}#{frag}  (no such page: {path})")
        elif (path, frag) not in anchors:
            bad.append(f"{url}#{frag}  (page exists, fragment does not)")

    if bad:
        print(f"  card-link gate: FAIL — {len(bad)} unresolvable card link(s):")
        for b in bad:
            print(f"    {b}")
        return 1

    n_pages = len({u for u, _ in found})
    print(
        f"  card-link gate: PASS — {len(found)} distinct card links "
        f"across {n_pages} glossary URLs, all resolve in myst.xref.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
