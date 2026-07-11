"""Link-integrity + count/hygiene drift gate for the progenax MyST docs site.

Why this exists
---------------
``myst build`` does **not** validate relative ``.md`` link targets. A site can
build cleanly (exit 0, zero content warnings) and still ship Markdown links that
point at files which do not exist — e.g. ``[bib](../99-bibliography/index.md)``
when only ``bibliography.md`` exists. mystmd resolves *cross-references*
(``(label)=`` + ``[](#label)``) and *citations* (``[@key]``), but a bare
relative-path link to a sibling ``.md`` file is passed through untouched. This
script is the missing gate: it walks the docs tree, resolves every ``.md`` link
relative to its containing file, and fails (nonzero exit) if any target is
missing.

It also surfaces two *informational* classes of doc drift that do not fail the
build (so this stays specifically a **link gate**):

  (b) hardcoded test-count phrasings in prose (``1234 tests``,
      ``956 unit tests``, …) — these drift as the suite grows and should point
      at the generated test dashboard / "see CI for the live count".
  (c) hygiene strings that leak local/internal paths into a public site
      (``/Users/``, ``.claude-work``, ``docs/core-papers/``, ``docs/plans/``,
      ``docs/notes/``).

Usage
-----
Run from the website root (``docs/website``):

    python scripts/check_links_and_counts.py                  # scan all pages
    python scripts/check_links_and_counts.py path/to/page.md  # scan a subset

With no file arguments it scans ``**/*.md`` under the website root, excluding
the ``_build/`` output directory. With arguments it scans exactly those files
(paths relative to CWD or absolute).

Exit code
---------
* nonzero  -> at least one broken ``.md`` link was found
* zero     -> no broken ``.md`` links (count/hygiene findings do NOT affect it)

Stdlib only — no third-party dependencies.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Website root = parent of this script's directory (docs/website/scripts/..).
WEBSITE_ROOT = Path(__file__).resolve().parent.parent

# Directories never scanned (build output is generated, not source).
EXCLUDE_DIRS = {"_build", "node_modules", "__pycache__"}

# (a) Markdown links whose target is a .md file (optionally with #anchor).
#     Matches ](target.md) and ](target.md#anchor). Captures the path only.
#     http(s):// links are filtered out after matching.
MD_LINK_RE = re.compile(r"\]\(\s*([^)\s]+?\.md)(?:#[^)\s]*)?\s*\)")

# (b) Hardcoded test-count phrasings in prose. Two complementary patterns:
#     - "N tests" / "N released-core unit tests" / "N validation tests" etc.
#     - a looser "NNN tests" catch for 3-4 digit bare counts.
COUNT_RES = [
    re.compile(
        r"\b\d{2,4}\s+(?:released-core\s+)?"
        r"(?:unit|integration|validation|experimental|physics-validation)?\s*tests\b"
    ),
    re.compile(r"\b\d{3,4}\s+tests\b"),
]

# (c) Hygiene strings that should not appear in published pages.
HYGIENE_STRINGS = [
    "/Users/",
    ".claude-work",
    "docs/core-papers/",
    "docs/plans/",
    "docs/notes/",
]


# ---------------------------------------------------------------------------
# Finding records
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single location-tagged result, rendered as ``file:line``."""

    path: Path
    line: int
    detail: str

    def location(self) -> str:
        try:
            rel = self.path.relative_to(WEBSITE_ROOT)
        except ValueError:
            rel = self.path
        return f"{rel}:{self.line}"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_markdown_files(args: list[str]) -> list[Path]:
    """Return the list of .md files to scan.

    With no args: every ``*.md`` under WEBSITE_ROOT excluding EXCLUDE_DIRS.
    With args: exactly those paths (resolved), keeping only existing .md files.
    """
    if args:
        files: list[Path] = []
        for arg in args:
            p = Path(arg)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.suffix == ".md" and p.is_file():
                files.append(p)
            else:
                print(f"  (skipping non-existent / non-.md argument: {arg})")
        return files

    files = []
    for p in sorted(WEBSITE_ROOT.rglob("*.md")):
        if any(part in EXCLUDE_DIRS for part in p.relative_to(WEBSITE_ROOT).parts):
            continue
        files.append(p)
    return files


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------


def scan_broken_links(path: Path, lines: list[str]) -> list[Finding]:
    """(a) Report every relative .md link whose target file does not exist."""
    findings: list[Finding] = []
    base_dir = path.parent
    for lineno, line in enumerate(lines, start=1):
        for match in MD_LINK_RE.finditer(line):
            target = match.group(1)
            # Skip external links.
            if target.startswith(("http://", "https://", "//", "mailto:")):
                continue
            resolved = (base_dir / target).resolve()
            if not resolved.is_file():
                findings.append(
                    Finding(path, lineno, f"-> {target} (resolved: missing)")
                )
    return findings


def load_toc_files() -> set[str]:
    """Website-root-relative paths of every ``file:`` entry in myst.yml.

    mystmd builds ONLY toc-listed pages, so a relative link whose target exists
    on disk but is absent from the toc still 404s on the built site (the
    ``interface-with-gravax`` half-state, audit F3). Stdlib-only parse: the toc
    is a simple ``- file: <path>`` list (``hidden: true`` entries count as
    listed — they are built and URL-reachable, just out of the nav).
    """
    myst = WEBSITE_ROOT / "myst.yml"
    entries = re.findall(r"^\s*-\s*file:\s*(\S+)", myst.read_text(), re.M)
    return set(entries)


def scan_toc_membership(
    path: Path, lines: list[str], toc: set[str]
) -> list[Finding]:
    """(d) GATE: every resolving .md link must point at a toc-listed page."""
    findings: list[Finding] = []
    base_dir = path.parent
    for lineno, line in enumerate(lines, start=1):
        for match in MD_LINK_RE.finditer(line):
            target = match.group(1)
            if target.startswith(("http://", "https://", "//", "mailto:")):
                continue
            resolved = (base_dir / target).resolve()
            if not resolved.is_file():
                continue  # scan_broken_links owns the missing-file class
            try:
                rel = resolved.relative_to(WEBSITE_ROOT).as_posix()
            except ValueError:
                continue  # outside the site tree (e.g. repo README links)
            if rel not in toc:
                findings.append(
                    Finding(path, lineno, f"-> {target} (exists but NOT in the myst.yml toc — unbuilt on the site)")
                )
    return findings


def scan_count_drift(path: Path, lines: list[str]) -> list[Finding]:
    """(b) Flag hardcoded test-count phrasings (informational)."""
    findings: list[Finding] = []
    for lineno, line in enumerate(lines, start=1):
        seen: set[str] = set()
        for regex in COUNT_RES:
            for match in regex.finditer(line):
                text = match.group(0).strip()
                if text not in seen:
                    seen.add(text)
                    findings.append(Finding(path, lineno, repr(text)))
    return findings


def scan_hygiene(path: Path, lines: list[str]) -> list[Finding]:
    """(c) Flag local/internal path-leak strings (informational)."""
    findings: list[Finding] = []
    for lineno, line in enumerate(lines, start=1):
        for needle in HYGIENE_STRINGS:
            if needle in line:
                findings.append(Finding(path, lineno, repr(needle)))
    return findings


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_section(title: str, findings: list[Finding]) -> None:
    print(f"\n{title} ({len(findings)})")
    print("-" * len(f"{title} ({len(findings)})"))
    if not findings:
        print("  (none)")
        return
    for f in findings:
        print(f"  {f.location()}  {f.detail}")


def main(argv: list[str]) -> int:
    files = discover_markdown_files(argv)
    if not files:
        print("No markdown files to scan.")
        return 0

    broken: list[Finding] = []
    untoccd: list[Finding] = []
    counts: list[Finding] = []
    hygiene: list[Finding] = []
    toc = load_toc_files()

    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover
            print(f"  (could not read {path}: {exc})")
            continue
        broken.extend(scan_broken_links(path, lines))
        untoccd.extend(scan_toc_membership(path, lines, toc))
        counts.extend(scan_count_drift(path, lines))
        hygiene.extend(scan_hygiene(path, lines))

    print(f"Scanned {len(files)} markdown file(s) under {WEBSITE_ROOT}")

    print_section("BROKEN .md LINKS (gate: nonzero exit if any)", broken)
    print_section("LINKS TO NON-TOC PAGES (gate: nonzero exit if any)", untoccd)
    print_section("HARDCODED TEST-COUNT PHRASINGS (informational)", counts)
    print_section("HYGIENE / PATH-LEAK STRINGS (informational)", hygiene)

    print("\nSUMMARY")
    print("-------")
    print(f"  broken .md links : {len(broken)}  <- gate")
    print(f"  non-toc targets  : {len(untoccd)}  <- gate")
    print(f"  count phrasings  : {len(counts)}  (informational)")
    print(f"  hygiene strings  : {len(hygiene)}  (informational)")

    if broken or untoccd:
        print(f"\nFAIL: {len(broken)} broken + {len(untoccd)} non-toc .md link(s).")
        return 1
    print("\nPASS: no broken or non-toc .md links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
