"""Auto-generate the progenax API reference for the MyST-MD website.

Run from the website root:

    python scripts/build_api_reference.py

For each public submodule of progenax (per ``progenax.__all__``), this
script walks the module via :mod:`inspect`, extracts signatures and
docstrings, and emits one MyST page per top-level module under
``30-api/`` plus an alphabetical ``30-api/full-symbol-index.md``.

Pages contain module-namespaced anchors so theory chapters can
cross-link via ``[](../30-api/profiles.md#api-profiles-plummerprofile)``.

The script uses only Python stdlib — no Griffe dependency. Sufficient
for progenax's docstring style (numpy-style with parameter descriptions
and references blocks).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re as _re
import sys
from pathlib import Path

# Modules whose public symbols feed the API reference.
# Order: alphabetical by module name (browse-by-module navigation).
PUBLIC_MODULES: list[str] = [
    "progenax.analytical",
    "progenax.binaries",
    "progenax.builders",
    "progenax.builders_cluster",
    "progenax.cluster",
    "progenax.diagnostics",
    "progenax.dynamics",
    "progenax.imf",
    "progenax.kinematics",
    "progenax.profiles",
    "progenax.protocols",
    "progenax.tidal",
]

# Output directory (relative to script's invocation cwd; usually website root)
API_DIR = Path("30-api")

# GitHub blob URL prefix for source-link generation
GITHUB_BLOB = "https://github.com/jaxstro/progenax/blob/main"


# ---------------------------------------------------------------------------
# Registry cross-links (ADR-0034 provenance cards + the grad-audit gate).
# Loaded once; failures degrade to no-badges (the API reference must build
# even if a registry file moves).
# ---------------------------------------------------------------------------


def _load_card_index() -> dict[str, tuple[str, str]]:
    """{qualname -> (family_stem, model)} from docs/provenance/registry/*.yaml."""
    index: dict[str, tuple[str, str]] = {}
    reg = Path(__file__).resolve().parents[3] / "docs" / "provenance" / "registry"
    try:
        import yaml

        for path in sorted(reg.glob("*.yaml")):
            for card in yaml.safe_load(path.read_text()) or []:
                for ref in card.get("code_refs", []):
                    qual = ref.split("::")[-1].split(".")[0]
                    index.setdefault(qual, (path.stem, card["model"]))
    except Exception:
        return {}
    return index


def _load_grad_audit_counts() -> dict[str, int]:
    """{symbol -> number of grad-audit registry cases} from the committed JSON."""
    counts: dict[str, int] = {}
    path = (
        Path(__file__).resolve().parents[3]
        / "validation" / "data" / "grad_audit_results.json"
    )
    try:
        rows = json.loads(path.read_text())
    except Exception:
        return {}
    for row in rows:
        sym = _re.split(r"[.\[]", row.get("id", ""))[0]
        if sym:
            counts[sym] = counts.get(sym, 0) + 1
    return counts


_CARD_INDEX = _load_card_index()
_GRAD_COUNTS = _load_grad_audit_counts()


def _badges(name: str) -> str:
    """One markdown line of registry badges for a symbol, or empty."""
    bits = []
    if name in _CARD_INDEX:
        fam, model = _CARD_INDEX[name]
        bits.append(
            f"[📇 model card](../15-model-reference/{fam}.md#card-{model})"
        )
    if name in _GRAD_COUNTS:
        n = _GRAD_COUNTS[name]
        bits.append(
            f"[∇ gradient-verified — {n} audit case{'s' if n > 1 else ''}]"
            "(../50-validation/differentiability-audit.md)"
        )
    return " · ".join(bits)


def _is_public(name: str) -> bool:
    """A name is public iff it does not start with underscore."""
    return not name.startswith("_")


def _format_signature(obj) -> str:
    """Return the signature string for a callable/class, or an empty string.

    Collapses the jaxtyping/ArrayLike union that inspect stringifies as
    ``Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, bool, int,
    float, complex]`` (x5 parameters on some constructors) into the readable
    alias ``ArrayLike`` (docs audit, Medium: unreadable generated signatures).
    """
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return ""
    text = str(sig)
    for union, alias in (
        ("Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, bool, int, float, complex, NoneType]", "ArrayLike | None"),
        ("Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, bool, int, float, complex]", "ArrayLike"),
        ("Union[Array, ndarray, bool, number, bool, int, float, complex]", "ArrayLike"),
        ("jax.Array | numpy.ndarray | numpy.bool | numpy.number | bool | int | float | complex", "ArrayLike"),
        # the jaxtyping PRNG-key union (Key[''] | UInt32[2] | UInt32[4])
        ("Union[jaxtyping.Key[Array, ''], jaxtyping.UInt32[Array, '2'], jaxtyping.UInt32[Array, '4']]", "PRNGKeyArray"),
    ):
        text = text.replace(union, alias)
    # Shorten the module-qualified jaxtyping shape annotations for readability.
    text = text.replace("jaxtyping.", "")
    return text


_SECTION_RE = None  # compiled lazily


def _format_docstring(obj) -> str:
    """Render a Google-style docstring as structured markdown.

    ``Args:``/``Attributes:`` blocks become parameter tables; ``Returns:``/
    ``Raises:`` become bolded one-liners; other recognized sections become
    bold-titled paragraphs. Anything unparseable falls back to the raw
    docstring (the reference must never lose content to the formatter).
    """
    import re

    doc = inspect.getdoc(obj)
    if not doc:
        return "*(no docstring)*"
    try:
        section_re = re.compile(
            r"^(Args|Arguments|Attributes|Returns|Yields|Raises|Note|Notes|"
            r"References|Example|Examples|Warning)\s*:\s*$",
            re.M,
        )
        parts: list[tuple[str, str]] = []
        last, last_name = 0, ""
        for m in section_re.finditer(doc):
            parts.append((last_name, doc[last : m.start()].rstrip()))
            last_name, last = m.group(1), m.end()
        parts.append((last_name, doc[last:].rstrip()))

        out: list[str] = []
        for name, body in parts:
            if not body.strip():
                continue
            if name in ("Args", "Arguments", "Attributes"):
                rows = []
                current = None
                for line in body.splitlines():
                    m = re.match(r"^\s{2,}(\*{0,2}\w+)\s*:\s*(.*)$", line)
                    if m and not line.startswith(" " * 12):
                        current = [m.group(1), m.group(2).strip()]
                        rows.append(current)
                    elif current is not None and line.strip():
                        current[1] += " " + line.strip()
                if not rows:
                    raise ValueError("unparsed args block")
                out.append(f"**{name}**")
                out.append("")
                out.append("| Parameter | Description |")
                out.append("|---|---|")
                for pname, desc in rows:
                    desc = desc.replace("|", "\\|")
                    out.append(f"| `{pname}` | {desc} |")
                out.append("")
            elif name in ("Returns", "Yields", "Raises"):
                text = " ".join(x.strip() for x in body.splitlines() if x.strip())
                out.append(f"**{name}:** {text}")
                out.append("")
            elif name:
                out.append(f"**{name}.** " + "\n".join(
                    x.strip() for x in body.splitlines()
                ).strip())
                out.append("")
            else:
                out.append(body)
                out.append("")
        return "\n".join(out).rstrip()
    except Exception:
        return doc


def _source_link(obj, package_root: Path) -> str | None:
    """Return a relative source-file link for `obj`, or None if not findable."""
    try:
        path = inspect.getsourcefile(obj)
        line = inspect.getsourcelines(obj)[1]
    except (TypeError, OSError):
        return None
    if path is None:
        return None
    p = Path(path)
    try:
        rel = p.relative_to(package_root)
    except ValueError:
        # Object not in our source tree
        return None
    # package_root is <repo>/src (src-layout), but the GitHub blob URL needs the
    # REPO-relative path — without the "src/" prefix every source link 404s
    # (docs audit, Medium). Emit src/<rel> so the displayed path is honest too.
    return f"src/{rel.as_posix()}#L{line}"


def _anchor(short_module: str, symbol: str) -> str:
    """Return a stable anchor that is unique across the whole MyST site."""
    return f"api-{short_module.replace('.', '-')}-{symbol.lower()}"


def _collect_public_symbols(module_name: str):
    """Yield (name, obj) for each public symbol in `module.__all__`."""
    mod = importlib.import_module(module_name)
    public = getattr(mod, "__all__", None)
    if public is None:
        # Fall back to non-underscore attributes
        public = [n for n in dir(mod) if _is_public(n)]
    for name in public:
        try:
            obj = getattr(mod, name)
        except AttributeError:
            continue
        yield name, obj


def _classify(obj) -> str:
    """Return 'class', 'function', 'protocol', or 'value'."""
    if inspect.isclass(obj):
        # Check if it's a Protocol (typing.Protocol subclass)
        try:
            if hasattr(obj, "_is_protocol") and obj._is_protocol:
                return "protocol"
        except Exception:
            pass
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj) or callable(obj):
        return "function"
    return "value"


def _emit_module_page(module_name: str, out_path: Path, package_root: Path) -> int:
    """Write the MyST page for a single module. Returns symbol count."""
    short = module_name.removeprefix("progenax.")
    symbols = list(_collect_public_symbols(module_name))

    lines: list[str] = []
    lines.append("---")
    lines.append(f"title: '`{module_name}`'")
    lines.append(
        f"description: Auto-generated API reference for `{module_name}` — "
        f"signatures and docstrings of every public symbol."
    )
    lines.append("---")
    lines.append("")
    lines.append(f"# `{module_name}`")
    lines.append("")
    lines.append(
        "*Auto-generated by `scripts/build_api_reference.py` from progenax "
        "source. Re-run after public-API changes; the script is idempotent.*"
    )
    lines.append("")
    lines.append(f"Module path: `progenax/{short.replace('.', '/')}/`")
    lines.append("")
    lines.append(f"Public symbols: **{len(symbols)}**")
    lines.append("")

    if not symbols:
        lines.append("*No public symbols found.*")
        out_path.write_text("\n".join(lines) + "\n")
        return 0

    # Table of contents within the page
    lines.append("## Contents")
    lines.append("")
    for name, _obj in symbols:
        lines.append(f"- [`{name}`](#{_anchor(short, name)})")
    lines.append("")

    # Per-symbol section
    for name, obj in symbols:
        kind = _classify(obj)
        sig = _format_signature(obj)
        doc = _format_docstring(obj)
        src = _source_link(obj, package_root)

        lines.append(f"({_anchor(short, name)})=")
        lines.append(f"## `{short}.{name}`")
        lines.append("")

        kind_label = {
            "class": "*class*",
            "protocol": "*protocol*",
            "function": "*function*",
            "value": "*value*",
        }[kind]
        lines.append(kind_label)
        badge_line = _badges(name)
        if badge_line:
            lines.append("")
            lines.append(badge_line)
        lines.append("")

        if sig:
            lines.append("```python")
            lines.append(f"{name}{sig}")
            lines.append("```")
            lines.append("")

        # Docstring as block-quote-ish; keep raw markdown in MyST
        # (MyST renders multi-line docstrings naturally).
        lines.append(doc)
        lines.append("")

        if src:
            lines.append(
                f"*Source: [`{src}`]({GITHUB_BLOB}/{src})*"
            )
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    return len(symbols)


def _emit_full_symbol_index(
    all_symbols: list[tuple[str, str, str]],
    out_path: Path,
) -> None:
    """Emit the alphabetical full-symbol index.

    Parameters
    ----------
    all_symbols
        List of (symbol_name, kind, owning_module_short_name) tuples.
    """
    sorted_syms = sorted(all_symbols, key=lambda t: t[0].lower())

    lines: list[str] = []
    lines.append("---")
    lines.append("title: Full symbol index")
    lines.append(
        "description: Alphabetical index of every public progenax symbol "
        "with classification and link to the per-module page."
    )
    lines.append("---")
    lines.append("")
    lines.append("# Full symbol index")
    lines.append("")
    lines.append(
        "*Auto-generated alphabetical index of every public progenax symbol. "
        "Click the symbol name to jump to its per-module page entry.*"
    )
    lines.append("")
    lines.append(f"Total public symbols: **{len(sorted_syms)}** across all submodules.")
    lines.append("")
    lines.append(
        "> This counts every symbol in each submodule's `__all__`. The top-level "
        "`progenax` package re-exports a curated subset for convenience "
        "(e.g. `from progenax import PlummerProfile`); import the submodule "
        "(e.g. `from progenax.imf import ...`) to reach the rest."
    )
    lines.append("")
    lines.append("| Symbol | Kind | Module |")
    lines.append("|--------|------|--------|")

    for name, kind, mod_short in sorted_syms:
        page = f"{mod_short}.md"
        lines.append(
            f"| [`{name}`]({page}#{_anchor(mod_short, name)}) | {kind} | `progenax.{mod_short}` |"
        )

    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=API_DIR,
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=None,
        help="Root of the progenax source package (auto-detected if omitted)",
    )
    args = parser.parse_args(argv)

    # Auto-detect package root by importing progenax and getting its file path
    if args.package_root is None:
        progenax = importlib.import_module("progenax")
        args.package_root = Path(progenax.__file__).resolve().parent.parent

    args.out.mkdir(parents=True, exist_ok=True)

    all_symbols: list[tuple[str, str, str]] = []
    total = 0

    for module in PUBLIC_MODULES:
        short = module.removeprefix("progenax.")
        page_name = short.replace(".", "-") + ".md"
        try:
            n = _emit_module_page(module, args.out / page_name, args.package_root)
        except Exception as exc:
            print(f"WARNING: {module} failed: {exc}", file=sys.stderr)
            n = 0
            continue
        total += n
        # Re-collect for the index
        for name, obj in _collect_public_symbols(module):
            all_symbols.append((name, _classify(obj), short))

    _emit_full_symbol_index(all_symbols, args.out / "full-symbol-index.md")

    print(
        f"Wrote {len(PUBLIC_MODULES)} module pages + 1 index to {args.out}/"
        f" ({total} public symbols total)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
