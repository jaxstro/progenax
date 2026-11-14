"""Auto-generate the progenax API reference for the MyST-MD website.

Run from the website root:

    python scripts/build_api_reference.py

For each supported public submodule of progenax (per its ``__all__``), this
script walks the module via :mod:`inspect`, extracts signatures and
docstrings, and emits one MyST page per top-level module under
``30-api/`` plus an alphabetical ``30-api/full-symbol-index.md``.

Pages contain module-namespaced anchors so theory chapters can
cross-link via ``[](../30-api/profiles.md#api-profiles-plummerprofile)``.

Introspection and markup conversion use Python stdlib. Optional registry badges
use the existing PyYAML development dependency. Google and NumPy docstring
sections, RST roles and directives are translated to MyST.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re as _re
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import get_origin

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
    "progenax.stellar",
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
        / "validation"
        / "data"
        / "grad_audit_results.json"
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
        bits.append(f"[📇 model card](../15-model-reference/{fam}.md#card-{model})")
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
        (
            "Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, bool, int, float, complex, NoneType]",
            "ArrayLike | None",
        ),
        (
            "Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, bool, int, float, complex]",
            "ArrayLike",
        ),
        ("Union[Array, ndarray, bool, number, bool, int, float, complex]", "ArrayLike"),
        (
            "jax.Array | numpy.ndarray | numpy.bool | numpy.number | bool | int | float | complex",
            "ArrayLike",
        ),
        # the jaxtyping PRNG-key union (Key[''] | UInt32[2] | UInt32[4])
        (
            "Union[jaxtyping.Key[Array, ''], jaxtyping.UInt32[Array, '2'], jaxtyping.UInt32[Array, '4']]",
            "PRNGKeyArray",
        ),
    ):
        text = text.replace(union, alias)
    # Shorten the module-qualified jaxtyping shape annotations for readability.
    text = text.replace("jaxtyping.", "")
    return text


_SECTIONS = (
    "Args",
    "Arguments",
    "Parameters",
    "Other Parameters",
    "Attributes",
    "Returns",
    "Yields",
    "Raises",
    "Note",
    "Notes",
    "References",
    "Example",
    "Examples",
    "Warning",
    "Warnings",
    "See Also",
)
_ROLE = _re.compile(r":(?:py:)?(\w+):`([^`]+)`")
_DIRECTIVE = _re.compile(r"^( *)\.\. ([-\w]+)::\s*(.*)$", _re.M)


def _convert_markup(text: str, links: dict[str, str], context: str) -> str:
    """Translate observed RST syntax, leaving literal code and MyST fences intact."""

    def role(match):
        kind, target = match.groups()
        if kind == "math":
            return f"${target}$"
        explicit = _re.fullmatch(r"(.+?)\s*<([^>]+)>", target)
        label, target = explicit.groups() if explicit else (target, target)
        target = target.lstrip("~")
        if not explicit and label.startswith("~"):
            label = target.rsplit(".", 1)[-1]
        # Resolve local class methods first, then module exports, then unique names.
        scopes = [context, context.rsplit(".", 1)[0]] if context else []
        href = next(
            (
                links[key]
                for key in [*(f"{scope}.{target}" for scope in scopes), target]
                if key in links
            ),
            None,
        )
        return f"[`{label}`]({href})" if href else f"`{label}`"

    lines = text.splitlines()
    out = []
    i = 0
    code_fence = None
    while i < len(lines):
        line = lines[i]
        fence = _re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            if code_fence is None:
                code_fence = fence.group(1)
            elif fence.group(1)[0] == code_fence[0] and len(fence.group(1)) >= len(
                code_fence
            ):
                code_fence = None
            out.append(line)
            i += 1
            continue
        if code_fence or line.startswith("    "):
            out.append(line)
            i += 1
            continue
        directive = _DIRECTIVE.match(line)
        if directive:
            indent, kind, first = directive.groups()
            body = [first] if first else []
            i += 1
            continuation = []
            while i < len(lines) and (
                not lines[i].strip()
                or len(lines[i]) - len(lines[i].lstrip()) > len(indent)
            ):
                continuation.append(lines[i])
                i += 1
            body_text = "\n".join(
                body + [textwrap.dedent("\n".join(continuation)).strip()]
            ).strip()
            if kind in ("warning", "note", "seealso", "tip", "important", "caution"):
                out.extend(
                    [
                        f":::{{{kind}}}",
                        _convert_markup(body_text, links, context),
                        ":::",
                        "",
                    ]
                )
            elif kind == "math":
                out.extend(["$$", body_text, "$$", ""])
            elif kind in ("code-block", "code"):
                language = first or "text"
                out.extend(
                    [
                        f"```{language}",
                        textwrap.dedent("\n".join(continuation)).strip(),
                        "```",
                        "",
                    ]
                )
            else:
                raise ValueError(f"unsupported docstring directive: {kind}")
            continue
        line = _ROLE.sub(role, line)
        line = _re.sub(r"``([^`]+)``", r"`\1`", line)
        out.append(line)
        i += 1
    return "\n".join(out).rstrip()


def _format_docstring(
    obj, links: dict[str, str] | None = None, context: str = ""
) -> str:
    """Render Google/NumPy sections without flattening equations, prose or examples."""
    doc = inspect.getdoc(obj)
    if not doc:
        return "*(no docstring)*"
    # Headings are unindented after inspect.getdoc; code within sections stays indented.
    names = "|".join(_re.escape(name) for name in _SECTIONS)
    headings = _re.compile(rf"^({names})(?:\s*:\s*$|\s*\n-{{3,}}\s*$)", _re.M)
    parts = []
    last, name = 0, ""
    for match in headings.finditer(doc):
        parts.append((name, doc[last : match.start()].strip("\n")))
        name, last = match.group(1), match.end()
    parts.append((name, doc[last:].strip("\n")))
    out = []
    for name, body in parts:
        # A Google section can be followed by unindented prose/directives before
        # the next named section. Dedent its leading block, not that later block.
        first_line = next((line for line in body.splitlines() if line.strip()), "")
        indent = len(first_line) - len(first_line.lstrip())
        if name and indent:
            normalized = []
            leading = True
            for line in body.splitlines():
                if line.strip() and not line.startswith(" " * indent):
                    leading = False
                normalized.append(line[indent:] if leading else line)
            body = "\n".join(normalized)
        body = textwrap.dedent(body).strip("\n")
        if not body.strip():
            continue
        if name:
            out.extend([f"**{name}**", ""])
        if name in ("Example", "Examples"):
            # Fence doctest blocks only; preserve explanatory prose and existing fences.
            lines = body.splitlines()
            i = 0
            fenced = False
            while i < len(lines):
                if _re.match(r"^\s*(`{3,}|~{3,})", lines[i]):
                    fenced = not fenced
                if not fenced and lines[i].lstrip().startswith(">>>"):
                    block = []
                    while i < len(lines) and lines[i].strip():
                        block.append(lines[i])
                        i += 1
                    out.extend(
                        ["```pycon", textwrap.dedent("\n".join(block)), "```", ""]
                    )
                else:
                    out.append(lines[i])
                    i += 1
        elif name in (
            "Args",
            "Arguments",
            "Parameters",
            "Other Parameters",
            "Attributes",
        ):
            # Keep complex/unparsed sections verbatim rather than dropping their preamble.
            rows = list(
                _re.finditer(
                    r"^(\*{0,2}\w+(?:,\s*\w+)*(?:\s*\([^\n]+\))?)\s*:\s*([^\n]*)$",
                    body,
                    _re.M,
                )
            )
            if rows and not body[: rows[0].start()].strip():
                entries = []
                for j, row in enumerate(rows):
                    end = rows[j + 1].start() if j + 1 < len(rows) else len(body)
                    description = textwrap.dedent(body[row.end() : end]).strip()
                    entries.append(
                        (row.group(1).strip(), row.group(2).strip(), description)
                    )
                # Retain the site's existing tables for simple parameter prose.
                # Paragraphs, equations and nested examples need block structure.
                simple = all(
                    "\n\n" not in description
                    and not _re.search(
                        r"(?m)^\s*(?:[-*] |\.\. |```|:::|\$\$|>>>| {4})", description
                    )
                    for _parameter, _first, description in entries
                )
                if simple:
                    out.extend(["| Parameter | Description |", "|---|---|"])
                    for parameter, first, description in entries:
                        combined = " ".join(
                            (first + " " + description).split()
                        ).replace("|", "\\|")
                        out.append(f"| `{parameter}` | {combined} |")
                else:
                    for parameter, first, description in entries:
                        out.extend([f"- **`{parameter}`**: {first}", ""])
                        if description:
                            out.extend([textwrap.indent(description, "  "), ""])
            else:
                out.append(body)
        else:
            out.append(body)
        out.append("")
    return _convert_markup("\n".join(out).rstrip(), links or {}, context)


def _collect_class_members(cls):
    """Public methods/properties, nearest definition first; exclude dependency machinery."""
    seen = set()
    members = []
    for owner in cls.__mro__:
        if not owner.__module__.startswith("progenax."):
            continue
        for name, descriptor in sorted(owner.__dict__.items()):
            if not _is_public(name) or name in seen:
                continue
            seen.add(name)
            if isinstance(descriptor, property):
                if descriptor.__doc__:
                    members.append((name, descriptor.fget, "property", owner))
            elif isinstance(descriptor, (classmethod, staticmethod)):
                kind = (
                    "classmethod"
                    if isinstance(descriptor, classmethod)
                    else "staticmethod"
                )
                members.append((name, getattr(cls, name), kind, owner))
            elif inspect.isfunction(descriptor) or (
                callable(descriptor) and hasattr(descriptor, "__wrapped__")
            ):
                members.append((name, descriptor, "method", owner))
    return sorted(members, key=lambda member: member[0])


def _reference_links(catalog) -> dict[str, str]:
    """Resolve public API roles only when they identify one object/contract."""
    candidates = {}

    def add(key, obj, href):
        candidates.setdefault(key, []).append((obj, href))

    for module, symbols in catalog.items():
        short = module.removeprefix("progenax.")
        add(module, importlib.import_module(module), f"{short}.md")
        for name, obj in symbols:
            href = f"{short}.md#{_anchor(short, name)}"
            add(f"{module}.{name}", obj, href)
            add(name, obj, href)
            qual = getattr(obj, "__qualname__", None)
            if qual:
                add(f"{obj.__module__}.{qual}", obj, href)
            if inspect.isclass(obj):
                for member, method, _kind, owner in _collect_class_members(obj):
                    href = f"{short}.md#{_anchor(short, name + '-' + member)}"
                    identity = (owner, member)
                    add(f"{module}.{name}.{member}", identity, href)
                    add(f"{name}.{member}", identity, href)
                    add(
                        f"{owner.__module__}.{owner.__qualname__}.{member}",
                        identity,
                        href,
                    )
    links = {}
    for key, options in candidates.items():
        first = options[0][0]
        if all(
            obj is first or (isinstance(obj, tuple) and obj == first)
            for obj, _href in options
        ):
            links[key] = options[0][1]
    links["DEFAULT_UNITS"] = links["progenax.DEFAULT_UNITS"] = (
        "index.md#api-progenax-default_units"
    )
    return links


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
        except AttributeError as exc:
            raise ValueError(
                f"{module_name}.{name}: declared export is missing"
            ) from exc
        yield name, obj


def _classify(obj) -> str:
    """Classify exports, including callable typing aliases such as Literal."""
    if get_origin(obj) is not None:
        return "type alias"
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


def _emit_module_page(
    module_name: str, out_path: Path, package_root: Path, *, catalog=None, links=None
) -> int:
    """Write the MyST page for a single module. Returns symbol count."""
    short = module_name.removeprefix("progenax.")
    if catalog is None:
        catalog = {
            module: list(_collect_public_symbols(module)) for module in PUBLIC_MODULES
        }
    links = _reference_links(catalog) if links is None else links
    symbols = catalog[module_name]

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
    lines.append(f"Import path: `{module_name}`")
    lines.append("")
    lines.append(f"Public symbols: **{len(symbols)}**")
    lines.append("")

    if not symbols:
        lines.append("*No public symbols found.*")
        out_path.write_text("\n".join(lines).rstrip() + "\n")
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
        sig = "" if kind == "type alias" else _format_signature(obj)
        try:
            doc = (
                f"```python\n{name} = {str(obj).replace('typing.', '')}\n```"
                if kind == "type alias"
                else _format_docstring(obj, links, f"{module_name}.{name}")
            )
        except Exception as exc:
            raise ValueError(f"{module_name}.{name}: {exc}") from exc
        src = _source_link(obj, package_root)

        lines.append(f"({_anchor(short, name)})=")
        lines.append(f"## `{short}.{name}`")
        lines.append("")

        kind_label = {
            "class": "*class*",
            "protocol": "*protocol*",
            "function": "*function*",
            "value": "*value*",
            "type alias": "*type alias*",
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
            lines.append(f"*Source: [`{src}`]({GITHUB_BLOB}/{src})*")
            lines.append("")

        if inspect.isclass(obj):
            for member_name, method, member_kind, owner in _collect_class_members(obj):
                member = name + "-" + member_name
                lines.extend(
                    [
                        f"({_anchor(short, member)})=",
                        f"### `{name}.{member_name}`",
                        "",
                        f"*{member_kind}*",
                        "",
                    ]
                )
                inherited = None
                if owner is not obj:
                    for owner_module, exports in catalog.items():
                        for owner_name, exported in exports:
                            if exported is owner:
                                owner_short = owner_module.removeprefix("progenax.")
                                inherited = f"{owner_short}.md#{_anchor(owner_short, owner_name + '-' + member_name)}"
                                break
                        if inherited:
                            break
                if inherited:
                    lines.extend(
                        [
                            f"Inherited from [`{owner.__name__}.{member_name}`]({inherited}).",
                            "",
                        ]
                    )
                else:
                    if member_kind != "property":
                        lines.extend(
                            [
                                "```python",
                                f"{member_name}{_format_signature(method)}",
                                "```",
                                "",
                            ]
                        )
                    try:
                        lines.extend(
                            [
                                _format_docstring(
                                    method, links, f"{module_name}.{name}"
                                ),
                                "",
                            ]
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"{module_name}.{name}.{member_name}: {exc}"
                        ) from exc
                    source = _source_link(method, package_root)
                    if source:
                        lines.extend(
                            [f"*Source: [`{source}`]({GITHUB_BLOB}/{source})*", ""]
                        )

    out_path.write_text("\n".join(lines).rstrip() + "\n")
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
    lines.append(
        f"Total API entries: **{len(sorted_syms)}**, including public methods and documented properties."
    )
    lines.append("")
    lines.append(
        "> Entries include each submodule export, public class members and `DEFAULT_UNITS`. The top-level "
        "`progenax` package re-exports a curated subset for convenience "
        "(e.g. `from progenax import PlummerProfile`); import the submodule "
        "(e.g. `from progenax.imf import ...`) to reach the rest."
    )
    lines.append("")
    lines.append("| Symbol | Kind | Module |")
    lines.append("|--------|------|--------|")

    for name, kind, mod_short in sorted_syms:
        page = f"{mod_short}.md"
        anchor_name = name.replace(".", "-")
        module_label = f"progenax.{mod_short}"
        if mod_short == "progenax":
            page, module_label = "index.md", "progenax"
        lines.append(
            f"| [`{name}`]({page}#{_anchor(mod_short, anchor_name)}) | {kind} | `{module_label}` |"
        )

    lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n")


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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare regenerated files without modifying the reference",
    )
    args = parser.parse_args(argv)

    # Auto-detect package root by importing progenax and getting its file path
    if args.package_root is None:
        progenax = importlib.import_module("progenax")
        args.package_root = Path(progenax.__file__).resolve().parent.parent

    # Stage generation: a failed contract never leaves a partial successful reference.
    try:
        catalog = {
            module: list(_collect_public_symbols(module)) for module in PUBLIC_MODULES
        }
        links = _reference_links(catalog)
        root = importlib.import_module("progenax")
        for name in root.__all__:
            obj = getattr(root, name)
            if name != "DEFAULT_UNITS" and not any(
                name == exported_name and obj is exported
                for symbols in catalog.values()
                for exported_name, exported in symbols
            ):
                raise ValueError(f"progenax.{name}: root export has no API owner")
        with tempfile.TemporaryDirectory(prefix="progenax-api-") as temporary:
            staging = Path(temporary)
            all_symbols = [("DEFAULT_UNITS", "value", "progenax")]
            total = 0
            for module, symbols in catalog.items():
                short = module.removeprefix("progenax.")
                total += _emit_module_page(
                    module,
                    staging / f"{short}.md",
                    args.package_root,
                    catalog=catalog,
                    links=links,
                )
                for name, obj in symbols:
                    all_symbols.append((name, _classify(obj), short))
                    if inspect.isclass(obj):
                        all_symbols.extend(
                            (f"{name}.{member}", kind, short)
                            for member, _method, kind, _owner in _collect_class_members(
                                obj
                            )
                        )
            _emit_full_symbol_index(all_symbols, staging / "full-symbol-index.md")
            stale = []
            for generated in sorted(staging.glob("*.md")):
                target = args.out / generated.name
                if args.check:
                    if (
                        not target.exists()
                        or target.read_bytes() != generated.read_bytes()
                    ):
                        stale.append(str(target))
                else:
                    args.out.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(generated.read_bytes())
            if stale:
                print("API reference is stale:\n" + "\n".join(stale), file=sys.stderr)
                return 1
    except Exception as exc:
        print(f"ERROR: API generation failed: {exc}", file=sys.stderr)
        return 1
    action = "Checked" if args.check else "Wrote"
    print(
        f"{action} {len(PUBLIC_MODULES)} module pages + 1 index ({total} exports, {len(all_symbols)} API entries)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
