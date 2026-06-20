"""Render the generated test-dashboard JSON to a MyST matrix page.

Split out of ``scripts/build_test_dashboard.py`` so the generator stays under the
500-LOC file cap (the table-building logic is the bulk of ``--render``). This is a
plain SCRIPT module — pure string assembly, no JAX, no side effects beyond
returning the page text. ``build_test_dashboard.py --render`` imports
:func:`render_dashboard_page` and writes its output to
``docs/website/50-validation/test-dashboard.md``.

The page is GENERATED — never hand-edited. It opens with frontmatter + an intro
paragraph that says so and names the regenerating command, then a MyST
``{list-table}`` colon-fence matrix: one row per module, columns

    module | unit | integration | validation | line-cov % | grad-audit fill |
    slowest test | validation PASS
"""

from __future__ import annotations

_GENERATING_CMD = "uv run python scripts/build_test_dashboard.py --emit --render"


def _line_cov_cell(line_cov: float | None) -> str:
    """A per-module line-cov cell: a percentage, or the Phase-2 pending marker."""
    if line_cov is None:
        return "pending (Phase 2)"
    return f"{line_cov:.1f}"


def _grad_audit_fill(registries: dict) -> str:
    """One-cell summary of the (repo-wide) grad-audit registry fill.

    The grad-audit registry is the only one built in Phase 1; the other three
    report ``not-built``. We show ``AUDITED/(AUDITED+EXEMPT)`` + the hazard count
    when built, else ``not built``. This is a repo-wide figure (not per-module), so
    every module row shows the same value — the column documents that the
    differentiability registry is live and hazard-free.
    """
    block = registries.get("differentiability", {})
    if block.get("status") != "built":
        return "not built"
    audited = block.get("audited", 0)
    exempt = sum(block.get("exempt", {}).values())
    hazards = block.get("hazards", 0)
    return f"{audited}/{audited + exempt} audited, {hazards} haz"


def _slowest_cell(durations: dict, module: str) -> str:
    """The slowest test for ``module`` from the durations block, or a dash.

    ``durations`` is either ``{"status": "not-measured"}`` (no committed artifact)
    or ``{module: {"slowest_test": str, "seconds": float}}``. We surface only the
    test node tail + seconds to keep the cell short.
    """
    entry = durations.get(module)
    if not isinstance(entry, dict):
        return "—"
    node = entry.get("slowest_test", "")
    tail = node.rsplit("::", 1)[-1] if "::" in node else node
    seconds = entry.get("seconds")
    if seconds is None:
        return tail or "—"
    return f"{tail} ({seconds:.1f}s)"


def _validation_pass_cell(validation_scripts: dict, module: str) -> str:
    """PASS/FAIL/unknown for a module's ``validate_<module>.py`` script, if any.

    ``validation_scripts`` maps ``validate_*.py`` -> exit code (or ``"unknown"``).
    We match a script named ``validate_<module>.py`` (exact stem). Exit ``0`` ->
    PASS, a nonzero int -> FAIL, ``"unknown"`` -> "—". Modules with no matching
    validate script also show "—".
    """
    code = validation_scripts.get(f"validate_{module}.py")
    if code == 0:
        return "PASS"
    if isinstance(code, int):
        return "FAIL"
    return "—"


def _matrix_rows(dashboard: dict) -> list[str]:
    """Build the ``{list-table}`` body rows (one per module, sorted)."""
    registries = dashboard.get("registries", {})
    durations = dashboard.get("durations", {})
    validation_scripts = dashboard.get("validation_scripts", {})
    grad_fill = _grad_audit_fill(registries)  # repo-wide; same on every row

    rows: list[str] = []
    for module, counts in sorted(dashboard.get("modules", {}).items()):
        cells = [
            f"`{module}`",
            str(counts.get("unit", 0)),
            str(counts.get("integration", 0)),
            str(counts.get("validation", 0)),
            _line_cov_cell(counts.get("line_cov")),
            grad_fill,
            _slowest_cell(durations, module),
            _validation_pass_cell(validation_scripts, module),
        ]
        # MyST list-table row: a top-level bullet, then one nested bullet per cell.
        block = [f"* - {cells[0]}"]
        block += [f"  - {cell}" for cell in cells[1:]]
        rows.append("\n".join(block))
    return rows


# Human labels for the four registries, in dashboard/key order. The prose sentence
# derives BUILT/not-built from the JSON so it can never drift from the actual blocks.
_REGISTRY_LABELS = {
    "differentiability": "differentiability",
    "api_coverage": "API-coverage",
    "physics_validation": "physics-validation",
    "provenance": "provenance",
}


def _registry_status_phrase(registries: dict) -> str:
    """A prose sentence naming which registries are built vs not (derived from JSON).

    Partitions the four registry blocks by their ``status`` so the intro paragraph
    cannot claim a registry is "not built" once it lands. An ``api_coverage`` block
    additionally reports its ``untested`` hole count when built-but-not-full.
    """
    built, not_built = [], []
    for key, label in _REGISTRY_LABELS.items():
        block = registries.get(key, {})
        if block.get("status") == "built":
            tag = label
            if block.get("full") is False and "untested" in block:
                tag += f" ({block['untested']} hole(s))"
            built.append(tag)
        else:
            not_built.append(label)
    parts = []
    if built:
        parts.append(f"Built registries: {', '.join(built)}.")
    if not_built:
        parts.append(f"Not built yet: {', '.join(not_built)}.")
    return " ".join(parts)


def _validation_scripts_section(validation_scripts: dict) -> list[str]:
    """A standalone table of EVERY ``validate_*.py`` -> PASS/FAIL/—.

    The per-module matrix only surfaces scripts whose stem matches a census module
    (``validate_<module>.py``), so 22 of 24 never appear there. This section lists ALL
    of them straight from ``validation/data/validation_runs.json`` so a failing
    validation script is visible on the page, not just in the JSON.
    """
    if not validation_scripts:
        return []
    rows: list[str] = []
    for name, code in sorted(validation_scripts.items()):
        if code == 0:
            verdict = "PASS"
        elif isinstance(code, int):
            verdict = f"FAIL (exit {code})"
        else:
            verdict = "—"
        rows.append(f"* - `{name}`\n  - {verdict}")
    n_fail = sum(
        1 for c in validation_scripts.values() if isinstance(c, int) and c != 0
    )
    return [
        "## Validation scripts",
        "",
        f"Exit status of every `scripts/validate_*.py` "
        f"(from `validation/data/validation_runs.json`): "
        f"**{len(validation_scripts)} scripts, {n_fail} failing**.",
        "",
        "```{list-table} Validation script runs",
        ":header-rows: 1",
        ":align: left",
        "",
        "* - Script\n  - Status",
        "\n".join(rows),
        "```",
        "",
    ]


def render_dashboard_page(dashboard: dict) -> str:
    """Return the full MyST page text for the dashboard JSON.

    Frontmatter (title/description) + a GENERATED-banner intro naming the
    regenerating command + the ``{list-table}`` matrix.
    """
    generated_utc = dashboard.get("generated_utc", "unknown")
    gate = dashboard.get("gate", {})
    registries = dashboard.get("registries", {})
    line_cov = gate.get("line_cov_measured")
    line_cov_str = (
        "pending (Phase 2, full-suite `--cov` run)"
        if line_cov is None
        else f"{line_cov:.1f}% (floor {gate.get('line_cov_floor')}%)"
    )
    registry_phrase = _registry_status_phrase(registries)

    header = [
        "* - Module",
        "  - Unit",
        "  - Integration",
        "  - Validation",
        "  - Line-cov %",
        "  - Grad-audit fill",
        "  - Slowest test",
        "  - Validation PASS",
    ]

    lines = [
        "---",
        "title: Test Dashboard",
        "description: Generated single source of truth for progenax's per-module "
        "test census, line coverage, registry fill, durations, and validation runs.",
        "---",
        "# Test Dashboard",
        "",
        "```{warning}",
        "This page is **generated** — do not hand-edit. Regenerate it with",
        "",
        f"    {_GENERATING_CMD}",
        "",
        "which stamps the timestamp, emits "
        "`validation/data/test_dashboard.json`, and re-renders this page.",
        "```",
        "",
        f"Generated at `{generated_utc}`. Line coverage: {line_cov_str}. "
        "`Line-cov %` cells read **pending (Phase 2)** until the committed "
        "full-suite `coverage.json` exists, then show the statement-weighted "
        "per-directory coverage. The `Grad-audit fill` column reports the "
        "repo-wide differentiability registry "
        f"(audited / audited+exempt, hazard count). {registry_phrase}",
        "",
        "```{list-table} Per-module test + coverage matrix",
        ":header-rows: 1",
        ":align: left",
        "",
        "\n".join(header),
        "\n".join(_matrix_rows(dashboard)),
        "```",
        "",
        *_validation_scripts_section(dashboard.get("validation_scripts", {})),
    ]
    return "\n".join(lines)
